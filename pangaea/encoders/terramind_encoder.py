import gzip
import json
import logging
import random
import xarray as xr
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
from abc import ABC, abstractmethod
from huggingface_hub import hf_hub_download
from PIL import Image
import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from einops import rearrange, repeat, reduce
import math
from functools import partial
from logging import Logger
import hashlib
import torch.nn as nn


from pangaea.encoders.base import Encoder

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def softmax1(tensor):
    # See https://www.evanmiller.org/attention-is-off-by-one.html
    return F.pad(tensor, (0,1)).softmax(dim=-1)[...,:-1]

def build_1d_sincos_posemb(max_len, embed_dim=1024, temperature=10000.):
    """Sine-cosine positional embeddings from MoCo-v3, adapted back to 1d

    Returns positional embedding of shape (1, N, D)
    """
    arange = torch.arange(max_len, dtype=torch.float32) # Shape (N,)
    assert embed_dim % 2 == 0, 'Embed dimension must be divisible by 2 for 1D sin-cos position embedding'
    pos_dim = embed_dim // 2
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim # Shape (D/2,)
    omega = 1. / (temperature ** omega)
    out = torch.einsum('n,d->nd', [arange, omega]) # Outer product, shape (N, D/2)
    pos_emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1).unsqueeze(0) # Shape (1, N, D)
    return pos_emb

def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.0):
    """Sine-cosine positional embeddings as used in MoCo-v3

    Returns positional embedding of shape (1, N, D) where N = W*H
    """
    grid_w = torch.arange(w, dtype=torch.float32) # Shape (W,)
    grid_h = torch.arange(h, dtype=torch.float32) # Shape (H, )
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij') # Shapes (W, H)
    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim # Shape (D/4,)
    omega = 1. / (temperature ** omega)
    out_w = torch.einsum('n,d->nd', [grid_w.reshape(-1), omega]) # Outer product, shape (W*H, D/4)
    out_h = torch.einsum('n,d->nd', [grid_h.reshape(-1), omega]) # Outer product, shape (W*H, D/4)
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1).unsqueeze(0) # Shape (1, W*H, D)
    return pos_emb


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks). 
    Implementation from timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class LayerNorm(nn.Module):
    """Custom implementation of LayerNorm with the option to disable the bias term"""
    def __init__(self, normalized_shape: int, eps=1e-5, bias=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_buffer("bias", torch.zeros(normalized_shape))

        # Normalized shape must be a tuple for F.layer_norm
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, eps=self.eps)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GatedMlp(nn.Module):
    """Implements SwiGLU and other gated feed-forward layers from Noam Shazeer's paper: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, bias=True):
        super().__init__()
        out_features = out_features or in_features
        # If gated, multiply hidden_dim by 2/3 to account for extra matmul
        hidden_features = int(2 * (hidden_features or in_features) / 3) 
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)

    def forward(self, x):
        x = self.fc2(self.act(self.fc1(x)) * self.fc3(x))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True, attn_drop=0., proj_drop=0., allow_zero_attn=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.allow_zero_attn = allow_zero_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1) # Unsqueeze attention mask for multi-head
            attn = attn.masked_fill(mask, -torch.finfo(attn.dtype).max)

        if self.allow_zero_attn:
            attn = softmax1(attn)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class NormAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True,  norm_layer=nn.LayerNorm, attn_drop=0., proj_drop=0., allow_zero_attn=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.allow_zero_attn = allow_zero_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = norm_layer(head_dim)
        self.k_norm = norm_layer(head_dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1) # Unsqueeze for multi-head
            attn = attn.masked_fill(mask, -torch.finfo(attn.dtype).max)

        if self.allow_zero_attn:
            attn = softmax1(attn)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, proj_bias=True, mlp_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, gated_mlp=False, qk_norm=False, allow_zero_attn=False):
        super().__init__()
        self.norm1 = norm_layer(dim)

        if not qk_norm:
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop, allow_zero_attn=allow_zero_attn)
        else:
            self.attn = NormAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, norm_layer=norm_layer, attn_drop=attn_drop, proj_drop=drop, allow_zero_attn=allow_zero_attn)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        if not gated_mlp:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, bias=mlp_bias, drop=drop)
        else:
            self.mlp = GatedMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, bias=mlp_bias)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

#-------------------------
def get_pil_resample_mode(resample_mode: str):
    """
    Returns the PIL resampling mode for the given resample mode string.

    Args:
        resample_mode: Resampling mode string
    """
    if resample_mode is None:
        return None
    elif resample_mode == "bilinear":
        return Image.Resampling.BILINEAR if hasattr(Image, 'Resampling') else Image.BILINEAR
    elif resample_mode == "bicubic":
        return Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC
    elif resample_mode == "nearest":
        return Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST
    else:
        raise ValueError(f"Resample mode {resample_mode} is not supported.")


class AbstractTransform(ABC):

    @abstractmethod
    def load(self, sample):
        pass

    @abstractmethod
    def preprocess(self, sample):
        pass

    @abstractmethod
    def image_augment(self, v, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        pass

    @abstractmethod
    def postprocess(self, v):
        pass


class ImageTransform(AbstractTransform):

    @staticmethod
    def pil_loader(path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        # with open(path, 'rb') as f:
        #     img = Image.open(f)
        img = Image.open(path)
        return img

    def zarr_loader(task: str, path: str):
        # print(f"loading data from... {path}")
        try:
            zarr_data = xr.open_zarr(path)
            bands = zarr_data["bands"].values
        except Exception as e:
            logging.error(f'Error while loading {path}')
            logging.exception(e)
            raise e
        # print(f"bands.shape = {bands.shape}")
        return bands

    @staticmethod
    def image_hflip(img: Image, flip: bool):
        """Crop and resize an image

        :param img: Image to crop and resize
        :param flip: Whether to flip the image
        :return: Flipped image (if flip = True)
        """
        if flip:
            img = TF.hflip(img)
        return img

    @staticmethod
    def image_crop_and_resize(img: Image, crop_coords: Tuple, target_size: Tuple, resample_mode: str = None):
        """Crop and resize an image

        :param img: Image to crop and resize
        :param crop_coords: Coordinates of the crop (top, left, h, w)
        :param target_size: Coordinates of the resize (height, width)
        :return: Cropped and resized image
        """

        top, left, h, w = crop_coords
        resize_height, resize_width = target_size
        img = TF.crop(img, top, left, h, w)
        resample_mode = get_pil_resample_mode(resample_mode)
        img = img.resize((resize_height, resize_width))
        return img

    @staticmethod
    def satellite_image_crop_and_resize(img: Image, crop_coords: Tuple, target_size: Tuple, resample_mode: str = None):
        """Crop and resize an image

        :param img: Image to crop and resize
        :param crop_coords: Coordinates of the crop (top, left, h, w)
        :param target_size: Coordinates of the resize (height, width)
        :return: Cropped and resized image
        """

        top, left, h, w = crop_coords
        img = TF.crop(img, top, left, h, w)
        return img


class Sen1GRDTransform(ImageTransform):

    def __init__(self):
        self.sen1_mean = SENTINEL1GRD_MEAN
        self.sen1_std = SENTINEL1GRD_STD

    def load(self, path):
        sample = self.zarr_loader(path)
        return sample

    def preprocess(self, sample):
        sample = torch.Tensor(sample)
        return sample

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        assert img.shape[-1] == 264, f"Image shape inconsistent. {img.shape} is invalid"
        img = img.squeeze()
        img = TF.normalize(img, mean=self.sen1_mean, std=self.sen1_std)
        img = self.satellite_image_crop_and_resize(img, crop_coords, target_size, resample_mode=resample_mode)
        img = self.image_hflip(img, flip)
        # img = img.unsqueeze(dim=2)
        # print(f"Sentinel-1 training data shape: {img.shape}")
        return img

    def postprocess(self, sample):
        return sample


class Sen1RTCTransform(Sen1GRDTransform):

    def __init__(self):
        self.sen1_mean = SENTINEL1RTC_MEAN
        self.sen1_std = SENTINEL1RTC_STD


class DEMTransform(ImageTransform):

    def __init__(self):
        self.dem_mean = DEM_MEAN
        self.dem_std = DEM_STD

    def load(self, path):
        sample = self.zarr_loader(path)
        return sample

    def preprocess(self, sample):
        sample = torch.Tensor(sample)
        return sample

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        assert img.shape[-1] == 264, f"Image shape inconsistent. {img.shape} is invalid"
        img = img.squeeze()
        img = TF.normalize(img, mean=self.dem_mean, std=self.dem_std)
        img = self.satellite_image_crop_and_resize(img, crop_coords, target_size, resample_mode=resample_mode)
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample):
        return sample


class NDVITransform(ImageTransform):

    def __init__(self):
        self.ndvi_mean = NDVI_MEAN
        self.ndvi_std = NDVI_STD

    def load(self, path):
        sample = self.zarr_loader(path)
        return sample

    def preprocess(self, sample):
        sample = torch.Tensor(sample)
        return sample

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        assert img.shape[-1] == 264, f"Image shape inconsistent. {img.shape} is invalid"
        img = img.squeeze()
        img = TF.normalize(img, mean=self.ndvi_mean, std=self.ndvi_std)
        img = self.satellite_image_crop_and_resize(img, crop_coords, target_size, resample_mode=resample_mode)
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample):
        return sample


class Sen2L2ATransform(ImageTransform):

    def __init__(self):
        self.sen2l2a_mean = SENTINEL2L2A_MEAN
        self.sen2l2a_std = SENTINEL2L2A_STD

    def load(self, path):
        # TODO: Instead of converting to RGB here, do it either in the preprocess or the postprocess step. Makes it compatible with wds dataloading.
        sample = self.zarr_loader(path)
        return sample

    def preprocess(self, sample):
        if isinstance(sample, np.ndarray) and np.issubdtype(sample.dtype, np.integer):
            sample = sample.astype(np.int32)  # because of limited support for torch.uint16
        sample = torch.Tensor(sample)
        return sample

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        assert img.shape[-1] == 264, f"Image shape inconsistent. {img.shape} is invalid"

        img = img.squeeze()
        img = TF.normalize(img, mean=self.sen2l2a_mean, std=self.sen2l2a_std)
        img = self.satellite_image_crop_and_resize(img, crop_coords, target_size, resample_mode=resample_mode)
        img = self.image_hflip(img, flip)
        # img = img.unsqueeze(dim=2)
        # print(f"Sentinel-2-L2A training data shape: {img.shape}")
        return img

    def postprocess(self, sample):
        return sample


class Sen2L1CTransform(ImageTransform):

    def __init__(self):
        self.sen2l1c_mean = SENTINEL2L1C_MEAN
        self.sen2l1c_std = SENTINEL2L1C_STD

    def load(self, path):
        # TODO: Instead of converting to RGB here, do it either in the preprocess or the postprocess step. Makes it compatible with wds dataloading.
        sample = self.zarr_loader(path)
        return sample

    def preprocess(self, sample):
        if isinstance(sample, np.ndarray) and np.issubdtype(sample.dtype, np.integer):
            sample = sample.astype(np.int32)  # because of limited support for torch.uint16
        sample = torch.Tensor(sample)
        return sample

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        assert img.shape[-1] == 264, f"Image shape inconsistent. {img.shape} is invalid"

        img = img.squeeze()
        img = TF.normalize(img, mean=self.sen2l1c_mean, std=self.sen2l1c_std)
        img = self.satellite_image_crop_and_resize(img, crop_coords, target_size, resample_mode=resample_mode)
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample):
        return sample


class Sen2RGBTransform(ImageTransform):

    def __init__(self):
        self.sen2rgb_mean = SENTINEL2RGB_MEAN
        self.sen2rgb_std = SENTINEL2RGB_STD

    def load(self, path):
        # TODO: Instead of converting to RGB here, do it either in the preprocess or the postprocess step. Makes it compatible with wds dataloading.
        sample = self.zarr_loader(path)
        return sample

    def preprocess(self, sample):
        if isinstance(sample, np.ndarray) and np.issubdtype(sample.dtype, np.integer):
            sample = sample.astype(np.int32)  # because of limited support for torch.uint16
        sample = torch.Tensor(sample)
        return sample

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        assert img.shape[-1] == 264, f"Image shape inconsistent. {img.shape} is invalid"

        img = img.squeeze()
        img = TF.normalize(img, mean=self.sen2rgb_mean, std=self.sen2rgb_std)
        img = self.satellite_image_crop_and_resize(img, crop_coords, target_size, resample_mode=resample_mode)
        img = self.image_hflip(img, flip)
        # img = img.unsqueeze(dim=2)
        # print(f"Sentinel-2-L2A training data shape: {img.shape}")
        return img

    def postprocess(self, sample):
        return sample


class LulcTransform(ImageTransform):

    def __init__(self):
        pass

    def load(self, path):
        zarr_data = xr.open_zarr(path)
        sample = zarr_data["bands"].transpose("sample", "time", "band", "y", "x").values
        sample = torch.tensor(sample).long()

        # Get 1-hot encoding using the class indices
        sample = sample.squeeze(dim=2)  # Remove the band dimension
        one_hot = F.one_hot(sample, num_classes=10).to(torch.float32)
        one_hot = rearrange(one_hot, "s t y x c -> s t c y x")

        return one_hot

    def preprocess(self, sample):
        return sample

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        assert img.shape[-1] == 264, f"Image shape inconsistent. {img.shape} is invalid"
        img = img.squeeze()
        img = self.satellite_image_crop_and_resize(img, crop_coords, target_size, resample_mode=resample_mode)
        img = self.image_hflip(img, flip)

        return img

    def postprocess(self, sample):
        return sample


# Placeholder class for all raw satellite inputs
class UntokTransform(ImageTransform):

    def __init__(self):
        self.mean = (None)
        self.std = (None)

    def load(self, path):
        sample = np.load(path).astype(int)
        return sample

    def preprocess(self, sample):
        sample = torch.Tensor(sample)
        return sample

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        img = TF.normalize(img, mean=self.mean, std=self.std)
        img = self.satellite_image_crop_and_resize(img, crop_coords, target_size, resample_mode=resample_mode)
        img = self.image_hflip(img, flip)

        return img

    def postprocess(self, sample):
        return sample


# For the untokenized S2 samples during pre-training
class UntokSen2L2ATransform(UntokTransform):

    def __init__(self):
        self.mean = SENTINEL2L2A_MEAN
        self.std = SENTINEL2L2A_STD


class UntokSen2L1CTransform(UntokTransform):

    def __init__(self):
        self.mean = SENTINEL2L1C_MEAN
        self.std = SENTINEL2L1C_STD


class UntokSen2RGBTransform(UntokTransform):

    def __init__(self):
        self.mean = SENTINEL2RGB_MEAN
        self.std = SENTINEL2RGB_STD


# For the untokenized S1 samples during pre-training
class UntokSen1GRDTransform(UntokTransform):

    def __init__(self):
        self.mean = SENTINEL1GRD_MEAN
        self.std = SENTINEL1GRD_STD


class UntokSen1RTCTransform(UntokTransform):

    def __init__(self):
        self.mean = SENTINEL1RTC_MEAN
        self.std = SENTINEL1RTC_STD


class UntokDEMTransform(UntokTransform):

    def __init__(self):
        self.mean = DEM_MEAN
        self.std = DEM_STD


class RGBTransform(ImageTransform):

    def __init__(self, imagenet_default_mean_and_std=True, color_jitter=False, color_jitter_strength=0.5):
        self.rgb_mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        self.rgb_std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        self.color_jitter = color_jitter
        self.color_jitter_transform = self.random_color_jitter(color_jitter_strength)

    def random_color_jitter(self, strength=0.5):
        # Color Jitter from Pix2Seq and SimCLR
        # Source: https://github.com/google-research/pix2seq/blob/main/data/data_utils.py#L114
        t = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.8 * strength, contrast=0.8 * strength, saturation=0.8 * strength,
                                         hue=0.2 * strength)], p=0.8),
            T.RandomApply([T.Grayscale(num_output_channels=3)], p=0.2),
        ])

        return t

    def rgb_to_tensor(self, img):
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.rgb_mean, std=self.rgb_std)
        return img

    def load(self, path):
        # TODO: Instead of converting to RGB here, do it either in the preprocess or the postprocess step. Makes it compatible with wds dataloading.
        sample = self.pil_loader(path)
        return sample

    def preprocess(self, sample):
        sample = sample.convert('RGB')

        if self.color_jitter:
            sample = self.color_jitter_transform(sample)

        return sample

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        img = self.image_crop_and_resize(img, crop_coords, target_size, resample_mode=resample_mode)
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample):
        sample = self.rgb_to_tensor(sample)
        return sample


class DepthTransform(ImageTransform):

    def __init__(self, standardize_depth=True):
        self.standardize_depth = standardize_depth

    def depth_to_tensor(self, img):
        img = torch.Tensor(img / (2 ** 16 - 1.0))
        img = img.unsqueeze(0)  # 1 x H x W
        if self.standardize_depth:
            img = self.truncated_depth_standardization(img)
        return img

    @staticmethod
    def truncated_depth_standardization(depth, thresh: float = 0.1):
        """Truncated depth standardization

        :param depth: Depth map
        :param thresh: Threshold
        :return: Robustly standardized depth map
        """
        # Flatten depth and remove bottom and top 10% of values
        trunc_depth = torch.sort(depth.reshape(-1), dim=0)[0]
        trunc_depth = trunc_depth[int(thresh * trunc_depth.shape[0]): int((1 - thresh) * trunc_depth.shape[0])]
        return (depth - trunc_depth.mean()) / torch.sqrt(trunc_depth.var() + 1e-6)

    def load(self, path):
        sample = self.pil_loader(path)
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        img = self.image_crop_and_resize(img, crop_coords, target_size, resample_mode=resample_mode)
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample):
        sample = np.array(sample)
        sample = self.depth_to_tensor(sample)
        return sample


class NormalTransform(ImageTransform):

    def __init__(self, standardize_surface_normals=False):
        self.normal_mean = (0.5, 0.5, 0.5) if not standardize_surface_normals else IMAGENET_SURFACE_NORMAL_MEAN
        self.normal_std = (0.5, 0.5, 0.5) if not standardize_surface_normals else IMAGENET_SURFACE_NORMAL_STD

    def normal_to_tensor(self, img):
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.normal_mean, std=self.normal_std)
        return img

    def load(self, path):
        sample = self.pil_loader(path)
        return sample

    def preprocess(self, sample):
        return sample

    def image_hflip(self, img: Image, flip: bool):
        if flip:
            img = TF.hflip(img)
            flipped_np = np.array(img)
            flipped_np[:, :, 0] = 255 - flipped_np[:, :, 0]
            img = Image.fromarray(flipped_np)

        return img

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        img = self.image_crop_and_resize(img, crop_coords, target_size, resample_mode=resample_mode)
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample):
        sample = self.normal_to_tensor(sample)
        return sample


class SemsegTransform(ImageTransform):

    def __init__(self, scale_factor=1.0, shift_idx_by_one=False, id_mapping: Optional[Dict] = None,
                 select_channel=None):
        self.scale_factor = scale_factor
        self.shift_idx_by_one = shift_idx_by_one
        self.id_mapping = id_mapping
        self.select_channel = select_channel

    def map_semseg_values(self, sample):
        sample = np.asarray(sample)
        mapping_fn = lambda x: self.id_mapping.get(x, x)
        sample = np.vectorize(mapping_fn)(sample)
        sample = Image.fromarray(sample, mode='P')
        return sample

    def semseg_to_tensor(self, img):
        # Rescale to scale factor
        if self.scale_factor != 1.0:
            target_height, target_width = int(img.height * self.scale_factor), int(img.width * self.scale_factor)
            img = img.resize((target_width, target_height))
        # Using pil_to_tensor keeps it in uint8, to_tensor converts it to float (rescaled to [0, 1])
        img = TF.pil_to_tensor(img).to(torch.long).squeeze(0)
        # 255->0, 254->0, all else shifted up by one
        return img

    def load(self, path):
        sample = self.pil_loader(path)
        if self.select_channel is not None:
            sample = sample.split()[self.select_channel]
        return sample

    def preprocess(self, sample):
        sample = sample.convert('P')

        if self.id_mapping is not None:
            sample = self.map_semseg_values(sample)

        if self.shift_idx_by_one:
            sample = np.asarray(sample)
            sample = sample + 1
            sample = Image.fromarray(sample, mode='P')

        return sample

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        # Value for padding with TF.crop is always 0.
        # Override resampling mode to 'nearest' for semseg
        img = self.image_crop_and_resize(img, crop_coords, target_size, resample_mode='nearest')
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample):
        img = self.semseg_to_tensor(sample)
        return img


class MaskTransform(ImageTransform):

    def __init__(self, mask_pool_size=1):
        assert isinstance(mask_pool_size, int)
        self.mask_pool_size = mask_pool_size  # Use to expand masks

    def mask_to_tensor(self, img):
        mask = TF.to_tensor(img)
        if self.mask_pool_size > 1:
            mask = reduce(mask, 'c (h1 h2) (w1 w2) -> c h1 w1', 'min', h2=self.mask_pool_size, w2=self.mask_pool_size)
            mask = repeat(mask, 'c h1 w1 -> c (h1 h2) (w1 w2)', h2=self.mask_pool_size, w2=self.mask_pool_size)
        return (mask == 1.0)

    def load(self, path):
        sample = self.pil_loader(path)
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(self, img, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        # Override resampling mode to 'nearest' for masks
        img = self.image_crop_and_resize(img, crop_coords, target_size, resample_mode='nearest')
        img = self.image_hflip(img, flip)
        return img

    def postprocess(self, sample):
        sample = self.mask_to_tensor(sample)
        return sample


class TokTransform(AbstractTransform):

    def __init__(self):
        pass

    def load(self, path):
        sample = np.load(path).astype(int)
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(self, v, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        if rand_aug_idx is None:
            raise ValueError(
                "Crop settings / augmentation index are missing but a pre-tokenized modality is being used")
        v = torch.tensor(v[rand_aug_idx])
        return v

    def postprocess(self, sample):
        return sample


class DetectionTransform(AbstractTransform):

    def __init__(self, det_threshold=0.6, det_max_instances=None, bbox_order='dist_to_orig', coord_bins=1000,
                 min_visibility=0.0, return_raw=False):
        self.det_threshold = det_threshold
        self.det_max_instances = det_max_instances
        self.coord_bins = coord_bins
        self.min_visibility = min_visibility
        self.return_raw = return_raw

        if bbox_order == 'area':
            self.bbox_order = self.order_bboxes_by_area
        elif bbox_order == 'score':
            self.bbox_order = self.order_bboxes_by_score
        elif bbox_order == 'random':
            self.bbox_order = self.shuffle_bboxes
        else:
            self.bbox_order = self.order_bboxes_by_dist_to_orig

    @staticmethod
    def order_bboxes_by_area(bboxes):
        return sorted(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)

    @staticmethod
    def order_bboxes_by_dist_to_orig(bboxes):
        return sorted(bboxes, key=lambda x: x[0] ** 2 + x[1] ** 2)

    @staticmethod
    def order_bboxes_by_score(bboxes):
        return sorted(bboxes, key=lambda x: x[5], reverse=True)

    @staticmethod
    def shuffle_bboxes(bboxes):
        return sorted(bboxes, key=lambda x: random.random())

    def convert_detection_instance(self, instances):
        """Convert instances dict to list of lists where each list takes the form:
        [xmin, ymin, xmax, ymax, class_name, score]
        """

        instances = [inst['boxes'] + [inst['class_name'], inst['score']] for inst in instances if
                     inst['score'] >= self.det_threshold]
        return instances

    def bboxes_hflip(self, bboxes: List[Tuple], image_size: Tuple, flip: bool):
        image_height, image_width = image_size
        if flip:
            bboxes = [tuple(A.bbox_hflip(bbox[:4], rows=image_height, cols=image_width)) + tuple(bbox[4:])
                      for bbox in bboxes]

        return bboxes

    def bboxes_crop_and_resize(self, bboxes: List[Tuple], crop_coords: Tuple, orig_size: Tuple):
        """Crop and resize bounding boxes

        Args:
            bboxes: Bounding boxes to crop and resize
            crop_coords: Coordinates of the crop (top, left, h, w)
            orig_size: Size of the original image

        Returns:
            Cropped and resized bounding boxes
        """
        orig_height, orig_width = orig_size
        top, left, h, w = crop_coords
        xmin, ymin, xmax, ymax = left, top, left + w, top + h
        bboxes = [tuple(A.bbox_crop(bbox[:4], x_min=xmin, y_min=ymin, x_max=xmax, y_max=ymax, rows=orig_height,
                                    cols=orig_width)) + tuple(bbox[4:])
                  for bbox in bboxes]
        bboxes = A.core.bbox_utils.filter_bboxes(bboxes, rows=h, cols=w, min_visibility=self.min_visibility)
        # No need to resize, bounding boxes in albumentations format are scale invariant

        return bboxes

    def order_and_filter_bboxes(self, bboxes):
        if self.det_max_instances is not None and len(bboxes) > self.det_max_instances:
            bboxes = self.order_bboxes_by_score(bboxes)[:self.det_max_instances]

        return self.bbox_order(bboxes)

    def convert_bboxes_to_string(self, bboxes: List[Tuple]):
        """Convert bounding boxes to a string

        Args:
            bboxes: Bounding boxes

        Returns:
            String representation of the bounding boxes
        """
        # Remove score, quantize coordinates
        bins = self.coord_bins

        bboxes = [
            [
                f"xmin={round(xmin * (bins - 1))}",
                f"ymin={round(ymin * (bins - 1))}",
                f"xmax={round(xmax * (bins - 1))}",
                f"ymax={round(ymax * (bins - 1))}",
                cls,
            ]
            for (xmin, ymin, xmax, ymax, cls, score) in bboxes
        ]
        # Convert each bounding box to a string
        bboxes = [' '.join(b) for b in bboxes]
        # Convert the list to a str
        return ' '.join(bboxes)

    def load(self, path):
        with open(path, 'r') as f:
            sample = json.load(f)

        return sample

    def preprocess(self, sample):
        instances = sample['instances']
        return self.convert_detection_instance(instances)

    def image_augment(self, bboxes: List[Tuple], crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx=None, resample_mode: str = None):
        bboxes = self.bboxes_crop_and_resize(bboxes, crop_coords, orig_size)
        bboxes = self.bboxes_hflip(bboxes, target_size, flip)
        bboxes = self.order_and_filter_bboxes(bboxes)
        return bboxes

    def postprocess(self, bboxes):
        if self.return_raw:
            return bboxes
        bboxes = self.convert_bboxes_to_string(bboxes)
        return bboxes


class CaptionTransform(AbstractTransform):

    def __init__(self, aligned_captions=True, no_aug=False):
        self.aligned_captions = aligned_captions
        self.no_aug = no_aug

    def load(self, path):
        # Caption can either be stored as .txt or .json.gz (in which case it's a list of dicts)
        if path.endswith('.txt'):
            sample = Path(path).read_text()
        elif path.endswith('.json'):
            with open(path, 'r') as f:
                sample = json.load(f)
        elif path.endswith('.json.gz'):
            with gzip.open(path, 'rb') as f:
                sample = json.load(f)
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(self, val, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):

        if isinstance(val, list) or isinstance(val, tuple):
            if self.aligned_captions:
                val = val[0] if rand_aug_idx is None else val[rand_aug_idx]
            else:
                val = random.choice(val) if not self.no_aug else val[0]

        if isinstance(val, dict):
            # If each caption is saved as a dict, extract the string
            val = val["caption"]
        assert isinstance(val, str)

        return val

    def postprocess(self, sample):
        return sample


class CoordsTransform(AbstractTransform):

    def load(self, path):
        sample = Path(path).read_text()
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(self, val, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        return val

    def postprocess(self, sample):
        return sample


class CropSettingsTransform(AbstractTransform):

    def load(self, path):
        sample = np.load(path).astype(int)
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(self, val, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        raise NotImplementedError("CropSettingsTransform is not meant to be used for image augmentation")

    def postprocess(self, sample):
        return sample


class IdentityTransform(AbstractTransform):

    def load(self, path):
        raise NotImplementedError("IdentityTransform does not support loading")

    def preprocess(self, sample):
        return sample

    def image_augment(self, val, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        return val

    def postprocess(self, sample):
        return sample


class JSONTransform(AbstractTransform):

    def load(self, path):
        if path.endswith('.json'):
            with open(path, 'r') as f:
                sample = json.load(f)
        elif path.endswith('.json.gz'):
            with gzip.open(path, 'rb') as f:
                sample = json.load(f)
        return sample

    def preprocess(self, sample):
        return sample

    def image_augment(self, val, crop_coords: Tuple, flip: bool, orig_size: Tuple, target_size: Tuple,
                      rand_aug_idx: Optional[int], resample_mode: str = None):
        return val

    def postprocess(self, sample):
        return sample
#--------------------------------

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)

IMAGENET_SURFACE_NORMAL_MEAN = (0.501, 0.405, 0.137)
IMAGENET_SURFACE_NORMAL_STD = (0.114, 0.165, 0.081)

SEG_IGNORE_INDEX = 255
SEG_IGNORE_INDEX_V2 = 0
PAD_MASK_VALUE = 254
COCO_SEMSEG_NUM_CLASSES = 133 + 1  # One extra class for no-class
ADE20K_SEMSEG_NUM_CLASSES = 150 + 1  # One extra class for no-class
HYPERSIM_SEMSEG_NUM_CLASSES = 41

SENTINEL1GRD_MEAN = (-12.599, -20.293) #VV, VH
SENTINEL1GRD_STD = (5.195, 5.890) #VV, VH
SENTINEL1RTC_MEAN = (-10.93, -17.329) #VV, VH
SENTINEL1RTC_STD = (4.391, 4.459) #VV, VH


SENTINEL2L2A_MEAN = (1794.311, 1925.161, 2183.128, 2338.041, 2667.254, 3233.633, 3460.960, 3555.569, 3619.542, 3703.298, 3406.497, 2841.645)
SENTINEL2L2A_STD = (1164.883, 1205.586, 1223.713, 1399.638, 1403.298, 1378.513, 1434.924, 1491.141, 1454.089, 1660.395, 1473.248, 1365.080)
SENTINEL2L1C_MEAN = (3357.089, 3137.385, 3018.788, 3082.986, 3295.651, 3854.537, 4122.849, 4040.56, 4306.481, 2473.847, 1506.07, 3472.825, 2838.929)
SENTINEL2L1C_STD = (1624.683, 1675.806, 1557.708, 1833.702, 1823.738, 1733.977, 1732.131, 1679.732, 1727.26, 1024.687, 442.165, 1331.411, 1160.419)

SENTINEL2RGB_MEAN = (87.271, 80.931, 66.667)
SENTINEL2RGB_STD = (58.767, 47.663, 42.631)

DEM_MEAN = (435.726,)
DEM_STD = (560.326,)

NDVI_MEAN = (0.207,)
NDVI_STD = (0.398,)

NUM_TIMESTAMPS = 4

ORIGINAL_IMAGE_SIZES = {"sen1grd@264":264,
                        "sen1rtc@264":264,
                        "sen2l2a@264":264,
                        "sen2l1c@264":264,
                      }

CROP_IMAGE_SIZES = {"sen1grd@264":256,
                    "sen1rtc@264":256,
                    "sen2l2a@264":256,
                    "sen2l1c@264":256,
                }

IMAGE_TASKS = {'rgb', 'depth', 'semseg', 'semseg_hypersim', 'semseg_coco', 'semseg_ade20k', 'normal'}
DETECTION_TASKS = {'det'} # 'det_coco', 'det_lvis'
TEXT_TASKS = {'caption'}
VISION_TASKS = IMAGE_TASKS | DETECTION_TASKS
SEQUENCE_TASKS = DETECTION_TASKS | TEXT_TASKS

NYU_MEAN = 2070.7764
NYU_STD = 777.5723

#--------------------------------


def generate_uint15_hash(seed_str):
    """Generates a hash of the seed string as an unsigned int15 integer"""
    return int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest(), 16) % (2**15)

#--------------------------------

class SequenceEncoderEmbedding(nn.Module):
    """Embedding module for encoding sequence inputs, like captions or a sequence of objects.

    Args:
        vocab_size: Vocabulary size
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 1D sin-cos positional embeddings
        max_sincos_pos_emb: Maximum allowed length for sin-cos positional embeddings
        padding_idx: Padding index for word embedding
    """

    def __init__(self,
                 vocab_size: int,
                 max_length: int,
                 dim_tokens: int | None = None,
                 sincos_pos_emb: bool = True,
                 max_sincos_pos_emb: int = 512,
                 padding_idx: int = 0,
                 **kwargs
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.padding_idx = padding_idx
        self.max_sincos_pos_emb = max_sincos_pos_emb

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of embedding module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        if self.sincos_pos_emb:
            if self.max_length > self.max_sincos_pos_emb:
                raise ValueError(f"Max length ({self.max_length}) is greater than the number of posembs ({self.max_sincos_pos_emb}")
            pos_emb = build_1d_sincos_posemb(max_len=self.max_sincos_pos_emb, embed_dim=self.dim_tokens)[:self.max_length]
            self.register_buffer("pos_emb", pos_emb) # self.pos_emb is now a buffer for FSDP
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_length, self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding
        self.token_emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.dim_tokens,
                                     padding_idx=self.padding_idx)


    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d : torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming sequence of ids to sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (dict[str, torch.Tensor]): Modality dict with at least the following keys:
                - 'tensor' (torch.Tensor): Input token sequence for each batch. Shape (B, L) where B is the batch size and L is the sequence length.
                - 'input_mask' (torch.Tensor): Mask for valid tokens in the input sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, L, D).
        """
        if not isinstance(d, dict):
            d = {
                'tensor': d,
                'input_mask': torch.zeros_like(d, dtype=torch.bool),  # No masking
            }

        ids = d['tensor']
        B = ids.shape[0]
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'

        # Map to embedding
        x = self.token_emb(ids)

        expanded_pos_emb = repeat(self.pos_emb, "() n d -> b n d", b=B)
        # Input pos encoding
        input_mask = d['input_mask']
        input_pos_id = (~input_mask).int().cumsum(dim=1) - 1
        input_pos_id[input_mask] = 0
        input_pos_emb = torch.gather(expanded_pos_emb, dim=1, index=repeat(input_pos_id, "b n -> b n d", d=expanded_pos_emb.shape[2]))
        input_pos_emb[input_mask] = 0

        x_emb = input_pos_emb + self.mod_emb

        d['x'] = x
        d['emb'] = x_emb
        return d
    
class ImageTokenEncoderEmbedding(nn.Module):
    """Embedding module for tokenized spatial inputs.

    Args:
        vocab_size: Vocabulary size
        patch_size: Int or tuple of the patch size over the full image size.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        image_size: Default image size. Used to initialize size of positional embeddings.
    """
    def __init__(self,
                 vocab_size: int,
                 patch_size: int | tuple[int,int] = 16,
                 dim_tokens: int | None = None,
                 sincos_pos_emb: bool = True,
                 image_size: int | tuple[int] = 224,
                 **kwargs):

        super().__init__()
        self.vocab_size = vocab_size
        self.patch_size = pair(patch_size)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // patch_size) * (self.image_size[1] // patch_size)

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // self.patch_size[0]
        w_posemb = self.image_size[1] // self.patch_size[1]
        if self.sincos_pos_emb:
            pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.register_buffer("pos_emb", pos_emb) # self.pos_emb is now a buffer for FSDP
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, (h_posemb * w_posemb), self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding
        self.token_emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.dim_tokens)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming image tokens to a sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (torch.Tensor, dict[str, torch.Tensor]): Modality dict with at least the following key:
                - 'tensor' (torch.Tensor): Input image tokens for each batch. Shape (B, H, W) where B is the batch size, and H, W are height and width of the tokenized image.                - 'input_mask' (torch.Tensor): Mask for valid tokens in the input sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            dict[str, torch.Tensor]: Modality dictionary with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, H*W, D).
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, H*W, D).
        """
        if not isinstance(d, dict):
            d = {'tensor': d}

        ids = d['tensor']
        B = ids.shape[0]
        ids = ids.reshape(B, -1)

        # Map to embedding
        x = self.token_emb(ids)

        # Create positional embedding + modality embedding
        x_emb = repeat(self.pos_emb + self.mod_emb, '() n d -> b n d', b=B)

        d['x'] = x
        d['emb'] = x_emb

        return d


class ImageEncoderEmbedding(nn.Module):
    """Embedding module for spatial inputs, like images or feature maps.
    Creates tokens from patches over the image.

    This adapter / embedding differs from the one of MultiMAE by taking as input a dict and
     separating positional embeddings and modality embeddings from the input projection
     Input projection is 'x', posemb + modemb is 'emb'

    Args:
        num_channels: Number of input channels of the image/feature map
        patch_size: Int or tuple of the patch size over the full image size.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        image_size: Default image size. Used to initialize size of positional embeddings.
    """
    def __init__(self,
                 num_channels: int,
                 patch_size: int | tuple[int,int],
                 dim_tokens: int | None = None,
                 sincos_pos_emb: bool = True,
                 image_size: int | tuple[int] = 224,
                 **kwargs
                 ):

        super().__init__()
        self.num_channels = num_channels
        self.patch_size = pair(patch_size)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // patch_size) * (self.image_size[1] // patch_size)

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of encoder that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // self.patch_size[0]
        w_posemb = self.image_size[1] // self.patch_size[1]
        if self.sincos_pos_emb:
            pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.register_buffer("pos_emb", pos_emb) # self.pos_emb is now a buffer for FSDP
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, (h_posemb * w_posemb), self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Image -> tokens projection
        # No bias term here, so modality embedding fully comes from self.mod_emb
        self.proj = nn.Linear(self.num_channels * self.patch_size[0] * self.patch_size[1], self.dim_tokens, bias=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, d: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming image to sequence of tokens.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (torch.Tensor, dict[str, torch.Tensor]): Modality dict with at least the following key:
                - 'tensor' (torch.Tensor): Input image for each batch. Shape (B, C, H, W) where B is the batch size, C is the number of channels, and H, W are height and width of the image.

                
        Returns:
            dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, (H / PH) * (W / PW), D), where PH and PW are the patch sizes
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the input sequence. Shape (B, (H / PH) * (W / PW), D)
        """
        if not isinstance(d, dict):
            d = {'tensor': d}

        x = d['tensor']
        B, C, H, W = x.shape
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'
        assert (H % self.patch_size[0] == 0) and (W % self.patch_size[1] == 0), f'Image sizes {H}x{W} must be divisible by patch sizes {self.patch_size[0]}x{self.patch_size[1]}'

        # Create patches [B, C, H, W] -> [B, (H*W), C]
        x_patch = self.proj(rearrange(x, 'b d (nh ph) (nw pw) -> b (nh nw) (ph pw d)', ph=self.patch_size[0], pw=self.patch_size[1]))

        if (H, W) != self.image_size:
            # Interpolate embedding if required
            pos_emb = self.interpolate_pos_encoding(self.pos_emb.clone(), H, W)
        else:
            pos_emb = self.pos_emb

        # Create positional embedding + modality embedding
        x_emb = repeat(pos_emb + self.mod_emb, '() n d -> b n d', b=B)

        d['x'] = x_patch
        d['emb'] = x_emb

        return d

    def interpolate_pos_encoding(self, pos_embeddings: torch.Tensor, height, width) \
            -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - transformers.models.vit.modeling_vit.ViTEmbeddings.interpolate_pos_encoding
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_positions = pos_embeddings.shape[1]
        new_height = height // self.patch_size[0]
        new_width = width // self.patch_size[1]

        # Assuming squared default image size
        sqrt_num_positions = int(num_positions**0.5)
        pos_embeddings = pos_embeddings.reshape(1, sqrt_num_positions, sqrt_num_positions, self.dim_tokens)
        pos_embeddings = pos_embeddings.permute(0, 3, 1, 2)

        pos_embeddings = nn.functional.interpolate(
            pos_embeddings,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        pos_embeddings = pos_embeddings.permute(0, 2, 3, 1).view(1, -1, self.dim_tokens)

        return pos_embeddings


#--------------------------------


class SequenceDecoderEmbedding(nn.Module):
    """Embedding module for sequence inputs, like captions or a sequence of objects.

    Args:
        vocab_size: Vocabulary size
        max_length: Maximum number of tokens in the sequence
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 1D sin-cos positional embeddings
        padding_idx: Padding index for word embedding
        share_embedding: Set to True to share input and output embedding weights
    """
    def __init__(self,
                 vocab_size: int,
                 max_length: int,
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 max_sincos_pos_emb: int = 512,
                 padding_idx: int = 0,
                 share_embedding: bool = True,
                 **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.padding_idx = padding_idx
        self.max_sincos_pos_emb = max_sincos_pos_emb
        self.share_embedding = share_embedding

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of embedding module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes

        if self.sincos_pos_emb:
            if self.max_length > self.max_sincos_pos_emb:
                raise ValueError(f"Max length ({self.max_length}) is greater than the number of posembs ({self.max_sincos_pos_emb}")
            # Get all posembs, than truncate up to max length
            pos_emb = build_1d_sincos_posemb(max_len=self.max_sincos_pos_emb, embed_dim=self.dim_tokens)[:self.max_length]
            self.register_buffer("pos_emb", pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.max_length, self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding
        self.token_emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.dim_tokens, padding_idx=self.padding_idx)

        # Output projection layer
        self.to_logits = nn.Linear(self.dim_tokens, self.vocab_size, bias=False)

        if self.share_embedding:
            # Share input and output embedding weights
            self.to_logits.weight = self.token_emb.weight


    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward_embed(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through embedding module, transforming sequence of ids to sequence of embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict, with at least the following keys:
                - 'tensor' (torch.Tensor): Token sequence for each batch. Shape (B, L) where B is the batch size and L is the sequence length.
                - 'target_mask' (torch.Tensor): Mask for valid tokens in the target sequence (set to 0 for valid tokens and 1 otherwise). Shape (B, L).

        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence. Shape (B, L, D) where D is the embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the target sequence. Shape (B, L, D).
                - 'ids' (torch.Tensor): Original token sequence from input dict. Shape (B, L).
        """
        ids = d['tensor']
        B = ids.shape[0]
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'

        # Map to embedding
        x = self.token_emb(ids)

        expanded_pos_emb = repeat(self.pos_emb, "() n d -> b n d", b=B)

        # Target pos encoding
        target_mask = d['target_mask']
        target_pos_id = (~target_mask).int().cumsum(dim=1) - 1
        target_pos_id[target_mask] = 0
        # Sometimes target sequence is over max length, it will be truncated in decoder
        target_pos_id[target_pos_id >= self.max_length] = 0
        target_pos_emb = torch.gather(expanded_pos_emb, dim=1, index=repeat(target_pos_id, "b n -> b n d", d=expanded_pos_emb.shape[2]))
        target_pos_emb[target_mask] = 0

        x_emb = target_pos_emb + self.mod_emb


        d['x'] = x
        d['emb'] = x_emb
        d['ids'] = d['tensor']

        return d

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through output projection layer, transforming sequence of embeddings to logits.

        Args:
            x (torch.Tensor): Output tokens from the decoder. Shape (B, M, D)
        
        Returns:
            torch.Tensor: Logits for each token in the sequence. Shape (B, M, V)
        """
        logits = self.to_logits(x)
        return logits



class ImageTokenDecoderEmbedding(nn.Module):
    """Embedding module for tokenized spatial inputs.

    Args:
        vocab_size: Vocabulary size
        patch_size: Int or tuple of the patch size over the full image size.
        dim_tokens: Dimension of output tokens. Can be set using init method.
        sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
        image_size: Default image size. Used to initialize size of positional embeddings.
        share_embedding: Set to True to share input and output embedding weights
    """
    def __init__(self,
                 vocab_size: int,
                 patch_size: Union[int, Tuple[int,int]] = 16,
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = True,
                 image_size: Union[int, Tuple[int]] = 224,
                 share_embedding: bool = True,
                 **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_size = pair(patch_size)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        self.share_embedding = share_embedding

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768, init_std=0.02):
        """
        Initialize parts of module that are dependent on dimension of tokens.
        Should be called when setting up FourM.

        Args:
            dim_tokens: Dimension of tokens
            init_std: Standard deviation of init
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // self.patch_size[0]
        w_posemb = self.image_size[1] // self.patch_size[1]
        if self.sincos_pos_emb:
            pos_emb = build_2d_sincos_posemb(h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens)
            self.register_buffer("pos_emb", pos_emb)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, (h_posemb * w_posemb), self.dim_tokens))
            nn.init.normal_(self.pos_emb, std=init_std)

        self.mod_emb = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))
        nn.init.normal_(self.mod_emb, std=init_std)

        # Token embedding (not needed if only masked tokens are given as input, but can be useful to train Token Critic)
        self.token_emb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.dim_tokens)

        # Output projection layer
        self.to_logits = nn.Linear(self.dim_tokens, self.vocab_size, bias=False)

        if self.share_embedding:
            # Share input and output embedding weights
            self.to_logits.weight = self.token_emb.weight

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward_embed(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the embedding module, transforming tokenized spatial inputs to embeddings.
        Creates corresponding modality and positional embeddings and adds them to the dict.

        Args:
            d (Dict[str, torch.Tensor]): Modality dict, with at least the following key:
                - 'tensor' (torch.Tensor): Modality tokens for each batch (e.g. from tokenized images). Shape (B, H, W) where B is the batch size, H and W are height and width after tokenization.


        Returns:
            Dict[str, torch.Tensor]: Modality dict with added keys:
                - 'x' (torch.Tensor): Embedded token sequence, which is replaced by mask tokens in the 4M decoder. Shape (B, H*W, D) where D is the embedding dimension.
                - 'emb' (torch.Tensor): Sum of positional and modality embeddings for the token sequence. Shape (B, H*W, D).
                - 'ids' (torch.Tensor): Reshaped token sequence from input dict, flattened in the spatial dimensions. Shape (B, H*W).
        """
        ids = d['tensor']
        B = ids.shape[0]
        ids = ids.reshape(B, -1)

        # Map to embedding
        x = self.token_emb(ids)

        # Create positional embedding + modality embedding
        x_emb = repeat(self.pos_emb + self.mod_emb, '() n d -> b n d', b=B)

        d['x'] = x
        d['emb'] = x_emb
        d['ids'] = ids
        return d

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through output projection layer, transforming sequence of embeddings to logits.

        Args:
            x (torch.Tensor): Output tokens from the decoder. Shape (B, M, D)
        
        Returns:
            torch.Tensor: Logits for each token in the sequence. Shape (B, M, V)
        """
        logits = self.to_logits(x)
        return logits

#--------------------------------

MODALITY_INFO = {
     'sen1grd@264': {
        'input_size': 264,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=2),
        'decoder_embedding': partial(ImageEncoderEmbedding, num_channels=2),
        # 'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'num_channels': 2,
        'id': generate_uint15_hash('sen1grd@264'),
        'path': 'S1GRD/',
    },
    'sen1rtc@264': {
        'input_size': 264,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=2),
        'decoder_embedding': partial(ImageEncoderEmbedding, num_channels=2),
        # 'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': None,  # Will be set to 196
        'type': 'img',
        'num_channels': 2,
        'id': generate_uint15_hash('sen1rtc@264'),
        'path': 'S1RTC/',
    },
    'sen2l2a@264': {
        'input_size': 264,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=12),
        'decoder_embedding': partial(ImageEncoderEmbedding, num_channels=12),
        # 'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'num_channels': 12,
        'id': generate_uint15_hash('sen2l2a@264'),
        'path': 'S2L2A/',
    },
    'sen2l1c@264': {
        'input_size': 264,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=12),
        'decoder_embedding': partial(ImageEncoderEmbedding, num_channels=12),
        # 'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'num_channels': 13,
        'id': generate_uint15_hash('sen2l1c@264'),
        'path': 'S2L1C/',
    },
    'sen2rgb@264': {
        'input_size': 264,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=3),
        'decoder_embedding': partial(ImageEncoderEmbedding, num_channels=3),
        # 'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'num_channels': 3,
        'id': generate_uint15_hash('sen2rgb@264'),
        'path': 'S2RGB/',
    },
    'lulc@264': {
        'input_size': 264,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=9),
        'decoder_embedding': partial(ImageEncoderEmbedding, num_channels=9),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'num_channels': 9,
        'id': generate_uint15_hash('lulc@264'),
        'path': 'LULC/',
    },
    'dem@264': {
        'input_size': 264,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=1),
        'decoder_embedding': partial(ImageEncoderEmbedding, num_channels=1),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'num_channels': 1,
        'id': generate_uint15_hash('dem@264'),
        'path': 'DEM/',
    },
    'ndvi@264': {
        'input_size': 264,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=1),
        'decoder_embedding': partial(ImageEncoderEmbedding, num_channels=1),
        'min_tokens': 0,
        'max_tokens': None,  # Will be set to 196
        'type': 'img',
        'num_channels': 1,
        'id': generate_uint15_hash('ndvi@264'),
        'path': 'NDVI/',
    },
    'untok_sen2l2a@224': {  # untokenized version
        'input_size': 224,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=12),
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'num_channels': 12,
        'id': generate_uint15_hash('untok_sen2l2a@224'),
        'path': 'S2L2A_untokenized',
    },
    'untok_sen2l1c@224': {  # untokenized version
        'input_size': 224,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=13),
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': None,  # Will be set to 196
        'type': 'img',
        'num_channels': 13,
        'id': generate_uint15_hash('untok_sen2l1c@224'),
        'path': 'S2L1C_untokenized',
    },
    'untok_sen2rgb@224': {  # untokenized version
        'input_size': 224,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=13),
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': None,  # Will be set to 196
        'type': 'img',
        'num_channels': 3,
        'id': generate_uint15_hash('untok_sen2rgb@224'),
        'path': 'S2RGB_untokenized',
    },
    'untok_sen1grd@224': {  # untokenized version
        'input_size': 224,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=2),
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'num_channels': 2,
        'id': generate_uint15_hash('untok_sen1grd@224'),
        'path': 'S1GRD_untokenized',
    },
    'untok_sen1rtc@224': {  # untokenized version
        'input_size': 224,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=2),
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': None,  # Will be set to 196
        'type': 'img',
        'num_channels': 2,
        'id': generate_uint15_hash('untok_sen1rtc@224'),
        'path': 'S1RTC_untokenized',
    },
    'tok_sen1grd@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 15360,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=15360),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=15360),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_sen1grd@224'),
        'pretokenized': True,
        'path': 'S1GRD_tokens',
    },
    'tok_sen1rtc@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 15360,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=15360),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=15360),
        'min_tokens': 0,
        'max_tokens': None,  # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_sen1rtc@224'),
        'pretokenized': True,
        'path': 'S1RTC_tokens',
    },
    'tok_sen2l2a@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 15360,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=15360),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=15360),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_sen2l2a@224'),
        'pretokenized': True,
        'path': 'S2L2A_tokens',
    },
    'tok_lulc@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 4375,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=4375),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=4375),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_lulc@224'),
        'pretokenized': True,
        'path': 'LULC_tokens',
    },
    'untok_dem@224': {  # untokenized version
        'input_size': 224,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=1),
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': None,  # Will be set to 196
        'type': 'img',
        'num_channels': 1,
        'id': generate_uint15_hash('untok_dem@224'),
        'path': 'DEM_untokenized',
    },
    'tok_dem@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 15360,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=15360),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=15360),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_dem@224'),
        'pretokenized': True,
        'path': 'DEM_tokens',
    },
    'tok_ndvi@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 15360,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=15360),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=15360),
        'min_tokens': 0,
        'max_tokens': None,  # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_ndvi@224'),
        'pretokenized': True,
        'path': 'NDVI_tokens',
    },
    ### Natural image/text domains
    'rgb@224': {
        'input_size': 224,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=3),
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'num_channels': 3,
        'id': generate_uint15_hash('rgb@224'),
        'path': 'rgb',
    },
    'caption': {
        'vocab_size': 30_000,
        'encoder_embedding': partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        'decoder_embedding': partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        'min_tokens': 0,
        'max_tokens': 256,
        'type': 'seq',
        'id': generate_uint15_hash('caption'),
        'path': 'captions_txt',
    },
    'coords': {
        'vocab_size': 30_000,
        'encoder_embedding': partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        'decoder_embedding': partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        'min_tokens': 0,
        'max_tokens': 256,
        'type': 'seq',
        'id': generate_uint15_hash('coords'),
        'path': 'coords',
    },
    'det': { 
        'vocab_size': 30_000,
        'encoder_embedding': partial(SequenceEncoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        'decoder_embedding': partial(SequenceDecoderEmbedding, vocab_size=30_000, max_length=256, padding_idx=0),
        'min_tokens': 0,
        'max_tokens': 256,
        'type': 'seq',
        'id': generate_uint15_hash('det'),
    },
    'tok_rgb@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 16384,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=16384),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=16384),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_rgb@224'),
        'pretokenized': True,
    },
    'tok_depth@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_depth@224'),
        'pretokenized': True,
    },
    'tok_normal@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_normal@224'),
        'pretokenized': True,
    },
    'tok_semseg@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 4096,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=4096),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=4096),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_semseg@224'),
        'pretokenized': True,
    },
    'tok_clip@224': {
        'input_size': 224,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 196
        'type': 'img',
        'id': generate_uint15_hash('tok_clip@224'),
        'pretokenized': True,
    },
    ### 224->448 super resolution modalities
    'rgb@448': {
        'input_size': 448,
        'patch_size': 16,
        'encoder_embedding': partial(ImageEncoderEmbedding, num_channels=3),
        'decoder_embedding': None,
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'num_channels': 3,
        'id': generate_uint15_hash('rgb@448'),
        'path': 'rgb',
    },
    'tok_rgb@448': {
        'input_size': 448,
        'patch_size': 16,
        'vocab_size': 16384,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=16384),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=16384),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'id': generate_uint15_hash('tok_rgb@448'),
        'pretokenized': True,
    },
    'tok_depth@448': {
        'input_size': 448,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'id': generate_uint15_hash('tok_depth@448'),
        'pretokenized': True,
    },
    'tok_normal@448': {
        'input_size': 448,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'id': generate_uint15_hash('tok_normal@448'),
        'pretokenized': True,
    },
    'tok_semseg@448': {
        'input_size': 448,
        'patch_size': 16,
        'vocab_size': 4096,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=4096),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=4096),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'id': generate_uint15_hash('tok_semseg@448'),
        'pretokenized': True,
    },
    'tok_clip@448': {
        'input_size': 448,
        'patch_size': 16,
        'vocab_size': 8192,
        'encoder_embedding': partial(ImageTokenEncoderEmbedding, vocab_size=8192),
        'decoder_embedding': partial(ImageTokenDecoderEmbedding, vocab_size=8192),
        'min_tokens': 0,
        'max_tokens': None, # Will be set to 784
        'type': 'img',
        'id': generate_uint15_hash('tok_clip@448'),
        'pretokenized': True,
    },
}


def build_modality_embeddings(modalities, img_size=None, dim=None, patch_size=None):
    mod_embeddings = {}
    mod_name_mapping = {}
    for modality in modalities:
        # New modalities can be provided as {'name': <num_channels>}
        if isinstance(modality, dict):
            for key, value in modality.items():
                if isinstance(value, nn.Module):
                    mod_embeddings[key] = value
                elif isinstance(value, int):
                    mod_embeddings[key] = ImageEncoderEmbedding(num_channels=value, dim_tokens=dim, image_size=img_size,
                                                                    patch_size=patch_size, sincos_pos_emb=True)
                else:
                    raise ValueError(f'Modalities must be provided as a list of strings or dicts, or as a dict with '
                                     f'the values being nn.Module or int (number of channels of the modality). '
                                     f'Found {key}: {value} ({type(value)})')
                mod_name_mapping[key] = key
            continue

        # Cover multiple naming conventions
        modality_renamed = (modality.lower()
                            .replace('s2', 'sen2')
                            .replace('s1', 'sen1')
                            .replace('text', 'caption')
                            .replace('coordinates', 'coords')
                            )

        # Get modality key in MODALITY_INFO
        if 'sen2l2a' in modality_renamed:
            key = 'untok_sen2l2a@224'
        elif 'sen2l1c' in modality_renamed:
            key = 'untok_sen2l1c@224'
        elif 'sen1rtc' in modality_renamed:
            key = 'untok_sen1rtc@224'
        elif 'sen1grd' in modality_renamed:
            key = 'untok_sen1grd@224'
        elif 'rgb' in modality_renamed:
            key = 'untok_sen2rgb@224'
        elif 'dem' in modality_renamed:
            key = 'untok_dem@224'
        else:
            key = modality

        if key in MODALITY_INFO.keys():
            mod_info = MODALITY_INFO[key]
            mod_embeddings[key] = mod_info['encoder_embedding'](image_size=img_size, dim_tokens=dim, **mod_info)
            mod_name_mapping[modality] = key  # Requires manual mapping for loading model weights
        else:
            raise NotImplementedError(f'Could not find modality {modality} in default modality info.')

    return mod_embeddings, mod_name_mapping


class TerraMindViT(Encoder):
    """Modified TerraMind model, adapted to behave as a raw data-only ViT.

    Args:
        img_size (int): Input image size.
        modalities (list, dict, optional): List of modality keys and dicts, or dict with modality keys and values being
            ints (num_channels of modality) or nn.Module (patch embedding layer).
        merge_method (str, optional): Specify how the output is merged for further processing. One of 'mean', 'max',
            'concat', 'dict', or None. 'mean', 'max', and 'concat' are dropping all sequence modality tokens, split all
            image modality tokens and reduce the by applying the appropriate method. 'dict' splits all tokens into a
            dictionary {'modality': torch.Tensor}. Defaults to 'mean'.
        patch_size (int): Patch size.
        in_chans (int): Number of input image channels.
        dim (int): Patch embedding dimension.
        encoder_depth (int): Depth of ViT / number of encoder blocks.
        num_heads (int): Number of attention heads in each ViT block.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool): If True, add a learnable bias to query, key, value.
        proj_bias (bool): If True, adds a bias to the attention out proj layer.
        mlp_bias (bool): If True, adds a learnable bias for the feedforward.
        drop_path_rate (float): Stochastic depth rate.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        modality_drop_rate (float): Drop modality inputs during training.
        act_layer (nn.Module): Activation layer.
        norm_layer (nn.Module): Normalization layer.
        gated_mlp (bool): If True, makes the feedforward gated (e.g., for SwiGLU)
        qk_norm (bool): If True, normalizes the query and keys (as in ViT-22B)
        use_act_checkpoint (bool): If True, use activation checkpointing.
        encoder_norm (bool): If True, adds a norm layer after the last encoder block.
    """
    def __init__(
        self,

        # newww
        encoder_weights: str | Path,
        input_size: int,
        input_bands: dict[str, list[str]],
        output_layers: int | list[int],
        output_dim: int | list[int],
        download_url: str,
        # old
        encoder_norm: bool = True,
        img_size: int = 224,
        modalities: list | dict[str, int | nn.Module] | None = None,
        merge_method: str | None = 'mean',
        patch_size: int = 16,
        in_chans: int = 3,
        dim: int = 768,
        encoder_depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_bias: bool = True,
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        modality_drop_rate: float = 0.0,
        act_layer: torch.Tensor = nn.GELU,
        norm_layer: partial | nn.Module = partial(LayerNorm, eps=1e-6),
        gated_mlp: bool = False,  # Make the feedforward gated for e.g. SwiGLU
        qk_norm: bool = False,
    ):
        super().__init__(        # newww
        model_name = "terramind",
        input_bands = input_bands,
        input_size = input_size,
        embed_dim = dim,
        output_layers = output_layers,
        output_dim = output_dim ,
        multi_temporal = False,
        multi_temporal_output = False,
        pyramid_output = False,
        encoder_weights = encoder_weights,
        download_url = download_url,
        )

        # print(modalities)
       
        modalities = list(modalities)
        # print(type(modalities))

        if modalities is None or len(modalities) == 0:
            # Init new image modality
            modalities = [{'image': in_chans}]
        elif isinstance(modalities, dict):
            modalities = [modalities]
        elif not isinstance(modalities, list):
            raise ValueError(f'Modalities must be None, a list of modality keys or a dict with ints/embedding layers.')

        # Build embedding layers for all defined modalities
        mod_embeddings, mod_name_mapping = build_modality_embeddings(modalities, img_size=img_size, dim=dim,
                                                                     patch_size=patch_size)
        self.encoder_embeddings = nn.ModuleDict(mod_embeddings)
        self.mod_name_mapping = mod_name_mapping
        self.modalities = list(mod_name_mapping.keys())  # Further code expects list
        self.patch_size = patch_size

        self.img_size = img_size
        self.merge_method = merge_method
        self.image_modalities = [key for key, value in self.encoder_embeddings.items()
                                 if isinstance(value, ImageEncoderEmbedding)]
        self.modality_drop_rate = modality_drop_rate
        assert 0 <= self.modality_drop_rate <= 1, "modality_drop_rate must be in [0, 1]"
        # New learned parameter for handling missing modalities
        if modality_drop_rate and self.merge_method == 'concat':
            self.missing_mod_token = nn.Parameter(torch.Tensor(dim))

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, encoder_depth)]

        self.encoder = nn.ModuleList([
            Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias,
                  mlp_bias=mlp_bias, drop_path=dpr[i], drop=drop_rate, attn_drop=attn_drop_rate, act_layer=act_layer,
                  norm_layer=norm_layer, gated_mlp=gated_mlp, qk_norm=qk_norm)
            for i in range(encoder_depth)
        ])

        # Needed for terratorch decoders
        if merge_method == 'concat':
            self.out_channels = [dim * len(self.image_modalities) for i in range(encoder_depth)]
        else:
            self.out_channels = [dim for i in range(encoder_depth)]

        self.encoder_norm = norm_layer(dim) if encoder_norm else nn.Identity()

        # Weight init
        self.init_weights()

    def init_weights(self):
        """Weight initialization following MAE's initialization scheme"""

        for name, m in self.named_modules():
            # Skipping tokenizers to avoid reinitializing them
            if "tokenizer" in name:
                continue
            # Linear
            elif isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm) or isinstance(m, LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

            # Embedding
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            # Conv2d
            elif isinstance(m, nn.Conv2d):
                if '.proj' in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = set()

        for mod, emb_module in self.encoder_embeddings.items():
            if hasattr(emb_module, 'no_weight_decay'):
                to_skip = emb_module.no_weight_decay()
                to_skip = set([f'encoder_embeddings.{mod}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def forward(self, image: dict[str, torch.Tensor] | torch.Tensor | None = None, **kwargs) -> list[torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            d (dict, torch.Tensor): Dict of inputs or input tensor with shape (B, C, H, W)

            Alternatively, keyword arguments with modality=tensor.

        Returns:
            list[torch.Tensor]: List of transformer layer outputs. Shape (B, L, D).
        """
        # # Handle single image modality
        # if isinstance(d, torch.Tensor):
        #     # Assuming first modality
        #     d = {self.modalities[0]: d}
        # elif d is None:
        #     d = {}
        #     assert len(kwargs), "No input provided."

        # # Add additional keyword args to input dict
        # for key, value in kwargs.items():
        #     d[key] = value

        d = {}
        if "sar" in image:
            d["S1GRD"] = image["sar"].squeeze(2)
        if "optical" in image:
            d["S2L2A"] = image["optical"].squeeze(2)

        if self.training and self.modality_drop_rate:
            # Drop random modalities during training
            for key in random.sample(list(d.keys()), k=len(d) - 1):
                if random.random() < self.modality_drop_rate:
                    _ = d.pop(key)

        # print(self.mod_name_mapping)
        # print(image["optical"].shape)
        # print(image["sar"].shape)

        x = []
        num_tokens = []
        image_mod = []
        for mod, tensor in d.items():
            assert mod in self.mod_name_mapping.keys(), \
                f'No patch embedding layer found for modality {mod}.'

            # print(tensor.shape)
            # print(self.encoder_embeddings[self.mod_name_mapping[mod]])
            mod_dict = self.encoder_embeddings[self.mod_name_mapping[mod]](tensor)
            # print(mod_dict['x'].shape)
            # print(mod_dict['emb'].shape)
            # Add embeddings to patchified data
            x.append(mod_dict['x'] + mod_dict['emb'])
            num_tokens.append(mod_dict['x'].shape[-2])
            image_mod.append(self.mod_name_mapping[mod] in self.image_modalities)

        # Concatenate along token dim
        x = torch.cat(x, dim=1)  # Shape: (B, N, D)

        out = []

        # for i, block in enumerate(self.encoder):
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i in self.output_layers:
                out.append(x.clone())

        # print(out[0].shape)

        out[-1] = self.encoder_norm(x)  # Shape: (B, N, D)

        def _unstack_image_modalities(x):
            x = torch.split(x, num_tokens, dim=1)  # Split tokens by modality
            x = [m for m, keep in zip(x, image_mod) if keep]  # Drop sequence modalities
            x = torch.stack(x, dim=1)  # (B, M, N, D)
            return x

        # Merge tokens from different modalities
        if self.merge_method == 'mean':
            out = [_unstack_image_modalities(x) for x in out]
            out = [x.mean(dim=1) for x in out]

        elif self.merge_method == 'max':
            out = [_unstack_image_modalities(x) for x in out]
            out = [x.max(dim=1)[0] for x in out]

        elif self.merge_method == 'concat':
            out = [_unstack_image_modalities(x) for x in out]
            if len(d) < len(self.image_modalities):
                # Handle missing modalities with missing_mod_token
                num_missing = len(self.image_modalities) - len(d)
                missing_tokens = self.missing_mod_token.repeat(out[-1].shape[0], num_missing, out[-1].shape[2], 1)
                out = [torch.cat([x, missing_tokens], dim=1) for x in out]
            # Concat along embedding dim
            out = [torch.cat(x.unbind(dim=1), dim=-1) for x in out]

        elif self.merge_method == 'dict':
            out = [torch.split(x, num_tokens, dim=1) for x in out]
            out = [{mod: x[i] for i, mod in enumerate(d.keys())} for x in out]

        elif self.merge_method is None:
            pass  # Do nothing
        else:
            raise NotImplementedError(f'Merging method {self.merge_method} is not implemented. '
                                      f'Select one of mean, max or concat.')

        # print(out[0].shape)
        # print(out[-1].shape)
        # print(out[0].shape)

        out = [
            x.permute(0, 2, 1)
            .view(
                x.shape[0],
                -1,
                self.img_size // self.patch_size,
                self.img_size // self.patch_size,
            )
            .contiguous()
            for x in out
        ]

        # print(out[0].shape)

        return out

        # return final_output/


    def load_encoder_weights(self, logger: Logger) -> None:
        pass

#=-------------------------------------------

logger = logging.getLogger('terramind')

# Model definitions
# __all__ = [
#     # pre-trained models
#     'terramind_v01_base',
#     'terramind_v1_base',
# ]

# pretrained_weights = {
#         "terramind_v01_base": {
#             "hf_hub_id": "FAST-EO/TerraMind-0.1-base",
#             "hf_hub_filename": "TerraMind_v01_base.pt",
#         },
#         "terramind_v1_base": {
#             "hf_hub_id": "FAST-EO/TerraMind-1.0-base",
#             "hf_hub_filename": "TerraMind_v1_base.pt",
#         },
#     }

PRETRAINED_BANDS = {
    'untok_sen2l2a@224': [
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "CIRRUS",
        "SWIR_1",
        "SWIR_2",
    ],
    'untok_sen2l1c@224': [
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "WATER_VAPOR"
        "CIRRUS",
        "SWIR_1",
        "SWIR_2",
    ],
    'untok_sen2rgb@224': ["RED", "GREEN", "BLUE"],
    'untok_sen1grd@224': ["VV", "VH"],
    'untok_sen1rtc@224': ["VV", "VH"],
    'untok_dem@224': ["DEM"],
}


def select_modality_patch_embed_weights(model: TerraMindViT, bands: dict[str, list], pretrained_bands: dict[str, list]):
    """
    Update patch embeddings weights for each provided modality by selecting the pretrained weights for each band.
    Args:
         model (TerraMindViT): model
         bands (dict[str, list]): Bands with format {<modality>: [<band names>]}
         pretrained_bands (dict[str, list]): Pretrained bands of the model with format {<modality>: [<band names>]}
    """
    # Update modality names to match model layer names
    bands = {model.mod_name_mapping[k]: v for k, v in bands.items()}
    for mod, mod_bands in bands.items():
        if mod not in pretrained_bands:
            logger.info(f"Cannot load band weights for modality {mod}, not found in pretrained bands.")
            continue

        pixel_count = model.encoder_embeddings[mod].patch_size[0] * model.encoder_embeddings[mod].patch_size[1]

        pretrained_weight = model.encoder_embeddings[mod].proj.weight.clone()
        # Init new projection layer with updated number of channels
        model.encoder_embeddings[mod].proj = nn.Linear(
            pixel_count * len(mod_bands),
            model.encoder_embeddings[mod].dim_tokens,
            bias=False
        )
        temp_weight = model.encoder_embeddings[mod].proj.weight.clone()

        # Reshape to [dim, pixel, band]
        temp_weight = temp_weight.view(temp_weight.shape[0], pixel_count, -1)
        pretrained_weight = pretrained_weight.view(pretrained_weight.shape[0], pixel_count, -1)

        # Copy weights of bands
        for index, band in enumerate(mod_bands):
            if band in pretrained_bands[mod]:
                logging.info(f"Loaded weights for {band} in position {index} of patch embed")
                pretrained_index = pretrained_bands[mod].index(band)
                temp_weight[..., index] = pretrained_weight[..., pretrained_index]

        # Update model weights
        model.encoder_embeddings[mod].proj.weight = nn.Parameter(temp_weight.view(temp_weight.shape[0], -1))

    return model


def checkpoint_filter_fn_vit(state_dict, model: TerraMindViT) -> dict:
    """Manually filter pre-trained weights for TerraMind ViT to enable strict weight loading."""

    model_state_dict = model.state_dict()
    clean_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict:
            if v.shape == model_state_dict[k].shape:
                clean_dict[k] = v
            else:
                logger.warning(f"Shape for {k} ({list(v.shape)}) does not match model weights "
                               f"({list(model_state_dict[k].shape)}), skipping weights.")

    missing_params = set(model_state_dict.keys()) - set(clean_dict.keys())
    for k in missing_params:
        logger.warning(f"Weights for {k} are missing in state dict, using random initialization.")
        clean_dict[k] = model_state_dict[k]

    state_dict = clean_dict

    return state_dict

def build_terrammind_vit(
        variant: str = None,
        pretrained: bool = False,
        encoder_weights: str | None = None,
        bands: dict[str, list] | None = None,
        pretrained_bands: dict[str, list] | None = None,
        # output_layers: list = None,
        **kwargs):

    model = TerraMindViT(
        encoder_weights=encoder_weights,
        **kwargs)

    if encoder_weights is not None:
        # Load model from checkpoint
        state_dict = torch.load(encoder_weights, map_location="cpu", weights_only=True)
        loaded_keys = model.load_state_dict(state_dict, strict=False)
        if loaded_keys.missing_keys:
            logger.warning(f"Missing keys in encoder_weights {encoder_weights}: {loaded_keys.missing_keys}")
        if loaded_keys.unexpected_keys:
            logger.warning(f"Missing keys in encoder_weights {encoder_weights}: {loaded_keys.missing_keys}")

    # elif pretrained:
    #     # Load model from Hugging Face
    #     state_dict_file = hf_hub_download(repo_id=pretrained_weights[variant]['hf_hub_id'],
    #                                       filename=pretrained_weights[variant]['hf_hub_filename'])
    #     state_dict = torch.load(state_dict_file, map_location="cpu", weights_only=True)
    #     state_dict = checkpoint_filter_fn_vit(state_dict, model)
    #     model.load_state_dict(state_dict, strict=True)

    if bands is not None:
        model = select_modality_patch_embed_weights(model, bands, pretrained_bands)

    return model


# @TERRATORCH_BACKBONE_REGISTRY.register
def terramind_v1_base(**kwargs):
    model = build_terrammind_vit(
        variant='terramind_v1_base',
        encoder_depth=12,
        dim=768,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        pretrained_bands=PRETRAINED_BANDS,
        **kwargs
    )
    return model

def terramind_v1_large(**kwargs):
    model = build_terrammind_vit(
        variant='terramind_v1_large',
        encoder_depth=24,
        dim=1024,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
        act_layer=nn.SiLU,
        gated_mlp=True,
        pretrained_bands=PRETRAINED_BANDS,
        **kwargs
    )
    return model


# # @TERRATORCH_BACKBONE_REGISTRY.register
# def terramind_v01_base(**kwargs):
#     model = build_terrammind_vit(
#         variant='terramind_v01_base',
#         encoder_depth=12,
#         dim=768,
#         num_heads=12,
#         mlp_ratio=4,
#         qkv_bias=False,
#         proj_bias=False,
#         mlp_bias=False,
#         norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
#         act_layer=nn.SiLU,
#         gated_mlp=True,
#         pretrained_bands={'untok_sen2l2a@224': PRETRAINED_BANDS['untok_sen2l2a@224']},
#         **kwargs
#     )
#     return model

# def terramind_v1_large_experimental_500b(**kwargs):
#     model = build_terrammind_vit(
#         variant='terramind_v1_large_experimental_500b',
#         encoder_depth=24,
#         dim=1024,
#         num_heads=16,
#         mlp_ratio=4,
#         qkv_bias=False,
#         proj_bias=False,
#         mlp_bias=False,
#         norm_layer=partial(LayerNorm, eps=1e-6, bias=False),
#         act_layer=nn.SiLU,
#         gated_mlp=True,
#         pretrained_bands=PRETRAINED_BANDS,
#         **kwargs
#     )
#     return model

# if __name__ == "__main__":
#     vit = terramind_v1_base()
#     print(vit.mod_name_mapping)