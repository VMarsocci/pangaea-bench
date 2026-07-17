# Copyright (c) IBM Corp. 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# transformers: https://github.com/huggingface/transformers
# --------------------------------------------------------

from logging import Logger
from pathlib import Path

import warnings
import logging
from anyio import Path
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import to_2tuple
from timm.models.vision_transformer import Block

from pangaea.encoders.base import Encoder

logger = logging.getLogger(__name__)


def get_3d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 3D sin/cos positional embeddings.

    Args:
        embed_dim (int):
            Embedding dimension.
        grid_size (tuple[int, int, int] | list[int]):
            The grid depth, height and width.
        add_cls_token (bool, *optional*, defaults to False):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size[0]*grid_size[1]*grid_size[2], embed_dim) or
        (1+grid_size[0]*grid_size[1]*grid_size[2], embed_dim): the position embeddings (with or without cls token)
    """

    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def _get_1d_sincos_embed_from_grid_torch(embed_dim: int, pos: torch.Tensor):
    """ Modified torch version of *get_1d_sincos_pos_embed_from_grid()*.

        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,) - must be float dtype!
        out: (M, D)
    """
    assert embed_dim % 2 == 0
    assert pos.dtype in [torch.float32, torch.float16, torch.bfloat16]

    omega = torch.arange(embed_dim // 2, dtype=pos.dtype).to(pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)

    return emb


def _init_weights(module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def _interpolate_pos_encoding(
        pos_embed: torch.Tensor,
        grid_size: tuple[int, int, int] | list[int],
        patch_size: tuple[int, int, int] | list[int],
        shape: tuple[int, int, int],
        embed_dim: int,
):
    """
    Adapted from:
    - transformers.models.vit.modeling_vit.ViTEmbeddings.interpolate_pos_encoding,
    - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194
    """
    t, h, w = shape
    t_patches = t // patch_size[0]
    h_patches = h // patch_size[1]
    w_patches = w // patch_size[2]

    if [t_patches, h_patches, w_patches] == grid_size:
        # No interpolation needed
        return pos_embed
    if t_patches != grid_size[0]:
        # Re-compute pos embedding to handle changed num_frames
        new_grid_size = (t_patches, *grid_size[1:])
        new_pos_embed = get_3d_sincos_pos_embed(pos_embed.shape[-1], new_grid_size, add_cls_token=True)
        new_pos_embed = torch.from_numpy(new_pos_embed).float().unsqueeze(0)
    else:
        new_grid_size = grid_size
        new_pos_embed = pos_embed

    class_pos_embed, patch_pos_embed = new_pos_embed[:, :1], new_pos_embed[:, 1:]

    patch_pos_embed = patch_pos_embed.reshape(*new_grid_size, embed_dim).permute(0, 3, 1, 2)

    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed,
        size=(h_patches, w_patches),
        mode='bicubic',
        align_corners=True,
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, embed_dim)

    return torch.cat((class_pos_embed, patch_pos_embed), dim=1)


class PatchEmbed(nn.Module):
    """3D version of timm.models.vision_transformer.PatchEmbed"""
    def __init__(
            self,
            input_size: tuple[int, int, int] = (1, 224, 224),
            patch_size: tuple[int, int, int] = (1, 16, 16),
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: nn.Module | None = None,
            flatten: bool = True,
            bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.grid_size = [s // p for s, p in zip(self.input_size, self.patch_size)]
        assert self.grid_size >= [1, 1, 1], "Patch size is bigger than input size."
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape

        if T / self.patch_size[0] % 1 or H / self.patch_size[1] % 1 or W / self.patch_size[2] % 1:
            warnings.warn(f"Input {x.shape[-3:]} is not divisible by patch size {self.patch_size}."
                          f"The border will be ignored, add backbone_padding for pixel-wise tasks.")

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x


class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.year_embed_dim = embed_dim // 2
        self.julian_day_embed_dim = embed_dim - self.year_embed_dim

        # If trainable, initialize scale with small number
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, temporal_coords: torch.Tensor, tokens_per_frame: int | None = None):
        """
        temporal_coords: year and day-of-year info with shape (B, T, 2).
        tokens_per_frame: number of tokens for each frame in the sample. If provided, embeddings will be
            repeated over T dimension, and final shape is (B, T*tokens_per_frame, embed_dim).
        """
        shape = temporal_coords.shape[:2] + (-1,)  # B, T, -1

        year = _get_1d_sincos_embed_from_grid_torch(
            self.year_embed_dim, temporal_coords[:, :, 0].flatten()).reshape(shape)
        julian_day = _get_1d_sincos_embed_from_grid_torch(
            self.julian_day_embed_dim, temporal_coords[:, :, 1].flatten()).reshape(shape)

        embedding = self.scale * torch.cat([year, julian_day], dim=-1)

        if tokens_per_frame is not None:
            embedding = torch.repeat_interleave(embedding, tokens_per_frame, dim=1)

        return embedding  # B, T*tokens_per_frame, embed_dim


class LocationEncoder(nn.Module):
    def __init__(self, embed_dim: int, trainable_scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.lat_embed_dim = embed_dim // 2
        self.lon_embed_dim = embed_dim - self.lat_embed_dim

        # If trainable, initialize scale with small number
        if trainable_scale:
            self.scale = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, location_coords: torch.Tensor):
        """
        location_coords: lat and lon info with shape (B, 2).
        """
        shape = location_coords.shape[:1] + (1, -1)  # B, 1, -1

        lat = _get_1d_sincos_embed_from_grid_torch(
                self.lat_embed_dim, location_coords[:, 0].flatten()).reshape(shape)
        lon = _get_1d_sincos_embed_from_grid_torch(
                self.lon_embed_dim, location_coords[:, 1].flatten()).reshape(shape)

        embedding = self.scale * torch.cat([lat, lon], dim=-1)

        return embedding  # B, 1, embed_dim


class Prithvi2_Encoder(Encoder):
    """ Prithvi ViT Encoder"""
    def __init__(self,
                 encoder_weights: str | Path,
                 download_url: str,
                 input_bands: dict[str, list[str]],
                 input_size: int | tuple[int, int] = 224,
                 output_dim: int | list[int] = 1024,
                 output_layers: int | list[int] = [5, 11, 17, 23],
                 patch_size: int | tuple[int, int, int] = (1, 16, 16),
                 num_frames: int = 1,
                 in_chans: int = 3,
                 embed_dim: int = 1024,
                 depth: int = 24,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.,
                 norm_layer: nn.Module = nn.LayerNorm,
                 coords_encoding: list[str] | None = None,
                 coords_scale_learn: bool = False,
                 drop_path: float = 0.,
                 ** kwargs,
                ):
        super().__init__(
            model_name="Prithvi2",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=True,
            multi_temporal_output=True,
            pyramid_output=False,
            download_url=download_url,
        )

        self.in_chans = in_chans
        self.num_frames = num_frames if isinstance(num_frames, int) and num_frames > 0 else 1
        self.embed_dim = embed_dim
        self.img_size = [input_size, input_size] if isinstance(input_size, int) else input_size
        self.img_size = to_2tuple(self.img_size)
        if isinstance(patch_size, int):
            patch_size = (1, patch_size, patch_size)

        self.np = self.img_size[-1] // patch_size[-1]
        

        # 3D patch embedding
        self.patch_embed = PatchEmbed(
            input_size=(self.num_frames,) + self.img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.out_channels = [embed_dim * self.patch_embed.grid_size[0]] * depth

        # Optional temporal and location embedding
        coords_encoding = coords_encoding or []
        self.temporal_encoding = 'time' in coords_encoding
        self.location_encoding = 'location' in coords_encoding
        if self.temporal_encoding:
            assert patch_size[0] == 1, f"With temporal encoding, patch_size[0] must be 1, received {patch_size[0]}"
            self.temporal_embed_enc = TemporalEncoder(embed_dim, coords_scale_learn)
        if self.location_encoding:
            self.location_embed_enc = LocationEncoder(embed_dim, coords_scale_learn)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer("pos_embed", torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))

        # Transformer layers
        self.blocks = []
        for i in range(depth):
            self.blocks.append(Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                                     drop_path=drop_path,))
        self.blocks = nn.ModuleList(self.blocks)

        self.norm = norm_layer(embed_dim)

        self.initialize_weights()


    def load_encoder_weights(self, logger: Logger) -> None:
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)
        pretrained_model = {key.replace("encoder.", ""): value for key, value in pretrained_model.items()}
        k = pretrained_model.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained_model[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
            else:
                pretrained_encoder[name] = pretrained_model[name]
            # print(name)
        # print(k)

        self.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, add_cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)

    def random_masking(self, sequence, mask_ratio, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling. Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`torch.FloatTensor` of shape `(batch_size, sequence_length, dim)`)
            mask_ratio (float): mask ratio to use.
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - mask_ratio))

        if noise is None:
            noise = torch.rand(batch_size, seq_length, device=sequence.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1).to(sequence.device)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(sequence.device)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def interpolate_pos_encoding(self, sample_shape: tuple[int, int, int]):

        pos_embed = _interpolate_pos_encoding(
            pos_embed=self.pos_embed,
            grid_size=self.patch_embed.grid_size,
            patch_size=self.patch_embed.patch_size,
            shape=sample_shape,
            embed_dim=self.embed_dim,
        )
        return pos_embed

    # def forward(
    #     self, image,
    #     temporal_coords: None | torch.Tensor = None,
    #     location_coords: None | torch.Tensor = None,
    #     mask_ratio=0.75
    # ):
    #     x = image["optical"]
    #     if len(x.shape) == 4 and self.patch_embed.input_size[0] == 1:
    #         # add time dim
    #         x = x.unsqueeze(2)
    #     sample_shape = x.shape[-3:]

    #     # embed patches
    #     x = self.patch_embed(x)

    #     pos_embed = self.interpolate_pos_encoding(sample_shape)
    #     # add pos embed w/o cls token
    #     x = x + pos_embed[:, 1:, :]

    #     if self.temporal_encoding and temporal_coords is not None:
    #         num_tokens_per_frame = x.shape[1] // self.num_frames
    #         temporal_encoding = self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
    #         x = x + temporal_encoding
    #     if self.location_encoding and location_coords is not None:
    #         location_encoding = self.location_embed_enc(location_coords)
    #         x = x + location_encoding

    #     # masking: length -> length * mask_ratio
    #     x, mask, ids_restore = self.random_masking(x, mask_ratio)

    #     # append cls token
    #     cls_token = self.cls_token + pos_embed[:, :1, :]
    #     cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    #     x = torch.cat((cls_tokens, x), dim=1)

    #     # apply Transformer blocks
    #     for block in self.blocks:
    #         x = block(x)
    #     x = self.norm(x)

    #     return x, mask, ids_restore

    def forward(
        self,
        image,
        temporal_coords: None | torch.Tensor = None,
        location_coords: None | torch.Tensor = None,
    ) -> list[torch.Tensor]:
        
        x = image["optical"]
        if len(x.shape) == 4 and self.patch_embed.input_size[0] == 1:
            # add time dim
            x = x.unsqueeze(2)
        sample_shape = x.shape[-3:]

        # embed patches
        x = self.patch_embed(x)

        pos_embed = self.interpolate_pos_encoding(sample_shape)
        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        if self.temporal_encoding and temporal_coords is not None:
            num_tokens_per_frame = x.shape[1] // self.num_frames
            temporal_encoding = self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
            x = x + temporal_encoding
        if self.location_encoding and location_coords is not None:
            location_encoding = self.location_embed_enc(location_coords)
            x = x + location_encoding

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.output_layers:
                out = (
                    x[:, 1:, :]
                    .permute(0, 2, 1)
                    .view(
                        x.shape[0],
                        -1,
                        self.num_frames,
                        self.np,
                        self.np,
                    )
                    .squeeze(2)
                    .contiguous()
                )
                output.append(out)

        return output

    def prepare_features_for_image_model(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        out = []
        effective_time_dim = self.patch_embed.input_size[0] // self.patch_embed.patch_size[0]
        for x in features:
            x_no_token = x[:, 1:, :]
            number_of_tokens = x_no_token.shape[1]
            tokens_per_timestep = number_of_tokens // effective_time_dim
            h = int(np.sqrt(tokens_per_timestep))
            encoded = rearrange(
                x_no_token,
                "batch (t h w) e -> batch (t e) h w",
                e=self.embed_dim,
                t=effective_time_dim,
                h=h,
            )
            out.append(encoded)
        return out