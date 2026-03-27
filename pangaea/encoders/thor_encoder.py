import math
import warnings
import logging
from collections.abc import Iterable, Sequence
from functools import partial
from itertools import repeat
from logging import Logger
from pathlib import Path
from typing import Any, Final, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from timm.layers import (
    DropPath,
    Mlp,
    use_fused_attn,
)
from timm.models.vision_transformer import LayerScale
from torch import Tensor, nn, vmap

from pangaea.encoders.base import Encoder

logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


pangaea_to_thor_band_map = {
    "B1": "S2:CoastAerosal",
    "B2": "S2:Blue",
    "B3": "S2:Green",
    "B4": "S2:Red",
    "B5": "S2:RE1",
    "B6": "S2:RE2",
    "B7": "S2:RE3",
    "B8": "S2:NIR",
    "B8A": "S2:RE4",
    "B9": "S2:WaterVapor",
    # "B10 # SWIR - Cirrus (60m) # We don't have this band
    "B11": "S2:SWIR1",
    "B12": "S2:SWIR2",
    "VV": "S1:IW-VV",
    "VH": "S1:IW-VH",
    "ASC_VV": "S1:IW-VV",
    "ASC_VH": "S1:IW-VH",
    "DSC_VV": "S1:IW-VV",
    "DSC_VH": "S1:IW-VH",
}


DEFAULT_GROUPS = [
    ["S2:Red", "S2:Green", "S2:Blue", "S2:NIR"],
    ["S2:RE1", "S2:RE2", "S2:RE3", "S2:RE4", "S2:SWIR1", "S2:SWIR2"],
    ["S2:CoastAerosal", "S2:WaterVapor"],
    ["S1:IW-VH", "S1:IW-VV", "S1:EW-VH", "S1:EW-VV"],
    ["S1:IW-HV", "S1:IW-HH", "S1:EW-HV", "S1:EW-HH"],
    [
        "S3:Oa01_reflectance",
        "S3:Oa02_reflectance",
        "S3:Oa03_reflectance",
        "S3:Oa04_reflectance",
        "S3:Oa05_reflectance",
        "S3:Oa06_reflectance",
        "S3:Oa07_reflectance",
    ],
    [
        "S3:Oa08_reflectance",
        "S3:Oa09_reflectance",
        "S3:Oa10_reflectance",
        "S3:Oa11_reflectance",
        "S3:Oa12_reflectance",
        "S3:Oa13_reflectance",
        "S3:Oa14_reflectance",
    ],
    [
        "S3:Oa15_reflectance",
        "S3:Oa16_reflectance",
        "S3:Oa17_reflectance",
        "S3:Oa18_reflectance",
        "S3:Oa19_reflectance",
        "S3:Oa20_reflectance",
        "S3:Oa21_reflectance",
    ],
    [
        "S3:S1_reflectance_an",
        "S3:S2_reflectance_an",
        "S3:S3_reflectance_an",
        "S3:S4_reflectance_an",
        "S3:S5_reflectance_an",
        "S3:S6_reflectance_an",
    ],
    ["S3:S7_BT_in", "S3:S8_BT_in", "S3:S9_BT_in"],
]

DEFAULT_CHANNELS = {
    "S2:Red": {
        "GSD": 10,
        "patch_size": 16,  # px
    },
    "S2:Green": {
        "GSD": 10,
        "patch_size": 16,  # px
    },
    "S2:Blue": {
        "GSD": 10,
        "patch_size": 16,  # px
    },
    "S2:NIR": {
        "GSD": 10,
        "patch_size": 16,  # px
    },
    "S2:RE1": {
        "GSD": 20,
        "patch_size": 16,  # px
    },
    "S2:RE2": {
        "GSD": 20,
        "patch_size": 16,  # px
    },
    "S2:RE3": {
        "GSD": 20,
        "patch_size": 16,  # px
    },
    "S2:RE4": {
        "GSD": 20,
        "patch_size": 16,  # px
    },
    "S2:SWIR1": {
        "GSD": 20,
        "patch_size": 16,  # px
    },
    "S2:SWIR2": {
        "GSD": 20,
        "patch_size": 16,  # px
    },
    "S2:CoastAerosal": {
        "GSD": 60,
        "patch_size": 16,  # px
    },
    "S2:WaterVapor": {
        "GSD": 60,
        "patch_size": 16,  # px
    },
    "S1:IW-VV": {
        "GSD": 10,
        "patch_size": 16,  # px
        "patch_embed_name": "S1:VV",
    },
    "S1:IW-VH": {
        "GSD": 10,
        "patch_size": 16,  # px
        "patch_embed_name": "S1:VH",
    },
    "S1:IW-HV": {
        "GSD": 10,
        "patch_size": 16,  # px
        "patch_embed_name": "S1:HV",
    },
    "S1:IW-HH": {
        "GSD": 10,
        "patch_size": 16,  # px
        "patch_embed_name": "S1:HH",
    },
    "S1:EW-VV": {
        "GSD": 10,
        "patch_size": 16,  # px
        "patch_embed_name": "S1:VV",
    },
    "S1:EW-VH": {
        "GSD": 10,
        "patch_size": 16,  # px
        "patch_embed_name": "S1:VH",
    },
    "S1:EW-HV": {
        "GSD": 10,
        "patch_size": 16,  # px
        "patch_embed_name": "S1:HV",
    },
    "S1:EW-HH": {
        "GSD": 10,
        "patch_size": 16,  # px
        "patch_embed_name": "S1:HH",
    },
    "S3:Oa01_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa02_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa03_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa04_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa05_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa06_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa07_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa08_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa09_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa10_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa11_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa12_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa13_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa14_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa15_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa16_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa17_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa18_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa19_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa20_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:Oa21_reflectance": {
        "GSD": 240,  # GSD 240 interp
        "patch_size": 16,  # px
    },
    "S3:S1_reflectance_an": {
        "GSD": 480,  # GSD 480 interp
        "patch_size": 16,  # px
    },
    "S3:S2_reflectance_an": {
        "GSD": 480,  # GSD 480 interp
        "patch_size": 16,  # px
    },
    "S3:S3_reflectance_an": {
        "GSD": 480,  # GSD 480 interp
        "patch_size": 16,  # px
    },
    "S3:S4_reflectance_an": {
        "GSD": 480,  # GSD 480 interp
        "patch_size": 16,  # px
    },
    "S3:S5_reflectance_an": {
        "GSD": 480,  # GSD 480 interp
        "patch_size": 16,  # px
    },
    "S3:S6_reflectance_an": {
        "GSD": 480,  # GSD 480 interp
        "patch_size": 16,  # px
    },
    "S3:S7_BT_in": {
        "GSD": 960,  # GSD 960 interp
        "patch_size": 16,  # px
    },
    "S3:S8_BT_in": {
        "GSD": 960,  # GSD 960 interp
        "patch_size": 16,  # px
    },
    "S3:S9_BT_in": {
        "GSD": 960,  # GSD 960 interp
        "patch_size": 16,  # px
    },
}


def to_2tuple(x: Any) -> tuple[int, int]:
    if isinstance(x, Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))


class IndFlexiPatchEmbed(nn.Module):
    def __init__(
        self,
        ground_covers: list[int],
        channels: dict[str, dict[str, int]],
        channel_rename_map: dict[str, str] | None = None,
        embed_dim: int = 768,
        norm_layer: nn.Module | None = None,
        flatten: bool = True,
        bias: bool = True,
        patch_size_seqs: dict[str, Sequence[int]] | Sequence[int] = (
            4,
            6,
            8,
            10,
            12,
            16,
        ),
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        """2D image to patch embedding w/ flexible patch sizes, for multiple product bands
        Extended from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L24

        Args:
            ground_cover: Ground cover size in meters
            channels: Dictionary of product bands and their parameters (GSD, num_patch)
            channel_rename_map: Dictionary of channel names to rename, to i.e., use same parameters for different bands
            embed_dim: Network embedding dimension size
            norm_layer: Optional normalization layer
            flatten: Whether to flatten the spatial dimensions of the output
            bias: Whether to use bias in convolution
            patch_size_seqs: Dict of List of patch sizes for each band or list of patch sizes for all bands to
                randomly sample from, unvalidated patch sizes are dropped
            patch_size_probs: Optional Dict of list of probabilities for each band or list of probabilities for all
                bands to sample corresponding patch_size_seqs elements. If None, then uniform distribution is used
            interpolation: Resize interpolation type
            antialias: Whether to apply antialiasing resizing
        """
        super().__init__()
        self.interpolation = interpolation
        self.antialias = antialias
        self.flatten = flatten
        self.channels = channels
        self.channel_rename_map = channel_rename_map
        self.embed_dim = embed_dim
        self.patch_sizes = {}
        proj_dict = {}
        for product_band, params in channels.items():
            kernel_size = params["patch_size"]
            self.patch_sizes[product_band] = to_2tuple(kernel_size)
            if channel_rename_map and product_band in channel_rename_map:
                product_band = channel_rename_map[product_band]
            if product_band in proj_dict:
                logger.info(f"Product band {product_band} already added, skipping")
                continue
            proj_dict[product_band] = nn.Conv2d(
                1, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=bias
            )
        self.patch_embed = nn.ModuleDict(proj_dict)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        if not isinstance(patch_size_seqs, dict):
            patch_size_seqs = {
                product_band: patch_size_seqs for product_band in channels
            }

        # filter valid patch size seqs
        for product_band in patch_size_seqs.keys():
            _patch_size_seq = []
            for patch_size in patch_size_seqs[product_band]:
                product_gsd = channels[product_band]["GSD"]
                for ground_cover in ground_covers:
                    product_num_patch = ground_cover // patch_size // product_gsd
                    if patch_size * product_num_patch * product_gsd != ground_cover:
                        # Skip patch sizes that don't add up for the given ground cover
                        continue
                    if patch_size not in _patch_size_seq:
                        _patch_size_seq.append(patch_size)

            if len(_patch_size_seq) == 0:
                msg = (
                    f"No valid patch sizes for {product_band} for ground cover {ground_covers} and GSD {product_gsd}"
                    f" with patch size seq {patch_size_seqs[product_band]}"
                )
                logger.warning(msg)
                continue

            logger.info(f"product_band: {product_band}, patch_size_seq: {_patch_size_seq}")
            patch_size_seqs[product_band] = sorted(_patch_size_seq)

        self.patch_size_seqs = patch_size_seqs
        self.pinvs = {}

    def _resize(self, x: Tensor, shape: tuple[int, int]) -> Tensor:
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias,
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(
        self, old_shape: tuple[int, int], new_shape: tuple[int, int]
    ) -> Tensor:
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    def resize_patch_embed(
        self,
        patch_embed: Tensor,
        patch_size: tuple[int, int],
        new_patch_size: tuple[int, int],
    ):
        """Resize patch_embed to target resolution via pseudo-inverse resizing"""
        # Return original kernel if no resize is necessary
        if patch_size == new_patch_size:
            return patch_embed

        # Calculate pseudo-inverse of resize matrix
        if patch_size not in self.pinvs or new_patch_size not in self.pinvs[patch_size]:
            if patch_size not in self.pinvs:
                self.pinvs[patch_size] = {}
            self.pinvs[patch_size][new_patch_size] = self._calculate_pinv(
                patch_size, new_patch_size
            )
        pinv = self.pinvs[patch_size][new_patch_size]
        pinv = pinv.to(patch_embed.device)

        def resample_patch_embed(patch_embed: Tensor):
            h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

        v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

        return v_resample_patch_embed(patch_embed)

    def forward(
        self,
        x: dict[str, Tensor],
        patch_sizes: dict[str, tuple[int, int]] | None = None,
    ) -> Tensor | tuple[Tensor, tuple[int, int]]:
        if patch_sizes is None:
            # During evaluation use base patch sizes if not specified
            patch_sizes = self.patch_sizes

        patch_embed_dict = {}
        for product_band, data in x.items():
            if product_band not in patch_sizes:
                logger.info(f"Skipping product band: {product_band}")
                continue

            patch_size = patch_sizes[product_band]
            patch_size = to_2tuple(patch_size)

            patch_embed_name = (
                self.channel_rename_map[product_band]
                if self.channel_rename_map
                else product_band
            )

            # Resize conv weights
            if patch_size == self.patch_sizes[product_band]:
                weight = self.patch_embed[patch_embed_name].weight
            else:
                weight = self.resize_patch_embed(
                    self.patch_embed[patch_embed_name].weight,
                    self.patch_sizes[product_band],
                    patch_size,
                )

            # Apply conv with resized weights
            data = F.conv2d(
                input=data,
                weight=weight,
                bias=self.patch_embed[patch_embed_name].bias,
                stride=patch_size,
            )

            if self.flatten:
                data = data.flatten(2).transpose(1, 2)  # BCHW -> BNC

            data = self.norm(data)
            patch_embed_dict[product_band] = data

        return patch_embed_dict


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    Generate 1D sincos positional embedding.
    Args:
        embed_dim: Output dimension for each position
        pos: A list of positions to be encoded: size (M,)
    Returns:
        Positional embedding: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.double()


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# Copied from timm.models.vision_transformer
# and adapted to support alibi
# furthermore, we use the flexi attention implementation if available
class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, alibi: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=alibi,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if alibi is not None:
                attn = attn + alibi
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self, x: torch.Tensor, alibi: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), alibi)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


def get_slopes(n):
    """
    Get slopes for attention bias calculation in alibi attention.
    Args:
        n: Number of attention heads
    Returns:
        List of slopes for each attention head
    """

    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


@torch.jit.script
def get_alibi_thor(
    metadata: dict[str, dict[str, int]],
    available_groups: dict[str, list[str]],
    slopes: torch.Tensor,
    offset: float | int = 0.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    2D Alibi implementation using Euclidean distance between patches
    Args:
        metadata: Metadata of the input data with GSD, patch_size, and num_patch information
        available_groups: Available groups of the input data
        slopes: Slopes for attention heads, used to scale the distance, should be a tensor of shape (num_heads,)
        offset: Offset to add to the distance, useful for cls token
        device: Device for computation
        dtype: Data type for computation
    Returns:
        distances: Alibi tensor (batch_size, num_heads, num_patches, num_patches),
        where num_patches is the total number of patches for all groups
    """
    num_patches = 0
    all_points = []
    max_patch_gsd_size = 0

    for _group_name, group_members in available_groups.items():
        first_member = group_members[0]
        product_gsd = metadata[first_member]["GSD"]
        product_num_patch = metadata[first_member]["num_patch"]
        product_patch_size = metadata[first_member]["patch_size"]
        num_patches += int(product_num_patch**2)
        max_patch_gsd_size = max(max_patch_gsd_size, product_patch_size * product_gsd)

        line_of_points = torch.arange(0, product_num_patch, dtype=dtype, device=device)
        line_of_points *= product_patch_size
        line_of_points += product_patch_size / 2
        line_of_points *= product_gsd

        points = torch.cartesian_prod(line_of_points, line_of_points)
        all_points.append(points)

    points = torch.cat(all_points, dim=0)

    # Normalize points by largest GSD
    points = points / max_patch_gsd_size

    attention_heads = slopes.shape[0]
    slopes = slopes.unsqueeze(1).unsqueeze(2)
    distances = torch.cdist(points, points)
    distances += float(offset)
    distances = distances.unsqueeze(0)
    distances = distances * slopes * -1
    distances = distances.view(-1, attention_heads, num_patches, num_patches)

    return distances


class THOR_Encoder(Encoder):
    def __init__(
        self,
        encoder_weights: str | Path,
        input_size: int,
        patch_size: int,
        input_bands: dict[str, list[str]],
        output_layers: list[int],
        download_url: str = "",
        ground_cover: int | None = None,
        output_aggr: int | str = "concat",
        #### Default parameters vit 'base_encoder_alibi_patch_embed' ###
        ref_patch_size: int = 4,
        patch_size_seqs: dict[str, Sequence[int]] | Sequence[int] | None = None,
        channels: dict[str, dict[str, int]] = DEFAULT_CHANNELS,
        groups: list[list] = DEFAULT_GROUPS,
        aggr_type: str = "subsetmean",
        ### Base ViT parameters ##########################################
        embed_dim=768,
        depth=12,
        num_heads=12,
        embed_band=True,
        band_embed_dim=128,
        embed_prod=False,
        prod_embed_dim=0,
        embed_patch_size=True,
        pad_prod_embed_null=False,
        pad_band_embed_null=False,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        if ground_cover is None:
            ground_cover = input_size * 10
            logger.info(
                f"Assuming ground_cover is {ground_cover} m based on input_size {input_size} px"
            )
        self.ground_cover = ground_cover

        bands = [
            pangaea_to_thor_band_map[band]
            for modality_bands in input_bands.values()
            for band in modality_bands
        ]
        self.bands = bands
        logger.info(f"Bands: {self.bands}")

        self.aggr_type = aggr_type
        if patch_size < ref_patch_size:
            msg = (
                f"The model was trained with a ref_patch_size of {ref_patch_size}, but the input patch size is {patch_size}."
                f" Overriding the ref_patch_size to {patch_size}. This may lead to suboptimal performance."
            )
            warnings.warn(msg)
            ref_patch_size = patch_size
        self.ref_patch_size = ref_patch_size
        if patch_size_seqs is None:
            patch_size_seqs = [patch_size]

        # Backwards compat channels
        self.channels = {}
        self.channel_rename_map = {}
        for channel, params in channels.items():
            if "num_patch" in params and "patch_size" not in params:
                params["patch_size"] = (
                    self.ground_cover // params["num_patch"] // params["GSD"]
                )
            if "patch_size" in params and "num_patch" not in params:
                if isinstance(patch_size_seqs, dict):
                    min_patch_size_seq = min(patch_size_seqs[channel])
                elif isinstance(patch_size_seqs, Sequence):
                    min_patch_size_seq = min(patch_size_seqs)
                patch_size = min(params["patch_size"], min_patch_size_seq)

                params["num_patch"] = self.ground_cover // patch_size // params["GSD"]

            rename_name = params.pop("patch_embed_name", channel)
            if rename_name in self.channel_rename_map:
                raise ValueError(f"Duplicate patch embed name {rename_name} found.")
            self.channel_rename_map[channel] = rename_name
            self.channels[channel] = params

        # Default groups
        self.groups = self.validate_group(
            groups
        )  # {'group0':[product_band, ...], 'group1': ..., ...}

        available_groups: dict[str, list[str]] = self.get_available_groups({
            band: None for band in bands
        })

        self.available_groups = available_groups

        self.output_aggr = output_aggr
        if self.output_aggr == "concat":
            output_dim = len(available_groups) * embed_dim
        elif self.output_aggr in ["mean", "sum", "max", "min"]:
            output_dim = embed_dim
        elif isinstance(self.output_aggr, int):
            output_dim = embed_dim
        else:
            raise ValueError(f"Unknown output_aggr {self.output_aggr}")

        logger.info(f"THOR Encoder output_dim: {output_dim}")

        size_map = {192: "tiny", 384: "small", 768: "base", 1024: "large"}

        super().__init__(
            model_name=f"thor_{size_map[embed_dim]}_encoder",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,  # my_model_embed_dim, fixed parameters
            output_dim=output_dim,
            output_layers=output_layers,
            pyramid_output=False,
            download_url=download_url,
            multi_temporal=False,  # wether support multi-temporal, fixed parametersfixed parameters
            multi_temporal_output=False,  # wether the output of the model has a temporal dimension
        )
        logger.info(f"Available groups: {available_groups}")
        self.min_gsd = min([params["GSD"] for params in self.channels.values()])

        self.ind_patch_embed = IndFlexiPatchEmbed(
            ground_covers=[self.ground_cover],
            channels=self.channels,
            channel_rename_map=self.channel_rename_map,
            embed_dim=embed_dim,
            patch_size_seqs=patch_size_seqs,
        )

        # Initialize embedding
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.embed_prod = embed_prod
        self.prod_embed_dim = prod_embed_dim if embed_prod else 0
        self.pad_prod_embed_null = pad_prod_embed_null
        self.embed_band = embed_band
        self.band_embed_dim = band_embed_dim if embed_band else 0
        self.pad_band_embed_null = pad_band_embed_null
        self.pos_embed_dim = embed_dim - prod_embed_dim - band_embed_dim
        self.embed_patch_size = embed_patch_size
        self.patch_size_embed_dim = self.pos_embed_dim if embed_patch_size else 0

        self.register_buffer("encoder_slopes", torch.tensor(get_slopes(num_heads)))

        self.init_embeds()

        # Initialize transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def init_embeds(self):
        self.band_embed, self.prod_embed = self.initialize_embedding(
            self.band_embed_dim,
            self.prod_embed_dim,
        )

    def validate_group(self, groups):
        valid_groups = {}
        if groups is None:
            for group_idx, prodcut_band in enumerate(self.channels.keys()):
                valid_groups[f"group{group_idx}"] = [prodcut_band]
            return valid_groups
        found_bands = []
        for group_idx, group in enumerate(groups):
            group_product = None
            group_gsd = None
            group_patch_size = None
            valid_groups[f"group{group_idx}"] = []
            for product_band in group:
                product, _ = product_band.split(":")
                group_product = product if group_product is None else group_product
                group_gsd = (
                    self.channels[product_band]["GSD"]
                    if group_gsd is None
                    else group_gsd
                )
                group_patch_size = (
                    self.channels[product_band]["patch_size"]
                    if group_patch_size is None
                    else group_patch_size
                )
                if self.channels[product_band]["GSD"] != group_gsd:
                    msg = f"GSD {self.channels[product_band]['GSD']} in group {group} does not match with GSD {group_gsd} in the same group."
                    raise ValueError(msg)
                if self.channels[product_band]["patch_size"] != group_patch_size:
                    msg = f"Patch size {self.channels[product_band]['patch_size']} in group {group} does not match with patch size {group_patch_size} in the same group."
                    raise ValueError(msg)
                valid_groups[f"group{group_idx}"].append(product_band)
                found_bands.append(product_band)

        # Remove bands from channels that are not present in the groups
        keys = list(self.channels.keys())
        remove_bands = set(keys) - set(found_bands)
        for band in remove_bands:
            del self.channels[band]
        if remove_bands:
            warnings.warn(
                f"Removed bands {remove_bands} from channels that are not present in the groups."
            )

        return valid_groups

    def initialize_embedding(
        self,
        band_embed_dim: int,
        prod_embed_dim: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        band_embed = {}
        if self.embed_band:
            embed = get_1d_sincos_pos_embed_from_grid(
                band_embed_dim, torch.arange(len(self.groups)).cpu().numpy()
            )
            for i, spectral_group in enumerate(self.groups.keys()):
                band_embed[spectral_group] = torch.from_numpy(embed[i]).float()

        prod_embed = {}
        if self.embed_prod:
            unique_prod = {
                prduct_band.split(":")[0]: None for prduct_band in self.channels.keys()
            }
            embed = get_1d_sincos_pos_embed_from_grid(
                prod_embed_dim, torch.arange(len(unique_prod)).cpu().numpy()
            )
            for i, prod in enumerate(unique_prod.keys()):
                if self.pad_prod_embed_null:
                    prod_embed[prod] = torch.zeros_like(
                        torch.from_numpy(embed[i])
                    ).float()
                else:
                    prod_embed[prod] = torch.from_numpy(embed[i]).float()

        band_embed = nn.ParameterDict(band_embed).requires_grad_(False)
        prod_embed = nn.ParameterDict(prod_embed).requires_grad_(False)

        return band_embed, prod_embed

    def initialize_weights(self) -> None:
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_available_groups(
        self, original_input: dict[str, torch.Tensor | None]
    ) -> dict[str, list[str]]:
        available_groups: dict[str, list[str]] = {}
        for group_name, group_member in self.groups.items():
            for member in group_member:
                if member in original_input:
                    if group_name not in available_groups:
                        available_groups[group_name] = [member]
                    else:
                        available_groups[group_name].append(member)
        return available_groups

    def get_channel_params(
        self,
        patch_embed: dict[str, torch.Tensor],
        metadata: dict[str, dict[str, int]] | None = None,
        ground_cover: int | None = None,
    ) -> dict[str, dict[str, int]]:
        """Get GSD, num_patch, patch_size for each channel in the input data.
        Args:
            patch_embed: patch embeddings of the input data {'product_band': (B, N, C)}
            metadata: metadata of the input data
        """
        channel_params = {
            product_band: {"GSD": self.channels[product_band]["GSD"]}
            for product_band in patch_embed.keys()
        }

        # Override with default if available
        if ground_cover is None:
            ground_cover = self.ground_cover

        assert isinstance(ground_cover, int), (
            f"Ground cover {ground_cover} is not defined as int, please provide a valid ground cover."
        )

        for product_band in patch_embed:
            # Override with metadata if available
            if (
                metadata is not None
                and product_band in metadata
                and "GSD" in metadata[product_band]
            ):
                channel_params[product_band]["GSD"] = metadata[product_band]["GSD"]
            num_patch = int(patch_embed[product_band].shape[1] ** 0.5)
            patch_size = (
                ground_cover // num_patch // channel_params[product_band]["GSD"]
            )
            channel_params[product_band]["num_patch"] = num_patch
            channel_params[product_band]["patch_size"] = patch_size
            assert (
                patch_size * channel_params[product_band]["GSD"] * num_patch
                == ground_cover
            ), (
                f"Patch size {patch_size} * GSD {channel_params[product_band]['GSD']} * num_patch {num_patch} does not match ground cover {ground_cover}"
            )

        return channel_params

    def get_encoder_auxilliary_embed(
        self,
        original_input: dict[str, torch.Tensor],
        available_groups: dict[str, list[str]],
        channel_params: dict[str, dict[str, int]],
    ) -> dict[str, torch.Tensor]:
        auxilliary_embed = {}
        for group_name, group_member in available_groups.items():
            product_band = group_member[0]
            product, band = product_band.split(":")

            data = original_input[product_band]

            if self.embed_patch_size:
                patch_sizes = (
                    torch.ones(
                        (channel_params[product_band]["num_patch"] ** 2),
                        device=data.device,
                        dtype=data.dtype,
                    )
                    * channel_params[product_band]["patch_size"]
                    * channel_params[product_band]["GSD"]
                    / (self.ref_patch_size * self.min_gsd)
                )

                group_pos_embed = get_1d_sincos_pos_embed_from_grid_torch(
                    self.patch_size_embed_dim, pos=patch_sizes
                ).to(data.dtype)
            else:
                group_pos_embed = torch.zeros(
                    (
                        channel_params[product_band]["num_patch"] ** 2,
                        self.pos_embed_dim,
                    ),
                    device=data.device,
                    dtype=data.dtype,
                )

            auxilliary_embed[group_name] = group_pos_embed

            if self.embed_band:
                auxilliary_embed[group_name] = torch.cat(
                    (
                        auxilliary_embed[group_name],
                        self.band_embed[group_name].expand(
                            auxilliary_embed[group_name].shape[0], -1
                        ),
                    ),
                    dim=-1,
                )
            if self.embed_prod:
                auxilliary_embed[group_name] = torch.cat(
                    (
                        auxilliary_embed[group_name],
                        self.prod_embed[product].expand(
                            auxilliary_embed[group_name].shape[0], -1
                        ),
                    ),
                    dim=-1,
                )

            band_ground_cover = int(
                data.shape[-1] * channel_params[product_band]["GSD"]
            )
            if band_ground_cover != self.ground_cover:
                raise ValueError(
                    f"Input ground cover for {product_band} is {band_ground_cover}x{band_ground_cover}, (image shape:{data.shape[-2:]}) "
                    f"which does match grid of {channel_params[product_band]['num_patch']}x{channel_params[product_band]['num_patch']}"
                    f"with patch size {channel_params[product_band]['patch_size']} and GSD {channel_params[product_band]['GSD']}."
                    f" patches for defined ground cover {self.ground_cover}."
                )

        return auxilliary_embed

    def aggregate_by_group(
        self,
        patch_embed: dict[str, torch.Tensor],
        available_groups: dict[str, list[str]],
    ) -> dict[str, torch.Tensor]:
        group_embed = {}
        for group_name, group_member in available_groups.items():
            if self.aggr_type == "subsetmean":
                to_stack = [
                    patch_embed[product_band]
                    for product_band in group_member
                    if product_band in patch_embed
                ]
                if len(to_stack) > 0:
                    group_embed[group_name] = torch.stack(to_stack, -1).mean(-1)
            elif self.aggr_type == "subsetsum":
                to_stack = [
                    patch_embed[product_band]
                    for product_band in group_member
                    if product_band in patch_embed
                ]
                if len(to_stack) > 0:
                    group_embed[group_name] = torch.stack(to_stack, -1).sum(-1)

        return group_embed

    def forward_intermediates(
        self,
        inp: dict[str, torch.Tensor],
        metadata: dict[str, dict[str, int]] | None = None,
        ground_cover: int | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, dict[str, int]]]:
        """Forward pass with intermediate outputs."""

        # TODO: add mode option for using i.e., smallest patch size or normal patch size
        patch_embeds = self.ind_patch_embed(
            inp,
            patch_sizes={
                p: min(p_sizes)
                for p, p_sizes in self.ind_patch_embed.patch_size_seqs.items()
            },
        )
        # {'product:band': (B, N_n, D), ...}

        # Get available groups
        available_groups = self.get_available_groups(patch_embeds)
        # {'group0': [product_band, ...], ...}

        # Get channel parameters
        channel_params = self.get_channel_params(patch_embeds, metadata, ground_cover)

        # Aggregate embeddings by group
        group_embeds = self.aggregate_by_group(patch_embeds, available_groups)
        # {'group0': (B, N_n, D), ...}

        # Get auxiliary embeddings (positional, band)
        aux_embeds = self.get_encoder_auxilliary_embed(
            inp, available_groups, channel_params
        )
        # {'group0': (N_n, D), ...}

        # Combine group embeddings with auxiliary embeddings
        x = torch.cat(
            [
                group_embeds[group_name] + aux_embeds[group_name]
                for group_name in group_embeds.keys()
            ],
            dim=1,
        )
        # (B, T, D)

        # Compute alibi attention bias
        alibi = get_alibi_thor(
            channel_params,
            available_groups,
            slopes=self.encoder_slopes,
            offset=0.0,
            device=x.device,
            dtype=x.dtype,
        )
        alibi = alibi.expand(x.shape[0], -1, -1, -1)

        # apply Transformer blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, alibi)
            # if i == len(self.blocks) - 1:
            #     x = self.norm(x) # TODO: drop?
            if i in self.output_layers:
                output.append(x)

        return output, channel_params

    def _post_process(
        self, features: list[torch.Tensor], channel_params: dict[str, dict[str, int]]
    ) -> list[torch.Tensor]:
        """Stack embeddings for each group, requires interpolation to the highest num_patch."""
        highest_num_patch = max(
            channel_params[product_band]["num_patch"]
            for product_band in channel_params.keys()
        )

        out_features = []
        for feature in features:
            start_idx = 0
            out = []
            # Important that we iterate through this in the same order we encoded
            for group_member in self.available_groups.values():
                if len(group_member) == 0:
                    continue
                product_band = group_member[0]

                num_patch = channel_params[product_band]["num_patch"]

                x_ = feature[:, start_idx : start_idx + num_patch**2, :].reshape(
                    -1, num_patch, num_patch, self.embed_dim
                )
                x_ = x_.permute(0, 3, 1, 2)  # B, C, H, W

                # Interpolate if needed
                if num_patch != highest_num_patch and self.output_aggr in [
                    "concat",
                    "mean",
                    "sum",
                    "max",
                    "min",
                ]:
                    x_ = F.interpolate(
                        x_,
                        size=(highest_num_patch, highest_num_patch),
                        mode="bilinear",
                    )

                out.append(x_)
                start_idx += num_patch**2

            if start_idx != feature.shape[1]:
                raise ValueError(
                    f"Number of patches {start_idx} does not match number of patches in input {feature.shape[1]}"
                )

            if self.output_aggr == "concat":
                # Concatenate all group features
                out = torch.cat(out, dim=1)
            elif self.output_aggr == "mean":
                # Mean all group features
                out = torch.mean(torch.stack(out, dim=0), dim=0)
            elif self.output_aggr == "sum":
                # Sum all group features
                out = torch.sum(torch.stack(out, dim=0), dim=0)
            elif self.output_aggr == "max":
                # Max all group features
                out = torch.max(torch.stack(out, dim=0), dim=0)[0]
            elif self.output_aggr == "min":
                # Min all group features
                out = torch.min(torch.stack(out, dim=0), dim=0)[0]
            elif isinstance(self.output_aggr, int):
                # Select the group feature at the specified index
                out = out[self.output_aggr]
            out_features.append(out)

        return out_features

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def load_encoder_weights(self, logger: Logger) -> None:
        pretrained_model = torch.load(
            self.encoder_weights, map_location="cpu", weights_only=True
        )["state_dict"]
        prefix = "mae."
        k = pretrained_model.keys()
        renamed_keys = {
            key.replace(prefix, ""): key for key in k if key.startswith(prefix)
        }
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.named_parameters():
            if name not in renamed_keys:
                missing[name] = param.shape
            elif pretrained_model[renamed_keys[name]].shape != param.shape:
                incompatible_shape[name] = (
                    param.shape,
                    pretrained_model[renamed_keys[name]].shape,
                )
            else:
                pretrained_encoder[name] = pretrained_model[renamed_keys[name]]
        unused_keys = set(renamed_keys.keys()) - set(pretrained_encoder.keys())
        for key in unused_keys:
            if key in renamed_keys:
                # These are keys that were in the pretrained model but not used,
                # typically decoder weights.
                logger.debug(f"Unused key {key} in pretrained model")
            else:
                logger.warning(f"Key {key} not found in pretrained model")

        if missing:
            logger.warning(
               f"Some keys from the pretrained model were not found in the current model: {missing}"
            )

        if incompatible_shape:
            logger.warning(
                f"Some parameters have incompatible shapes: {incompatible_shape}"
            )
            raise ValueError("Incompatible parameter shapes found, have you loaded the correct size model?")

        self.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)
        logger.info("Loaded encoder weights successfully.")

    # def _preprocess_input(self, x):
    #     """Preprocess input data for the model."""
    #     # Concatenate all modalities
    #     x = torch.concat([x[modality] for modality in x.keys()], dim=1)

    #     x = {
    #         channel: F.interpolate(
    #             x[:, [i], :, :],
    #             (
    #                 int(self.ground_cover / self.channels[channel]["GSD"]),
    #                 int(self.ground_cover / self.channels[channel]["GSD"]),
    #             ),
    #             mode="bilinear",
    #         )
    #         for i, channel in enumerate(self.bands)
    #     }

    #     return x

    def _preprocess_input(self, x):
        """Preprocess input data for the model."""
        # Concatenate all modalities
        # print(f"x keys: {x.keys()}")
        # print(f"x shapes before concat: {[x[modality].shape for modality in x.keys()]}")
        x = torch.concat([x[modality] for modality in x.keys()], dim=1)
        
        # Ensure x has batch dimension
        # print(f"x.dim() before unsqueeze: {x.dim()}")
        if x.dim() != 3:
            x = x.squeeze(2)

        x = {
            channel: F.interpolate(
                x[:, [i], :, :],
                (
                    int(self.ground_cover / self.channels[channel]["GSD"]),
                    int(self.ground_cover / self.channels[channel]["GSD"]),
                ),
                mode="bilinear",
            )
            for i, channel in enumerate(self.bands)
        }

        return x

    def forward(self, x: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Foward pass of the encoder.

        Args:
            x (dict[str, torch.Tensor]): encoder's input structured as a dictionary:
            x = {modality1: tensor1, modality2: tensor2, ...}, e.g. x = {"optical": tensor1, "sar": tensor2}.
            If the encoder is multi-temporal (self.multi_temporal==True), input tensor shape is (B C T H W) with C the
            number of bands required by the encoder for the given modality and T the number of time steps. If the
            encoder is not multi-temporal, input tensor shape is (B C H W) with C the number of bands required by the
            encoder for the given modality.

        Returns:
            list[torch.Tensor]: list of the embeddings for each modality. For single-temporal encoders, the list's
            elements are of shape (B, embed_dim, H', W'). For multi-temporal encoders, the list's elements are of shape
            (B, C', T, H', W') with T the number of time steps if the encoder does not have any time-merging strategy,
            else (B, C', H', W') if the encoder has a time-merging strategy (where C'==self.output_dim).
        """
        x = self._preprocess_input(x)
        outputs, channel_params = self.forward_intermediates(x)
        outputs = self._post_process(outputs, channel_params=channel_params)
        return outputs
    
_large_cfg = {
    "embed_dim": 1024,
    "depth": 24,
    "num_heads": 16,
    "band_embed_dim": 256,
}

_base_cfg = {
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "band_embed_dim": 128,
}

_small_cfg = {
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "band_embed_dim": 64,
}

_tiny_cfg = {
    "embed_dim": 192,
    "depth": 12,
    "num_heads": 3,
    "band_embed_dim": 32,
}

def thor_tiny_encoder(**kwargs) -> THOR_Encoder:
    """THOR Tiny Encoder with Alibi Attention and Flexivit Patch Embedding V1

    Args:
        **kwargs: keyword arguments for the THOR_Encoder class

    Returns:
        THOR_Encoder: THOR Tiny Encoder with Alibi Attention and Patch Embedding V1
    """
    model = THOR_Encoder(
        **_tiny_cfg,
        **kwargs,
    )
    return model

def thor_small_encoder(**kwargs) -> THOR_Encoder:
    """THOR Small Encoder with Alibi Attention and Flexivit Patch Embedding V1

    Args:
        **kwargs: keyword arguments for the THOR_Encoder class

    Returns:
        THOR_Encoder: THOR Small Encoder with Alibi Attention and Flexivit Patch Embedding V1
    """
    model = THOR_Encoder(
        **_small_cfg,
        **kwargs,
    )
    return model


def thor_base_encoder(**kwargs) -> THOR_Encoder:
    """THOR Base Encoder with Alibi Attention and Flexivit Patch Embedding V1

    Args:
        **kwargs: keyword arguments for the THOR_Encoder class

    Returns:
        THOR_Encoder: THOR Base Encoder with Alibi Attention and Flexivit Patch Embedding V1
    """
    model = THOR_Encoder(
        **_base_cfg,
        **kwargs,
    )
    return model

def thor_large_encoder(**kwargs) -> THOR_Encoder:
    """THOR Large Encoder with Alibi Attention and Flexivit Patch Embedding V1

    Args:
        **kwargs: keyword arguments for the THOR_Encoder class

    Returns:
        THOR_Encoder: THOR Large Encoder with Alibi Attention and Flexivit Patch Embedding V1
    """
    model = THOR_Encoder(
        **_large_cfg,
        **kwargs,
    )
    return model


__all__ = [
    "thor_tiny_encoder",
    "thor_small_encoder",
    "thor_base_encoder",
    "thor_large_encoder",
]