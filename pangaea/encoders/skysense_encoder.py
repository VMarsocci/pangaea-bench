"""SkySense encoder wrapper for the Pangaea benchmark, adapted from https://github.com/Jack-bo1220/SkySense.

Three encoder variants are exposed:
- ``SkySenseS2``    : Sentinel-2 ViT-Large only (optical).
- ``SkySenseS2S1``  : Sentinel-2 + Sentinel-1 ViT-Large pair, outputs are
                      channel-concatenated at each output layer.
- ``SkySenseVHR``   : VHR SwinTransformerV2-Huge backbone only.

The cross-modal fusion transformer from the original SkySense paper is
intentionally excluded so that Pangaea's standardised decoder fusion
architectures (UPerNet, LTAE, etc.) can be used for fair comparison.

All model code (VisionTransformer, SwinTransformerV2 and their building
blocks) is inlined from the SkySense reference implementation, adapted to
use only standard PyTorch — no mmcv or mmengine dependency.

Checkpoint format
-----------------
Each backbone is expected as a **standalone state dict** (a plain
``torch.load`` result with parameter names matching the backbone directly,
no wrapping keys).  The three backbone files are configured via separate
config keys:

- ``encoder_weights``     → Sentinel-2 ViT-Large state dict (all variants)
- ``s1_encoder_weights``  → Sentinel-1 ViT-Large state dict (S2S1)
- ``vhr_encoder_weights`` → VHR SwinTransformerV2-Huge state dict (VHR)

For ``SkySenseVHR`` the single ``encoder_weights`` key points to the VHR
backbone file.
"""

import logging
import math
from copy import deepcopy
from logging import Logger
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.init import trunc_normal_
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from pangaea.encoders.base import Encoder


# ===========================================================================
# Pure-PyTorch building blocks
# (replacements for mmcv.cnn.bricks.transformer and mmengine.model)
# ===========================================================================

class _DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class _PatchEmbed(nn.Module):
    """2-D image-to-patch embedding.

    Replacement for ``mmcv.cnn.bricks.transformer.PatchEmbed``.
    ``forward`` returns ``(tokens, (H', W'))`` with tokens of shape
    ``(B, H'*W', embed_dims)``, matching the mmcv interface.
    ``init_out_size`` is set when ``input_size`` is provided (needed by Swin).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 768,
        conv_type: str = "Conv2d",    # ignored; always Conv2d
        kernel_size: int = 16,
        stride: int = 16,
        padding: int | str = 0,
        norm_cfg=None,                # if not None a LayerNorm is applied post-embed
        input_size: int | None = None,
        init_cfg=None,                # accepted but ignored (mmcv compat)
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = embed_dims

        # mmcv uses "corner" (asymmetric) padding for non-divisible sizes.
        # SkySense input sizes are always multiples of patch_size, so 0 is
        # equivalent.
        if padding == "corner":
            padding = 0

        self.projection = nn.Conv2d(
            in_channels, embed_dims,
            kernel_size=kernel_size, stride=stride, padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dims) if norm_cfg is not None else None

        if input_size is not None:
            if isinstance(input_size, int):
                input_size = (input_size, input_size)
            h = (input_size[0] + 2 * padding - kernel_size) // stride + 1
            w = (input_size[1] + 2 * padding - kernel_size) // stride + 1
            self.init_out_size: tuple[int, int] = (h, w)
        else:
            self.init_out_size = None

    def forward(self, x: torch.Tensor):
        x = self.projection(x)              # (B, C, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, H'*W', C)
        if self.norm is not None:
            x = self.norm(x)
        return x, (H, W)


class _MHA(nn.Module):
    """Self-attention wrapping ``nn.MultiheadAttention``.

    Replacement for ``mmcv.cnn.bricks.transformer.MultiheadAttention``.
    Supports the ``identity`` residual shortcut used by the ViT encoder layer.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        batch_first: bool = True,
        bias: bool = True,
        **kwargs,    # absorb any extra mmcv-style kwargs
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads,
            dropout=attn_drop, bias=bias, batch_first=batch_first,
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = query
        out, _ = self.attn(query, key, value)
        out = self.proj_drop(out)
        if identity is not None:
            return identity + out
        return out


class _FFN(nn.Module):
    """Two-layer feed-forward network with optional DropPath.

    Replacement for ``mmcv.cnn.bricks.transformer.FFN``.
    Supports both the ``identity`` residual (``add_identity=True``, ViT style)
    and the manual residual in SwinBlockV2 (``add_identity=False``).
    """

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        num_fcs: int = 2,
        ffn_drop: float = 0.0,
        dropout_layer=None,        # dict(type='DropPath', drop_prob=p) or None
        act_cfg=None,              # accepted but ignored; always GELU
        add_identity: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert num_fcs >= 2
        self.add_identity = add_identity

        # Mirror mmcv FFN structure so checkpoint keys match:
        #   layers.0 = nn.Sequential(Linear, GELU, Dropout)  → keys: layers.0.0.*
        #   layers.1 = nn.Linear                             → keys: layers.1.*
        #   layers.2 = nn.Dropout                            → no params
        layers: list[nn.Module] = []
        in_dim = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Sequential(
                nn.Linear(in_dim, feedforward_channels),
                nn.GELU(),
                nn.Dropout(ffn_drop),
            ))
            in_dim = feedforward_channels
        layers.append(nn.Linear(in_dim, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

        drop_prob = 0.0
        if isinstance(dropout_layer, dict):
            drop_prob = dropout_layer.get("drop_prob", 0.0)
        self.drop_path = _DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, identity: torch.Tensor | None = None) -> torch.Tensor:
        out = self.drop_path(self.layers(x))
        if self.add_identity:
            return (identity if identity is not None else x) + out
        return out


# ===========================================================================
# VisionTransformer (ViT-Large)
# Used for both the Sentinel-2 and Sentinel-1 backbones in SkySense.
# ===========================================================================

class _TransformerEncoderLayer(nn.Module):
    """Single encoder layer of the SkySense ViT backbone (pre-norm)."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_fcs: int = 2,
        qkv_bias: bool = True,
        act_cfg: dict = dict(type="GELU"),
        norm_cfg: dict = dict(type="LN"),
        batch_first: bool = True,
        attn_cfg: dict = dict(),
        ffn_cfg: dict = dict(),
        with_cp: bool = False,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dims)

        _attn_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            batch_first=batch_first,
            bias=qkv_bias,
        )
        _attn_cfg.update(attn_cfg)
        self.attn = _MHA(**_attn_cfg)

        self.ln2 = nn.LayerNorm(embed_dims)

        _ffn_cfg = dict(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=(
                dict(type="DropPath", drop_prob=drop_path_rate)
                if drop_path_rate > 0
                else None
            ),
            act_cfg=act_cfg,
        )
        _ffn_cfg.update(ffn_cfg)
        self.ffn = _FFN(**_ffn_cfg)
        self.with_cp = with_cp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _inner(x):
            x = self.attn(self.ln1(x), identity=x)
            x = self.ffn(self.ln2(x), identity=x)
            return x

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner, x)
        return _inner(x)


class _VisionTransformer(nn.Module):
    """ViT-Large backbone as used in SkySense for S2 / S1 inputs.

    Key differences from the canonical ViT:
    - patch_size=4 (instead of 16) for 64×64 input images.
    - Supports arbitrary input sizes at inference via dynamic pos_embed
      resize (bicubic interpolation).
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 10,
        embed_dims: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        out_indices=-1,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.3,
        with_cls_token: bool = True,
        output_cls_token: bool = False,
        norm_cfg: dict = dict(type="LN"),
        act_cfg: dict = dict(type="GELU"),
        patch_norm: bool = False,
        final_norm: bool = False,
        interpolate_mode: str = "bicubic",
        num_fcs: int = 2,
        norm_eval: bool = False,
        with_cp: bool = False,
        pretrained=None,
        init_cfg=None,    # accepted but ignored (mmcv compat)
    ):
        super().__init__()

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2

        if output_cls_token:
            assert with_cls_token, (
                "with_cls_token must be True when output_cls_token=True"
            )

        self.img_size = img_size
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.norm_eval = norm_eval
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token

        self.patch_embed = _PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            kernel_size=patch_size,
            stride=patch_size,
            padding="corner",
            norm_cfg=norm_cfg if patch_norm else None,
        )

        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            self.out_indices = [num_layers - 1 if out_indices == -1 else out_indices]
        elif isinstance(out_indices, (list, tuple)):
            self.out_indices = list(out_indices)
        else:
            raise TypeError("out_indices must be int, list or tuple")

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=mlp_ratio * embed_dims,
                attn_drop_rate=attn_drop_rate,
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                num_fcs=num_fcs,
                qkv_bias=qkv_bias,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                batch_first=True,
            )
            for i in range(num_layers)
        ])

        self.final_norm = final_norm
        self.final_ln = nn.LayerNorm(embed_dims) if final_norm else None
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    if "ffn" in n:
                        nn.init.normal_(m.bias, mean=0.0, std=1e-6)
                    else:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, _BatchNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _pos_embedding(self, patched_img, hw_shape, pos_embed):
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            pos_h = self.img_size[0] // self.patch_size
            pos_w = self.img_size[1] // self.patch_size
            pos_embed = self._resize_pos_embed(
                pos_embed, hw_shape, (pos_h, pos_w), self.interpolate_mode
            )
        return self.drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def _resize_pos_embed(pos_embed, input_shape, pos_shape, mode):
        assert pos_embed.ndim == 3
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0:1]
        pos_embed_weight = pos_embed[:, -(pos_h * pos_w):]
        pos_embed_weight = (
            pos_embed_weight.reshape(1, pos_h, pos_w, pos_embed.shape[2])
            .permute(0, 3, 1, 2)
        )
        pos_embed_weight = F.interpolate(
            pos_embed_weight, size=input_shape, align_corners=False, mode=mode
        )
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        return torch.cat((cls_token_weight, pos_embed_weight), dim=1)

    def forward(self, inputs: torch.Tensor):
        B = inputs.shape[0]
        x, hw_shape = self.patch_embed(inputs)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embedding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            x = x[:, 1:]

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1 and self.final_norm and self.final_ln is not None:
                x = self.final_ln(x)
            if i in self.out_indices:
                out = x[:, 1:] if self.with_cls_token else x
                B_, _, C = out.shape
                out = (
                    out.reshape(B_, hw_shape[0], hw_shape[1], C)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

        return tuple(outs)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()


# ===========================================================================
# SwinTransformerV2-Huge
# Used for the VHR (high-resolution optical) backbone in SkySense.
# ===========================================================================

def _resize_swin_pos_embed(
    pos_embed: torch.Tensor,
    src_shape,
    dst_shape,
    mode: str = "bicubic",
    num_extra_tokens: int = 1,
) -> torch.Tensor:
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    C = pos_embed.shape[2]
    extra_tokens = pos_embed[:, :num_extra_tokens]
    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)
    dst_weight = F.interpolate(
        src_weight, size=(dst_h, dst_w), align_corners=False, mode=mode
    )
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
    return torch.cat([extra_tokens, dst_weight], dim=1)


class _PatchMerging(nn.Module):
    """Swin V2 patch merging (downsampling) layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        norm_cfg: dict = dict(type="LN"),
        is_post_norm: bool = False,
        init_cfg=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        dilation = to_2tuple(dilation)
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
        sample_dim = kernel_size[0] * kernel_size[1] * in_channels
        self.is_post_norm = is_post_norm
        if norm_cfg is not None:
            norm_dim = out_channels if is_post_norm else sample_dim
            self.norm = nn.LayerNorm(norm_dim)
        else:
            self.norm = None
        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, input_size):
        B, L, C = x.shape
        H, W = input_size
        assert L == H * W
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.sampler(x)
        x = x.transpose(1, 2)
        out_H = (
            H + 2 * self._padding[0]
            - self._dilation[0] * (self._kernel_size[0] - 1) - 1
        ) // self._stride[0] + 1
        out_W = (
            W + 2 * self._padding[1]
            - self._dilation[1] * (self._kernel_size[1] - 1) - 1
        ) // self._stride[1] + 1
        output_size = (out_H, out_W)
        if self.norm is not None and not self.is_post_norm:
            x = self.norm(x)
        x = self.reduction(x)
        if self.norm is not None and self.is_post_norm:
            x = self.norm(x)
        return x, output_size


class _WindowMSAV2(nn.Module):
    """Swin V2 window-based multi-head self-attention with scaled cosine
    attention and log-spaced continuous relative position bias."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size,
        pretrained_window_size=(0, 0),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_cfg=None,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.num_heads = num_heads
        self.pretrained_window_size = pretrained_window_size

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        relative_coords_h = torch.arange(
            -(window_size[0] - 1), window_size[0], dtype=torch.float32
        )
        relative_coords_w = torch.arange(
            -(window_size[1] - 1), window_size[1], dtype=torch.float32
        )
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij")
        ).permute(1, 2, 0).contiguous().unsqueeze(0)

        norm_h = max(pretrained_window_size[0] - 1, window_size[0] - 1, 1)
        norm_w = max(pretrained_window_size[1] - 1, window_size[1] - 1, 1)
        if pretrained_window_size[0] > 0:
            norm_h = pretrained_window_size[0] - 1
            norm_w = pretrained_window_size[1] - 1
        relative_coords_table[:, :, :, 0] /= norm_h
        relative_coords_table[:, :, :, 1] /= norm_w
        relative_coords_table *= 8
        relative_coords_table = (
            torch.sign(relative_coords_table)
            * torch.log2(torch.abs(relative_coords_table) + 1.0)
            / math.log2(8)
        )
        self.register_buffer("relative_coords_table", relative_coords_table)

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w], indexing="ij")
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(embed_dims))
        self.v_bias = nn.Parameter(torch.zeros(embed_dims))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def _get_relative_position_bias(self, ws: int, device: torch.device) -> torch.Tensor:
        """Compute relative position bias for an arbitrary square window size ws×ws."""
        if ws == self.window_size[0]:
            # Use pre-computed buffers (common case)
            coords_table = self.relative_coords_table
            rel_idx = self.relative_position_index
        else:
            # Recompute on-the-fly for smaller windows (e.g. when feature map < window_size)
            rel_h = torch.arange(-(ws - 1), ws, dtype=torch.float32, device=device)
            rel_w = torch.arange(-(ws - 1), ws, dtype=torch.float32, device=device)
            coords_table = torch.stack(
                torch.meshgrid([rel_h, rel_w], indexing="ij")
            ).permute(1, 2, 0).contiguous().unsqueeze(0)
            norm_h = self.pretrained_window_size[0] - 1 if self.pretrained_window_size[0] > 0 else max(ws - 1, 1)
            norm_w = self.pretrained_window_size[1] - 1 if self.pretrained_window_size[1] > 0 else max(ws - 1, 1)
            coords_table[:, :, :, 0] /= norm_h
            coords_table[:, :, :, 1] /= norm_w
            coords_table *= 8
            coords_table = (
                torch.sign(coords_table)
                * torch.log2(torch.abs(coords_table) + 1.0)
                / math.log2(8)
            )
            coords = torch.stack(
                torch.meshgrid([torch.arange(ws, device=device), torch.arange(ws, device=device)], indexing="ij")
            )
            coords_flat = torch.flatten(coords, 1)
            rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
            rel_coords = rel_coords.permute(1, 2, 0).contiguous()
            rel_coords[:, :, 0] += ws - 1
            rel_coords[:, :, 1] += ws - 1
            rel_coords[:, :, 0] *= 2 * ws - 1
            rel_idx = rel_coords.sum(-1)

        bias_table = self.cpb_mlp(coords_table).view(-1, self.num_heads)
        bias = bias_table[rel_idx.view(-1)].view(ws * ws, ws * ws, -1)
        bias = bias.permute(2, 0, 1).contiguous()
        return 16 * torch.sigmoid(bias)

    def forward(self, query: torch.Tensor, key=None, value=None, mask=None):
        B_, N, C = query.shape
        ws = int(N ** 0.5)  # actual window size (may differ from self.window_size when map < window)
        qkv_bias = torch.cat((
            self.q_bias,
            torch.zeros_like(self.v_bias, requires_grad=False),
            self.v_bias,
        ))
        qkv = F.linear(query, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale

        relative_position_bias = self._get_relative_position_bias(ws, query.device)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = (
                attn.view(B_ // nW, nW, self.num_heads, N, N)
                + mask.unsqueeze(1).unsqueeze(0)
            )
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _ShiftWindowMSA(nn.Module):
    """Shifted window multi-head self-attention for Swin V2."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dropout_layer: dict = dict(type="DropPath", drop_prob=0.0),
        pad_small_map: bool = False,
        window_msa=None,
        msa_cfg: dict = dict(),
        init_cfg=None,
    ):
        super().__init__()
        if window_msa is None:
            window_msa = _WindowMSAV2
        self.shift_size = shift_size
        self.window_size = window_size
        self.pad_small_map = pad_small_map
        self.w_msa = window_msa(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            **msa_cfg,
        )
        drop_prob = dropout_layer.get("drop_prob", 0.0) if isinstance(dropout_layer, dict) else 0.0
        self.drop = _DropPath(drop_prob) if drop_prob > 0.0 else nn.Identity()

    def forward(self, query: torch.Tensor, hw_shape):
        H, W = hw_shape
        B, L, C = query.shape
        assert L == H * W
        query = query.view(B, H, W, C)

        window_size = self.window_size
        shift_size = self.shift_size
        if min(H, W) <= window_size:
            shift_size = 0
            window_size = min(H, W)

        pad_b = (window_size - H % window_size) % window_size
        pad_r = (window_size - W % window_size) % window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = query.shape

        if shift_size > 0:
            shifted_query = torch.roll(query, shifts=(-shift_size, -shift_size), dims=(1, 2))
            attn_mask = self._get_attn_mask(Hp, Wp, window_size, shift_size, query.device)
        else:
            shifted_query = query
            attn_mask = None

        query_windows = self._window_partition(shifted_query, window_size)
        query_windows = query_windows.view(-1, window_size * window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, window_size, window_size, C)
        shifted_x = self._window_reverse(attn_windows, window_size, Hp, Wp)

        if shift_size > 0:
            x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        return self.drop(x)

    @staticmethod
    def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    @staticmethod
    def _window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    @staticmethod
    def _get_attn_mask(H, W, window_size, shift_size, device):
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = _ShiftWindowMSA._window_partition(img_mask, window_size)
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)


class _SwinBlockV2(nn.Module):
    """Swin Transformer V2 block with post-normalisation."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        window_size: int = 8,
        shift: bool = False,
        extra_norm: bool = False,
        ffn_ratio: float = 4.0,
        drop_path: float = 0.0,
        pad_small_map: bool = False,
        attn_cfgs: dict = dict(),
        ffn_cfgs: dict = dict(),
        norm_cfg: dict = dict(type="LN"),
        with_cp: bool = False,
        pretrained_window_size: int = 0,
        init_cfg=None,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.extra_norm = extra_norm

        _attn_cfgs = {
            "embed_dims": embed_dims,
            "num_heads": num_heads,
            "shift_size": window_size // 2 if shift else 0,
            "window_size": window_size,
            "dropout_layer": dict(type="DropPath", drop_prob=drop_path),
            "pad_small_map": pad_small_map,
            "window_msa": _WindowMSAV2,
            "msa_cfg": dict(pretrained_window_size=to_2tuple(pretrained_window_size)),
            **attn_cfgs,
        }
        self.attn = _ShiftWindowMSA(**_attn_cfgs)
        self.norm1 = nn.LayerNorm(embed_dims)

        _ffn_cfgs = {
            "embed_dims": embed_dims,
            "feedforward_channels": int(embed_dims * ffn_ratio),
            "num_fcs": 2,
            "ffn_drop": 0,
            "dropout_layer": dict(type="DropPath", drop_prob=drop_path),
            "act_cfg": dict(type="GELU"),
            "add_identity": False,
            **ffn_cfgs,
        }
        self.ffn = _FFN(**_ffn_cfgs)
        self.norm2 = nn.LayerNorm(embed_dims)

        if self.extra_norm:
            self.norm3 = nn.LayerNorm(embed_dims)

    def forward(self, x: torch.Tensor, hw_shape):
        def _inner(x):
            identity = x
            x = self.attn(x, hw_shape)
            x = self.norm1(x)
            x = x + identity

            identity = x
            x = self.ffn(x)
            x = self.norm2(x)
            x = x + identity

            if self.extra_norm:
                x = self.norm3(x)
            return x

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner, x)
        return _inner(x)


class _SwinBlockV2Sequence(nn.Module):
    """A sequence of Swin V2 blocks with optional patch-merging downsampling."""

    def __init__(
        self,
        embed_dims: int,
        depth: int,
        num_heads: int,
        window_size: int = 8,
        downsample: bool = False,
        downsample_cfg: dict = dict(),
        drop_paths=0.0,
        block_cfgs=dict(),
        with_cp: bool = False,
        pad_small_map: bool = False,
        extra_norm_every_n_blocks: int = 0,
        pretrained_window_size: int = 0,
        init_cfg=None,
    ):
        super().__init__()

        if not isinstance(drop_paths, Sequence):
            drop_paths = [drop_paths] * depth
        if not isinstance(block_cfgs, Sequence):
            block_cfgs = [deepcopy(block_cfgs) for _ in range(depth)]

        if downsample:
            self.out_channels = 2 * embed_dims
            _ds_cfg = {
                "in_channels": embed_dims,
                "out_channels": self.out_channels,
                "norm_cfg": dict(type="LN"),
                **downsample_cfg,
            }
            self.downsample = _PatchMerging(**_ds_cfg)
        else:
            self.out_channels = embed_dims
            self.downsample = None

        self.blocks = nn.ModuleList()
        for i in range(depth):
            extra_norm = bool(
                extra_norm_every_n_blocks and (i + 1) % extra_norm_every_n_blocks == 0
            )
            _block_cfg = {
                "embed_dims": self.out_channels,
                "num_heads": num_heads,
                "window_size": window_size,
                "shift": i % 2 != 0,
                "extra_norm": extra_norm,
                "drop_path": drop_paths[i],
                "with_cp": with_cp,
                "pad_small_map": pad_small_map,
                "pretrained_window_size": pretrained_window_size,
                **block_cfgs[i],
            }
            self.blocks.append(_SwinBlockV2(**_block_cfg))

    def forward(self, x: torch.Tensor, in_shape):
        if self.downsample:
            x, out_shape = self.downsample(x, in_shape)
        else:
            out_shape = in_shape
        for block in self.blocks:
            x = block(x, out_shape)
        return x, out_shape


class _SwinTransformerV2Huge(nn.Module):
    """SwinTransformerV2-Huge backbone as used in SkySense for VHR inputs.

    Hardcoded to the Huge architecture to match SkySense pre-training.
    ``out_indices`` selects which stages to return (0–3).
    """

    _ARCH = {
        "embed_dims": 352,
        "depths": [2, 2, 18, 2],
        "num_heads": [8, 16, 32, 64],
        "extra_norm_every_n_blocks": 6,
    }

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        window_size: int = 8,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        out_indices=(3,),
        use_abs_pos_embed: bool = False,
        interpolate_mode: str = "bicubic",
        with_cp: bool = False,
        frozen_stages: int = -1,
        norm_eval: bool = False,
        pad_small_map: bool = False,
        norm_cfg: dict = dict(type="LN"),
        stage_cfgs: dict = dict(downsample_cfg=dict(is_post_norm=True)),
        patch_cfg: dict = dict(),
        pretrained_window_sizes=None,
        init_cfg=None,
    ):
        super().__init__()
        if pretrained_window_sizes is None:
            pretrained_window_sizes = [0, 0, 0, 0]

        arch = self._ARCH
        self.embed_dims = arch["embed_dims"]
        self.depths = arch["depths"]
        self.num_heads = arch["num_heads"]
        self.extra_norm_every_n_blocks = arch["extra_norm_every_n_blocks"]
        self.num_layers = len(self.depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        self.interpolate_mode = interpolate_mode
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.num_extra_tokens = 0

        if isinstance(window_size, int):
            self.window_sizes = [window_size] * self.num_layers
        else:
            assert len(window_size) == self.num_layers
            self.window_sizes = list(window_size)

        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type="Conv2d",
            kernel_size=patch_size,
            stride=patch_size,
            norm_cfg=dict(type="LN"),
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = _PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size

        if self.use_abs_pos_embed:
            num_patches = self.patch_resolution[0] * self.patch_resolution[1]
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.embed_dims)
            )

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        total_depth = sum(self.depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.stages = nn.ModuleList()
        embed_dims_list = [self.embed_dims]
        for i, (depth, num_heads) in enumerate(zip(self.depths, self.num_heads)):
            _stage_cfg = deepcopy(stage_cfgs) if isinstance(stage_cfgs, dict) else deepcopy(stage_cfgs[i])
            downsample = i > 0
            _s_cfg = {
                "embed_dims": embed_dims_list[-1],
                "depth": depth,
                "num_heads": num_heads,
                "window_size": self.window_sizes[i],
                "downsample": downsample,
                "drop_paths": dpr[:depth],
                "with_cp": with_cp,
                "pad_small_map": pad_small_map,
                "extra_norm_every_n_blocks": self.extra_norm_every_n_blocks,
                "pretrained_window_size": pretrained_window_sizes[i],
                **_stage_cfg,
            }
            stage = _SwinBlockV2Sequence(**_s_cfg)
            self.stages.append(stage)
            dpr = dpr[depth:]
            embed_dims_list.append(stage.out_channels)

        # The SkySense checkpoint only contains a norm for the *last* requested stage.
        # Intermediate stages use nn.Identity() to avoid spurious "missing parameter"
        # warnings when loading the pre-trained weights.
        last_out_idx = max(out_indices)
        for i in out_indices:
            if i == last_out_idx and norm_cfg:
                norm_layer = nn.LayerNorm(embed_dims_list[i + 1])
            else:
                norm_layer = nn.Identity()
            self.add_module(f"norm{i}", norm_layer)

    def forward(self, x: torch.Tensor):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + _resize_swin_pos_embed(
                self.absolute_pos_embed,
                self.patch_resolution,
                hw_shape,
                self.interpolate_mode,
                self.num_extra_tokens,
            )
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                out = norm_layer(x)
                out = (
                    out.view(-1, *hw_shape, stage.out_channels)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(out)

        return tuple(outs)

    def train(self, mode: bool = True):
        super().train(mode)
        # freeze stages if requested
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
        for i in range(0, self.frozen_stages + 1):
            m = self.stages[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            if i in self.out_indices:
                for param in getattr(self, f"norm{i}").parameters():
                    param.requires_grad = False
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

# ===========================================================================
# Helpers: load sub-model state dict (Pangaea pattern)
# ===========================================================================

def _extract_and_load(
    submodule: nn.Module,
    state_dict: dict,
    prefix: str,
    submodule_name: str,
    logger: Logger,
) -> tuple[dict, dict]:
    """Strip *prefix* from matching keys and load into *submodule*.

    Returns ``(missing, incompatible_shape)`` dicts for ``parameters_warning``.
    """
    extracted = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    missing: dict = {}
    incompatible_shape: dict = {}
    matched: dict = {}

    for name, param in submodule.named_parameters():
        full_name = f"{submodule_name}.{name}" if submodule_name else name
        if name not in extracted:
            missing[full_name] = param.shape
        elif extracted[name].shape != param.shape:
            incompatible_shape[full_name] = (param.shape, extracted[name].shape)
        else:
            matched[name] = extracted[name]

    submodule.load_state_dict(matched, strict=False)
    return missing, incompatible_shape


def _load_backbone(
    submodule: nn.Module,
    path: str | Path,
    submodule_name: str,
    logger: Logger,
) -> tuple[dict, dict]:
    """Load backbone weights from a standalone state-dict checkpoint file."""
    state_dict: dict = torch.load(path, map_location="cpu")
    return _extract_and_load(submodule, state_dict, "", submodule_name, logger)


# ===========================================================================
# Pangaea Encoder classes
# ===========================================================================

class SkySenseS2(Encoder):
    """SkySense Sentinel-2 ViT-Large encoder.

    Uses only the Sentinel-2 (optical) backbone from the SkySense checkpoint.
    All 24 transformer layers are available; ``output_layers`` selects which
    ones to return as feature maps.

    The backbone operates on images of shape ``(B, C, H, W)`` where *C* equals
    the number of bands listed in ``input_bands["optical"]`` after band
    filtering by the Pangaea preprocessor.
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_size: int,
        input_bands: dict[str, list[str]],
        output_layers: list[int],
        output_dim: int | list[int],
        download_url: str = "",
        drop_path_rate: float = 0.3,
    ) -> None:
        super().__init__(
            model_name="skysense_s2",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=1024,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
        )

        n_optical = len(input_bands["optical"])
        _out_indices = list(output_layers)
        self.backbone = _VisionTransformer(
            img_size=input_size,
            patch_size=4,
            in_channels=n_optical,
            embed_dims=1024,
            num_layers=24,
            num_heads=16,
            mlp_ratio=4,
            out_indices=_out_indices,
            drop_path_rate=drop_path_rate,
            with_cls_token=True,
            final_norm=False,
        )

    def load_encoder_weights(self, logger: Logger) -> None:
        missing, incompatible = _load_backbone(
            self.backbone, self.encoder_weights, "backbone", logger,
        )
        self.parameters_warning(missing, incompatible, logger)

    def forward(self, image: dict[str, torch.Tensor], metadata=None) -> list[torch.Tensor]:
        x = image["optical"]
        if x.ndim == 5:          # (B, C, T, H, W) — squeeze temporal dim
            x = x.squeeze(2)
        return list(self.backbone(x))


class SkySenseS2S1(Encoder):
    """SkySense Sentinel-2 + Sentinel-1 dual ViT-Large encoder.

    Both optical and SAR backbones process their respective inputs
    independently, then their feature maps are concatenated along the channel
    dimension at each output layer.  Output dimension is therefore
    ``2 × 1024 = 2048`` per output layer.

    Expects ``input_bands`` with both ``"optical"`` and ``"sar"`` keys.
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_size: int,
        input_bands: dict[str, list[str]],
        output_layers: list[int],
        output_dim: int | list[int],
        s1_encoder_weights: str | Path,
        download_url: str = "",
        drop_path_rate: float = 0.3,
    ) -> None:
        super().__init__(
            model_name="skysense_s2s1",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=1024,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
        )
        self.s1_encoder_weights = s1_encoder_weights

        n_optical = len(input_bands["optical"])
        n_sar = len(input_bands["sar"])
        _out_indices = list(output_layers)

        self.s2_backbone = _VisionTransformer(
            img_size=input_size,
            patch_size=4,
            in_channels=n_optical,
            embed_dims=1024,
            num_layers=24,
            num_heads=16,
            mlp_ratio=4,
            out_indices=_out_indices,
            drop_path_rate=drop_path_rate,
            with_cls_token=True,
            final_norm=False,
        )
        self.s1_backbone = _VisionTransformer(
            img_size=input_size,
            patch_size=4,
            in_channels=n_sar,
            embed_dims=1024,
            num_layers=24,
            num_heads=16,
            mlp_ratio=4,
            out_indices=_out_indices,
            drop_path_rate=drop_path_rate,
            with_cls_token=True,
            final_norm=False,
        )

    def load_encoder_weights(self, logger: Logger) -> None:
        m1, i1 = _load_backbone(
            self.s2_backbone, self.encoder_weights, "s2_backbone", logger,
        )
        m2, i2 = _load_backbone(
            self.s1_backbone, self.s1_encoder_weights, "s1_backbone", logger,
        )
        self.parameters_warning({**m1, **m2}, {**i1, **i2}, logger)

    def forward(self, image: dict[str, torch.Tensor], metadata=None) -> list[torch.Tensor]:
        optical = image["optical"]
        sar = image["sar"]
        if optical.ndim == 5:
            optical = optical.squeeze(2)
        if sar.ndim == 5:
            sar = sar.squeeze(2)

        s2_feats = self.s2_backbone(optical)
        s1_feats = self.s1_backbone(sar)

        return [
            torch.cat([f2, f1], dim=1)
            for f2, f1 in zip(s2_feats, s1_feats)
        ]

class SkySenseVHR(Encoder):
    """SkySense VHR (high-resolution) SwinTransformerV2-Huge encoder only.

    Uses only the VHR/high-resolution backbone from the SkySense checkpoint,
    which is a SwinTransformerV2-Huge model pre-trained for 3-channel RGB images.
    Produces a 4-level spatial pyramid directly from the Swin stages without
    any fusion with ViT features.

    Output channels at each stage: [352, 704, 1408, 2816]
    Strides: [4, 8, 16, 32] (relative to input size)

    Expects ``input_bands`` with key ``"optical"`` (will use first 3 channels as R,G,B).
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_size: int,
        input_bands: dict[str, list[str]],
        output_layers: list[int],
        output_dim: int | list[int],
        download_url: str = "",
        drop_path_rate: float = 0.1,
        window_size: int = 8,
    ) -> None:
        assert len(output_layers) == 4, (
            "SkySenseVHR requires exactly 4 output_layers (one per Swin stage). "
            f"Got {len(output_layers)}."
        )
        # SkySenseVHR always outputs all 4 pyramid levels with fixed channel dims
        swin_channels = [352, 704, 1408, 2816]
        super().__init__(
            model_name="skysense_vhr",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=352,
            output_layers=output_layers,
            output_dim=swin_channels,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=True,
            download_url=download_url,
        )

        # VHR uses 3-channel RGB (R, G, B from optical modality)
        self.vhr_backbone = _SwinTransformerV2Huge(
            img_size=input_size,
            patch_size=4,
            in_channels=3,
            window_size=window_size,
            drop_path_rate=drop_path_rate,
            out_indices=(0, 1, 2, 3),
            use_abs_pos_embed=False,
            pad_small_map=True,
        )

    def load_encoder_weights(self, logger: Logger) -> None:
        m_vhr, i_vhr = _load_backbone(
            self.vhr_backbone, self.encoder_weights, "vhr_backbone", logger,
        )
        self.parameters_warning(m_vhr, i_vhr, logger)

    def forward(self, image: dict[str, torch.Tensor], metadata=None) -> list[torch.Tensor]:
        vhr = image["optical"]
        if vhr.ndim == 5:          # (B, C, T, H, W) — squeeze temporal dim
            vhr = vhr.squeeze(2)

        # Extract RGB in the order expected by the SwinV2-Huge backbone (R, G, B).
        # The config lists input_bands as [B4, B3, B2] (Red, Green, Blue) so that
        # the preprocessor delivers channels already in correct RGB order → take [:3].
        vhr = vhr[:, :3, :, :]

        # Return 4-level pyramid from SwinTransformerV2-Huge
        return list(self.vhr_backbone(vhr))
