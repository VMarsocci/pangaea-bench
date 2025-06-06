# Obtained from: https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT

from functools import partial
from logging import Logger
from pathlib import Path

import torch
import torch.nn as nn
from timm.layers import to_2tuple
from timm.models.vision_transformer import DropPath, Mlp

from pangaea.encoders.base import Encoder
from pangaea.encoders.pos_embed import interpolate_pos_embed


class SpectralGPT_Encoder(Encoder):
    """
    Paper: https://arxiv.org/abs/2311.07113
    Attributes:
        output_layers (int | list[int]): The layers from which to extract the output.
        num_frames (int): Number of frames, set to 1.
        patch_embed (PatchEmbed): Patch embedding layer.
        patchemb_input_size (tuple): Input size after patch embedding.
        cls_embed (bool): Whether to use class embedding.
        sep_pos_embed (bool): Whether to use separate positional embeddings for spatial and temporal dimensions.
        patch_size (int): Size of the patches.
        cls_token (nn.Parameter): Class token parameter.
        pos_embed_spatial (nn.Parameter): Spatial positional embedding parameter.
        pos_embed_temporal (nn.Parameter): Temporal positional embedding parameter.
        pos_embed_class (nn.Parameter): Class positional embedding parameter.
        pos_embed (nn.Parameter): Positional embedding parameter.
        blocks (nn.ModuleList): List of transformer blocks.
    Methods:
        __init__(self, encoder_weights: str | Path, input_size: int, input_bands: dict[str, list[str]], output_layers: int | list[int], in_chans: int = 3, t_patch_size: int = 3, patch_size: int = 16, output_dim: int = 768, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0, no_qkv_bias: bool = False, drop_path_rate: float = 0.5, norm_layer=nn.LayerNorm, sep_pos_embed=True, cls_embed=False):
            Initializes the SpectralGPT_Encoder with the given parameters.
        load_encoder_weights(self, logger: Logger) -> None:
            Loads the encoder weights from a pretrained model and handles any missing or incompatible shapes.
        forward(self, image):
            Forward pass of the encoder. Processes the input image and returns the output from the specified layers.
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_size: int,
        input_bands: dict[str, list[str]],
        output_layers: int | list[int],
        output_dim: int | list[int],
        download_url: str,
        in_chans: int = 3,
        t_patch_size: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        no_qkv_bias: bool = False,
        drop_path_rate: float = 0.5,
        norm_layer=nn.LayerNorm,
        sep_pos_embed=True,
        cls_embed=False,
    ):
        super().__init__(
            model_name="SpectralGPT",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
        )

        self.output_layers = output_layers

        # refer to: https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT/blob/ab6b965eb20c0e7ec373eb48ec99a67f711d4906/models_mae_spectral.py#L477
        self.num_frames = 1

        self.patch_embed = PatchEmbed(
            self.input_size,
            patch_size,
            self.num_frames,
            embed_dim,
            in_chans,
            t_patch_size,
        )

        num_patches = self.patch_embed.num_patches

        self.patchemb_input_size = self.patch_embed.input_size

        self.cls_embed = cls_embed
        self.sep_pos_embed = sep_pos_embed
        self.patch_size = patch_size

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1,
                    self.patchemb_input_size[1] * self.patchemb_input_size[2],
                    embed_dim,
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.patchemb_input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim), requires_grad=True
            )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    drop_path=dpr[i],
                    attn_func=partial(
                        Attention,
                        input_size=self.patchemb_input_size,
                    ),
                )
                for i in range(depth)
            ]
        )

    def load_encoder_weights(self, logger: Logger) -> None:
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)
        pretrained_model = pretrained_model["model"]
        interpolate_pos_embed(self, pretrained_model)

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

        self.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)

    def forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        # input image of shape B C H W
        x = image["optical"].unsqueeze(-3)  # B C H W -> B C 1 H W

        x = x.permute(0, 2, 1, 3, 4)  # for this model: B, T, C, H, W
        x = self.patch_embed(x)
        N, T, L, C = x.shape  # T: number of bands; L: spatial

        x = x.view([N, T * L, C])

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.patchemb_input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.patchemb_input_size[1] * self.patchemb_input_size[2],
                dim=1,
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            pos_embed = self.pos_embed[:, :, :]
        x = x + pos_embed

        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.output_layers:
                if self.cls_embed:
                    out = x[:, 1:]
                else:
                    out = x
                out = out.view(N, T, L, C).transpose(2, 3).flatten(1, 2)
                out = (
                    out.view(
                        x.shape[0],
                        -1,
                        self.input_size // self.patch_size,
                        self.input_size // self.patch_size,
                    ).contiguous()
                )
                output.append(out)

        return output


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        # temporal related:
        frames=32,
        t_patch_size=4,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0
        assert frames % t_patch_size == 0
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (frames // t_patch_size)
        )
        self.input_size = (
            frames // t_patch_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )

        self.frames = frames
        self.t_patch_size = t_patch_size
        self.img_size = img_size
        self.num_patches = num_patches

        self.grid_size = img_size[0] // patch_size[0]
        self.t_grid_size = frames // t_patch_size

        kernel_size = [t_patch_size] + list(patch_size)

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size
        )

    def forward(self, x):
        B, C, T, H, W = x.shape  # b, 1, in_chans, H, W

        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        assert T == self.frames
        x = self.proj(x).flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert self.input_size[1] == self.input_size[2]

    def forward(self, x):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x


class Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
