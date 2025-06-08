# Adapted from: https://github.com/bair-climate-initiative/scale-mae/

from functools import partial
from logging import Logger
from pathlib import Path

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed

from pangaea.encoders.base import Encoder
from pangaea.encoders.pos_embed import get_2d_sincos_pos_embed_with_resolution


class PatchEmbedUnSafe(PatchEmbed):
    """Image to Patch Embedding"""

    def forward(self, x):
        B, C, H, W = x.shape
        # Dropped size check in timm
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ScaleMAE_Encoder(Encoder):
    """
    Paper: https://arxiv.org/pdf/2212.14532
    Attributes:
        output_layers (int | list[int]): The layers from which to output features.
        img_size (int): The size of the input image.
        patch_size (int): The size of the patches to divide the image into.
        input_res (torch.Tensor): The input resolution.
        patch_embed (PatchEmbedUnSafe): The patch embedding layer.
        cls_token (nn.Parameter): The class token parameter.
        blocks (nn.ModuleList): The list of transformer blocks.
    Methods:
        __init__(encoder_weights: str | Path, input_size: int, input_bands: dict[str, list[str]], output_layers: int | list[int], embed_dim: int = 1024, patch_size: int = 16, in_chans: int = 3, depth: int = 24, num_heads: int = 16, mlp_ratio: float = 4., qkv_bias: bool = True, input_res: float = 1.0, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
            Initializes the ScaleMAE_Encoder with the given parameters.
        initialize_weights():
            Initializes the weights of the model.
        _init_weights(m):
            Initializes the weights of the given module.
        load_encoder_weights(logger: Logger) -> None:
            Loads the encoder weights from a pretrained model.
        forward(image):
            Forward pass of the encoder.
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_size: int,
        input_bands: dict[str, list[str]],
        output_layers: int | list[int],
        output_dim: int | list[int],
        download_url: str,
        embed_dim: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        input_res: float = 1.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__(
            model_name="ScaleMAE",
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

        self.img_size = input_size
        self.patch_size = patch_size

        self.input_res = torch.tensor([input_res]).float().cpu()

        self.patch_embed = PatchEmbedUnSafe(
            self.img_size, patch_size, in_chans, embed_dim
        )
        # num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(
        #    torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        # )  # fixed sin-cos embedding
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        # self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_encoder_weights(self, logger: Logger) -> None:
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)["model"]
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

    def forward(self, image):
        x = image["optical"].squeeze(2)
        B, _, h, w = x.shape
        x = self.patch_embed(x)

        num_patches = int(
            (h * w) / (self.patch_embed.patch_size[0] * self.patch_embed.patch_size[1])
        )
        pos_embed = get_2d_sincos_pos_embed_with_resolution(
            x.shape[-1],
            int(num_patches**0.5),
            self.input_res,
            cls_token=True,
            device=x.device,
        )

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed.to(x.dtype)

        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.output_layers:
                out = (
                    x[:, 1:]
                    .permute(0, 2, 1)
                    .view(
                        x.shape[0],
                        -1,
                        self.img_size // self.patch_size,
                        self.img_size // self.patch_size,
                    )
                    .contiguous()
                )
                output.append(out)

        return output
