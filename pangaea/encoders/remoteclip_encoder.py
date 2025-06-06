import math
from collections import OrderedDict
from logging import Logger
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pangaea.encoders.base import Encoder

from .pos_embed import get_2d_sincos_pos_embed


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(
            x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps
        )
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        scaled_cosine: bool = False,
        scale_heads: bool = False,
        logit_scale_max: float = math.log(1.0 / 0.01),
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.logit_scale_max = logit_scale_max
        self.use_fsdpa = hasattr(nn.functional, "scaled_dot_product_attention")

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1)))
            )
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        x = x.transpose(0, 1)

        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.reshape(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.reshape(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.reshape(L, N * self.num_heads, -1).transpose(0, 1)

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        if self.logit_scale is not None:
            attn = torch.bmm(
                F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2)
            )
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = torch.bmm(attn, v)
        else:
            if self.use_fsdpa:
                x = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
            else:
                q = q * self.scale
                attn = torch.bmm(q, k.transpose(-1, -2))
                if attn_mask is not None:
                    attn += attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = torch.bmm(attn, v)

        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)

        x = x.transpose(0, 1).reshape(L, N, C)

        x = x.transpose(0, 1)

        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )

    def attention(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = (
            self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        )
        v_x = (
            self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        )
        x = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        output_layers,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.output_layers = output_layers

        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(layers)
            ]
        )

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, "int8_original_dtype"):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        output = []
        for i, r in enumerate(self.resblocks):
            x = r(x, attn_mask=attn_mask)
            if i in self.output_layers:
                # out = x[:, 1:].permute(0, 2, 1).view(-1, self.embed_dim, self.img_size // self.patch_size, self.img_size // self.patch_size).contiguous()
                output.append(x)

        return output


class RemoteCLIP_Encoder(Encoder):
    """
    Paper: https://arxiv.org/pdf/2306.11029
    Attributes:
        output_layers (int | list[int]): The layers from which to extract the output.
        image_size (tuple[int, int]): The size of the input image.
        patch_size (tuple[int, int]): The size of the patches extracted from the input image.
        grid_size (tuple[int, int]): The size of the grid formed by the patches.
        width (int): The width of the convolutional layer.
        conv1 (nn.Conv2d): The first convolutional layer.
        class_embedding (nn.Parameter): The class embedding parameter.
        positional_embedding (nn.Parameter): The positional embedding parameter.
        patch_dropout (nn.Module): The patch dropout layer.
        ln_pre (nn.Module): The layer normalization layer.
        transformer (Transformer): The transformer module.
    Methods:
        __init__(self, encoder_weights: str | Path, input_bands: dict[str, list[str]], input_size: int, embed_dim: int, patch_size: int, width: int, head_width: int, layers: int, mlp_ratio: float, output_layers: int | list[int], ls_init_value: float = None, patch_dropout: float = 0., no_ln_pre: bool = False, pos_embed_type: str = 'learnable', act_layer: Callable = nn.GELU, norm_layer: Callable = LayerNorm):
            Initializes the RemoteCLIP_Encoder with the given parameters.
        freeze(self):
            Freezes the parameters of the encoder to prevent them from being updated during training.
        load_encoder_weights(self, logger: Logger) -> None:
            Loads the encoder weights from a pre-trained model and logs any missing or incompatible parameters.
        forward(self, image):
            Performs a forward pass through the encoder with the given input image.
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_bands: dict[str, list[str]],
        input_size: int,
        embed_dim: int,
        patch_size: int,
        width: int,
        head_width: int,
        layers: int,
        mlp_ratio: float,
        output_layers: int | list[int],
        output_dim: int | list[int],
        download_url: str,
        ls_init_value: float | None = None,
        patch_dropout: float = 0.0,
        no_ln_pre: bool = False,
        pos_embed_type: str = "learnable",
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
    ):
        super().__init__(
            model_name="remoteclip_encoder",
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

        image_height, image_width = self.image_size = (self.input_size, self.input_size)
        patch_height, patch_width = self.patch_size = (patch_size, patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.width = width

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # class embeddings and positional embeddings
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == "learnable":
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
            )
        elif pos_embed_type == "sin_cos_2d":
            # fixed sin-cos embedding
            assert (
                self.grid_size[0] == self.grid_size[1]
            ), "currently sin cos 2d pos embedding only supports square input"
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width),
                requires_grad=False,
            )
            pos_embed_type = get_2d_sincos_pos_embed(
                width, self.grid_size[0], cls_token=True
            )
            self.positional_embedding.data.copy_(
                torch.from_numpy(pos_embed_type).float()
            )
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = (
            PatchDropout(patch_dropout) if patch_dropout > 0.0 else nn.Identity()
        )

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)

        # print(self.output_layers)
        heads = width // head_width
        self.transformer = Transformer(
            width,
            layers,
            heads,
            self.output_layers,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def load_encoder_weights(self, logger: Logger) -> None:
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)
        visual_only_model = {}
        for k, v in pretrained_model.items():
            if k.startswith("visual."):
                visual_only_model[k.replace("visual.", "")] = v
        pretrained_model = visual_only_model

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
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1
        )
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        output = self.transformer(x)

        output = [
            out[:, 1:]
            .permute(0, 2, 1)
            .view(
                x.shape[0],
                -1,
                self.image_size[0] // self.patch_size[0],
                self.image_size[1] // self.patch_size[1],
            )
            .contiguous()
            for out in output
        ]

        return output
