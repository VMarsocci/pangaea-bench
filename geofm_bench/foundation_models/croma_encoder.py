# Adapted from: https://github.com/antofuller/CROMA

import itertools
import math

import torch
from einops import rearrange
from torch import einsum, nn


class CROMA_OPTICAL_Encoder(nn.Module):
    def __init__(self, cfg, size="base", image_resolution=120):
        super().__init__()

        self.input_bands = cfg["input_bands"]
        self.output_layers = cfg["output_layers"]
        self.model_name = "CROMA_OPTICAL"
        self.img_size = image_resolution

        if size == "base":
            self.encoder_dim = 768
            self.encoder_depth = 12
            self.num_heads = 16
            self.patch_size = 8
        else:
            # large by default
            self.encoder_dim = 1024
            self.encoder_depth = 24
            self.num_heads = 16
            self.patch_size = 8

        self.num_patches = int((image_resolution / 8) ** 2)
        self.s2_channels = 12  # fixed at 12 multispectral optical channels
        self.attn_bias = get_2dalibi(
            num_heads=self.num_heads, num_patches=self.num_patches
        )

        self.s2_encoder = ViT(
            dim=self.encoder_dim, depth=self.encoder_depth, in_channels=self.s2_channels
        )

    def forward(self, image):
        output = self.s2_encoder(
            image["optical"].squeeze(2),
            self.attn_bias.to(image["optical"].device),
            self.output_layers,
        )  # (bsz, num_patches, encoder_dim)

        output = [
            x.permute(0, 2, 1)
            .view(
                x.shape[0],
                -1,
                self.img_size // self.patch_size,
                self.img_size // self.patch_size,
            )
            .contiguous()
            for x in output
        ]

        return output

    def load_encoder_weights(self, pretrained_path):
        pretrained_model = torch.load(pretrained_path, map_location="cpu")["s2_encoder"]
        k = pretrained_model.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.s2_encoder.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained_model[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
            else:
                pretrained_encoder[name] = pretrained_model[name]

        msg = self.s2_encoder.load_state_dict(pretrained_encoder, strict=False)

        return missing, incompatible_shape


class CROMA_SAR_Encoder(nn.Module):
    def __init__(self, cfg, size="base", image_resolution=120):
        super().__init__()

        self.input_bands = cfg["input_bands"]
        self.output_layers = cfg["output_layers"]
        self.model_name = "CROMA_SAR"
        self.img_size = image_resolution

        if size == "base":
            self.encoder_dim = 768
            self.encoder_depth = 12
            self.num_heads = 16
            self.patch_size = 8
        else:
            # large by default
            self.encoder_dim = 1024
            self.encoder_depth = 24
            self.num_heads = 16
            self.patch_size = 8

        self.num_patches = int((image_resolution / 8) ** 2)
        self.s1_channels = 2  # fixed at 2 SAR backscatter channels
        self.attn_bias = get_2dalibi(
            num_heads=self.num_heads, num_patches=self.num_patches
        )

        self.s1_encoder = ViT(
            dim=self.encoder_dim,
            depth=int(self.encoder_depth / 2),
            in_channels=self.s1_channels,
        )

    def forward(self, image):
        # output = []

        output = self.s1_encoder(
            image["sar"].squeeze(2),
            self.attn_bias.to(image["sar"].device),
            self.output_layers,
        )  # (bsz, num_patches, encoder_dim)

        output = [
            x.permute(0, 2, 1)
            .view(
                x.shape[0],
                -1,
                self.img_size // self.patch_size,
                self.img_size // self.patch_size,
            )
            .contiguous()
            for x in output
        ]

        return output

    def load_encoder_weights(self, pretrained_path):
        pretrained_model = torch.load(pretrained_path, map_location="cpu")["s1_encoder"]
        k = pretrained_model.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.s1_encoder.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained_model[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
            else:
                pretrained_encoder[name] = pretrained_model[name]

        msg = self.s1_encoder.load_state_dict(pretrained_encoder, strict=False)

        return missing, incompatible_shape


class CROMA_JOINT_Encoder(nn.Module):
    def __init__(self, cfg, size="base", image_resolution=120):
        super().__init__()

        self.input_bands = cfg["input_bands"]
        self.output_layers = cfg["output_layers"]
        self.model_name = "CROMA"
        self.img_size = image_resolution

        if size == "base":
            self.encoder_dim = 768
            self.encoder_depth = 12
            self.num_heads = 16
            self.patch_size = 8
        else:
            # large by default
            self.encoder_dim = 1024
            self.encoder_depth = 24
            self.num_heads = 16
            self.patch_size = 8

        self.num_patches = int((image_resolution / 8) ** 2)
        self.s1_channels = 2  # fixed at 2 SAR backscatter channels
        self.s2_channels = 12  # fixed at 12 multispectral optical channels
        self.attn_bias = get_2dalibi(
            num_heads=self.num_heads, num_patches=self.num_patches
        )

        self.s1_encoder = ViT(
            dim=self.encoder_dim,
            depth=int(self.encoder_depth / 2),
            in_channels=self.s1_channels,
        )
        self.s2_encoder = ViT(
            dim=self.encoder_dim, depth=self.encoder_depth, in_channels=self.s2_channels
        )
        self.cross_encoder = BaseTransformerCrossAttn(
            dim=self.encoder_dim,
            depth=int(self.encoder_depth / 2),
            num_heads=self.num_heads,
        )

    def forward(self, image):
        attn_bias = self.attn_bias.to(image["optical"].device)
        SAR_encodings = self.s1_encoder(
            image["sar"].squeeze(2), attn_bias
        )  # (bsz, num_patches, encoder_dim)
        optical_encodings = self.s2_encoder(
            image["optical"].squeeze(2), attn_bias
        )  # (bsz, num_patches, encoder_dim)
        output = self.cross_encoder(
            x=SAR_encodings,
            context=optical_encodings,
            relative_position_bias=attn_bias,
            output_layers=self.output_layers,
        )

        output = [
            x.permute(0, 2, 1)
            .view(
                x.shape[0],
                -1,
                self.img_size // self.patch_size,
                self.img_size // self.patch_size,
            )
            .contiguous()
            for x in output
        ]

        return output

    def load_encoder_weights(self, pretrained_path):
        pretrained_model = torch.load(pretrained_path, map_location="cpu")
        combined_state_dict = {}
        for prefix, module in pretrained_model.items():
            for k, v in module.items():
                combined_state_dict[
                    prefix.replace("joint_encoder", "cross_encoder") + "." + k
                ] = v

        pretrained_model = combined_state_dict

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

        msg = self.load_state_dict(pretrained_encoder, strict=False)

        return missing, incompatible_shape


def get_2dalibi(num_heads, num_patches):
    # inspired by: https://github.com/ofirpress/attention_with_linear_biases
    points = list(
        itertools.product(
            range(int(math.sqrt(num_patches))), range(int(math.sqrt(num_patches)))
        )
    )

    def get_slopes(n):
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

    slopes = torch.Tensor(get_slopes(num_heads)).unsqueeze(1)
    idxs = []
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist * slopes * -1)
    all_bias = torch.cat(idxs, dim=1)
    return all_bias.view(1, num_heads, num_patches, num_patches)


class FFN(nn.Module):
    def __init__(
        self,
        dim,
        mult=4,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),  # (BSZ, num_patches, inner_dim)
            nn.GELU(),  # (BSZ, num_patches, inner_dim)
            nn.Dropout(dropout),  # (BSZ, num_patches, inner_dim)
            nn.Linear(inner_dim, dim),  # (BSZ, num_patches, dim)
        )
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        return self.net(x)  # (BSZ, num_patches, dim)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be evenly divisible by num_heads"
        dim_head = int(dim / num_heads)
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, relative_position_bias):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  # (BSZ, num_patches, dim)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v)
        )  # (BSZ, num_heads, num_patches, dim_head)

        attention_scores = (
            einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        )  # (BSZ, num_heads, num_patches, num_patches)
        attention_scores = (
            attention_scores + relative_position_bias
        )  # (BSZ, num_heads, num_patches, num_patches)

        attn = attention_scores.softmax(
            dim=-1
        )  # (BSZ, num_heads, num_patches, num_patches)
        attn = self.dropout(attn)  # (BSZ, num_heads, num_patches, num_patches)

        out = einsum(
            "b h i j, b h j d -> b h i d", attn, v
        )  # (BSZ, num_heads, num_patches, dim_head)
        out = rearrange(out, "b h n d -> b n (h d)")  # (BSZ, num_patches, dim)
        return self.to_out(out)  # (BSZ, num_patches, dim)


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be evenly divisible by num_heads"
        dim_head = int(dim / num_heads)
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, relative_position_bias):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        context = self.input_norm(context)  # (BSZ, num_patches, dim)

        q = self.to_q(x)  # (BSZ, num_patches, dim)
        k = self.to_k(context)  # (BSZ, num_patches, dim)
        v = self.to_v(context)  # (BSZ, num_patches, dim)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v)
        )  # (BSZ, num_heads, num_patches, dim_head)

        attention_scores = (
            einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        )  # (BSZ, num_heads, num_patches, num_patches)
        attention_scores = (
            attention_scores + relative_position_bias
        )  # (BSZ, num_heads, num_patches, num_patches)

        attn = attention_scores.softmax(
            dim=-1
        )  # (BSZ, num_heads, num_patches, num_patches)
        attn = self.dropout(attn)  # (BSZ, num_heads, num_patches, num_patches)

        out = einsum(
            "b h i j, b h j d -> b h i d", attn, v
        )  # (BSZ, num_heads, num_patches, dim_head)
        out = rearrange(out, "b h n d -> b n (h d)")  # (BSZ, num_patches, dim)
        return self.to_out(out)  # (BSZ, num_patches, dim)


class BaseTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        final_norm=True,
    ):
        super().__init__()
        self.final_norm = final_norm
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                        FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        if self.final_norm:
            self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, relative_position_bias=False, output_layers=None):
        output = []
        for i, layer in enumerate(self.layers):
            self_attn, ffn = layer
            x = self_attn(x, relative_position_bias) + x  # (BSZ, num_patches, dim)
            x = ffn(x) + x  # (BSZ, num_patches, dim)

            if output_layers is not None and i in output_layers:
                output.append(x)

        if self.final_norm:
            if output_layers is None:
                x = self.norm_out(x)
            else:
                output[-1] = self.norm_out(output[-1])

        if output_layers is None:
            return x
        else:
            return output


class BaseTransformerCrossAttn(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                        CrossAttention(
                            dim=dim, num_heads=num_heads, dropout=attn_dropout
                        ),
                        FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, context, relative_position_bias, output_layers):
        output = []

        for i, layer in enumerate(self.layers):
            self_attn, cross_attn, ffn = layer
            x = self_attn(x, relative_position_bias) + x  # (BSZ, num_patches, dim)
            x = (
                cross_attn(x, context, relative_position_bias) + x
            )  # (BSZ, num_patches, dim)
            x = ffn(x) + x  # (BSZ, num_patches, dim)
            if output_layers is not None and i in output_layers:
                output.append(x)

        if output_layers is None:
            x = self.norm_out(x)
            return x
        else:
            output[-1] = self.norm_out(output[-1])
            return output


class ViT(nn.Module):
    def __init__(self, dim, depth, in_channels):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.dim = dim
        self.num_heads = 16  # always 16, for base and large models
        self.patch_size = 8  # always 8, for base and large models

        pixels_per_patch = int(self.patch_size * self.patch_size * in_channels)
        self.linear_input = nn.Linear(pixels_per_patch, self.dim)
        self.transformer = BaseTransformer(
            dim=self.dim,
            depth=self.depth,
            num_heads=self.num_heads,
        )

    def forward(self, imgs, attn_bias, output_layers=None):
        x = rearrange(
            imgs,
            "b c (h i) (w j) -> b (h w) (c i j)",
            i=self.patch_size,
            j=self.patch_size,
        )
        # x is shape -> (bsz, num_patches, self.channels*self.patch_size*self.patch_size)

        x = self.linear_input(x)  # (bsz, num_patches, dim)
        output = self.transformer(
            x, relative_position_bias=attn_bias, output_layers=output_layers
        )

        return output
