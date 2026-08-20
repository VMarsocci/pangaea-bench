# RAMEN Encoder 

from logging import Logger
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.vision_transformer import Block

import copy
import numpy as np

from pangaea.encoders.base import Encoder
from pangaea.encoders.pos_embed import get_1d_sincos_pos_embed_from_grid_torch, get_2d_sincos_pos_embed_with_resolution

class RAMEN_Encoder(Encoder):
    """
        Paper: https://arxiv.org/abs/2512.05025
        Attributes:
            output_layers (int | list[int]): The layers from which to extract the output.
            img_size (int): The size of the input image.
            input_res (float): The resolution of the input image in ground sampling distance (e.g. 10.0 for Sentinel-2).
            res (float): The resolution of the output image in ground sampling distance.
            wavelengths (dict): A dictionary mapping each modality to a list of wavelengths for each band (radar images expect strings for corresponding polarizations).
            embed_dim (int): The embedding dimension.
        Methods:
            __init__(encoder_weights, input_bands, input_size, embed_dim, output_layers, wavelengths, input_res, res, depth=12, num_heads=12, mlp_ratio=4.0, class_token=True, qvk_bias=True, qk_scale=None, drop_rate=0.0, drop_path_rate=0.0, attn_drop_rate=0.0):
                Initializes the RAMEN_Encoder with the given parameters.
            forward(image):
                Forward pass of the encoder. Takes an input image and returns the encoded output.
            load_encoder_weights(logger):
                Loads the encoder weights from a pretrained model and logs any missing or incompatible parameters.
        """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_bands: dict[str, list[str]],
        input_size: int,
        embed_dim: int,
        output_dim: int | list[int],
        output_layers: int | list[int],
        download_url: str,
        input_res: float = 10.0,
        output_res: float = 40.0,
        wavelengths: dict = {},
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        class_token: bool = True,
        qvk_bias: bool = True,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__(
            model_name="ramen_encoder",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim*len(input_bands),
            multi_temporal=True,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
        )

        self.output_layers = output_layers
        self.img_size = input_size

        self.input_bands = input_bands

        self.modalities = list(self.input_bands.keys())
        self.input_res = input_res
        self.output_res = output_res
        self.wavelengths = {m: [wavelengths[m][band] for band in self.input_bands[m]] for m in self.modalities}
        self.embed_dim = embed_dim


        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None

        self.spectral_projector = SpectralProjector(embed_dim, embed_dim*2)
        if "sar" in self.modalities:
            self.radar_projector = RadarProjector(embed_dim, embed_dim*2)
        if "dem" in self.modalities:
            self.dem_projector = DemProjector(embed_dim, embed_dim*2)

        self.resampler = ScaleResampler(embed_dim)

        self.ltae = LTAE2d(
                    in_channels=embed_dim,
                    n_head=16,
                    d_k=16,
                    mlp= [embed_dim, embed_dim*2, embed_dim],
                    dropout=0.,
                    mlp_in= [embed_dim],
                    T= 367,
                    in_norm=True,
                    return_att=False,
                    positional_encoding=True,
                )        

        self.in_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qvk_bias,
                qk_norm=qk_scale,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.effective_size = int(self.input_size * (self.input_res / self.output_res))
        self.pos_embed = get_2d_sincos_pos_embed_with_resolution(
            embed_dim,
            self.effective_size,
            torch.tensor([self.output_res]),
            cls_token=True,
            )

    def forward(self, x):
        device = x[self.modalities[0]].device
        dtype = x[self.modalities[0]].dtype
        output = []
        out = {}
        out['pos_embed'] = self.pos_embed.to(device=device, dtype=dtype)
        for modality in self.modalities:
            x_mod = x[modality] # [B, C, T, H, W]
            B, C, T, H, W = x_mod.shape
            x_mod = x_mod.permute(0, 2, 3, 4, 1).contiguous() # [B, T, H, W, C]

            if modality in ["sar"]:
                spectral_encoding = self.radar_projector(
                    self.wavelengths[modality], device
                )
            elif modality in ["dem"]:
                spectral_encoding = self.dem_projector(
                    self.wavelengths[modality], device
                )
            else:
                spectral_encoding = self.spectral_projector(
                    torch.Tensor(self.wavelengths[modality]).to(device=device, dtype=dtype)
                )

            out_mod = x_mod @ spectral_encoding # [B, T, H, W, C] @ [C, D] -> [B, T, H, W, D]
            out_mod = out_mod.permute(0, 1, 4, 2, 3).reshape(B * T, -1, H, W).contiguous() # [B*T, D, H, W]

            expected_size = self.effective_size
            scale = (self.input_res / self.output_res)
            out_mod = self.resampler(out_mod, scale)
                    
            if out_mod.shape[2] != expected_size or out_mod.shape[3] != expected_size:
                out_mod = F.interpolate(out_mod, size=(expected_size, expected_size), mode='bilinear')
                
            if T > 1:
                out_mod = out_mod.view(B, T, -1, expected_size, expected_size)  # [B, T, D, H, W]
                out_mod = self.ltae(
                    out_mod, batch_positions=x['_'.join([modality, "dates"])], pad_mask=None
                )  # [B, D, H, W]

            out_mod = out_mod.flatten(2).transpose(1, 2)  # [B, (H*W), D]
            out_mod = self.in_norm(out_mod)
            out[modality] = out_mod

        tokens = torch.cat([out[modality] for modality in self.modalities], dim=1)
        tokens = tokens + out['pos_embed'][:, 1:, :].repeat(1, len(self.modalities), 1)

        if self.cls_token is not None:
            cls_tokens = (self.cls_token + out['pos_embed'][:, :1, :]).expand(tokens.shape[0], -1, -1)
            tokens = torch.cat((cls_tokens, tokens), dim=1)

        for j, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if j == len(self.blocks) - 1:
                tokens = self.norm(tokens)
            if j in self.output_layers:
                if len(self.modalities) > 1:
                    n_tok = tokens[:, 1:, :].shape[1] // len(self.modalities)
                    patch_tokens = [tokens[:, 1 + i*n_tok : 1 + (i+1)*n_tok, :] for i in range(len(self.modalities))]
                    patch_tokens = torch.cat(patch_tokens, dim=-1)
                    out = (
                        patch_tokens
                        .permute(0, 2, 1)
                        .view(
                            patch_tokens.shape[0],
                            -1,
                            expected_size,
                            expected_size,
                        )
                        .contiguous()
                    )
                else:
                    out = (
                        tokens[:, 1:]
                        .permute(0, 2, 1)
                        .view(
                            tokens.shape[0],
                            -1,
                            expected_size,
                            expected_size,
                        )
                        .contiguous()
                    )
                output.append(out)

        return output

    def load_encoder_weights(self, logger: Logger) -> None:
        
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)
        k = pretrained_model["state_dict"].keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.named_parameters():
            clean_name = name
            for prefix in ["model.encoder.", "encoder."]:
                if name.startswith(prefix):
                    clean_name = name[len(prefix):]
                    break
            if clean_name not in k:
                missing[clean_name] = param.shape
            elif pretrained_model["state_dict"][clean_name].shape != param.shape:
                incompatible_shape[clean_name] = (param.shape, pretrained_model["state_dict"][clean_name].shape)
            else:
                pretrained_encoder[clean_name] = pretrained_model["state_dict"][clean_name]

        self.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)

class RAMEN_Encoder_MonoTemporal(Encoder):
    """
        Paper: https://arxiv.org/abs/2512.05025
        RAMEN encode input in mono-temporal mode, process each timesteps independently to reproduce results of the paper for multi-temporal tasks.
        Attributes:
            output_layers (int | list[int]): The layers from which to extract the output.
            img_size (int): The size of the input image.
            input_res (float): The resolution of the input image in ground sampling distance (e.g. 10.0 for Sentinel-2).
            res (float): The resolution of the output image in ground sampling distance.
            wavelengths (dict): A dictionary mapping each modality to a list of wavelengths for each band (radar images expect strings for corresponding polarizations).
            embed_dim (int): The embedding dimension.
        Methods:
            __init__(encoder_weights, input_bands, input_size, embed_dim, output_layers, wavelengths, input_res, res, depth=12, num_heads=12, mlp_ratio=4.0, class_token=True, qvk_bias=True, qk_scale=None, drop_rate=0.0, drop_path_rate=0.0, attn_drop_rate=0.0):
                Initializes the RAMEN_Encoder with the given parameters.
            forward(image):
                Forward pass of the encoder. Takes an input image and returns the encoded output.
            load_encoder_weights(logger):
                Loads the encoder weights from a pretrained model and logs any missing or incompatible parameters.
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_bands: dict[str, list[str]],
        input_size: int,
        embed_dim: int,
        output_dim: int | list[int],
        output_layers: int | list[int],
        download_url: str,
        input_res: float = 10.0,
        output_res: float = 40.0,
        wavelengths: dict = {},
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        class_token: bool = True,
        qvk_bias: bool = True,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__(
            model_name="ramen_encoder",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim*len(input_bands),
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
        )

        self.output_layers = output_layers
        self.img_size = input_size

        self.input_bands = input_bands

        self.modalities = self.input_bands.keys()
        self.input_res = input_res
        self.output_res = output_res
        self.wavelengths = {m: [wavelengths[m][band] for band in self.input_bands[m]] for m in self.modalities}
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None

        self.spectral_projector = SpectralProjector(embed_dim, embed_dim*2)
        if "sar" in self.modalities:
            self.radar_projector = RadarProjector(embed_dim, embed_dim*2)
        if "dem" in self.modalities:
            self.dem_projector = DemProjector(embed_dim, embed_dim*2)

        self.resampler = ScaleResampler(embed_dim) 

        self.in_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qvk_bias,
                qk_norm=qk_scale,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.effective_size = int(self.input_size * (self.input_res / self.output_res))
        self.pos_embed = get_2d_sincos_pos_embed_with_resolution(
            embed_dim,
            self.effective_size,
            torch.tensor([self.output_res]),
            cls_token=True,
            )

    def forward(self, x):
        device = x[self.modalities[0]].device
        dtype = x[self.modalities[0]].dtype
        output = []
        out = {}
        out['pos_embed'] = self.pos_embed.to(device=device, dtype=dtype)
        for modality in self.modalities:
            x_mod = x[modality] # [B, C, H, W]
            x_mod = x_mod.permute(0, 2, 3, 1).contiguous() # [B, H, W, C]

            if modality in ["sar"]:
                spectral_encoding = self.radar_projector(
                    self.wavelengths[modality], device
                )
            elif modality in ["dem"]:
                spectral_encoding = self.dem_projector(
                    self.wavelengths[modality], device
                )
            else:
                spectral_encoding = self.spectral_projector(
                    torch.Tensor(self.wavelengths[modality]).to(device=device, dtype=dtype)
                )
            out_mod = x_mod @ spectral_encoding # [B, H, W, C] @ [C, D] -> [B, H, W, D]
            out_mod = out_mod.permute(0, 3, 1, 2).contiguous() # [B, D, H, W]

            expected_size = self.effective_size
            scale = (self.input_res / self.output_res)
            out_mod = self.resampler(out_mod, scale)
                    
            if out_mod.shape[2] != expected_size or out_mod.shape[3] != expected_size:
                out_mod = F.interpolate(out_mod, size=(expected_size, expected_size), mode='bilinear')

            out_mod = out_mod.flatten(2).transpose(1, 2)  # [B, (H*W), D]
            out_mod = self.in_norm(out_mod)
            out[modality] = out_mod

        tokens = torch.cat([out[modality] for modality in self.modalities], dim=1)
        tokens = tokens + out['pos_embed'][:, 1:, :].repeat(1, len(self.modalities), 1)

        if self.cls_token is not None:
            cls_tokens = (self.cls_token + out['pos_embed'][:, :1, :]).expand(tokens.shape[0], -1, -1)
            tokens = torch.cat((cls_tokens, tokens), dim=1)

        for j, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if j == len(self.blocks) - 1:
                tokens = self.norm(tokens)
            if j in self.output_layers:
                if len(self.modalities) > 1:
                    n_tok = tokens[:, 1:, :].shape[1] // len(self.modalities)
                    patch_tokens = [tokens[:, 1 + i*n_tok : 1 + (i+1)*n_tok, :] for i in range(len(self.modalities))]
                    patch_tokens = torch.cat(patch_tokens, dim=-1)
                    out = (
                        patch_tokens
                        .permute(0, 2, 1)
                        .view(
                            patch_tokens.shape[0],
                            -1,
                            expected_size,
                            expected_size,
                        )
                        .contiguous()
                    )
                else:
                    out = (
                        tokens[:, 1:]
                        .permute(0, 2, 1)
                        .view(
                            tokens.shape[0],
                            -1,
                            expected_size,
                            expected_size,
                        )
                        .contiguous()
                    )
                output.append(out)

        return output

    def load_encoder_weights(self, logger: Logger) -> None:
        
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)
        k = pretrained_model["state_dict"].keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.named_parameters():
            clean_name = name
            for prefix in ["model.encoder.", "encoder."]:
                if name.startswith(prefix):
                    clean_name = name[len(prefix):]
                    break
            if clean_name not in k:
                missing[clean_name] = param.shape
            elif pretrained_model["state_dict"][clean_name].shape != param.shape:
                incompatible_shape[clean_name] = (param.shape, pretrained_model["state_dict"][clean_name].shape)
            else:
                pretrained_encoder[clean_name] = pretrained_model["state_dict"][clean_name]

        self.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)

class ScaleEncoding(nn.Module):
    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: Dimension of the positional encoding.
        """
        super().__init__()
        assert embed_dim % 2 == 0, "Embedding dimension must be even"
        self.embed_dim = embed_dim

    def forward(self, scales: Tensor) -> Tensor:
        """
        Args:
            wavelengths: Tensor of spectral positions [C].
                        e.g., scales of each input channel.
        Returns:
            pos_emb: [C, embed_dim]
        """

        scales = torch.log(scales)
        pos_emb = get_1d_sincos_pos_embed_from_grid_torch(self.embed_dim, scales)
        
        return pos_emb

class ScaleResampler(nn.Module):
    
    def __init__(self, embed_dim: int, num_experts: int = 4):

        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1) for _ in range(num_experts)
        ])

        self.scale_encoding = ScaleEncoding(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, num_experts),
        )
        self.temperature = nn.Parameter(torch.tensor(0.7))

    def forward(self, x, scale):
        
        x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)

        scale_tensor = torch.tensor([scale], device=x.device, dtype=x.dtype)
        scale_emb = self.scale_encoding(scale_tensor)

        weights = (self.mlp(scale_emb) / self.temperature.clamp(min=1e-3)).softmax(dim=-1)    

        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            out += weights[0, i] * expert(x)
        return x + out

class SpectralEncoding(nn.Module):

    def __init__(self, embed_dim: int, normalize: bool = False):
        """
        Args:
            embed_dim: Dimension of the positional encoding.
            normalize: Whether to normalize wavelengths before encoding.
        """
        super().__init__()
        assert embed_dim % 2 == 0, "Embedding dimension must be even"
        self.embed_dim = embed_dim
        self.normalize = normalize

    def forward(self, wavelengths: Tensor) -> Tensor:
        """
        Args:
            wavelengths: Tensor of spectral positions [C].
                        e.g., wavelengths of each input channel.
        Returns:
            pos_emb: [C, embed_dim]
        """
        if self.normalize:
            wavelengths = (wavelengths - 350.0) / (2500.0 - 350.0)
            wavelengths = wavelengths.clamp(0, 1)

        pos_emb = get_1d_sincos_pos_embed_from_grid_torch(self.embed_dim, wavelengths)
        return pos_emb

class SpectralProjector(nn.Module):

    def __init__(self, embed_dim=768, hidden_dim=768):

        super().__init__()
        self.embed_dim = embed_dim

        self.spectral_encoding = SpectralEncoding(embed_dim, normalize=False)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, wavelengths: Tensor) -> Tensor:
        """
        Args:
            wavelengths: Tensor of spectral positions [C].
                        e.g., wavelengths of each input channel.
        Returns:
            pos_emb: [C, embed_dim]
        """
        spectral_encodings = self.spectral_encoding(wavelengths)
        return self.mlp(spectral_encodings)

class RadarProjector(nn.Module):

    def __init__(self, embed_dim=768, hidden_dim=768):

        super().__init__()
        self.embed_dim = embed_dim

        self.asc_vv_param = nn.Parameter(torch.empty(embed_dim))
        self.asc_vh_param = nn.Parameter(torch.empty(embed_dim))
        self.asc_hv_param = nn.Parameter(torch.empty(embed_dim))
        self.asc_hh_param = nn.Parameter(torch.empty(embed_dim))
        
        self.des_vv_param = nn.Parameter(torch.empty(embed_dim))
        self.des_vh_param = nn.Parameter(torch.empty(embed_dim))
        self.des_hv_param = nn.Parameter(torch.empty(embed_dim))
        self.des_hh_param = nn.Parameter(torch.empty(embed_dim))

        for name, param in self.named_parameters():
            if "param" in name:
                nn.init.normal_(param, 0.0, 0.02)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.pol_map = {
            "asc_vv": self.asc_vv_param,
            "asc_vh": self.asc_vh_param,
            "asc_hv": self.asc_hv_param,
            "asc_hh": self.asc_hh_param,
            "des_vv": self.des_vv_param,
            "des_vh": self.des_vh_param,
            "des_hv": self.des_hv_param,
            "des_hh": self.des_hh_param,
        }

    def forward(self, polarizations: list, device=None) -> torch.Tensor:
        """
        Args:
            polarizations: List of polarization strings ['vv', 'vh', 'hh', 'hv'] of length [C].
            device: torch.device where the output should be placed.
        Returns:
            pos_emb: [C, hidden_dim]
        """
        device = device or self.asc_vv_param.device 

        try:
            embeddings = torch.stack(
                [self.pol_map[pol.lower()].to(device) for pol in polarizations],
                dim=0
            )
        except KeyError as e:
            raise ValueError(f"Unknown polarization: {e.args[0]}. "
                             f"Valid options: {list(self.pol_map.keys())}")

        return self.mlp(embeddings)                  # [C, embed_dim]

class DemProjector(nn.Module):

    def __init__(self, embed_dim=768, hidden_dim=768):

        super().__init__()
        self.embed_dim = embed_dim

        self.dsm_param = nn.Parameter(torch.empty(embed_dim))
        self.dtm_param = nn.Parameter(torch.empty(embed_dim))
        self.slope_param = nn.Parameter(torch.empty(embed_dim))

        for name, param in self.named_parameters():
            if "param" in name:
                nn.init.normal_(param, 0.0, 0.02)


        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.dem_map = {
            "dsm": self.dsm_param,
            "dtm": self.dtm_param,
            "slope": self.slope_param
        }

    def forward(self, polarizations: list, device=None) -> torch.Tensor:
        """
        Args:
            polarizations: List of polarization strings ['vv', 'vh', 'hh', 'hv'] of length [C].
        Returns:
            pos_emb: [C, hidden_dim]
        """
        embeddings = []
        device = device or self.dsm_param.device 

        try:
            embeddings = torch.stack(
                [self.dem_map[pol.lower()].to(device) for pol in polarizations],
                dim=0
            )
        except KeyError as e:
            raise ValueError(f"Unknown model: {e.args[0]}. "
                             f"Valid options: {list(self.dem_map.keys())}")

        return self.mlp(embeddings)                          



class PositionalEncoder(nn.Module):
    def __init__(self, d, T=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )
        self.updated_location = False

    def forward(self, batch_positions):
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table


class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=[256, 128],
        dropout=0.2,
        mlp_in = [10, 32, 128],
        T=1000,
        in_norm=True,
        return_att=False,
        return_att_full=False,
        positional_encoding=True,
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.return_att_full = return_att_full
        self.n_head = n_head

        if len(mlp_in) > 0:
            self.d_model = mlp_in[-1]
            mlp_in.insert(0, self.in_channels)
            layers = []
            for i in range(len(mlp_in) - 1):
                layers.extend(
                    [
                        nn.Linear(mlp_in[i], mlp_in[i + 1]),
                        nn.BatchNorm1d(mlp_in[i + 1]),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )

            self.inconv  = nn.Sequential(*layers)
        else:
            self.d_model = in_channels
            self.inconv = None

        self.mlp.insert(0, self.d_model)

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        
        if in_norm:
            self.in_norm = nn.GroupNorm(
                num_groups=n_head,
                num_channels=mlp_in[-1] if len(mlp_in) > 0 else in_channels,
            )
        else:
            self.in_norm = nn.Identity()
        
        #self.out_norm = nn.GroupNorm(
        #    num_groups=n_head,
        #    num_channels=mlp[-1],
        #)

        self.out_norm = None
        
        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)

        if self.inconv is not None:
            out_reshape = out.view(-1, out.size(-1))
            out_reshape = self.inconv(out_reshape)
            out = out_reshape.view(out.shape[0], out.shape[1], -1)

        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = (
            out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        )  # Concatenate heads
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )  # head x b x t x h x w

        if self.return_att:
            return out, torch.mean(attn, dim=(0, 3, 4), keepdim=True).squeeze()
        elif self.return_att_full:
            return out, attn
        else:
            return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(
            -1, d_k
        )  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        # compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn
