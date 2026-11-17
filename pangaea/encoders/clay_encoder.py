from logging import Logger
from typing import cast

import torch
from claymodel.model import Encoder as ClayCoreEncoder
from torch import nn

from pangaea.encoders.base import Encoder

CLAY_V1_5_STATE_DICT_PREFIX = "model.encoder."

CLAY_V1_5_ARCH: dict[str, int] = {
    "dim": 1024,
    "depth": 24,
    "heads": 16,
    "dim_head": 64,
    "mlp_ratio": 4,
}


class ClayEncoder(Encoder):
    """Clay v1.5 encoder; iterates ``transformer.layers`` directly because Clay's ``Encoder.forward`` exposes only the final output, not intermediates."""

    def __init__(
        self,
        input_bands: dict[str, list[str]],
        input_size: int,
        output_layers: list[int],
        gsd: float,
        wavelengths: list[float],
        patch_size: int = 8,
        encoder_weights: str = "",
        download_url: str = "",
    ):
        n_bands = sum(len(b) for b in input_bands.values())
        if len(wavelengths) != n_bands:
            raise ValueError(
                f"len(wavelengths)={len(wavelengths)} must equal "
                f"total input bands={n_bands}"
            )

        dim = CLAY_V1_5_ARCH["dim"]
        super().__init__(
            model_name="clay-v1.5",
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=dim,
            output_layers=output_layers,
            output_dim=dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            encoder_weights=encoder_weights,
            download_url=download_url,
        )

        self.patch_size = patch_size
        self.gsd = float(gsd)
        self.register_buffer(
            "wavelengths",
            torch.tensor(wavelengths, dtype=torch.float32),
            persistent=False,
        )

        self.backbone = ClayCoreEncoder(
            mask_ratio=0.0,
            patch_size=patch_size,
            shuffle=False,
            **CLAY_V1_5_ARCH,
        )

    def load_encoder_weights(self, logger: Logger) -> None:
        ckpt = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)
        full_sd = ckpt.get("state_dict", ckpt)
        encoder_sd = {
            k[len(CLAY_V1_5_STATE_DICT_PREFIX) :]: v
            for k, v in full_sd.items()
            if k.startswith(CLAY_V1_5_STATE_DICT_PREFIX)
        }
        self.backbone.load_state_dict(encoder_sd, strict=True)

    def forward(self, x: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        tensor = torch.cat([x[k].squeeze(2) for k in x.keys()], dim=1)
        b, _, h, w = tensor.shape
        device = tensor.device

        # Zero time/latlon: Clay was pretrained with these as auxiliary conditioning,
        # TODO : if we wanna add temporal/geospatial to clay
        time = torch.zeros(b, 4, device=device)
        latlon = torch.zeros(b, 4, device=device)
        gsd_t = torch.tensor(self.gsd, device=device)
        waves = cast(torch.Tensor, self.wavelengths).to(device=device)

        patches, _ = self.backbone.to_patch_embed(tensor, waves)
        patches = self.backbone.add_encodings(patches, time, latlon, gsd_t)

        cls_tokens = self.backbone.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls_tokens, patches), dim=1)

        h_grid = h // self.patch_size
        w_grid = w // self.patch_size
        outputs: list[torch.Tensor] = []
        layers = self.backbone.transformer.layers
        for i, layer in enumerate(layers):
            pair = cast(nn.ModuleList, layer)
            attn, ff = pair[0], pair[1]
            tokens = attn(tokens) + tokens
            tokens = ff(tokens) + tokens
            if i in self.output_layers:
                feat = (
                    tokens[:, 1:]
                    .transpose(1, 2)
                    .reshape(b, -1, h_grid, w_grid)
                    .contiguous()
                )
                outputs.append(feat)
        return outputs
