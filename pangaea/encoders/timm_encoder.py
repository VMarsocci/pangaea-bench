from logging import Logger
from typing import Any, cast

import timm
import torch
from torch import nn

from pangaea.encoders.base import Encoder


class TimmEncoder(Encoder):
    """Wraps any timm backbone, CNN via ``features_only`` or ViT via ``forward_intermediates``."""

    def __init__(
        self,
        model_name: str,
        input_bands: dict[str, list[str]],
        input_size: int,
        output_layers: list[int],
        encoder_type: str,
    ):
        if encoder_type not in {"cnn", "vit"}:
            raise ValueError(
                f"encoder_type must be 'cnn' or 'vit', got {encoder_type!r}"
            )

        in_chans = sum(len(bands) for bands in input_bands.values())

        if encoder_type == "cnn":
            create_kwargs: dict[str, Any] = {
                "in_chans": in_chans,
                "features_only": True,
                "out_indices": tuple(output_layers),
            }
        else:
            create_kwargs = {
                "in_chans": in_chans,
                "num_classes": 0,
                "dynamic_img_size": True,
            }

        backbone: nn.Module = timm.create_model(
            model_name, pretrained=False, **cast(Any, create_kwargs)
        )

        embed_dim: int
        output_dim: int | list[int]
        if encoder_type == "cnn":
            channels = list(cast(Any, backbone).feature_info.channels())
            embed_dim = int(channels[-1])
            output_dim = list(channels)
            pyramid_output = True
        else:
            embed_dim = int(cast(Any, backbone).embed_dim)
            output_dim = embed_dim
            pyramid_output = False

        super().__init__(
            model_name=model_name,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=pyramid_output,
            encoder_weights="",
            download_url="",
        )

        self.encoder_type = encoder_type
        self.backbone = backbone
        self._create_kwargs = create_kwargs

    def forward(self, x: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        tensor = torch.cat([x[k].squeeze(2) for k in x.keys()], dim=1)
        if self.encoder_type == "cnn":
            return list(cast(Any, self.backbone)(tensor))
        intermediates = cast(Any, self.backbone).forward_intermediates(
            tensor,
            indices=self.output_layers,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        return list(intermediates)

    def load_encoder_weights(self, logger: Logger) -> None:
        # timm fuses architecture build + checkpoint resolution inside create_model;
        # we instantiate a temporary pretrained twin and copy its state_dict over.
        ref = timm.create_model(
            self.model_name, pretrained=True, **cast(Any, self._create_kwargs)
        )
        self.backbone.load_state_dict(ref.state_dict())
        del ref
