from logging import Logger
from pathlib import Path

import torch

from pangaea.encoders.base import Encoder


class AnySat_Encoder(Encoder):
    def __init__(
        self,
        encoder_weights: str | Path,
        input_bands: dict[str, list[str]],
        input_size: int,
        output_layers: int | list[int],
        output_dim: int | list[int],
        download_url: str | None,
        embed_dim: int = 768,
        model_size: str = "base",
        flash_attn: bool = False,
        patch_size: int = 10,
        output: str = "patch",
        output_modality: str = "auto",
    ) -> None:
        super().__init__(
            model_name=f"AnySat-{model_size}",
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

        self.patch_size = patch_size
        self.output = output
        self.output_modality = output_modality

        self.model = torch.hub.load(
            "gastruc/anysat",
            "anysat",
            pretrained=False,
            model_size=model_size,
            flash_attn=flash_attn,
        )

    @staticmethod
    def _to_anysat_timeseries(x: torch.Tensor, modality_name: str) -> torch.Tensor:
        if x.ndim == 4:
            return x.unsqueeze(1)
        if x.ndim == 5:
            return x.permute(0, 2, 1, 3, 4).contiguous()
        raise ValueError(f"Unexpected {modality_name} input shape: {tuple(x.shape)}")

    def load_encoder_weights(self, logger: Logger) -> None:
        if not self.encoder_weights:
            logger.warning("No AnySat checkpoint specified; using random initialization.")
            return

        checkpoint = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        msg = self.model.model.load_state_dict(state_dict, strict=False)

        if msg.missing_keys:
            logger.warning("Missing parameters: %s", msg.missing_keys)
        if msg.unexpected_keys:
            logger.warning("Unexpected parameters: %s", msg.unexpected_keys)

    def forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        data = {}
        target_modality = self.output_modality

        if "optical" in image:
            s2 = self._to_anysat_timeseries(image["optical"], "optical")
            b, t, _, _, _ = s2.shape
            data["s2"] = s2
            data["s2_dates"] = torch.zeros((b, t), device=s2.device, dtype=torch.long)
            if target_modality == "auto":
                target_modality = "s2"

        if "sar" in image:
            s1 = self._to_anysat_timeseries(image["sar"], "sar")
            b, t, c, _, _ = s1.shape
            sar_key = "s1" if c == 3 else "s1-asc"
            data[sar_key] = s1
            data[f"{sar_key}_dates"] = torch.zeros((b, t), device=s1.device, dtype=torch.long)
            if target_modality == "auto" and "optical" not in image:
                target_modality = sar_key

        if not data:
            raise ValueError("AnySat_Encoder expects at least one modality: 'optical' or 'sar'.")

        features = self.model(
            data,
            patch_size=self.patch_size,
            output=self.output,
            output_modality=target_modality,
        )

        if features.ndim == 2:
            features = features.unsqueeze(-1).unsqueeze(-1)
        elif features.ndim == 3:
            if features.shape[1] == self.embed_dim:
                n = features.shape[2]
                side = int(n**0.5)
                features = features.view(features.shape[0], features.shape[1], side, side)
            else:
                n = features.shape[1]
                side = int(n**0.5)
                features = features.permute(0, 2, 1).contiguous().view(
                    features.shape[0], features.shape[2], side, side
                )

        return [features for _ in self.output_layers]
