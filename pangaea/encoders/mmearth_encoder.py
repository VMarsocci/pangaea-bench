import math
from collections import OrderedDict
from logging import Logger
from pathlib import Path

import torch

from pangaea.encoders.base import Encoder


class MMEarthConvNeXtV2Encoder(Encoder):
    """MMEarth ConvNeXtV2 encoder wrapper for Pangaea benchmark."""

    def __init__(
        self,
        encoder_weights: str | Path,
        input_size: int,
        input_bands: dict[str, list[str]],
        output_layers: int | list[int],
        output_dim: int | list[int],
        download_url: str | None,
        model_name: str = "convnextv2_atto",
        patch_size: int = 8,
        use_orig_stem: bool = False,
        num_classes: int = 1000,
        embed_dim: int = 320,
        ckpt_name: str = "pt-all_mod_atto_1M_64_uncertainty_56-8",
    ) -> None:
        super().__init__(
            model_name=f"MMEarth-{model_name}",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=True,
            download_url=download_url,
        )

        self.mmearth_model_name = model_name
        self.patch_size = patch_size
        self.use_orig_stem = use_orig_stem
        self.num_classes = num_classes
        self.ckpt_name = ckpt_name

        self.model = torch.hub.load(
            "vishalned/MMEarth-train",
            "MPMAE",
            model_name=self.mmearth_model_name,
            ckpt_name=self.ckpt_name,
            pretrained=False,
            linear_probe=True,
            in_chans=len(input_bands["optical"]),
            patch_size=self.patch_size,
            img_size=self.input_size,
            num_classes=self.num_classes,
            use_orig_stem=self.use_orig_stem,
        )

    @staticmethod
    def _remap_checkpoint_keys(
        checkpoint: dict[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        remapped = OrderedDict()
        for key, value in checkpoint.items():
            if key.startswith("encoder"):
                key = ".".join(key.split(".")[1:])

            if key.endswith("kernel"):
                key = ".".join(key.split(".")[:-1])
                new_key = key + ".weight"
                if len(value.shape) == 3:
                    kv, in_dim, out_dim = value.shape
                    ks = int(math.sqrt(kv))
                    remapped[new_key] = (
                        value.permute(2, 1, 0)
                        .reshape(out_dim, in_dim, ks, ks)
                        .transpose(3, 2)
                    )
                elif len(value.shape) == 2:
                    kv, dim = value.shape
                    ks = int(math.sqrt(kv))
                    remapped[new_key] = (
                        value.permute(1, 0).reshape(dim, 1, ks, ks).transpose(3, 2)
                    )
                continue
            elif "ln" in key or "linear" in key:
                split_key = key.split(".")
                split_key.pop(-2)
                new_key = ".".join(split_key)
            else:
                new_key = key

            remapped[new_key] = value

        for key, value in list(remapped.items()):
            if key.endswith("bias") and len(value.shape) != 1:
                remapped[key] = value.reshape(-1)
            elif "grn" in key:
                remapped[key] = value.unsqueeze(0).unsqueeze(1)

        return remapped

    def load_encoder_weights(self, logger: Logger) -> None:
        if not self.encoder_weights:
            logger.warning("No MMEarth checkpoint specified; using random initialization.")
            return

        checkpoint = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)
        checkpoint_model = checkpoint["model"] if "model" in checkpoint else checkpoint

        filtered_checkpoint = {}
        for key, value in checkpoint_model.items():
            if "decoder" in key or "mask_token" in key or "proj" in key or "pred" in key:
                continue
            filtered_checkpoint[key] = value

        remapped = self._remap_checkpoint_keys(filtered_checkpoint)

        current_state = self.model.state_dict()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}

        for name, param in current_state.items():
            if name not in remapped:
                missing[name] = param.shape
            elif remapped[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, remapped[name].shape)
            else:
                pretrained_encoder[name] = remapped[name]

        self.model.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)

    def forward(self, images: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        x = images["optical"].squeeze(2).float()

        stage_outputs = []
        if self.model.use_orig_stem:
            x = self.model.stem_orig(x)
        else:
            x = self.model.initial_conv(x)
            x = self.model.stem(x)

        x = self.model.stages[0](x)
        stage_outputs.append(x)

        for i in range(3):
            x = self.model.downsample_layers[i](x)
            x = self.model.stages[i + 1](x)
            stage_outputs.append(x)

        return [stage_outputs[i] for i in self.output_layers]
