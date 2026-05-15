from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from pangaea.decoders.upernet import Feature2Pyramid
from pangaea.encoders.base import Encoder


class FPNNeck(nn.Module):
    """torchvision FeaturePyramidNetwork wrapper emitting 4 FPN levels plus a `pool` extra level."""

    def __init__(self, in_channels_list: list[int], out_channels: int) -> None:
        super().__init__()
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, feats: list[torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        return self.fpn(OrderedDict((str(i), f) for i, f in enumerate(feats)))


class BackboneWrapper(nn.Module):
    """Pangaea Encoder -> Feature2Pyramid -> FPN, exposed as a torchvision FasterRCNN backbone.

    UNet is not supported (forward returns one extra level and inverted channel order),
    matching the SegUPerNet constraint. Adapted from terratorch's BackboneWrapper
    (terratorch.models.object_detection_model_factory:222-256).
    """

    def __init__(
        self,
        encoder: Encoder,
        modality: str,
        fpn_out_channels: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.modality = modality

        in_channels_list = list(encoder.output_dim)
        if len(in_channels_list) != 4:
            raise ValueError(
                f"DetFasterRCNN requires encoders that emit 4 feature maps "
                f"(MultiScaleRoIAlign + anchor config are 4+pool); "
                f"{type(encoder).__name__} emits {len(in_channels_list)}"
            )
        rescales = [1.0, 1.0, 1.0, 1.0] if encoder.pyramid_output else [4, 2, 1, 0.5]
        self.feature_pyramid = Feature2Pyramid(
            embed_dim=in_channels_list, rescales=rescales
        )
        self.fpn = FPNNeck(
            in_channels_list=in_channels_list, out_channels=fpn_out_channels
        )
        self.out_channels = fpn_out_channels

    def forward(self, image: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        # Pangaea encoders are written for (B, C, T, H, W); torchvision hands us (B, C, H, W).
        if image.dim() == 4:
            image = image.unsqueeze(2)
        feats = self.encoder({self.modality: image})
        feats = self.feature_pyramid(feats)
        return self.fpn(list(feats))


class IdentityRCNNTransform(GeneralizedRCNNTransform):
    """GeneralizedRCNNTransform with normalize/resize/postprocess overridden to identity.

    Pangaea's preprocessor already normalizes and sizes inputs to the encoder, so the
    parent's ImageNet-style normalize and min/max resize would corrupt them. We keep the
    parent only for batching list[Tensor] into an ImageList. Adapted from terratorch's
    TerratorchGeneralizedRCNNTransform (terratorch.models.utils:346-398).
    """

    def __init__(self) -> None:
        # size_divisible=1 is load-bearing: the default 32 pads batches up to a multiple of
        # 32, which breaks encoders with fixed-size positional embeddings (e.g., CROMA at 120).
        super().__init__(
            min_size=1,
            max_size=10_000,
            image_mean=[0.0],
            image_std=[1.0],
            size_divisible=1,
        )

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def resize(
        self,
        image: torch.Tensor,
        target: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        return image, target

    def postprocess(self, result, image_shapes, original_image_sizes):
        return result
