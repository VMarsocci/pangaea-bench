from typing import cast

import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

from pangaea.decoders.base import Decoder
from pangaea.decoders.det_necks import BackboneWrapper
from pangaea.encoders.base import Encoder


class DetFasterRCNN(Decoder):
    """Faster R-CNN (torchvision) on top of a pangaea encoder + Feature2Pyramid + FPN.

    Loss is owned by torchvision (RPN cls/reg + RoI cls/reg); pangaea passes criterion=none.
    finetune=False freezes the encoder; necks/RPN/RoI stay trainable.
    """

    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
        fpn_out_channels: int,
        anchor_sizes: list[list[int]],
        anchor_aspect_ratios: list[float],
        roi_output_size: int,
        score_threshold: float,
        nms_threshold: float,
        detections_per_image: int,
    ) -> None:
        super().__init__(encoder=encoder, num_classes=num_classes, finetune=finetune)

        self.model_name = "Det_FasterRCNN"
        if not self.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False

        modality = list(encoder.input_bands.keys())[0]
        self.modality = modality
        self.backbone = BackboneWrapper(
            encoder=encoder, modality=modality, fpn_out_channels=fpn_out_channels
        )

        anchor_generator = AnchorGenerator(
            sizes=tuple(tuple(s) for s in anchor_sizes),
            aspect_ratios=(tuple(anchor_aspect_ratios),) * len(anchor_sizes),
        )
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=roi_output_size,
            sampling_ratio=2,
        )

        # image_mean/std as zeros/ones make torchvision's normalize a no-op; pangaea's
        # preprocessor already normalized and sized inputs to encoder.input_size.
        num_channels = len(encoder.input_bands[modality])
        self.detector = FasterRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=encoder.input_size,
            max_size=encoder.input_size,
            image_mean=[0.0] * num_channels,
            image_std=[1.0] * num_channels,
            box_score_thresh=score_threshold,
            box_nms_thresh=nms_threshold,
            box_detections_per_img=detections_per_image,
        )
        # Default size_divisible=32 pads batches up to a multiple of 32, which breaks
        # encoders with fixed-size positional embeddings (e.g., CROMA at 120).
        cast(GeneralizedRCNNTransform, self.detector.transform).size_divisible = 1

    def forward(
        self,
        img: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Train: returns torchvision's loss dict. Eval: list of per-image {boxes, labels, scores}."""
        feat = img[self.modality]
        if feat.shape[2] != 1:
            raise ValueError(
                f"detection is single-temporal scope; got T={feat.shape[2]} "
                f"for modality {self.modality!r}"
            )
        # torchvision input: list[Tensor(C,H,W)]; pangaea input: Tensor(B,C,T,H,W) with T=1
        return self.detector(list(feat[:, :, 0].unbind(0)), targets)
