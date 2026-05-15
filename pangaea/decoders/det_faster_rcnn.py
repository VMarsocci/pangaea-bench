import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from pangaea.decoders.base import Decoder
from pangaea.decoders.det_necks import BackboneWrapper, IdentityRCNNTransform
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

        self.detector = FasterRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=encoder.input_size,
            max_size=encoder.input_size,
            box_score_thresh=score_threshold,
            box_nms_thresh=nms_threshold,
            box_detections_per_img=detections_per_image,
        )
        self.detector.transform = IdentityRCNNTransform()

    def forward(
        self,
        img: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Train: returns torchvision's loss dict. Eval: list of per-image {boxes, labels, scores}."""
        feat = img[self.modality]
        # torchvision wants list[Tensor(C,H,W)]; BackboneWrapper re-introduces T=1 before the encoder.
        if feat.dim() == 5:
            if feat.shape[2] != 1:
                raise ValueError(
                    f"detection is single-temporal scope; got T={feat.shape[2]} "
                    f"for modality {self.modality!r}"
                )
            feat = feat[:, :, 0, :, :]
        return self.detector(list(feat.unbind(0)), targets)
