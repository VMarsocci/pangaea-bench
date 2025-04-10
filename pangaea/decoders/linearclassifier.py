import torch
import torch.nn as nn
import torch.nn.functional as F

from pangaea.decoders.base import Decoder
from pangaea.decoders.ltae import LTAE2d, LTAEChannelAdaptor
from pangaea.encoders.base import Encoder


class LinearClassifier(Decoder):

    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
        feature_multiplier: int = 1,
        in_channels: list[int] | None = None,
    ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            finetune=finetune,
        )

        self.model_name = "LinearClassifier"
        self.encoder = encoder
        self.finetune = finetune
        self.feature_multiplier = feature_multiplier

        if not self.finetune:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.input_layers = self.encoder.output_layers
        self.input_layers_num = len(self.input_layers)

        if in_channels is None:
            self.in_channels = [
                dim * feature_multiplier for dim in self.encoder.output_dim
            ]
        else:
            self.in_channels = [dim * feature_multiplier for dim in in_channels]

        self.in_channels = sum(self.in_channels)
        # self.in_channels = self.in_channels[-1]
        

        self.num_classes = num_classes
        self.linear_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),   
            nn.Flatten(),
            nn.Linear(self.in_channels, self.num_classes))
        
        # self.linear_head = nn.Linear(self.in_channels, num_classes)


    
    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        """Compute the segmentation output.

        Args:
            img (dict[str, torch.Tensor]): input data structured as a dictionary:
            img = {modality1: tensor1, modality2: tensor2, ...}, e.g. img = {"optical": tensor1, "sar": tensor2}.
            with tensor1 and tensor2 of shape (B C T=1 H W) with C the number of encoders'bands for the given modality.
            output_shape (torch.Size | None, optional): output's spatial dims (H, W) (equals to the target spatial dims).
            Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape (B, num_classes, H', W') with (H' W') coressponding to the output_shape.
        """

        # img[modality] of shape [B C T=1 H W]
        if self.encoder.multi_temporal:
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder(img)
            else:
                feat = self.encoder(img)

            # multi_temporal models can return either (B C' T=1 H' W')
            # or (B C' H' W'), we need (B C' H' W')
            if self.encoder.multi_temporal_output:
                feat = [f.squeeze(-3) for f in feat]

        else:
            # remove the temporal dim
            # [B C T=1 H W] -> [B C H W]
            if not self.finetune:
                with torch.no_grad():
                    feat = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})
            else:
                feat = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})

        shapes = torch.tensor([f.shape[2:] for f in feat])  # Extract H, W for each tensor
        max_h, max_w = shapes.max(dim=0).values
        max_h = max_h.item()
        max_w = max_w.item()
        resized_feats = []
        for f in feat:
            if f.shape[2:] != (max_h, max_w):
                resized_feats.append(F.interpolate(f, size=(max_h, max_w), mode='bilinear', align_corners=False))
            else:
                resized_feats.append(f)
        
        final_feat = torch.cat(resized_feats, dim=1)
        # final_feat = resized_feats[-1]
        
        logits = self.linear_head(final_feat)

        return logits