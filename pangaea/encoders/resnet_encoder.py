from collections import OrderedDict
from logging import Logger
from typing import Sequence

from omegaconf import OmegaConf
import torch
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import torch.nn as nn

from pangaea.encoders.base import Encoder


class ResNetEncoder(Encoder):
    """
    ResNet Encoder for Supervised Baseline, pretrained on ImageNet.
    It supports single time frame inputs with optical bands.

    Args:
        output_layers (str | list[str]): The layers from which to extract the output.
        input_bands (dict[str, list[str]]): Band names, specifically expecting the 'optical' key with a list of bands.
        depth (int): The depth of the ResNet encoder in [18, 34, 50, 101, 152].
    """

    def __init__(
            self,
            output_layers: str | list[str],
            input_bands: dict[str, list[str]], 
            input_size: int, 
            depth: int,
            encoder_weights: str | None = None, 
    ):
        super().__init__(
            model_name=f"Resnet-{depth}",
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=[64, 128, 256, 512,] if depth <= 34 else [256, 512, 1024, 2048,],
            output_dim=[64, 128, 256, 512,] if depth <= 34 else [256, 512, 1024, 2048,],
            output_layers=output_layers,
            multi_temporal=False,
            pyramid_output=True,
            multi_temporal_output=False,
            encoder_weights=encoder_weights,
            download_url="",
        )

        assert depth in [18, 34, 50, 101, 152], f"ResNet depth {depth} is not supported"

        if depth == 18:
            self.resnet = resnet18(weights=encoder_weights)
        elif depth == 34:
            self.resnet = resnet34(weights=encoder_weights)
        elif depth == 50:
            self.resnet = resnet50(weights=encoder_weights)
        elif depth == 101:
            self.resnet = resnet101(weights=encoder_weights)
        elif depth == 152:
            self.resnet = resnet152(weights=encoder_weights)

        # remove the classifier and avg pooling
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.backbone = create_feature_extractor(self.resnet, {k:k for k in output_layers})


    def forward(self, image):
        x = image["optical"].squeeze(2) # squeeze the time dimension
        return [v for v in self.backbone(x).values()]
    

    def load_encoder_weights(self, logger: Logger) -> None:
        pass

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

class ResNetEncoderSAR(Encoder):
    """
    ResNet Encoder for Supervised Baseline, pretrained on ImageNet.
    It supports single time frame inputs with optical bands.

    Args:
        output_layers (str | list[str]): The layers from which to extract the output.
        input_bands (dict[str, list[str]]): Band names, specifically expecting the 'optical' key with a list of bands.
        depth (int): The depth of the ResNet encoder in [18, 34, 50, 101, 152].
    """

    def __init__(
            self,
            output_layers: str | list[str],
            input_bands: dict[str, list[str]], 
            input_size: int, 
            depth: int,
            encoder_weights: str | None = None, 
    ):
        super().__init__(
            model_name=f"Resnet-{depth}",
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=[64, 128, 256, 512,] if depth <= 34 else [256, 512, 1024, 2048,],
            output_dim=[64, 128, 256, 512,] if depth <= 34 else [256, 512, 1024, 2048,],
            output_layers=output_layers,
            multi_temporal=False,
            pyramid_output=True,
            multi_temporal_output=False,
            encoder_weights=encoder_weights,
            download_url="",
        )

        assert depth in [18, 34, 50, 101, 152], f"ResNet depth {depth} is not supported"

        if depth == 18:
            self.resnet = resnet18(weights=encoder_weights)
        elif depth == 34:
            self.resnet = resnet34(weights=encoder_weights)
        elif depth == 50:
            self.resnet = resnet50(weights=encoder_weights)
        elif depth == 101:
            self.resnet = resnet101(weights=encoder_weights)
        elif depth == 152:
            self.resnet = resnet152(weights=encoder_weights)

        # remove the classifier and avg pooling
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.backbone = create_feature_extractor(self.resnet, {k:k for k in output_layers})


    def forward(self, image):
        x = image["sar"].squeeze(2) # squeeze the time dimension
        return [v for v in self.backbone(x).values()]
    

    def load_encoder_weights(self, logger: Logger) -> None:
        pass

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
