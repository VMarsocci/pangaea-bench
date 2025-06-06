# Adapted from: https://github.com/allenai/satlaspretrain_models/

import collections
from enum import Enum, auto
from io import BytesIO
from logging import Logger
from pathlib import Path

import requests
import torch
import torch.nn
import torchvision

from pangaea.encoders.base import Encoder


class Backbone(Enum):
    SWINB = auto()
    SWINT = auto()
    RESNET50 = auto()
    RESNET152 = auto()


SatlasPretrain_weights = {
    "Sentinel2_SwinB_SI_RGB": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_si_rgb.pth?download=true",
        "backbone": Backbone.SWINB,
        "num_channels": 3,
        "multi_image": False,
    },
    "Sentinel2_SwinB_MI_RGB": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_mi_rgb.pth?download=true",
        "backbone": Backbone.SWINB,
        "num_channels": 3,
        "multi_image": True,
    },
    "Sentinel2_SwinB_SI_MS": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_si_ms.pth?download=true",
        "backbone": Backbone.SWINB,
        "num_channels": 9,
        "multi_image": False,
    },
    "Sentinel2_SwinB_MI_MS": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_mi_ms.pth?download=true",
        "backbone": Backbone.SWINB,
        "num_channels": 9,
        "multi_image": True,
    },
    "Sentinel1_SwinB_SI": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel1_swinb_si.pth?download=true",
        "backbone": Backbone.SWINB,
        "num_channels": 2,
        "multi_image": False,
    },
    "Sentinel1_SwinB_MI": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel1_swinb_mi.pth?download=true",
        "backbone": Backbone.SWINB,
        "num_channels": 2,
        "multi_image": True,
    },
    "Landsat_SwinB_SI": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/landsat_swinb_si.pth?download=true",
        "backbone": Backbone.SWINB,
        "num_channels": 11,
        "multi_image": False,
    },
    "Landsat_SwinB_MI": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/landsat_swinb_mi.pth?download=true",
        "backbone": Backbone.SWINB,
        "num_channels": 11,
        "multi_image": True,
    },
    "Aerial_SwinB_SI": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/aerial_swinb_si.pth?download=true",
        "backbone": Backbone.SWINB,
        "num_channels": 3,
        "multi_image": False,
    },
    "Aerial_SwinB_MI": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/aerial_swinb_mi.pth?download=true",
        "backbone": Backbone.SWINB,
        "num_channels": 3,
        "multi_image": True,
    },
    "Sentinel2_SwinT_SI_RGB": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swint_si_rgb.pth?download=true",
        "backbone": Backbone.SWINT,
        "num_channels": 3,
        "multi_image": False,
    },
    "Sentinel2_SwinT_SI_MS": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swint_si_ms.pth?download=true",
        "backbone": Backbone.SWINT,
        "num_channels": 9,
        "multi_image": False,
    },
    "Sentinel2_SwinT_MI_RGB": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swint_mi_rgb.pth?download=true",
        "backbone": Backbone.SWINT,
        "num_channels": 3,
        "multi_image": True,
    },
    "Sentinel2_SwinT_MI_MS": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swint_mi_ms.pth?download=true",
        "backbone": Backbone.SWINT,
        "num_channels": 9,
        "multi_image": True,
    },
    "Sentinel2_Resnet50_SI_RGB": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet50_si_rgb.pth?download=true",
        "backbone": Backbone.RESNET50,
        "num_channels": 3,
        "multi_image": False,
    },
    "Sentinel2_Resnet50_SI_MS": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet50_si_ms.pth?download=true",
        "backbone": Backbone.RESNET50,
        "num_channels": 9,
        "multi_image": False,
    },
    "Sentinel2_Resnet50_MI_RGB": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet50_mi_rgb.pth?download=true",
        "backbone": Backbone.RESNET50,
        "num_channels": 3,
        "multi_image": True,
    },
    "Sentinel2_Resnet50_MI_MS": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet50_mi_ms.pth?download=true",
        "backbone": Backbone.RESNET50,
        "num_channels": 9,
        "multi_image": True,
    },
    "Sentinel2_Resnet152_SI_RGB": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet152_si_rgb.pth?download=true",
        "backbone": Backbone.RESNET152,
        "num_channels": 3,
        "multi_image": False,
    },
    "Sentinel2_Resnet152_SI_MS": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet152_si_ms.pth?download=true",
        "backbone": Backbone.RESNET152,
        "num_channels": 9,
        "multi_image": False,
    },
    "Sentinel2_Resnet152_MI_RGB": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet152_mi_rgb.pth?download=true",
        "backbone": Backbone.RESNET152,
        "num_channels": 3,
        "multi_image": True,
    },
    "Sentinel2_Resnet152_MI_MS": {
        "url": "https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_resnet152_mi_ms.pth?download=true",
        "backbone": Backbone.RESNET152,
        "num_channels": 9,
        "multi_image": True,
    },
}


def adjust_state_dict_prefix(
    state_dict, needed, prefix=None, prefix_allowed_count=None
):
    """
    Adjusts the keys in the state dictionary by replacing 'backbone.backbone' prefix with 'backbone'.

    Args:
        state_dict (dict): Original state dictionary with 'backbone.backbone' prefixes.

    Returns:
        dict: Modified state dictionary with corrected prefixes.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Assure we're only keeping keys that we need for the current model component.
        if needed not in key:
            continue

        # Update the key prefixes to match what the model expects.
        if prefix is not None:
            while key.count(prefix) > prefix_allowed_count:
                key = key.replace(prefix, "", 1)

        new_state_dict[key] = value
    return new_state_dict


class FPN(torch.nn.Module):
    def __init__(self, backbone_channels):
        super(FPN, self).__init__()

        out_channels = 128
        in_channels_list = [ch[1] for ch in backbone_channels]
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=in_channels_list, out_channels=out_channels
        )

        self.out_channels = [[ch[0], out_channels] for ch in backbone_channels]

    def forward(self, x):
        inp = collections.OrderedDict(
            [("feat{}".format(i), el) for i, el in enumerate(x)]
        )
        output = self.fpn(inp)
        output = list(output.values())

        return output


class Upsample(torch.nn.Module):
    # Computes an output feature map at 1x the input resolution.
    # It just applies a series of transpose convolution layers on the
    # highest resolution features from the backbone (FPN should be applied first).

    def __init__(self, backbone_channels):
        super(Upsample, self).__init__()
        self.in_channels = backbone_channels

        out_channels = backbone_channels[0][1]
        self.out_channels = [(1, out_channels)] + backbone_channels

        layers = []
        depth, ch = backbone_channels[0]
        while depth > 1:
            next_ch = max(ch // 2, out_channels)
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(ch, ch, 3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.ConvTranspose2d(ch, next_ch, 4, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            layers.append(layer)
            ch = next_ch
            depth /= 2

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        output = self.layers(x[0])
        return [output] + x


class SwinBackbone(torch.nn.Module):
    def __init__(self, num_channels, arch):
        super(SwinBackbone, self).__init__()

        if arch == "swinb":
            self.backbone = torchvision.models.swin_v2_b()
            self.out_channels = [
                [4, 128],
                [8, 256],
                [16, 512],
                [32, 1024],
            ]
            self.embed_dim = 128

        elif arch == "swint":
            self.backbone = torchvision.models.swin_v2_t()
            self.out_channels = [
                [4, 96],
                [8, 192],
                [16, 384],
                [32, 768],
            ]

            self.embed_dim = 96
        else:
            raise ValueError("Backbone architecture not supported.")

        self.backbone.features[0][0] = torch.nn.Conv2d(
            num_channels,
            self.backbone.features[0][0].out_channels,
            kernel_size=(4, 4),
            stride=(4, 4),
        )

    def forward(self, x):
        outputs = []
        for layer in self.backbone.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]


class ResnetBackbone(torch.nn.Module):
    def __init__(self, num_channels, arch="resnet50"):
        super(ResnetBackbone, self).__init__()

        if arch == "resnet50":
            self.resnet = torchvision.models.resnet.resnet50(weights=None)
            ch = [256, 512, 1024, 2048]
        elif arch == "resnet152":
            self.resnet = torchvision.models.resnet.resnet152(weights=None)
            ch = [256, 512, 1024, 2048]
        else:
            raise ValueError("Backbone architecture not supported.")

        self.resnet.conv1 = torch.nn.Conv2d(
            num_channels,
            self.resnet.conv1.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.out_channels = [
            [4, ch[0]],
            [8, ch[1]],
            [16, ch[2]],
            [32, ch[3]],
        ]

    def train(self, mode=True):
        super(ResnetBackbone, self).train(mode)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        layer1 = self.resnet.layer1(x)
        layer2 = self.resnet.layer2(layer1)
        layer3 = self.resnet.layer3(layer2)
        layer4 = self.resnet.layer4(layer3)

        return [layer1, layer2, layer3, layer4]


class AggregationBackbone(torch.nn.Module):
    def __init__(self, num_channels, backbone):
        super(AggregationBackbone, self).__init__()

        # Number of channels to pass to underlying backbone.
        self.image_channels = num_channels

        # Prepare underlying backbone.
        self.backbone = backbone

        self.embed_dim = self.backbone.embed_dim

        # Features from images within each group are aggregated separately.
        # Then the output is the concatenation across groups.
        # e.g. [[0], [1, 2]] to compare first image against the others
        self.groups = [[0, 1, 2, 3, 4, 5, 6, 7]]

        ngroups = len(self.groups)
        self.out_channels = [
            (depth, ngroups * count) for (depth, count) in self.backbone.out_channels
        ]

        self.aggregation_op = "max"

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.reshape(B, C * T, H, W)
        # First get features of each image.
        all_features = []
        for i in range(0, x.shape[1], self.image_channels):
            features = self.backbone(x[:, i : i + self.image_channels, :, :])
            all_features.append(features)

        # Now compute aggregation over each group.
        # We handle each depth separately.
        l = []
        for feature_idx in range(len(all_features[0])):
            aggregated_features = []
            for group in self.groups:
                group_features = []
                for image_idx in group:
                    # We may input fewer than the maximum number of images.
                    # So here we skip image indices in the group that aren't available.
                    if image_idx >= len(all_features):
                        continue

                    group_features.append(all_features[image_idx][feature_idx])
                # Resulting group features are (depth, batch, C, height, width).
                group_features = torch.stack(group_features, dim=0)

                if self.aggregation_op == "max":
                    group_features = torch.amax(group_features, dim=0)

                aggregated_features.append(group_features)

            # Finally we concatenate across groups.
            aggregated_features = torch.cat(aggregated_features, dim=1)

            l.append(aggregated_features)

        return l


class SatlasNet_Encoder(Encoder):
    """
    Paper: https://arxiv.org/pdf/2211.15660
    Attributes:
        model_identifier (str): Identifier for the model, used to fetch pretrained weights.
        weights_url (str): URL to download the pretrained weights.
        in_chans (int): Number of input channels for the model.
        multi_image (bool): Indicates if the model is for multi-image input.
        backbone_arch (str): Architecture of the backbone model.
        out_dim (int): Output dimension of the encoder.
        backbone (nn.Module): Backbone model initialized based on the architecture.
        embed_dim (int): Embedding dimension of the backbone model.
        fpn (FPN, optional): Feature Pyramid Network, if enabled.
        upsample (Upsample, optional): Upsample module for the FPN, if enabled.
    Methods:
        __init__(self, input_bands: dict[str, list[str]], input_size: int, output_dim: int, model_identifier: str, fpn=False):
            Initializes the SatlasNet_Encoder with the given parameters.
        _initialize_backbone(self, num_channels, backbone_arch, multi_image):
            Initializes the backbone model based on the specified architecture.
        load_encoder_weights(self, logger: Logger) -> None:
            Downloads and loads pretrained weights for the encoder and optionally for the FPN.
        forward(self, imgs):
            Defines the forward pass of the encoder model.
    """

    def __init__(
        self,
        input_bands: dict[str, list[str]],
        input_size: int,
        output_dim: int | list[int],
        output_layers: int | list[int],
        model_identifier: str,
        encoder_weights: str | Path,
        download_url: str,
        fpn=False,
    ):
        """
        Initializes a model, based on desired imagery source and model components.
        """
        super().__init__(
            model_name="satlas_pretrain",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=768,  # will be overwritten by the backbone
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False if "_SI_" in model_identifier else True,
            multi_temporal_output=False,
            pyramid_output=True,
            download_url=download_url,
        )

        # Validate that the model identifier is supported.
        self.model_identifier = model_identifier
        if self.model_identifier not in SatlasPretrain_weights.keys():
            raise ValueError("Invalid model_identifier. See SatlasPretrain_weights.")

        model_info = SatlasPretrain_weights[self.model_identifier]
        self.weights_url = model_info["url"]
        self.in_chans = model_info["num_channels"]
        self.multi_image = model_info["multi_image"]
        self.backbone_arch = model_info["backbone"]


        self.backbone = self._initialize_backbone(
            self.in_chans, self.backbone_arch, self.multi_image
        )

        self.embed_dim = self.backbone.embed_dim

        # Whether or not to feed imagery through the pretrained Feature Pyramid Network after the backbone.
        if fpn:
            self.fpn = FPN(self.backbone.out_channels)
            self.upsample = Upsample(self.fpn.out_channels)
        else:
            self.fpn = None

    def _initialize_backbone(self, num_channels, backbone_arch, multi_image):
        # Load backbone model according to specified architecture.
        if backbone_arch == Backbone.SWINB or backbone_arch == "Backbone.SWINB":
            backbone = SwinBackbone(num_channels, arch="swinb")
        elif backbone_arch == Backbone.SWINT or backbone_arch == "Backbone.SWINT":
            backbone = SwinBackbone(num_channels, arch="swint")
        elif backbone_arch == Backbone.RESNET50 or backbone_arch == "Backbone.RESNET50":
            backbone = ResnetBackbone(num_channels, arch="resnet50")
        elif (
            backbone_arch == Backbone.RESNET152 or backbone_arch == "Backbone.RESNET152"
        ):
            backbone = ResnetBackbone(num_channels, arch="resnet152")
        else:
            raise ValueError("Unsupported backbone architecture.")

        # If using a model for multi-image, need the Aggretation to wrap underlying backbone model.
        if self.multi_image:
            backbone = AggregationBackbone(num_channels, backbone)

        return backbone

    def load_encoder_weights(self, logger: Logger) -> None:
        """
        Find and load pretrained SatlasPretrain weights, based on the model_identifier argument.
        Option to load pretrained FPN.
        """

        response = requests.get(self.weights_url)
        if response.status_code == 200:
            weights_file = BytesIO(response.content)
        else:
            raise Exception(f"Failed to download weights from {self.weights_url}")

        pretrained_model = torch.load(weights_file, map_location=torch.device("cpu"), weights_only=False)

        # If using a model for multi-image, need the Aggretation to wrap underlying backbone model.
        prefix, prefix_allowed_count = None, None
        if self.backbone_arch in ["Backbone.RESNET50", "Backbone.RESNET152"]:
            prefix_allowed_count = 0
        elif self.multi_image:
            prefix_allowed_count = 2
        else:
            prefix_allowed_count = 1

        state_dict_backbone = adjust_state_dict_prefix(
            pretrained_model, "backbone", "backbone.", prefix_allowed_count
        )

        missing = {}
        incompatible_shape = {}
        for name, param in self.backbone.named_parameters():
            if name not in state_dict_backbone:
                missing[name] = param.shape
            elif state_dict_backbone[name].shape != param.shape:
                incompatible_shape[name] = (
                    param.shape,
                    state_dict_backbone[name].shape,
                )

        self.backbone.load_state_dict(state_dict_backbone)

        if self.fpn:
            state_dict_fpn = adjust_state_dict_prefix(
                pretrained_model, "fpn", "intermediates.0.", 0
            )
            for name, param in self.fpn.named_parameters():
                if name not in state_dict_fpn:
                    missing[name] = param.shape
                elif state_dict_fpn[name].shape != param.shape:
                    incompatible_shape[name] = (param.shape, state_dict_fpn[name].shape)

            self.fpn.load_state_dict(state_dict_fpn)

        self.parameters_warning(missing, incompatible_shape, logger)

    def forward(self, imgs):
        # Define forward pass
        if not isinstance(self.backbone, AggregationBackbone):
            imgs["optical"] = imgs["optical"].squeeze(2)

        x = self.backbone(imgs["optical"])

        if self.fpn:
            x = self.fpn(x)
            x = self.upsample(x)

        output = []
        for i in range(len(x)):
            output.append(x[i])

        return output
