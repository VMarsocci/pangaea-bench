from collections import OrderedDict
from logging import Logger
from typing import Sequence

import torch
import torch.nn as nn

from pangaea.encoders.base import Encoder


class UNet(Encoder):
    """
    UNet Encoder for Supervised Baseline, to be trained from scratch.
    It supports single time frame inputs with optical bands

    Args:
        input_bands (dict[str, list[str]]): Band names, specifically expecting the 'optical' key with a list of bands.
        input_size (int): Size of the input images (height and width).
        topology (Sequence[int]): The number of feature channels at each stage of the U-Net encoder.

    """

    def __init__(
        self,
        input_bands: dict[str, list[str]],
        input_size: int,
        topology: Sequence[int],
        output_dim: int | list[int],
        download_url: str,
        encoder_weights: str | None = None,
    ):
        super().__init__(
            model_name="unet_encoder",
            encoder_weights=encoder_weights,  # no pre-trained weights, train from scratch
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=0,
            output_dim=output_dim,
            output_layers=None,
            multi_temporal=False,  # single time frame
            multi_temporal_output=False,
            pyramid_output=True,
            download_url=download_url,
        )

        if "optical" in input_bands and "sar" in input_bands:
            self.in_channels = len(input_bands["optical"]) + len(input_bands["sar"])
        elif "optical" in input_bands:
            self.in_channels = len(input_bands["optical"])
        elif "sar" in input_bands:
            self.in_channels = len(input_bands["sar"])
        else:
            raise ValueError("No valid bands found in the input_bands")
        self.topology = topology

        self.in_conv = InConv(self.in_channels, self.topology[0], DoubleConv)
        self.encoder = UNet_Encoder(self.topology)

    def forward(self, image):
        
        if "optical" in image and "sar" in image:
            # Concatenate optical and sar bands
            x = torch.cat([image["optical"].squeeze(2), image["sar"].squeeze(2)], dim=1)
        elif "optical" in image:
            x = image["optical"].squeeze(2)
        elif "sar" in image:
            x = image["sar"].squeeze(2)
        else:
            raise ValueError("No valid bands found in the image")
        
        feat = self.in_conv(x)
        output = self.encoder(feat)
        return output

    def load_encoder_weights(self, logger: Logger) -> None:
        pass


class UNetMT(Encoder):
    """
    Multi Temporal UNet Encoder for Supervised Baseline, to be trained from scratch.
    It supports single time frame inputs with optical bands

    Args:
        input_bands (dict[str, list[str]]): Band names, specifically expecting the 'optical' key with a list of bands.
        input_size (int): Size of the input images (height and width).
        topology (Sequence[int]): The number of feature channels at each stage of the U-Net encoder.

    """

    def __init__(
        self,
        input_bands: dict[str, list[str]],
        input_size: int,
        multi_temporal: int,
        topology: Sequence[int],
        output_dim: int | list[int],
        download_url: str,
        encoder_weights: str | None = None,
    ):
        super().__init__(
            model_name="unet_encoder",
            encoder_weights=encoder_weights,  # no pre-trained weights, train from scratch
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=0,
            output_dim=output_dim,
            output_layers=None,
            multi_temporal=multi_temporal,
            multi_temporal_output=False,
            pyramid_output=True,
            download_url=download_url,
        )

        if "optical" in input_bands and "sar" in input_bands:
            self.in_channels = len(input_bands["optical"]) + len(input_bands["sar"])
        elif "optical" in input_bands:
            self.in_channels = len(input_bands["optical"])
        elif "sar" in input_bands:
            self.in_channels = len(input_bands["sar"])
        else:
            raise ValueError("No valid bands found in the input_bands")
            
        self.topology = topology

        self.in_conv = InConv(self.in_channels, self.topology[0], DoubleConv)
        self.encoder = UNet_Encoder(self.topology)

        self.time_merging = DoubleConv(
            in_ch=self.in_channels * self.multi_temporal, out_ch=self.in_channels
        )

    def forward(self, image):
        if "optical" in image and "sar" in image:
            # Concatenate optical and sar bands
            x = torch.cat([image["optical"].squeeze(2), image["sar"].squeeze(2)], dim=1)
        elif "optical" in image:
            x = image["optical"].squeeze(2)
        elif "sar" in image:
            x = image["sar"].squeeze(2)
        else:
            raise ValueError("No valid bands found in the image")
            
        b, c, t, h, w = x.shape
        # merge time and channels dimension
        x = x.reshape(b, c * t, h, w)
        x = self.time_merging(x)

        feat = self.in_conv(x)
        output = self.encoder(feat)
        return output

    def load_encoder_weights(self, logger: Logger) -> None:
        pass


class UNet_Encoder(nn.Module):
    """
    UNet Encoder class that defines the architecture of the encoder part of the UNet.

    Args:
        topology (Sequence[int]): A sequence of integers defining the number of channels
                                  at each layer of the encoder.
    """

    def __init__(self, topology: Sequence[int]):
        super(UNet_Encoder, self).__init__()

        self.topology = topology

        # Variable scale
        down_topo = topology
        down_dict = OrderedDict()
        n_layers = len(down_topo)

        # Downward layers
        for idx in range(n_layers):
            is_not_last_layer = idx != n_layers - 1
            in_dim = down_topo[idx]
            out_dim = (
                down_topo[idx + 1] if is_not_last_layer else down_topo[idx]
            )  # last layer
            layer = Down(in_dim, out_dim, DoubleConv)
            down_dict[f"down{idx + 1}"] = layer
        self.down_seq = nn.ModuleDict(down_dict)

    def forward(self, x1: torch.Tensor) -> list:
        inputs = [x1]
        for layer in self.down_seq.values():
            out = layer(inputs[-1])
            inputs.append(out)

        inputs.reverse()
        return inputs


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(InConv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(Down, self).__init__()

        self.mpconv = nn.Sequential(nn.MaxPool2d(2), conv_block(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x
