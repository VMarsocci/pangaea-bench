from logging import Logger

import torch
import numpy as np
import timm

from pangaea.encoders.base import Encoder


class TimmEncoder(Encoder):
    """Timm Encoder class."""

    def __init__(
        self,
        timm_model_name: str,
        input_bands: dict[str, list[str]],
        input_size: int,
    ) -> None:
        """Initialize the TimmEncoder.

        Args:
            model_name (str): Name of the model.
            input_bands (dict[str, list[str]]): Input bands.
            input_size (int): Input size.
            embed_dim (int): Embedding dimension.
            output_layers (list[int]): Output layers.
            logger (Logger, optional): Logger instance. Defaults to None.
        """

        super().__init__(
            model_name=timm_model_name,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=0,  # will be set after model is created
            output_layers=[],
            output_dim=0,  # will be set after model is created
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False, # will be set after model is created
            encoder_weights="",
            download_url="",
        )

        try:
            self.model = timm.create_model(
                timm_model_name,
                pretrained=True,
                features_only=True,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"Error loading Timm model {timm_model_name}. "
                "Please ensure that the timm_model_name parameter is correct and that you have internet access to download the weights."
            ) from e

        self.embed_dim = self.model.model.embed_dim
        self.output_dim = self.model.feature_info.channels()
        self.output_layers = list(range(len(self.output_dim)))
        self.pyramid_output = np.unique(self.output_dim).shape[0] > 1 # pyramid if multiple unique output dims

    def forward(self, image):
        x = torch.cat(
            [image[modality].squeeze(2) for modality in self.input_bands], dim=1
        )

        pred = self.model(x)
        return [v for v in pred]

    def load_encoder_weights(self, logger: Logger) -> None:
        pass
