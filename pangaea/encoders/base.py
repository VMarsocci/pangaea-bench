import os
import urllib.error
import urllib.request
from logging import Logger
from pathlib import Path

import gdown
import torch
import torch.nn as nn
import tqdm


class DownloadProgressBar:
    def __init__(self, text="Downloading..."):
        self.pbar = None
        self.text = text

    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm.tqdm(
                desc=self.text,
                total=total_size,
                unit="b",
                unit_scale=True,
                unit_divisor=1024,
            )

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded - self.pbar.n)
        else:
            self.pbar.close()
            self.pbar = None


class Encoder(nn.Module):
    """Base class for encoder."""

    def __init__(
        self,
        model_name: str,
        input_bands: dict[str, list[str]],
        input_size: int,
        embed_dim: int,
        output_dim: int,
        multi_temporal: bool,
        encoder_weights: str | Path,
        download_url: str,
    ) -> None:
        """Initialize the Encoder.

        Args:
            model_name (str): name of the model.
            input_bands (dict[str, list[str]]): list of the input bands for each modality.
            dictionary with keys as the modality and values as the list of bands.
            input_size (int): size of the input image.
            embed_dim (int): dimension of the embedding used by the encoder.
            output_dim (int): dimension of the embedding output by the encoder, accepted by the decoder.
            multi_temporal (bool): whether the model is multi-temporal or not.
            encoder_weights (str | Path): path to the encoder weights.
            download_url (str): url to download the model.
        """
        super().__init__()
        self.model_name = model_name
        self.input_bands = input_bands
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.encoder_weights = encoder_weights
        self.multi_temporal = multi_temporal
        self.download_url = download_url

        # download_model if necessary
        self.download_model()

    def load_encoder_weights(self, logger: Logger) -> None:
        """Load the encoder weights.

        Args:
            logger (Logger): logger to log the information.

        Raises:
            NotImplementedError: raise if the method is not implemented.
        """
        raise NotImplementedError

    def parameters_warning(
        self,
        missing: dict[str, torch.Size],
        incompatible_shape: dict[str, tuple[torch.Size, torch.Size]],
        logger: Logger,
    ) -> None:
        """Print warning messages for missing or incompatible parameters

        Args:
            missing (dict[str, torch.Size]): list of missing parameters.
            incompatible_shape (dict[str, tuple[torch.Size, torch.Size]]): list of incompatible parameters.
            logger (Logger): logger to log the information.
        """
        if missing:
            logger.warning(
                "Missing parameters:\n"
                + "\n".join("%s: %s" % (k, v) for k, v in sorted(missing.items()))
            )
        if incompatible_shape:
            logger.warning(
                "Incompatible parameters:\n"
                + "\n".join(
                    "%s: expected %s but found %s" % (k, v[0], v[1])
                    for k, v in sorted(incompatible_shape.items())
                )
            )

    def freeze(self) -> None:
        """Freeze encoder's parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the encoder.

        Args:
            x (torch.Tensor): input image.

        Raises:
            NotImplementedError: raise if the method is not implemented.

        Returns:
            torch.Tensor: embedding generated by the encoder.
        """
        raise NotImplementedError

    def download_model(self) -> None:
        if self.download_url and not os.path.isfile(self.encoder_weights):
            # TODO: change this path
            os.makedirs("pretrained_models", exist_ok=True)

            pbar = DownloadProgressBar(f"Downloading {self.encoder_weights}")

            if self.download_url.startswith("https://drive.google.com/"):
                gdown.download(self.download_url, self.encoder_weights)
            else:
                try:
                    urllib.request.urlretrieve(
                        self.download_url,
                        self.encoder_weights,
                        pbar,
                    )
                except urllib.error.HTTPError as e:
                    print(
                        "Error while downloading model: The server couldn't fulfill the request."
                    )
                    print("Error code: ", e.code)
                except urllib.error.URLError as e:
                    print("Error while downloading model: Failed to reach a server.")
                    print("Reason: ", e.reason)
