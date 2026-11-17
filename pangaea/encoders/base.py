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
    """Download progress bar."""

    def __init__(self, text: str = "Downloading...") -> None:
        """Initialize the DownloadProgressBar.

        Args:
            text (str, optional): pbar text. Defaults to "Downloading...".
        """
        self.pbar = None
        self.text = text

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        """Update the progress bar.

        Args:
            block_num (int): number of blocks.
            block_size (int): size of the blocks.
            total_size (int): total size of the download.
        """

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
        output_layers: list[int],
        output_dim: int | list[int],
        multi_temporal: bool,
        multi_temporal_output: bool,
        pyramid_output: bool,
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
            multi_temporal (bool): whether the model output is multi-temporal or not.
            encoder_weights (str | Path): path to the encoder weights.
            download_url (str): url to download the model.
        """
        super().__init__()
        self.model_name = model_name
        self.input_bands = input_bands
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.output_layers = output_layers
        self.output_dim = (
            [output_dim for _ in output_layers]
            if isinstance(output_dim, int)
            else list(output_dim)
        )
        self.encoder_weights = encoder_weights
        self.multi_temporal = multi_temporal
        self.multi_temporal_output = multi_temporal_output

        self.pyramid_output = pyramid_output
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

    def enforce_single_temporal(self):
        return
        # self.multi_temporal = False
        # self.multi_temporal_fusion = False

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

    def forward(self, x: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Foward pass of the encoder.

        Args:
            x (dict[str, torch.Tensor]): encoder's input structured as a dictionary:
            x = {modality1: tensor1, modality2: tensor2, ...}, e.g. x = {"optical": tensor1, "sar": tensor2}.
            If the encoder is multi-temporal (self.multi_temporal==True), input tensor shape is (B C T H W) with C the
            number of bands required by the encoder for the given modality and T the number of time steps. If the
            encoder is not multi-temporal, input tensor shape is (B C H W) with C the number of bands required by the
            encoder for the given modality.
        Raises:
            NotImplementedError: raise if the method is not implemented.

        Returns:
            list[torch.Tensor]: list of the embeddings for each modality. For single-temporal encoders, the list's
            elements are of shape (B, embed_dim, H', W'). For multi-temporal encoders, the list's elements are of shape
            (B, C', T, H', W') with T the number of time steps if the encoder does not have any time-merging strategy,
            else (B, C', H', W') if the encoder has a time-merging strategy (where C'==self.output_dim).
        """
        raise NotImplementedError

    def download_model(self) -> None:
        """Download the model if the weights are not already downloaded."""
        if self.download_url and not os.path.isfile(self.encoder_weights):
            os.makedirs(
                os.path.dirname(self.encoder_weights) or ".", exist_ok=True
            )

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
