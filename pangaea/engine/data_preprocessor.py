import math
import numbers
import random
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from hydra.utils import instantiate


class BasePreprocessor:
    """Base class for preprocessor."""

    def __init__(
        self,
    ) -> None:
        return

    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        raise NotImplementedError

    def update_meta(self, meta):
        raise NotImplementedError

    def check_dimension(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]):
        """check dimension (C, T, H, W) of data"""
        for k, v in data["image"].items():
            if len(v.shape) != 4:
                raise AssertionError(
                    f"Image dimension must be 4 (C, T, H, W), Got {str(len(v.shape))}"
                )

        if len(data["target"].shape) not in (0, 1, 2):
            raise AssertionError(
                f"Target dimension must be 0 (for classification - single label) or 1 (for classification - multi-label) or 2 (for dense prediction), got {str(len(data['target'].shape))}"
            )

    def check_size(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]):
        """check if data size is equal"""
        base_shape = data["image"][list(data["image"].keys())[0]].shape

        for k, v in data["image"].items():
            if v.shape[1:] != base_shape[1:]:
                shape = {k: tuple(v.shape[1:]) for k, v in data["image"].items()}
                raise AssertionError(
                    f"Image size (T, H, W) from all modalities must be equal, Got {str(shape)}"
                )

        if base_shape[-2:] != data["target"].shape[-2:]:
            raise AssertionError(
                f"Image size and target size (H, W) must be equal, Got {str(tuple(base_shape[-2:]))} and {str(tuple(data['target'].shape[-2:]))}"
            )


class Preprocessor(BasePreprocessor):
    """A series of base preprocessors that preprocess images and targets."""

    def __init__(self, preprocessor_cfg, dataset_cfg, encoder_cfg) -> None:
        """Build preprocessors defined in preprocessor_cfg.
        Args:
            preprocessor_cfg: preprocessor config
            dataset_cfg: dataset config
            encoder_cfg: encoder config
        """
        super().__init__()
        # initialize the meta statistics/info of the input data and target encoder
        meta = {}
        meta["dataset_img_size"] = dataset_cfg["img_size"]
        meta["encoder_input_size"] = encoder_cfg["input_size"]
        meta["dataset_bands"] = dataset_cfg["bands"]
        meta["encoder_bands"] = encoder_cfg["input_bands"]
        meta["multi_modal"] = dataset_cfg["multi_modal"]
        meta["multi_temporal"] = dataset_cfg["multi_temporal"]

        meta["data_bands"] = dataset_cfg["bands"]
        meta["data_img_size"] = dataset_cfg["img_size"]
        meta["data_mean"] = {
            k: torch.tensor(v) for k, v in dataset_cfg["data_mean"].items()
        }
        meta["data_std"] = {
            k: torch.tensor(v) for k, v in dataset_cfg["data_std"].items()
        }
        meta["data_min"] = {
            k: torch.tensor(v) for k, v in dataset_cfg["data_min"].items()
        }
        meta["data_max"] = {
            k: torch.tensor(v) for k, v in dataset_cfg["data_max"].items()
        }

        meta["ignore_index"] = dataset_cfg["ignore_index"]
        meta["class_distribution"] = torch.tensor(dataset_cfg["distribution"])

        self.preprocessor = []

        # build the preprocessor and update the meta for the next
        for preprocess in preprocessor_cfg:
            preprocessor = instantiate(preprocess, **meta)
            meta = preprocessor.update_meta(meta)
            self.preprocessor.append(preprocessor)

    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """preprocess images and targets step by step.
        Args:
            data (dict): input data.
        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary following the format
            {"image":
                {
                encoder_modality_1: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                ...
                encoder_modality_N: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                 },
            "target": torch.Tensor of shape (H W),
             "metadata": dict}.
        """
        # self.check_dimension(data)
        # for process in self.preprocessor:
        #     data = process(data)

        return data


class BandFilter(BasePreprocessor):
    def __init__(self, **meta) -> None:
        """Intialize the BandFilter.
        Args:
            meta: statistics/info of the input data and target encoder
                data_bands: bands of incoming data
                encoder_bands: expected bands by encoder
        """
        super().__init__()

        self.used_bands_indices = {}

        for k in meta["data_bands"].keys():
            if k not in meta["encoder_bands"].keys():
                continue
            self.used_bands_indices[k] = torch.tensor(
                [
                    meta["data_bands"][k].index(b)
                    for b in meta["encoder_bands"][k]
                    if b in meta["data_bands"][k]
                ],
                dtype=torch.long,
            )

        if not self.used_bands_indices:
            raise ValueError("No nontrivial input bands after BandFilter!")

    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Filter redundant bands from the data.
        Args:
            data (dict): input data.
        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary following the format
            {"image":
                {
                encoder_modality_1: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                ...
                encoder_modality_N: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                 },
            "target": torch.Tensor of shape (H W),
             "metadata": dict}.
        """

        data["image"] = {
            k: data["image"][k][v] for k, v in self.used_bands_indices.items()
        }

        return data

    def update_meta(self, meta):
        """Tracking the meta statistics/info for next processor."""
        for k in list(meta["data_bands"].keys()):
            if k not in self.used_bands_indices.keys():
                meta["data_bands"].pop(k, None)
                meta["data_mean"].pop(k, None)
                meta["data_std"].pop(k, None)
                meta["data_min"].pop(k, None)
                meta["data_max"].pop(k, None)
            else:
                meta["data_bands"][k] = [
                    meta["data_bands"][k][i.item()] for i in self.used_bands_indices[k]
                ]
                meta["data_mean"][k] = meta["data_mean"][k][self.used_bands_indices[k]]
                meta["data_std"][k] = meta["data_std"][k][self.used_bands_indices[k]]
                meta["data_min"][k] = meta["data_min"][k][self.used_bands_indices[k]]
                meta["data_max"][k] = meta["data_max"][k][self.used_bands_indices[k]]

        return meta


class BandPadding(BasePreprocessor):
    def __init__(self, fill_value: float = 0.0, **meta) -> None:
        """Intialize the BandPadding.
        Args:
            fill_value (float): fill value for padding.
            meta: statistics/info of the input data and target encoder
                data_bands: bands of incoming data
                encoder_bands: expected bands by encoder
        """
        super().__init__()

        self.fill_value = fill_value
        self.data_img_size = meta["data_img_size"]

        self.encoder_bands = meta["encoder_bands"]
        self.avail_bands_mask, self.used_bands_indices = {}, {}
        for k in meta["encoder_bands"].keys():
            if k in meta["data_bands"].keys():
                self.avail_bands_mask[k] = torch.tensor(
                    [b in meta["data_bands"][k] for b in meta["encoder_bands"][k]],
                    dtype=torch.bool,
                )
                self.used_bands_indices[k] = torch.tensor(
                    [
                        meta["data_bands"][k].index(b)
                        for b in meta["encoder_bands"][k]
                        if b in meta["data_bands"][k]
                    ],
                    dtype=torch.long,
                )
            else:
                self.avail_bands_mask[k] = torch.zeros(
                    len(meta["encoder_bands"][k]), dtype=torch.bool
                )
        if not self.used_bands_indices:
            raise ValueError("No nontrivial input bands after BandPadding!")

    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        for k in self.avail_bands_mask.keys():
            if k in self.used_bands_indices.keys():
                size = self.avail_bands_mask[k].shape + data["image"][k].shape[1:]
                padded_image = torch.full(
                    size, fill_value=self.fill_value, dtype=data["image"][k].dtype
                )
                padded_image[self.avail_bands_mask[k]] = data["image"][k][
                    self.used_bands_indices[k]
                ]
            else:
                reference = data["image"](list(data["image"].keys())[0])
                size = self.avail_bands_mask[k].shape + reference.shape[1:]
                padded_image = torch.full(
                    size, fill_value=self.fill_value, dtype=reference.dtype
                )

            data["image"][k] = padded_image
        return data

    def update_meta(self, meta):
        """Tracking the meta statistics/info for next processor."""
        meta["data_bands"] = meta["encoder_bands"]
        for k in self.avail_bands_mask.keys():
            size = self.avail_bands_mask[k].shape
            meta["data_mean"][k] = torch.full(
                size, fill_value=self.fill_value, dtype=torch.float
            )
            meta["data_std"][k] = torch.ones(size, dtype=torch.float)
            meta["data_min"][k] = torch.full(
                size, fill_value=self.fill_value, dtype=torch.float
            )
            meta["data_max"][k] = torch.full(
                size, fill_value=self.fill_value, dtype=torch.float
            )
            if self.used_bands_indices[k] is not None:
                meta["data_mean"][k][self.avail_bands_mask[k]] = meta["data_mean"][k][
                    self.used_bands_indices[k]
                ]
                meta["data_std"][k][self.avail_bands_mask[k]] = meta["data_std"][k][
                    self.used_bands_indices[k]
                ]
                meta["data_min"][k][self.avail_bands_mask[k]] = meta["data_min"][k][
                    self.used_bands_indices[k]
                ]
                meta["data_max"][k][self.avail_bands_mask[k]] = meta["data_max"][k][
                    self.used_bands_indices[k]
                ]
        return meta


class NormalizeMeanStd(BasePreprocessor):
    def __init__(
        self,
        **meta,
    ) -> None:
        """Initialize the NormalizeMeanStd.
        Args:
            meta: statistics/info of the input data and target encoder
                data_mean: global mean of incoming data
                data_std: global std of incoming data
        """
        super().__init__()

        self.data_mean = meta["data_mean"]
        self.data_std = meta["data_std"]

    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Apply Mean/Std Normalization to the data.
        Args:
            data (dict): input data.
        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary following the format
            {"image":
                {
                encoder_modality_1: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                ...
                encoder_modality_N: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                 },
            "target": torch.Tensor of shape (H W),
             "metadata": dict}.
        """

        for k in self.data_mean.keys():
            data["image"][k].sub_(self.data_mean[k].view(-1, 1, 1, 1)).div_(
                self.data_std[k].view(-1, 1, 1, 1)
            )
        return data

    def update_meta(self, meta):
        """Tracking the meta statistics/info for next processor."""
        meta["data_mean"] = {
            k: torch.zeros_like(v) for k, v in meta["data_mean"].items()
        }
        meta["data_std"] = {k: torch.ones_like(v) for k, v in meta["data_std"].items()}
        meta["data_min"] = {
            k: (v - meta["data_mean"][k]) / meta["data_std"][k]
            for k, v in meta["data_min"].items()
        }
        meta["data_max"] = {
            k: (v - meta["data_mean"][k]) / meta["data_std"][k]
            for k, v in meta["data_max"].items()
        }

        return meta


class NormalizeMinMax(BasePreprocessor):
    def __init__(
        self,
        **meta,
    ) -> None:
        """Initialize the NormalizeMinMax.
        Args:
            meta: statistics/info of the input data and target encoder
                data_min: global minimum value of incoming data
                data_max: global maximum value of incoming data
        """
        super().__init__()

        self.data_min = meta["data_min"]
        self.data_max = meta["data_max"]

    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Apply Min/Max Normalization to the data.
        Args:
            data (dict): input data.
        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary following the format
            {"image":
                {
                encoder_modality_1: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                ...
                encoder_modality_N: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                 },
            "target": torch.Tensor of shape (H W),
             "metadata": dict}.
        """

        for k in self.data_min.keys():
            data["image"][k].sub_(self.data_min[k].view(-1, 1, 1, 1)).div_(
                (self.data_max[k] - self.data_min[k]).view(-1, 1, 1, 1)
            )
        return data

    def update_meta(self, meta):
        """Tracking the meta statistics/info for next processor."""
        meta["data_mean"] = {
            k: (v - meta["data_min"][k]) / (meta["data_max"][k] - meta["data_min"][k])
            for k, v in meta["data_mean"].items()
        }
        meta["data_std"] = {
            k: v / (meta["data_max"][k] - meta["data_min"][k])
            for k, v in meta["data_std"].items()
        }
        meta["data_min"] = {
            k: torch.zeros_like(v) for k, v in meta["data_mean"].items()
        }
        meta["data_max"] = {k: torch.ones_like(v) for k, v in meta["data_std"].items()}

        return meta


class RandomCrop(BasePreprocessor):
    def __init__(
        self, size: int | Sequence[int], pad_if_needed: bool = False, **meta
    ) -> None:
        """Initialize the RandomCrop preprocessor.
        Args:
            size (int): crop size.
            pad_if_needed (bool, optional): whether to pad. Defaults to False.
            meta: statistics/info of the input data and target encoder
                data_mean: global mean value of incoming data for potential padding
                ignore_index: ignore index for potential padding
        """
        super().__init__()

        self.size = tuple(
            _setup_size(
                size, error_msg="Please provide only two dimensions (h, w) for size."
            )
        )
        self.pad_if_needed = pad_if_needed
        self.pad_value = meta["data_mean"]
        self.ignore_index = meta["ignore_index"]

    def get_params(self, data: dict) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            data (dict): input data.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = data["image"][list(data["image"].keys())[0]].shape[-2:]
        th, tw = self.size
        if h < th or w < tw:
            raise ValueError(
                f"Required crop size {(th, tw)} is larger than input image size {(h, w)}"
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def check_pad(
        self,
        data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        _, t, height, width = data["image"][list(data["image"].keys())[0]].shape

        if height < self.size[0] or width < self.size[1]:
            pad_img = max(self.size[0] - height, 0), max(self.size[1] - width, 0)
            height, width = height + 2 * pad_img[0], width + 2 * pad_img[1]
            for k, v in data["image"].items():
                padded_img = (
                    self.pad_value[k].reshape(-1, 1, 1, 1).repeat(1, t, height, width)
                )
                padded_img[:, :, pad_img[0] : -pad_img[0], pad_img[1] : -pad_img[1]] = v
                data["image"][k] = padded_img

            data["target"] = TF.pad(
                data["target"],
                padding=padded_img,
                fill=self.ignore_index,
                padding_mode="constant",
            )

        return data

    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Random crop the data.
        Args:
            data (dict): input data.
        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary following the format
            {"image":
                {
                encoder_modality_1: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                ...
                encoder_modality_N: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                 },
            "target": torch.Tensor of shape (H W),
             "metadata": dict}.
        """
        self.check_size(data)

        if self.pad_if_needed:
            data = self.check_pad(data)

        i, j, h, w = self.get_params(data=data)

        for k, v in data["image"].items():
            data["image"][k] = TF.crop(v, i, j, h, w)

        data["target"] = TF.crop(data["target"], i, j, h, w)

        return data

    def update_meta(self, meta):
        """Tracking the meta statistics/info for next processor."""
        meta["data_img_size"] = self.size[0]
        return meta


class RandomCropToEncoder(RandomCrop):
    def __init__(self, pad_if_needed: bool = False, **meta) -> None:
        """Initialize the RandomCropToEncoder preprocessor.
        Args:
            size (int): crop size.
            pad_if_needed (bool, optional): whether to pad. Defaults to False.
            meta: statistics/info of the input data and target encoder
                data_mean: global mean value of incoming data for potential padding
                ignore_index: ignore index for potential padding
        """
        size = meta["encoder_input_size"]
        super().__init__(size, pad_if_needed, **meta)


class FocusRandomCrop(RandomCrop):
    def __init__(self, size: int, pad_if_needed: bool = False, **meta) -> None:
        """Initialize the FocusRandomCrop preprocessor.
        Args:
            size (int): crop size.
            pad_if_needed (bool, optional): whether to pad. Defaults to False.
            meta: statistics/info of the input data and target encoder
                data_mean: global mean value of incoming data for potential padding
                ignore_index: ignore index for potential padding
        """
        super().__init__(size, pad_if_needed, **meta)

    def get_params(self, data: dict) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.

        Args:
            data (dict): input data.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """

        h, w = data["target"].shape
        th, tw = self.size

        if h < th or w < tw:
            raise ValueError(
                f"Required crop size {(th, tw)} is larger than input image size {(h, w)}"
            )

        if w == tw and h == th:
            return 0, 0, h, w

        valid_map = data["target"] != self.ignore_index
        idx = torch.arange(0, h * w)[valid_map.flatten()]
        sample = idx[random.randint(0, idx.shape[0] - 1)]
        y, x = sample // w, sample % w

        i = random.randint(max(0, y - th), min(y, h - th + 1))
        j = random.randint(max(0, x - tw), min(x, w - tw + 1))

        return i, j, th, tw


class FocusRandomCropToEncoder(FocusRandomCrop):
    def __init__(self, pad_if_needed: bool = False, **meta) -> None:
        """Initialize the FocusRandomCropToEncoder preprocessor.
        Args:
            pad_if_needed (bool, optional): whether to pad. Defaults to False.
            meta: statistics/info of the input data and target encoder
                data_mean: global mean value of incoming data for potential padding
                ignore_index: ignore index for potential padding
        """
        size = meta["encoder_input_size"]
        super().__init__(size, pad_if_needed, **meta)


class ImportanceRandomCrop(RandomCrop):
    def __init__(
        self, size, pad_if_needed: bool = False, num_trials: int = 10, **meta
    ) -> None:
        """Initialize the FocusRandomCrop preprocessor.
        Args:
            size (int): crop size.
            pad_if_needed (bool, optional): whether to pad. Defaults to False.
            num_trials (int, optional): number of trials. Defaults to 10.
            meta: statistics/info of the input data and target encoder
                data_mean: global mean value of incoming data for potential padding
                ignore_index: ignore index for potential padding
        """
        super().__init__(size, pad_if_needed, **meta)

        self.num_trials = num_trials
        self.class_weight = 1 / meta["class_distribution"]

    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Random crop the data.
        Args:
            data (dict): input data.
        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary following the format
            {"image":
                {
                encoder_modality_1: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                ...
                encoder_modality_N: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                 },
            "target": torch.Tensor of shape (H W),
             "metadata": dict}.
        """
        self.check_size(data)

        if self.pad_if_needed:
            data, height, width = self.check_pad(data)
        else:
            _, _, height, width = data["image"][list(data["image"].keys())[0]].shape

        valid = data["target"] != self.ignore_index
        weight_map = torch.full(
            size=data["target"].shape, fill_value=1e-6, dtype=torch.float
        )
        weight_map[valid] = self.class_weight[data["target"][valid]]

        crop_candidates = [self.get_params(data) for _ in range(self.num_trials)]
        crop_weights = [
            weight_map[i : i + h, j : j + w].sum().item() / (h * w)
            for i, j, h, w in crop_candidates
        ]
        crop_weights = np.array(crop_weights)
        crop_weights = crop_weights / crop_weights.sum()

        crop_idx = np.random.choice(self.num_trials, p=crop_weights)
        i, j, h, w = crop_candidates[crop_idx]

        for k, v in data["image"].items():
            data["image"][k] = TF.crop(v, i, j, h, w)

        data["target"] = TF.crop(data["target"], i, j, h, w)

        return data


class ImportanceRandomCropToEncoder(ImportanceRandomCrop):
    def __init__(
        self, pad_if_needed: bool = False, num_trials: int = 10, **meta
    ) -> None:
        """Initialize the FocusRandomCrop preprocessor.
        Args:
            size (int): crop size.
            pad_if_needed (bool, optional): whether to pad. Defaults to False.
            num_trials (int, optional): number of trials. Defaults to 10.
            meta: statistics/info of the input data and target encoder
                data_mean: global mean value of incoming data for potential padding
                ignore_index: ignore index for potential padding
        """
        size = meta["encoder_input_size"]
        super().__init__(size, pad_if_needed, num_trials, **meta)


class Resize(BasePreprocessor):
    def __init__(
        self,
        size: int | Sequence[int],
        interpolation=T.InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
        resize_target: bool = True,
        **meta,
    ) -> None:
        """Initialize the Resize preprocessor.
        Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
        antialias (bool, optional): Whether to apply antialiasing.
        resize_target (bool, optional): Whether to resize the target
        meta: statistics/info of the input data and target encoder
        """
        super().__init__()

        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = _setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        )

        if isinstance(interpolation, int):
            interpolation = TF._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias
        self.resize_target = resize_target

    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Resize the data.
        Args:
            data (dict): input data.
        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary following the format
            {"image":
                {
                encoder_modality_1: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                ...
                encoder_modality_N: torch.Tensor of shape (C T H W) (T=1 if single timeframe),
                 },
            "target": torch.Tensor of shape (H W),
             "metadata": dict}.
        """
        for k, v in data["image"].items():
            data["image"][k] = TF.resize(
                data["image"][k],
                self.size,
                interpolation=self.interpolation,
                antialias=self.antialias,
            )

        if self.resize_target:
            if torch.is_floating_point(data["target"]):
                data["target"] = TF.resize(
                    data["target"].unsqueeze(0),
                    size=self.size,
                    interpolation=T.InterpolationMode.BILINEAR,
                ).squeeze(0)
            else:
                data["target"] = TF.resize(
                    data["target"].unsqueeze(0),
                    size=self.size,
                    interpolation=T.InterpolationMode.NEAREST,
                ).squeeze(0)

        return data

    def update_meta(self, meta):
        """Tracking the meta statistics/info for next processor."""
        meta["data_img_size"] = self.size[0]
        return meta


class ResizeToEncoder(Resize):
    def __init__(
        self,
        interpolation=T.InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
        resize_target: bool = False,
        **meta,
    ) -> None:
        """Initialize the ResizeToEncoder preprocessor.
        Args:
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
        antialias (bool, optional): Whether to apply antialiasing.
        resize_target (bool, optional): Whether to resize the target
        meta: statistics/info of the input data and target encoder
        """
        size = meta["encoder_input_size"]
        super().__init__(size, interpolation, antialias, resize_target, **meta)


class RandomResizedCrop(BasePreprocessor):
    def __init__(
        self,
        size: int | Sequence[int],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.3333333333333333),
        interpolation=T.InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
        resize_target: bool = True,
        **meta,
    ) -> None:
        """Initialize the RandomResizedCrop preprocessor.
        Args:
        size (int): crop size.
        scale (list): range of scale of the origin size cropped
        ratio (list): range of aspect ratio of the origin aspect ratio cropped
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
        antialias (bool, optional): Whether to apply antialiasing.
        resize_target (bool, optional): Whether to resize the target
        meta: statistics/info of the input data and target encoder
        """
        super().__init__()
        self.size = _setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        )

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError("Scale and ratio should be of kind (min, max)")

        if isinstance(interpolation, int):
            interpolation = TF._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias
        self.scale = scale
        self.ratio = ratio
        self.resize_target = resize_target

    @staticmethod
    def get_params(
        input_size: Tuple[int, int],
        scale: Tuple[float, float],
        ratio: Tuple[float, float],
    ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        height, width = input_size
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        self.check_size(data)

        _, t, h_img, w_img = data["image"][list(data["image"].keys())[0]].shape

        i, j, h, w = self.get_params((h_img, w_img), self.scale, self.ratio)

        for k, v in data["image"].items():
            data["image"][k] = TF.resized_crop(
                data["image"][k],
                i,
                j,
                h,
                w,
                self.size,
                self.interpolation,
                antialias=self.antialias,
            )

        if self.resize_target:
            if torch.is_floating_point(data["target"]):
                data["target"] = TF.resized_crop(
                    data["target"].unsqueeze(0),
                    i,
                    j,
                    h,
                    w,
                    self.size,
                    T.InterpolationMode.BILINEAR,
                ).squeeze(0)
            else:
                data["target"] = TF.resized_crop(
                    data["target"].unsqueeze(0),
                    i,
                    j,
                    h,
                    w,
                    self.size,
                    T.InterpolationMode.NEAREST,
                ).squeeze(0)

        else:
            data["target"] = TF.crop(data["target"], i, j, h, w)

        return data

    def update_meta(self, meta):
        """Tracking the meta statistics/info for next processor."""
        meta["data_img_size"] = self.size[0]
        return meta


class RandomResizedCropToEncoder(RandomResizedCrop):
    def __init__(
        self,
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.3333333333333333),
        interpolation=T.InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
        resize_target: bool = True,
        **meta,
    ) -> None:
        """Initialize the RandomResizedCropToEncoder preprocessor.
        Args:
        scale (list): range of scale of the origin size cropped
        ratio (list): range of aspect ratio of the origin aspect ratio cropped
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
        antialias (bool, optional): Whether to apply antialiasing.
        resize_target (bool, optional): Whether to resize the target
        meta: statistics/info of the input data and target encoder
        """
        size = meta["encoder_input_size"]
        super().__init__(
            size, scale, ratio, interpolation, antialias, resize_target, **meta
        )


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

