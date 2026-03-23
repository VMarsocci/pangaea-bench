import os
from glob import glob

import numpy as np
import tifffile as tiff
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from pangaea.datasets.base import RawGeoFMDataset


class FiveBillionPixels(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list[int],
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
        download_url: str,
        auto_download: bool,
        use_cmyk: bool,
    ):
        """Initialize the FiveBillionPixels dataset.
        Link to original dataset: https://x-ytong.github.io/project/Five-Billion-Pixels.html

        Args:
            split (str): split of the dataset (train, val, test).
            dataset_name (str): dataset name.
            multi_modal (bool): if the dataset is multi-modal.
            multi_temporal (int): number of temporal frames.
            root_path (str): root path of the dataset.
            classes (list): classes of the dataset.
            num_classes (int): number of classes.
            ignore_index (int): index to ignore for metrics and loss.
            img_size (int): size of the image.
            bands (dict[str, list[str]]): bands of the dataset.
            distribution (list[int]): class distribution.
            data_mean (dict[str, list[str]]): mean for each band for each modality.
            Dictionary with keys as the modality and values as the list of means.
            e.g. {"s2": [b1_mean, ..., bn_mean], "s1": [b1_mean, ..., bn_mean]}
            data_std (dict[str, list[str]]): str for each band for each modality.
            Dictionary with keys as the modality and values as the list of stds.
            e.g. {"s2": [b1_std, ..., bn_std], "s1": [b1_std, ..., bn_std]}
            data_min (dict[str, list[str]]): min for each band for each modality.
            Dictionary with keys as the modality and values as the list of mins.
            e.g. {"s2": [b1_min, ..., bn_min], "s1": [b1_min, ..., bn_min]}
            data_max (dict[str, list[str]]): max for each band for each modality.
            Dictionary with keys as the modality and values as the list of maxs.
            e.g. {"s2": [b1_max, ..., bn_max], "s1": [b1_max, ..., bn_max]}
            download_url (str): url to download the dataset.
            auto_download (bool): whether to download the dataset automatically.
            use_cmyk (bool): wheter to use cmyk or RGB-NIR colours for images.
        """
        super(FiveBillionPixels, self).__init__(
            split=split,
            dataset_name=dataset_name,
            multi_modal=multi_modal,
            multi_temporal=multi_temporal,
            root_path=root_path,
            classes=classes,
            num_classes=num_classes,
            ignore_index=ignore_index,
            img_size=img_size,
            bands=bands,
            distribution=distribution,
            data_mean=data_mean,
            data_std=data_std,
            data_min=data_min,
            data_max=data_max,
            download_url=download_url,
            auto_download=auto_download,
            # use_cmyk=use_cmyk,
        )

        self._base_dir = root_path
        self.classes = classes
        self.use_cmyk = use_cmyk
        self.split = split

        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.classes = classes
        self.img_size = img_size
        self.distribution = distribution
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.download_url = download_url
        self.auto_download = auto_download

        self._image_dir = sorted(
            glob(os.path.join(self._base_dir, self.split, "images", "*.tif"))
        )
        self._label_dir = sorted(
            glob(os.path.join(self._base_dir, self.split, "labels", "*.tif"))
        )

    def __len__(self):
        return len(self._image_dir)

    def __getitem__(self, index):
        filename = os.path.basename(self._image_dir[index])
        if self.use_cmyk:
            image = Image.open(self._image_dir[index]).convert("CMYK")
            image = TF.pil_to_tensor(image)
        else:
            image = tiff.imread(
                self._image_dir[index]
            )  # .convert('CMYK') #check it also on the normalization
            image = image.astype(np.float32)  # Convert to float32
            image = torch.from_numpy(image).permute(2, 0, 1)

            # output image shape (C T=1 H W)
            image = image.unsqueeze(1)
        target = tiff.imread(self._label_dir[index])
        target = target.astype(np.int64)  # Convert to int64 (since it's a mask)
        target = torch.from_numpy(target).long()

        output = {
            "image": {
                "optical": image,
            },
            "target": target,
            "metadata": {"filename": filename},
        }

        return output

    # @staticmethod
    # def get_splits(dataset_config):
    #     dataset_train = FiveBillionPixels(dataset_config, split="train")
    #     dataset_val = FiveBillionPixels(dataset_config, split="val")
    #     dataset_test = FiveBillionPixels(dataset_config, split="test")
    #     return dataset_train, dataset_val, dataset_test
