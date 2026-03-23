# Source: https://github.com/cloudtostreet/Sen1Floods11

import os

import geopandas
import numpy as np
import pandas as pd
import rasterio
import torch

from pangaea.datasets.base import RawGeoFMDataset
from pangaea.datasets.utils import download_bucket_concurrently


class Sen1Floods11(RawGeoFMDataset):
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
        gcs_bucket: str,
    ):
        """Initialize the Sen1Floods11 dataset.
        Link: https://github.com/cloudtostreet/Sen1Floods11

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
            gcs_bucket (str): subset for downloading the dataset.
        """

        self.gcs_bucket = gcs_bucket

        super(Sen1Floods11, self).__init__(
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
        )

        self.root_path = root_path
        self.classes = classes
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

        self.split_mapping = {"train": "train", "val": "valid", "test": "test"}

        split_file = os.path.join(
            self.root_path,
            "v1.1",
            f"splits/flood_handlabeled/flood_{self.split_mapping[split]}_data.csv",
        )
        metadata_file = os.path.join(
            self.root_path, "v1.1", "Sen1Floods11_Metadata.geojson"
        )
        data_root = os.path.join(
            self.root_path, "v1.1", "data/flood_events/HandLabeled/"
        )

        self.metadata = geopandas.read_file(metadata_file)

        with open(split_file) as f:
            file_list = f.readlines()

        file_list = [f.rstrip().split(",") for f in file_list]

        self.s1_image_list = [
            os.path.join(data_root, "S1Hand", f[0]) for f in file_list
        ]
        self.s2_image_list = [
            os.path.join(data_root, "S2Hand", f[0].replace("S1Hand", "S2Hand"))
            for f in file_list
        ]
        self.target_list = [
            os.path.join(data_root, "LabelHand", f[1]) for f in file_list
        ]

    def __len__(self):
        return len(self.s1_image_list)

    def _get_date(self, index):
        file_name = self.s2_image_list[index]
        location = os.path.basename(file_name).split("_")[0]
        if self.metadata[self.metadata["location"] == location].shape[0] != 1:
            date = pd.to_datetime("13-10-1998", dayfirst=True)
        else:
            date = pd.to_datetime(
                self.metadata[self.metadata["location"] == location]["s2_date"].item()
            )
        date_np = np.zeros((1, 3))
        date_np[0, 0] = date.year
        date_np[0, 1] = date.dayofyear - 1  # base 0
        date_np[0, 2] = date.hour
        return date_np

    def __getitem__(self, index):
        filename = os.path.basename(self.s2_image_list[index])
        with rasterio.open(self.s2_image_list[index]) as src:
            s2_image = src.read()

        with rasterio.open(self.s1_image_list[index]) as src:
            s1_image = src.read()
            # Convert the missing values (clouds etc.)
            s1_image = np.nan_to_num(s1_image)

        with rasterio.open(self.target_list[index]) as src:
            target = src.read(1)

        timestamp = self._get_date(index)

        s2_image = torch.from_numpy(s2_image).float()
        s1_image = torch.from_numpy(s1_image).float()
        target = torch.from_numpy(target).long()

        output = {
            "image": {
                "optical": s2_image.unsqueeze(1),
                "sar": s1_image.unsqueeze(1),
            },
            "target": target,
            "metadata": {
                "filename": filename,
                "timestamp": timestamp,
            },
        }

        return output

    @staticmethod
    def download(self, silent=False):
        if os.path.exists(self.root_path):
            if not silent:
                print(
                    "Sen1Floods11 Dataset folder exists, skipping downloading dataset."
                )
            return
        download_bucket_concurrently(self.gcs_bucket, self.root_path)
