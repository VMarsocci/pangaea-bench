###
# Open-Canopy Dataset
# original code https://github.com/fajwel/Open-Canopy
###

import json
import os

import numpy as np
import rasterio
import torch

from pangaea.datasets.base import RawGeoFMDataset

class OpenCanopy(RawGeoFMDataset):
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
    ):
        """Initialize the Open-Canopy dataset.

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
        """
        super(OpenCanopy, self).__init__(
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

        assert split in ["train", "val", "test"], "Split must be train, val or test"
        with open("data/canopy_height/geometries.geojson", "r") as f:
            self.metadata = json.load(f)

        # delete all geometries that are not in the split
        self.metadata["features"] = [
            feature for feature in self.metadata["features"] if feature["properties"]["split"] == split
        ]
    

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Get the item at index i.

        Args:
            i (int): index of the item.

        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary follwing the format
            {"image":
                {"optical": torch.Tensor,
                 "sar": torch.Tensor},
            "target": torch.Tensor,
             "metadata": dict}.
        """

        tile = self.metadata["features"][i]
        filename = tile["properties"]["image_name"]
        year = filename.split("_")[-1][:4]
        spot_folder = f"data/canopy_height/{year}/spot/"
        lidar_folder = f"data/canopy_height/{year}/lidar/"
        spot_path = os.path.join(spot_folder, filename)
        lidar_path = os.path.join(lidar_folder, "compressed_lidar_" + filename.split("_")[-1])

        coords = tile["geometry"]["coordinates"]

        with rasterio.open(spot_path) as src:
            window = rasterio.windows.from_bounds(
                coords[0][0][0], coords[0][0][1], coords[0][2][0], coords[0][2][1],
                transform=src.transform
            )
            rgbir = torch.Tensor(src.read(window=window))
        
        with rasterio.open(lidar_path) as src:
            canopy_height = torch.Tensor(src.read(1, window=window))
        
        return {
            "image": {
                "optical": rgbir.to(torch.float).unsqueeze(1),
            },
            "target": canopy_height.to(torch.float),
            "metadata": {},
        }

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: length of the dataset.
        """
        return len(self.metadata["features"])

    @staticmethod
    def download(self):
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="AI4Forest/Open-Canopy",
            repo_type="dataset",
            local_dir="data",
        )
