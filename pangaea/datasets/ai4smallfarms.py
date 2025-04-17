import os
import pathlib
from glob import glob

import numpy as np
import requests
import torch
from pyDataverse.api import DataAccessApi, NativeApi
from tifffile import imread

from pangaea.datasets.base import RawGeoFMDataset


class AI4SmallFarms(RawGeoFMDataset):
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
        """Initialize the AI4SmallFarms dataset.
            Link: https://phys-techsciences.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/dans-xy6-ngg6

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
        super(AI4SmallFarms, self).__init__(
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

        self.root_path = pathlib.Path(root_path)
        self.split = split
        self.image_dir = self.root_path.joinpath(f"sentinel-2-asia/{split}/images")
        self.mask_dir = self.root_path.joinpath(f"sentinel-2-asia/{split}/masks")
        self.image_list = sorted(glob(str(self.image_dir.joinpath("*.tif"))))
        self.mask_list = sorted(glob(str(self.mask_dir.joinpath("*.tif"))))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = imread(pathlib.Path(self.image_list[index]))
        target = imread(
            pathlib.Path(self.mask_list[index])
        )  # Assuming target is a single band

        # Convert the image and target to supported types
        image = image.astype(np.float32)  # Convert to float32
        target = target.astype(np.int64)  # Convert to int64 (since it's a mask)

        image = torch.from_numpy(image).permute(2, 0, 1)
        target = (torch.from_numpy(target) / 255).long()

        # Handle invalid data if any
        invalid_mask = torch.isnan(image)
        image[invalid_mask] = 0

        # output image shape (C T=1 H W)
        image = image.unsqueeze(1)

        return {
            "image": {
                "optical": image,
            },
            "target": target,
            "metadata": {},
        }

    @staticmethod
    def download(self, silent=False):
        root_path = pathlib.Path(self.root_path)

        # Create the root directory if it does not exist
        if not root_path.exists():
            root_path.mkdir(parents=True, exist_ok=True)

        if root_path.exists() and any(root_path.iterdir()):
            if not silent:
                print(f"Dataset already exists at {root_path}. Skipping download.")
            return

        # Set up the Dataverse API
        base_url = self.download_url
        api = NativeApi(base_url)
        data_api = DataAccessApi(base_url)
        DOI = "doi:10.17026/dans-xy6-ngg6"

        # Fetch dataset files using NativeAPI
        try:
            dataset = api.get_dataset(DOI)
            files_list = dataset.json()["data"]["latestVersion"]["files"]
        except Exception as e:
            if not silent:
                print(f"Error retrieving dataset metadata: {e}")
            return

        if not files_list:
            if not silent:
                print(f"No files found for DOI: {DOI}")
            return

        # Process each file and download
        for file in files_list:
            filename = file["dataFile"]["filename"]
            file_id = file["dataFile"]["id"]
            directory_label = file.get("directoryLabel", "")
            dv_path = os.path.join(directory_label, filename)

            # Construct the full path for the file
            file_path = root_path / dv_path

            # Create subdirectories if they do not exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if not silent:
                print(f"Downloading file: {filename}, id: {file_id} to {file_path}")

            try:
                # Download the file using its ID
                response = data_api.get_datafile(file_id, is_pid=False)
                response.raise_for_status()  # Check if the request was successful
                with open(file_path, "wb") as f:
                    f.write(response.content)
            except requests.exceptions.HTTPError as err:
                if err.response.status_code == 404:
                    if not silent:
                        print(
                            f"File {filename} with id {file_id} not found (404). Skipping."
                        )
                else:
                    if not silent:
                        print(
                            f"Error downloading file {filename} with id {file_id}: {err}"
                        )
                    raise

        # **Cleanup: Remove unwanted files and directories**
        unwanted_paths = [
            os.path.join(self.root_path, "easy-migration.zip"),
            os.path.join(self.root_path, "readme.md"),
            os.path.join(
                self.root_path, "sentinel-2-asia", "benchmark.qgz"
            ),
            os.path.join(
                self.root_path, "sentinel-2-asia", "tiles_asia.gpkg"
            ),
            os.path.join(self.root_path, "sentinel-2-asia", "reference"),
            os.path.join(
                self.root_path, "sentinel-2-asia", "test", "output"
            ),
            os.path.join(self.root_path, "sentinel-2-nl"),
        ]

        # Remove unwanted files and directories
        for path in unwanted_paths:
            if os.path.exists(path):
                if os.path.isdir(path):
                    os.system(f"rm -rf {path}")  # Remove directories
                else:
                    os.remove(path)  # Remove files
                if not silent:
                    print(f"Removed unwanted path: {path}")

        os.rename(os.path.join(self.root_path, "sentinel-2-asia/validate"), os.path.join(self.root_path, "sentinel-2-asia/val"))

        if not silent:
            print(f"Downloaded dataset to {self.root_path}")
