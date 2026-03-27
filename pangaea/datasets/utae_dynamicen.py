import os
import numpy as np
import rasterio
import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
from datetime import datetime
# import torchvision.transforms.functional as TF
# import cv2

# import random
# from PIL import Image

from pangaea.datasets.base import RawGeoFMDataset

# from utils.registry import DATASET_REGISTRY

# @DATASET_REGISTRY.register()
class DynamicEarthNet(RawGeoFMDataset):
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
        sample_dates: list[int]
    ):
        """Initialize the DynamicEarthNet dataset.
        Link: https://github.com/aysim/dynnet

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
        super(DynamicEarthNet, self).__init__(
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
        self.ignore_index = ignore_index
        self.split = split
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.classes = classes
        self.img_size = img_size
        self.distribution = distribution
        self.num_classes = num_classes
        self.download_url = download_url
        self.auto_download = auto_download

        self.sample_dates = [str(d).rjust(2,'0') for d in sample_dates]

        self.files = []

        reference_date = "2018-01-01"
        self.reference_date = datetime(*map(int, reference_date.split("-")))

        self.set_files()


    def set_files(self):
        self.file_list = os.path.join(self.root_path, "dynnet_training_splits", f"{self.split}" + ".txt")

        file_list = [line.rstrip().split(' ') for line in tuple(open(self.file_list, "r"))]
        #for
        self.files, self.labels, self.year_months = list(zip(*file_list))
        self.files = [f.replace('/reprocess-cropped/UTM-24000/', '/planet/') for f in self.files]

        self.all_sequences = []
        for f, ym in zip(self.files, self.year_months):
            images = []
            for date in self.sample_dates:
                image_file = os.path.join(self.root_path, f[1:], f"{ym}-{date}.npy")
                assert os.path.isfile(image_file), f"{image_file} does not exist"
                images.append(image_file)
            self.all_sequences.append(images)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = os.path.basename(self.labels[index])
        images = [np.load(seq) for seq in self.all_sequences[index]]
        images = torch.from_numpy(np.stack(images, axis=0)).transpose(0, 1).float()
        label = torch.from_numpy(np.load(os.path.join(self.root_path, self.labels[index][1:].replace('tif', 'npy')))).long()

        output = {
            'image': {
                'optical': images,
            },
            'target': label,
            'metadata': {'filename': filename}
        }

        return output


    @staticmethod
    def download(self, silent=False):
        pass
