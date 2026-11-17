import io
import os
import urllib.request

import numpy as np
import tacoreader.v1 as tacoreader
import torch
from PIL import Image

from pangaea.datasets.base import RawGeoFMDataset
from pangaea.datasets.utils import read_tortilla_subfile_bytes

_TORTILLA_FILE = "geobench_mpv4ger.tortilla"
_RGB_BAND_INDEX = {"B4": 0, "B3": 1, "B2": 2}


class mPv4ger(RawGeoFMDataset):
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
        """Initialize the m-PV4-Ger dataset.
            Link: https://github.com/ServiceNow/geo-bench
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
        super(mPv4ger, self).__init__(
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

        self.root_path = root_path
        self.split = split

        df = tacoreader.load(os.path.join(self.root_path, _TORTILLA_FILE))
        self.dataset = df[df["tortilla:data_split"] == self.split].reset_index(
            drop=True
        )
        self._labels = [int(v) for v in self.dataset["label"]]
        self._band_indices = [_RGB_BAND_INDEX[b] for b in bands["optical"]]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        vsi_path = self.dataset.read(index)
        png = read_tortilla_subfile_bytes(vsi_path)
        arr = np.array(Image.open(io.BytesIO(png)), dtype=np.float32)
        image = torch.from_numpy(arr).permute(2, 0, 1)[self._band_indices].unsqueeze(1)
        return {
            "image": {"optical": image},
            "target": torch.tensor(self._labels[index], dtype=torch.int64),
            "metadata": {},
        }

    def download(self, silent=False):
        os.makedirs(self.root_path, exist_ok=True)
        urllib.request.urlretrieve(
            self.download_url, os.path.join(self.root_path, _TORTILLA_FILE)
        )
