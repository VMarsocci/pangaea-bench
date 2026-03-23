import os
import time 
import pathlib
import urllib.request
import urllib.error
import zipfile

from glob import glob
import rasterio
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from pangaea.datasets.utils import DownloadProgressBar
from pangaea.datasets.base import RawGeoFMDataset

###############################################################
# MADOS DATASET                                               #
###############################################################

# @DATASET_REGISTRY.register()
class MADOS(RawGeoFMDataset):
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
        """Initialize the MADOS dataset.
        Link: https://marine-pollution.github.io/index.html

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
        super(MADOS, self).__init__(
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

        self.ROIs_split = np.genfromtxt(os.path.join(self.root_path, 'splits', f'{split}_X.txt'), dtype='str')

        self.image_list = []
        self.target_list = []

        self.tiles = sorted(glob(os.path.join(self.root_path, '*')))

        for tile in self.tiles:
            splits = [f.split('_cl_')[-1] for f in glob(os.path.join(tile, '10', '*_cl_*'))]

            for crop in splits:
                crop_name = os.path.basename(tile) + '_' + crop.split('.tif')[0]

                if crop_name in self.ROIs_split:
                    all_bands = glob(os.path.join(tile, '*', '*L2R_rhorc*_' + crop))
                    all_bands = sorted(all_bands, key=self.get_band)
                    # all_bands = np.array(all_bands)

                    self.image_list.append(all_bands)

                    cl_path = os.path.join(tile, '10', os.path.basename(tile) + '_L2R_cl_' + crop)
                    self.target_list.append(cl_path)

    def __len__(self):
        return len(self.image_list)

    def getnames(self):
        return self.ROIs_split

    def __getitem__(self, index):
        filename = os.path.basename(self.target_list[index])

        all_bands = self.image_list[index]
        current_image = []
        for c, band in enumerate(all_bands):
            upscale_factor = int(os.path.basename(os.path.dirname(band))) // 10
            with rasterio.open(band, mode='r') as src:
                this_band = src.read(1,
                                     out_shape=(int(src.height * upscale_factor), int(src.width * upscale_factor)),
                                     resampling=rasterio.enums.Resampling.nearest
                                     )
                this_band = torch.from_numpy(this_band)
                #this_band[torch.isnan(this_band)] = self.data_mean['optical'][c]
                current_image.append(this_band)

        image = torch.stack(current_image)
        invalid_mask = torch.isnan(image)
        image[invalid_mask] = 0

        # images must be of shape (C T H W)
        image = image.unsqueeze(1)

        with rasterio.open(self.target_list[index], mode='r') as src:
            target = src.read(1)
        target = torch.from_numpy(target.astype(np.int64))
        target = target - 1

        output = {
            'image': {
                'optical': image,
            },
            'target': target,
            'metadata': {'filename': filename}
        }

        return output

    @staticmethod
    def get_band(path):
        return int(path.split('_')[-2])

    @staticmethod
    def download(self, silent=False):
        output_path = pathlib.Path(self.root_path)
        url = self.download_url

        existing_dirs = list(output_path.glob("Scene_*"))
        if existing_dirs:
            if not silent:
                print("MADOS Dataset folder exists, skipping downloading dataset.")
            return

        output_path.mkdir(parents=True, exist_ok=True)

        temp_file_name = f"temp_{hex(int(time.time()))}_MADOS.zip"
        pbar = DownloadProgressBar()

        try:
            urllib.request.urlretrieve(url, output_path / temp_file_name, pbar)
        except urllib.error.HTTPError as e:
            print('Error while downloading dataset: The server couldn\'t fulfill the request.')
            print('Error code: ', e.code)
            return
        except urllib.error.URLError as e:
            print('Error while downloading dataset: Failed to reach a server.')
            print('Reason: ', e.reason)
            return

        with zipfile.ZipFile(output_path / temp_file_name, 'r') as zip_ref:
            print(f"Extracting to {output_path} ...")
            # Remove top-level dir in ZIP file for nicer data dir structure
            members = []
            for zipinfo in zip_ref.infolist():
                new_path = os.path.join(*(zipinfo.filename.split(os.path.sep)[1:]))
                zipinfo.filename = str(new_path)
                members.append(zipinfo)

            zip_ref.extractall(output_path, members)
            print("done.")

        (output_path / temp_file_name).unlink()
