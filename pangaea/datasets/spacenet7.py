'''
SpaceNet 7 dataset: https://spacenet.ai/sn7-challenge/
'''

import os
import time
from pathlib import Path
import urllib.request
import urllib.error
import tarfile
import shutil
import gdown

import json
from glob import glob
import rasterio
import numpy as np

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T

from abc import abstractmethod

from pangaea.datasets.utils import DownloadProgressBar
from pangaea.datasets.base import RawGeoFMDataset
# from utils.registry import DATASET_REGISTRY

# train/val/test split from https://doi.org/10.3390/rs15215135
SN7_TRAIN = [
    'L15-0331E-1257N_1327_3160_13',
    'L15-0358E-1220N_1433_3310_13',
    'L15-0457E-1135N_1831_3648_13',
    'L15-0487E-1246N_1950_3207_13',
    'L15-0577E-1243N_2309_3217_13',
    'L15-0586E-1127N_2345_3680_13',
    'L15-0595E-1278N_2383_3079_13',
    'L15-0632E-0892N_2528_4620_13',
    'L15-0683E-1006N_2732_4164_13',
    'L15-0924E-1108N_3699_3757_13',
    'L15-1015E-1062N_4061_3941_13',
    'L15-1138E-1216N_4553_3325_13',
    'L15-1203E-1203N_4815_3378_13',
    'L15-1204E-1202N_4816_3380_13',
    'L15-1209E-1113N_4838_3737_13',
    'L15-1210E-1025N_4840_4088_13',
    'L15-1276E-1107N_5105_3761_13',
    'L15-1298E-1322N_5193_2903_13',
    'L15-1389E-1284N_5557_3054_13',
    'L15-1438E-1134N_5753_3655_13',
    'L15-1439E-1134N_5759_3655_13',
    'L15-1481E-1119N_5927_3715_13',
    'L15-1538E-1163N_6154_3539_13',
    'L15-1615E-1206N_6460_3366_13',
    'L15-1669E-1153N_6678_3579_13',
    'L15-1669E-1160N_6679_3549_13',
    'L15-1672E-1207N_6691_3363_13',
    'L15-1703E-1219N_6813_3313_13',
    'L15-1709E-1112N_6838_3742_13',
    'L15-1716E-1211N_6864_3345_13',
]
SN7_VAL = [
    'L15-0357E-1223N_1429_3296_13',
    'L15-0361E-1300N_1446_2989_13',
    'L15-0368E-1245N_1474_3210_13',
    'L15-0566E-1185N_2265_3451_13',
    'L15-0614E-0946N_2459_4406_13',
    'L15-0760E-0887N_3041_4643_13',
    'L15-1014E-1375N_4056_2688_13',
    'L15-1049E-1370N_4196_2710_13',
    'L15-1185E-0935N_4742_4450_13',
    'L15-1289E-1169N_5156_3514_13',
    'L15-1296E-1198N_5184_3399_13',
    'L15-1615E-1205N_6460_3370_13',
    'L15-1617E-1207N_6468_3360_13',
    'L15-1669E-1160N_6678_3548_13',
    'L15-1748E-1247N_6993_3202_13',
]
SN7_TEST = [
    'L15-0387E-1276N_1549_3087_13',
    'L15-0434E-1218N_1736_3318_13',
    'L15-0506E-1204N_2027_3374_13',
    'L15-0544E-1228N_2176_3279_13',
    'L15-0977E-1187N_3911_3441_13',
    'L15-1025E-1366N_4102_2726_13',
    'L15-1172E-1306N_4688_2967_13',
    'L15-1200E-0847N_4802_4803_13',
    'L15-1204E-1204N_4819_3372_13',
    'L15-1335E-1166N_5342_3524_13',
    'L15-1479E-1101N_5916_3785_13',
    'L15-1690E-1211N_6763_3346_13',
    'L15-1691E-1211N_6764_3347_13',
    'L15-1848E-0793N_7394_5018_13',
]


###############################################################
# SPACENET 7 DATASET                                             #
###############################################################

class AbstractSN7(RawGeoFMDataset):

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
        domain_shift: bool,
        i_split: int,
        j_split: int,
    ):
        """Initialize the SpaceNet dataset.
        Link: https://spacenet.ai/sn7-challenge/

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
            domain_shift (bool): wheter to perform domain adaptation evaluation.
            i_splt (int): .
            j_split (int): . #ISSUES
        """
        super().__init__(
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


        self.root_path = Path(root_path)
        metadata_file = self.root_path / 'metadata_train.json'
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        self.sn7_img_size = 1024  # size of the SpaceNet 7 images
        self.img_size = img_size  # size used for tiling the images
        assert self.sn7_img_size % self.img_size == 0

        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.classes = classes
        self.distribution = distribution
        self.num_classes = self.class_num = num_classes
        self.ignore_index = ignore_index
        self.download_url = download_url
        self.auto_download = auto_download
        
        self.distribution = distribution
        self.domain_shift = domain_shift
        self.i_split = i_split
        self.j_split = j_split
        self.sn7_aois = list(SN7_TRAIN) + list(SN7_VAL) + list(SN7_TEST)

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def load_planet_mosaic(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / 'train' / aoi_id / 'images_masked'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}.tif'
        with rasterio.open(str(file), mode='r') as src:
            img = src.read(out_shape=(self.sn7_img_size, self.sn7_img_size), resampling=rasterio.enums.Resampling.nearest)
        # 4th band (last oen) is alpha band
        img = img[:-1]
        return img.astype(np.float32)

    def load_building_label(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        folder = self.root_path / 'train' / aoi_id / 'labels_raster'
        file = folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.tif'
        with rasterio.open(str(file), mode='r') as src:
            label = src.read(out_shape=(self.sn7_img_size, self.sn7_img_size), resampling=rasterio.enums.Resampling.nearest)
        label = (label > 0).squeeze()
        return label.astype(np.int64)

    def load_change_label(self, aoi_id: str, year_t1: int, month_t1: int, year_t2: int, month_t2) -> np.ndarray:
        building_t1 = self.load_building_label(aoi_id, year_t1, month_t1)
        building_t2 = self.load_building_label(aoi_id, year_t2, month_t2)
        change = np.not_equal(building_t1, building_t2)
        return change.astype(np.int64)

    @staticmethod
    def get_band(path):
        return int(path.split('_')[-2])

    @staticmethod
    def download(self, silent=False):
        output_path = Path(self.root_path)

        if not output_path.exists():
            output_path.mkdir()
        else:
            if not silent:
                print("SpaceNet 7 Dataset folder exists, skipping downloading dataset.")
            return

        # download from Google Drive
        url = self.download_url
        tar_file = output_path / f'spacenet7.tar.gz'
        gdown.download(url, str(tar_file), quiet=False)

        try:
            with tarfile.open(tar_file, 'r:gz') as tar:
                print(f"Extracting to {output_path} ...")
                tar.extractall(path=output_path)
                print(f"Successfully extracted {tar_file} to {output_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

        temp_dataset_path = output_path / 'spacenet7'
        for item in temp_dataset_path.iterdir():
            destination = output_path / item.name
            try:
                shutil.move(str(item), str(destination))
                print(f"Moved {item} to {destination}")
            except Exception as e:
                print(f"Failed to move {item}: {e}")
        tar_file.unlink()
        # temp_dataset_path.unlink()


# @DATASET_REGISTRY.register()
class SN7MAPPING(AbstractSN7):
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
        domain_shift: bool,
        i_split: int,
        j_split: int,
    ):
        """Initialize the SpaceNet dataset for building mapping.
        """
        super().__init__(
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
            domain_shift=domain_shift,
            i_split=i_split,
            j_split=j_split,
        )

        self.split = split
        self.items = []

        if self.domain_shift:  # split by AOI ids
            if split == 'train':
                self.aoi_ids = list(SN7_TRAIN)
            elif split == 'val':
                self.aoi_ids = list(SN7_VAL)
            elif split == 'test':
                self.aoi_ids = list(SN7_TEST)
            else:
                raise Exception('Unkown split')

            # adding timestamps (only if label exists and not masked) for each AOI
            for aoi_id in self.aoi_ids:
                timestamps = list(self.metadata[aoi_id])
                for timestamp in timestamps:
                    if not timestamp['mask'] and timestamp['label']:
                        item = {
                            'aoi_id': timestamp['aoi_id'],
                            'year': timestamp['year'],
                            'month': timestamp['month'],
                        }
                        # tiling the timestamps
                        for i in range(0, self.sn7_img_size, self.img_size):
                            for j in range(0, self.sn7_img_size, self.img_size):
                                item['i'] = i
                                item['j'] = j
                                self.items.append(dict(item))
        
        else:  # within-scenes split
            assert self.i_split % self.img_size == 0 and self.j_split % self.img_size == 0
            assert self.img_size <= self.i_split and self.img_size <= self.j_split
            self.aoi_ids = list(self.sn7_aois)
            for aoi_id in self.aoi_ids:
                timestamps = list(self.metadata[aoi_id])
                for timestamp in timestamps:
                    if not timestamp['mask'] and timestamp['label']:
                        item = {
                            'aoi_id': timestamp['aoi_id'],
                            'year': timestamp['year'],
                            'month': timestamp['month'],
                        }
                        if split == 'train':
                            i_min, i_max = 0, self.i_split
                            j_min, j_max = 0, self.sn7_img_size
                        elif split == 'val':
                            i_min, i_max = self.i_split, self.sn7_img_size
                            j_min, j_max = 0, self.j_split
                        elif split == 'test':
                            i_min, i_max = self.i_split, self.sn7_img_size
                            j_min, j_max = self.j_split, self.sn7_img_size
                        else:
                            raise Exception('Unkown split')
                        # tiling the timestamps
                        for i in range(i_min, i_max, self.img_size):
                            for j in range(j_min, j_max, self.img_size):
                                item['i'] = i
                                item['j'] = j
                                self.items.append(dict(item))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        item = self.items[index]
        aoi_id, year, month = item['aoi_id'], int(item['year']), int(item['month'])
        filename = f"{aoi_id}_{year}_{month}_{item['i']}_{item['j']}"

        image = self.load_planet_mosaic(aoi_id, year, month)
        target = self.load_building_label(aoi_id, year, month)

        # cut to tile
        i, j = item['i'], item['j']
        image = image[:, i:i + self.img_size, j:j + self.img_size]
        target = target[i:i + self.img_size, j:j + self.img_size]

        image = torch.from_numpy(image)
        target = torch.from_numpy(target)
        # weight = torch.empty(target.shape)
        # for i, freq in enumerate(self.distribution):
        #     weight[target == i] = 1 - freq

        output = {
            'image': {
                'optical': image,
            },
            'target': target,
            # 'weight': weight,
            'metadata': {'filename': filename}
        }

        return output

class SN7CD(AbstractSN7):
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
        domain_shift: bool,
        i_split: int,
        j_split: int,
        dataset_multiplier: int,
        minimum_temporal_gap: int,
    ):
        """Initialize the SpaceNet dataset for change detection.

            ...
            eval_mode (bool): select if evaluation is happening. Instanciate true for val and test
            dataset_multiplier (int): multiplies sample in dataset during training.
        """
        super().__init__(
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
            domain_shift=domain_shift,
            i_split=i_split,
            j_split=j_split,
        )

        self.T = self.multi_temporal
        assert self.T > 1

        self.eval_mode = False if split == 'train' else True
        self.multiplier = 1 if self.eval_mode else dataset_multiplier
        self.min_gap = minimum_temporal_gap

        self.split = split
        self.items = []

        if self.domain_shift:  # split by AOI ids
            if split == 'train':
                self.aoi_ids = list(SN7_TRAIN)
            elif split == 'val':
                self.aoi_ids = list(SN7_VAL)
            elif split == 'test':
                self.aoi_ids = list(SN7_TEST)
            else:
                raise Exception('Unkown split')

            # adding timestamps (only if label exists and not masked) for each AOI
            for aoi_id in self.aoi_ids:
                item = { 'aoi_id': aoi_id }
                # tiling the timestamps
                for i in range(0, self.sn7_img_size, self.img_size):
                    for j in range(0, self.sn7_img_size, self.img_size):
                        item['i'] = i
                        item['j'] = j
                        self.items.append(dict(item))
        
        else:  # within-scenes split
            assert self.i_split % self.img_size == 0 and self.j_split % self.img_size == 0
            assert self.img_size <= self.i_split and self.img_size <= self.j_split
            self.aoi_ids = list(self.sn7_aois)
            for aoi_id in self.aoi_ids:
                item = { 'aoi_id': aoi_id }
                if split == 'train':
                    i_min, i_max = 0, self.i_split
                    j_min, j_max = 0, self.sn7_img_size
                elif split == 'val':
                    i_min, i_max = self.i_split, self.sn7_img_size
                    j_min, j_max = 0, self.j_split
                elif split == 'test':
                    i_min, i_max = self.i_split, self.sn7_img_size
                    j_min, j_max = self.j_split, self.sn7_img_size
                else:
                    raise Exception('Unkown split')
                # tiling the timestamps
                for i in range(i_min, i_max, self.img_size):
                    for j in range(j_min, j_max, self.img_size):
                        item['i'] = i
                        item['j'] = j
                        self.items.append(dict(item))

        self.items = self.multiplier * list(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        aoi_id = item['aoi_id']

        # determine timestamps for t1 and t2 (random for train and first-last for eval)
        timestamps = [ts for ts in self.metadata[aoi_id] if not ts['mask'] and ts['label']]
        if self.eval_mode:
            t_values = list(np.linspace(0, len(timestamps), self.T, endpoint=False, dtype=int))
        else:
            if self.T == 2:
                # t_values = [0, -1]
                t1 = np.random.randint(0, len(timestamps) - self.min_gap)
                t2 = np.random.randint(t1 + self.min_gap, len(timestamps))
                t_values = [t1, t2]
            else:  # randomly add intermediate timestamps
                t_values = [0] + sorted(np.random.randint(1, len(timestamps) - 1, size=self.T - 2)) + [-1]

        timestamps = sorted([timestamps[t] for t in t_values], key=lambda ts: int(ts['year']) * 12 + int(ts['month']))

        # load images according to timestamps
        image = np.stack([self.load_planet_mosaic(aoi_id, ts['year'], ts['month']) for ts in timestamps])

        # Reshaping tensor (T, C, H, W) to (C, T, H, W)
        image = image.transpose(1, 0, 2, 3)
        image = torch.from_numpy(image)

        # change label between first (0) and last (-1) timestamp
        year_t1, month_t1 = timestamps[0]['year'], timestamps[0]['month']
        year_t2, month_t2 = timestamps[-1]['year'], timestamps[-1]['month']
        filename = f"{aoi_id}_{year_t1}_{month_t1}_{year_t2}_{month_t2}_{item['i']}_{item['j']}"
        target = self.load_change_label(aoi_id, year_t1, month_t1, year_t2, month_t2)
        target = torch.from_numpy(target).long()

        # cut to tile
        i, j = item['i'], item['j']
        image = image[:, :, i:i + self.img_size, j:j + self.img_size]
        target = target[i:i + self.img_size, j:j + self.img_size]

        # weight for oversampling
        weight = torch.empty(target.shape)
        for i, freq in enumerate(self.distribution):
            weight[target == i] = 1 - freq

        output = {
            'image': {
                'optical': image,
            },
            'target': target,
            'weight': weight,
            'metadata': {'filename': filename}
        }

        return output

    # @staticmethod
    # def get_splits(dataset_config):
    #     dataset_train = SN7CD(cfg=dataset_config, split='train', eval_mode=False)
    #     dataset_val = SN7CD(cfg=dataset_config, split='val', eval_mode=True)
    #     dataset_test = SN7CD(cfg=dataset_config, split='test', eval_mode=True)
    #     return dataset_train, dataset_val, dataset_test

