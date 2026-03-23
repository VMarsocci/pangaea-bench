# Adapted from https://github.com/sustainlab-group/sustainbench/blob/main/sustainbench/datasets/croptypemapping_dataset.py

import os
import json
import gdown
import tarfile
import zipfile
import shutil

import numpy as np
import pandas as pd

import torch

from pangaea.datasets.base import RawGeoFMDataset
# from utils.registry import DATASET_REGISTRY


# @DATASET_REGISTRY.register()
class CropTypeMappingSouthSudan(RawGeoFMDataset):
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
        use_pad: bool,
    ):
        """Initialize the CropTypeMappingSouthSudan dataset.
        Link: https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg2/crop_type_mapping_ghana-ss.html#download

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
            use_pad (bool): wheter to pad or not the images.
        """
        super(CropTypeMappingSouthSudan, self).__init__(
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
            # use_pad=use_pad
        )

        self.root_path = root_path
        self.classes = classes
        self.split = split
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}
        self.split_mapping = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}

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

        self.country = 'southsudan'
        self.use_pad = use_pad
        self.grid_size = multi_temporal

        split_df = pd.read_csv(os.path.join(self.root_path, self.country, 'list_eval_partition.csv'))
        self.split_array = split_df['partition'].values

        split_mask = self.split_array == self.split_dict[split]
        self.split_indices = np.where(split_mask)[0]
        self.ori_ids = torch.from_numpy(split_df['id'].values)

    
    def __getitem__(self, idx):
        id = self.split_indices[idx]
        loc_id = f'{self.ori_ids[id]:06d}'

        file_path = os.path.join(self.root_path, self.country, 'npy', f'{self.country}_{loc_id}.npz')
        try:
            images = np.load(file_path)
            s1 = torch.from_numpy(images['s1'])[:2, ...].float()   # only use VV and VH bands
            s2 = torch.from_numpy(images['s2']).float() 

            s1  = torch.flip(s1, dims=[3])  # flip the time dimension
            s2 = torch.flip(s2, dims=[3])  

            if self.use_pad:
                s1 = self.pad_or_crop(s1)
                s2 = self.pad_or_crop(s2)


            s1 = torch.permute(s1, (0, 3, 1, 2))  # C, T, H, W
            s2 = torch.permute(s2, (0, 3, 1, 2))  # C, T, H, W


            label = np.load(os.path.join(self.root_path, self.country, 'truth', f'{self.country}_{loc_id}.npz'))['truth']
            label = self._mapping_label(label)
            label = torch.from_numpy(label).long()

            metadata = self.get_metadata(idx)
            
            output = {
                'image': {
                    'optical': s2,
                    'sar': s1
                },
                'target': label,
                'metadata': metadata
            }

            return output
        
        except zipfile.BadZipFile:
            print(f"BadZipFile: {file_path}. This file is skipped.")
            return self.__getitem__(idx + 1)
            
        
    def _mapping_label(self, label):
        # The dataset uses top four crop types 
        # Reference: https://openaccess.thecvf.com/content_CVPRW_2019/papers/cv4gc/Rustowicz_Semantic_Segmentation_of_Crop_Type_in_Africa_A_Novel_Dataset_CVPRW_2019_paper.pdf

        transformed_label = np.full_like(label, -1)
        mapping = {1: 0, 2: 1, 3: 2, 4: 3}

        for k, v in mapping.items():
            transformed_label[label == k] = v

        return transformed_label
    
    def __len__(self):
        return len(self.split_indices)

    def get_dates(self, json_file):
        """
        Converts json dates into tensor containing dates
        """
        dates = np.array(json_file['dates'])
        dates = np.char.replace(dates, '-', '')
        dates = torch.from_numpy(dates.astype(np.int32))
        return dates
    
    def get_metadata(self, idx):
        """
        Returns metadata for a given idx.
        Dates are returned as integers in format {Year}{Month}{Day}
        """
        id = self.split_indices[idx]
        loc_id = f'{self.ori_ids[id]:06d}'

        s1_json = json.loads(open(os.path.join(self.root_path, self.country, 's1', f's1_{self.country}_{loc_id}.json'), 'r').read())
        s1 = self.get_dates(s1_json)

        s2_json = json.loads(open(os.path.join(self.root_path, self.country, 's2', f's2_{self.country}_{loc_id}.json'), 'r').read())
        s2 = self.get_dates(s2_json)

        s1 = torch.flip(s1, dims=[0])  # flip the time dimension
        s2 = torch.flip(s2, dims=[0])
    
        if self.use_pad:
            s1 = self.pad_or_crop(s1)
            s2 = self.pad_or_crop(s2)

        filename = f'{self.country}_{loc_id}.npz'
        return {'sar': s1, 'optical': s2, 'filename': filename}
    
    def pad_or_crop(self, tensor):
        '''
        Right pads or crops tensor to GRID_SIZE.
        '''
        # if self.grid_size >= tensor.shape[-1]:
        pad_size = self.grid_size - tensor.shape[-1]
        tensor = torch.nn.functional.pad(input=tensor, pad=(0, pad_size), value=0)
        # else:
        #     tensor = tensor[..., :self.grid_size]
        return tensor

    @staticmethod
    def download(self, silent=False):
        if os.path.exists(self.root_path):
            if not silent:
                print("CropTypeMapping Dataset folder exists, skipping downloading dataset.")
            return

        output_path = self.root_path
        os.makedirs(output_path, exist_ok=True)
        url = self.download_url

        temp_file = os.path.join(output_path, "archive.tar.gz")

        try:
            gdown.download_folder(url=url,output=output_path, quiet=False, use_cookies=False)
        except Exception as e:
                print(f'{temp_file} may be corrupted. Please try deleting it and rerunning.\n')
                print(f"Exception: ", e)

        with tarfile.open(temp_file, 'r:gz') as tar:
            print(f"Extracting {temp_file}...")
            tar.extractall(path=output_path)
        
        os.remove(temp_file)
        for item in os.listdir(os.path.join(output_path, 'africa_crop_type_mapping_v1.0')):
            shutil.move(os.path.join(output_path, 'africa_crop_type_mapping_v1.0', item), os.path.join(output_path, item))
        os.rmdir(os.path.join(output_path, 'africa_crop_type_mapping_v1.0'))

