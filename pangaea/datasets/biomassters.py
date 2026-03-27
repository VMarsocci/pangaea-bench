import numpy as np
import torch
import pandas as pd
import pathlib
import rasterio
from tifffile import imread
from os.path import join as opj

from pangaea.datasets.utils import read_tif
from pangaea.datasets.base import RawGeoFMDataset

def read_imgs(multi_temporal, temp , fname, data_dir, img_size):
    imgs_s1, imgs_s2, mask = [], [], []
    if multi_temporal==1:
        month_list = [temp]        
    else:        
        month_list = list(range(int(multi_temporal)))
    
    for month in month_list:
        
        s1_fname = '%s_%s_%02d.tif' % (str.split(fname, '_')[0], 'S1', month)
        s2_fname = '%s_%s_%02d.tif' % (str.split(fname, '_')[0], 'S2', month)

        s1_filepath = data_dir.joinpath(s1_fname)
        if s1_filepath.exists():
            img_s1 = imread(s1_filepath)
            m = img_s1 == -9999
            img_s1 = img_s1.astype('float32')
            img_s1 = np.where(m, 0, img_s1)
        else:            
            img_s1 = np.zeros((img_size, img_size) + (4,), dtype='float32')
        
        s2_filepath = data_dir.joinpath(s2_fname)
        if s2_filepath.exists():
            img_s2 = imread(s2_filepath)
            img_s2 = img_s2.astype('float32')
        else:            
            img_s2 = np.zeros((img_size, img_size) + (11,), dtype='float32')
        
        img_s1 = np.transpose(img_s1, (2, 0, 1))
        img_s2 = np.transpose(img_s2, (2, 0, 1))
        imgs_s1.append(img_s1)
        imgs_s2.append(img_s2)
        mask.append(False)

    mask = np.array(mask)

    imgs_s1 = np.stack(imgs_s1, axis=1)
    imgs_s2 = np.stack(imgs_s2, axis=1)
    return imgs_s1, imgs_s2, mask

class BioMassters(RawGeoFMDataset):
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
        temp: int,
    ):
        """Initialize the BioMassters dataset.
        Link: https://huggingface.co/datasets/nascetti-a/BioMassters

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
            temp (int): which temporal frame to use when using single temporal.
        """
        super(BioMassters, self).__init__(
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
        self.multi_temporal = multi_temporal
        self.temp = temp
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
        
        self.data_path = pathlib.Path(self.root_path).joinpath(f"{split}_Data_list.csv")
        self.id_list = pd.read_csv(self.data_path)['chip_id']
        
        self.split_path = 'train' if split == 'val' else split
        self.dir_features = pathlib.Path(self.root_path).joinpath(f'{self.split_path}_features')
        self.dir_labels = pathlib.Path(self.root_path).joinpath( f'{self.split_path}_agbm')

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):

        chip_id = self.id_list.iloc[index]
        fname = str(chip_id)+'_agbm.tif'
        
        imgs_s1, imgs_s2, mask = read_imgs(self.multi_temporal, self.temp, fname, self.dir_features, self.img_size)
        with rasterio.open(self.dir_labels.joinpath(fname)) as lbl:
            target = lbl.read(1)
        target = np.nan_to_num(target)

        imgs_s1 = torch.from_numpy(imgs_s1).float()
        imgs_s2 = torch.from_numpy(imgs_s2).float()
        target = torch.from_numpy(target).float()

        return {
            'image': {
                    'optical': imgs_s2,
                    'sar' : imgs_s1,
                    },
            'target': target,  
            'metadata': {'masks':mask, 'filename': fname}
        }

    # @staticmethod
    # def download(self, silent=False):
    #     pass
