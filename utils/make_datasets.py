# -*- coding: utf-8 -*-
''' 
Authors: Yuru Jia, Valerio Marsocci
'''

import os

from datasets.mados import MADOS
from datasets.croptypemapping import CropTypeMappingDataset
from datasets.sen1floods11 import Sen1Floods11
from datasets.hlsburnscars import BurnScarsDataset
from datasets.xView2 import xView2

def make_dataset(ds_name, path, **kwargs):
    datasets = {
        "mados": MADOS,
        "crop_type_mapping": CropTypeMappingDataset,
        "sen1floods11": Sen1Floods11,
        "hlsburnscars": BurnScarsDataset,      
        "xView2": xView2,
    }

    if ds_name not in datasets:
        raise ValueError(f"{ds_name} is not yet supported.")
    
    if ds_name == "mados":
        dataset_train = MADOS(path, mode="train")
        dataset_val = MADOS(path, mode="val")
        dataset_test = MADOS(path, mode="test")
    elif ds_name == "crop_type_mapping":
        dataset = CropTypeMappingDataset(data_dir=path, split_scheme='southsudan', calculate_bands=False, normalize=True)
        dataset_train = dataset.get_subset('train')
        dataset_val = dataset.get_subset('val')
        dataset_test = dataset.get_subset('test')
    elif ds_name == "sen1floods11":
        dataset_train = Sen1Floods11(data_root=path, split="train")
        dataset_val = Sen1Floods11(data_root=path, split="val")
        dataset_test = Sen1Floods11(data_root=path, split="test")
    elif ds_name == "hlsburnscars":
        dataset_train = BurnScarsDataset(data_root=path, split="training")
        dataset_val = BurnScarsDataset(data_root=path, split="validation")
        dataset_test = dataset_val
    elif ds_name == "xView2":
        dataset_train = xView2(data_root=path, split="train")
        dataset_val = xView2(data_root=path, split="val")
        dataset_test = xView2(data_root=path, split="test")
    return dataset_train, dataset_val, dataset_test
