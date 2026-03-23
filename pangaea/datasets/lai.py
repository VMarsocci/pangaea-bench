import os
import numpy as np
import torch
import pandas as pd
import pathlib
from tifffile import imread
import rasterio
from os.path import join as opj
from typing import Dict, List, Optional

from pangaea.datasets.base import RawGeoFMDataset
from pangaea.datasets.utils import read_tif


class LAI(RawGeoFMDataset):
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
        modalities: Optional[List[str]] = None,
    ):
        """Initialize the LAI dataset.
        
        Dataset structure (after split):
        root_path/
        ├── train/
        │   ├── FAPAR/
        │   │   ├── FAPAR_patch_256px_x000_y000.tif
        │   │   └── ...
        │   ├── FCOVER/
        │   ├── LAI/
        │   ├── SENTINEL2/
        │   └── SENTINEL1/
        ├── val/
        │   ├── FAPAR/
        │   ├── FCOVER/
        │   ├── LAI/
        │   ├── SENTINEL2/
        │   └── SENTINEL1/
        └── test/
            ├── FAPAR/
            ├── FCOVER/
            ├── LAI/
            ├── SENTINEL2/
            └── SENTINEL1/

        Args:
            split (str): split of the dataset (train, val, test).
            dataset_name (str): dataset name.
            multi_modal (bool): if the dataset is multi-modal.
            multi_temporal (int): number of temporal frames.
            root_path (str): root path of the dataset.
            classes (list): classes of the dataset (target modalities to predict).
            num_classes (int): number of classes.
            ignore_index (int): index to ignore for metrics and loss.
            img_size (int): size of the image. 
            bands (dict[str, list[str]]): bands of the dataset.
            distribution (list[int]): class distribution.
            data_mean (dict[str, list[str]]): mean for each band for each modality. 
            Dictionary with keys as the modality and values as the list of means.
            e.g. {"sentinel2": [b1_mean, ..., bn_mean], "sentinel1": [b1_mean, ..., bn_mean]}
            data_std (dict[str, list[str]]): std for each band for each modality.
            Dictionary with keys as the modality and values as the list of stds.
            e.g. {"sentinel2": [b1_std, ..., bn_std], "sentinel1": [b1_std, ..., bn_std]}
            data_min (dict[str, list[str]]): min for each band for each modality.
            Dictionary with keys as the modality and values as the list of mins.
            e.g. {"sentinel2": [b1_min, ..., bn_min], "sentinel1": [b1_min, ..., bn_min]}
            data_max (dict[str, list[str]]): max for each band for each modality.
            Dictionary with keys as the modality and values as the list of maxs.
            e.g. {"sentinel2": [b1_max, ..., bn_max], "sentinel1": [b1_max, ..., bn_max]}
            download_url (str): url to download the dataset.
            auto_download (bool): whether to download the dataset automatically.
            modalities (Optional[List[str]]): list of input modalities to use. Default: ['SENTINEL1', 'SENTINEL2'].
        """
        super(LAI, self).__init__(
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
        
        # Dataset-specific attributes
        self.root_path = pathlib.Path(root_path)
        self.split = split
        self.multi_temporal = multi_temporal
        self.img_size = img_size
        
        # Default input modalities if not specified
        if modalities is None:
            self.modalities = ['SENTINEL2', 'SENTINEL1']
        else:
            self.modalities = modalities
            
        # Target modalities based on classes parameter
        target_mapping = {
            'FAPAR': 'FAPAR',
            'FCOVER': 'FCOVER', 
            'LAI': 'LAI',
            'fapar': 'FAPAR',
            'fcover': 'FCOVER',
            'lai': 'LAI'
        }
        
        self.target_modalities = []
        for class_name in classes:
            if class_name in target_mapping:
                self.target_modalities.append(target_mapping[class_name])
            else:
                self.target_modalities.append(class_name)
        
        if len(self.target_modalities) == 0:
            self.target_modalities = ['FAPAR', 'FCOVER', 'LAI']
        
        # Check if dataset has split structure (train/val/test folders)
        has_split_structure = (self.root_path / 'train').exists()
        
        if has_split_structure:
            # New structure: root_path/train/SENTINEL2/...
            split_root = self.root_path / self.split
            if not split_root.exists():
                raise FileNotFoundError(f"Split directory not found: {split_root}")
        else:
            # Old structure: root_path/SENTINEL2/...
            split_root = self.root_path
        
        # Initialize paths for each input modality
        self.modality_paths = {}
        for modality in self.modalities:
            self.modality_paths[modality] = split_root / modality
            
        # Initialize paths for target modalities
        self.target_paths = {}
        for modality in self.target_modalities:
            self.target_paths[modality] = split_root / modality
        
        # Get list of patch IDs by scanning SENTINEL2 folder
        primary_modality = 'SENTINEL2'
        primary_path = self.modality_paths[primary_modality]
        
        if not primary_path.exists():
            raise FileNotFoundError(f"Primary modality path does not exist: {primary_path}")
        
        # Extract patch coordinates from filenames
        self.patch_ids = []
        for file_path in primary_path.glob(f"{primary_modality}_*.tif"):
            filename = file_path.stem
            parts = filename.split('_')
            if len(parts) >= 4:
                x_part = parts[-2]
                y_part = parts[-1]
                patch_id = f"{x_part}_{y_part}"
                self.patch_ids.append(patch_id)
            else:
                patch_id = filename.replace(f"{primary_modality}_", "")
                self.patch_ids.append(patch_id)
        
        self.patch_ids.sort()
        
        if len(self.patch_ids) == 0:
            raise ValueError(f"No patches found in {primary_path}")
        
        print(f"Loaded {len(self.patch_ids)} patches for {self.split} split")

    def _load_modality_image(self, modality: str, patch_id: str) -> np.ndarray:
        """Load a single modality image for a given patch ID.
        
        Args:
            modality (str): Name of the modality (e.g., 'SENTINEL2', 'SENTINEL1')
            patch_id (str): Patch identifier (e.g., 'x000_y000')
            
        Returns:
            np.ndarray: Image array with shape (channels, height, width)
        """
        # if modality == 'SENTINEL1':
        #     filename = f"S1_patch_256px_{patch_id}.tif"
        # else:
        filename = f"{modality}_patch_256px_{patch_id}.tif"
            
        filepath = self.modality_paths[modality] / filename
        
        if not filepath.exists():
            print(f"Warning: File not found {filepath}, filling with zeros")
            default_channels = 1
            return np.zeros((default_channels, self.img_size, self.img_size), dtype=np.float32)
        
        try:
            with rasterio.open(filepath) as src:
                image = src.read()
                image = image.astype(np.float32)
                image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
                return image
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            try:
                image = imread(filepath)
                if len(image.shape) == 2:
                    image = image[np.newaxis, :, :]
                elif len(image.shape) == 3:
                    image = np.transpose(image, (2, 0, 1))
                
                image = image.astype(np.float32)
                image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
                return image
            except Exception as e2:
                print(f"Error with fallback loading {filepath}: {e2}")
                return np.zeros((1, self.img_size, self.img_size), dtype=np.float32)

    def _load_target_image(self, modality: str, patch_id: str) -> np.ndarray:
        """Load a target vegetation index image for a given patch ID.
        
        Args:
            modality (str): Name of the target modality (e.g., 'FAPAR', 'FCOVER', 'LAI')
            patch_id (str): Patch identifier (e.g., 'x000_y000')
            
        Returns:
            np.ndarray: Target image array with shape (1, height, width)
        """
        filename = f"{modality}_patch_256px_{patch_id}.tif"
        filepath = self.target_paths[modality] / filename
        
        if not filepath.exists():
            print(f"Warning: Target file not found {filepath}, filling with zeros")
            return np.zeros((1, self.img_size, self.img_size), dtype=np.float32)
        
        try:
            with rasterio.open(filepath) as src:
                image = src.read()
                if len(image.shape) == 2:
                    image = image[np.newaxis, :, :]
                image = image.astype(np.float32)
                image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
                return image
                
        except Exception as e:
            print(f"Error loading target {filepath}: {e}")
            return np.zeros((1, self.img_size, self.img_size), dtype=np.float32)

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.patch_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Returns the index-th item of the dataset.
        
        Args:
            index (int): Index of the patch to retrieve
            
        Returns:
            dict: Dictionary containing:
                - 'image': Dict with modality names as keys and image tensors as values
                - 'target': Target tensor
                - 'metadata': Dict with additional metadata
        """
        patch_id = self.patch_ids[index]
        
        # Load input modalities
        modality_images = {}
        
        for modality in self.modalities:
            image = self._load_modality_image(modality, patch_id)
            image_tensor = torch.from_numpy(image).float()
            image_tensor = image_tensor.unsqueeze(1)  # Add temporal dimension
            
            # Map to standard framework keys
            if modality == 'SENTINEL2':
                modality_images['optical'] = image_tensor
            elif modality == 'SENTINEL1':
                modality_images['sar'] = image_tensor
        
        # Load target vegetation indices
        target_images = []
        for target_modality in self.target_modalities:
            target_img = self._load_target_image(target_modality, patch_id)
            target_images.append(target_img[0])
        
        # Stack targets
        if len(target_images) == 1:
            target = target_images[0]
        else:
            target = np.stack(target_images, axis=0)
            
        target = torch.from_numpy(target).float()
        
        # Metadata
        primary_modality = self.modalities[0]
        filename = f"{primary_modality}_patch_256px_{patch_id}.tif"
        metadata = {
            'filename': filename,
            'patch_id': patch_id,
            'input_modalities': self.modalities,
            'target_modalities': self.target_modalities,
            'masks': np.array([False] * len(self.modalities))
        }
        
        return {
            'image': modality_images,
            'target': target,
            'metadata': metadata
        }

    @staticmethod
    def download(self, silent=False):
        """Download method - implement if needed for automatic dataset download."""
        if not silent:
            print("Download method not implemented. Please manually place data in the specified root_path.")
        pass