import os
import torch
import numpy as np
import pandas as pd
import rasterio as rio
from pangaea.datasets.base import RawGeoFMDataset
from skimage.morphology import binary_erosion, disk
from typing import Tuple, Union, List, Dict
import json

class CHCDataset(RawGeoFMDataset):
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
        img_merging: str = 'first',
        min_height: float = 3.0,
        masking: list[str] = ['veg'],
        weighting: str = 'none',
        confidence_z_value: float = 1.645,
        oversampling_factor: float = 1.0,
        rm_vegetation_edges: bool = True,
        standardized_test: bool = True
    ):
        '''
        Dataset for canopy height change regression

        This dataset is used for change regression in canopy height models (CHM) from 2018 to 2023, using a PlanetScope
        time series of 6 years as input. The dataset provides pre-computed CHM differences as targets, together with
        the target variance. Differences are zero in no-vegetation or invalid areas, all of which can be masked out
        by the vegetation and validity masks, respectively.

        Requires a CSV file with the following columns:
            - sample_loc_id: unique identifier for the sample location
            - year: year of the sample (e.g., 2019, 2020)
            - split: split of the sample (e.g., 'train', 'val', 'test')
            - fname_planetscope: filename of the image for the sample location and year
        For each location min. 2 years must be available.

        Args:
            img_merging (str): how to merge images for each year if multiple are available, options are 'first', 'random', 'avg'.
            min_height (float): minimum height of vegetation to be considered as vegetation in CHM.
            masking (list): list of masks to apply, options are 'veg' for vegetation mask and 'z_value' for confidence-based masking.
            weighting (str): weighting scheme for valid pixels, options are 'none' for no weighting, 'var_inv' for inverse of the variance, 'z_cutoff' for binary weights based on the confidence_z_value.
            confidence_z_value (float): z-value for change confidence to consider a change as valid. Used for masking and weighting based on change confidence.
            oversampling_factor (float): factor to oversample locations with high change density.
            rm_vegetation_edges (bool): whether to remove vegetation edges based on morphological filtering to reduce edge effects in the CHM.
            standardized_test (bool): whether to apply a standardized masking and weighting scheme for the test set to ensure comparability of results. If True, the same masking and weighting will be applied to all test samples, regardless of the masking and weighting arguments provided.
        '''
        super(CHCDataset, self).__init__(
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
        assert multi_temporal == 6
        assert all(m in ['veg', 'z_value'] for m in masking)
        assert weighting in ['none', 'var_inv', 'z_cutoff']
        assert img_merging in ['first', 'random', 'avg']

        self.img_merging = img_merging
        self.patch_size = 1024 # size of CHC patches, change if other image resolution used (e.g., Sentinel-2, 10m)
        self.nodata = -1
        self.confidence_z_value = confidence_z_value
        self.rm_vegetation_edges = rm_vegetation_edges
        self.chm_dir = 'chm_diff'
        self.img_dir = 'planetscope'

        if standardized_test and split == 'test':
            masking = ['veg', 'z_value']
            weighting = 'none'
        self.masking = masking
        self.weighting = weighting

        # CHM settings
        self.min_height = min_height

        # load index file
        self.index_df = pd.read_csv(os.path.join(self.root_path, 'index.csv'))
        self.index_df = self.index_df[self.index_df.groupby('sample_loc_id')['year'].transform('nunique') > 1]
        if split is not None:
            self.index_df = self.index_df[self.index_df['split'] == split]

        self.items = np.sort(self.index_df.sample_loc_id.unique())

        if (oversampling_factor > 1.0) and (split == 'train'):
            # oversample locations with high change density
            if not 'density' in self.index_df.columns:
                densities = np.ones_like(self.items, dtype=np.float32)
            else:
                densities = self.index_df.groupby('sample_loc_id')['density'].first().loc[self.items].values
            sampling_probs = densities / densities.sum()
            n_additional = int(len(self.items) * (oversampling_factor - 1.0))
            additional_items = np.random.choice(self.items, size=n_additional, replace=True, p=sampling_probs)
            self.items = np.concatenate([self.items, additional_items])
            self.items = np.sort(self.items)

        assert 'year' in self.index_df.columns
        assert 'sample_loc_id' in self.index_df.columns
        assert 'fname_planetscope' in self.index_df.columns

    def __len__(self):
        # Return the total number of samples
        return len(self.items)

    def _get_samples(self, index:int, img_selection:Union[str, None]=None) -> Tuple[List[List[str]], str]:
        loc_idx = self.items[index]
        yrs_at_loc = self.index_df[self.index_df['sample_loc_id'] == loc_idx].sort_values('year')['year'].unique().tolist()[:self.multi_temporal]
        if len(yrs_at_loc) < 2:
            raise ValueError(f"Sample location {loc_idx} has only one year of data: {yrs_at_loc[0]}.")
        sample_df = self.index_df[(self.index_df['sample_loc_id'] == loc_idx) & (self.index_df['year'].isin(yrs_at_loc))]
        
        if img_selection == 'first':
            sample_df = sample_df.groupby('year').first().reset_index()
        elif img_selection == 'random':
            sample_df = sample_df.groupby('year').sample(n=1, random_state=None).reset_index(drop=True)
        elif img_selection == 'avg':
            pass # averaging will be done in _get_img anyways

        # fill gaps if less years are available than multi_temporal
        if len(yrs_at_loc) < self.multi_temporal:
            if yrs_at_loc[-1] - yrs_at_loc[0] + 1 > self.multi_temporal:
                yrs_at_loc = list(yrs_at_loc) + [None] * (self.multi_temporal - len(yrs_at_loc))
            else:
                full_yrs = np.arange(yrs_at_loc[0], yrs_at_loc[0] + self.multi_temporal)
                yrs_at_loc = [yr if yr in yrs_at_loc else None for yr in full_yrs]

        fnames_img = []
        for yr in yrs_at_loc:
            if yr is not None:
                fnames_img.append(sample_df[sample_df['year'] == yr]['fname_planetscope'].to_list())
            else:
                fnames_img.append([]) # None

        fname_diff = f'{str(loc_idx)}.tif'

        return fnames_img, fname_diff

    def _get_raster(self, fname:str, dir) -> Tuple[np.ma.MaskedArray, dict]:
        """Load a raster file.
        
        Resamples the raster to the patch size.
        """
        with rio.open(os.path.join(self.root_path, dir, fname)) as src:
            data = src.read(masked=True)
            meta = src.meta

        assert data.shape == (meta['count'], self.patch_size, self.patch_size)
        if meta['crs'] is not None:
            meta['crs'] = meta['crs'].to_wkt()
        meta['dtype'] = 'float32'
        return data.astype(np.float32), meta

    def _get_img(self, fnames:List[List[str]]) -> tuple[np.ma.MaskedArray, List[dict]]:
        '''
        Load and preprocess images for two years. If several images are available for a year,
        they are averaged.
        '''

        imgs = []
        metas = []
        for year_fnames in fnames:
            if len(year_fnames) == 0:
                # create empty image
                img = np.ma.masked_all((self.img_size, self.img_size, len(self.bands['optical'])), dtype=np.float32)
                meta = {
                    'count': len(self.bands['optical']),
                    'dtype': 'float32',
                    'crs': None,
                    'transform': None,
                    'width': self.img_size,
                    'height': self.img_size,
                    'nodata': -1
                }
            else:
                img_list = []
                for f in year_fnames:
                    rst, meta = self._get_raster(f, self.img_dir)
                    img_list.append(rst.transpose(1, 2, 0)) # (H, W, C) format
                img = np.stack(img_list, axis=0) # (N, H, W, C) format
                img = img.mean(axis=0) # (H, W, C) format

            imgs.append(img)
            metas.append(meta)

        imgs = np.ma.stack(imgs) # (T, H, W, C) format

        return imgs, metas
    
    def _filter_veg_edges(self, veg_mask: np.ndarray) -> np.ndarray:
        """Filter vegetation mask to remove edge effects.

        Args:
            veg_mask (np.ndarray): Input vegetation mask.

        Returns:
            np.ndarray: Filtered vegetation mask.
        """
        veg_mask = veg_mask.astype(bool)
        veg_mask = binary_erosion(veg_mask, footprint=disk(1)[np.newaxis, :, :])
        return veg_mask

    def __getitem__(self, index:int) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor, Dict[str, Union[int, str, List[List[str]], torch.Tensor]]]]:
        """Returns the i-th item of the dataset.

        Args:
            i (int): index of the item

        Returns:
            dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary follwing the format
            {"image":
                {
                "optical": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),
                    "sar": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),
                    },
            "target": torch.Tensor of shape (H W) of type torch.int64 for segmentation, torch.float for
            regression datasets.,
                "metadata": dict}.
        """
        fnames_img, fname_diff = self._get_samples(index, img_selection=self.img_merging)

        # create image stack
        image, metas = self._get_img(fnames_img)
        image = image.transpose(3, 0, 1, 2) # (C T H W) format
        image = image.filled(self.nodata)
        image = torch.from_numpy(image).float()

        # get change (0 in no-vegetation areas, NaN in invalid areas)
        diff, meta_diff = self._get_raster(fname_diff, self.chm_dir) # (3, H, W), channels: diff, h_max, var
        diff = diff.filled(np.nan)
        valid_mask = np.isfinite(diff).all(axis=0, keepdims=True)
        chm_diff = diff[0:1] # keep dims
        max_height = diff[1:2]
        var_diff = diff[2:3]

        # mask-out changes in low vegetation areas
        vegetation_mask = (max_height >= self.min_height)
        if self.rm_vegetation_edges:
            vegetation_mask = self._filter_veg_edges(vegetation_mask)

        # masking
        mask = (chm_diff > -30) & (chm_diff < 10) & valid_mask  # keep changes within [-30, 10] range
        if 'veg' in self.masking:
            mask = mask & vegetation_mask
        if 'z_value' in self.masking:
            mask = mask & (np.abs(chm_diff) >= self.confidence_z_value * np.sqrt(var_diff)) # keep changes above confidence threshold

        # weights
        if self.weighting == 'var_inv':
            weights = np.where(var_diff > 0, 1 / var_diff, 0.)
        elif self.weighting == 'z_cutoff':
            weights = np.where(np.abs(chm_diff) >= self.confidence_z_value * np.sqrt(var_diff), 1.0, 0.0)
        else:
            weights = np.ones_like(var_diff)
        
        chm_diff = np.where(mask, chm_diff, 0.0)
        var_diff = np.where(mask, var_diff, 0.0)
        weights = np.where(mask, weights, 0.0)

        assert chm_diff.shape == (1, self.patch_size, self.patch_size)
        assert var_diff.shape == (1, self.patch_size, self.patch_size)
        assert mask.shape == (1, self.patch_size, self.patch_size)
        assert vegetation_mask.shape == (1, self.patch_size, self.patch_size)

        return_dict = {
            'image': {'optical': image}, # (C, T, H, W)
            'target': torch.from_numpy(chm_diff).float().squeeze(0), # (H, W),
            'weights': torch.from_numpy(weights).float().squeeze(0), # (H, W)
            'metadata': {
                'sample_loc_id': self.items[index],
                'var_diff': torch.from_numpy(var_diff).float().squeeze(0), # (H, W)
                'mask': torch.from_numpy(mask), # (1, H, W) boolean valid data mask
                'vegetation_mask': torch.from_numpy(vegetation_mask), # (1, H, W) boolean vegetation mask
                'meta_planetscope': json.dumps(metas), # prevent recursive collating
                'meta_diff': json.dumps(meta_diff),
                'fnames_planetscope': json.dumps(fnames_img),
                'fnames_diff': json.dumps(fname_diff),
            }
        }

        return return_dict

    @staticmethod
    def download(silent=False):
        # Implement if your dataset requires downloading
        pass