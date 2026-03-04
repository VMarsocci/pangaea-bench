import torch
from pangaea.datasets.base import RawGeoFMDataset

import numpy as np
import h5py
from os.path import join, isfile
from tqdm import tqdm
import requests
import pathlib

REF_BIOMES = {
    20: 'Shrubs', 30: 'Herbaceous vegetation', 40: 'Cultivated', 90: 'Herbaceous wetland',
    111: 'Closed-ENL', 112: 'Closed-EBL', 114: 'Closed-DBL', 115: 'Closed-mixed',
    116: 'Closed-other', 121: 'Open-ENL', 122: 'Open-EBL', 124: 'Open-DBL',
    125: 'Open-mixed', 126: 'Open-other'
}

def download_from_zenodo(api_url, target_filenames, output_dir):
    """
    This function downloads specific files from a Zenodo record. It first retrieves
    the metadata for the record to find the download URLs for the specified files,
    and then downloads only those files.

    Args:
    - record_id (str): The Zenodo record ID (e.g., "18485030").
    - target_filenames (list of str): List of filenames to download from the record.
    - output_dir (str): Directory where the downloaded files should be saved.

    Returns:
    - None: The function saves the files to the specified output directory.
    """
    
    # Fetch metadata to find the specific download URLs
    response = requests.get(api_url)
    response.raise_for_status()
    files_in_record = response.json().get('files', [])
    to_download = {f['key']: f['links']['self'] for f in files_in_record if f['key'] in target_filenames}
    if not to_download:
        print("None of the specified files were found in this record.")
        return

    # Download the files
    for filename, download_url in to_download.items():
        file_path = join(output_dir, filename)
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(file_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=filename
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024): # 1MB chunks
                    _ = f.write(chunk)
                    _ = pbar.update(len(chunk))
    print(f"\nDone. Files saved to '{output_dir}'")


class AGBDLite(RawGeoFMDataset):
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
        target: str,
        eval_big: bool, 
        lite_chunk_size: int
    ):
        super(AGBDLite, self).__init__(
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

        assert split in ['train', 'val', 'test'], "split must be one of 'train', 'val', or 'test'"
        self.mode = split
        self.eval_big = eval_big
        self.target = target
        assert self.target == 'biome', "Only biome classification is currently supported for AGBD-Lite segmentation. Please set target to 'biome'."
        self.lite_chunk_size = lite_chunk_size
        self.patch_size = img_size
        self.zenodo_record = "18485030"
        if auto_download: self.download(self)

        self.s2_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        if self.eval_big and self.mode == 'test' : self.fname = 'AGBD-test.h5'
        else: self.fname = f'AGBD-Lite-{self.mode}.h5'
        self.f_handle = h5py.File(join(self.root_path, self.fname), 'r')

        with h5py.File(join(self.root_path, self.fname), 'r') as f:
            gedi_length = len(f['GEDI']['agbd'])
            total_length = (gedi_length // self.lite_chunk_size) + (1 if (gedi_length % self.lite_chunk_size != 0) else 0)
        self.gedi_length, self.length = gedi_length, total_length

        # Prepare to map the biome classes to 0-13 for Cross Entropy loss
        keys = torch.tensor(sorted(REF_BIOMES.keys()))
        self.biome_lookup = torch.full((max(keys.max(), 255) + 1,), -1)
        self.biome_lookup[keys] = torch.arange(len(keys))

    def __len__(self):
        # Return the total number of samples
        return int(self.length)

    def __getitem__(self, n):
        """Returns the i-th item of the dataset.

        Args:
            i (int): index of the item

        Raises:
            NotImplementedError: raise if the method is not implemented

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

        # Find the file, tile, and row index corresponding to this chunk
        idx_start = n * self.lite_chunk_size
        idx_end = min(idx_start + self.lite_chunk_size, self.gedi_length)   
        f = self.f_handle

        # Sentinel-2 bands ------------------------------------------------------------------------
        
        # Set the order and indices for the Sentinel-2 bands
        if not hasattr(self, 's2_order') : self.s2_order = list(f['S2_bands'].attrs['order'])
        if not hasattr(self, 's2_indices') : self.s2_indices = [self.s2_order.index(band) for band in self.s2_bands]

        # Get the bands
        s2_bands = f['S2_bands'][idx_start : idx_end, :, :, self.s2_indices].astype(np.float32)

        # Get the BOA offset, if it exists
        if 'S2_boa_offset' in f['Sentinel_metadata'].keys() : s2_boa_offset = f['Sentinel_metadata']['S2_boa_offset'][idx_start : idx_end].astype(np.float32)
        else: s2_boa_offset = np.full((s2_bands.shape[0],), 0, dtype = np.float32)
        s2_boa_offset = s2_boa_offset[:, np.newaxis, np.newaxis, np.newaxis]

        # Get the surface reflectance values
        sr_bands = (s2_bands - s2_boa_offset * 1000) / 10000
        sr_bands[s2_bands == 0] = 0
        sr_bands[sr_bands < 0] = 0

        # SAR bands (from ALOS-PALSAR-2) ----------------------------------------------------------

        # Set the order for the ALOS bands
        if not hasattr(self, 'alos_order') : self.alos_order = f['ALOS_bands'].attrs['order']

        # Get the bands as gamma naught values
        alos_bands = f['ALOS_bands'][idx_start : idx_end, :, :, :].astype(np.float32)
        alos_bands = np.where(alos_bands == 0, -9999.0, 10 * np.log10(np.power(alos_bands, 2)) - 83.0)

        # Target data -----------------------------------------------------------------------------
        lc = torch.from_numpy(np.array(f['LC'][idx_start : idx_end, :, :, 0])).long()
        target, _ = torch.mode(lc.flatten(start_dim=1), dim = 1)
        target = self.biome_lookup[target]
        
        # TEMPORARY
        placeholder_target = torch.full_like(lc, fill_value = self.ignore_index, dtype = torch.float if self.target in ['agbd', 'rh98'] else torch.long)
        placeholder_target[:, self.patch_size // 2, self.patch_size // 2] = target
        target = placeholder_target
        # TEMPORARY
        
        # Metadata (if needed) --------------------------------------------------------------------
        region = torch.from_numpy(np.array(f['GEDI']['region_cla'][idx_start : idx_end]))
        biome = lc[:, self.patch_size // 2, self.patch_size // 2]

        # Convert to tensors and return -----------------------------------------------------------
        sr_bands = torch.from_numpy(sr_bands).float()
        sr_bands = sr_bands.permute(0, 3, 1, 2).unsqueeze(2) # Change to (B, C, 1, H, W)
        alos_bands = torch.from_numpy(alos_bands).float()
        alos_bands = alos_bands.permute(0, 3, 1, 2).unsqueeze(2) # Change to (B, C, 1, H, W)

        # TEMPORARY
        sr_bands = sr_bands.squeeze(0)
        alos_bands = alos_bands.squeeze(0)
        target = target.squeeze(0)
        region = region.squeeze(0)
        biome = biome.squeeze(0)

        return {
            'image': {
                'optical': sr_bands,
                'sar': alos_bands
                },
            'target': target,
            'metadata': {
                'region': region,
                'biome': biome
            }
        }

    @staticmethod
    def download(self, silent=False):
        
        root_path = pathlib.Path(self.root_path)

        # Create the root directory if it does not exist
        if not root_path.exists(): root_path.mkdir(parents=True, exist_ok=True)
        if root_path.exists() :
            if self.eval_big and not isfile(join(root_path, 'AGBD-test.h5')) :
                print(f"AGBD-Lite test file does not exist at {root_path}. Downloading test file.")
                download_from_zenodo(self.download_url, ["AGBD-test.h5"], output_dir=root_path)
            if isfile(join(root_path, 'AGBD-Lite-train.h5')) and isfile(join(root_path, 'AGBD-Lite-val.h5')) and isfile(join(root_path, 'AGBD-Lite-test.h5'))  :
                if not silent:
                    print(f"AGBD-Lite files exist at {root_path}. Skipping download.")
                return

        # Download the files from https://zenodo.org/records/18485030 
        fnames = ["AGBD-Lite-train.h5", "AGBD-Lite-val.h5", "AGBD-Lite-test.h5"]
        fnames += ["AGBD-test.h5"] if self.eval_big else []
        download_from_zenodo(self.download_url, fnames, output_dir=root_path)