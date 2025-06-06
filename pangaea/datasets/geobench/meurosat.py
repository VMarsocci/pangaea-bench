import os
import numpy as np
import torch
from pangaea.datasets.base import RawGeoFMDataset
from pangaea.datasets.utils import decompress_zip_with_progress
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import subprocess
import sys
try:
    import geobench
except ImportError:
    print("geobench not found. Installing via pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "geobench"])
    import geobench


class mEuroSat(torch.utils.data.Dataset):
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
        """Initialize the mEuroSat dataset.
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
        super(mEuroSat, self).__init__()
        self.split = split
        self.dataset_name = dataset_name
        self.multi_modal = multi_modal
        self.multi_temporal = multi_temporal
        self.root_path = root_path
        self.classes = classes
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.img_size = img_size
        self.bands = bands
        self.distribution = distribution
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.download_url = download_url
        self.auto_download = auto_download

        if not os.path.exists(self.root_path):
            self.download(self)



        
        split_mapping = {'train': 'train', 'val': 'valid', 'test': 'test'}
        task = geobench.load_task_specs(self.root_path)
        self.dataset = task.get_dataset(split=split_mapping[self.split])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        all_band_names = (
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "08A",
            "09",
            "10",
            "11",
            "12",
        )
        rgb_bands = ("04", "03", "02")

        BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}
        image, band_names = sample.pack_to_3d(band_names=BAND_SETS["all"])
        label = sample.label
        filename = sample.sample_name
        
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() 
        
        # image=image.unsqueeze(1)

        return {
            "image": image,
            "target": torch.tensor(label, dtype=torch.int64),
            # "metadata": {
            #     "filename": filename},
        }
        
    def download(self, silent=False):
        local_directory = Path(os.getenv("GEO_BENCH_DIR"))
        dataset_repo = self.download_url

        local_directory.mkdir(parents=True, exist_ok=True)

        api = HfApi()
        dataset_files = api.list_repo_files(repo_id=dataset_repo, repo_type="dataset")

        for file in dataset_files:

            if file not in ['classification_v1.0/m-eurosat.zip', 'classification_v1.0/normalizer.json']:
                continue

            local_file_path = local_directory / file

            local_file_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Downloading {file}...")
            hf_hub_download(
                repo_id=dataset_repo,
                filename=file,
                cache_dir=local_directory,
                local_dir=local_directory,
                repo_type="dataset",
            )
            if file.endswith(".zip"):
                print(f"Decompressing ...")
                decompress_zip_with_progress(local_directory / file)

    