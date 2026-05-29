# Sources: 
# - https://github.com/PaulBorneP/Xview2_Strong_Baseline/blob/master/datasets/base_dataset.py
# - https://github.com/PaulBorneP/Xview2_Strong_Baseline/blob/master/datasets/supervised_dataset.py
# - https://github.com/PaulBorneP/Xview2_Strong_Baseline/blob/master/legacy/datasets.py

from typing import Sequence, Dict, Any, Union, Literal, Tuple
import shutil
import time
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pathlib
from glob import glob
from sklearn.model_selection import train_test_split
import cv2
import urllib
import tarfile

from pangaea.datasets.utils import DownloadProgressBar
from pangaea.datasets.base import RawGeoFMDataset

_HF_REVISION = "49dfe4fa206d43720695350a597d067e6892844a"
_HF_SPLITS = ("train", "test", "tier3")


class xView2(RawGeoFMDataset):
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
        oversample_building_damage: bool
    ):
        """Initialize the xView2 dataset.
        Link: https://xview2.org/dataset
        Note that this dataset must be downloaded from the competition website, which you will need to sign up for.

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
            oversample_building_damage (bool): whether to oversample images with building damage
        """
        super(xView2, self).__init__(
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
        self.split = split
        self.bands = bands
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
        self.oversample_building_damage = oversample_building_damage

        self.all_files = self.get_all_files()

    def get_all_files(self) -> Sequence[str]:
        all_files = []
        if self.split == "test":
            data_dirs = [os.path.join(self.root_path, "test")]
        else:
            # Train and val consist of the smaller train and the larger tier3 set.
            data_dirs = [os.path.join(self.root_path, d) for d in ["train", "tier3"]]

        for d in data_dirs:
            for f in sorted(os.listdir(os.path.join(d, 'images'))):
                if '_pre_disaster.png' in f:
                    all_files.append(os.path.join(d, 'images', f))
        
        if self.split != "test":
            train_val_idcs = self.get_stratified_train_val_split(all_files)
            
            if self.split == "train" and self.oversample_building_damage:
                train_val_idcs[self.split] = self.oversample_building_files(all_files, train_val_idcs[self.split])

            
            all_files = [all_files[i] for i in train_val_idcs[self.split]]
            

        return all_files

    @staticmethod
    def get_stratified_train_val_split(all_files) -> Tuple[Sequence[int], Sequence[int]]:

       # Fixed stratified sample to split data into train/val. 
       # This keeps 90% of datapoints belonging to an individual event in the training set and puts the remaining 10% in the validation set. 
        disaster_names = list(
            map(lambda path: pathlib.Path(path).name.split("_")[0], all_files))
        train_idxs, val_idxs = train_test_split(np.arange(len(all_files)),
                                                test_size=0.1,
                                                random_state=23,
                                                stratify=disaster_names)
        return {"train": train_idxs, "val": val_idxs}

    def oversample_building_files(self, all_files, train_idxs):
        # Oversamples buildings on the image-level, by including each image with any building pixels twice in the training set. 
        file_classes = []
        for i, fn in enumerate(all_files):
            fl = np.zeros((4,), dtype=bool)
            # Only read images that are included in train_idxs
            if i in train_idxs: 
                msk1 = cv2.imread(fn.replace('/images/', '/masks/').replace('_pre_disaster', '_post_disaster'),
                                cv2.IMREAD_UNCHANGED)
                for c in range(1, 5):
                    fl[c - 1] = c in msk1
            file_classes.append(fl)
        file_classes = np.asarray(file_classes)

        new_train_idxs = []
        for i in train_idxs:
            new_train_idxs.append(i)
            # If any building damage was present in the image, add the image to the training set a second time.
            if file_classes[i, 1:].max():
                new_train_idxs.append(i)
            # If minor or medium damage were present, add it a third time, since these two classes are very hard to detect.
            # Source: https://github.com/DIUx-xView/xView2_first_place/blob/master/train34_cls.py
            if file_classes[i, 1:3].max():
                new_train_idxs.append(i)
        train_idxs = np.asarray(new_train_idxs)
        return train_idxs


    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor,  Any, str]]:
        
        fn = self.all_files[idx]

        img_pre = cv2.imread(fn, cv2.IMREAD_COLOR)
        img_post = cv2.imread(fn.replace('_pre_', '_post_'), cv2.IMREAD_COLOR)


        #msk_pre = cv2.imread(fn.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED)
        msk_post = cv2.imread(fn.replace('/images/', '/masks/').replace(
            '_pre_disaster', '_post_disaster'), cv2.IMREAD_UNCHANGED)
        
        #msk = np.stack([msk_pre, msk_post], axis=0)
        msk = msk_post
        img = np.stack([img_pre, img_post], axis=0) 

        # Reshaping tensors from (T, H, W, C) to (C, T, H, W)
        img = torch.from_numpy(img.transpose((3, 0, 1, 2))).float()
        # img_pre = torch.from_numpy(img_pre.transpose((2, 0, 1))).float()
        # img_post = torch.from_numpy(img_post.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk).long()


        return {
            'image': {
                    'optical': img,
                    },
            'target': msk,  
            'metadata': {"filename":fn}
        }


    @staticmethod
    def download(self, silent=False):
        output_path = pathlib.Path(self.root_path)
        url = self.download_url

        if output_path.exists() and (output_path / "train" / "images").exists():
            if not silent:
                print("xView2 dataset folder exists, skipping downloading dataset.")
            return

        output_path.mkdir(parents=True, exist_ok=True)

        # HF dataset repo IDs match "<owner>/<repo>" with no URL scheme.
        if "://" not in url and url.count("/") == 1:
            xView2._materialize_from_hf_mirror(self, output_path, url, silent=silent)
            return

        temp_file_name = f"temp_{hex(int(time.time()))}_hls_burn_scars.tar.gz"
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

        with tarfile.open(output_path / temp_file_name, 'r') as tar:
            print(f"Extracting to {output_path} ...")
            tar.extractall(output_path)
            print("done.")

        os.remove(output_path / temp_file_name)

    @staticmethod
    def _materialize_from_hf_mirror(self, output_path, repo_id, silent=False):
        """Stream PNG bytes from the parquet mirror onto disk in the
        on-disk layout `xview2.py` already expects. The mirror stores raw
        `read_bytes()` of each PNG, so this is bit-identical to the
        xview2.org tarball.
        """
        from huggingface_hub import snapshot_download
        import pyarrow.parquet as pq

        if not silent:
            print(f"Downloading xView2 parquet shards from HF mirror {repo_id} ...")

        staging = output_path / "_hf_parquet"
        staging.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=_HF_REVISION,
            local_dir=str(staging),
            allow_patterns=["xview2/*.parquet"],
        )

        for split in _HF_SPLITS:
            shards = sorted(glob(str(staging / "xview2" / f"{split}-*.parquet")))
            if not shards:
                if not silent:
                    print(f"  no {split} shards found, skipping")
                continue
            images_dir = output_path / split / "images"
            masks_dir = output_path / split / "masks"
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
            n_written = 0
            for shard in shards:
                table = pq.read_table(
                    shard, columns=["image_id", "pre_image", "post_image", "post_mask"]
                )
                for i in range(len(table)):
                    iid = table["image_id"][i].as_py()
                    pre = table["pre_image"][i].as_py()["bytes"]
                    post = table["post_image"][i].as_py()["bytes"]
                    mask = table["post_mask"][i].as_py()["bytes"]
                    (images_dir / f"{iid}_pre_disaster.png").write_bytes(pre)
                    (images_dir / f"{iid}_post_disaster.png").write_bytes(post)
                    (masks_dir / f"{iid}_post_disaster.png").write_bytes(mask)
                    n_written += 1
            if not silent:
                print(f"  wrote {n_written} sample(s) to {output_path}/{split}/")

        shutil.rmtree(staging, ignore_errors=True)

if __name__=="__main__":
    dataset = xView2(
        split="train", 
        dataset_name="xView2",
        root_path="./data/xView2",
        download_url="https://the-dataset-is-not-publicly-available.com",
        auto_download=False,
        img_size=1024,
        multi_temporal=False,
        multi_modal=False, 
        classes=["No building", "No damage","Minor damage","Major damage","Destroyed"],
        num_classes=5,
        ignore_index=-1,
        bands=["B4", "B3", "B2"],
        distribution = [0.9415, 0.0448, 0.0049, 0.0057, 0.0031],
        data_mean=[66.7703, 88.4452, 85.1047],
        data_std=[48.3066, 51.9129, 62.7612],
        data_min=[0.0, 0.0, 0.0],
        data_max=[255, 255, 255],
    )
    x,y = dataset[0]
    print(x["optical"].shape, y.shape)