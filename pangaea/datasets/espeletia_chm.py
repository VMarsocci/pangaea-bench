import os
from pathlib import Path, PurePosixPath
import shutil
import time 
import urllib.request
import urllib.error
import zipfile
import numpy as np
import pandas as pd
import rasterio
import torch

from pangaea.datasets.base import RawGeoFMDataset
from pangaea.datasets.utils import DownloadProgressBar


class EspeletiaCHM(RawGeoFMDataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        multi_modal: bool,
        multi_temporal: int | bool,
        root_path: str,
        classes: list,
        num_classes: int,
        ignore_index: int,
        img_size: int,
        bands: dict[str, list[str]],
        distribution: list,
        data_mean: dict[str, list[float]],
        data_std: dict[str, list[float]],
        data_min: dict[str, list[float]],
        data_max: dict[str, list[float]],
        download_url: str,
        auto_download: bool,
    ):
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
        self.split = split

        split_csv = self.root_path / split / "labels.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")

        self.df = pd.read_csv(split_csv)

        required_cols = ["patch_id", "label", "ms_path", "dsm_path", "chm_path"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' not found in {split_csv}")

        self.samples = self.df.to_dict("records")

    def __len__(self):
        return len(self.samples)

    def _read_tif(self, path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read()  # (C, H, W)
        return arr

    def __getitem__(self, index: int):
        sample = self.samples[index]

        ms_path = self.root_path / sample["ms_path"]
        dsm_path = self.root_path / sample["dsm_path"]
        chm_path = self.root_path / sample["chm_path"]

        if not ms_path.exists():
            raise FileNotFoundError(f"MS patch not found: {ms_path}")
        if not dsm_path.exists():
            raise FileNotFoundError(f"DSM patch not found: {dsm_path}")
        if not chm_path.exists():
            raise FileNotFoundError(f"CHM patch not found: {chm_path}")

        ms = self._read_tif(ms_path).astype(np.float32)
        dsm = self._read_tif(dsm_path).astype(np.float32)
        chm = self._read_tif(chm_path).astype(np.float32)


        ms = torch.from_numpy(ms).float()
        dsm = torch.from_numpy(dsm).float()
        chm = torch.from_numpy(chm).float()

        # Pangaea expects C T H W
        ms = ms.unsqueeze(1)   # [6, 1, H, W]
        dsm = dsm.unsqueeze(1) # [1, 1, H, W]
        chm = chm.squeeze(0) # [H, W]
        #print(chm.shape)

        output = {
            "image":{
                "optical": ms,     # (C_ms, 1, H, W)
                "dem": dsm,        # (1, 1, H, W)
            },
            "target": chm.clip(0.,50.),
            "metadata": {
                "patch_id": sample.get("patch_id", str(index)),
                "polygon_id": sample.get("polygon_id", -1),
                "description": sample.get("description", ""),
                "ms_path": str(ms_path),
                "chm_path": str(chm_path),
            }
        }
        return output

    @staticmethod
    def download(self, silent: bool = False):
        output_path = Path(self.root_path)
        url = self.download_url

        existing_dirs = list(output_path.glob("espeletia_*"))
        if existing_dirs:
            if not silent:
                print("Espeletia Dataset folder exists, skipping downloading dataset.")
            return

        output_path.mkdir(parents=True, exist_ok=True)

        temp_file_name = f"temp_{hex(int(time.time()))}_Espeletia.zip"
        pbar = DownloadProgressBar()

        try:
            urllib.request.urlretrieve(url, output_path / temp_file_name, pbar)
        except urllib.error.HTTPError as e:
            print("Error while downloading dataset: The server couldn't fulfill the request.")
            print("Error code:", e.code)
            return
        except urllib.error.URLError as e:
            print("Error while downloading dataset: Failed to reach a server.")
            print("Reason:", e.reason)
            return

        temp_zip = output_path / temp_file_name

        # Clean up stale wrong files from previous failed extraction
        for split_name in ["train", "val", "test"]:
            split_path = output_path / split_name
            if split_path.exists() and split_path.is_file():
                split_path.unlink()

        with zipfile.ZipFile(temp_zip, "r") as zip_ref:
            if not silent:
                print(f"Extracting to {output_path} ...")

            for member in zip_ref.infolist():
                raw_path = PurePosixPath(member.filename)

                # Skip empty names and directory entries
                if not member.filename or member.is_dir():
                    continue

                parts = raw_path.parts

                # Expect paths like:
                # espeletia_scene_cls_chm/train/ms/file.tif
                if len(parts) < 2:
                    continue

                # Remove the top-level folder
                rel_path = Path(*parts[1:])

                # Skip accidental container-only entries like "train", "val", "test"
                if len(rel_path.parts) == 1 and rel_path.suffix == "":
                    continue

                target_path = output_path / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)

                with zip_ref.open(member) as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            if not silent:
                print("done.")

        temp_zip.unlink()
