import io
import json
import os

import h5py
import rasterio
import tacoreader.v1 as tacoreader
import torch
from huggingface_hub import hf_hub_download

from pangaea.datasets.base import RawGeoFMDataset
from pangaea.datasets.utils import read_tortilla_subfile_bytes

_TORTILLA_FILE = "geobench_nzcattle.tortilla"
_HF_SUBPATH = f"nzcattle/{_TORTILLA_FILE}"
_HF_REVISION = "d94af9521e6870516b93faffc1c840d08a57a157"
_SPLIT_MAP = {"train": "train", "val": "validation", "test": "test"}
_RGB_BAND_INDEX = {"B4": 0, "B3": 1, "B2": 2}


class NZCattleDet(RawGeoFMDataset):
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
        """GEO-Bench-2 NZ-Cattle detection (RGB aerial)."""
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

        self._band_indices = [_RGB_BAND_INDEX[b] for b in bands["optical"]]

        tortilla_path = os.path.join(root_path, _TORTILLA_FILE)
        df = tacoreader.load(tortilla_path)
        target_split = _SPLIT_MAP[split]
        self._df = df[df["tortilla:data_split"] == target_split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, i: int):
        sample_row = self._df.read(i)
        image_path = sample_row.read(0)
        annot_ref = sample_row.read(1)

        with rasterio.open(image_path) as src:
            data = src.read(out_dtype="float32")
        image = torch.from_numpy(data[self._band_indices, :, :]).unsqueeze(1)

        annot_bytes = annot_ref if isinstance(annot_ref, (bytes, bytearray)) else read_tortilla_subfile_bytes(annot_ref)
        with h5py.File(io.BytesIO(annot_bytes), "r") as f:
            annotations = json.loads(f.attrs["annotation"])["boxes"]

        if annotations:
            boxes = torch.tensor(
                [
                    [b["bbox"][0], b["bbox"][1], b["bbox"][0] + b["bbox"][2], b["bbox"][1] + b["bbox"][3]]
                    for b in annotations
                ],
                dtype=torch.float32,
            )
            labels = torch.tensor(
                [b["category_id"] for b in annotations], dtype=torch.int64
            )
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return {
            "image": {"optical": image},
            "boxes": boxes,
            "labels": labels,
            "metadata": {},
        }

    @staticmethod
    def download(self, silent: bool = False) -> None:
        os.makedirs(self.root_path, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=self.download_url,
            repo_type="dataset",
            filename=_HF_SUBPATH,
            revision=_HF_REVISION,
            local_dir=self.root_path,
        )
        target = os.path.join(self.root_path, _TORTILLA_FILE)
        if os.path.abspath(downloaded) != os.path.abspath(target):
            os.replace(downloaded, target)
