import io
import os
from glob import glob

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from pangaea.datasets.base import RawGeoFMDataset

_HF_REVISION = "65adb1eacfd6fa7c8070ab21f840b392fc9fbb55"
_SPLIT_MAP = {"train": "train", "val": "validation", "test": "test"}


def _decode_image(cell: object) -> Image.Image:
    if isinstance(cell, dict) and "bytes" in cell and cell["bytes"] is not None:
        return Image.open(io.BytesIO(cell["bytes"]))
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell))
    raise TypeError(f"Unsupported parquet image cell type: {type(cell)!r}")


class HOTBuildings(RawGeoFMDataset):
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
        """HOT VHR building segmentation.

        Source: https://huggingface.co/datasets/hotosm/vhr-building-segmentation.
        Binary building masks over RGB aerial imagery, distributed as parquet
        shards with embedded PNG-encoded images and masks. Splits (train,
        validation, test) are defined by the upstream parquet shard names.
        """
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

        prefix = _SPLIT_MAP[split]
        data_dir = os.path.join(root_path, "data")
        shards = sorted(glob(os.path.join(data_dir, f"{prefix}-*.parquet")))
        if not shards:
            raise FileNotFoundError(
                f"No parquet shards for split '{split}' under {data_dir}"
            )
        tables = [pq.read_table(p, columns=["image", "mask"]) for p in shards]
        table = pa.concat_tables(tables)
        self._images = table["image"]
        self._masks = table["mask"]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        img_pil = _decode_image(self._images[index].as_py()).convert("RGB")
        mask_pil = _decode_image(self._masks[index].as_py())

        img = np.asarray(img_pil, dtype=np.float32) / 255.0
        image = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(1)

        mask_np = np.asarray(mask_pil)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        target = torch.from_numpy((mask_np > 0).astype(np.int64))

        return {
            "image": {"optical": image},
            "target": target,
            "metadata": {},
        }

    @staticmethod
    def download(self, silent: bool = False) -> None:
        os.makedirs(self.root_path, exist_ok=True)
        snapshot_download(
            repo_id=self.download_url,
            repo_type="dataset",
            revision=_HF_REVISION,
            local_dir=self.root_path,
            allow_patterns=["data/*.parquet", "norm_stats.json"],
        )
