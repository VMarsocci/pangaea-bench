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

_HF_REVISION = "89dcb32efad356a5cdc6da5285ad6c872216529a"
_SPLIT_MAP = {"train": "train", "val": "val", "test": "test"}
_BAND_INDEX = {"B8": 0, "B4": 1, "B3": 2, "B2": 3}


class PureForest(RawGeoFMDataset):
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
        """PureForest tree species classification (18 classes).

        Loads from the parquet mirror at
        `kshitijrajsharma/pangaea2-vhr` (config `pureforest`). The mirror
        ships 4-channel PNGs in order (NIR, R, G, B); band selection is
        driven by `bands.optical` via the YAML.
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

        self._band_indices = [_BAND_INDEX[b] for b in bands["optical"]]
        prefix = _SPLIT_MAP[split]
        shard_dir = os.path.join(root_path, "pureforest")
        shards = sorted(glob(os.path.join(shard_dir, f"{prefix}-*.parquet")))
        if not shards:
            raise FileNotFoundError(
                f"No PureForest shards for split '{split}' under {shard_dir}"
            )
        table = pa.concat_tables(
            [pq.read_table(p, columns=["image", "label"]) for p in shards]
        )
        self._images = table["image"]
        self._labels = table["label"]

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        cell = self._images[index].as_py()
        if isinstance(cell, dict) and cell.get("bytes") is not None:
            img_bytes = cell["bytes"]
        else:
            img_bytes = cell
        img = Image.open(io.BytesIO(img_bytes))
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        arr = arr.transpose(2, 0, 1)
        image = torch.from_numpy(arr[self._band_indices, :, :]).unsqueeze(1)

        label = int(self._labels[index].as_py())
        target = torch.tensor(label, dtype=torch.int64)

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
            allow_patterns=["pureforest/*.parquet", "pureforest/label_map.json"],
        )
