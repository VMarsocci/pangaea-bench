import io
import os
from glob import glob

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_ASSETS_CACHE
from PIL import Image
from sklearn.model_selection import train_test_split

from pangaea.datasets.base import RawGeoFMDataset

_HF_REVISION = "49dfe4fa206d43720695350a597d067e6892844a"
_AOI_ORDER = ("AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum")
_SPLIT_SEED = 234
_COLUMNS = ["aoi", "image_tif", "road_mask"]
_ARROW_CACHE = "sn3_roads.arrow"


def _read_shard(shard: str) -> pa.Table:
    """Read a shard's needed columns, widening image_tif to large_binary.

    500 image blobs overflow the int32 offsets of plain binary (>2 GB chunk).
    """
    table = pq.read_table(shard, columns=_COLUMNS)
    idx = table.schema.get_field_index("image_tif")
    return table.set_column(
        idx, "image_tif", table["image_tif"].cast(pa.large_binary())
    )


def _build_arrow_cache(shards: list[str], cache_path: str) -> None:
    """Convert the parquet shards into one uncompressed Arrow IPC file.

    Memory-mapping this file gives O(1) row reads without holding the dataset
    in RAM, which a single 500-row parquet group cannot.
    """
    tmp_path = cache_path + ".tmp"
    first, *rest = shards
    table = _read_shard(first)
    with pa.ipc.new_file(tmp_path, table.schema) as writer:
        writer.write_table(table)
        for shard in rest:
            writer.write_table(_read_shard(shard))
    os.replace(tmp_path, cache_path)


class SpaceNet3(RawGeoFMDataset):
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
        """SpaceNet 3 road extraction (binary segmentation).

        Loads from the parquet mirror at `kshitijrajsharma/pangaea2-vhr`
        (config `sn3_roads`). Upstream `test` is unlabeled (standard SpaceNet
        challenge format), so we carve train/val/test (70/20/10) from upstream
        `train` via a deterministic per-AOI random split with seed 234.
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

        shard_dir = os.path.join(root_path, "sn3_roads")
        shards = sorted(glob(os.path.join(shard_dir, "train-*.parquet")))
        if not shards:
            raise FileNotFoundError(f"No SN3 train shards under {shard_dir}")

        # Derived cache lives in the HF cache (honors HF_HOME on offline nodes),
        cache_dir = os.path.join(HF_ASSETS_CACHE, "pangaea", "spacenet3", _HF_REVISION)
        cache_path = os.path.join(cache_dir, _ARROW_CACHE)
        if not os.path.exists(cache_path):
            os.makedirs(cache_dir, exist_ok=True)
            _build_arrow_cache(shards, cache_path)

        self._mmap = pa.memory_map(cache_path, "r")
        self._table = pa.ipc.open_file(self._mmap).read_all()

        mask_struct = self._table["road_mask"].combine_chunks()
        mask_valid = mask_struct.field("bytes").is_valid().to_pylist()
        aois = self._table["aoi"].to_pylist()
        labeled = [(i, aoi) for i, (aoi, ok) in enumerate(zip(aois, mask_valid)) if ok]

        indices_for_split: list[int] = []
        for aoi in _AOI_ORDER:
            aoi_indices = [i for i, a in labeled if a == aoi]
            if not aoi_indices:
                continue

            train_idx, holdout_idx = train_test_split(
                aoi_indices, test_size=0.30, random_state=_SPLIT_SEED, shuffle=True
            )
            val_idx, test_idx = train_test_split(
                holdout_idx, test_size=1 / 3, random_state=_SPLIT_SEED, shuffle=True
            )

            picked = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
            indices_for_split.extend(picked)

        self._indices = indices_for_split

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        row = self._indices[i]

        tif_bytes = self._table["image_tif"][row].as_py()
        with rasterio.open(io.BytesIO(tif_bytes)) as src:
            arr = src.read()
        image = torch.from_numpy(arr.astype(np.float32)).unsqueeze(1)

        mask_bytes = self._table["road_mask"][row].as_py()["bytes"]
        mask_np = np.asarray(Image.open(io.BytesIO(mask_bytes)))
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
            allow_patterns=["sn3_roads/train-*.parquet", "sn3_roads/stats.json"],
        )
