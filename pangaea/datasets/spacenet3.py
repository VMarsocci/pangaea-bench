import io
import os
from glob import glob

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from sklearn.model_selection import train_test_split

from pangaea.datasets.base import RawGeoFMDataset

_HF_REVISION = "49dfe4fa206d43720695350a597d067e6892844a"
_AOI_ORDER = ("AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum")
_SPLIT_SEED = 234


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

        Loads from the parquet mirror at
        `kshitijrajsharma/pangaea2-vhr` (config `sn3_roads`). Upstream `test`
        is unlabeled (standard SpaceNet challenge format), so we carve
        train/val/test (70/20/10) from upstream `train` via a deterministic
        per-AOI random split with seed 234.
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
        tables = []
        for p in shards:
            t = pq.read_table(p, columns=["aoi", "image_tif", "road_mask"])
            t = t.set_column(
                t.schema.get_field_index("image_tif"),
                "image_tif",
                t["image_tif"].cast(pa.large_binary()),
            )
            tables.append(t)
        table = pa.concat_tables(tables, promote_options="default")

        # Drop rows whose road_mask is missing. The HF README claims only
        # upstream test masks are null, but ~8 percent of train rows are too,
        # so we filter explicitly to keep the labeled subset.
        aois = table["aoi"].to_pylist()
        mask_col = table["road_mask"]
        labeled_indices = []
        labeled_aois = []
        for i in range(len(table)):
            cell = mask_col[i].as_py()
            has_bytes = cell is not None and (
                (isinstance(cell, dict) and cell.get("bytes") is not None)
                or isinstance(cell, (bytes, bytearray))
            )
            if has_bytes:
                labeled_indices.append(i)
                labeled_aois.append(aois[i])

        indices_for_split: list[int] = []
        for aoi in _AOI_ORDER:
            aoi_indices = [
                labeled_indices[k] for k, a in enumerate(labeled_aois) if a == aoi
            ]
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

        self._image_tifs = table["image_tif"].take(indices_for_split)
        self._road_masks = table["road_mask"].take(indices_for_split)

    def __len__(self) -> int:
        return len(self._image_tifs)

    def __getitem__(
        self, index: int
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        tif_bytes = self._image_tifs[index].as_py()
        with rasterio.open(io.BytesIO(tif_bytes)) as src:
            arr = src.read()
        image = torch.from_numpy(arr.astype(np.float32)).unsqueeze(1)

        mask_cell = self._road_masks[index].as_py()
        if isinstance(mask_cell, dict) and mask_cell.get("bytes") is not None:
            mask_bytes = mask_cell["bytes"]
        else:
            mask_bytes = mask_cell
        mask_img = Image.open(io.BytesIO(mask_bytes))
        mask_np = np.asarray(mask_img)
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
