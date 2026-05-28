import os

import rasterio
import tacoreader.v1 as tacoreader
import torch
from huggingface_hub import hf_hub_download

from pangaea.datasets.base import RawGeoFMDataset

_TORTILLA_FILE = "geobench_flair2.tortilla"
_HF_SUBPATH = f"flair2/{_TORTILLA_FILE}"
_HF_REVISION = "919dc522d29adc3837b4f6a6175530c212a32e37"
_SPLIT_MAP = {"train": "train", "val": "validation", "test": "test"}
_AERIAL_BAND_INDEX = {"B4": 0, "B3": 1, "B2": 2, "B8": 3}


class FLAIR2(RawGeoFMDataset):
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
        """GEO-Bench-2 FLAIR-2 land cover semantic segmentation.

        Wraps the `geobench_flair2.tortilla` artifact published by AI Alliance
        and mirrored to `kshitijrajsharma/geobench2-vhr`. Splits, hashes, and
        per-pixel labels follow GEO-Bench-2 (https://arxiv.org/abs/2511.15658).
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

        self._band_indices = [_AERIAL_BAND_INDEX[b] for b in bands["optical"]]

        tortilla_path = os.path.join(root_path, _TORTILLA_FILE)
        df = tacoreader.load(tortilla_path)
        target_split = _SPLIT_MAP[split]
        self._df = df[df["tortilla:data_split"] == target_split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        sample_row = self._df.read(index)
        aerial_path = sample_row.read(0)
        mask_path = sample_row.read(1)

        with rasterio.open(aerial_path) as src:
            data = src.read()

        image = torch.from_numpy(data[self._band_indices, :, :]).float().unsqueeze(1)

        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        target = torch.from_numpy(mask).long()
        target[target > 13] = 13
        target -= 1

        return {
            "image": {"optical": image},
            "target": target,
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
