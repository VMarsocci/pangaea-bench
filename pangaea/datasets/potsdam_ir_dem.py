import os
import pathlib
import shutil
import zipfile
from glob import glob

import numpy as np
import requests
import rasterio
import torch
from PIL import Image
from tqdm import tqdm

from pangaea.datasets.base import RawGeoFMDataset
from pangaea.datasets.utils import DownloadProgressBar

import os
import re
import random


def parse_rgbir_id(filename: str) -> str:
    """
    top_potsdam_2_10_RGBIR.tif -> 2_10
    """
    m = re.match(r"top_potsdam_(\d+)_(\d+)_RGBIR\.tif$", filename)
    if m is None:
        raise ValueError(f"Unexpected RGBIR filename format: {filename}")
    return f"{int(m.group(1))}_{int(m.group(2))}"


def parse_label_id(filename: str) -> str:
    """
    top_potsdam_2_10_label.tif -> 2_10
    """
    m = re.match(r"top_potsdam_(\d+)_(\d+)_label\.tif$", filename)
    if m is None:
        raise ValueError(f"Unexpected label filename format: {filename}")
    return f"{int(m.group(1))}_{int(m.group(2))}"


def parse_dsm_norm_id(filename: str) -> str:
    """
    dsm_potsdam_02_10_normalized_lastools.jpg -> 2_10
    """
    m = re.match(r"dsm_potsdam_(\d+)_(\d+)_normalized_lastools\.jpg$", filename)
    if m is None:
        raise ValueError(f"Unexpected DSM filename format: {filename}")
    return f"{int(m.group(1))}_{int(m.group(2))}"

def build_file_map(directory: str, suffix: str, parser_fn):
    files = [f for f in os.listdir(directory) if f.endswith(suffix)]
    file_map = {}
    for f in files:
        tile_id = parser_fn(f)
        file_map[tile_id] = os.path.join(directory, f)
    return file_map

class PotsdamIRDEM(RawGeoFMDataset):
    def __init__(
        self,
        download_url: str,
        auto_download: bool,
        download_password: str,
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
        data_mean: dict[str, list[float]],
        data_std: dict[str, list[float]],
        data_min: dict[str, list[float]],
        data_max: dict[str, list[float]],
    ):
        """Initialize the ISPRS Potsdam dataset.
            Link: https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx

        Args:
            download_url (str): url to download the dataset.
            auto_download (bool): whether to download the dataset automatically.
            download_password (str): password to download the dataset.
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
        self.download_password = download_password

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

        self.root_path = pathlib.Path(root_path)
        self.split = split

        self.optical_list = sorted(glob(str(self.root_path / split / "optical" / "*.tif")))
        self.dem_list = sorted(glob(str(self.root_path / split / "dem" / "*.tif")))
        self.mask_list = sorted(glob(str(self.root_path / split / "labels" / "*.tif")))

        assert len(self.optical_list) == len(self.dem_list) == len(self.mask_list), (
            f"Mismatch in dataset file counts for split={split}: "
            f"optical={len(self.optical_list)}, dem={len(self.dem_list)}, labels={len(self.mask_list)}"
        )

        self.class_colors = [
            (255, 255, 255),  # impervious surfaces
            (0, 0, 255),      # building
            (0, 255, 255),    # low vegetation
            (0, 255, 0),      # tree
            (255, 255, 0),    # car
            (255, 0, 0),      # clutter/background
        ]
        self.class_color_tensor = torch.tensor(self.class_colors, dtype=torch.uint8)

    def __len__(self):
        return len(self.optical_list)

    def __getitem__(self, index):
        with rasterio.open(self.optical_list[index]) as src:
            optical = torch.from_numpy(src.read()).float()  # [4, H, W]

        with rasterio.open(self.dem_list[index]) as src:
            dem = torch.from_numpy(src.read(1)).float()     # [H, W]

        label_img = np.array(Image.open(self.mask_list[index]))

        if label_img.ndim == 2:
            target = torch.from_numpy(label_img).long()
        else:
            label_t = torch.from_numpy(label_img).permute(2, 0, 1).to(torch.uint8)  # [3, H, W]

            matches = torch.stack(
                [
                    torch.all(label_t == color.view(3, 1, 1), dim=0)
                    for color in self.class_color_tensor
                ],
                dim=0,
            )  # [num_classes, H, W]

            target = matches.float().argmax(dim=0).long()
            unknown = ~matches.any(dim=0)
            target[unknown] = self.ignore_index

        optical[torch.isnan(optical)] = 0
        dem[torch.isnan(dem)] = 0

        # Pangaea expects C T H W
        optical = optical.unsqueeze(1)      # [4, 1, H, W]
        dem = dem.unsqueeze(0).unsqueeze(1) # [1, 1, H, W]

        return {
            "image": {
                "optical": optical,
                "dem": dem,
            },
            "target": target,
            "metadata": {},
        }
        
    @staticmethod
    def download(self, silent: bool = False):
        s = requests.session()

        # Fetch tokens from the protected page
        response = s.get(self.download_url)
        response.raise_for_status()
        html = response.text

        csrf_middleware_token = html.split('name="csrfmiddlewaretoken" value="')[1].split('"')[0]
        token = html.split('name="token" value="')[1].split('"')[0]

        data = {
            "csrfmiddlewaretoken": csrf_middleware_token,
            "token": token,
            "password": self.download_password,
        }

        out_dir = str(self.root_path)
        os.makedirs(out_dir, exist_ok=True)

        zip_path = os.path.join(out_dir, "potsdam.zip")
        pbar = DownloadProgressBar()

        try:
            with s.post(
                self.download_url + "?dl=1",
                data=data,
                stream=True,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            ) as response:
                response.raise_for_status()

                total_size = int(response.headers.get("Content-Length", 0))
                with open(zip_path, "wb") as f:
                    for i, chunk in enumerate(response.iter_content(chunk_size=8192)):
                        if chunk:
                            f.write(chunk)
                            if total_size > 0 and not silent:
                                pbar(i, 8192, total_size)

        except requests.exceptions.HTTPError as e:
            print("HTTP error while downloading Potsdam dataset.")
            print(e)
            return

        except requests.exceptions.RequestException as e:
            print("Request error while downloading Potsdam dataset.")
            print(e)
            return

        raw_dir = os.path.join(out_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # Extract only the inner archives we need
        # Extract only the inner archives we need
        print("Extracting inner archives...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extract("Potsdam/5_Labels_for_participants.zip", raw_dir)
            zip_ref.extract("Potsdam/5_Labels_all.zip", raw_dir)
            zip_ref.extract("Potsdam/4_Ortho_RGBIR.zip", raw_dir)
            zip_ref.extract("Potsdam/1_DSM_normalisation.zip", raw_dir)

        print("Extracting training labels...")
        with zipfile.ZipFile(os.path.join(raw_dir, "Potsdam", "5_Labels_for_participants.zip"), "r") as zip_ref:
            zip_ref.extractall(os.path.join(raw_dir, "5_Labels_for_participants"))

        print("Extracting all labels...")
        with zipfile.ZipFile(os.path.join(raw_dir, "Potsdam", "5_Labels_all.zip"), "r") as zip_ref:
            zip_ref.extractall(os.path.join(raw_dir, "5_Labels_all"))

        print("Extracting RGBIR images...")
        with zipfile.ZipFile(os.path.join(raw_dir, "Potsdam", "4_Ortho_RGBIR.zip"), "r") as zip_ref:
            zip_ref.extractall(os.path.join(raw_dir, "4_Ortho_RGBIR"))

        print("Extracting DSMs...")
        with zipfile.ZipFile(os.path.join(raw_dir, "Potsdam", "1_DSM_normalisation.zip"), "r") as zip_ref:
            zip_ref.extractall(os.path.join(raw_dir, "1_DSM_normalisation"))

        # Flatten nested extracted folders if needed
        flatten_if_single_nested_dir(os.path.join(raw_dir, "4_Ortho_RGBIR"))
        flatten_if_single_nested_dir(os.path.join(raw_dir, "1_DSM_normalisation"))
        flatten_if_single_nested_dir(os.path.join(raw_dir, "5_Labels_for_participants"))
        flatten_if_single_nested_dir(os.path.join(raw_dir, "5_Labels_all"))

        image_dir = os.path.join(raw_dir, "4_Ortho_RGBIR")
        dsm_dir = os.path.join(raw_dir, "1_DSM_normalisation")
        label_all_dir = os.path.join(raw_dir, "5_Labels_all")
        label_train_dir = os.path.join(raw_dir, "5_Labels_for_participants")

        rgbir_map = build_file_map(image_dir, ".tif", parse_rgbir_id)
        dsm_map = build_file_map(dsm_dir, "_normalized_lastools.jpg", parse_dsm_norm_id)
        label_all_map = build_file_map(label_all_dir, ".tif", parse_label_id)
        label_train_map = build_file_map(label_train_dir, ".tif", parse_label_id)

        train_numbers = sorted(label_train_map.keys())
        # Tiles with labels that are not part of the original participant
        # training set form the held-out pool.
        heldout_numbers = sorted(
            k for k in label_all_map.keys()
            if k not in train_numbers
        )

        if len(heldout_numbers) < 2:
            raise RuntimeError(
                "At least two held-out source tiles are required to create "
                "separate validation and test splits. "
                f"Found {len(heldout_numbers)}."
            )

        # Divide the held-out source tiles into validation and test sets.
        # The seed makes the split reproducible.
        split_rng = random.Random(42)
        split_rng.shuffle(heldout_numbers)

        split_index = len(heldout_numbers) // 2
        val_numbers = sorted(heldout_numbers[:split_index])
        test_numbers = sorted(heldout_numbers[split_index:])

        print(
            "Source-tile split: "
            f"{len(train_numbers)} train, "
            f"{len(val_numbers)} validation, "
            f"{len(test_numbers)} test."
        )
        print(f"Validation tile IDs: {val_numbers}")
        print(f"Test tile IDs: {test_numbers}")

        # Validate every source tile before beginning the expensive tiling step.
        split_numbers = {
            "training": train_numbers,
            "validation": val_numbers,
            "test": test_numbers,
        }

        # Create consistent directory structure
        for split_name, numbers in split_numbers.items():
            missing_rgbir = [k for k in numbers if k not in rgbir_map]
            missing_dsm = [k for k in numbers if k not in dsm_map]
            missing_labels = [k for k in numbers if k not in label_all_map]

            if missing_rgbir:
                raise FileNotFoundError(
                    f"Missing RGBIR files for {split_name} tiles, "
                    f"e.g. {missing_rgbir[:5]}"
                )

            if missing_dsm:
                raise FileNotFoundError(
                    f"Missing DSM files for {split_name} tiles, "
                    f"e.g. {missing_dsm[:5]}"
                )

            if missing_labels:
                raise FileNotFoundError(
                    f"Missing label files for {split_name} tiles, "
                    f"e.g. {missing_labels[:5]}"
                )
            
        for split_name in ["train", "val", "test"]:
            os.makedirs(
                os.path.join(out_dir, split_name, "optical"),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(out_dir, split_name, "dem"),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(out_dir, split_name, "labels"),
                exist_ok=True,
            )    

        print("Tiling training images...")
        tile_and_save_split(
            numbers=train_numbers,
            rgbir_map=rgbir_map,
            dsm_map=dsm_map,
            label_map=label_all_map,
            out_dir=out_dir,
            save_folder="train",
            tile_size=256,
            overlap=0,
        )

        print("Tiling validation images...")
        tile_and_save_split(
            numbers=val_numbers,
            rgbir_map=rgbir_map,
            dsm_map=dsm_map,
            label_map=label_all_map,
            out_dir=out_dir,
            save_folder="val",
            tile_size=256,
            overlap=0,
        )

        print("Tiling test images...")
        tile_and_save_split(
            numbers=test_numbers,
            rgbir_map=rgbir_map,
            dsm_map=dsm_map,
            label_map=label_all_map,
            out_dir=out_dir,
            save_folder="test",
            tile_size=256,
            overlap=0,
        )

        print("Cleaning temporary files...")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(raw_dir):
            shutil.rmtree(raw_dir)


def tile_and_save_split(
    numbers: list[str],
    rgbir_map: dict[str, str],
    dsm_map: dict[str, str],
    label_map: dict[str, str],
    out_dir: str,
    save_folder: str,
    tile_size: int = 512,
    overlap: int = 128,
):
    i = 0
    for tile_id in tqdm(numbers):
        if tile_id not in rgbir_map:
            raise FileNotFoundError(f"Missing RGBIR file for tile {tile_id}")
        if tile_id not in dsm_map:
            raise FileNotFoundError(f"Missing DSM file for tile {tile_id}")
        if tile_id not in label_map:
            raise FileNotFoundError(f"Missing label file for tile {tile_id}")

        image_path = rgbir_map[tile_id]
        dsm_path = dsm_map[tile_id]
        label_path = label_map[tile_id]

        with rasterio.open(image_path) as src_img:
            image = src_img.read()  # [4, H, W]
            image = np.transpose(image, (1, 2, 0))  # [H, W, 4]

        dsm = np.array(Image.open(dsm_path))  # [H, W] expected for normalized jpg
        label = np.array(Image.open(label_path))  # [H, W, 3]

        if dsm.ndim == 3:
            # just in case JPG loads as HWC with repeated channels
            if dsm.shape[2] == 1:
                dsm = dsm[:, :, 0]
            elif dsm.shape[2] == 3:
                # if saved as RGB grayscale jpg, take one channel
                dsm = dsm[:, :, 0]
            else:
                raise ValueError(f"Unexpected DSM shape for {dsm_path}: {dsm.shape}")

        h = min(image.shape[0], dsm.shape[0], label.shape[0])
        w = min(image.shape[1], dsm.shape[1], label.shape[1])

        if image.shape[:2] != (h, w) or dsm.shape[:2] != (h, w) or label.shape[:2] != (h, w):
            print(
                f"Warning: spatial mismatch for tile {tile_id}. "
                f"Cropping to common size ({h}, {w}) from "
                f"image={image.shape}, dsm={dsm.shape}, label={label.shape}"
            )

        image = image[:h, :w, ...]
        dsm = dsm[:h, :w]
        label = label[:h, :w, ...]

        image_tiles = tile_image(image, tile_size=tile_size, overlap=overlap)
        dsm_tiles = tile_image(dsm, tile_size=tile_size, overlap=overlap)
        label_tiles = tile_image(label, tile_size=tile_size, overlap=overlap)

        assert len(image_tiles) == len(dsm_tiles) == len(label_tiles), (
            f"Tile count mismatch for {tile_id}: "
            f"image={len(image_tiles)}, dsm={len(dsm_tiles)}, label={len(label_tiles)}"
        )

        for image_tile, dsm_tile, label_tile in zip(image_tiles, dsm_tiles, label_tiles):
            save_multiband_tiff(
                os.path.join(out_dir, save_folder, "optical", f"{i}.tif"),
                image_tile,
            )
            save_singleband_tiff(
                os.path.join(out_dir, save_folder, "dem", f"{i}.tif"),
                dsm_tile,
            )
            Image.fromarray(label_tile).save(
                os.path.join(out_dir, save_folder, "labels", f"{i}.tif")
            )
            i += 1


def tile_image(image, tile_size: int = 256, overlap: int = 0):
    stride = tile_size - overlap
    tiles = []

    for y in range(0, image.shape[0] - tile_size + 1, stride):
        for x in range(0, image.shape[1] - tile_size + 1, stride):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)

    return tiles


def save_multiband_tiff(path: str, array: np.ndarray):
    """
    Save [H, W, C] array as multiband TIFF.
    """
    if array.ndim != 3:
        raise ValueError(f"Expected [H, W, C] array, got shape {array.shape}")

    h, w, c = array.shape
    array = np.transpose(array, (2, 0, 1))  # [C, H, W]

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype=array.dtype,
    ) as dst:
        dst.write(array)


def save_singleband_tiff(path: str, array: np.ndarray):
    """
    Save [H, W] array as single-band TIFF.
    """
    if array.ndim != 2:
        raise ValueError(f"Expected [H, W] array, got shape {array.shape}")

    h, w = array.shape

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=array.dtype,
    ) as dst:
        dst.write(array, 1)


def flatten_if_single_nested_dir(path: str):
    """
    If extraction creates path/<single_subdir>/*, move contents one level up.
    """
    if not os.path.isdir(path):
        return

    entries = os.listdir(path)
    if len(entries) != 1:
        return

    inner = os.path.join(path, entries[0])
    if not os.path.isdir(inner):
        return

    for name in os.listdir(inner):
        shutil.move(os.path.join(inner, name), os.path.join(path, name))

    os.rmdir(inner)


def image_number(filename: str) -> str:
    parts = filename.split("_")
    return parts[-3] + "_" + parts[-2]


def image_filename(number: str) -> str:
    return f"top_potsdam_{number}_RGBIR.tif"


def dsm_filename(number: str) -> str:
    return f"top_potsdam_{number}_DSM.tif"


def label_filename(number: str) -> str:
    return f"top_potsdam_{number}_label.tif"