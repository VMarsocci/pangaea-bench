import pathlib
from glob import glob
import os
import urllib
import urllib.request
import requests 
import shutil 

import numpy as np
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2
import zipfile 
from tqdm import tqdm
from PIL import Image

from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from pangaea.datasets.utils import DownloadProgressBar
from pangaea.datasets.base import RawGeoFMDataset


class Potsdam(RawGeoFMDataset):
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
        data_mean: dict[str, list[str]],
        data_std: dict[str, list[str]],
        data_min: dict[str, list[str]],
        data_max: dict[str, list[str]],
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

        super(Potsdam, self).__init__(
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

        self.data_mean = data_mean
        self.data_std = data_std
        self.data_min = data_min
        self.data_max = data_max
        self.classes = classes
        self.img_size = img_size
        self.distribution = distribution
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.root_path = pathlib.Path(root_path)
        self.split = split
        self.image_dir = self.root_path.joinpath(split)
        self.image_list = sorted(glob(str(self.image_dir.joinpath("images", "*.png"))))
        self.mask_list = sorted(glob(str(self.image_dir.joinpath("labels", "*.png"))))

        self.class_colors = [(255,255,255), (0,0,255), (0,255,255), (0,255,0), (255,255,0), (255,0,0)]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = read_image(pathlib.Path(self.image_list[index]))
        target = read_image(
            pathlib.Path(self.mask_list[index])
        )
        # target is a rgb image with each class as a different color
        # convert to a single channel image with each pixel as the class index
        target = torch.argmax(
            torch.stack(
                [
                    torch.all(target == torch.tensor(color).view(3, 1, 1), dim=0).long()
                    for color in self.class_colors
                ]
            ),
            dim=0,
        )

        # Convert the image and target to supported types
        image = v2.ToDtype(torch.float32)(image)  # Convert to float32
        target = v2.ToDtype(torch.int64)(target)  # Convert to int64 (since it's a mask)

        target = target.long()

        # Handle invalid data if any
        invalid_mask = torch.isnan(image)
        image[invalid_mask] = 5

        # image must have C T H W format, add time dimension
        image = image.unsqueeze(1)

        return {
            "image": {
                "optical": image,
            },
            "target": target,
            "metadata": {},
        }

    @staticmethod
    def download(self, silent=False):
        s = requests.session()
        # fetch tokens
        response = s.get(self.download_url)
        html = response.text

        # sfcrsf_token = response.headers.get("Set-Cookie").split(";")[0].split("=")[1]
        crsf_middleware_token = html.split("name=\"csrfmiddlewaretoken\" value=\"")[1].split("\"")[0]
        token = html.split("name=\"token\" value=\"")[1].split("\"")[0]

        data = {
                "csrfmiddlewaretoken": crsf_middleware_token,
                "token": token,
                "password": self.download_password,
            }


        out_dir = self.root_path
        # ensure the directory exists
        os.makedirs(out_dir, exist_ok=True)

        pbar = DownloadProgressBar()

        try:
            with s.post(self.download_url+"?dl=1", data=data, stream=True, headers={'Content-Type': 'application/x-www-form-urlencoded'}) as response:
                response.raise_for_status()

                tot_size = int(response.headers['Content-Length'])
                with open(os.path.join(out_dir, "potsdam.zip"), 'wb') as f:
                    for i, chunk in enumerate(response.iter_content(chunk_size=8192)):
                        f.write(chunk)
                        pbar(i,8192,tot_size)
                
        except requests.exceptions.HTTPError as e:
            print('Error while downloading dataset: The server couldn\'t fulfill the request.')
            print('Error code: ', e.code)
            return

        except requests.exceptions.InvalidURL as e:
            print('Error while downloading dataset: Failed to reach a server.')
            print('Reason: ', e.reason)
            return
        
        out_dir = self.root_path
        # unzip
        print("Extracting inner archives...")
        with zipfile.ZipFile(os.path.join(out_dir, "potsdam.zip"), 'r') as zip_ref:
            zip_ref.extract("Potsdam/5_Labels_for_participants.zip", os.path.join(out_dir, "raw"))
            zip_ref.extract("Potsdam/5_Labels_all.zip", os.path.join(out_dir, "raw"))
            zip_ref.extract("Potsdam/3_Ortho_IRRG.zip", os.path.join(out_dir, "raw"))

        print("Extracting train labels...")
        with zipfile.ZipFile(os.path.join(out_dir, "raw", "Potsdam", "5_Labels_for_participants.zip"), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(out_dir,"raw", "5_Labels_for_participants"))

        print("Extracting test labels...")
        with zipfile.ZipFile(os.path.join(out_dir, "raw","Potsdam", "5_Labels_all.zip"), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(out_dir,"raw", "5_Labels_all"))
        
        print("Extracting images...")
        with zipfile.ZipFile(os.path.join(out_dir, "raw","Potsdam", "3_Ortho_IRRG.zip"), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(out_dir,"raw", "3_Ortho_IRRG"))
        
        os.rename(os.path.join(out_dir, "raw", "3_Ortho_IRRG", "3_Ortho_IRRG"), os.path.join(out_dir, "raw", "3_Ortho_IRRG", "3_Ortho_IRRG_1"))
        shutil.move(os.path.join(out_dir, "raw", "3_Ortho_IRRG", "3_Ortho_IRRG_1"), os.path.join(out_dir, "raw"))
        os.removedirs(os.path.join(out_dir, "raw", "3_Ortho_IRRG"))
        os.rename(os.path.join(out_dir, "raw", "3_Ortho_IRRG_1"), os.path.join(out_dir, "raw", "3_Ortho_IRRG"))

        os.rename(os.path.join(out_dir, "raw", "5_Labels_for_participants", "5_Labels_for_participants"), os.path.join(out_dir, "raw", "5_Labels_for_participants", "5_Labels_for_participants_1"))
        shutil.move(os.path.join(out_dir, "raw", "5_Labels_for_participants", "5_Labels_for_participants_1"), os.path.join(out_dir, "raw"))
        os.removedirs(os.path.join(out_dir, "raw", "5_Labels_for_participants"))
        os.rename(os.path.join(out_dir, "raw", "5_Labels_for_participants_1"), os.path.join(out_dir, "raw", "5_Labels_for_participants"))

        images = os.listdir(os.path.join(out_dir, "raw", "3_Ortho_IRRG"))
        labels = os.listdir(os.path.join(out_dir, "raw", "5_Labels_all"))
        labels_train = os.listdir(os.path.join(out_dir, "raw", "5_Labels_for_participants"))

        images, labels, labels_train = [list(filter(lambda x: x.endswith(".tif"), data)) for data in [images, labels, labels_train]]

        train_numbers = [image_number(filename) for filename in labels_train]
        test_numbers = [image_number(filename) for filename in labels if image_number(filename) not in train_numbers]

        os.makedirs(f"{out_dir}/train/images")
        os.makedirs(f"{out_dir}/train/labels")
        os.makedirs(f"{out_dir}/test/images")
        os.makedirs(f"{out_dir}/test/labels")
        os.makedirs(f"{out_dir}/val/images")
        os.makedirs(f"{out_dir}/val/labels")
        print("tiling train images...")
        i = 0
        for full_size_image_number in tqdm(train_numbers):
            image = Image.open(f"{out_dir}/raw/3_Ortho_IRRG/{image_filename(full_size_image_number)}")
            label = Image.open(f"{out_dir}/raw/5_Labels_all/{label_filename(full_size_image_number)}")
            image = np.array(image)
            label = np.array(label)
            image_tiles = tile_image(image)
            label_tiles = tile_image(label)
            save_folder = "train"
            for image_tile, label_tile in zip(image_tiles, label_tiles):
                Image.fromarray(image_tile).save(f"{out_dir}/{save_folder}/images/{i}.png")
                Image.fromarray(label_tile).save(f"{out_dir}/{save_folder}/labels/{i}.png")
                i += 1
        
        print("tiling val images...")
        i = 0
        for full_size_image_number in tqdm(test_numbers):
            image = Image.open(f"{out_dir}/raw/3_Ortho_IRRG/{image_filename(full_size_image_number)}")
            label = Image.open(f"{out_dir}/raw/5_Labels_all/{label_filename(full_size_image_number)}")
            image = np.array(image)
            label = np.array(label)
            image_tiles = tile_image(image)
            label_tiles = tile_image(label)
            save_folder = "val"
            for image_tile, label_tile in zip(image_tiles, label_tiles):
                Image.fromarray(image_tile).save(f"{out_dir}/{save_folder}/images/{i}.png")
                Image.fromarray(label_tile).save(f"{out_dir}/{save_folder}/labels/{i}.png")
                i += 1
        
        os.remove(os.path.join(out_dir, "potsdam.zip"))
        shutil.rmtree(os.path.join(out_dir, "raw"))
    
def tile_image(image, tile_size=512, overlap=128):
    stride = tile_size - overlap
    tiles = []
    for y in range(0, image.shape[0] - tile_size + 1, stride):
        for x in range(0, image.shape[1] - tile_size + 1, stride):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append(tile)
    return tiles

def image_number(filename:str) -> str:
    return filename.split("_")[-3]+"_"+filename.split("_")[-2]

def image_filename(number:str) -> str:
    return f"top_potsdam_{number}_IRRG.tif"

def label_filename(number:str) -> str:
    return f"top_potsdam_{number}_label.tif"
