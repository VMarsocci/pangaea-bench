import torch
import torch.nn.functional as F

import numpy as np
import pdb
import logging


class DataPreprocessor(torch.utils.data.Dataset):
    def __init__(self, dataset, args, encoder_cfg, dataset_cfg):
        self.dataset = dataset
        self.encoder_config = encoder_cfg
        self.data_config = dataset_cfg
        self.root_path = dataset_cfg['root_path']
        self.data_mean = dataset_cfg['data_mean']
        self.data_std = dataset_cfg['data_std']
        self.classes = dataset_cfg['classes']
        self.class_num = len(self.classes)
        self.split = dataset.split

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class SegPreprocessor(DataPreprocessor):

    def __init__(self, dataset, args, encoder_cfg, dataset_cfg):
        super().__init__(dataset, args, encoder_cfg, dataset_cfg)

        self.preprocessor = {}
        self.preprocessor['optical'] = OpticalShapeAdaptor(args, encoder_cfg, dataset_cfg)
        self.preprocessor['rgb'] = RGBShapeAdaptor(args, encoder_cfg, dataset_cfg)
        # TO DO: other modalities


    def __getitem__(self, index):
        data = self.dataset[index]
        image = {}

        for k, v in data['image'].items():
            # print(k, v)
            image[k] = self.preprocessor[k](v)

        target = data['target'].long()


        # print(image["optical"].size())
        # print(image["rgb"].size())

        return image, target


class OpticalShapeAdaptor():
    def __init__(self, args, encoder_cfg, dataset_cfg):
        self.dataset_bands = dataset_cfg["bands"]['optical']
        self.input_bands = encoder_cfg["input_bands"]['optical']
        self.input_size = encoder_cfg["input_size"]
        self.temporal_input = encoder_cfg["temporal_input"]

        self.used_bands_mask = torch.tensor([b in self.input_bands for b in self.dataset_bands], dtype=torch.bool)
        self.avail_bands_mask = torch.tensor([b in self.dataset_bands for b in self.input_bands], dtype=torch.bool)
        # self.used_bands_indices = torch.tensor([self.dataset_bands.index(b) if b in self.input_bands else -1 for b in self.dataset_bands], dtype=torch.long)
        # self.avail_bands_indices = torch.tensor([self.dataset_bands.index(b) if b in self.dataset_bands else None for b in self.input_bands])
        self.avail_bands_indices = torch.tensor([self.dataset_bands.index(b) if b in self.dataset_bands else -1 for b in self.input_bands], dtype=torch.long)
                
        self.need_padded = self.avail_bands_mask.sum() < len(self.input_bands)

        self.logger = logging.getLogger()

        self.logger.info("Available bands in dataset: {}".format(' '.join(str(b) for b in self.dataset_bands)))
        self.logger.info("Required bands in encoder: {}".format(' '.join(str(b) for b in self.input_bands)))
        if self.need_padded:
            self.logger.info("Unavailable bands {} are padded with zeros".format(
                ' '.join(str(b) for b in np.array(self.input_bands)[self.avail_bands_mask.logical_not()])))


    def __call__(self, optical_image):
        padded_image = torch.cat([torch.zeros_like(optical_image[0: 1]), optical_image], dim=0)
        optical_image = padded_image[self.avail_bands_indices + 1]
        optical_image = F.interpolate(optical_image.unsqueeze(0), (self.input_size, self.input_size), mode='bilinear', align_corners=False)
        optical_image = optical_image.squeeze(0)

        if self.temporal_input and len(optical_image.shape) == 3:
            optical_image = optical_image.unsqueeze(1)

        return optical_image
    

class RGBShapeAdaptor():
    def __init__(self, args, encoder_cfg, dataset_cfg):
        self.dataset_bands = dataset_cfg["bands"]['optical']
        self.input_bands = encoder_cfg["input_bands"]['rgb']
        self.input_size = encoder_cfg["input_size"]
        self.temporal_input = encoder_cfg["temporal_input"]

        self.used_bands_mask = torch.tensor([b in self.input_bands for b in self.dataset_bands], dtype=torch.bool)
        self.avail_bands_mask = torch.tensor([b in self.dataset_bands for b in self.input_bands], dtype=torch.bool)
        # self.used_bands_indices = torch.tensor([self.dataset_bands.index(b) if b in self.input_bands else -1 for b in self.dataset_bands], dtype=torch.long)
        # self.avail_bands_indices = torch.tensor([self.dataset_bands.index(b) if b in self.dataset_bands else None for b in self.input_bands])
        self.avail_bands_indices = torch.tensor([self.dataset_bands.index(b) if b in self.dataset_bands else -1 for b in self.input_bands], dtype=torch.long)
                
        self.need_padded = self.avail_bands_mask.sum() < len(self.input_bands)

        self.logger = logging.getLogger()

        self.logger.info("Available bands in dataset: {}".format(' '.join(str(b) for b in self.dataset_bands)))
        self.logger.info("Required bands in encoder: {}".format(' '.join(str(b) for b in self.input_bands)))
        if self.need_padded:
            self.logger.info("Unavailable bands {} are padded with zeros".format(
                ' '.join(str(b) for b in np.array(self.input_bands)[self.avail_bands_mask.logical_not()])))


    def __call__(self, optical_image):
        padded_image = torch.cat([torch.zeros_like(optical_image[0: 1]), optical_image], dim=0)
        optical_image = padded_image[self.avail_bands_indices + 1]
        optical_image = F.interpolate(optical_image.unsqueeze(0), (self.input_size, self.input_size), mode='bilinear', align_corners=False)
        optical_image = optical_image.squeeze(0)

        if self.temporal_input and len(optical_image.shape) == 3:
            optical_image = optical_image.unsqueeze(1)

        return optical_image



