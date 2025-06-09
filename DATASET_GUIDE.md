# Dataset Guide

This document provides a detailed overview of the datasets used in this repository. For each dataset, you will find instructions on how to prepare the data, along with command-line examples for running models. 

*DISCLAIMER*: please consider that we provide the detailed overview for the datasets included in the original repo. Community-contributed datasets may not come with pre-defined command-line examples in this repository. Feel free to adapt the existing examples based on your use case. 
## 📚 Table of Contents

- [HLSBurnScars](#hlsburnscars)
- [MADOS](#mados)
- [PASTIS-R](#pastis-r)
- [Sen1Floods11](#sen1floods11)
- [xView2](#xview2)
- [FiveBillionPixels](#fivebillionpixels)
- [DynamicEarthNet](#dynamicearthnet)
- [Crop Type Mapping (South Sudan)](#crop-type-mapping-south-sudan)
- [SpaceNet 7](#spacenet-7)
- [AI4SmallFarms](#ai4smallfarms)
- [BioMassters](#biomassters)

### 🧪 Community-Contributed Datasets
- [Potsdam](#potsdam)
- [Geo-Bench Datasets](#geo-bench-datasets)
  - [Multi-label Classification (e.g., m-BigEarthNet)](#for-multi-label-classification-eg-m-bigearthnet)
  - [Single-label Classification (e.g., m-EuroSat, m-Brick-Kiln)](#for-single-label-classification-ie-m-eurosat-m-brick-kiln-m-forestnet-m-pv4ger-m-so2sat)
  - [Semantic Segmentation (e.g., m-NZ-Cattle, m-SA-Crop-Type)](#for-semantic-segmentation-ie-m-cashew-plantation-m-chesapeake-landcover-m-neontree-m-nz-cattle-m-pv4ger-seg-and-m-sa-crop-type)

---

### HLSBurnScars

- The code supports automatic downloading of the dataset into `./data` folder. 
- The basic experiment uses mean and std values for normalization and applies random cropping to align images with the size used for GFMs pretraining.
   Below is a CLI example for running the experiment with the RemoteClip pretrained encoder and UperNet segmentation decoder:

  ```
   torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=hlsburnscars \
   encoder=remoteclip \
   decoder=seg_upernet\
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation
  ```
  
### MADOS

- The code supports automatic downloading of the dataset into `./data` folder. 
- Random cropping to encoder size is done with focus cropping. This avoids batches with no loss, caused by the high ratio of unlabeled pixels ion the dataset.
- The basic experiment uses mean and std values for normalization and applies random cropping to align images with the size used for GFMs pretraining.
   Below is a CLI example for running the experiment with the RemoteClip pretrained encoder and UperNet segmentation decoder:

  ```
   torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=mados \
   encoder=remoteclip \
   decoder=seg_upernet\
   preprocessing=seg_focus_crop \
   criterion=cross_entropy \
   task=segmentation
  ```
  
### PASTIS-R

- The code supports automatic downloading of the dataset into `./data` folder.
- Images are 128x128 patches, so a resize is needed to match input_size requirements of the encoders.
- For models that don't support multi-temporal data, each time frame is processed separately for feature extraction and then mapped into a single representation. This setup requires the configuration file `configs/decoder/seg_upernet_mt_ltae.yaml`. Additionally, in the dataset configuration, specify the number of time frames, for example, `multi_temporal: 6`. Below is a CLI example for running the experiment using the RemoteCLIP pretrained encoder and multi-temporal UPerNet with L-TAE processing of temporal information:

  ```
  torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=pastis \
   encoder=remoteclip \
   decoder=seg_upernet_mt_ltae \
   preprocessing=seg_resize \
   criterion=cross_entropy \
   task.evaluator.inference_mode=whole \
   task=segmentation
  ```
  
###  Sen1Floods11

- The code supports automatic downloading of the dataset into `./data` folder. 
- The basic experiment uses mean and std values for normalization and applies random cropping to align images with the size used for GFMs pretraining.
   Below is a CLI example for running the experiment with the RemoteClip pretrained encoder and UperNet segmentation decoder:

  ```
   torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=sen1floods11 \
   encoder=remoteclip \
   decoder=seg_upernet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation
  ```
  
### xView2

- The dataset needs to be downloaded manually from the official website. This requires a registration and accepting the terms and conditions. On the download page, we need the datasets under `Datasets from the Challenge`, excluding the holdout set. Extract the datasets in the `./data/xView2/` folder, such that it contains e.g. `./data/xView2/tier3/images/...`.
- The `tier3` set does not come up labels in the form of images, so we first need to create them from the respective JSON data. We create a `masks` folder on the level of the `images` folder by running:

  ```
   python datasets/xView2_create_masks.py
   ```
- The basic experimental setup for this dataset is a change detection task. Two images showing the same location are encoded using a foundation model as encoder. A smaller UPerNet model is trained to compute the 5-class segmentation mask from these encodings. Below is a CLI example for running the experiment with the Prithvi pretrained encoder:
   ```
   torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=xview2 \
   encoder=prithvi \
   decoder=seg_siamupernet_conc \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=change_detection
   ```

### FiveBillionPixels

- The code is not supporting the automatic download. It will come soon.
- With respect to the original datasets, the images were cropped in 520x520 tiles, without overlapping. Few pixels on the borders of each original image got lost. The new class distribution is visible in the respective config.
- In the config, you can specify if using the CMYK or the NIR-RGB colour distribution. 
- The basic experiment uses mean and std values for normalization and applies random cropping to align images with the size used for GFMs pretraining.
   Below is a CLI example for running the experiment with the RemoteClip pretrained encoder and UperNet segmentation decoder:

  ```
   torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=fivebillionpixels \
   encoder=remoteclip \
   decoder=seg_upernet\
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation
  ```

### DynamicEarthNet

- The code supports automatic downloading of the dataset into `./data` folder.
- The basic experimental setup for this dataset is a multi-temporal semantic segmentation task. For models that don't support multi-temporal data, each time frame is processed separately for feature extraction and then mapped into a single representation. This setup requires the configuration file `configs/decoder/seg_upernet_mt_ltae.yaml` or `configs/decoder/seg_upernet_mt_linear.yaml`. Additionally, in the dataset configuration, specify the number of time frames, for example, `multi_temporal: 6`. Below is a CLI example for running the experiment using the RemoteCLIP pretrained encoder and multi-temporal UPerNet with L-TAE processing of temporal information:

  ```
  torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=dynamicen \
   encoder=remoteclip \
   decoder=seg_upernet_mt_ltae \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation
  ```
  
###  Crop Type Mapping (South Sudan)

- The code supports automatic downloading of the dataset into `./data` folder.
- Images are 64x64 patches, so a resize is needed to match input_size requirements of the encoders.
- The original dataset contains corrupted files, which are skipped during the experiment. We follow the dataset paper to use the most frequent 4 classes and the others are ignored.
- The basic experimental setup for this dataset is a multi-temporal multi-modal semantic segmentation task. For models that don't support multi-temporal data, each time frame is processed separately for feature extraction and then mapped into a single representation. This setup requires the configuration file `configs/decoder/seg_upernet_mt_linear.yaml` or `configs/decoder/seg_upernet_mt_ltae.yaml`. Additionally, in the dataset configuration, specify the number of time frames, for example, `multi_temporal: 6`, where the latest six images are selected for both optical and SAR data. Below is a CLI example for running the experiment using the RemoteCLIP pretrained encoder and multi-temporal UPerNet with L-TAE processing of temporal information:

  ```
  torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=croptypemapping \
   encoder=remoteclip \
   decoder=seg_upernet_mt_ltae \
   preprocessing=seg_resize \
   criterion=cross_entropy \
   task=segmentation \
   task.evaluator.inference_mode=whole 
  ```

### SpaceNet 7

- The code supports automatic downloading of the dataset into `./data` folder.
- The basic experiment uses mean and std values for normalization and applies random cropping to align images with the size used for GFMs pretraining.
- The dataset supports building mapping and change detection.
- Below is a CLI example for running the building mapping (i.e. single temporal semantic segmentation) experiment with the RemoteClip pretrained encoder and UperNet segmentation decoder:

  ```
   torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=spacenet7 \
   encoder=remoteclip \
   decoder=seg_upernet \
   preprocessing=seg_default \
   criterion=dice \
   task=segmentation
  ```
- Here is an example to run change detection:
    ```
   torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=spacenet7cd \
   encoder=remoteclip \
   decoder=seg_siamupernet_conc \
   preprocessing=seg_default \
   criterion=dice \
   task=change_detection
  ```

###  AI4SmallFarms

- The code supports automatic downloading of the dataset into `./data` folder.
- The original dataset contains vector files as well as Google Maps (GM) files, which are skipped during the experiment. Only the .tif Sentinel-2 images and delineation labels are kept after downloading.
- The dataset is uni-temporal, and the labels contain only two classes (farm boundary or background). For training using the RemoteCLIP encoder, the following command should be used:

  ```
   torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=ai4smallfarms \
   encoder=remoteclip \
   decoder=seg_upernet \
   preprocessing=seg_default \
   criterion=dice \
   task=segmentation \
   data_replicate=2 \
   task.trainer.best_metric_key=IoU
  ```
  
### BioMassters
- The code is not supporting the automatic download. It will come soon.
- The dataset is multi-modal and multi-temporal, so a default command of using DOFA model is:
  ```
   torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=biomassters \
   encoder=dofa \
   decoder=reg_upernet_mt_ltae \
   preprocessing=reg_default \
   criterion=mse \
   task=regression
  ```
- If you want to try single temporal regression, you can use:
    ```
    torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=biomassters \
   encoder=dofa \
   decoder=reg_upernet \
   preprocessing=reg_default \
   criterion=mse \
   task=regression
   ```
  In this case, you can specify in the `temp` parameter which frame you want to use.

---
**Note**: The following datasets are **community-contributed** and are not part of the original benchmark repository. 
### Potsdam
   ```
   torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=potsdam \
   encoder=dofa \
   decoder=seg_upernet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation
  ```
### Geo-Bench Datasets 
Note that `export GEO_BENCH_DIR=YOUR/PATH/DIR` is required.
-  For multi-label linear classification, e.g., m-BigEarthNet
    ```
    export GEO_BENCH_DIR=YOUR/PATH/DIR
    torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py  \
      --config-name=train \
      dataset=mbigearthnet \
      encoder=dofa  \
      decoder=cls_linear  \
      preprocessing=cls_resize \
      criterion=binary_cross_entropy \
      task=linear_classification_multi_label \
      finetune=false
    ```

-  For single-label linear classification, i.e., m-EuroSat, m-Brick-Kiln, m-ForestNet, m-PV4Ger, m-So2Sat
    ```
      export GEO_BENCH_DIR=YOUR/PATH/DIR
      torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py  \
        --config-name=train \
        dataset=meurosat \
        encoder=remoteclip  \
        decoder=cls_linear  \
        preprocessing=cls_resize \
        criterion=cross_entropy \
        task=linear_classification \
        finetune=false
      ```

-  For semantic segmentation, i.e., m-Cashew-Plantation, m-Chesapeake-Landcover, m-NeonTree, m-NZ-Cattle, m-PV4Ger-Seg and m-SA-Crop-Type
    ```
      export GEO_BENCH_DIR=YOUR/PATH/DIR
      torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py  \
        --config-name=train \
        dataset=mnz-cattle \
        encoder=dofa  \
        decoder=seg_upernet  \
        preprocessing=seg_default \
        criterion=cross_entropy \
        task=segmentation \
        finetune=false
      ```

-  For KNN probe classification, i.e., m-EuroSat, m-Brick-Kiln, m-ForestNet, m-PV4Ger, m-So2Sat
    ```
      export GEO_BENCH_DIR=YOUR/PATH/DIR
      torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py  \
        --config-name=train \
        dataset=meurosat \
        encoder=remoteclip  \
        decoder=cls_knn  \
        preprocessing=cls_resize \
        criterion=none \
        task=knn_probe \
        batch_size=32 \
        finetune=false
      ```
    Note that for KNN probe:
    - The criterion is set to `none` since no training is performed
    - The batch size can be larger since we're only doing inference
    - `finetune` is set to `false` as we're only using the pre-trained encoder

-  For multi-label KNN probe classification, i.e., m-BigEarthNet
    ```
      export GEO_BENCH_DIR=YOUR/PATH/DIR
      torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py  \
        --config-name=train \
        dataset=mbigearthnet \
        encoder=remoteclip  \
        decoder=cls_knn_multilabel  \
        preprocessing=cls_resize \
        criterion=none \
        task=knn_probe_multi_label \
        batch_size=32 \
        finetune=false
      ```
    Note that for multi-label KNN probe:
    - The criterion is set to `none` since no training is performed
    - The batch size can be larger since we're only doing inference
    - `finetune` is set to `false` as we're only using the pre-trained encoder
    - The task is set to `knn_probe_multi_label` to handle multiple labels per sample