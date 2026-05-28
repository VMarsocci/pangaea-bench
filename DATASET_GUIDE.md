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
- [Open-Canopy](#open-canopy)
- [FLAIR-2 (GEO-Bench-2)](#flair-2-geo-bench-2)
- [SpaceNet 2 (GEO-Bench-2)](#spacenet-2-geo-bench-2)
- [HOT Buildings](#hot-buildings)
- [Geo-Bench Datasets](#geo-bench-datasets)
  - [Multi-label Classification (e.g., m-BigEarthNet)](#for-multi-label-classification-eg-m-bigearthnet)
  - [Single-label Classification (e.g., m-EuroSat, m-Brick-Kiln)](#for-single-label-classification-ie-m-eurosat-m-brick-kiln-m-forestnet-m-pv4ger-m-so2sat)
  - [Semantic Segmentation (e.g., m-NZ-Cattle, m-SA-Crop-Type)](#for-semantic-segmentation-ie-m-cashew-plantation-m-chesapeake-landcover-m-neontree-m-nz-cattle-m-pv4ger-seg-and-m-sa-crop-type)
  - [SpaceNet 3 (roads)](#spacenet-3-roads-native)
  - [PureForest](#pureforest)

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
### Open-Canopy
   ```
    torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
    --config-name=train \
    dataset=opencanopy \
    encoder=dofa \
    decoder=reg_upernet \
    preprocessing=reg_default \
    criterion=mse \
    task=regression
  ```

### FLAIR-2 (GEO-Bench-2)

- Land cover semantic segmentation over 13 classes from IGN aerial imagery, packaged as the GEO-Bench-2 `geobench_flair2.tortilla` artifact.
- Splits, hashes, and class indexing follow [GEO-Bench-2](https://arxiv.org/abs/2511.15658). Splits are stored inside the tortilla file (`tortilla:data_split` column) and are domain-stratified by FLAIR's `D###` domain IDs. Local counts: train 4049, val 1022, test 3022.
- The basic experiment uses RGB only and resizes to encoder input size. To use the aerial NIR band as well, add `B8` to `bands.optical` (mapped to channel index 3 in the aerial TIFF).

  ```
   torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=flair2 \
   encoder=remoteclip \
   decoder=seg_upernet \
   preprocessing=seg_resize \
   criterion=cross_entropy \
   task=segmentation
  ```

### SpaceNet 2 (GEO-Bench-2)

- Building footprint semantic segmentation (3 classes: background slot, no-building, building) over WorldView-3 imagery, packaged as the GEO-Bench-2 `geobench_spacenet2.tortilla` artifact.
- Splits follow [GEO-Bench-2](https://arxiv.org/abs/2511.15658), city-based checkerboard split for spatial separation. Local counts: train 5186, val 1461, test 2961. Native tile size is 650x650.
- Auto-downloads `spacenet2/geobench_spacenet2.tortilla` from `kshitijrajsharma/geobench2-vhr` (~43 GB, mirrors AI Alliance's `aialliance/spacenet2`).
- The mask is shifted by +1 (matching the upstream `geobench_v2` convention) so labels become `{1=no-building, 2=building}` and the class-0 slot is reserved.
- Default config uses RGB (`B4 B3 B2`) extracted from the 8-band Worldview multispectral bands; the panchromatic band is not loaded.

  ```
   torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=spacenet2 \
   encoder=remoteclip \
   decoder=seg_upernet \
   preprocessing=seg_resize \
   criterion=cross_entropy \
   task=segmentation
  ```

### HOT Buildings

- Binary building segmentation (background, building) over RGB VHR aerial + satellite imagery, distributed by HOTOSM.
- Source: [`hotosm/vhr-building-segmentation`](https://huggingface.co/datasets/hotosm/vhr-building-segmentation) on Hugging Face. Splits (`train`, `validation`, `test`) match the upstream parquet shard names. Local counts: train 57890, val 7237, test 7236, all 256x256.
- Auto-downloads parquet shards and `norm_stats.json` via `huggingface_hub.snapshot_download`. Image and mask cells are PNG-encoded inside parquet; pixel values are scaled to 0..1 in the dataset and normalised in the preprocessor.
- The class distribution is not authoritative (no per-class pixel counts shipped upstream), so the YAML uses a placeholder `[0.5, 0.5]` for `distribution`.

  ```
   torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=hot_buildings \
   encoder=remoteclip \
   decoder=seg_upernet \
   preprocessing=seg_resize \
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

### SpaceNet 3 (roads)

- Binary road segmentation over WorldView-3 PS-RGB (0.3 m).
- Loads from the parquet mirror at [`kshitijrajsharma/pangaea2-vhr`](https://huggingface.co/datasets/kshitijrajsharma/pangaea2-vhr) (config `sn3_roads`), pinned to revision `49dfe4f…`. Auto-downloads only the labeled `train` shards (~12 GB).
- Upstream `test` is unlabeled (standard SpaceNet challenge format), so we carve **train / val / test (70 / 20 / 10) from upstream `train`, stratified per-AOI** (Vegas / Paris / Shanghai / Khartoum), `random_state=234`. This mirrors the GEO-Bench-2 SN2 implementation style ([Lacoste et al. 2025, §6.1.11](https://arxiv.org/abs/2511.15658)).
- Images decode as uint16 GeoTIFF via `rasterio.open(io.BytesIO(...))` (PIL would silently truncate uint16 RGB).
- Image size 1300x1300; encoder-time resize handled by preprocessing.

  ```
   torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=spacenet3 \
   encoder=remoteclip \
   decoder=seg_upernet \
   preprocessing=seg_default \
   criterion=cross_entropy \
   task=segmentation
  ```

### PureForest

- Tree-species classification, **13 classes** (single-label) over 0.2 m IGN BD ORTHO aerial NRGB (NIR + R + G + B). Labels follow the official [PureForest dictionary](https://huggingface.co/datasets/IGNF/PureForest/blob/main/metadata/PureForestID-dictionnary.csv), which hierarchically groups 18 tree species into 13 semantic classes.
- Loads from the parquet mirror at [`kshitijrajsharma/pangaea2-vhr`](https://huggingface.co/datasets/kshitijrajsharma/pangaea2-vhr) (config `pureforest`), pinned to revision `89dcb32…`. Splits (`train` / `val` / `test`) are defined by the upstream shard names.
- The mirror ships PNGs in channel order `(NIR, R, G, B)`; band selection is driven by `bands.optical` in the YAML (default RGB = `B4 B3 B2`, mapped to channel indices `[1, 2, 3]`).
- The parquet `label` column is the official 13-class `class_index` (0..12); the underlying 18-species name remains available as `class_name` and `species` columns for traceability.

  ```
   torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
   --config-name=train \
   dataset=pureforest \
   encoder=remoteclip \
   decoder=cls_linear \
   preprocessing=cls_resize \
   criterion=cross_entropy \
   task=linear_classification \
   finetune=false
  ```