## Contributing 

We welcome all forms of contributions, including but not limited to the following.

- Introduce new geospatial foundation models
- Incorporate downstream datasets
- Add new decoder heads
- Fix typo or bugs

### Workflow

1. fork and pull the latest repository
2. checkout a new branch (do not use the main branch for PRs)
3. commit your changes
4. create a PR

Note: For significant modifications or any bugs spotting, please consider opening an issue for discussion beforehand.

## Code structure 

### engine
In `engine`, basic modules in the training pipeline are defined including `data_preprocessor`, `trainer` and `evaluator`.
1. `data_preprocessor` selects the bands needed by an encoder and pads unavailable bands with zeros, and different **augmentations**.
2. `trainer` supports mixed precision/distributed training and print training stats and metrics in real time.
3. `evaluator` can be called independently and evaluate a model also in distributed way and compute per class metrics.

### datasets
1. The implementations are simplified and standardized.
2. Dataset metas are read from configs, including newly added classes (name), ignore_index, and so on.
3. Check the example later to quick start contributing.

### encoders
In `encoders`, you can find all the supported (foundation) models.
1. Support multi-stage output that may be needed by segmentors, specified by output layers in encoder config.
2. Check the example later to quick start contributing.

### decoders
In `decoders`, you can find all the supported decoders.
1. The UperNet implementation is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/main)
3. We support UPerNet for unitemporal semantic segmentation, UPerNetCD for change detection and MTUPerNet for multitemporal semantic segmentation
4. for multi-temporal, L-TAE and linear projection are supported

## Adding new features

### Adding a new geospatial foundation model

We have designed the repo to allow for benchmarking your own model with minimal effort. Follow the steps below to integrate your model:

1. **Implement an Encoder Class**:

   - In `pangaea/encoders/`, create a new Python file named after your model (e.g., `my_model_encoder.py`).
   - Implement a class that inherits from `Encoder`. You can check it in `pangaea/encoders/base.py`.
   - Be sure that your dataset is instantiated with all the required parameters from the `Encoder`. You can also add new parameters or fix some parameters from the `Encoder` that are not changing in your model (e.g. `multi_temporal`).
   - Implement the required methods: `__init__`, `load_encoder_weights`, and `forward`.
   - **Example**:

     ```python
     import torch.nn as nn
     
     from pangaea.encoders.base import Encoder
     
     class MyModel(Encoder):
         def __init__(
             self,
             encoder_weights: str | Path,
             input_size: int,
             input_bands: dict[str, list[str]],
             output_layers: int | list[int],
             in_chans: int,              #newly added parameter
         ) -> None:
             super().__init__(
                 model_name="my_model_name",
                 encoder_weights=encoder_weights,
                 input_bands=input_bands,
                 input_size=input_size,
                 embed_dim=768,        # my_model_embed_dim, fixed parameters
                 output_dim=768,       # my_model_output_dim, fixed parameters
                 multi_temporal=False, # wether support multi-temporal, fixed parametersfixed parameters
                 multi_temporal_ouput=False, # wether the output of the model has a temporal dimension
             )
     
            self.in_chans = in_chans    #newly added parameter

             # Initialize your model architecture here
             # For example:
             self.backbone = nn.Sequential(
                 nn.Conv2d(in_chans, 64, kernel_size=3, padding=1),
                 nn.ReLU(),
                 # Add more layers as needed
             )
             # Specify output layers if applicable

         def load_encoder_weights(self, pretrained_path: str) -> None:
             # Load pretrained weights
             state_dict = torch.load(pretrained_path, map_location='cpu')
             self.load_state_dict(state_dict, strict=False)
             print(f"Loaded encoder weights from {pretrained_path}")

         def forward(self, x: dict[str, torch.Tensor]) -> list[torch.Tensor]:
             """Foward pass of the encoder.
 
             Args:
                 x (dict[str, torch.Tensor]): encoder's input structured as a dictionary:
                 x = {modality1: tensor1, modality2: tensor2, ...}, e.g. x = {"optical": tensor1, "sar": tensor2}.
                 If the encoder is multi-temporal (self.multi_temporal==True), input tensor shape is (B C T H W) with C the
                 number of bands required by the encoder for the given modality and T the number of time steps. If the
                 encoder is not multi-temporal, input tensor shape is (B C H W) with C the number of bands required by the
                 encoder for the given modality.
 
             Returns:
                 list[torch.Tensor]: list of the embeddings for each modality. For single-temporal encoders, the list's
                 elements are of shape (B, embed_dim, H', W'). For multi-temporal encoders, the list's elements are of shape
                 (B, C', T, H', W') with T the number of time steps if the encoder does not have any time-merging strategy,
                 else (B, C', H', W') if the encoder has a time-merging strategy (where C'==self.output_dim).
             """
             x = image['optical']
             outputs = []
             # Forward pass through the model
             for idx, layer in enumerate(self.backbone):
                 x = layer(x)
                 if idx in self.output_layers:
                     outputs.append(x)
             return outputs
     ```

2. **Create an Encoder Configuration File**:

   - In `configs/encoder/`, create a new YAML file named after your model (e.g., `my_model.yaml`).
   - Define model-specific parameters, including `encoder_weights`, `input_bands`,`input_size` and any model architecture arguments. Specifically, indicate `_target_` to point out your implemeted model
   - **Example**:

     ```yaml
      _target_: pangaea.encoders.my_model_encoder.MyModel
      encoder_weights: ./pretrained_models/my_model_weights.pth
      download_url: https://path.to.your.model/weights.pth
      
      input_size: 120  
      in_chans: 3
      embed_dim: 768
      patch_size: 16
      num_heads: 12
      depth: 12
      mlp_ratio: 4
      
      input_bands:
        optical:
          - B2
          - B3
          - B4
      
      output_layers:
        - 3
        - 5
        - 7
        - 11
     ```
     
3. **Run Training with Your Model**:
   - Use the `run.py` script with your encoder configuration.
   - **Example Command**:

     ```bash
      torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
      --config-name=train \
      dataset=hlsburnscars \
      encoder=my_model \
      decoder=seg_upernet \
      preprocessing=seg_default \
      criterion=cross_entropy \
      task=segmentation
     ```

### Adding a new downstream dataset

We have designed the repo to allow for using your own datasets with minimal effort. Follow the steps below to integrate your dataset:

1. **Implement a Dataset Class**:

   - In the `pangaea/datasets/` directory, create a new Python file named after your dataset (e.g., `my_dataset.py`).
   - Implement a class that inherits from `RawGeoFMDataset`. You can check it in `pangaea/datasets/base.py`.
   - Be sure that your dataset is instantiated with all the required parameters from the `GeoFMDataset`. You can also add new parameters.
   - Implement the required methods: `__init__`, `__len__`, `__getitem__`, and `download` (if applicable, otherwise a `NotImplementedError is raised`).
   - **Example**:

     ```python
     import torch
     from pangaea.datasets.base import RawGeoFMDataset

     class MyDataset(RawGeoFMDataset):
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
             temp: int, #newly added parameter
         ):
             super(MyDataset, self).__init__(
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

             self.temp = temp #newly added parameter
             # Initialize file lists or data structures here

         def __len__(self):
             # Return the total number of samples
             return len(self.file_list)

         def __getitem__(self, index):
            """Returns the i-th item of the dataset.

            Args:
                i (int): index of the item

            Raises:
                NotImplementedError: raise if the method is not implemented

            Returns:
                dict[str, torch.Tensor | dict[str, torch.Tensor]]: output dictionary follwing the format
                {"image":
                    {
                    "optical": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),
                     "sar": torch.Tensor of shape (C T H W) (where T=1 if single-temporal dataset),
                     },
                "target": torch.Tensor of shape (H W) of type torch.int64 for segmentation, torch.float for
                regression datasets.,
                 "metadata": dict}.
            """
             # Load your data and labels here
             image = ...  # Load image
             target = ...  # Load target label or mask

             # Convert to tensors
             image = torch.tensor(image, dtype=torch.float32)
             target = torch.tensor(target, dtype=torch.long)

             return {
                 'image': {'optical': image},
                 'target': target,
                 'metadata': {}
             }

         @staticmethod
         def download(self, silent=False):
             # Implement if your dataset requires downloading
             pass
     ```
2. **Create a Dataset Configuration File**:

   - Navigate to `configs/dataset/` and create a new YAML file named after your dataset (e.g., `my_dataset.yaml`).
   - Indicate your implemented dataset class in `_target_`.
   - Define all necessary dataset parameters such as `dataset_name`, `root_path`, `img_size`, `bands`, `data_mean`, `data_std`, `num_classes`, and class labels. Check `GeoFMDataset` class for more details in `pangaea/datasets/base.py`.
     
   - **Example**:

     ```yaml
     _target_: pangaea.datasets.utae_dynamicen.DynamicEarthNet
     dataset_name: MyDataset
     root_path: ./data/my_data_dir
     download_url: None
     auto_download: False
     img_size: 256
     multi_temporal: 6
     multi_modal: False
     ignore_index: -1
     num_classes: 3
     classes:
       - Class1
       - Class2
       - Class3
     distribution:
       - 0.2
       - 0.4
       - 0.4
     bands:
       optical:
         - B1
         - B2
         - B3
     data_mean:
       optical:
         - 0.485
         - 0.456
         - 0.404
     data_std:
       optical:
         - 0.229
         - 0.224
         - 0.225
     data_min:
       optical:
         - 0.
         - 0.
         - 0.
     data_max:
       optical:
         - 1.
         - 1.
         - 1.
     ```

3. **Adjust the Augmentation Pipeline**:

   - If your dataset requires specific preprocessing or augmentation, create or modify an augmentation configuration file in `configs/preprocessing/`.
   - Ensure that all preprocessing steps (e.g., normalization, resizing) match your dataset's requirements.
   - If your specific preprocessing or augmentation are not implemented, please implement them in `pangaea/engine/data_preprocessor.py`

4. **Run Training**:
   - Use the `run.py` script with your dataset and augmentation configurations.
   - **Example Command**:

     ```bash
      torchrun --nnodes=1 --nproc_per_node=1 pangaea/run.py \
      --config-name=train \
      dataset=my_dataset \
      encoder=prithvi \
      decoder=seg_upernet_mt_ltae \
      preprocessing=seg_default \
      criterion=cross_entropy \
      task=segmentation
     ```
