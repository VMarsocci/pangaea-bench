"""Structured config schema for Pangaea.

To add a new top-level config parameter:
    1. Add it here with the correct Python type.
    2. Set the matching value in ``configs/train.yaml`` and/or
       ``configs/test.yaml`` (the YAML always wins over the default
       defined here).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PangaeaConfig:
    # Runtime                                                           
    train: bool = True
    seed: int = 234
    work_dir: str = "./data/results/"

    # Logging                                                           
    use_wandb: bool = False
    wandb_run_id: Optional[str] = None

    # Data loading                                                      
    num_workers: int = 4
    batch_size: int = 8
    test_num_workers: int = 4
    test_batch_size: int = 1

    # Experiment / fine-tuning
    finetune: bool = False
    ckpt_dir: Optional[str] = None
    limited_label_train: float = 1.0
    limited_label_val: float = 1.0
    limited_label_strategy: str = "stratified"
    stratification_bins: int = 3
    data_replicate: int = 1
    use_final_ckpt: bool = False

    # Cross-testing
    cross_test_dataset: Optional[str] = None
    cross_test_root_path: Optional[str] = None
    cross_test_preprocessing: Optional[str] = None
    test_log_name: Optional[str] = None

    # Prediction output
    save_predictions: bool = False
    save_visualizations: bool = False
    save_output_dir: Optional[str] = None
    max_samples_to_visualize: int = -1
    overwrite: bool = True
