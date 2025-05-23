import random
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from pangaea.datasets.base import GeoFMDataset
from pangaea.datasets.base import GeoFMSubset

# Calculate image-wise class distributions for segmentation
def calculate_class_distributions(dataset: GeoFMDataset|GeoFMSubset):
    num_classes = dataset.num_classes
    ignore_index = dataset.ignore_index
    class_distributions = []

    for idx in tqdm(range(len(dataset)), desc="Calculating class distributions per sample"):
        target = dataset[idx]['target']

        if ignore_index is not None:
            target=target[(target != ignore_index)]

        total_pixels = target.numel()
        if total_pixels == 0:
            class_distributions.append([0] * num_classes)
            continue
        else:
            class_counts = [(target == i).sum().item() for i in range(num_classes)]
            class_ratios = [count / total_pixels for count in class_counts]
            class_distributions.append(class_ratios)

    return np.array(class_distributions)


# Calculate image-wise distributions for regression
def calculate_regression_distributions(dataset: GeoFMDataset|GeoFMSubset):
    distributions = []

    for idx in tqdm(range(len(dataset)), desc="Calculating regression distributions per sample"):
        target = dataset[idx]['target']
        mean_value = target.mean().item()  # Example for patch-wise mean; adjust as needed for other metrics
        distributions.append(mean_value)

    return np.array(distributions)


# Function to bin class distributions using ceil
def bin_class_distributions(class_distributions, num_bins=3, logger=None):
    logger.info(f"Class distributions are being binned into {num_bins} categories using ceil")
    
    bin_edges = np.linspace(0, 1, num_bins + 1)[1]
    binned_distributions = np.ceil(class_distributions / bin_edges).astype(int) - 1
    return binned_distributions



# Function to bin regression distributions
def bin_regression_distributions(regression_distributions, num_bins=3, logger=None):
    logger.info(f"Regression distributions are being binned into {num_bins} categories")
    # Define the range for binning based on minimum and maximum values in regression distributions
    binned_distributions = np.digitize(
        regression_distributions, 
        np.linspace(regression_distributions.min(), regression_distributions.max(), num_bins + 1)
    ) - 1
    return binned_distributions

def balance_cls_indices(
         dataset:GeoFMDataset|GeoFMSubset, 
         strategy, 
         label_fraction=1.0, 
         logger=None):
     
     indices_by_class = defaultdict(list)
     
     n_samples = len(dataset)
     for idx in range(n_samples):
         label = dataset[idx]['target']
         indices_by_class[label].append(idx)
         
     selected_idx = []
     # For each class, sample the same fraction of indices
     if strategy == "stratified":
         for label, indices in indices_by_class.items():
             num_to_select = max(1, int(len(indices) * label_fraction))
             selected_idx.extend(random.sample(indices, num_to_select))
     else:
         raise NotImplementedError
     
     other_idx = list(set(range(len(dataset))) - set(selected_idx))
     
     return selected_idx, other_idx


def balance_seg_indices(
        dataset:GeoFMDataset|GeoFMSubset, 
        strategy, 
        label_fraction=1.0, 
        num_bins=3, 
        logger=None):
    """
    Balances and selects indices from a segmentation dataset based on the specified strategy.

    Args:
    dataset : GeoFMDataset | GeoFMSubset
        The dataset from which to select indices, typically containing geospatial segmentation data.
    
    strategy : str
        The strategy to use for selecting indices. Options include:
        - "stratified": Proportionally selects indices from each class bin based on the class distribution.
        - "oversampled": Prioritizes and selects indices from bins with lower class representation.
    
    label_fraction : float, optional, default=1.0
        The fraction of labels (indices) to select from each class or bin. Values should be between 0 and 1.
    
    num_bins : int, optional, default=3
        The number of bins to divide the class distributions into, used for stratification or oversampling.
    
    logger : object, optional
        A logger object for tracking progress or logging messages (e.g., `logging.Logger`)

    ------
    
    Returns:
    selected_idx : list of int
        The indices of the selected samples based on the strategy and label fraction.

    other_idx : list of int
        The remaining indices that were not selected.

    """
    # Calculate class distributions with progress tracking
    class_distributions = calculate_class_distributions(dataset)

    # Bin the class distributions
    binned_distributions = bin_class_distributions(class_distributions, num_bins=num_bins, logger=logger)
    combined_bins = np.apply_along_axis(lambda row: ''.join(map(str, row)), axis=1, arr=binned_distributions)

    indices_per_bin = {}
    for idx, bin_id in enumerate(combined_bins):
        if bin_id not in indices_per_bin:
            indices_per_bin[bin_id] = []
        indices_per_bin[bin_id].append(idx)

    if strategy == "stratified":
        # Select a proportion of indices from each bin   
        selected_idx = []
        for bin_id, indices in indices_per_bin.items():
            num_to_select = int(max(1, len(indices) * label_fraction))  # Ensure at least one index is selected
            selected_idx.extend(np.random.choice(indices, num_to_select, replace=False))
    elif strategy == "oversampled":
        # Prioritize the bins with the lowest values
        sorted_indices = np.argsort(combined_bins)
        selected_idx = sorted_indices[:int(len(dataset) * label_fraction)]

    # Determine the remaining indices not selected
    other_idx = list(set(range(len(dataset))) - set(selected_idx))

    return selected_idx, other_idx


def balance_reg_indices(
        dataset:GeoFMDataset|GeoFMSubset, 
        strategy, 
        label_fraction=1.0, 
        num_bins=3, 
        logger=None):

    """
    Balances and selects indices from a regression dataset based on the specified strategy.

    Args:
    dataset : GeoFMDataset | GeoFMSubset
        The dataset from which to select indices, typically containing geospatial regression data.
    
    strategy : str
        The strategy to use for selecting indices. Options include:
        - "stratified": Proportionally selects indices from each bin based on the binned regression distributions.
        - "oversampled": Prioritizes and selects indices from bins with lower representation.
    
    label_fraction : float, optional, default=1.0
        The fraction of indices to select from each bin. Values should be between 0 and 1.
    
    num_bins : int, optional, default=3
        The number of bins to divide the regression distributions into, used for stratification or oversampling.
    
    logger : object, optional
        A logger object for tracking progress or logging messages (e.g., `logging.Logger`). If None, no logging is performed.
    
    ------
    
    Returns:
    selected_idx : list of int
        The indices of the selected samples based on the strategy and label fraction.

    other_idx : list of int
        The remaining indices that were not selected.

    """

    regression_distributions = calculate_regression_distributions(dataset)
    binned_distributions = bin_regression_distributions(regression_distributions, num_bins=num_bins, logger=logger)

    indices_per_bin = {i: [] for i in range(num_bins)}

    # Populate the indices per bin
    for index, bin_index in enumerate(binned_distributions):
        if bin_index in indices_per_bin:
            indices_per_bin[bin_index].append(index)
    
    if strategy == "stratified":
        # Select fraction of indices from each bin
        selected_idx = []
        for bin_index, indices in indices_per_bin.items():
            num_to_select = int(max(1, len(indices) * label_fraction))  # Ensure at least one index is selected
            selected_idx.extend(np.random.choice(indices, num_to_select, replace=False))
    elif strategy == "oversampled":
        # Prioritize bins with underrepresented values (e.g., high biomass samples)
        sorted_indices = np.argsort(binned_distributions)
        selected_idx = sorted_indices[:int(len(dataset) * label_fraction)]

    other_idx = list(set(range(len(dataset))) - set(selected_idx))

    return selected_idx, other_idx


# Function to get subset indices based on the strategy, supporting both classification and regression
def get_subset_indices(dataset: GeoFMDataset, 
                       task="segmentation",
                       strategy="random", 
                       label_fraction=0.5, 
                       num_bins=3, 
                       logger=None):
    logger.info(
        f"Creating a subset of the {dataset.split} dataset using {strategy} strategy, with {label_fraction * 100}% of labels utilized."
    )
    assert strategy in ["random", "stratified", "oversampled"], "Unsupported dataset subsampling strategy"
    
    if strategy == "random":
        n_samples = len(dataset)
        indices = random.sample(
            range(n_samples), int(n_samples * label_fraction)
        )
        return indices
    
    elif task == "classification" or task == "classification_multi_label":
         indices, _ = balance_cls_indices(
             dataset, strategy=strategy, label_fraction=label_fraction, logger=logger
         )
    
    elif task == "segmentation" or task == "change_detection":
        indices, _ = balance_seg_indices(
            dataset, strategy=strategy, label_fraction=label_fraction, num_bins=num_bins, logger=logger
        )
    elif task == "regression":
        indices, _ = balance_reg_indices(
            dataset, strategy=strategy, label_fraction=label_fraction, num_bins=num_bins, logger=logger
        )
    
    return indices


