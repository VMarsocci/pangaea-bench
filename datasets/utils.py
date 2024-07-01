import os
import tqdm
import importlib
import logging
import concurrent.futures
from google.cloud.storage import Client
from utils.utils import load_class  


# Utility progress bar handler for urlretrieve
class DownloadProgressBar:
    def __init__(self):
        self.pbar = None
    
    def __call__(self, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm.tqdm(desc="Downloading...", total=total_size, unit="b", unit_scale=True, unit_divisor=1024)

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded - self.pbar.n)
        else:
            self.pbar.close()
            self.pbar = None


def download_bucket_concurrently(bucket_name, destination_directory=""):
    """Adapted from: https://cloud.google.com/storage/docs/samples/storage-transfer-manager-download-bucket#storage_transfer_manager_download_bucket-python
    Download all of the blobs in a bucket, concurrently in a thread pool.

    The filename of each blob once downloaded is derived from the blob name and
    the `destination_directory `parameter.

    Directories will be created automatically as needed, for instance to
    accommodate blob names that include slashes.
    """

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The directory on your computer to which to download all of the files. This
    # string is prepended (with os.path.join()) to the name of each blob to form
    # the full path. Relative paths and absolute paths are both accepted. An
    # empty string means "the current working directory". Note that this
    # parameter allows accepts directory traversal ("../" etc.) and is not
    # intended for unsanitized end user input.
    # destination_directory = ""

    storage_client = Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)

    blobs = list(bucket.list_blobs())
    print(f"Downloading {len(blobs)} files from GSC...")

    # results = transfer_manager.download_many_to_path(
    #     bucket,
    #     blob_names,
    #     destination_directory=destination_directory,
    #     max_workers=workers,
    #     raise_exception=True
    # )

    blob_file_pairs = []

    for blob in blobs:
        path = os.path.join(destination_directory, blob.name)
        directory, _ = os.path.split(path)
        os.makedirs(directory, exist_ok=True)
        blob_file_pairs.append((blob, path))

    def download_blob_file_pair(blob_file_pair):
        blob, path = blob_file_pair
        return blob.download_to_filename(path)

    results = []
    with tqdm.tqdm(total=len(blob_file_pairs)) as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(download_blob_file_pair, arg): arg for arg in blob_file_pairs}
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                pbar.update(1)

    # for blob, path in tqdm.tqdm(blob_file_pairs):
    #     result = blob.download_to_filename(path)
    #     results.append(result)

    for blob, result in zip(blobs, results):
        # The results list is either `None` or an exception for each blob in
        # the input list, in order.
        name = blob.name
        if isinstance(result, Exception):
            print("Failed to download {} due to exception: {}".format(name, result))
        else:
            print("Downloaded {} to {}.".format(name, destination_directory + name))


def make_dataset(dataset_config):
    dataset = load_class(dataset_config['dataset'])

    if hasattr(dataset, 'download') and callable(dataset.download):
        dataset.download(dataset_config, silent=True)
    else:
        logging.getLogger().warning(f"Dataset {dataset_config['dataset']} doesn't implement autodownload, you might have to download it manually.")

    if hasattr(dataset, 'get_splits') and callable(dataset.get_splits):
        return dataset.get_splits(dataset_config)
    else:
        raise TypeError(f"Please make sure your dataset {dataset_config['dataset']} implements a get_splits method.")
