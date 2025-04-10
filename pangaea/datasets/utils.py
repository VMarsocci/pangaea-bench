import os
import tqdm
import rasterio
import pathlib
import concurrent.futures
from google.cloud.storage import Client
import zipfile


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

    for blob, result in zip(blobs, results):
        # The results list is either `None` or an exception for each blob in
        # the input list, in order.
        name = blob.name
        if isinstance(result, Exception):
            print("Failed to download {} due to exception: {}".format(name, result))
        else:
            print("Downloaded {} to {}.".format(name, destination_directory + name))


def read_tif(file: pathlib.Path):
    with rasterio.open(file) as dataset:
        arr = dataset.read()  # (bands X height X width)
    return arr.transpose((1, 2, 0))


def read_tif_with_metadata(file: pathlib.Path):
    with rasterio.open(file) as dataset:
        arr = dataset.read()  # (bands X height X width)
        transform = dataset.transform
        crs = dataset.crs
    return arr.transpose((1, 2, 0)), transform, crs


def decompress_zip_with_progress(zip_file_path, extract_to_folder=None):
    """Decompress a zip file with a progress bar and remove the symlink."""
    if extract_to_folder is None:
        extract_to_folder = zip_file_path.parent

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        file_names = zip_ref.namelist()
        total_files = len(file_names)

        # Initialize the progress bar with the total number of files
        with tqdm.tqdm(total=total_files, unit="file", desc=f"Extracting {zip_file_path.name}") as pbar:
            for file in file_names:
                # Extract each file
                zip_ref.extract(file, extract_to_folder)
                # Update the progress bar
                pbar.update(1)

    # remove zip file
    zip_file_path.unlink()