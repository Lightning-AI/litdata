import hashlib
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from litdata.constants import _DEFAULT_CACHE_DIR, _INDEX_FILENAME
from litdata.streaming.downloader import get_downloader_cls
from litdata.streaming.item_loader import BaseItemLoader, TokensLoader
from litdata.streaming.resolver import Dir
from litdata.utilities.subsample import my_subsampled_filenames_and_roi, shuffle_lists_together


def subsample_streaming_dataset(
    input_dir: Dir,
    item_loader: Optional[BaseItemLoader] = None,
    subsample: float = 1.0,
    shuffle: bool = False,
    seed: int = 42,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Subsample streaming dataset.

    But before doing that, we will do some preprocessing:
    - Make sure input_dir contains cache path and remote url.
    - Check if `index.json` file exists in cache path.
    - If not, download from remote url. If remote url doesn't contain `index.json` file, raise error.
    - Once download, load chunks from `index.json` file.
    - Once chunks are ready, subsample (chunk filenames, region_of_interest).

    """
    my_subsampled_files: List[str] = []
    my_roi: List[Tuple[int, int]] = []

    # Make sure input_dir contains cache path and remote url
    if _should_replace_path(input_dir.path):
        cache_path = _try_create_cache_dir(input_dir=input_dir.path if input_dir.path else input_dir.url)
        if cache_path is not None:
            input_dir.path = cache_path

    assert input_dir.path is not None

    cache_index_filepath = os.path.join(input_dir.path, _INDEX_FILENAME)

    # Check if `index.json` file exists in cache path
    if not os.path.exists(cache_index_filepath) and isinstance(input_dir.url, str):
        assert input_dir.url is not None
        downloader = get_downloader_cls(input_dir.url, input_dir.path, [])
        downloader.download_file(os.path.join(input_dir.url, _INDEX_FILENAME), cache_index_filepath)

    if os.path.exists(os.path.join(input_dir.path, _INDEX_FILENAME)):
        # load chunks from `index.json` file
        with open(os.path.join(input_dir.path, _INDEX_FILENAME)) as f:
            data = json.load(f)
            original_chunks = data["chunks"]
    else:
        raise ValueError(
            f"The provided dataset `{input_dir.path}` doesn't contain any {_INDEX_FILENAME} file."
            " HINT: Did you successfully optimize a dataset to the provided `input_dir`?"
        )

    assert len(original_chunks) > 0, f"No chunks found in the `{input_dir}/index.json` file"

    # create a (chunk_start, chunk_end) list to indicate our subsample from where we can read.
    my_roi = generate_roi(original_chunks, item_loader)

    # shuffle lists together
    if shuffle and not math.isclose(subsample, 1.0):
        # checking if subsample is 1, as if user wants complete data, then let, shuffler and sampler do the work
        original_chunks, my_roi = shuffle_lists_together(original_chunks, my_roi, seed)

    num_items_to_subsample = int(sum([roi[1] - roi[0] for roi in my_roi]) * subsample)

    my_subsampled_files, my_roi, _, _ = my_subsampled_filenames_and_roi(original_chunks, my_roi, target)

    return my_subsampled_files, my_roi


def _should_replace_path(path: Optional[str]) -> bool:
    """Whether the input path is a special path to be replaced."""
    if path is None or path == "":
        return True

    return path.startswith("/teamspace/datasets/") or path.startswith("/teamspace/s3_connections/")


def _try_create_cache_dir(input_dir: Optional[str]) -> Optional[str]:
    hash_object = hashlib.md5((input_dir or "").encode())  # noqa: S324
    if "LIGHTNING_CLUSTER_ID" not in os.environ or "LIGHTNING_CLOUD_PROJECT_ID" not in os.environ:
        cache_dir = os.path.join(_DEFAULT_CACHE_DIR, hash_object.hexdigest())
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    cache_dir = os.path.join("/cache", "chunks", hash_object.hexdigest())
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def generate_roi(chunks: List[Dict[str, Any]], item_loader: Optional[BaseItemLoader] = None) -> List[Tuple[int, int]]:
    "Generates default region_of_interest for chunks."
    my_roi = []

    if isinstance(item_loader, TokensLoader):
        for idx, chunk in enumerate(chunks):
            dim = chunk["dim"]
            num_blocks = dim // item_loader._block_size
            end = num_blocks
            my_roi.append((0, end))

    else:
        for i, chunk in enumerate(chunks):
            end = chunk["chunk_size"]
            my_roi.append((0, end))

    return my_roi
