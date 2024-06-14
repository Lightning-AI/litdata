import os
import json
import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from litdata.streaming.resolver import Dir
from litdata.constants import _INDEX_FILENAME, _DEFAULT_CACHE_DIR
from litdata.streaming.downloader import get_downloader_cls
from litdata.streaming.item_loader import BaseItemLoader, TokensLoader


def subsample_streaming_dataset(input_dir: Dir, item_loader: Optional[BaseItemLoader] = None, subsample: float = 1.0)->Tuple[List[str], List[Tuple[int,int]]]:
    """
    Subsample streaming dataset.
    
    But before doing that, we will do some preprocessing:
    - Make sure input_dir contains cache path and remote url.
    - Check if `index.json` file exists in cache path.
    - If not, download from remote url. If remote url doesn't contain `index.json` file, raise error.
    - Once download, load chunks from `index.json` file.
    - Once chunks are ready, subsample them on the basis of `item loader` and return (chunks, region_of_interest).
    """
    my_subsampled_files: List[str] = []
    my_roi: List[Tuple[int,int]] = []

    # Make sure input_dir contains cache path and remote url
    if _should_replace_path(input_dir.path):
        cache_path = _try_create_cache_dir(
            input_dir=input_dir.path if input_dir.path else input_dir.url
        )
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
    if isinstance(item_loader, TokensLoader):
        my_subsampled_files, my_roi = token_loader_sample_chunk_and_generate_interval(
            original_chunks, subsample, item_loader._block_size
        )
    else:
        my_subsampled_files, my_roi = sample_chunk_and_generate_interval(original_chunks, subsample)

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


def _generate_subsample_intervals_for_token_loader(
    chunks: List[Dict[str, Any]], block_size: int, last_left_subsample_count: int
) -> List[Tuple[int, int]]:
    intervals = []
    begin = 0
    end = 0
    for idx, chunk in enumerate(chunks):
        dim = chunk["dim"]
        num_blocks = dim // block_size
        end += num_blocks
        start_idx, end_idx = begin, end
        if idx == len(chunks) - 1 and last_left_subsample_count > 0:
            end_idx = last_left_subsample_count
        intervals.append((start_idx, end_idx))
        begin += num_blocks
    return intervals


def sample_chunk_and_generate_interval(
    chunks: List[Dict[str, Any]], subsample: float
) -> Tuple[List[str], List[Tuple[int, int]]]:
    total_chunk_length = len(chunks) * chunks[0]["chunk_size"]
    new_subsample_length = int(total_chunk_length * subsample)
    complete_subsample_chunk = new_subsample_length // chunks[0]["chunk_size"]
    last_left_subsample_count = new_subsample_length - (complete_subsample_chunk * chunks[0]["chunk_size"])

    chunk_count = complete_subsample_chunk
    if last_left_subsample_count > 0:
        chunk_count += 1
        # sampled chunks
        chunks = random.sample(chunks, chunk_count)

    region_of_interest = _generate_subsample_intervals(chunks, last_left_subsample_count)

    my_subsampled_files = [chunk["filename"] for chunk in chunks]
    return my_subsampled_files, region_of_interest


def token_loader_sample_chunk_and_generate_interval(
    chunks: List[Dict[str, Any]], subsample: float, block_size: int
) -> Tuple[List[str], List[Tuple[int, int]]]:
    total_chunk_length = len(chunks) * chunks[0]["dim"]
    new_subsample_length = int(total_chunk_length * subsample)
    complete_subsample_chunk = new_subsample_length // chunks[0]["dim"]
    last_left_subsample_count = new_subsample_length - (complete_subsample_chunk * chunks[0]["dim"])

    chunk_count = complete_subsample_chunk
    if last_left_subsample_count > 0:
        chunk_count += 1
        # sampled chunks
        chunks = random.sample(chunks, chunk_count)

    region_of_interest = _generate_subsample_intervals_for_token_loader(chunks, block_size, last_left_subsample_count)
    my_subsampled_files = [chunk["filename"] for chunk in chunks]

    return my_subsampled_files, region_of_interest


def _generate_subsample_intervals(
    my_chunk_arr: List[Dict[str, Any]], last_left_subsample_count: int
) -> List[Tuple[int, int]]:
    """Generates a list of intervals that the dataset is allowed to read, based on the sizes of chunks."""
    intervals = []
    begin = 0
    end = 0
    for i, chunk in enumerate(my_chunk_arr):
        if i == len(my_chunk_arr) - 1 and last_left_subsample_count > 0:
            end += last_left_subsample_count
        else:
            end += chunk["chunk_size"]

        intervals.append((begin, end))
        begin += chunk["chunk_size"]

    return intervals
