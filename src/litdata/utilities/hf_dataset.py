"""Contains utility functions for indexing and streaming HF datasets."""

import os

from litdata.constants import _INDEX_FILENAME
from litdata.streaming.writer import index_parquet_dataset
from litdata.utilities.parquet import default_cache_dir


def index_hf_dataset(hf_url: str) -> str:
    """Index a Hugging Face dataset and return the cache directory path.

    Args:
        hf_url (str): The URL of the Hugging Face dataset.

    Returns:
        str: The path to the cache directory containing the index.
    """
    if not hf_url.startswith("hf://"):
        raise ValueError(
            f"Invalid Hugging Face dataset URL: {hf_url}. "
            "The URL should start with 'hf://'. Please check the URL and try again."
        )

    cache_dir = default_cache_dir(hf_url)
    cache_index_path = os.path.join(cache_dir, _INDEX_FILENAME)

    if os.path.exists(cache_index_path):
        print(f"Found existing HF {_INDEX_FILENAME} file for {hf_url} at {cache_index_path}.")
    else:
        print(f"Indexing HF dataset from {hf_url} into {cache_index_path}.")
        index_parquet_dataset(hf_url, cache_dir, num_workers=os.cpu_count())

    return cache_dir
