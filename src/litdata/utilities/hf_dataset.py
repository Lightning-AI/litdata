"""Contains utility functions for indexing and streaming HF datasets."""

import os

from litdata.constants import _INDEX_FILENAME
from litdata.streaming.writer import index_parquet_dataset
from litdata.utilities.parquet import default_cache_dir


def index_hf_dataset(hf_url: str) -> str:
    if not hf_url.startswith("hf://"):
        raise ValueError(f"Invalid Hugging Face dataset URL: {hf_url}. Expected URL to start with 'hf://'.")

    cache_dir = default_cache_dir(hf_url)

    cache_index_path = os.path.join(cache_dir, _INDEX_FILENAME)
    print("=" * 50)

    if os.path.exists(cache_index_path):
        print(f"Found HF index.json file in {cache_index_path}.")
    else:
        print("Indexing HF dataset...")
        index_parquet_dataset(hf_url, cache_dir, remove_after_indexing=False, num_workers=os.cpu_count())

    print("=" * 50)

    return cache_dir
