import json
import os
import random
from copy import deepcopy
from typing import Any, Dict, List

from litdata import StreamingDataset
from litdata.constants import _INDEX_FILENAME
from litdata.utilities.subsample import subsample_filenames_and_roi, shuffle_lists_together


def train_test_split(
    streaming_dataset: StreamingDataset, splits: List[float], seed: int = 42
) -> List[StreamingDataset]:
    """Splits a StreamingDataset into multiple subsets for training, testing, and validation.

    This function splits a StreamingDataset into multiple non-overlapping subsets based on the provided proportions.
    These subsets can be used for training, testing, and validation purposes.

    Args:
        streaming_dataset (StreamingDataset): An instance of StreamingDataset that needs to be split.
        splits (List[float]): A list of floats representing the proportion of data to be allocated to each split
                             (e.g., [0.8, 0.1, 0.1] for 80% training, 10% testing, and 10% validation).

    Returns:
        List[StreamingDataset]: A list of StreamingDataset instances, where each element represents a split of the
                                original dataset according to the proportions specified in the 'splits' argument.

    Raises:
        ValueError: If any element in the 'splits' list is not a float between 0 (inclusive) and 1 (exclusive).
        ValueError: If the sum of the values in the 'splits' list is greater than 1.
        Exception: If the provided StreamingDataset is already a subsample (not currently supported).

    """
    if any(not isinstance(split, float) for split in splits):
        raise ValueError("Each split should be a float.")

    if not all(0 < _f <= 1 for _f in splits):
        raise ValueError("Each Split should be a float with each value in [0,1].")

    if sum(splits) > 1:
        raise ValueError("Splits' sum must be less than 1.")

    # we need subsampled chunk filenames, original chunk file, and subsampled_roi

    dummy_streaming_dataset = deepcopy(streaming_dataset)
    dummy_subsampled_chunk_filename = dummy_streaming_dataset.subsampled_files
    dummy_subsampled_roi = dummy_streaming_dataset.region_of_interest
    subsampled_chunks: List[Dict[str, Any]] = []

    input_dir = dummy_streaming_dataset.input_dir
    assert input_dir.path

    if os.path.exists(os.path.join(input_dir.path, _INDEX_FILENAME)):
        # load chunks from `index.json` file
        with open(os.path.join(input_dir.path, _INDEX_FILENAME)) as f:
            data = json.load(f)
            original_chunks = data["chunks"]
            subsampled_chunks = [
                _org_chunk
                for _org_chunk in original_chunks
                if _org_chunk["filename"] in dummy_subsampled_chunk_filename
            ]
    else:
        raise ValueError("Couldn't load original chunk file.")

    new_datasets = [deepcopy(streaming_dataset) for _ in splits]

    dataset_length = sum([my_roi[1] - my_roi[0] for my_roi in dummy_subsampled_roi])

    for i, split in enumerate(splits):
        item_count = int(dataset_length * split)

        random.seed(seed)
        subsampled_chunks, dummy_subsampled_roi = shuffle_lists_together(subsampled_chunks, dummy_subsampled_roi, seed)
        curr_chunk_filename, curr_chunk_roi, left_chunks, left_roi = subsample_filenames_and_roi(
            subsampled_chunks, dummy_subsampled_roi, item_count
        )

        new_datasets[i].subsampled_files = curr_chunk_filename
        new_datasets[i].region_of_interest = curr_chunk_roi
        subsampled_chunks = left_chunks
        dummy_subsampled_roi = left_roi
        
    
    # undo all the properties associated with original dataset
    default_properties: Dict[str, Any] = {
        "cache": None,
        "worker_env": None,
        "worker_chunks": [],
        "worker_intervals": [],
        "current_indexes": [],
        "chunk_index": 0,
        "num_chunks": None,
        "global_index": 0,
        "index": 0,
        "has_triggered_download": False,
        "min_items_per_replica": None,
        "current_epoch": 1,
        "random_state": None,
        "shuffler": None,
        "_state_dict": None,
        "num_workers": None,
        "batch_size": None,
    }

    for new_dataset in new_datasets:
        for prop, value in default_properties.items():
            setattr(new_dataset, prop, value)

    return new_datasets
