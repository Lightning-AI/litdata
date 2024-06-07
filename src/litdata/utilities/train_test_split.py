from copy import deepcopy
from typing import List

import numpy as np

from litdata import StreamingDataset


def train_test_split(streaming_dataset: StreamingDataset, splits: List[float]) -> List[StreamingDataset]:
    """Split a StreamingDataset into multiple subsets for purposes such as training, testing, and validation.

    Arguments:
        streaming_dataset (StreamingDataset): An instance of StreamingDataset that needs to be split.
        splits (List[float]): List of floats representing the proportion of data to be allocated to each split
                            (e.g., [0.8, 0.1, 0.1] for train, test, and validation).

    Returns:
        List[StreamingDataset]: A list of StreamingDataset instances, each corresponding to the proportions specified
                                in the splits argument.

    """
    if any(not isinstance(split, float) for split in splits):
        raise ValueError("Each split should be a float.")

    if not all(0 < _f <= 1 for _f in splits):
        raise ValueError("Each Split should be a float with each value in [0,1].")

    if sum(splits) > 1:
        raise ValueError("Splits' sum must be less than 1.")

    my_datasets = [deepcopy(streaming_dataset) for _ in splits]

    fraction = [0] + np.cumsum(splits).tolist()
    for i in range(len(splits)):
        my_datasets[i]._modify_subsample_interval(fraction[i], fraction[i + 1])

    return my_datasets
