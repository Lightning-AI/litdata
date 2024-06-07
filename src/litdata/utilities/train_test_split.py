from copy import deepcopy
from typing import List

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
    if any([not isinstance(split, float) for split in splits]) or not (
        all([_f > 0 and _f <= 1 for _f in splits]) and sum(splits) <= 1
    ):
        raise ValueError("Split should be float with each value in [0,1] and max sum can be 1.")

    my_datasets = [deepcopy(streaming_dataset) for _ in splits]

    frac_start = 0
    frac_end = 0
    for i in range(len(splits)):
        frac_end += splits[i]
        my_datasets[i]._modify_subsample_interval(frac_start, frac_end)
        frac_start += splits[i]

    return my_datasets
