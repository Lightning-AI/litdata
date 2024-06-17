from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def shuffle_lists_together(
    list1: List[Any], list2: List[Any], random_seed_sampler: Optional[np.random.RandomState] = None, seed: int = 42
) -> Tuple[List[Any], List[Any]]:
    """Shuffles list1 and applies the same shuffle order to list2.

    Args:
        list1: The first list to shuffle.
        list2: The second list to shuffle in correspondence with list1.
        random_seed_sampler: Random seed sampler to be used for shuffling
        seed: Seed to use incase random_seed_sampler is not provided

    Returns:
        A tuple containing the shuffled versions of list1 and list2.

    """
    # Sanity check
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same size.")

    if random_seed_sampler is None:
        random_seed_sampler = np.random.RandomState([seed])

    # Shuffle the first list
    shuffled_indices = list(range(len(list1)))

    shuffled_indices = random_seed_sampler.permutation(shuffled_indices).tolist()

    # Apply the same shuffle order to the second list
    shuffled_list1 = [list1[i] for i in shuffled_indices]
    shuffled_list2 = [list2[i] for i in shuffled_indices]

    # Return the shuffled lists
    return shuffled_list1, shuffled_list2


def subsample_filenames_and_roi(
    chunks: List[Dict[str, Any]], roi_list: List[Tuple[int, int]], target: int
) -> Tuple[List[str], List[Tuple[int, int]], List[Dict[str, Any]], List[Tuple[int, int]]]:
    assert len(chunks) == len(roi_list)

    cumsum_sizes = np.cumsum([r[-1] for r in roi_list])

    match = np.argmax(cumsum_sizes > target)
    left_chunk_filenames = [c["filename"] for c in chunks[: match + 1]]
    left_chunk_roi = [r[-1] for r in roi_list[: match + 1]]
    left_chunk_roi[-1] = target if match == 0 else (target - cumsum_sizes[match - 1])

    assert np.sum(left_chunk_roi) == target

    right_chunk_filenames = chunks[match:]
    right_chunk_roi = [r[-1] for r in roi_list[match:]]
    right_chunk_roi[0] -= left_chunk_roi[-1]

    # exact match
    if left_chunk_roi[-1] == 0:
        left_chunk_filenames = left_chunk_filenames[:-1]
        left_chunk_roi = left_chunk_roi[:-1]

    return (
        left_chunk_filenames,
        [(0, r) for r in left_chunk_roi],
        right_chunk_filenames,
        [(0, r) for r in right_chunk_roi],
    )
