from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def shuffle_lists_together(
    list1: List[Any], list2: List[Any], random_seed_sampler: Optional[np.random.RandomState] = None
) -> Tuple[List[Any], List[Any]]:
    """Shuffles list1 and applies the same shuffle order to list2.

    Args:
        list1: The first list to shuffle.
        list2: The second list to shuffle in correspondence with list1.
        random_seed_sampler: Random seed sampler to be used for shuffling

    Returns:
        A tuple containing the shuffled versions of list1 and list2.

    """
    # Sanity check
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same size.")

    if random_seed_sampler is None:
        random_seed_sampler = np.random.RandomState([42])

    # Shuffle the first list
    shuffled_indices = list(range(len(list1)))

    shuffled_indices = random_seed_sampler.permutation(shuffled_indices).tolist()

    # Apply the same shuffle order to the second list
    shuffled_list1 = [list1[i] for i in shuffled_indices]
    shuffled_list2 = [list2[i] for i in shuffled_indices]

    # Return the shuffled lists
    return shuffled_list1, shuffled_list2


def target_sum_problem_with_space_optimization(
    roi_list: List[Tuple[int, int]], target: int
) -> List[Optional[Tuple[int, ...]]]:
    """Solves the target sum problem with space optimization, finding subsets of minimum items that add up to the
    target.

    This is an extension of a standard dynamic programming problem, (target sum and 0-1 knapsack).
    We're not in (0-1) knapsack case, as we can take subset of a chunk.

    Args:
        roi_list: A list of tuples, each containing two integers representing a range.
        target: An integer representing the target sum.

    Returns:
        A list of optional tuples, each containing the indices of elements from roi_list that sum up to the target.
        If no such subset exists, the entry will be None.

    """
    # instantiate 2  1-D table with the size of the target+1 with None.
    # it will keep track of if some combination can actually make that sum

    prev_row: List[Optional[Tuple[int, ...]]] = [None for _ in range(target + 1)]
    curr_row: List[Optional[Tuple[int, ...]]] = [None for _ in range(target + 1)]

    # all the 0th column of table to be empty tuple, marking they can be achieved using some elements of list
    prev_row[0] = ()

    # mark the first element of the list as the only element that can be achieved
    curr_chunk_size = roi_list[0][1] - roi_list[0][0]
    if curr_chunk_size <= target:
        prev_row[curr_chunk_size] = (0,)

    for i in range(1, len(roi_list)):
        for j in range(target + 1):
            if prev_row[j] is not None:
                curr_chunk_size = roi_list[i][1] - roi_list[i][0]
                if curr_chunk_size + j <= target:
                    new_probable_tuple = prev_row[j]
                    assert new_probable_tuple is not None
                    new_probable_tuple += (i,)

                    if (prev_row[j + curr_chunk_size]) is not None:
                        prev_row_after_curr_chunk_size = prev_row[j + curr_chunk_size]
                        assert prev_row_after_curr_chunk_size is not None
                        if len(new_probable_tuple) >= len(
                            prev_row_after_curr_chunk_size
                        ):  # check if the new tuple is better than the previous one (less elements)
                            curr_row[j + curr_chunk_size] = prev_row[j + curr_chunk_size]
                        else:
                            curr_row[j + curr_chunk_size] = new_probable_tuple
                    else:
                        curr_row[j + curr_chunk_size] = new_probable_tuple

                if curr_row[j] is None:
                    curr_row[j] = prev_row[j]

                else:
                    curr_row_col = curr_row[j]
                    prev_row_col = prev_row[j]
                    assert curr_row_col is not None
                    assert prev_row_col is not None

                    curr_row[j] = curr_row[j] if len(curr_row_col) <= len(prev_row_col) else prev_row[j]

        prev_row = curr_row
        curr_row = [None for _ in range(target + 1)]

    return prev_row


def subsample_filenames_and_roi(
    chunks: List[Dict[str, Any]], roi_list: List[Tuple[int, int]], target: int
) -> Tuple[List[str], List[Tuple[int, int]], List[Dict[str, Any]], List[Tuple[int, int]]]:
    """Selects a subset of filenames and ROIs that best match the target sum, with the remaining items returned
    separately.

    Args:
        chunks: A list of dictionaries, each containing metadata including 'filename'.
        roi_list: A list of tuples, each containing two integers representing a range.
        target: An integer representing the target sum.

    Returns:
        A tuple containing four lists:
            - List of filenames corresponding to the selected chunks.
            - List of ROIs corresponding to the selected chunks.
            - List of remaining chunk dictionaries not included in the target sum.
            - List of remaining ROIs not included in the target sum.

    """
    assert len(chunks) == len(roi_list)

    complete_roi_lists = target_sum_problem_with_space_optimization(roi_list, target)

    # iterate from end, and for the first non-None value,
    # sum up the complete roi chunks, and then try to accomodate for left
    i = len(complete_roi_lists) - 1
    while i >= 0 and complete_roi_lists[i] is None:
        i -= 1

    assert i >= 0

    complete_roi_list = complete_roi_lists[i]
    assert complete_roi_list is not None

    subsampled_chunk_filenames = [chunks[i]["filename"] for i in complete_roi_list]
    subsampled_roi = [roi_list[i] for i in complete_roi_list]

    left_chunks = [chunks[i] for i in range(len(chunks)) if i not in complete_roi_list]
    left_roi = [roi_list[i] for i in range(len(chunks)) if i not in complete_roi_list]

    sum_of_complete_overlap_roi = sum([i[1] - i[0] for i in subsampled_roi])
    left_item_count = target - sum_of_complete_overlap_roi

    while left_item_count > 0:
        top_left_chunk = left_chunks.pop(0)
        top_left_roi = left_roi.pop(0)
        top_left_roi_item_count = top_left_roi[1] - top_left_roi[0]

        if top_left_roi_item_count <= left_item_count:
            subsampled_chunk_filenames.append(top_left_chunk["filename"])
            subsampled_roi.append(top_left_roi)
            left_item_count -= top_left_roi[1] - top_left_roi[0]

        else:
            subsampled_chunk_filenames.append(top_left_chunk["filename"])
            left_chunks.append(
                top_left_chunk
            )  # it will also be available for other splits, as not all roi is exhausted
            subsampled_roi.append((top_left_roi[0], top_left_roi[0] + left_item_count))
            left_roi.append((top_left_roi[0] + left_item_count, top_left_roi[1]))
            left_item_count = 0

    return subsampled_chunk_filenames, subsampled_roi, left_chunks, left_roi
