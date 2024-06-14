import random
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from litdata import StreamingDataset
from litdata.utilities.dataset_utilities import _generate_subsample_intervals


def train_test_split(streaming_dataset: StreamingDataset, splits: List[float]) -> List[StreamingDataset]:
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

    # streaming dataset should not be a subsample itself.
    # We don't support this feature as of yet.
    # But, if someone implementing in future:
    #   HINT: think of using priority queue to get the largest RoI chunks which will be allocated first.
    if is_dataset_subsample(streaming_dataset.chunks, streaming_dataset.region_of_interest):
        raise Exception("Splitting subsample dataset is not supported.")

    original_chunk = deepcopy(streaming_dataset.chunks)

    my_datasets = [deepcopy(streaming_dataset) for _ in splits]

    my_new_chunk_list, my_new_roi_list = split_modify_chunk_and_roi(original_chunk, splits)

    for i, curr_new_dataset in enumerate(my_datasets):
        curr_new_dataset.chunks = my_new_chunk_list[i]
        curr_new_dataset.region_of_interest = my_new_roi_list[i]

    return my_datasets


def is_dataset_subsample(chunks: List[Dict[str, Any]], region_of_interest: List[Tuple[int, int]]) -> bool:
    """Checks if a StreamingDataset is a subsample of another dataset.

    This function determines if a StreamingDataset is a subsample based on the chunk sizes and region of interest (ROI) information. 
    A subsample dataset would have chunks with sizes smaller than the corresponding ROI ranges.

    Args:
        chunks (List[Dict[str, Any]]): A list of dictionaries representing the chunks in the StreamingDataset.
        region_of_interest (List[Tuple[int, int]]): A list of tuples representing the ROI for each chunk.

    Returns:
        bool: True if the StreamingDataset is a subsample, False otherwise.
    """
    for i, roi in enumerate(region_of_interest):
        start, end = roi
        if (end - start) != chunks[i]["chunk_size"]:
            return True

    return False


def sample_k_times(
    lst: List[Dict[str, Any]], n_list: List[int]
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Samples a list k times with a specified number of elements each time.

    This function samples a list multiple times, where each sample contains a specified number of elements. 
    It ensures that elements are not sampled more than once and removes them from the original list after being selected.

    Args:
        lst (List[Dict[str, Any]]): The list to be sampled from.
        n_list (List[int]): A list of integers representing the number of elements to sample each time.

    Returns:
        Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]: A tuple containing two elements. 
            - The first element is a list of lists, where each inner list represents a sample.
            - The second element is the remaining list after removing the sampled elements.

    Raises:
        ValueError: If the list does not have enough elements to fulfill all the sampling requests 
                    specified in the 'n_list' argument.
    """

    all_samples = []

    for n in n_list:
        # Ensure there are enough elements in the list to sample k times with n elements each
        if len(lst) < n:
            raise ValueError("List doesn't have enough elements to sample.")

        # Select n unique random elements
        sample = random.sample(lst, n)
        all_samples.append(sample)

        # Remove the selected elements from the list
        lst = [item for item in lst if item not in sample]

    return all_samples, lst


def split_modify_chunk_and_roi(
    chunk_list: List[Dict[str, Any]], splits: List[float]
) -> Tuple[List[List[Dict[str, Any]]], List[List[Tuple[int, int]]]]:
    """Splits chunks and ROIs based on specified proportions, considering chunk sizes.

    This function splits a list of chunks (dictionaries representing data units) and their corresponding ROIs (regions of interest) 
    into multiple subsets based on the provided proportions in the 'splits' argument. It prioritizes assigning whole chunks to 
    each split whenever possible, while still fulfilling the desired data allocation ratios.

    **Process:**

    1. **Analyze Chunk Information:**
       - Gets the size of each chunk from the 'chunk_size' key in the dictionaries.
       - Calculates the total length of the data considering all chunks.

    2. **Calculate Split Details:**
       - Converts the proportions in 'splits' to actual item counts for each split based on the total data length.
       - Determines the number of whole chunks that can be allocated to each split without exceeding the desired data size.
       - Calculates the remaining item count required for each split after allocating whole chunks.

    3. **Sample Chunks:**
       - Uses the `sample_k_times` function to select the required number of whole chunks for each split.
       - Generates the corresponding ROIs for the sampled chunks using `_generate_subsample_intervals` (assumed to be an internal function).

    4. **Handle Remaining Items:**
       - Iterates through each split.
       - While there are still items remaining for a split and there are unallocated chunks:
           - Selects the first remaining chunk and its starting index within the overall data.
           - Calculates how many items this chunk can contribute to the split based on its remaining size.
           - Updates the ROI for the split to include the newly assigned items.
           - If the chunk contributes all its remaining items, it's removed from the pool of unallocated chunks.
           - Otherwise, the chunk is added back to the pool with an updated starting index reflecting the used items.

    5. **Return Results:**
       - Returns a tuple containing two lists:
           - The first list contains sub-lists of chunks, where each sub-list represents the chunks assigned to a particular split.
           - The second list contains sub-lists of ROIs, where each sub-list corresponds to the ROIs for the chunks in the respective split.

    Args:
        chunk_list (List[Dict[str, Any]]): A list of dictionaries representing the chunks in the StreamingDataset.
        splits (List[float]): A list of floats representing the proportion of data to be allocated to each split.

    Returns:
        Tuple[List[List[Dict[str, Any]]], List[List[Tuple[int, int]]]]: A tuple containing the split chunks and ROIs.
    """

    chunk_size = chunk_list[0]["chunk_size"]
    total_chunk_length = len(chunk_list) * chunk_size
    each_split_item_count = [int(total_chunk_length * split) for split in splits]
    complete_chunk_overlap_list = [int(total_chunk_length * split) // chunk_size for split in splits]
    item_count_left_list = [
        each_split_item_count[i] - (complete_chunk_overlap_list[i] * chunk_size) for i in range(len(splits))
    ]

    new_chunk_list, remaining_chunk_list = sample_k_times(chunk_list, complete_chunk_overlap_list)
    new_roi_list = [_generate_subsample_intervals(new_curr_chunk, 0) for new_curr_chunk in new_chunk_list]

    remaining_chunk_start_idx = [0 for remaining_chunk in remaining_chunk_list]

    for i in range(len(splits)):
        while item_count_left_list[i] > 0:
            if len(remaining_chunk_list) == 0:
                # might be possible if chunks are of uneven size and we are 1-2 item short of exact split count
                break
            first_chunk_in_remaining_list = remaining_chunk_list.pop(0)
            first_chunk_start_idx = remaining_chunk_start_idx.pop(0)

            top_can_fulfill = first_chunk_in_remaining_list["chunk_size"] - first_chunk_start_idx

            last_roi_idx = len(new_chunk_list[i]) * chunk_size

            new_chunk_list[i].append(first_chunk_in_remaining_list)

            if top_can_fulfill == item_count_left_list[i]:
                new_roi_list[i].append(
                    (
                        last_roi_idx + first_chunk_start_idx,
                        last_roi_idx + first_chunk_start_idx + item_count_left_list[i],
                    )
                )

                item_count_left_list[i] = 0

            elif top_can_fulfill > item_count_left_list[i]:
                new_roi_list[i].append(
                    (
                        last_roi_idx + first_chunk_start_idx,
                        last_roi_idx + first_chunk_start_idx + item_count_left_list[i],
                    )
                )

                remaining_chunk_list.append(first_chunk_in_remaining_list)
                remaining_chunk_start_idx.append(first_chunk_start_idx + item_count_left_list[i])
                item_count_left_list[i] = 0

            else:
                new_roi_list[i].append((last_roi_idx + first_chunk_start_idx, last_roi_idx + chunk_size))
                item_count_left_list[i] -= top_can_fulfill

    return new_chunk_list, new_roi_list
