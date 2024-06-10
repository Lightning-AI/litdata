import random
import numpy as np
from copy import deepcopy
from typing import List, Tuple
from litdata import StreamingDataset
from litdata.streaming.dataset import _generate_subsample_intervals


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


def is_dataset_subsample(chunks: List[any], region_of_interest: List[Tuple[int,int]])->bool:
    for i, roi in enumerate(region_of_interest):
        start, end = roi
        if (end-start) != chunks[i]["chunk_size"]:
            return True
    
    return False


def sample_k_times(lst: List[any], n_list: List[int]):
    
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


def split_modify_chunk_and_roi(chunk_list: List[any], splits: List[float])->Tuple[List[any], List[Tuple[int,int]]]:
    chunk_size = chunk_list[0]["chunk_size"]
    total_chunk_length = len(chunk_list)*chunk_size
    each_split_item_count = [int(total_chunk_length*split) for split in splits]
    complete_chunk_overlap_list = [int(total_chunk_length*split)//chunk_size for split in splits]
    item_count_left_list = [each_split_item_count[i]-(complete_chunk_overlap_list[i]*chunk_size) for i in range(len(splits))]



    new_chunk_list, remaining_chunk_list = sample_k_times(chunk_list,complete_chunk_overlap_list)
    new_roi_list = [_generate_subsample_intervals(new_curr_chunk,0) for new_curr_chunk in new_chunk_list]

    remaining_chunk_start_idx = [0 for remaining_chunk in remaining_chunk_list]


    for i in range(len(splits)):
        while item_count_left_list[i]>0:
            if len(remaining_chunk_list)==0:
                # might be possible if chunks are of uneven size and we are 1-2 item short of exact split count
                return
            first_chunk_in_remaining_list = remaining_chunk_list.pop(0)
            first_chunk_start_idx = remaining_chunk_start_idx.pop(0)

            top_can_fulfill = first_chunk_in_remaining_list["chunk_size"] - first_chunk_start_idx
            

            last_roi_idx = len(new_chunk_list[i]) * chunk_size

            new_chunk_list[i].append(first_chunk_in_remaining_list)

            if top_can_fulfill == item_count_left_list[i]:
                new_roi_list[i].append((last_roi_idx+first_chunk_start_idx, last_roi_idx+first_chunk_start_idx+item_count_left_list[i]))

                item_count_left_list[i]=0

            elif top_can_fulfill > item_count_left_list[i]:
                new_roi_list[i].append((last_roi_idx+first_chunk_start_idx, last_roi_idx+first_chunk_start_idx+item_count_left_list[i]))

                remaining_chunk_list.append(first_chunk_in_remaining_list)
                remaining_chunk_start_idx.append(first_chunk_start_idx+item_count_left_list[i])
                item_count_left_list[i]=0

            
            else:
                new_roi_list[i].append((last_roi_idx+first_chunk_start_idx, last_roi_idx+chunk_size))
                item_count_left_list[i]-=top_can_fulfill

    return new_chunk_list, new_roi_list
