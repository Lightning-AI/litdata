import pytest
from litdata.utilities.subsample import (
    shuffle_lists_together,
    target_sum_problem_with_space_optimization,
    my_subsampled_filenames_and_roi
)


def test_shuffle_lists_together():
    list1 = [i**2 for i in range(10)]
    list2 = [i**4 for i in range(10)]

    shuffled_l1, shuffled_l2 = shuffle_lists_together(list1, list2)

    assert all([shuffled_l1[i]**2 == shuffled_l2[i] for i in range(len(list1))])

    with pytest.raises(ValueError, match="Lists must be of the same size"):
        l1 = [32, 54, 21]
        l2 = ["Apple", "Mango", "Orange", "Lichi"]
        shuffle_lists_together(l1,l2, seed=47)
    
    l1 = [32, 54, 21, 57]
    l2 = ["Apple", "Mango", "Orange", "Lichi"]
    shuffled_l1, shuffled_l2 = shuffle_lists_together(l1,l2, seed=47)

    assert all([l2[l1.index(shuffled_l1[i])] == shuffled_l2[i]
        for i in range(len(shuffled_l1))
    ])


def test_target_sum_problem_with_space_optimization():
    my_roi_list = [(0,50), (0,50), (0,50) ,(0,50)]
    target = 100

    final_table_row = target_sum_problem_with_space_optimization(my_roi_list, target)

    assert final_table_row[100] == (0,1)

    # -----------------------------------------------------

    my_roi_list = [(0,55), (0,5), (0,40) ,(0,70), (0,30)]
    target = 100

    final_table_row = target_sum_problem_with_space_optimization(my_roi_list, target)

    assert final_table_row[100] == (3,4)


def test_my_subsampled_filenames_and_roi():
    my_chunks = [
        {"filename":"1.txt"},
        {"filename":"2.txt"},
        {"filename":"3.txt"},
        {"filename":"4.txt"},
        {"filename":"5.txt"},
    ]

    my_roi_list = [(0,50), (0,25), (0,75), (0,35), (0,5)]

    total_chunk_roi_length = sum([roi[1]-roi[0] for roi in my_roi_list])

    subsample = 0.42

    target = int(total_chunk_roi_length * subsample)

    my_subsampled_chunk_filenames, my_roi_list, left_chunks, left_roi = my_subsampled_filenames_and_roi(my_chunks, my_roi_list, target)


    assert target == sum([roi[1]-roi[0] for roi in my_roi_list])
    assert total_chunk_roi_length - target == sum([roi[1]-roi[0] for roi in left_roi])
