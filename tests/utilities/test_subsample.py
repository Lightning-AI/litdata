import numpy as np
import pytest
from litdata.utilities.subsample import (
    shuffle_lists_together,
    subsample_filenames_and_roi,
)


def test_shuffle_lists_together():
    list1 = [i**2 for i in range(10)]
    list2 = [i**4 for i in range(10)]

    shuffled_l1, shuffled_l2 = shuffle_lists_together(list1, list2)

    assert all(shuffled_l1[i] ** 2 == shuffled_l2[i] for i in range(len(list1)))
    assert all(shuffled_l1[i] ** 2 == shuffled_l2[i] for i in range(len(list1)))

    l1 = [32, 54, 21]
    l2 = ["Apple", "Mango", "Orange", "Lichi"]
    with pytest.raises(ValueError, match="Lists must be of the same size"):
        shuffle_lists_together(l1, l2, random_seed_sampler=np.random.RandomState([17]))

    l1 = [32, 54, 21, 57]
    l2 = ["Apple", "Mango", "Orange", "Lichi"]
    shuffled_l1, shuffled_l2 = shuffle_lists_together(l1, l2, random_seed_sampler=np.random.RandomState([17, 90, 21]))

    assert all(l2[l1.index(shuffled_l1[i])] == shuffled_l2[i] for i in range(len(shuffled_l1)))
    assert all(l2[l1.index(shuffled_l1[i])] == shuffled_l2[i] for i in range(len(shuffled_l1)))


def test_subsample_filenames_and_roi():
    my_chunks = [
        {"filename": "1.txt"},
        {"filename": "2.txt"},
        {"filename": "3.txt"},
        {"filename": "4.txt"},
        {"filename": "5.txt"},
    ]

    roi_list = [(0, 50), (0, 25), (0, 75), (0, 35), (0, 5)]

    total_chunk_roi_length = sum(roi[1] - roi[0] for roi in roi_list)

    subsample = 0.42

    target = int(total_chunk_roi_length * subsample)

    _, roi_list, _, left_roi = subsample_filenames_and_roi(my_chunks, roi_list, target)

    assert target == sum(roi[1] - roi[0] for roi in roi_list)
    assert (total_chunk_roi_length - target) == sum(_roi[1] - _roi[0] for _roi in left_roi)


def test_subsample_filenames_and_roi_exact():
    my_chunks = [
        {"filename": "1.txt"},
        {"filename": "2.txt"},
        {"filename": "3.txt"},
        {"filename": "4.txt"},
    ]

    roi_list = [(0, 50), (0, 50), (0, 50), (0, 50)]

    total_chunk_roi_length = sum(roi[1] - roi[0] for roi in roi_list)

    subsample = 0.5

    target = int(total_chunk_roi_length * subsample)

    _, roi_list, _, right_roi = subsample_filenames_and_roi(my_chunks, roi_list, target)

    assert len(roi_list) == 2
    assert len(right_roi) == 2
    assert target == sum(roi[1] - roi[0] for roi in roi_list)
    assert (total_chunk_roi_length - target) == np.sum(right_roi)
