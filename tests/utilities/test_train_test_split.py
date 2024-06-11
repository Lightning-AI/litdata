import pytest
from litdata import train_test_split
from litdata.constants import _ZSTD_AVAILABLE
from litdata.streaming.cache import Cache
from litdata.streaming.dataset import StreamingDataset
from litdata.utilities.train_test_split import (
    is_dataset_subsample,
    sample_k_times,
    split_modify_chunk_and_roi,
)


def test_is_dataset_subsample():
    chunks = [
        {"chunk_size": 50},
        {"chunk_size": 50},
        {"chunk_size": 50},
        {"chunk_size": 32},
    ]

    # when roi is complete overlap
    region_of_interest = [(0, 50), (50, 100), (100, 150), (150, 182)]
    assert not is_dataset_subsample(chunks, region_of_interest)

    # when roi is a subsample
    region_of_interest = [(0, 50), (50, 100), (100, 150), (150, 162)]
    assert is_dataset_subsample(chunks, region_of_interest)


def test_sample_k_times():
    my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    my_n_list = [3, 5, 2]

    my_samples, my_remaining_list = sample_k_times(my_list, my_n_list)

    # random samples should be of specified length
    assert all(len(my_samples[i]) == n for i, n in enumerate(my_n_list))

    # remaining list length should also be correct
    assert len(my_remaining_list) == len(my_list) - sum(my_n_list)


def test_split_modify_chunk_and_roi():
    my_chunk_list = [
        {"chunk_size": 50, "file": 1},
        {"chunk_size": 50, "file": 2},
        {"chunk_size": 50, "file": 3},
        {"chunk_size": 50, "file": 4},
    ]
    my_splits = [0.8, 0.2]

    my_new_chunk_list, my_new_roi_list = split_modify_chunk_and_roi(my_chunk_list, my_splits)

    assert len(my_new_chunk_list[0]) == 4
    assert len(my_new_chunk_list[1]) == 1

    assert my_new_roi_list == [[(0, 50), (50, 100), (100, 150), (150, 160)], [(10, 50)]]

    # -----------------------------------------------

    my_splits = [0.35, 0.15, 0.15]  # 70, 30, 30 items

    my_new_chunk_list, my_new_roi_list = split_modify_chunk_and_roi(my_chunk_list, my_splits)

    assert len(my_new_chunk_list[0]) == 2  # 50, 20
    assert len(my_new_chunk_list[1]) == 1  # 30
    assert len(my_new_chunk_list[2]) == 1  # 30

    assert my_new_roi_list == [[(0, 50), (50, 70)], [(0, 30)], [(0, 30)]]  # each new chunk will be used

    # -----------------------------------------------

    my_splits = [0.45, 0.15, 0.15, 0.25]  # 90, 30, 30, 50 items

    my_new_chunk_list, my_new_roi_list = split_modify_chunk_and_roi(my_chunk_list, my_splits)

    assert len(my_new_chunk_list[0]) == 2  # 50, 40
    assert len(my_new_chunk_list[1]) == 1  # 30
    assert len(my_new_chunk_list[2]) == 2  # 30
    assert len(my_new_chunk_list[3]) == 1  # 50

    assert my_new_roi_list == [[(0, 50), (50, 90)], [(0, 30)], [(40, 50), (80, 100)], [(0, 50)]]


@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        pytest.param("zstd", marks=pytest.mark.skipif(condition=not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")),
    ],
)
def test_train_test_split(tmpdir, compression):
    cache = Cache(str(tmpdir), chunk_size=10, compression=compression)
    for i in range(100):
        cache[i] = i
    cache.done()
    cache.merge()

    my_streaming_dataset = StreamingDataset(input_dir=str(tmpdir))
    train_dataset, test_dataset = train_test_split(my_streaming_dataset, splits=[0.75, 0.25])

    assert len(train_dataset) == 75
    assert len(test_dataset) == 25
