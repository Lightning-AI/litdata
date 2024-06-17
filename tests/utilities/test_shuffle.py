from dataclasses import dataclass

from litdata.streaming.item_loader import Interval
from litdata.utilities.env import _DistributedEnv
from litdata.utilities.shuffle import (
    _associate_chunks_and_internals_to_ranks,
    _find_chunks_per_ranks_on_which_to_skip_deletion,
    _intra_node_chunk_shuffle,
)


def test_intra_node_chunk_shuffle():
    chunks_per_ranks = [[0, 1], [2, 3], [4, 5], [6, 7]]

    shuffled_indexes = _intra_node_chunk_shuffle(_DistributedEnv(4, 1, 1), chunks_per_ranks, 42, 2)
    assert shuffled_indexes == [5, 2, 0, 7, 6, 1, 3, 4]

    shuffled_indexes = _intra_node_chunk_shuffle(_DistributedEnv(4, 1, 2), chunks_per_ranks, 42, 2)
    assert shuffled_indexes == [3, 2, 1, 0, 7, 6, 5, 4]

    chunks_per_ranks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    shuffled_indexes = _intra_node_chunk_shuffle(_DistributedEnv(8, 7, 2), chunks_per_ranks, 42, 2)
    assert shuffled_indexes == [5, 2, 0, 7, 6, 1, 3, 4, 13, 10, 8, 15, 14, 9, 11, 12]


def test_associate_chunks_and_internals_to_ranks():

    indexes = [0, 1, 2, 3, 4, 5, 6, 7]
    chunk_intervals = [
        Interval(0, 0, 50, 50),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 50, 50),
    ]


    chunks_per_ranks, intervals_per_ranks = _associate_chunks_and_internals_to_ranks(
        _DistributedEnv(4, 1, 2),
        indexes,
        chunk_intervals,
        drop_last=True,
    )

    assert chunks_per_ranks == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert intervals_per_ranks == [
        [[0, 0, 50, 50], [0, 0, 50, 50]],
        [[0, 0, 50, 50], [0, 0, 50, 50]],
        [[0, 0, 50, 50], [0, 0, 50, 50]],
        [[0, 0, 50, 50], [0, 0, 50, 50]],
    ]

    chunk_intervals = [
        Interval(0, 0, 50, 50),
        Interval(0, 0, 150, 150),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 12, 12),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 27, 27),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 33, 33),
    ]


    chunks_per_ranks, intervals_per_ranks = _associate_chunks_and_internals_to_ranks(
        _DistributedEnv(4, 1, 2),
        indexes,
        chunk_intervals,
        drop_last=True,
    )

    assert chunks_per_ranks == [[0, 1], [1, 2], [2, 3, 4, 5], [5, 6, 7]]
    assert sum([interval[2] - interval[1] for interval in intervals_per_ranks[0]]) == 105
    assert sum([interval[2] - interval[1] for interval in intervals_per_ranks[1]]) == 105
    assert sum([interval[2] - interval[1] for interval in intervals_per_ranks[2]]) == 105
    assert sum([interval[2] - interval[1] for interval in intervals_per_ranks[3]]) == 105

    assert intervals_per_ranks == [
        [[0, 0, 50, 50], [0, 0, 55, 150]],
        [[0, 55, 150, 150], [0, 0, 10, 50]],
        [[0, 10, 50, 50], [0, 0, 12, 12], [0, 0, 50, 50], [0, 0, 3, 27]],
        [[0, 3, 27, 27], [0, 0, 50, 50], [0, 0, 31, 33]],
    ]

    chunk_intervals = [
        Interval(0, 0, 5, 5),
        Interval(0, 0, 150, 150),
        Interval(0, 0, 7, 7),
        Interval(0, 0, 12, 12),
        Interval(0, 0, 4, 4),
        Interval(0, 0, 27, 27),
        Interval(0, 0, 50, 50),
        Interval(0, 0, 1, 1),
    ]


    chunks_per_ranks, intervals_per_ranks = _associate_chunks_and_internals_to_ranks(
        _DistributedEnv(4, 1, 2),
        indexes,
        chunk_intervals,
        drop_last=True,
    )

    assert chunks_per_ranks == [[0, 1], [1], [1, 2, 3, 4, 5], [5, 6, 7]]
    assert sum([interval[2] - interval[1] for interval in intervals_per_ranks[0]]) == 64
    assert sum([interval[2] - interval[1] for interval in intervals_per_ranks[1]]) == 64
    assert sum([interval[2] - interval[1] for interval in intervals_per_ranks[2]]) == 64
    assert sum([interval[2] - interval[1] for interval in intervals_per_ranks[3]]) == 64
    assert intervals_per_ranks == [
        [[0, 0, 5, 5], [0, 0, 59, 150]],
        [[0, 59, 123, 150]],
        [[0, 123, 150, 150], [0, 0, 7, 7], [0, 0, 12, 12], [0, 0, 4, 4], [0, 0, 14, 27]],
        [[0, 14, 27, 27], [0, 0, 50, 50], [0, 0, 1, 1]],
    ]

    disable_deletion_ranks = _find_chunks_per_ranks_on_which_to_skip_deletion(1, chunks_per_ranks, intervals_per_ranks)
    assert disable_deletion_ranks == {1: [1], 2: [1], 3: [5]}
