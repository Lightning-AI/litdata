from litdata.streaming.item_loader import Interval
from litdata.utilities.env import _DistributedEnv
from litdata.utilities.shuffle import (
    _associate_chunks_and_internals_to_workers,
    _find_chunks_per_workers_on_which_to_skip_deletion,
    _get_shared_chunks,
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


def test_associate_chunks_and_internals_to_workers():
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

    workers_chunks, workers_intervals = _associate_chunks_and_internals_to_workers(
        _DistributedEnv(4, 1, 2),
        indexes,
        chunk_intervals,
        drop_last=True,
    )

    assert workers_chunks == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert workers_intervals == [
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

    workers_chunks, workers_intervals = _associate_chunks_and_internals_to_workers(
        _DistributedEnv(4, 1, 2),
        indexes,
        chunk_intervals,
        drop_last=True,
    )

    assert workers_chunks == [[0, 1], [1, 2], [2, 3, 4, 5], [5, 6, 7]]
    assert sum([interval[2] - interval[1] for interval in workers_intervals[0]]) == 105
    assert sum([interval[2] - interval[1] for interval in workers_intervals[1]]) == 105
    assert sum([interval[2] - interval[1] for interval in workers_intervals[2]]) == 105
    assert sum([interval[2] - interval[1] for interval in workers_intervals[3]]) == 105

    assert workers_intervals == [
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

    workers_chunks, workers_intervals = _associate_chunks_and_internals_to_workers(
        _DistributedEnv(4, 1, 2),
        indexes,
        chunk_intervals,
        drop_last=True,
    )

    assert workers_chunks == [[0, 1], [1], [1, 2, 3, 4, 5], [5, 6, 7]]
    assert sum([interval[2] - interval[1] for interval in workers_intervals[0]]) == 64
    assert sum([interval[2] - interval[1] for interval in workers_intervals[1]]) == 64
    assert sum([interval[2] - interval[1] for interval in workers_intervals[2]]) == 64
    assert sum([interval[2] - interval[1] for interval in workers_intervals[3]]) == 64
    assert workers_intervals == [
        [[0, 0, 5, 5], [0, 0, 59, 150]],
        [[0, 59, 123, 150]],
        [[0, 123, 150, 150], [0, 0, 7, 7], [0, 0, 12, 12], [0, 0, 4, 4], [0, 0, 14, 27]],
        [[0, 14, 27, 27], [0, 0, 50, 50], [0, 0, 1, 1]],
    ]


def test_get_shared_chunks():
    assert _get_shared_chunks([]) == {}
    assert _get_shared_chunks([[0]]) == {}
    assert _get_shared_chunks([[0], [1]]) == {}
    assert _get_shared_chunks([[0], [0, 1]]) == {0: [0, 1]}  # chunk 0 is shared by worker 0 and 1
    assert _get_shared_chunks([[0, 1], [1]]) == {1: [0, 1]}  # chunk 1 is shared by worker 0 and 1
    assert _get_shared_chunks([[2], [0, 1], [2, 3]]) == {2: [0, 2]}
    assert _get_shared_chunks([[2], [0, 1], [2, 3], [1, 4], [1]]) == {1: [1, 3, 4], 2: [0, 2]}


def test_find_chunks_per_workers_on_which_to_skip_deletion():
    # world size = 1, single worker
    chunks_to_disable = _find_chunks_per_workers_on_which_to_skip_deletion(
        num_workers=1,
        batch_size=1,
        workers_chunks=[[0]],
        workers_intervals=[[(0, 0, 50, 50)]],
    )
    assert chunks_to_disable == {}

    # world size = 1, multiple workers, no shared chunks
    chunks_to_disable = _find_chunks_per_workers_on_which_to_skip_deletion(
        num_workers=2,
        batch_size=5,
        workers_chunks=[[0], [1]],
        workers_intervals=[[(0, 0, 50, 50)], [(0, 0, 50, 50)]],
    )
    assert chunks_to_disable == {}

    # world size = 1, 2 workers sharing one chunk
    chunks_to_disable = _find_chunks_per_workers_on_which_to_skip_deletion(
        num_workers=2,
        batch_size=5,
        workers_chunks=[[0, 1], [1, 2]],
        workers_intervals=[[(0, 0, 50, 50), (0, 0, 25, 50)], [(0, 25, 50, 50), (0, 0, 50, 50)]],
    )
    assert chunks_to_disable == {1: [1]}

    # world size = 1, 4 workers sharing one chunk
    chunks_to_disable = _find_chunks_per_workers_on_which_to_skip_deletion(
        num_workers=4,
        batch_size=5,
        workers_chunks=[[0], [0], [0], [0]],
        workers_intervals=[[(0, 0, 50, 50)], [(0, 50, 100, 50)], [(0, 100, 150, 50)], [(0, 150, 200, 50)]],
    )
    assert chunks_to_disable == {0: [0, 1, 2]}

    # world size = 1, 4 workers sharing one chunk, different size
    chunks_to_disable = _find_chunks_per_workers_on_which_to_skip_deletion(
        num_workers=4,
        batch_size=5,
        workers_chunks=[[0], [0], [0], [0]],
        workers_intervals=[[(0, 0, 50, 50)], [(0, 50, 95, 50)], [(0, 95, 150, 50)], [(0, 150, 200, 50)]],
    )
    assert chunks_to_disable == {0: [0, 1, 3]}

    # world size 2, 2 workers per rank, varying batch size
    for batch_size in range(1, 6):
        chunks_to_disable = _find_chunks_per_workers_on_which_to_skip_deletion(
            num_workers=2,
            batch_size=batch_size,
            workers_chunks=[[0], [0], [0], [0]],
            workers_intervals=[
                [(0, 0, 50, 50)],
                [(0, 50, 95, 50)],
                [(0, 95, 145, 50)],
                [(0, 145, 205, 50)],  # last to access chunk 0
            ],
        )
        assert chunks_to_disable == {0: [0, 1, 2]}
        
    # world size 2, 2 workers per rank, sharing multiple chunks
    chunks_to_disable = _find_chunks_per_workers_on_which_to_skip_deletion(
        num_workers=2,
        batch_size=5,
        workers_chunks=[[0, 1], [3, 4], [1, 2], [4, 5]],
        workers_intervals=[
            [(0, 0, 50, 50), (0, 0, 50, 50)],
            [(0, 0, 50, 50), (0, 0, 50, 50)],
            [(0, 50, 100, 100), (0, 0, 50, 50)],
            [(0, 50, 100, 100), (0, 0, 50, 50)],
        ],
    )
    assert chunks_to_disable == {1: [2], 4: [3]}
