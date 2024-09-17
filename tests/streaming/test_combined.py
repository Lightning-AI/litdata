import os
import sys
from unittest.mock import ANY, MagicMock

import numpy as np
import pytest
import torch
from litdata.streaming.cache import Cache
from litdata.streaming.combined import CombinedStreamingDataset
from litdata.streaming.dataloader import StreamingDataLoader
from litdata.streaming.dataset import Dir, StreamingDataset
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader


class TestCombinedStreamingDataset(CombinedStreamingDataset):
    def _check_datasets(self, datasets) -> None:
        pass

    def reset_state_dict(self):
        pass


def test_combined_dataset_num_samples_yield():
    dataset = TestCombinedStreamingDataset(
        [range(10), range(0, -10, -1)], 42, weights=(0.5, 0.5), iterate_over_all=False
    )
    dataset_iter = iter(dataset)

    data = list(dataset_iter)
    assert data == [0, 0, 1, 2, -1, -2, -3, 3, 4, 5, 6, -4, 7, 8, -5, -6, 9, -7, -8]

    dataset = TestCombinedStreamingDataset(
        [range(10), range(0, -10, -1)], 37, weights=(0.5, 0.5), iterate_over_all=False
    )
    dataset_iter = iter(dataset)

    data = list(dataset_iter)
    assert data == [0, 0, -1, -2, -3, -4, -5, 1, -6, 2, -7, -8, 3, 4, -9, 5]

    dataset = TestCombinedStreamingDataset(
        [range(10), range(0, -10, -1)], 23, weights=(0.5, 0.5), iterate_over_all=False
    )
    dataset_iter = iter(dataset)

    data = [next(dataset_iter) for _ in range(5)]
    assert data == [0, -1, -2, 0, -3]
    assert dataset._iterator._num_samples_yielded == [1, 4]
    assert next(dataset_iter) == 1
    assert dataset._iterator._num_samples_yielded == [2, 4]


class Range:
    def __init__(self, start, end, step=1):
        self.values = list(range(start, end, step))

    def set_epoch(self, epoch):
        self.values = np.random.RandomState([42, epoch]).permutation(self.values).tolist()

    def __iter__(self):
        yield from self.values

    def __len__(self):
        return len(self.values)


def test_combined_dataset_iterate_over_all_4_datasets():
    dataset = TestCombinedStreamingDataset(
        [Range(0, 10), Range(10, 20), Range(20, 30), Range(30, 40)], 42, iterate_over_all=True
    )
    data = []
    for i in range(2):
        dataset.set_epoch(i)
        data.append(list(dataset))

    assert len(data[0]) == 40
    assert data[0][-3:] == [11, 18, 19]
    assert data[1][-3:] == [18, 11, 12]


def test_combined_dataset_num_samples_yield_iterate_over_all():
    dataset = TestCombinedStreamingDataset([range(10), range(0, -10, -1)], 42, iterate_over_all=True)
    assert len(dataset) == 20
    samples = list(dataset)
    assert len(samples) == 20


def test_drop_last_and_shuffle():
    dataset_mock_1 = MagicMock()
    dataset_mock_2 = MagicMock()
    dataset_mock_1.__len__.return_value = 1
    dataset_mock_2.__len__.return_value = 1

    dataset = TestCombinedStreamingDataset([dataset_mock_1, dataset_mock_2], 42, iterate_over_all=True)
    StreamingDataLoader(dataset, shuffle=True, drop_last=True)

    dataset_mock_1.set_shuffle.assert_called()
    dataset_mock_2.set_shuffle.assert_called()

    dataset_mock_1.set_drop_last.assert_called()
    dataset_mock_2.set_drop_last.assert_called()

    dataset_mock_1.set_num_workers.assert_called()
    dataset_mock_2.set_num_workers.assert_called()

    dataset_mock_1.set_batch_size.assert_called()
    dataset_mock_2.set_batch_size.assert_called()


class TestStatefulDataset:
    def __init__(self, size, step):
        self.size = size
        self.step = step
        self.counter = 0

    def __len__(self):
        return self.size

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter == self.size:
            raise StopIteration
        value = self.step * self.counter
        self.counter += 1
        return value

    def state_dict(self, *args, **kwargs):
        return {"counter": self.counter}

    def load_state_dict(self, state_dict):
        self.counter = state_dict["counter"]


def test_combined_dataset_state_dict():
    dataset = TestCombinedStreamingDataset(
        [TestStatefulDataset(10, 1), TestStatefulDataset(10, -1)],
        42,
        weights=(0.5, 0.5),
        iterate_over_all=False,
    )
    assert dataset.state_dict(0, 1) == {}
    dataset_iter = iter(dataset)
    assert dataset.state_dict(0, 1) == {"0": {"counter": 0}, "1": {"counter": 0}}

    dataset2 = TestCombinedStreamingDataset(
        [TestStatefulDataset(10, 1), TestStatefulDataset(10, -1)],
        42,
        weights=(0.5, 0.5),
        iterate_over_all=False,
    )
    assert dataset2.state_dict(0, 1) == {}

    data = []
    states = []
    for i, value in enumerate(dataset_iter):
        state = dataset.state_dict(i, 1)
        data.append(value)
        states.append(state)

    assert data == [0, 0, 1, 2, -1, -2, -3, 3, 4, 5, 6, -4, 7, 8, -5, -6, 9, -7, -8]
    assert states == [
        {"0": {"counter": 0}, "1": {"counter": 1}},
        {"0": {"counter": 1}, "1": {"counter": 1}},
        {"0": {"counter": 2}, "1": {"counter": 1}},
        {"0": {"counter": 3}, "1": {"counter": 1}},
        {"0": {"counter": 3}, "1": {"counter": 2}},
        {"0": {"counter": 3}, "1": {"counter": 3}},
        {"0": {"counter": 3}, "1": {"counter": 4}},
        {"0": {"counter": 4}, "1": {"counter": 4}},
        {"0": {"counter": 5}, "1": {"counter": 4}},
        {"0": {"counter": 6}, "1": {"counter": 4}},
        {"0": {"counter": 7}, "1": {"counter": 4}},
        {"0": {"counter": 7}, "1": {"counter": 5}},
        {"0": {"counter": 8}, "1": {"counter": 5}},
        {"0": {"counter": 9}, "1": {"counter": 5}},
        {"0": {"counter": 9}, "1": {"counter": 6}},
        {"0": {"counter": 9}, "1": {"counter": 7}},
        {"0": {"counter": 10}, "1": {"counter": 7}},
        {"0": {"counter": 10}, "1": {"counter": 8}},
        {"0": {"counter": 10}, "1": {"counter": 9}},
    ]

    dataset2 = TestCombinedStreamingDataset(
        [TestStatefulDataset(10, 1), TestStatefulDataset(10, -1)],
        42,
        weights=(0.5, 0.5),
        iterate_over_all=False,
    )
    assert dataset2.state_dict(0, 1) == {}
    dataset2_iter = iter(dataset2)

    data_2 = []
    for state in states:
        dataset.load_state_dict({"dataset": state})
        data_2.append(next(dataset2_iter))

    assert data == data_2


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ([1], [1]),
        ([2], [1]),
        ([2, 0.5], [0.8, 0.2]),
        ([1, 1, 1], [1 / 3, 1 / 3, 1 / 3]),
        ([0.3, 0, 0], [1.0, 0, 0]),
        (None, [1 / 3, 2 / 3]),
    ],
)
def test_combined_dataset_normalizes_weights(weights, expected):
    combined_dataset = TestCombinedStreamingDataset([[1], [2, 3]], weights=weights, iterate_over_all=False, seed=1)
    assert combined_dataset._weights == expected


class SimpleDataset(IterableDataset):
    def __init__(self, start, end):
        super().__init__()
        self._start = start
        self._end = end

    def __iter__(self):
        return iter(range(self._start, self._end))

    def state_dict(self, **kwargs):
        return kwargs

    def set_epoch(self, current_epoch):
        pass

    def set_shuffle(self, _):
        pass

    def set_drop_last(self, _):
        pass

    def set_batch_size(self, _):
        pass

    def set_num_workers(self, _):
        pass


def test_combined_dataset():
    dataset1 = SimpleDataset(0, 10)
    dataset2 = SimpleDataset(10, 20)
    dataset = TestCombinedStreamingDataset(
        datasets=[dataset1, dataset2], weights=[1.0, 0.0], iterate_over_all=False, seed=12345
    )

    res = list(dataset)
    assert res == list(range(0, 10))

    dataset1 = SimpleDataset(0, 10)
    dataset2 = SimpleDataset(10, 20)
    dataset = TestCombinedStreamingDataset(
        datasets=[dataset1, dataset2], weights=[0.0, 1.0], iterate_over_all=False, seed=12345
    )

    res = list(dataset)
    assert res == list(range(10, 20))

    dataset1 = SimpleDataset(0, 10)
    dataset2 = SimpleDataset(10, 20)
    dataset = TestCombinedStreamingDataset(
        datasets=[dataset1, dataset2], weights=[0.5, 0.5], iterate_over_all=False, seed=12345
    )

    res = list(dataset)
    assert 9 in res or 19 in res
    if len(res) > 10:
        assert 0 in res
        assert 10 in res

    dataset1 = SimpleDataset(0, 10)
    dataset2 = SimpleDataset(10, 20)
    dataset = TestCombinedStreamingDataset(
        datasets=[dataset1, dataset2], weights=[0.5, 0.5], iterate_over_all=False, seed=12345
    )
    dataloader = DataLoader(dataset, batch_size=2, num_workers=1)
    dataloader_iter = iter(dataloader)
    assert torch.equal(next(dataloader_iter), torch.Tensor([0, 1]))


@pytest.mark.parametrize("batch_size", [1, 2])
def test_combined_dataset_with_dataloader_and_one_worker(batch_size):
    dataset1 = SimpleDataset(0, 10)
    dataset2 = SimpleDataset(10, 20)
    dataset = TestCombinedStreamingDataset(
        datasets=[dataset1, dataset2], weights=[0.5, 0.5], iterate_over_all=False, seed=12345
    )
    dataloader = StreamingDataLoader(dataset, num_workers=1, batch_size=batch_size, prefetch_factor=1)
    dataloader_iter = iter(dataloader)

    if batch_size == 2:
        assert torch.equal(next(dataloader_iter), torch.Tensor([0, 1]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([10, 2]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([3, 4]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([11, 5]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([6, 7]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([12, 8]))

    else:
        assert torch.equal(next(dataloader_iter), torch.Tensor([0]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([1]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([10]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([2]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([3]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([4]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([11]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([5]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([6]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([7]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([12]))
        assert torch.equal(next(dataloader_iter), torch.Tensor([8]))

    assert dataloader.state_dict() == {
        "dataset": {
            "0": {"num_samples_yielded": 9, "num_workers": 1, "batch_size": batch_size},
            "1": {"num_samples_yielded": 3, "num_workers": 1, "batch_size": batch_size},
        },
        "current_epoch": 1,
        "latest_worker_idx": 0,
        "num_samples_yielded": {0: [9, 3]},
    }


@pytest.mark.skipif(sys.platform == "win32", reason="too slow in CI")
def test_combined_dataset_with_dataloader_2_epochs(tmpdir):
    data_dir_1 = os.path.join(tmpdir, "data_1")
    data_dir_2 = os.path.join(tmpdir, "data_2")
    cache_dir_1 = os.path.join(tmpdir, "cache_dir_1")
    cache_dir_2 = os.path.join(tmpdir, "cache_dir_2")

    os.makedirs(data_dir_1)
    os.makedirs(data_dir_2)
    os.makedirs(cache_dir_1)
    os.makedirs(cache_dir_2)

    cache = Cache(input_dir=str(data_dir_1), chunk_size=2)

    for i in range(10):
        cache[i] = i

    cache.done()
    cache.merge()

    cache = Cache(input_dir=str(data_dir_2), chunk_size=2)

    for i in range(10):
        cache[i] = i + 5

    cache.done()
    cache.merge()

    dataset1 = StreamingDataset(input_dir=Dir(cache_dir_1, data_dir_1), shuffle=True)
    dataset2 = StreamingDataset(input_dir=Dir(cache_dir_2, data_dir_2), shuffle=True)
    dataset = CombinedStreamingDataset(
        datasets=[dataset1, dataset2], weights=[0.5, 0.5], iterate_over_all=False, seed=12345
    )
    dataloader = StreamingDataLoader(dataset, num_workers=3, batch_size=2)

    assert dataset1.current_epoch == 1
    assert dataset2.current_epoch == 1

    batches_1 = []
    states_1 = []
    for batch in dataloader:
        batches_1.append(batch)
        states_1.append(dataloader.state_dict())

    assert dataset1.current_epoch == 1
    assert dataset2.current_epoch == 1

    batches_2 = []
    states_2 = []
    for batch in dataloader:
        batches_2.append(batch)
        states_2.append(dataloader.state_dict())
    assert dataset1.current_epoch == 2
    assert dataset2.current_epoch == 2

    assert sum(torch.equal(b1, b2) for b1, b2 in zip(batches_1, batches_2)) != len(batches_1)

    assert states_1 == [
        {
            "dataset": {
                "0": {
                    "num_samples_yielded": 2,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
                "1": {
                    "num_samples_yielded": 0,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
            },
            "current_epoch": 1,
            "latest_worker_idx": 0,
            "num_samples_yielded": {0: [2, 0]},
        },
        {
            "dataset": {
                "0": {
                    "num_samples_yielded": 4,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
                "1": {
                    "num_samples_yielded": 0,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
            },
            "current_epoch": 1,
            "latest_worker_idx": 1,
            "num_samples_yielded": {0: [2, 0], 1: [2, 0]},
        },
        {
            "dataset": {
                "0": {
                    "num_samples_yielded": 6,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
                "1": {
                    "num_samples_yielded": 0,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
            },
            "current_epoch": 1,
            "latest_worker_idx": 2,
            "num_samples_yielded": {0: [2, 0], 1: [2, 0], 2: [2, 0]},
        },
        {
            "dataset": {
                "0": {
                    "num_samples_yielded": 7,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
                "1": {
                    "num_samples_yielded": 1,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
            },
            "current_epoch": 1,
            "latest_worker_idx": 0,
            "num_samples_yielded": {0: [3, 1], 1: [2, 0], 2: [2, 0]},
        },
        {
            "dataset": {
                "0": {
                    "num_samples_yielded": 8,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
                "1": {
                    "num_samples_yielded": 2,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
            },
            "current_epoch": 1,
            "latest_worker_idx": 1,
            "num_samples_yielded": {0: [3, 1], 1: [3, 1], 2: [2, 0]},
        },
        {
            "dataset": {
                "0": {
                    "num_samples_yielded": 9,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
                "1": {
                    "num_samples_yielded": 3,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
            },
            "current_epoch": 1,
            "latest_worker_idx": 2,
            "num_samples_yielded": {0: [3, 1], 1: [3, 1], 2: [3, 1]},
        },
        {
            "dataset": {
                "0": {
                    "num_samples_yielded": 10,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
                "1": {
                    "num_samples_yielded": 3,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 1,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
            },
            "current_epoch": 1,
            "latest_worker_idx": 0,
            "num_samples_yielded": {0: [4, 1], 1: [3, 1], 2: [3, 1]},
        },
    ]

    assert states_2 == [
        {
            "dataset": {
                "0": {
                    "num_samples_yielded": 2,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
                "1": {
                    "num_samples_yielded": 0,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
            },
            "current_epoch": 2,
            "latest_worker_idx": 0,
            "num_samples_yielded": {0: [2, 0]},
        },
        {
            "dataset": {
                "0": {
                    "num_samples_yielded": 4,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
                "1": {
                    "num_samples_yielded": 0,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
            },
            "current_epoch": 2,
            "latest_worker_idx": 1,
            "num_samples_yielded": {0: [2, 0], 1: [2, 0]},
        },
        {
            "dataset": {
                "0": {
                    "num_samples_yielded": 6,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
                "1": {
                    "num_samples_yielded": 0,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
            },
            "current_epoch": 2,
            "latest_worker_idx": 2,
            "num_samples_yielded": {0: [2, 0], 1: [2, 0], 2: [2, 0]},
        },
        {
            "dataset": {
                "0": {
                    "num_samples_yielded": 7,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
                "1": {
                    "num_samples_yielded": 1,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
            },
            "current_epoch": 2,
            "latest_worker_idx": 0,
            "num_samples_yielded": {0: [3, 1], 1: [2, 0], 2: [2, 0]},
        },
        {
            "dataset": {
                "0": {
                    "num_samples_yielded": 8,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
                "1": {
                    "num_samples_yielded": 2,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
            },
            "current_epoch": 2,
            "latest_worker_idx": 1,
            "num_samples_yielded": {0: [3, 1], 1: [3, 1], 2: [2, 0]},
        },
        {
            "dataset": {
                "0": {
                    "num_samples_yielded": 9,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
                "1": {
                    "num_samples_yielded": 3,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
            },
            "current_epoch": 2,
            "latest_worker_idx": 2,
            "num_samples_yielded": {0: [3, 1], 1: [3, 1], 2: [3, 1]},
        },
        {
            "dataset": {
                "0": {
                    "num_samples_yielded": 10,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
                "1": {
                    "num_samples_yielded": 3,
                    "num_workers": 3,
                    "batch_size": 2,
                    "current_epoch": 2,
                    "input_dir_path": ANY,
                    "input_dir_url": ANY,
                    "item_loader": None,
                    "drop_last": False,
                    "seed": 42,
                    "world_size": 1,
                    "shuffle": True,
                    "subsampled_files": ANY,
                    "region_of_interest": ANY,
                },
            },
            "current_epoch": 2,
            "latest_worker_idx": 0,
            "num_samples_yielded": {0: [4, 1], 1: [3, 1], 2: [3, 1]},
        },
    ]

    dataloader.load_state_dict(states_2[1])

    assert dataloader.restore

    batches_23 = []
    states_23 = []
    for batch in dataloader:
        batches_23.append(batch)
        states_23.append(dataloader.state_dict())

    assert sum(not torch.equal(b1, b2) for b1, b2 in zip(batches_2[2:], batches_23)) == 0
    assert states_23[0]["current_epoch"] == 2

    assert not dataloader.restore


@pytest.mark.timeout(120)
def test_combined_dataset_dataloader_states_complete_iterations(tmpdir):
    datasets = [str(tmpdir.join(f"dataset_{i}")) for i in range(2)]
    for dataset in datasets:
        cache = Cache(input_dir=dataset, chunk_bytes="64MB")
        for i in range(50):
            cache[i] = i
        cache.done()
        cache.merge()

    dataset_1 = StreamingDataset(datasets[0], shuffle=True)
    dataset_2 = StreamingDataset(datasets[1], shuffle=True)
    combined_dataset = CombinedStreamingDataset(datasets=[dataset_1, dataset_2])

    # Test dataloader without explicit num workers
    dataloader = StreamingDataLoader(combined_dataset, batch_size=4)
    assert not dataloader.restore
    dataloader.load_state_dict(dataloader.state_dict())
    assert not dataloader.restore
    batch = next(iter(dataloader))
    assert len(batch) == 4, "Batch size should be 4"
    assert len(dataloader) == 25, "Dataloader length should be 25 (50+50 items / batch size 4)"

    # Test dataloader with num workers
    dataloader = StreamingDataLoader(combined_dataset, batch_size=4, num_workers=4)
    assert len(dataloader) == 25, "Dataloader length should be 25 (50+50 items / batch size 4)"

    # Verify dataloader state after complete last iteration
    for batch in dataloader:
        assert dataloader.current_epoch == 1, "Current epoch should be 1"
        pass
    dataloader.load_state_dict(dataloader.state_dict())
    assert not dataloader.restore
    for batch in dataloader:
        assert dataloader.current_epoch == 2, "Current epoch should be 2"
        pass
    assert not dataloader.restore


@pytest.mark.timeout(120)
def test_combined_dataset_dataloader_states_partial_iterations(tmpdir):
    datasets = [str(tmpdir.join(f"dataset_{i}")) for i in range(2)]
    for dataset in datasets:
        cache = Cache(input_dir=dataset, chunk_bytes="64MB")
        for i in range(50):
            cache[i] = i
        cache.done()
        cache.merge()

    dataset_1 = StreamingDataset(datasets[0], shuffle=True)
    dataset_2 = StreamingDataset(datasets[1], shuffle=True)
    combined_dataset = CombinedStreamingDataset(datasets=[dataset_1, dataset_2])

    # Verify dataloader state after partial last iteration
    dataloader = StreamingDataLoader(combined_dataset, batch_size=4, num_workers=4)

    for batch_idx, batch in enumerate(dataloader):
        assert dataloader.current_epoch == 1, "Current epoch should be 1"
        if batch_idx == 15:
            break
    state_dict = dataloader.state_dict()
    dataloader.load_state_dict(state_dict)
    assert dataloader.restore

    print(dataloader.state_dict())

    # Verify remaining batches in the first epoch
    count = 0
    for _ in dataloader:
        assert dataloader.current_epoch == 1, "Current epoch should be 1"
        count += 1
    assert count == 15, "There should be atleast 15 batches remaining in the first epoch"
    assert not dataloader.restore

    # Verify batches in the second epoch
    count = 0
    for _ in dataloader:
        assert dataloader.current_epoch == 2, "Current epoch should be 2"
        count += 1
    assert count >= 25, "There should be at least 25 batches in the second epoch"

    # TODO: Add more conditions to check the state of the dataloader
