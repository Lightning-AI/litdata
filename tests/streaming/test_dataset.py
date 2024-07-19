# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import random
import shutil
import sys
from time import sleep
from unittest import mock

import numpy as np
import pytest
import torch
from litdata import optimize, train_test_split
from litdata.constants import _ZSTD_AVAILABLE
from litdata.processing import functions
from litdata.streaming import Cache
from litdata.streaming import dataset as dataset_module
from litdata.streaming.dataloader import StreamingDataLoader
from litdata.streaming.dataset import (
    _INDEX_FILENAME,
    Dir,
    StreamingDataset,
    _replay_chunks_sampling,
    _replay_sampling,
)
from litdata.streaming.item_loader import TokensLoader
from litdata.streaming.shuffle import FullShuffle, NoShuffle
from litdata.utilities import dataset_utilities as dataset_utilities_module
from litdata.utilities.env import _DistributedEnv, _WorkerEnv
from litdata.utilities.shuffle import _associate_chunks_and_internals_to_workers
from torch.utils.data import DataLoader


def seed_everything(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        pytest.param("zstd", marks=pytest.mark.skipif(condition=not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")),
    ],
)
@pytest.mark.timeout(15)
def test_streaming_dataset(tmpdir, monkeypatch, compression):
    seed_everything(42)

    with pytest.raises(ValueError, match="The provided dataset"):
        dataset = StreamingDataset(input_dir=str(tmpdir))

    cache = Cache(str(tmpdir), chunk_size=10, compression=compression)
    for i in range(60):
        cache[i] = i
    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir))

    assert len(dataset) == 60
    for i in range(60):
        assert dataset[i] == i

    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 60
    for i in range(60):
        assert next(dataset_iter) == i

    dataloader = DataLoader(dataset, num_workers=2, batch_size=1)
    assert len(dataloader) == 60
    dataloader = DataLoader(dataset, num_workers=2, batch_size=2)
    assert len(dataloader) == 30


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        pytest.param("zstd", marks=pytest.mark.skipif(condition=not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")),
    ],
)
@pytest.mark.timeout(30)
def test_streaming_dataset_distributed_no_shuffle(drop_last, tmpdir, compression):
    seed_everything(42)

    cache = Cache(str(tmpdir), chunk_size=10, compression=compression)
    for i in range(101):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir), shuffle=False, drop_last=drop_last)
    assert not dataset.shuffle
    _ = dataset[0]  # init shuffler
    assert isinstance(dataset.shuffler, NoShuffle)

    for i in range(101):
        assert dataset[i] == i

    dataset.distributed_env = _DistributedEnv(1, 0, 1)
    assert len(dataset) == 101

    dataset.distributed_env = _DistributedEnv(2, 0, 1)
    assert len(dataset) == 50

    dataset.distributed_env = _DistributedEnv(2, 1, 1)
    assert len(dataset) == 50 + int(not drop_last)

    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 50 + int(not drop_last)

    dataset.distributed_env = _DistributedEnv(2, 0, 1)

    process_1_1 = list(dataset_iter)

    assert len(process_1_1) == 50
    assert process_1_1[:10] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dataset_iter = iter(dataset)

    assert len(dataset_iter) == 50
    process_1_2 = list(dataset_iter)
    assert process_1_2[:10] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    assert len(process_1_2) == 50

    dataset = StreamingDataset(input_dir=str(tmpdir), shuffle=False, drop_last=drop_last)
    dataset.distributed_env = _DistributedEnv(2, 1, 1)

    assert len(dataset) == 50 + int(not drop_last)
    dataset_iter = iter(dataset)

    process_2_1 = list(dataset_iter)
    assert process_2_1[:10] == [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

    assert len(process_2_1) == 50 + int(not drop_last)
    dataset_iter = iter(dataset)

    assert len(dataset_iter) == 50 + int(not drop_last)
    process_2_2 = list(dataset_iter)

    assert process_2_2[:10] == [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

    assert len(process_2_2) == 50 + int(not drop_last)

    _, workers_intervals = dataset.shuffler.get_chunks_and_intervals_per_workers(
        dataset.distributed_env, 1, 1, dataset.current_epoch
    )

    assert process_1_1 == process_1_2

    found_list = []
    for i in process_1_1:
        found = False
        for interval in workers_intervals[0]:
            if interval[1] <= i <= interval[2]:
                found = True
                break
        found_list.append(found)

    assert all(found_list) is True

    found_list = []
    for i in process_2_1:
        found = False
        for interval in workers_intervals[1]:
            if interval[1] <= i <= interval[2]:
                found = True
                break
        found_list.append(found)

    assert all(found_list) is True

    assert len([i for i in process_1_1 if i in process_2_1]) == 0
    assert len([i for i in process_1_2 if i in process_2_2]) == 0


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        pytest.param("zstd", marks=pytest.mark.skipif(condition=not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")),
    ],
)
@pytest.mark.timeout(60)
def test_streaming_dataset_distributed_full_shuffle_odd(drop_last, tmpdir, compression):
    seed_everything(42)

    cache = Cache(input_dir=str(tmpdir), chunk_size=10, compression=compression)
    for i in range(1097):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    assert dataset.shuffle
    _ = dataset[0]
    assert isinstance(dataset.shuffler, FullShuffle)

    for i in range(1097):
        assert dataset[i] == i

    dataset.distributed_env = _DistributedEnv(2, 0, 1)
    assert len(dataset) == 548
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 548
    process_1_1 = list(dataset_iter)
    assert process_1_1[:10] == [224, 227, 229, 226, 225, 222, 228, 221, 220, 223]
    assert len(process_1_1) == 548

    dataset_2 = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    iter(dataset_2)
    assert isinstance(dataset_2.shuffler, FullShuffle)
    dataset_2.distributed_env = _DistributedEnv(2, 1, 1)
    assert len(dataset_2) == 548 + int(not drop_last)
    dataset_2_iter = iter(dataset_2)
    assert len(dataset_2_iter) == 548 + int(not drop_last)
    process_2_1 = list(dataset_2_iter)
    assert process_2_1[:10] == [279, 278, 105, 104, 106, 107, 103, 101, 102, 109]
    assert len(process_2_1) == 548 + int(not drop_last)
    assert len([i for i in process_1_1 if i in process_2_1]) == 0


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        pytest.param(
            "zstd",
            marks=pytest.mark.skipif(
                condition=not _ZSTD_AVAILABLE or sys.platform == "darwin", reason="Requires: ['zstd']"
            ),
        ),
    ],
)
@pytest.mark.timeout(30)
def test_streaming_dataset_distributed_full_shuffle_even(drop_last, tmpdir, compression):
    seed_everything(42)

    cache = Cache(str(tmpdir), chunk_size=10, compression=compression)
    for i in range(1222):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    assert dataset.shuffle
    _ = dataset[0]
    assert isinstance(dataset.shuffler, FullShuffle)

    for i in range(1222):
        assert dataset[i] == i

    dataset.distributed_env = _DistributedEnv(2, 0, 1)
    assert len(dataset) == 611
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 611
    process_1_1 = list(dataset_iter)
    assert process_1_1[:10] == [818, 810, 812, 815, 814, 817, 813, 819, 816, 811]
    assert len(process_1_1) == 611

    dataset_2 = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    iter(dataset_2)
    assert isinstance(dataset_2.shuffler, FullShuffle)
    dataset_2.distributed_env = _DistributedEnv(2, 1, 1)
    assert len(dataset_2) == 611
    dataset_2_iter = iter(dataset_2)
    assert len(dataset_2_iter) == 611
    process_2_1 = list(dataset_2_iter)
    assert process_2_1[:10] == [181, 183, 186, 188, 187, 185, 189, 184, 182, 1092]
    assert len(process_2_1) == 611
    assert len([i for i in process_1_1 if i in process_2_1]) == 0


@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        pytest.param("zstd", marks=pytest.mark.skipif(condition=not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")),
    ],
)
@pytest.mark.timeout(60)
def test_streaming_dataset_distributed_full_shuffle_even_multi_nodes(drop_last, tmpdir, compression):
    seed_everything(42)

    cache = Cache(str(tmpdir), chunk_size=10, compression=compression)
    for i in range(1222):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    assert dataset.shuffle
    _ = dataset[0]
    assert isinstance(dataset.shuffler, FullShuffle)

    for i in range(1222):
        assert dataset[i] == i

    dataset.distributed_env = _DistributedEnv(4, 0, 2)
    assert len(dataset) == 305
    dataset_iter = iter(dataset)
    assert len(dataset_iter) == 305
    process_1_1 = list(dataset_iter)
    assert process_1_1[:10] == [817, 816, 812, 810, 814, 815, 819, 813, 818, 811]
    assert len(process_1_1) == 305

    dataset_2 = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    iter(dataset_2)
    assert isinstance(dataset_2.shuffler, FullShuffle)
    dataset_2.distributed_env = _DistributedEnv(4, 1, 2)
    assert len(dataset_2) == 305
    dataset_2_iter = iter(dataset_2)
    assert len(dataset_2_iter) == 305
    process_2_1 = list(dataset_2_iter)
    assert process_2_1[:10] == [1087, 1088, 1089, 1085, 1086, 4, 3, 0, 5, 1]
    assert len(process_2_1) == 305
    assert len([i for i in process_1_1 if i in process_2_1]) == 0

    dataset_2 = StreamingDataset(input_dir=str(tmpdir), shuffle=True, drop_last=drop_last)
    iter(dataset_2)
    assert isinstance(dataset_2.shuffler, FullShuffle)
    dataset_2.distributed_env = _DistributedEnv(4, 1, 2)
    dataset_2.current_epoch = 2
    assert len(dataset_2) == 310
    dataset_2_iter = iter(dataset_2)
    assert len(dataset_2_iter) == 310
    process_2_1 = list(dataset_2_iter)
    assert process_2_1[:10] == [1018, 1010, 1012, 1015, 1014, 1017, 1013, 1019, 1016, 1011]
    assert len(process_2_1) == 310
    assert len([i for i in process_1_1 if i in process_2_1]) != 0


def test_streaming_dataset_deepcopy(tmpdir):
    seed_everything(42)

    remote_dir = os.path.join(tmpdir, "remote_dir")

    os.makedirs(remote_dir, exist_ok=True)

    cache = Cache(remote_dir, chunk_size=10)
    for i in range(10):
        cache[i] = i

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=remote_dir, shuffle=True)
    assert dataset.cache is None
    iter(dataset)
    assert dataset.cache is not None
    assert dataset.cache._reader._prepare_thread is None
    dataset.cache._reader._prepare_thread = True
    dataloader = DataLoader(dataset, num_workers=1)

    batches = []
    for batch in dataloader:
        batches.append(batch)

    assert len(batches) == 10


def test_dataset_cache_recreation(tmpdir):
    """Test that we recreate the cache and other objects only when appropriate."""
    cache = Cache(str(tmpdir), chunk_size=10)
    for i in range(10):
        cache[i] = i
    cache.done()
    cache.merge()

    # repated `len()` calls
    dataset = StreamingDataset(input_dir=str(tmpdir))
    assert not dataset.cache
    assert not dataset.shuffler
    len(dataset)
    assert not dataset.cache
    shuffler = dataset.shuffler
    assert isinstance(shuffler, NoShuffle)
    len(dataset)
    assert dataset.shuffler is shuffler

    # repeated `iter()` calls
    dataset = StreamingDataset(input_dir=str(tmpdir))
    assert not dataset.cache
    assert not dataset.shuffler
    iter(dataset)
    cache = dataset.cache
    shuffler = dataset.shuffler
    assert isinstance(cache, Cache)
    assert isinstance(shuffler, NoShuffle)
    iter(dataset)
    assert isinstance(dataset.cache, Cache)
    assert isinstance(dataset.shuffler, NoShuffle)
    assert dataset.cache is not cache  # cache gets recreated
    assert dataset.shuffler is not shuffler  # shuffler gets recreated

    # repeated `getitem()` calls
    dataset = StreamingDataset(input_dir=str(tmpdir))
    assert not dataset.cache
    assert not dataset.shuffler
    _ = dataset[0]
    cache = dataset.cache
    shuffler = dataset.shuffler
    assert isinstance(cache, Cache)
    assert isinstance(shuffler, NoShuffle)
    _ = dataset[1]
    assert dataset.cache is cache  # cache gets reused
    assert dataset.shuffler is shuffler  # shuffler gets reused


def test_dataset_for_text_tokens(tmpdir):
    seed_everything(42)

    block_size = 1024 + 1
    cache = Cache(input_dir=str(tmpdir), chunk_size=block_size * 11, item_loader=TokensLoader(block_size))
    text_idxs_list = []

    counter = 0
    while True:
        text_ids = torch.randint(0, 1000, (np.random.randint(0, 1000),)).to(torch.int)
        text_idxs_list.append(text_ids)
        chunk_filepath = cache._add_item(counter, text_ids)
        if chunk_filepath:
            break
        counter += 1

    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir), item_loader=TokensLoader(block_size))

    assert len(dataset) == 10

    cache_0 = dataset[0]
    cache_1 = dataset[1]
    cache_2 = dataset[2]
    cache_3 = dataset[3]
    assert len(cache_0) == block_size
    assert len(cache_1) == block_size
    assert not torch.equal(cache_0, cache[1])
    indices = torch.cat(text_idxs_list, dim=0)
    assert torch.equal(cache_0, indices[: len(cache_0)])
    assert torch.equal(cache_1, indices[len(cache_0) : len(cache_0) + len(cache_1)])

    dataloader = DataLoader(StreamingDataset(input_dir=str(tmpdir), item_loader=TokensLoader(block_size)), batch_size=2)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == 0:
            assert torch.equal(torch.stack([cache_0, cache_1]), batch)
        elif batch_idx == 1:
            assert torch.equal(torch.stack([cache_2, cache_3]), batch)
        else:
            break


def test_dataset_for_text_tokens_multiple_workers(tmpdir):
    seed_everything(42)

    block_size = 10
    cache = Cache(input_dir=str(tmpdir), chunk_size=40, item_loader=TokensLoader(block_size))

    counter = 0
    for i in range(10):
        text_ids = torch.arange(counter, counter + 20).to(torch.int)
        cache[i] = text_ids
        counter += 20

    cache.done()
    cache.merge()

    for i in range(20):
        sequence = cache[i]
        assert sequence[0].item() == i * block_size
        assert sequence[-1].item() == (i + 1) * block_size - 1

    assert len(os.listdir(tmpdir)) == 6

    dataset = StreamingDataset(input_dir=str(tmpdir), item_loader=TokensLoader(block_size), shuffle=False)

    assert len(dataset) == 20

    dataloader = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=False)

    assert len(dataloader) == 10

    expected = [
        [0, 10],
        [100, 110],
        [20, 30],
        [120, 130],
        [40, 50],
        [140, 150],
        [60, 70],
        [160, 170],
        [80, 90],
        [180, 190],
    ]

    result = []
    for batch in dataloader:
        result.append(batch[:, 0].tolist())
    assert result == expected


def test_dataset_for_text_tokens_distributed_num_workers(tmpdir):
    seed_everything(42)

    block_size = 10
    cache = Cache(input_dir=str(tmpdir), chunk_size=40, item_loader=TokensLoader(block_size))

    counter = 0
    for i in range(10):
        text_ids = torch.arange(counter, counter + 20).to(torch.int)
        cache[i] = text_ids
        counter += 20

    cache.done()
    cache.merge()

    for i in range(20):
        sequence = cache[i]
        assert sequence[0].item() == i * block_size
        assert sequence[-1].item() == (i + 1) * block_size - 1

    assert len([f for f in os.listdir(tmpdir) if f.endswith(".bin")]) == 5

    dataset = StreamingDataset(input_dir=str(tmpdir), item_loader=TokensLoader(block_size), shuffle=False)

    assert len(dataset) == 20

    dataset.distributed_env = _DistributedEnv(2, 0, 1)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    assert len(dataloader) == 5

    expected = [[0, 10], [20, 30], [40, 50], [60, 70], [80, 90]]

    for batch_idx, batch in enumerate(dataloader):
        assert [batch[0][0].item(), batch[1][0].item()] == expected[batch_idx]

    dataset.distributed_env = _DistributedEnv(2, 1, 1)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    assert len(dataloader) == 5

    expected = [[100, 110], [120, 130], [140, 150], [160, 170], [180, 190]]

    for batch_idx, batch in enumerate(dataloader):
        assert [batch[0][0].item(), batch[1][0].item()] == expected[batch_idx]

    for batch_idx, batch in enumerate(dataloader):
        assert [batch[0][0].item(), batch[1][0].item()] == expected[batch_idx]


def optimize_fn(item):
    return torch.arange(item[0], item[0] + 20).to(torch.int)


def test_dataset_for_text_tokens_distributed_num_workers_end_to_end(tmpdir, monkeypatch):
    monkeypatch.setattr(functions, "_get_input_dir", lambda x: str(tmpdir))

    seed_everything(42)

    with open(tmpdir / "a.txt", "w") as f:
        f.write("hello")

    inputs = [(v, str(tmpdir / "a.txt")) for v in range(0, 200, 20)]

    cache_dir = os.path.join(tmpdir, "cache")
    output_dir = os.path.join(tmpdir, "target_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", cache_dir)

    functions.optimize(
        optimize_fn, inputs, output_dir=str(tmpdir), num_workers=2, chunk_size=2, reorder_files=False, num_downloaders=1
    )

    assert len([f for f in os.listdir(tmpdir) if f.endswith(".bin")]) == 10

    block_size = 10
    dataset = StreamingDataset(input_dir=str(tmpdir), item_loader=TokensLoader(block_size), shuffle=False)

    L = len(dataset)
    assert L == 20

    for i in range(L):
        sequence = dataset[i]
        assert sequence[0].item() == i * block_size
        assert sequence[-1].item() == (i + 1) * block_size - 1

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("GLOBAL_RANK", "0")
    monkeypatch.setenv("NNODES", "1")
    dataset = StreamingDataset(input_dir=str(tmpdir), item_loader=TokensLoader(block_size), shuffle=False)
    dataloader = StreamingDataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
    assert dataset.drop_last  # in distributed setting, this is forced automatically

    # L = 20, world size 2, num workers 2
    # L / (2 * 2) = 5 items per worker
    # drop last -> 4 items per worker
    # batch size = 2 -> 2 batches per worker -> len(dataloader) = 4
    assert len(dataloader) == 4

    expected = [[0, 10], [40, 50], [20, 30], [60, 70]]
    returned = []
    for batch in dataloader:
        returned.append(batch[:, 0].tolist())
    assert returned == expected

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("GLOBAL_RANK", "1")
    monkeypatch.setenv("NNODES", "1")
    dataset = StreamingDataset(input_dir=str(tmpdir), item_loader=TokensLoader(block_size), shuffle=False)
    dataloader = StreamingDataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
    assert dataset.drop_last  # in distributed setting, this is forced automatically

    assert len(dataloader) == 4

    expected = [[80, 90], [120, 130], [100, 110], [140, 150]]
    returned = []
    for batch in dataloader:
        returned.append(batch[:, 0].tolist())
    assert returned == expected


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows and MacOs")
def test_s3_streaming_dataset(monkeypatch):
    downloader = mock.MagicMock()

    def fn(remote_chunkpath: str, local_chunkpath: str):
        with open(local_chunkpath, "w") as f:
            json.dump({"chunks": [{"chunk_size": 2, "filename": "0.bin"}]}, f)

    downloader.download_file = fn

    monkeypatch.setattr(dataset_utilities_module, "get_downloader_cls", mock.MagicMock(return_value=downloader))

    dataset = StreamingDataset(input_dir="s3://pl-flash-data/optimized_tiny_imagenet")
    assert dataset.input_dir.url == "s3://pl-flash-data/optimized_tiny_imagenet"
    assert dataset.input_dir.path.endswith(
        "/chunks/597d6184e3ba942b36c8b6357a890033"
    )  # it won't be None, and a cache dir will be created


class EmulateS3StreamingDataset(StreamingDataset):
    def _create_cache(self, worker_env: _WorkerEnv) -> Cache:
        cache_dir = os.path.join(self.input_dir.path)
        os.makedirs(cache_dir, exist_ok=True)

        cache = Cache(
            input_dir=Dir(cache_dir, self.input_dir.url),
            item_loader=self.item_loader,
            chunk_bytes=1,
            serializers=self.serializers,
        )
        cache._reader._try_load_config()

        if not cache.filled:
            raise ValueError(
                f"The provided dataset `{self.input_dir}` doesn't contain any {_INDEX_FILENAME} file."
                " HINT: Did you successfully optimize a dataset to the provided `input_dir`?"
            )

        return cache


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows and MacOs")
def test_dataset_reshuffling_every_epoch(tmpdir):
    seed_everything(42)

    data_dir = os.path.join(tmpdir, "data")
    cache_dir = os.path.join(tmpdir, "cache_dir")

    os.makedirs(data_dir)
    os.makedirs(cache_dir)

    block_size = 20
    cache = Cache(input_dir=str(data_dir), chunk_size=40, item_loader=TokensLoader(block_size))

    counter = 0
    for i in range(100):
        text_ids = torch.arange(counter, counter + 20).to(torch.int)
        cache[i] = text_ids
        counter += 20

    cache.done()
    cache.merge()

    assert len([f for f in os.listdir(data_dir) if f.endswith(".bin")]) == 50

    dataset = EmulateS3StreamingDataset(
        input_dir=Dir(cache_dir, data_dir), item_loader=TokensLoader(block_size), shuffle=True
    )

    dataset.current_epoch = 1
    dataloader = StreamingDataLoader(dataset, num_workers=2, batch_size=2, prefetch_factor=1)

    dataloader_iter = iter(dataloader)

    _ = next(dataloader_iter)
    state_dict_0 = dataloader.state_dict()

    assert state_dict_0["dataset"]["num_samples_yielded"] == 2
    assert state_dict_0["dataset"]["num_workers"] == 2
    assert state_dict_0["dataset"]["batch_size"] == 2

    _ = next(dataloader_iter)
    state_dict_1 = dataloader.state_dict()
    assert state_dict_1["dataset"]["num_samples_yielded"] == 4
    assert state_dict_1["dataset"]["num_workers"] == 2
    assert state_dict_1["dataset"]["batch_size"] == 2

    batch_2 = next(dataloader_iter)
    state_dict_2 = dataloader.state_dict()
    assert state_dict_2["dataset"]["num_samples_yielded"] == 6
    assert state_dict_2["dataset"]["num_workers"] == 2
    assert state_dict_2["dataset"]["batch_size"] == 2

    dataset = EmulateS3StreamingDataset(
        input_dir=Dir(cache_dir, data_dir),
        item_loader=TokensLoader(block_size),
        shuffle=True,
    )

    dataloader = StreamingDataLoader(dataset, num_workers=2, batch_size=2, prefetch_factor=1)
    dataloader.load_state_dict(state_dict_1)

    dataloader_iter = iter(dataloader)
    batch_0_restart = next(dataloader_iter)

    state_dict_2 = dataloader.state_dict()["dataset"]

    assert state_dict_2["num_samples_yielded"] == 6
    assert state_dict_2["num_workers"] == 2
    assert state_dict_2["batch_size"] == 2

    assert torch.equal(batch_2, batch_0_restart)

    assert len(os.listdir(cache_dir)) >= 5


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows and MacOs")
def test_resumable_dataset_two_workers_2_epochs(tmpdir):
    seed_everything(42)

    data_dir = os.path.join(tmpdir, "data")
    cache_dir = os.path.join(tmpdir, "cache_dir")

    os.makedirs(data_dir)
    os.makedirs(cache_dir)

    block_size = 20
    cache = Cache(input_dir=str(data_dir), chunk_size=40, item_loader=TokensLoader(block_size))

    counter = 0
    for i in range(100):
        text_ids = torch.arange(counter, counter + 20).to(torch.int)
        cache[i] = text_ids
        counter += 20

    cache.done()
    cache.merge()

    assert len([f for f in os.listdir(data_dir) if f.endswith(".bin")]) == 50

    dataset = EmulateS3StreamingDataset(
        input_dir=Dir(cache_dir, data_dir), item_loader=TokensLoader(block_size), shuffle=True
    )

    dataloader = StreamingDataLoader(dataset, num_workers=2, batch_size=2, prefetch_factor=1, persistent_workers=True)

    batches_epoch_1 = []
    for batch in dataloader:
        batches_epoch_1.append(batch)

    assert len(os.listdir(cache_dir)) == 51

    batches_epoch_2 = []
    for batch in dataloader:
        batches_epoch_2.append(batch)

    assert len(os.listdir(cache_dir)) == 51
    assert not all(torch.equal(b1, b2) for b1, b2 in zip(batches_epoch_1, batches_epoch_2))


def _simple_preprocess(_):
    for _ in range(10):
        yield torch.randint(0, 100, size=(10,), dtype=torch.int64)


def _get_simulated_s3_dataloader(cache_dir, data_dir, shuffle=False):
    dataset = EmulateS3StreamingDataset(
        input_dir=Dir(cache_dir, data_dir),
        item_loader=TokensLoader(block_size=10),
        shuffle=shuffle,
    )
    return StreamingDataLoader(dataset, batch_size=2, num_workers=2)


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows and MacOs")
@mock.patch.dict(os.environ, {}, clear=True)
@pytest.mark.timeout(60)
@pytest.mark.parametrize("shuffle", [True, False])
def test_dataset_resume_on_future_chunks(shuffle, tmpdir, monkeypatch):
    """This test is constructed to test resuming from a chunk past the first chunk, when subsequent chunks don't have
    the same size."""
    s3_cache_dir = str(tmpdir / "s3cache")
    optimize_cache_dir = str(tmpdir / "optimize_cache")
    data_dir = str(tmpdir / "optimized")
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", optimize_cache_dir)

    optimize(
        fn=_simple_preprocess,
        inputs=list(range(8)),
        output_dir=str(tmpdir / "optimized"),
        chunk_size=190,
        num_workers=4,
        num_uploaders=1,
    )
    sleep(5)  # wait for copier/remover threads to complete
    assert set(os.listdir(tmpdir / "optimized")) == {
        "chunk-0-0.bin",
        "chunk-0-1.bin",
        "chunk-1-0.bin",
        "chunk-1-1.bin",
        "chunk-2-0.bin",
        "chunk-2-1.bin",
        "chunk-3-0.bin",
        "chunk-3-1.bin",
        "index.json",
    }

    os.mkdir(s3_cache_dir)
    train_dataloader = _get_simulated_s3_dataloader(s3_cache_dir, data_dir, shuffle=shuffle)
    batches_to_fetch = 16
    batch_to_resume_from = None
    dataloader_state = None

    for i, batch in enumerate(train_dataloader):
        if i == batches_to_fetch:
            dataloader_state = train_dataloader.state_dict()
        if i == batches_to_fetch + 1:
            batch_to_resume_from = batch
            break

    shutil.rmtree(s3_cache_dir)
    os.mkdir(s3_cache_dir)
    train_dataloader = _get_simulated_s3_dataloader(s3_cache_dir, data_dir, shuffle=shuffle)
    assert dataloader_state is not None
    assert batch_to_resume_from is not None
    train_dataloader.load_state_dict(dataloader_state)
    # The next batch after resuming must match what we should have gotten next in the initial loop
    assert torch.equal(next(iter(train_dataloader)), batch_to_resume_from)


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows and MacOs")
def test_dataset_valid_state(tmpdir, monkeypatch):
    seed_everything(42)

    data_dir = os.path.join(tmpdir, "data")
    cache_dir = os.path.join(tmpdir, "cache_dir")

    os.makedirs(data_dir)
    os.makedirs(cache_dir)

    block_size = 20
    cache = Cache(input_dir=str(data_dir), chunk_size=40, item_loader=TokensLoader(block_size))

    counter = 0
    for i in range(100):
        text_ids = torch.arange(counter, counter + 20).to(torch.int)
        cache[i] = text_ids
        counter += 20

    cache.done()
    cache.merge()

    dataset = EmulateS3StreamingDataset(
        input_dir=Dir(cache_dir, data_dir),
        item_loader=TokensLoader(block_size),
        shuffle=False,
        drop_last=False,
    )
    dataloader = DataLoader(dataset, num_workers=1, batch_size=2)
    dataloader_iter = iter(dataloader)
    next(dataloader_iter)

    sleep(1)

    state_dict = dataset.state_dict(0, 1, 2)

    dataset.load_state_dict(state_dict)
    dataset.worker_env = _WorkerEnv(world_size=1, rank=0)
    dataset.cache = cache

    dataset._validate_state_dict()

    state_dict["drop_last"] = True
    dataset.load_state_dict(state_dict)
    with pytest.raises(
        ValueError,
        match="The provided `drop_last` state doesn't match the current one. Found `False` instead of `True`.",
    ):
        dataset._validate_state_dict()

    state_dict["item_loader"] = {}
    dataset.load_state_dict(state_dict)
    with pytest.raises(
        ValueError,
        match="The provided `item_loader` state doesn't match the current one."
        " Found `{'block_size': 20}` instead of `{}`.",
    ):
        dataset._validate_state_dict()

    state_dict["seed"] = 12
    dataset.load_state_dict(state_dict)
    with pytest.raises(
        ValueError,
        match="The provided `seed` state doesn't match the current one. Found `42` instead of `12`.",
    ):
        dataset._validate_state_dict()

    state_dict["input_dir_url"] = "toto"
    dataset.load_state_dict(state_dict)
    with pytest.raises(
        ValueError,
        match=f"The provided `input_dir` URL state doesn't match the current one."
        f" Found `{data_dir}` instead of `toto`.",
    ):
        dataset._validate_state_dict()

    state_dict["input_dir_path"] = "toto"
    dataset.load_state_dict(state_dict)
    with pytest.raises(
        ValueError,
        match=f"The provided `input_dir` path state doesn't match the current one."
        f" Found `{cache_dir}` instead of `toto`.",
    ):
        dataset._validate_state_dict()

    state_dict["input_dir_path"] = "/teamspace/datasets/coco"
    dataset.load_state_dict(state_dict)
    with pytest.raises(
        ValueError,
        match=f"The provided `input_dir` path state doesn't match the current one. Found `{cache_dir}` instead of ",
    ):
        dataset._validate_state_dict()

    state_dict["num_workers"] = "8"
    dataset.load_state_dict(state_dict)
    with pytest.raises(
        ValueError,
        match="The provided `num_workers` state doesn't match the current one. Found `1` instead of `8`.",
    ):
        dataset._validate_state_dict()

    state_dict["shuffle"] = True
    dataset.load_state_dict(state_dict)
    with pytest.raises(
        ValueError,
        match="The provided `shuffle` state doesn't match the current one. Found `False` instead of `True`.",
    ):
        dataset._validate_state_dict()


def test_replay_sampling():
    assert _replay_sampling(27, 8, 2) == {0: 16, 1: 11}  # {0: 8 + 8, 1: 8 + 3}
    assert _replay_sampling(27, 7, 2) == {0: 14, 1: 13}  # {0: 7 + 7, 1: 7 + 6}
    assert _replay_sampling(27, 6, 2) == {0: 15, 1: 12}  # {0: 6 + 6 + 3, 1: 6 + 6}
    assert _replay_sampling(27, 5, 2) == {0: 15, 1: 12}  # {0: 5 + 5 + 5, 1: 5 + 5 + 2}
    assert _replay_sampling(27, 4, 2) == {0: 15, 1: 12}  # {0: 4 + 4 + 4 + 3, 1: 4 + 4 + 4}
    assert _replay_sampling(27, 8, 3) == {0: 11, 1: 8, 2: 8}  # {0: 8 + 3, 1: 8, 2: 8}
    assert _replay_sampling(27, 4, 3) == {0: 11, 1: 8, 2: 8}  # {0: 4 + 4 + 3, 1: 4 + 4, 2: 4 + 4}


def test_replay_chunks_sampling():
    chunks_replica = range(10)
    intervals_replica = [(i, i, i + 5, i + 5) for i in range(0, 50, 5)]
    workers_chunks, workers_intervals = _associate_chunks_and_internals_to_workers(
        _DistributedEnv(2, 0, 1), chunks_replica, intervals_replica
    )
    assert workers_chunks == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert workers_intervals == [
        [[0, 0, 5, 5], [5, 5, 10, 10], [10, 10, 15, 15], [15, 15, 20, 20], [20, 20, 25, 25]],
        [[25, 25, 30, 30], [30, 30, 35, 35], [35, 35, 40, 40], [40, 40, 45, 45], [45, 45, 50, 50]],
    ]
    workers_intervals = {i: workers_intervals[i] for i in range(len(workers_intervals))}
    assert _replay_chunks_sampling(workers_intervals, {0: 16, 1: 11}) == ({0: 3, 1: 2}, {0: 1, 1: 1})
    assert _replay_chunks_sampling(workers_intervals, {0: 14, 1: 13}) == ({0: 2, 1: 2}, {0: 4, 1: 3})
    assert _replay_chunks_sampling(workers_intervals, {0: 15, 1: 12}) == ({0: 3, 1: 2}, {0: 0, 1: 2})

    # Test that replay stops at the right chunk
    workers_intervals = {0: [(0, 0, 10, 10), (10, 10, 20, 20), (20, 20, 21, 21), (21, 21, 30, 30)]}
    indexes = {0: 15}
    # Replay should stop at chunk index 1, because 15 - 10 = 5, which fits into with chunk idx 1
    chunk_indexes, indexes = _replay_chunks_sampling(workers_intervals, indexes)
    assert chunk_indexes == {0: 1}
    assert indexes == {0: 5}


@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        pytest.param("zstd", marks=pytest.mark.skipif(condition=not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")),
    ],
)
def test_dataset_distributed_drop_last(tmpdir, monkeypatch, compression):
    class _DistributedEnvMock:
        def detect(cls):
            return _DistributedEnv(2, 0, 1)

    logger_mock = mock.MagicMock()

    monkeypatch.setattr(dataset_module, "_DistributedEnv", _DistributedEnvMock())
    monkeypatch.setattr(dataset_module, "logger", logger_mock)

    cache = Cache(str(tmpdir), chunk_size=10, compression=compression)
    for i in range(60):
        cache[i] = i
    cache.done()
    cache.merge()

    dataset = StreamingDataset(str(tmpdir), drop_last=None)
    assert dataset.drop_last

    dataset = StreamingDataset(str(tmpdir), drop_last=False)
    assert not dataset.drop_last

    warn_msg = logger_mock.warn._mock_mock_calls[0].args[0]
    expected_warn_msg = (
        "You're operating within a distributed environment and have disabled the `drop_last` option."
        " Please note that this configuration may lead to training interruptions"
        " if your system depends on distributed collectives."
    )
    assert expected_warn_msg == warn_msg


def test_subsample_streaming_dataset_with_token_loader(tmpdir, monkeypatch):
    monkeypatch.setattr(functions, "_get_input_dir", lambda x: str(tmpdir))

    seed_everything(42)

    with open(tmpdir / "a.txt", "w") as f:
        f.write("hello")

    inputs = [(v, str(tmpdir / "a.txt")) for v in range(0, 200, 20)]

    cache_dir = os.path.join(tmpdir, "cache")
    output_dir = os.path.join(tmpdir, "target_dir")
    os.makedirs(output_dir, exist_ok=True)
    monkeypatch.setenv("DATA_OPTIMIZER_CACHE_FOLDER", cache_dir)
    monkeypatch.setenv("DATA_OPTIMIZER_DATA_CACHE_FOLDER", cache_dir)

    functions.optimize(
        optimize_fn, inputs, output_dir=str(tmpdir), num_workers=2, chunk_size=2, reorder_files=False, num_downloaders=1
    )

    assert len([f for f in os.listdir(tmpdir) if f.endswith(".bin")]) == 10

    block_size = 10
    dataset1 = StreamingDataset(input_dir=str(tmpdir), item_loader=TokensLoader(block_size), shuffle=False)
    dataset2 = StreamingDataset(
        input_dir=str(tmpdir), item_loader=TokensLoader(block_size), shuffle=False, subsample=0.4
    )

    assert len(dataset2) == int(len(dataset1) * 0.4)


@pytest.mark.skipif(sys.platform == "win32", reason="Not tested on windows")
def test_dataset_with_mosaic_mds_data(tmpdir):
    from PIL import Image
    from streaming import MDSWriter
    # example taken from: https://github.com/mosaicml/streaming

    # A dictionary mapping input fields to their data types
    columns = {"image": "jpeg", "class": "int"}
    # Shard compression, if any
    compression = "zstd"
    # Save the samples as shards using MDSWriter
    with MDSWriter(out=str(tmpdir), columns=columns, compression=compression) as out:
        for i in range(10):
            sample = {
                "image": Image.fromarray(np.random.randint(0, 256, (32, 32, 3), np.uint8)),
                "class": i,
            }
            out.write(sample)
    dataset = StreamingDataset(input_dir=str(tmpdir))
    assert len(dataset) == 10
    for i in range(10):
        sample = dataset[i]
        assert sample["class"] == i

    assert [sample["class"] for sample in dataset[:]] == list(range(10))  # test slicing

    # -------------- train_test_split --------------

    train_ds, test_ds, val_ds = train_test_split(dataset, splits=[0.7, 0.2, 0.1])

    assert len(train_ds) == 7
    assert len(test_ds) == 2
    assert len(val_ds) == 1

    # -------------- subsample --------------

    dataset = StreamingDataset(input_dir=str(tmpdir), subsample=0.4)
    assert len(dataset) == 4
    assert [sample["class"] for sample in dataset[:]] == [0, 1, 2, 3]

    # -------------- works with dataloader --------------

    dataset = StreamingDataset(input_dir=str(tmpdir))
    dataloader = DataLoader(dataset, batch_size=4, drop_last=True)
    i = 0
    for batch in dataloader:
        assert len(batch["class"]) == 4
        assert len(batch["image"]) == 4
        assert list(batch["class"]) == [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
        i += 1

    dataloader = DataLoader(dataset, batch_size=4, drop_last=False)
    i = 0
    for batch in dataloader:
        if i == 2:
            # last batch is smaller than batch_size
            assert len(batch["class"]) == 2
            assert len(batch["image"]) == 2
            assert list(batch["class"]) == [4 * i, 4 * i + 1]
            break
        assert len(batch["class"]) == 4
        assert len(batch["image"]) == 4
        assert list(batch["class"]) == [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
        i += 1
