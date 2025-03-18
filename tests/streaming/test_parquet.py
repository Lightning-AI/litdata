import hashlib
import json
import os
import sys
from contextlib import nullcontext
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from litdata.constants import _INDEX_FILENAME
from litdata.streaming.dataset import StreamingDataset
from litdata.streaming.item_loader import ParquetLoader, PyTreeLoader
from litdata.streaming.writer import index_parquet_dataset
from litdata.utilities.hf_dataset import index_hf_dataset
from litdata.utilities.parquet import (
    CloudParquetDir,
    HFParquetDir,
    LocalParquetDir,
    default_cache_dir,
    get_parquet_indexer_cls,
)


#! TODO: Fix test failing on windows
# @pytest.mark.skipif(condition=sys.platform == "win32", reason="Fails on windows and test gets cancelled")
@pytest.mark.usefixtures("clean_pq_index_cache")
@pytest.mark.parametrize(
    ("pq_dir_url"),
    [
        None,
        "s3://some_bucket/some_path",
        "gs://some_bucket/some_path",
        "hf://datasets/some_org/some_repo/some_path",
    ],
)
@pytest.mark.parametrize(("num_worker"), [None, 1, 2, 4])
def test_parquet_index_write(
    monkeypatch, tmp_path, pq_data, huggingface_hub_fs_mock, fsspec_pq_mock, pq_dir_url, num_worker
):
    monkeypatch.setattr("litdata.utilities.parquet._HF_HUB_AVAILABLE", True)
    monkeypatch.setattr("litdata.utilities.parquet._FSSPEC_AVAILABLE", True)

    if pq_dir_url is None:
        pq_dir_url = os.path.join(tmp_path, "pq-dataset")

    cache_dir = default_cache_dir(pq_dir_url)

    index_file_path = os.path.join(tmp_path, "pq-dataset", _INDEX_FILENAME)
    if pq_dir_url.startswith("hf://"):
        index_file_path = os.path.join(cache_dir, _INDEX_FILENAME)

    assert not os.path.exists(index_file_path)

    # call the write_parquet_index fn
    if num_worker is None:
        index_parquet_dataset(pq_dir_url=pq_dir_url)
    else:
        index_parquet_dataset(pq_dir_url=pq_dir_url, num_workers=num_worker)

    assert os.path.exists(index_file_path)

    if pq_dir_url.startswith("hf://"):
        assert len(os.listdir(cache_dir)) == 1
    elif pq_dir_url.startswith(("gs://", "s3://")):
        assert len(os.listdir(cache_dir)) == 0

    # Read JSON file into a dictionary
    with open(index_file_path) as f:
        data = json.load(f)
        assert len(data["chunks"]) == 5
        for cnk in data["chunks"]:
            assert cnk["chunk_size"] == 5
        assert data["config"]["item_loader"] == "ParquetLoader"

    # no test for streaming on s3 and gs
    if pq_dir_url is None or pq_dir_url.startswith("hf://"):
        ds = StreamingDataset(pq_dir_url)

        assert len(ds) == 25  # 5 datasets for 5 loops

        for i, _ds in enumerate(ds):
            idx = i % 5
            assert len(_ds) == 3
            assert _ds[0] == pq_data["name"][idx]
            assert _ds[1] == pq_data["weight"][idx]
            assert _ds[2] == pq_data["height"][idx]


@pytest.mark.usefixtures("clean_pq_index_cache")
@patch("litdata.utilities.parquet._HF_HUB_AVAILABLE", False)
def test_index_hf_dataset(monkeypatch, tmp_path, huggingface_hub_fs_mock):
    monkeypatch.setattr("litdata.utilities.parquet._HF_HUB_AVAILABLE", True)

    with pytest.raises(ValueError, match="Invalid Hugging Face dataset URL"):
        index_hf_dataset("invalid_url")

    hf_url = "hf://datasets/some_org/some_repo/some_path"
    cache_dir = index_hf_dataset(hf_url)
    assert os.path.exists(cache_dir)
    assert len(os.listdir(cache_dir)) == 1
    assert os.path.exists(os.path.join(cache_dir, _INDEX_FILENAME))


def test_default_cache_dir(monkeypatch):
    os = ModuleType("os")
    os.path = Mock()
    monkeypatch.setattr("litdata.utilities.parquet.os", os)
    os.path.expanduser = Mock(return_value="/tmp/mock_path")  # noqa: S108

    def join_all_args(*args):
        # concatenate all paths with '/'
        assert all(isinstance(arg, str) for arg in args), "All arguments must be strings"
        is_root_route = args[0].startswith("/")
        joined_path = "/".join(arg.strip("/") for arg in args)
        if is_root_route:
            joined_path = "/" + joined_path
        return joined_path

    os.path.join = Mock(side_effect=join_all_args)
    os.makedirs = Mock()

    url = "pq://random_path/random_endpoint"
    url_hash = hashlib.sha256(url.encode()).hexdigest()

    cache_dir = default_cache_dir(url)

    assert os.path.expanduser.assert_called_once
    assert os.makedirs.assert_called_once

    expected_default_cache_dir = "/tmp/mock_path" + "/.cache" + "/litdata-cache-index-pq" + "/" + url_hash  # noqa: S108

    assert expected_default_cache_dir == cache_dir


#! TODO: Fix test failing on windows
@pytest.mark.skipif(condition=sys.platform == "win32", reason="Fails on windows bcoz of urllib.parse")
@pytest.mark.parametrize(
    ("pq_url", "cls", "expectation"),
    [
        ("s3://some_bucket/somepath", CloudParquetDir, nullcontext()),
        ("gs://some_bucket/somepath", CloudParquetDir, nullcontext()),
        ("hf://some_bucket/somepath", HFParquetDir, nullcontext()),
        ("local://some_bucket/somepath", LocalParquetDir, nullcontext()),
        ("/home/some_user/some_bucket/somepath", LocalParquetDir, nullcontext()),
        ("meow://some_bucket/somepath", None, pytest.raises(ValueError, match="The provided")),
    ],
)
def test_get_parquet_indexer_cls(pq_url, cls, expectation, monkeypatch, fsspec_mock, huggingface_hub_fs_mock):
    os = Mock()
    os.listdir = Mock(return_value=[])

    fsspec_fs_mock = Mock()
    fsspec_fs_mock.ls = Mock(return_value=[])
    fsspec_mock.filesystem = Mock(return_value=fsspec_fs_mock)

    hf_fs_mock = Mock()
    hf_fs_mock.ls = Mock(return_value=[])
    huggingface_hub_fs_mock.HfFileSystem = Mock(return_value=hf_fs_mock)

    monkeypatch.setattr("litdata.utilities.parquet.os", os)
    monkeypatch.setattr("litdata.utilities.parquet._HF_HUB_AVAILABLE", True)

    with expectation:
        indexer_obj = get_parquet_indexer_cls(pq_url)
        assert isinstance(indexer_obj, cls)


@pytest.mark.usefixtures("clean_pq_index_cache")
def test_stream_hf_parquet_dataset(huggingface_hub_fs_mock, pq_data):
    hf_url = "hf://datasets/some_org/some_repo/some_path"

    # Test case 1: Invalid item_loader
    with pytest.raises(ValueError, match="Invalid item_loader for hf://datasets."):
        StreamingDataset(hf_url, item_loader=PyTreeLoader)

    # Test case 2: Streaming without passing item_loader
    ds = StreamingDataset(hf_url)
    assert len(ds) == 25  # 5 datasets for 5 loops
    for i, _ds in enumerate(ds):
        idx = i % 5
        assert len(_ds) == 3
        assert _ds[0] == pq_data["name"][idx]
        assert _ds[1] == pq_data["weight"][idx]
        assert _ds[2] == pq_data["height"][idx]

    # Test case 3: Streaming with ParquetLoader as item_loader
    ds = StreamingDataset(hf_url, item_loader=ParquetLoader())
    assert len(ds) == 25
    for i, _ds in enumerate(ds):
        idx = i % 5
        assert len(_ds) == 3
        assert _ds[0] == pq_data["name"][idx]
        assert _ds[1] == pq_data["weight"][idx]
        assert _ds[2] == pq_data["height"][idx]

    # Test case 4: Streaming with ParquetLoader and low_memory=True
    ds = StreamingDataset(hf_url, item_loader=ParquetLoader(low_memory=True))
    assert len(ds) == 25
    for i, _ds in enumerate(ds):
        idx = i % 5
        assert len(_ds) == 3
        assert _ds[0] == pq_data["name"][idx]
        assert _ds[1] == pq_data["weight"][idx]
        assert _ds[2] == pq_data["height"][idx]

    # Test case 5: Streaming with ParquetLoader and low_memory=True and shuffle=True
    with pytest.raises(ValueError, match="You have enabled shuffling when using low memory with ParquetLoader."):
        StreamingDataset(hf_url, item_loader=ParquetLoader(low_memory=True), shuffle=True)
