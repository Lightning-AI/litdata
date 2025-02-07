import hashlib
import json
import os
from contextlib import nullcontext
from types import ModuleType
from unittest.mock import Mock

import pytest

from litdata.streaming.dataset import StreamingDataset
from litdata.streaming.item_loader import ParquetLoader
from litdata.streaming.writer import index_parquet_dataset
from litdata.utilities.parquet import (
    CloudParquetDir,
    HFParquetDir,
    LocalParquetDir,
    default_cache_dir,
    get_parquet_indexer_cls,
)


def test_parquet_index_write(tmp_path, pq_data, write_pq_data):
    index_file_path = os.path.join(tmp_path, "pq-dataset", "index.json")
    assert not os.path.exists(index_file_path)
    # call the write_parquet_index fn
    index_parquet_dataset(os.path.join(tmp_path, "pq-dataset"))
    assert os.path.exists(index_file_path)

    # Read JSON file into a dictionary
    with open(index_file_path) as f:
        data = json.load(f)
        assert len(data["chunks"]) == 5
        for cnk in data["chunks"]:
            assert cnk["chunk_size"] == 5
        assert data["config"]["item_loader"] == "ParquetLoader"
        assert data["config"]["data_format"] == ["String", "Float64", "Float64"]

    ds = StreamingDataset(os.path.join(tmp_path, "pq-dataset"), item_loader=ParquetLoader())

    assert len(ds) == 25  # 5 datasets for 5 loops

    for i, _ds in enumerate(ds):
        idx = i % 5
        assert len(_ds) == 3
        assert _ds[0] == pq_data["name"][idx]
        assert _ds[1] == pq_data["weight"][idx]
        assert _ds[2] == pq_data["height"][idx]


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
def test_get_parquet_indexer_cls(pq_url, cls, expectation, monkeypatch, fsspec_mock, huggingface_hub_mock):
    os = Mock()
    os.listdir = Mock(return_value=[])

    fsspec_fs_mock = Mock()
    fsspec_fs_mock.ls = Mock(return_value=[])
    fsspec_mock.filesystem = Mock(return_value=fsspec_fs_mock)

    hf_fs_mock = Mock()
    hf_fs_mock.ls = Mock(return_value=[])
    huggingface_hub_mock.HfFileSystem = Mock(return_value=hf_fs_mock)

    monkeypatch.setattr("litdata.utilities.parquet.os", os)
    monkeypatch.setattr("litdata.utilities.parquet._HF_HUB_AVAILABLE", True)
    with expectation:
        indexer_obj = get_parquet_indexer_cls(pq_url)
        assert isinstance(indexer_obj, cls)
