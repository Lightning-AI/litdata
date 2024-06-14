import os
import pytest
from unittest import mock
from litdata.utilities.dataset_utilities import (
    _should_replace_path,
    _try_create_cache_dir,
    generate_roi,
)

def test_should_replace_path():
    assert _should_replace_path(None)
    assert _should_replace_path("")
    assert not _should_replace_path(".../datasets/...")
    assert not _should_replace_path(".../s3__connections/...")
    assert _should_replace_path("/teamspace/datasets/...")
    assert _should_replace_path("/teamspace/s3_connections/...")
    assert not _should_replace_path("something_else")


def test_try_create_cache_dir():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert os.path.join("chunks", "100b8cad7cf2a56f6df78f171f97a1ec") in _try_create_cache_dir("any")

    # the cache dir creating at /cache requires root privileges, so we need to mock `os.makedirs()`
    with (
        mock.patch.dict("os.environ", {"LIGHTNING_CLUSTER_ID": "abc", "LIGHTNING_CLOUD_PROJECT_ID": "123"}),
        mock.patch("litdata.streaming.dataset.os.makedirs") as makedirs_mock,
    ):
        cache_dir_1 = _try_create_cache_dir("")
        cache_dir_2 = _try_create_cache_dir("ssdf")
        assert cache_dir_1 != cache_dir_2
        assert cache_dir_1 == os.path.join("/cache", "chunks", "d41d8cd98f00b204e9800998ecf8427e")
        assert len(makedirs_mock.mock_calls) == 2


def test_generate_roi():
    my_chunks = [{"chunk_size":30},{"chunk_size":50},{"chunk_size":20},{"chunk_size":10},]
    my_roi = generate_roi(my_chunks)

    assert my_roi == [(0,30), (0,50), (0,20), (0,10)]
