from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from litdata.constants import _NUMPY_DTYPES_MAPPING, _TORCH_DTYPES_MAPPING
from litdata.streaming import Cache, item_loader
from litdata.streaming.dataset import StreamingDataset
from litdata.streaming.item_loader import PyTreeLoader, TokensLoader


def test_serializer_setup():
    config_mock = MagicMock()
    config_mock.__getitem__.return_value = ["fake:12"]
    serializer_mock = MagicMock()
    item_loader = PyTreeLoader()
    item_loader.setup(config_mock, [], {"fake": serializer_mock})
    assert len(item_loader._serializers) == 2
    assert item_loader._serializers["fake:12"]


def test_pytreeloader_with_no_header_tensor_serializer(tmpdir):
    cache = Cache(str(tmpdir), chunk_size=10)
    assert isinstance(cache._reader._item_loader, PyTreeLoader)
    dtype_index_float = 1
    dtype_index_long = 18
    for i in range(10):
        cache[i] = {
            "float": i * torch.ones(10).to(_TORCH_DTYPES_MAPPING[dtype_index_float]),
            "long": i * torch.ones(10).to(_TORCH_DTYPES_MAPPING[dtype_index_long]),
        }

    data_format = [f"no_header_tensor:{dtype_index_float}", f"no_header_tensor:{dtype_index_long}"]
    assert cache._writer.get_config()["data_format"] == data_format
    cache.done()
    cache.merge()

    dataset = StreamingDataset(input_dir=str(tmpdir))
    for i in range(len(dataset)):
        item = dataset[i]
        assert torch.allclose(i * torch.ones(10).to(_TORCH_DTYPES_MAPPING[dtype_index_float]), item["float"])
        assert torch.allclose(i * torch.ones(10).to(_TORCH_DTYPES_MAPPING[dtype_index_long]), item["long"])


def test_tokensloader_with_no_header_numpy_serializer(tmpdir):
    cache = Cache(str(tmpdir), chunk_size=512, item_loader=TokensLoader())
    assert isinstance(cache._reader._item_loader, TokensLoader)

    dtype_index_int32 = 3
    dtype = _NUMPY_DTYPES_MAPPING[dtype_index_int32]

    for i in range(10):
        data = np.random.randint(0, 100, size=(256), dtype=dtype)
        cache._add_item(i, data)

    data_format = [f"no_header_numpy:{dtype_index_int32}"]
    assert cache._writer.get_config()["data_format"] == data_format
    cache.done()
    cache.merge()

    dataset = StreamingDataset(
        input_dir=str(tmpdir),
        drop_last=True,
        item_loader=TokensLoader(block_size=256),
    )

    for data in dataset:
        assert data.shape == (256,)
        assert data.dtype == dtype


class TestPyTreeLoader(PyTreeLoader):
    def force_download(self, chunk_index):
        assert chunk_index == 0
        super().force_download(chunk_index)
        raise Exception("worked")


def test_force_download(monkeypatch, tmpdir):
    monkeypatch.setattr(item_loader, "_FORCE_DOWNLOAD_TIME", 1)
    monkeypatch.setattr(item_loader, "_FORCE_DOWNLOAD_TIME", 1)
    loader = TestPyTreeLoader()

    config_mock = MagicMock()
    config_mock.__getitem__.return_value = ["fake:12"]
    serializer_mock = MagicMock()
    loader.setup(config_mock, [], {"fake": serializer_mock})

    with pytest.raises(Exception, match="worked"):
        loader.load_item_from_chunk(0, 0, "chunk_filepath", 0, 1)
