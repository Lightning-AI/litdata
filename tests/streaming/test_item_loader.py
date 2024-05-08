from unittest.mock import MagicMock

import torch
from litdata.constants import _TORCH_DTYPES_MAPPING
from litdata.streaming import Cache
from litdata.streaming.dataset import StreamingDataset
from litdata.streaming.item_loader import PyTreeLoader


def test_serializer_setup():
    config_mock = MagicMock()
    config_mock.__getitem__.return_value = ["fake:12"]
    serializer_mock = MagicMock()
    item_loader = PyTreeLoader()
    item_loader.setup(config_mock, [], {"fake": serializer_mock})
    serializer_mock.setup._mock_mock_calls[0].args[0] == "fake:12"


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
