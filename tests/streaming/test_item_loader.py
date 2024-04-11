from unittest.mock import MagicMock

from litdata.streaming.item_loader import PyTreeLoader


def test_serializer_setup():
    config_mock = MagicMock()
    config_mock.__getitem__.return_value = ["fake:12"]
    serializer_mock = MagicMock()
    item_loader = PyTreeLoader()
    item_loader.setup(config_mock, [], {"fake": serializer_mock})
    serializer_mock.setup._mock_mock_calls[0].args[0] == "fake:12"
