from unittest.mock import MagicMock

from litdata.processing import utilities as utilities_module
from litdata.processing.utilities import (
    extract_rank_and_index_from_filename,
    optimize_dns_context,
)


def test_optimize_dns_context(monkeypatch):
    popen_mock = MagicMock()

    monkeypatch.setattr(utilities_module, "_IS_IN_STUDIO", True)
    monkeypatch.setattr(utilities_module, "Popen", popen_mock)

    class FakeFile:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args, **kwargs):
            return self

        def readlines(self):
            return ["127.0.0.53"]

    monkeypatch.setitem(__builtins__, "open", MagicMock(return_value=FakeFile()))

    with optimize_dns_context(True):
        pass

    cmd = popen_mock._mock_call_args_list[0].args[0]
    expected_cmd = (
        "sudo /home/zeus/miniconda3/envs/cloudspace/bin/python"
        " -c 'from litdata.processing.utilities import _optimize_dns; _optimize_dns(True)'"
    )
    assert cmd == expected_cmd


def test_extract_rank_and_index_from_filename():
    file_names = [
        "chunk-0-0.bin",
        "chunk-0-0.compressionAlgorithm.bin",
        "chunk-1-4.bin",
        "chunk-1-9.compressionAlgorithm.bin",
        "chunk-22-10.bin",
        "chunk-2-3.compressionAlgorithm.bin",
        "chunk-31-3.bin",
        "chunk-3-110.compressionAlgorithm.bin",
    ]

    rank_and_index = [
        (0, 0),
        (0, 0),
        (1, 4),
        (1, 9),
        (22, 10),
        (2, 3),
        (31, 3),
        (3, 110),
    ]

    for idx, file_name in enumerate(file_names):
        rank, index = extract_rank_and_index_from_filename(file_name)
        assert rank == rank_and_index[idx][0]
        assert index == rank_and_index[idx][1]


def test_read_index_file_content():
    # TODO: Add a test for reading index file content on local and remote
    ...
