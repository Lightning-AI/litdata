import json
import os
from unittest.mock import MagicMock

import pytest
from litdata.processing import utilities as utilities_module
from litdata.processing.utilities import (
    append_index_json,
    delete_files_with_extension,
    move_files_between_dirs,
    optimize_dns_context,
    optimize_mode_utility,
    overwrite_index_json,
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


def test_append_index_json(tmpdir):
    temp_index = {"chunks": [{"chunk_size": 1, "filename": "chunk-0-0-abc.bin"}], "config": "config"}
    output_index = {"chunks": [{"chunk_size": 1, "filename": "chunk-0-0-def.bin"}], "config": "config"}

    final_index = append_index_json(temp_index, output_index)

    assert final_index == {
        "chunks": [
            {"chunk_size": 1, "filename": "chunk-0-0-def.bin"},  # output_index first
            {"chunk_size": 1, "filename": "chunk-0-0-abc.bin"},
        ],
        "config": "config",
    }

    # test failing case
    output_index = {"chunks": [{"chunk_size": 1, "filename": "chunk-0-0-def.bin"}], "config": "differnt-config"}
    with pytest.raises(ValueError, match="The config of the optimized dataset is different from the original one."):
        append_index_json(temp_index, output_index)


def test_overwrite_index_json(tmpdir):
    temp_index = {"chunks": [{"chunk_size": 1, "filename": "chunk-0-0-abc.bin"}], "config": "config"}
    output_index = {"chunks": [{"chunk_size": 2, "filename": "chunk-0-0-def.bin"}], "config": "config"}

    final_index = overwrite_index_json(temp_index, output_index)

    assert final_index == {"chunks": [{"chunk_size": 1, "filename": "chunk-0-0-abc.bin"}], "config": "config"}

    # test failing case
    output_index = {"chunks": [{"chunk_size": 1, "filename": "chunk-0-0-def.bin"}], "config": "differnt-config"}
    with pytest.raises(ValueError, match="The config of the optimized dataset is different from the original one."):
        overwrite_index_json(temp_index, output_index)


def test_optimize_mode_utility_append(tmpdir):
    output_dir = tmpdir.mkdir("output")
    temp_dir = tmpdir.mkdir("temp")

    filepath = os.path.join(output_dir, "index.json")
    with open(filepath, "w") as f:
        json.dump({"chunks": [{"chunk_size": 1, "filename": "chunk-0-0-abc.bin"}], "config": "config"}, f)

    filepath = os.path.join(temp_dir, "index.json")
    with open(filepath, "w") as f:
        json.dump({"chunks": [{"chunk_size": 1, "filename": "chunk-0-0-def.bin"}], "config": "config"}, f)

    optimize_mode_utility(temp_dir, output_dir, "append")

    assert not os.path.exists(temp_dir)

    with open(os.path.join(output_dir, "index.json")) as f:
        index = json.load(f)

    assert index == {
        "chunks": [
            {"chunk_size": 1, "filename": "chunk-0-0-abc.bin"},
            {"chunk_size": 1, "filename": "chunk-0-0-def.bin"},
        ],
        "config": "config",
    }

    # test failing case
    temp_dir = tmpdir.mkdir("temp")
    filepath = os.path.join(temp_dir, "index.json")
    with open(filepath, "w") as f:
        json.dump({"chunks": [{"chunk_size": 1, "filename": "chunk-0-0-def.bin"}], "config": "different-config"}, f)
    with pytest.raises(ValueError, match="The config of the optimized dataset is different from the original one."):
        optimize_mode_utility(temp_dir, output_dir, "append")


def test_optimize_mode_utility_overwrite(tmpdir):
    output_dir = tmpdir.mkdir("output")
    temp_dir = tmpdir.mkdir("temp")

    filepath = os.path.join(output_dir, "index.json")
    with open(filepath, "w") as f:
        json.dump({"chunks": [{"chunk_size": 1, "filename": "chunk-0-0-abc.bin"}], "config": "config"}, f)

    filepath = os.path.join(temp_dir, "index.json")
    with open(filepath, "w") as f:
        json.dump({"chunks": [{"chunk_size": 1, "filename": "chunk-0-0-def.bin"}], "config": "config"}, f)

    optimize_mode_utility(temp_dir, output_dir, "overwrite")

    assert not os.path.exists(temp_dir)

    with open(os.path.join(output_dir, "index.json")) as f:
        index = json.load(f)

    assert index == {
        "chunks": [
            {"chunk_size": 1, "filename": "chunk-0-0-def.bin"},
        ],
        "config": "config",
    }

    # test failing case
    temp_dir = tmpdir.mkdir("temp")
    filepath = os.path.join(temp_dir, "index.json")
    with open(filepath, "w") as f:
        json.dump({"chunks": [{"chunk_size": 1, "filename": "chunk-0-0-def.bin"}], "config": "different-config"}, f)

    with pytest.raises(ValueError, match="The config of the optimized dataset is different from the original one."):
        optimize_mode_utility(temp_dir, output_dir, "overwrite")


def test_move_files_between_dirs(tmpdir):
    temp_dir = tmpdir.mkdir("temp")
    output_dir = tmpdir.mkdir("output")
    filepath = os.path.join(temp_dir, "a.txt")
    with open(filepath, "w") as f:
        f.write("HERE")

    move_files_between_dirs(temp_dir, output_dir, "a.txt")

    assert os.path.exists(os.path.join(output_dir, "a.txt"))
    assert not os.path.exists(os.path.join(temp_dir, "a.txt"))


def test_delete_files_with_extension(tmpdir):
    # Create a temporary directory and files for testing
    test_dir = tmpdir.mkdir("test_dir")

    # Create files with .txt and .log extensions
    file1 = test_dir.join("file1.txt")
    file2 = test_dir.join("file2.txt")
    file3 = test_dir.join("file3.log")

    file1.write("content")
    file2.write("content")
    file3.write("content")

    # Check the files exist before deletion
    assert file1.check()
    assert file2.check()
    assert file3.check()

    # Call the function to delete .txt files
    delete_files_with_extension(str(test_dir), "txt")

    # Check the .txt files are deleted and .log file still exists
    assert not file1.check()
    assert not file2.check()
    assert file3.check()

    # Call the function to delete .log files
    delete_files_with_extension(str(test_dir), "log")

    # Check the .log file is deleted
    assert not file3.check()
