import os
import sys
from typing import Tuple
from unittest import mock

import pytest
from litdata import walk
from litdata.processing.functions import _get_input_dir, _resolve_dir, optimize
from litdata.streaming.dataset import StreamingDataset


@pytest.mark.skipif(sys.platform == "win32", reason="currently not supported for windows.")
def test_get_input_dir(tmpdir, monkeypatch):
    monkeypatch.setattr(os.path, "exists", mock.MagicMock(return_value=True))
    assert _get_input_dir(["/teamspace/studios/here/a", "/teamspace/studios/here/b"]) == "/teamspace/studios/here"

    exists_res = [False, True]

    def fn(*_, **__):
        return exists_res.pop(0)

    monkeypatch.setattr(os.path, "exists", fn)

    with pytest.raises(ValueError, match="The provided item  didn't contain any filepaths."):
        assert _get_input_dir(["", "/teamspace/studios/asd/b"])


def test_walk(tmpdir):
    for i in range(5):
        folder_path = os.path.join(tmpdir, str(i))
        os.makedirs(folder_path, exist_ok=True)
        for j in range(5):
            filepath = os.path.join(folder_path, f"{j}.txt")
            with open(filepath, "w") as f:
                f.write("hello world !")

    walks_os = sorted(os.walk(tmpdir))
    walks_function = sorted(walk(tmpdir))
    assert walks_os == walks_function


def test_get_input_dir_with_s3_path():
    input_dir = _get_input_dir(["s3://my_bucket/my_folder/a.txt"])
    assert input_dir == "s3://my_bucket/my_folder"
    input_dir = _resolve_dir(input_dir)
    assert not input_dir.path
    assert input_dir.url == "s3://my_bucket/my_folder"


def test_optimize_function_modes(tmpdir):
    output_dir = tmpdir.mkdir("output")
    output_dir = str(output_dir)

    def compress(index: int) -> Tuple[int, int]:
        return (index, index**2)

    def different_compress(index: int) -> Tuple[int, int, int]:
        return (index, index**2, index**3)

    # none mode
    optimize(
        fn=compress,
        inputs=list(range(1, 101)),
        output_dir=output_dir,
        chunk_bytes="64MB",
    )

    my_dataset = StreamingDataset(output_dir)
    assert len(my_dataset) == 100
    assert my_dataset[:] == [(i, i**2) for i in range(1, 101)]

    # append mode
    optimize(
        fn=compress,
        mode="append",
        inputs=list(range(101, 201)),
        output_dir=output_dir,
        chunk_bytes="64MB",
    )

    my_dataset = StreamingDataset(output_dir)
    assert len(my_dataset) == 200
    assert my_dataset[:] == [(i, i**2) for i in range(1, 201)]

    # overwrite mode
    optimize(
        fn=compress,
        mode="overwrite",
        inputs=list(range(201, 351)),
        output_dir=output_dir,
        chunk_bytes="64MB",
    )

    my_dataset = StreamingDataset(output_dir)
    assert len(my_dataset) == 150
    assert my_dataset[:] == [(i, i**2) for i in range(201, 351)]

    # failing case
    with pytest.raises(ValueError, match="The config of the optimized dataset is different from the original one."):
        # overwrite mode
        optimize(
            fn=different_compress,
            mode="overwrite",
            inputs=list(range(201, 351)),
            output_dir=output_dir,
            chunk_bytes="64MB",
        )
