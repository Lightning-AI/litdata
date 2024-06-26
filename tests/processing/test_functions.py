import os
import sys
from unittest import mock

import pytest
from litdata.processing.functions import _get_input_dir, _resolve_dir
from litdata import walk, optimize, StreamingDataset, StreamingDataLoader


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


def test_optimize(tmpdir):
    output_dir = str(tmpdir / "output_dir")

    
    def compress(index):
        return index, index**2
    
    def different_compress(index):
        return index, index**2, index**3
    
    optimize(
        fn=compress,
        inputs=list(range(100)),
        num_workers=1,
        output_dir=output_dir,
        chunk_bytes="64MB",
    )

    ds = StreamingDataset(output_dir)

    assert len(ds) == 100
    assert ds[:] == [(i, i**2) for i in range(100)]

    with pytest.raises(RuntimeError, match="HINT: If you want to append/overwrite to the existing dataset"):
        optimize(
            fn=compress,
            inputs=list(range(100,200)),
            num_workers=1,
            output_dir=output_dir,
            chunk_bytes="64MB",
        )
    
    with pytest.raises(ValueError, match="The provided `mode` should be either `append` or `overwrite`"):
        optimize(
            fn=compress,
            inputs=list(range(100,200)),
            num_workers=1,
            output_dir=output_dir,
            chunk_bytes="64MB",
            mode = "some-random-mode"
        )
    
    
    optimize(
            fn=compress,
            inputs=list(range(100,200)),
            num_workers=3,
            output_dir=output_dir,
            chunk_bytes="64MB",
            mode="overwrite"
    )

    ds = StreamingDataset(output_dir)

    assert len(ds) == 100
    assert ds[:] == [(i, i**2) for i in range(100, 200)]

    optimize(
            fn=compress,
            inputs=list(range(200,300)),
            num_workers=os.cpu_count(),
            output_dir=output_dir,
            chunk_bytes="64MB",
            mode="append"
    )

    ds = StreamingDataset(output_dir)

    assert len(ds) == 200
    assert ds[:] == [(i, i**2) for i in range(100, 300)]

    optimize(
            fn=compress,
            inputs=list(range(300,400)),
            num_workers=2,
            output_dir=output_dir,
            chunk_bytes="64MB",
            mode="append"
    )

    ds = StreamingDataset(output_dir)

    assert len(ds) == 300
    assert ds[:] == [(i, i**2) for i in range(100, 400)]

    with pytest.raises(Exception, match="The config isn't consistent between chunks"):
        optimize(
            fn=different_compress,
            inputs=list(range(100,200)),
            num_workers=1,
            output_dir=output_dir,
            chunk_bytes="64MB",
            mode = "append"
        )
    
    ds = StreamingDataset(output_dir)

    assert len(ds) == 300
    assert ds[:] == [(i, i**2) for i in range(100, 400)]

    optimize(
            fn=different_compress,
            inputs=list(range(800,900)),
            num_workers=1,
            output_dir=output_dir,
            chunk_bytes="64MB",
            mode = "overwrite"
        )
    
    ds = StreamingDataset(output_dir)

    assert len(ds) == 100
    assert ds[:] == [(i, i**2, i**3) for i in range(800, 900)]
