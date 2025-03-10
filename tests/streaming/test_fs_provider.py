import pytest

from litdata.streaming.fs_provider import GCPFsProvider, S3FsProvider, _get_fs_provider, get_bucket_and_path


def test_get_bucket_and_path():
    bucket, path = get_bucket_and_path("s3://bucket/path/to/file.txt")
    assert bucket == "bucket"
    assert path == "path/to/file.txt"

    bucket, path = get_bucket_and_path("s3://bucket/path/to/file.txt", "s3")
    assert bucket == "bucket"
    assert path == "path/to/file.txt"

    bucket, path = get_bucket_and_path("gs://bucket/path/to/file.txt", "gs")
    assert bucket == "bucket"
    assert path == "path/to/file.txt"


def test_get_fs_provider():
    fs_provider = _get_fs_provider("s3://bucket/path/to/file.txt")
    assert isinstance(fs_provider, S3FsProvider)

    fs_provider = _get_fs_provider("gs://bucket/path/to/file.txt")
    assert isinstance(fs_provider, GCPFsProvider)

    with pytest.raises(ValueError, match="Unsupported scheme"):
        _get_fs_provider("http://bucket/path/to/file.txt")
