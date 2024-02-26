import os
from unittest.mock import MagicMock

from litdata.streaming.downloader import S3Downloader, LocalDownloaderWithCache, subprocess, shutil


def test_s3_downloader_fast(tmpdir, monkeypatch):
    monkeypatch.setattr(os, "system", MagicMock(return_value=0))
    popen_mock = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", MagicMock(return_value=popen_mock))
    downloader = S3Downloader(tmpdir, tmpdir, [])
    downloader.download_file("s3://random_bucket/a.txt", os.path.join(tmpdir, "a.txt"))
    popen_mock.wait.assert_called()

def test_download_with_cache(tmpdir, monkeypatch):
    # Create a file to download/cache
    with open("a.txt", "w") as f:
        f.write("hello")

    try:
        local_downloader = LocalDownloaderWithCache(tmpdir, tmpdir, [])
        shutil_mock = MagicMock()
        monkeypatch.setattr(shutil, "copy", shutil_mock)
        local_downloader.download_file("local:a.txt", os.path.join(tmpdir, "a.txt"))
        shutil_mock.assert_called()
    finally:
        os.remove("a.txt")
