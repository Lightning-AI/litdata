import os
from unittest.mock import MagicMock

from litdata.streaming.downloader import (
    LocalDownloaderWithCache,
    shutil,
)


def test_download_with_cache(tmpdir, monkeypatch):
    # Create a file to download/cache
    with open("a.txt", "w") as f:
        f.write("hello")

    try:
        local_downloader = LocalDownloaderWithCache("file", tmpdir, tmpdir, [])
        shutil_mock = MagicMock()
        os_mock = MagicMock()
        monkeypatch.setattr(shutil, "copy", shutil_mock)
        monkeypatch.setattr(os, "rename", os_mock)

        local_downloader.download_file("local:a.txt", os.path.join(tmpdir, "a.txt"))
        shutil_mock.assert_called()
        os_mock.assert_called()
    finally:
        os.remove("a.txt")
