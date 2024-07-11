import os
import sys
from types import ModuleType
from unittest import mock
from unittest.mock import MagicMock

from litdata.streaming.downloader import (
    GCPDownloader,
    LocalDownloaderWithCache,
    S3Downloader,
    shutil,
    subprocess,
)


def test_s3_downloader_fast(tmpdir, monkeypatch):
    monkeypatch.setattr(os, "system", MagicMock(return_value=0))
    popen_mock = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", MagicMock(return_value=popen_mock))
    downloader = S3Downloader(tmpdir, tmpdir, [])
    downloader.download_file("s3://random_bucket/a.txt", os.path.join(tmpdir, "a.txt"))
    popen_mock.wait.assert_called()


@mock.patch("litdata.streaming.downloader._GOOGLE_STORAGE_AVAILABLE", True)
def test_gcp_downloader(tmpdir, monkeypatch):
    # Mocks for local imports
    google = ModuleType("google")
    monkeypatch.setitem(sys.modules, "google", google)
    google_cloud = ModuleType("cloud")
    monkeypatch.setitem(sys.modules, "google.cloud", google_cloud)
    google_cloud_storage = ModuleType("storage")
    monkeypatch.setitem(sys.modules, "google.cloud.storage", google_cloud_storage)
    google.cloud = google_cloud
    google.cloud.storage = google_cloud_storage

    # Create mock objects
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.download_to_filename = MagicMock()

    # Patch the storage client to return the mock client
    google_cloud_storage.Client = MagicMock(return_value=mock_client)

    # Configure the mock client to return the mock bucket and blob
    mock_client.bucket = MagicMock(return_value=mock_bucket)
    mock_bucket.blob = MagicMock(return_value=mock_blob)

    # Initialize the downloader
    downloader = GCPDownloader("gs://random_bucket", tmpdir, [])
    local_filepath = os.path.join(tmpdir, "a.txt")
    downloader.download_file("gs://random_bucket/a.txt", local_filepath)

    # Assert that the correct methods were called
    mock_client.bucket.assert_called_with("random_bucket")
    mock_bucket.blob.assert_called_with("a.txt")
    mock_blob.download_to_filename.assert_called_with(local_filepath)


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
