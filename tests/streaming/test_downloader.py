# ruff: noqa: S604
import os
from unittest import mock
from unittest.mock import MagicMock, patch

from litdata.streaming.downloader import (
    AzureDownloader,
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


@patch("os.system")
@patch("subprocess.Popen")
def test_s3_downloader_with_s5cmd_no_storage_options(popen_mock, system_mock, tmpdir):
    system_mock.return_value = 0  # Simulates s5cmd being available
    process_mock = MagicMock()
    popen_mock.return_value = process_mock

    # Initialize the S3Downloader without storage options
    downloader = S3Downloader("s3://random_bucket", str(tmpdir), [])

    # Action: Call the download_file method
    remote_filepath = "s3://random_bucket/sample_file.txt"
    local_filepath = os.path.join(tmpdir, "sample_file.txt")
    downloader.download_file(remote_filepath, local_filepath)

    # Assertion: Verify subprocess.Popen was called with correct arguments and no env variables
    popen_mock.assert_called_once_with(
        f"s5cmd cp {remote_filepath} {local_filepath}",
        shell=True,
        stdout=subprocess.PIPE,
        env=None,
    )
    process_mock.wait.assert_called_once()


@patch("os.system")
@patch("subprocess.Popen")
def test_s3_downloader_with_s5cmd_with_storage_options(popen_mock, system_mock, tmpdir):
    system_mock.return_value = 0  # Simulates s5cmd being available
    process_mock = MagicMock()
    popen_mock.return_value = process_mock

    storage_options = {"AWS_ACCESS_KEY_ID": "dummy_key", "AWS_SECRET_ACCESS_KEY": "dummy_secret"}

    # Initialize the S3Downloader with storage options
    downloader = S3Downloader("s3://random_bucket", str(tmpdir), [], storage_options)

    # Action: Call the download_file method
    remote_filepath = "s3://random_bucket/sample_file.txt"
    local_filepath = os.path.join(tmpdir, "sample_file.txt")
    downloader.download_file(remote_filepath, local_filepath)

    # Create expected environment variables by merging the current env with storage_options
    expected_env = os.environ.copy()
    expected_env.update(storage_options)

    # Assertion: Verify subprocess.Popen was called with the correct arguments and environment variables
    popen_mock.assert_called_once_with(
        f"s5cmd cp {remote_filepath} {local_filepath}",
        shell=True,
        stdout=subprocess.PIPE,
        env=expected_env,
    )
    process_mock.wait.assert_called_once()


@mock.patch("litdata.streaming.downloader._GOOGLE_STORAGE_AVAILABLE", True)
def test_gcp_downloader(tmpdir, monkeypatch, google_mock):
    # Create mock objects
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.download_to_filename = MagicMock()

    # Patch the storage client to return the mock client
    google_mock.cloud.storage.Client = MagicMock(return_value=mock_client)

    # Configure the mock client to return the mock bucket and blob
    mock_client.bucket = MagicMock(return_value=mock_bucket)
    mock_bucket.blob = MagicMock(return_value=mock_blob)

    # Initialize the downloader
    storage_options = {"project": "DUMMY_PROJECT"}
    downloader = GCPDownloader("gs://random_bucket", tmpdir, [], storage_options)
    local_filepath = os.path.join(tmpdir, "a.txt")
    downloader.download_file("gs://random_bucket/a.txt", local_filepath)

    # Assert that the correct methods were called
    google_mock.cloud.storage.Client.assert_called_with(**storage_options)
    mock_client.bucket.assert_called_with("random_bucket")
    mock_bucket.blob.assert_called_with("a.txt")
    mock_blob.download_to_filename.assert_called_with(local_filepath)


@mock.patch("litdata.streaming.downloader._AZURE_STORAGE_AVAILABLE", True)
def test_azure_downloader(tmpdir, monkeypatch, azure_mock):
    mock_blob = MagicMock()
    mock_blob_data = MagicMock()
    mock_blob.download_blob.return_value = mock_blob_data
    service_mock = MagicMock()
    service_mock.get_blob_client.return_value = mock_blob

    azure_mock.storage.blob.BlobServiceClient = MagicMock(return_value=service_mock)

    # Initialize the downloader
    storage_options = {"project": "DUMMY_PROJECT"}
    downloader = AzureDownloader("azure://random_bucket", tmpdir, [], storage_options)
    local_filepath = os.path.join(tmpdir, "a.txt")
    downloader.download_file("azure://random_bucket/a.txt", local_filepath)

    # Assert that the correct methods were called
    azure_mock.storage.blob.BlobServiceClient.assert_called_with(**storage_options)
    service_mock.get_blob_client.assert_called_with(container="random_bucket", blob="a.txt")
    mock_blob.download_blob.assert_called()
    mock_blob_data.readinto.assert_called()


def test_download_with_cache(tmpdir, monkeypatch):
    # Create a file to download/cache
    with open("a.txt", "w") as f:
        f.write("hello")

    try:
        local_downloader = LocalDownloaderWithCache(tmpdir, tmpdir, [])
        shutil_mock = MagicMock()
        os_mock = MagicMock()
        monkeypatch.setattr(shutil, "copy", shutil_mock)
        monkeypatch.setattr(os, "rename", os_mock)

        local_downloader.download_file("local:a.txt", os.path.join(tmpdir, "a.txt"))
        shutil_mock.assert_called()
        os_mock.assert_called()
    finally:
        os.remove("a.txt")
