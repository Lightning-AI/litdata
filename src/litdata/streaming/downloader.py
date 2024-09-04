# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from abc import ABC
from typing import Any, Dict, List, Optional, Union

import fsspec
from filelock import FileLock, Timeout

from litdata.constants import _INDEX_FILENAME

# from litdata.streaming.client import S3Client


class Downloader(ABC):
    def __init__(
        self,
        cloud_provider: str,
        remote_dir: str,
        cache_dir: str,
        chunks: List[Dict[str, Any]],
        storage_options: Optional[Dict] = {},
    ):
        print("-" * 80)
        print(f"{cloud_provider=}")
        print("-" * 80)

        self._remote_dir = remote_dir
        self._cache_dir = cache_dir
        self._chunks = chunks
        self.fs = fsspec.filesystem(cloud_provider, **storage_options)

    def download_chunk_from_index(self, chunk_index: int) -> None:
        chunk_filename = self._chunks[chunk_index]["filename"]
        local_chunkpath = os.path.join(self._cache_dir, chunk_filename)
        remote_chunkpath = os.path.join(self._remote_dir, chunk_filename)
        self.download_file(remote_chunkpath, local_chunkpath)

    def download_file(self, remote_chunkpath: str, local_chunkpath: str) -> None:
        pass


# class S3Downloader(Downloader):
#     def __init__(
#         self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
#     ):
#         super().__init__("s3", remote_dir, cache_dir, chunks, storage_options)
#         self._s5cmd_available = os.system("s5cmd > /dev/null 2>&1") == 0

#         if not self._s5cmd_available:
#             self._client = S3Client(storage_options=self._storage_options)

#     def download_file(self, remote_filepath: str, local_filepath: str) -> None:
#         obj = parse.urlparse(remote_filepath)

#         if obj.scheme != "s3":
#             raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for {remote_filepath}")

#         if os.path.exists(local_filepath):
#             return

#         try:
#             with FileLock(local_filepath + ".lock", timeout=3 if obj.path.endswith(_INDEX_FILENAME) else 0):
#                 if self._s5cmd_available:
#                     proc = subprocess.Popen(
#                         f"s5cmd cp {remote_filepath} {local_filepath}",
#                         shell=True,
#                         stdout=subprocess.PIPE,
#                     )
#                     proc.wait()
#                 else:
#                     from boto3.s3.transfer import TransferConfig

#                     extra_args: Dict[str, Any] = {}

#                     # try:
#                     #     with FileLock(local_filepath + ".lock", timeout=1):
#                     if not os.path.exists(local_filepath):
#                         # Issue: https://github.com/boto/boto3/issues/3113
#                         self._client.client.download_file(
#                             obj.netloc,
#                             obj.path.lstrip("/"),
#                             local_filepath,
#                             ExtraArgs=extra_args,
#                             Config=TransferConfig(use_threads=False),
#                         )
#         except Timeout:
#             # another process is responsible to download that file, continue
#             pass


# class GCPDownloader(Downloader):
#     def __init__(
#         self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
#     ):
#         if not _GOOGLE_STORAGE_AVAILABLE:
#             raise ModuleNotFoundError(str(_GOOGLE_STORAGE_AVAILABLE))

#         super().__init__("gs", remote_dir, cache_dir, chunks, storage_options)

#     def download_file(self, remote_filepath: str, local_filepath: str) -> None:
#         from google.cloud import storage

#         obj = parse.urlparse(remote_filepath)

#         if obj.scheme != "gs":
#             raise ValueError(f"Expected obj.scheme to be `gs`, instead, got {obj.scheme} for {remote_filepath}")

#         if os.path.exists(local_filepath):
#             return

#         try:
#             with FileLock(local_filepath + ".lock", timeout=3 if obj.path.endswith(_INDEX_FILENAME) else 0):
#                 bucket_name = obj.netloc
#                 key = obj.path
#                 # Remove the leading "/":
#                 if key[0] == "/":
#                     key = key[1:]

#                 client = storage.Client(**self._storage_options)
#                 bucket = client.bucket(bucket_name)
#                 blob = bucket.blob(key)
#                 blob.download_to_filename(local_filepath)
#         except Timeout:
#             # another process is responsible to download that file, continue
#             pass


# class AzureDownloader(Downloader):
#     def __init__(
#         self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
#     ):
#         if not _AZURE_STORAGE_AVAILABLE:
#             raise ModuleNotFoundError(str(_AZURE_STORAGE_AVAILABLE))

#         super().__init__("abfs", remote_dir, cache_dir, chunks, storage_options)

#     def download_file(self, remote_filepath: str, local_filepath: str) -> None:
#         from azure.storage.blob import BlobServiceClient

#         obj = parse.urlparse(remote_filepath)

#         if obj.scheme != "azure":
#             raise ValueError(
#                 f"Expected obj.scheme to be `azure`, instead, got {obj.scheme} for {remote_filepath}"
#             )

#         if os.path.exists(local_filepath):
#             return

#         try:
#             with FileLock(local_filepath + ".lock", timeout=3 if obj.path.endswith(_INDEX_FILENAME) else 0):
#                 service = BlobServiceClient(**self._storage_options)
#                 blob_client = service.get_blob_client(container=obj.netloc, blob=obj.path.lstrip("/"))
#                 with open(local_filepath, "wb") as download_file:
#                     blob_data = blob_client.download_blob()
#                     blob_data.readinto(download_file)

#         except Timeout:
#             # another process is responsible to download that file, continue
#             pass


class LocalDownloader(Downloader):
    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        if not os.path.exists(remote_filepath):
            raise FileNotFoundError(f"The provided remote_path doesn't exist: {remote_filepath}")

        if remote_filepath != local_filepath and not os.path.exists(local_filepath):
            shutil.copy(remote_filepath, local_filepath)


class LocalDownloaderWithCache(LocalDownloader):
    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        remote_filepath = remote_filepath.replace("local:", "")
        super().download_file(remote_filepath, local_filepath)


# _DOWNLOADERS = {
#     "s3://": S3Downloader,
#     "gs://": GCPDownloader,
#     "azure://": AzureDownloader,
#     "local:": LocalDownloaderWithCache,
#     "": LocalDownloader,
# }


_DOWNLOADERS = {
    "s3://": "s3",
    "gs://": "gs",
    "azure://": "abfs",
    "local:": "file",
    "": "file",
}


class FsspecDownloader(Downloader):
    def __init__(
        self,
        cloud_provider: str,
        remote_dir: str,
        cache_dir: str,
        chunks: List[Dict[str, Any]],
        storage_options: Optional[Dict] = {},
    ):
        remote_dir = remote_dir.replace("local:", "")
        self.is_local = False
        super().__init__(cloud_provider, remote_dir, cache_dir, chunks, storage_options)

    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        if os.path.exists(local_filepath) or remote_filepath == local_filepath:
            return
        try:
            with FileLock(local_filepath + ".lock", timeout=3):
                self.fs.get(remote_filepath, local_filepath, recursive=True)
            # remove the lock file
            if os.path.exists(local_filepath + ".lock"):
                os.remove(local_filepath + ".lock")
        except Timeout:
            # another process is responsible to download that file, continue
            pass


def does_file_exist(
    remote_filepath: str, cloud_provider: Union[str, None] = None, storage_options: Optional[Dict] = {}
) -> bool:
    if cloud_provider is None:
        cloud_provider = get_cloud_provider(remote_filepath)

    fs = fsspec.filesystem(cloud_provider, **storage_options)
    return fs.exists(remote_filepath)


def list_directory(
    remote_directory: str,
    detail: bool = False,
    cloud_provider: Union[str, None] = None,
    storage_options: Optional[Dict] = {},
) -> List[str]:
    """Returns a list of filenames in a remote directory."""
    if cloud_provider is None:
        cloud_provider = get_cloud_provider(remote_directory)

    fs = fsspec.filesystem(cloud_provider, **storage_options)
    return fs.ls(remote_directory, detail=detail)  # just return the filenames


def download_file_or_directory(remote_filepath: str, local_filepath: str, storage_options: Optional[Dict] = {}) -> None:
    """Download a file from the remote cloud storage."""
    try:
        with FileLock(local_filepath + ".lock", timeout=3):
            fs_cloud_provider = get_cloud_provider(remote_filepath)
            fs = fsspec.filesystem(fs_cloud_provider, **storage_options)
            fs.get(remote_filepath, local_filepath, recursive=True)
    except Timeout:
        # another process is responsible to download that file, continue
        pass


def upload_file_or_directory(local_filepath: str, remote_filepath: str, storage_options: Optional[Dict] = {}) -> None:
    """Upload a file to the remote cloud storage."""
    try:
        with FileLock(local_filepath + ".lock", timeout=3):
            fs_cloud_provider = get_cloud_provider(remote_filepath)
            fs = fsspec.filesystem(fs_cloud_provider, **storage_options)
            fs.put(local_filepath, remote_filepath, recursive=True)
    except Timeout:
        # another process is responsible to upload that file, continue
        pass


def copy_file_or_directory(
    remote_filepath_src: str, remote_filepath_tg: str, storage_options: Optional[Dict] = {}
) -> None:
    """Copy a file from src to target on the remote cloud storage."""
    fs_cloud_provider = get_cloud_provider(remote_filepath_src)
    fs = fsspec.filesystem(fs_cloud_provider, **storage_options)
    fs.copy(remote_filepath_src, remote_filepath_tg, recursive=True)


def remove_file_or_directory(remote_filepath: str, storage_options: Optional[Dict] = {}) -> None:
    """Remove a file from the remote cloud storage."""
    fs_cloud_provider = get_cloud_provider(remote_filepath)
    fs = fsspec.filesystem(fs_cloud_provider, **storage_options)
    fs.rm(remote_filepath, recursive=True)


def get_cloud_provider(remote_filepath: str) -> str:
    for k, fs_cloud_provider in _DOWNLOADERS.items():
        if str(remote_filepath).startswith(k):
            return fs_cloud_provider
    raise ValueError(f"The provided `remote_filepath` {remote_filepath} doesn't have a downloader associated.")


def get_downloader_cls(
    remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
) -> Downloader:
    for k, fs_cloud_provider in _DOWNLOADERS.items():
        if str(remote_dir).startswith(k):
            return FsspecDownloader(fs_cloud_provider, remote_dir, cache_dir, chunks, storage_options)
    raise ValueError(f"The provided `remote_dir` {remote_dir} doesn't have a downloader associated.")
