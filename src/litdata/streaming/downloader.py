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
import subprocess
from abc import ABC
from typing import Any, Dict, List, Optional
from urllib import parse

from filelock import FileLock, Timeout

from litdata.constants import _GOOGLE_STORAGE_AVAILABLE, _AZURE_STORAGE_AVAILABLE, _INDEX_FILENAME
from litdata.streaming.client import S3Client


class Downloader(ABC):
    def __init__(
        self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
    ):
        self._remote_dir = remote_dir
        self._cache_dir = cache_dir
        self._chunks = chunks
        self._storage_options = storage_options or {}

    def download_chunk_from_index(self, chunk_index: int) -> None:
        chunk_filename = self._chunks[chunk_index]["filename"]
        local_chunkpath = os.path.join(self._cache_dir, chunk_filename)
        remote_chunkpath = os.path.join(self._remote_dir, chunk_filename)
        self.download_file(remote_chunkpath, local_chunkpath)

    def download_file(self, remote_chunkpath: str, local_chunkpath: str) -> None:
        pass


class S3Downloader(Downloader):
    def __init__(
        self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
    ):
        super().__init__(remote_dir, cache_dir, chunks, storage_options)
        self._s5cmd_available = os.system("s5cmd > /dev/null 2>&1") == 0

        if not self._s5cmd_available:
            self._client = S3Client(storage_options=self._storage_options)

    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "s3":
            raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_filepath}")

        if os.path.exists(local_filepath):
            return

        try:
            with FileLock(local_filepath + ".lock", timeout=3 if obj.path.endswith(_INDEX_FILENAME) else 0):
                if self._s5cmd_available:
                    proc = subprocess.Popen(
                        f"s5cmd cp {remote_filepath} {local_filepath}",
                        shell=True,
                        stdout=subprocess.PIPE,
                    )
                    proc.wait()
                else:
                    from boto3.s3.transfer import TransferConfig

                    extra_args: Dict[str, Any] = {}

                    # try:
                    #     with FileLock(local_filepath + ".lock", timeout=1):
                    if not os.path.exists(local_filepath):
                        # Issue: https://github.com/boto/boto3/issues/3113
                        self._client.client.download_file(
                            obj.netloc,
                            obj.path.lstrip("/"),
                            local_filepath,
                            ExtraArgs=extra_args,
                            Config=TransferConfig(use_threads=False),
                        )
        except Timeout:
            # another process is responsible to download that file, continue
            pass


class GCPDownloader(Downloader):
    def __init__(
        self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
    ):
        if not _GOOGLE_STORAGE_AVAILABLE:
            raise ModuleNotFoundError(str(_GOOGLE_STORAGE_AVAILABLE))

        super().__init__(remote_dir, cache_dir, chunks, storage_options)

    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        from google.cloud import storage

        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "gs":
            raise ValueError(f"Expected obj.scheme to be `gs`, instead, got {obj.scheme} for remote={remote_filepath}")

        if os.path.exists(local_filepath):
            return

        try:
            with FileLock(local_filepath + ".lock", timeout=3 if obj.path.endswith(_INDEX_FILENAME) else 0):
                bucket_name = obj.netloc
                key = obj.path
                # Remove the leading "/":
                if key[0] == "/":
                    key = key[1:]

                client = storage.Client(**self._storage_options)
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(key)
                blob.download_to_filename(local_filepath)
        except Timeout:
            # another process is responsible to download that file, continue
            pass

class AzureDownloader(Downloader):
    def __init__(
        self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
    ):
        if not _AZURE_STORAGE_AVAILABLE:
            raise ModuleNotFoundError(str(_AZURE_STORAGE_AVAILABLE))

        super().__init__(remote_dir, cache_dir, chunks, storage_options)

    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
        from azure.storage.blob import BlobServiceClient

        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "azure":
            raise ValueError(f"Expected obj.scheme to be `azure`, instead, got {obj.scheme} for remote={remote_filepath}")

        if os.path.exists(local_filepath):
            return

        try:
            with FileLock(local_filepath + ".lock", timeout=3 if obj.path.endswith(_INDEX_FILENAME) else 0):
                directories = obj.path.lstrip('/').split('/')
                container_name = directories[0]
                blob_name = os.path.join(*directories[1:])
                service_client = BlobServiceClient.from_connection_string(self._storage_options["connection_string"])
                blob_client = service_client.get_blob_client(container=container_name, blob=blob_name)
                with open(local_filepath, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())
                
        except Timeout:
            # another process is responsible to download that file, continue
            pass

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


_DOWNLOADERS = {"s3://": S3Downloader, "gs://": GCPDownloader, "azure://":AzureDownloader, "local:": LocalDownloaderWithCache, "": LocalDownloader}


def get_downloader_cls(
    remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
) -> Downloader:
    for k, cls in _DOWNLOADERS.items():
        if str(remote_dir).startswith(k):
            return cls(remote_dir, cache_dir, chunks, storage_options)
    raise ValueError(f"The provided `remote_dir` {remote_dir} doesn't have a downloader associated.")
