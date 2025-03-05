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

import contextlib
import os
import shutil
import subprocess
from abc import ABC
from contextlib import suppress
from typing import Any, Dict, List, Optional
from urllib import parse

from filelock import FileLock, Timeout

from litdata.constants import (
    _AZURE_STORAGE_AVAILABLE,
    _DATABRICKS_SDK_AVAILABLE,
    _GOOGLE_STORAGE_AVAILABLE,
    _HF_HUB_AVAILABLE,
    _INDEX_FILENAME,
)
from litdata.streaming.client import S3Client


class Downloader(ABC):
    def __init__(
        self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
    ):
        self._remote_dir = remote_dir
        self._cache_dir = cache_dir
        self._chunks = chunks
        self._storage_options = storage_options or {}

    def _increment_local_lock(self, chunkpath: str) -> None:
        countpath = chunkpath + ".cnt"
        with suppress(Timeout), FileLock(countpath + ".lock", timeout=3):
            try:
                with open(countpath) as count_f:
                    curr_count = int(count_f.read().strip())
            except Exception:
                curr_count = 0
            curr_count += 1
            with open(countpath, "w+") as count_f:
                count_f.write(str(curr_count))

    def download_chunk_from_index(self, chunk_index: int) -> None:
        chunk_filename = self._chunks[chunk_index]["filename"]
        local_chunkpath = os.path.join(self._cache_dir, chunk_filename)
        remote_chunkpath = os.path.join(self._remote_dir, chunk_filename)

        self.download_file(remote_chunkpath, local_chunkpath, chunk_filename)

    def download_file(self, remote_chunkpath: str, local_chunkpath: str, remote_chunk_filename: str = "") -> None:
        pass


class S3Downloader(Downloader):
    def __init__(
        self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
    ):
        super().__init__(remote_dir, cache_dir, chunks, storage_options)
        self._s5cmd_available = os.system("s5cmd > /dev/null 2>&1") == 0

        if not self._s5cmd_available:
            self._client = S3Client(storage_options=self._storage_options)

    def download_file(self, remote_filepath: str, local_filepath: str, remote_chunk_filename: str = "") -> None:
        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "s3":
            raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for remote={remote_filepath}")

        if os.path.exists(local_filepath):
            return

        with suppress(Timeout), FileLock(
            local_filepath + ".lock", timeout=3 if obj.path.endswith(_INDEX_FILENAME) else 0
        ):
            if self._s5cmd_available:
                env = None
                if self._storage_options:
                    env = os.environ.copy()
                    env.update(self._storage_options)
                proc = subprocess.Popen(
                    f"s5cmd cp {remote_filepath} {local_filepath}",
                    shell=True,
                    stdout=subprocess.PIPE,
                    env=env,
                )
                proc.wait()
            else:
                from boto3.s3.transfer import TransferConfig

                extra_args: Dict[str, Any] = {}

                if not os.path.exists(local_filepath):
                    # Issue: https://github.com/boto/boto3/issues/3113
                    self._client.client.download_file(
                        obj.netloc,
                        obj.path.lstrip("/"),
                        local_filepath,
                        ExtraArgs=extra_args,
                        Config=TransferConfig(use_threads=False),
                    )


class GCPDownloader(Downloader):
    def __init__(
        self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
    ):
        if not _GOOGLE_STORAGE_AVAILABLE:
            raise ModuleNotFoundError(str(_GOOGLE_STORAGE_AVAILABLE))

        super().__init__(remote_dir, cache_dir, chunks, storage_options)

    def download_file(self, remote_filepath: str, local_filepath: str, remote_chunk_filename: str = "") -> None:
        from google.cloud import storage

        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "gs":
            raise ValueError(f"Expected obj.scheme to be `gs`, instead, got {obj.scheme} for remote={remote_filepath}")

        if os.path.exists(local_filepath):
            return

        with suppress(Timeout), FileLock(
            local_filepath + ".lock", timeout=3 if obj.path.endswith(_INDEX_FILENAME) else 0
        ):
            bucket_name = obj.netloc
            key = obj.path
            # Remove the leading "/":
            if key[0] == "/":
                key = key[1:]

            client = storage.Client(**self._storage_options)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(key)
            blob.download_to_filename(local_filepath)


class AzureDownloader(Downloader):
    def __init__(
        self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
    ):
        if not _AZURE_STORAGE_AVAILABLE:
            raise ModuleNotFoundError(str(_AZURE_STORAGE_AVAILABLE))

        super().__init__(remote_dir, cache_dir, chunks, storage_options)

    def download_file(self, remote_filepath: str, local_filepath: str, remote_chunk_filename: str = "") -> None:
        from azure.storage.blob import BlobServiceClient

        obj = parse.urlparse(remote_filepath)

        if obj.scheme != "azure":
            raise ValueError(
                f"Expected obj.scheme to be `azure`, instead, got {obj.scheme} for remote={remote_filepath}"
            )

        if os.path.exists(local_filepath):
            return

        with suppress(Timeout), FileLock(
            local_filepath + ".lock", timeout=3 if obj.path.endswith(_INDEX_FILENAME) else 0
        ):
            service = BlobServiceClient(**self._storage_options)
            blob_client = service.get_blob_client(container=obj.netloc, blob=obj.path.lstrip("/"))
            with open(local_filepath, "wb") as download_file:
                blob_data = blob_client.download_blob()
                blob_data.readinto(download_file)


class LocalDownloader(Downloader):
    def download_file(self, remote_filepath: str, local_filepath: str, remote_chunk_filename: str = "") -> None:
        if not os.path.exists(remote_filepath):
            raise FileNotFoundError(f"The provided remote_path doesn't exist: {remote_filepath}")

        with suppress(Timeout), FileLock(
            local_filepath + ".lock", timeout=3 if remote_filepath.endswith(_INDEX_FILENAME) else 0
        ):
            if remote_filepath == local_filepath or os.path.exists(local_filepath):
                return
            # make an atomic operation to be safe
            temp_file_path = local_filepath + ".tmp"
            shutil.copy(remote_filepath, temp_file_path)
            os.rename(temp_file_path, local_filepath)
            with contextlib.suppress(Exception):
                os.remove(local_filepath + ".lock")


class HFDownloader(Downloader):
    def __init__(
        self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
    ):
        if not _HF_HUB_AVAILABLE:
            raise ModuleNotFoundError(
                "Support for Downloading HF dataset depends on `huggingface_hub`.",
                "Please, run: `pip install huggingface_hub",
            )

        super().__init__(remote_dir, cache_dir, chunks, storage_options)
        from huggingface_hub import HfFileSystem

        self.fs = HfFileSystem()

    def download_file(self, remote_filepath: str, local_filepath: str, remote_chunk_filename: str = "") -> None:
        # for HF dataset downloading, we don't need remote_filepath, but remote_chunk_filename
        with suppress(Timeout), FileLock(local_filepath + ".lock", timeout=0):
            temp_path = local_filepath + ".tmp"  # Avoid partial writes
            try:
                with self.fs.open(remote_chunk_filename, "rb") as cloud_file, open(temp_path, "wb") as local_file:
                    for chunk in iter(lambda: cloud_file.read(4096), b""):  # Stream in 4KB chunks local_file.
                        local_file.write(chunk)

                os.rename(temp_path, local_filepath)  # Atomic move after successful write

            except Exception as e:
                print(f"Error processing {remote_chunk_filename}: {e}")

            finally:
                # Ensure cleanup of temp file if an error occurs
                if os.path.exists(temp_path):
                    os.remove(temp_path)


class DBFSDownloader(Downloader):
    def __init__(
        self,
        remote_dir: str,
        cache_dir: str,
        chunks: List[Dict[str, Any]],
        storage_options: Optional[Dict] = {},
    ):
        if not _DATABRICKS_SDK_AVAILABLE:
            raise ModuleNotFoundError(
                "Support for Downloading dbfs dataset depends on `databricks-sdk`.",
                "Please, run: `pip install databricks-sdk",
            )

        super().__init__(
            remote_dir=remote_dir,
            cache_dir=cache_dir,
            chunks=chunks,
            storage_options=storage_options,
        )

        from databricks.sdk import WorkspaceClient

        self._dbfs_client = WorkspaceClient(**self._storage_options)

    def download_file(self, remote_filepath: str, local_filepath: str, remote_chunk_filename: str = "") -> None:
        from databricks.sdk.core import DatabricksError

        assert self._dbfs_client is not None

        obj = parse.urlparse(remote_filepath)
        if obj.scheme != "dbfs":
            raise ValueError(f"Expected scheme to be `dbfs`, instead, got {obj.scheme} for remote={remote_filepath}")

        if os.path.exists(local_filepath):
            return

        with suppress(Timeout), FileLock(
            local_filepath + ".lock", timeout=3 if obj.path.endswith(_INDEX_FILENAME) else 0
        ):
            local_tmp = local_filepath + ".tmp"
            response = self._dbfs_client.files.download(obj.path).contents

            assert response is not None

            try:
                with response, open(local_tmp, "wb") as f:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        f.write(chunk)

                os.rename(local_tmp, local_filepath)
            except DatabricksError as e:
                if e.error_code == "PERMISSION_DENIED":
                    e.args = (
                        "Ensure the file path or credentials are set correctly. For "
                        + "Databricks Unity Catalog, file path must starts with `dbfs:/Volumes` "
                        + "and for Databricks File System, file path must starts with `dbfs`. "
                        + e.args[0],
                    )
                raise e
            except Exception as e:
                raise e
            finally:
                # Ensure cleanup of temp file if an error occurs
                if os.path.exists(local_tmp):
                    os.remove(local_tmp)


class LocalDownloaderWithCache(LocalDownloader):
    def download_file(self, remote_filepath: str, local_filepath: str, remote_chunk_filename: str = "") -> None:
        remote_filepath = remote_filepath.replace("local:", "")
        super().download_file(remote_filepath, local_filepath)


_DOWNLOADERS = {
    "s3://": S3Downloader,
    "gs://": GCPDownloader,
    "azure://": AzureDownloader,
    "hf://": HFDownloader,
    "dbfs:/": DBFSDownloader,
    "local:": LocalDownloaderWithCache,
    "": LocalDownloader,
}


def get_downloader_cls(
    remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
) -> Downloader:
    for k, cls in _DOWNLOADERS.items():
        if str(remote_dir).startswith(k):
            return cls(remote_dir, cache_dir, chunks, storage_options)
    raise ValueError(f"The provided `remote_dir` {remote_dir} doesn't have a downloader associated.")
