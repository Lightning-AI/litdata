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

import fsspec
from filelock import FileLock, Timeout

from litdata.constants import (
    _FSSPEC_AVAILABLE,
    _HF_HUB_AVAILABLE,
    _INDEX_FILENAME,
    _SUPPORTED_CLOUD_PROVIDERS,
)

_USE_S5CMD_FOR_S3 = True

_DEFAULT_STORAGE_OPTIONS = {
    "s3": {"config_kwargs": {"retries": {"max_attempts": 1000, "mode": "adaptive"}}},
}


def get_complete_storage_options(cloud_provider: str, storage_options: Optional[Dict] = {}) -> Dict:
    if storage_options is None:
        storage_options = {}
    if cloud_provider in _DEFAULT_STORAGE_OPTIONS:
        return {**_DEFAULT_STORAGE_OPTIONS[cloud_provider], **storage_options}
    return storage_options


def download_s3_file_via_s5cmd(remote_filepath: str, local_filepath: str) -> None:
    _s5cmd_available = os.system("s5cmd > /dev/null 2>&1") == 0

    if _s5cmd_available is False:
        raise ModuleNotFoundError(str(_s5cmd_available))

    obj = parse.urlparse(remote_filepath)

    if obj.scheme != "s3":
        raise ValueError(f"Expected obj.scheme to be `s3`, instead, got {obj.scheme} for {remote_filepath}")

    if os.path.exists(local_filepath):
        return

    try:
        with FileLock(local_filepath + ".lock", timeout=3 if obj.path.endswith(_INDEX_FILENAME) else 0):
            proc = subprocess.Popen(
                f"s5cmd cp {remote_filepath} {local_filepath}",
                shell=True,
                stdout=subprocess.PIPE,
            )
            proc.wait()
    except Timeout:
        # another process is responsible to download that file, continue
        pass


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
        self.download_file(remote_chunkpath, local_chunkpath, chunk_filename)

    def download_file(self, remote_chunkpath: str, local_chunkpath: str, remote_chunk_filename: str = "") -> None:
        pass


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


class FsspecDownloader(Downloader):
    def __init__(
        self, remote_dir: str, cache_dir: str, chunks: List[Dict[str, Any]], storage_options: Optional[Dict] = {}
    ):
        if not _FSSPEC_AVAILABLE:
            raise ModuleNotFoundError(str(_FSSPEC_AVAILABLE))
        cloud_provider = parse.urlparse(remote_dir).scheme
        if cloud_provider not in _SUPPORTED_CLOUD_PROVIDERS:
            raise ValueError(
                f"Cloud provider {cloud_provider} is not supported by LitData.",
                "Supported providers are: {_SUPPORTED_CLOUD_PROVIDERS}",
            )
        super().__init__(remote_dir, cache_dir, chunks, storage_options)
        self.fs = fsspec.filesystem(cloud_provider, **storage_options)
        self.cloud_provider = cloud_provider
        self.use_s5cmd = cloud_provider == "s3" and os.system("s5cmd > /dev/null 2>&1") == 0

    def download_file(self, remote_filepath: str, local_filepath: str, remote_chunk_filename: str = "") -> None:
        if os.path.exists(local_filepath) or remote_filepath == local_filepath:
            return
        if self.use_s5cmd and _USE_S5CMD_FOR_S3:
            download_s3_file_via_s5cmd(remote_filepath, local_filepath)
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


class LocalDownloaderWithCache(LocalDownloader):
    def download_file(self, remote_filepath: str, local_filepath: str, remote_chunk_filename: str = "") -> None:
        remote_filepath = remote_filepath.replace("local:", "")
        super().download_file(remote_filepath, local_filepath)


_DOWNLOADERS = {
    "s3://": FsspecDownloader,
    "gs://": FsspecDownloader,
    "azure://": FsspecDownloader,
    "hf://": HFDownloader,
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
