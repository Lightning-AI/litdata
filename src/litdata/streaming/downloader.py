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
from typing import Any, Dict, List, Optional, Union
from urllib import parse

import fsspec
from filelock import FileLock, Timeout

from litdata.constants import _INDEX_FILENAME

_USE_S5CMD_FOR_S3 = True


class Downloader(ABC):
    def __init__(
        self,
        cloud_provider: str,
        remote_dir: str,
        cache_dir: str,
        chunks: List[Dict[str, Any]],
        storage_options: Optional[Dict] = {},
    ):
        self._remote_dir = remote_dir
        self._cache_dir = cache_dir
        self._chunks = chunks
        self.fs = fsspec.filesystem(cloud_provider, **storage_options)

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

        try:
            with FileLock(local_filepath + ".lock", timeout=3 if remote_filepath.endswith(_INDEX_FILENAME) else 0):
                if remote_filepath != local_filepath and not os.path.exists(local_filepath):
                    # make an atomic operation to be safe
                    temp_file_path = local_filepath + ".tmp"
                    shutil.copy(remote_filepath, temp_file_path)
                    os.rename(temp_file_path, local_filepath)
                    with contextlib.suppress(Exception):
                        os.remove(local_filepath + ".lock")
        except Timeout:
            pass


class LocalDownloaderWithCache(LocalDownloader):
    def download_file(self, remote_filepath: str, local_filepath: str, remote_chunk_filename: str = "") -> None:
        remote_filepath = remote_filepath.replace("local:", "")
        super().download_file(remote_filepath, local_filepath)


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


_DOWNLOADERS = {
    "s3://": "s3",
    "gs://": "gs",
    "azure://": "abfs",
    "abfs://": "abfs",
    "local:": "file",
    "": "file",
}

_DEFAULT_STORAGE_OPTIONS = {
    "s3": {"config_kwargs": {"retries": {"max_attempts": 1000, "mode": "adaptive"}}},
}


def get_complete_storage_options(cloud_provider: str, storage_options: Optional[Dict] = {}) -> Dict:
    if storage_options is None:
        storage_options = {}
    if cloud_provider in _DEFAULT_STORAGE_OPTIONS:
        return {**_DEFAULT_STORAGE_OPTIONS[cloud_provider], **storage_options}
    return storage_options


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
        storage_options = get_complete_storage_options(cloud_provider, storage_options)
        super().__init__(cloud_provider, remote_dir, cache_dir, chunks, storage_options)
        self.cloud_provider = cloud_provider
        self.use_s5cmd = cloud_provider == "s3" and os.system("s5cmd > /dev/null 2>&1") == 0

    def download_file(self, remote_filepath: str, local_filepath: str) -> None:
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


def does_file_exist(
    remote_filepath: str, cloud_provider: Union[str, None] = None, storage_options: Optional[Dict] = {}
) -> bool:
    if cloud_provider is None:
        cloud_provider = get_cloud_provider(remote_filepath)
    storage_options = get_complete_storage_options(cloud_provider, storage_options)
    fs = fsspec.filesystem(cloud_provider, **storage_options)
    return fs.exists(remote_filepath)


def list_directory(
    remote_directory: str,
    detail: bool = False,
    cloud_provider: Optional[str] = None,
    storage_options: Optional[Dict] = {},
) -> List[str]:
    """Returns a list of filenames in a remote directory."""
    if cloud_provider is None:
        cloud_provider = get_cloud_provider(remote_directory)
    storage_options = get_complete_storage_options(cloud_provider, storage_options)
    fs = fsspec.filesystem(cloud_provider, **storage_options)
    return fs.ls(remote_directory, detail=detail)  # just return the filenames


def download_file_or_directory(remote_filepath: str, local_filepath: str, storage_options: Optional[Dict] = {}) -> None:
    """Download a file from the remote cloud storage."""
    fs_cloud_provider = get_cloud_provider(remote_filepath)
    use_s5cmd = fs_cloud_provider == "s3" and os.system("s5cmd > /dev/null 2>&1") == 0
    if use_s5cmd and _USE_S5CMD_FOR_S3:
        download_s3_file_via_s5cmd(remote_filepath, local_filepath)
        return
    try:
        with FileLock(local_filepath + ".lock", timeout=3):
            storage_options = get_complete_storage_options(fs_cloud_provider, storage_options)
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
            storage_options = get_complete_storage_options(fs_cloud_provider, storage_options)
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
    storage_options = get_complete_storage_options(fs_cloud_provider, storage_options)
    fs = fsspec.filesystem(fs_cloud_provider, **storage_options)
    fs.copy(remote_filepath_src, remote_filepath_tg, recursive=True)


def remove_file_or_directory(remote_filepath: str, storage_options: Optional[Dict] = {}) -> None:
    """Remove a file from the remote cloud storage."""
    fs_cloud_provider = get_cloud_provider(remote_filepath)
    storage_options = get_complete_storage_options(fs_cloud_provider, storage_options)
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
