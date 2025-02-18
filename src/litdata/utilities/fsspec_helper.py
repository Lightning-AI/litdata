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
from typing import Dict, List, Optional, Union

import fsspec
from filelock import FileLock, Timeout

from litdata.constants import _SUPPORTED_CLOUD_PROVIDERS
from litdata.streaming.downloader import _USE_S5CMD_FOR_S3, download_s3_file_via_s5cmd, get_complete_storage_options


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
    for _cloud_provider in _SUPPORTED_CLOUD_PROVIDERS:
        if remote_filepath.startswith(_cloud_provider):
            if _cloud_provider == "azure":
                return "abfs"
            return _cloud_provider

    raise ValueError(
        f"Cloud provider {remote_filepath} is not supported by LitData.",
        "Supported providers are: {_SUPPORTED_CLOUD_PROVIDERS}",
    )
