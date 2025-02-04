"""contains utility functions to return parquet files from local, s3, or gs."""

import json
import os
from abc import ABC, abstractmethod
from subprocess import Popen
from time import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union

from litdata.constants import _DATATROVE_AVAILABLE, _FSSPEC_AVAILABLE, _INDEX_FILENAME
from litdata.streaming.resolver import Dir, _resolve_dir


class ParquetDir(ABC):
    def __init__(
        self,
        dir_path: Optional[Union[str, Dir]],
        cache_path: Optional[str] = None,
        storage_options: Optional[Dict] = {},
    ):
        self.dir = _resolve_dir(dir_path)
        self.cache_path = cache_path
        self.storage_options = storage_options

    @abstractmethod
    def __iter__(self) -> Generator[Tuple[str, str, str], None, None]: ...

    @abstractmethod
    def write_index(self, chunks_info: List[Dict[str, Any]], config: Dict[str, Any]) -> None: ...


class LocalParquetDir(ParquetDir):
    def __init__(
        self,
        dir_path: Optional[Union[str, Dir]],
        cache_path: Optional[str] = None,
        storage_options: Optional[Dict] = {},
    ):
        super().__init__(dir_path, cache_path, storage_options)

    def __iter__(self) -> Generator[Tuple[str, str, str], None, None]:
        assert self.dir.path is not None
        assert self.dir.path != "", "Dir path can't be empty"

        for file_name in os.listdir(self.dir.path):
            if file_name.endswith(".parquet"):
                file_path = os.path.join(self.dir.path, file_name)
                yield file_name, file_path, None

    def write_index(self, chunks_info: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        # write to index.json file
        if self.cache_path is None:
            self.cache_path = self.dir.path
        assert self.cache_path is not None
        with open(os.path.join(self.cache_path, _INDEX_FILENAME), "w") as f:
            data = {"chunks": chunks_info, "config": config, "updated_at": str(time())}
            json.dump(data, f, sort_keys=True)

        print(f"Index file written to: {os.path.join(self.cache_path, _INDEX_FILENAME)}")


class CloudParquetDir(ParquetDir):
    def __init__(
        self,
        dir_path: Optional[Union[str, Dir]],
        cache_path: Optional[str] = None,
        storage_options: Optional[Dict] = None,
    ):
        if not _FSSPEC_AVAILABLE:
            raise ModuleNotFoundError(
                "Support for Indexing cloud parquet files depends on `fsspec`.", "Please, run: `pip install fsspec`"
            )

        super().__init__(dir_path, cache_path, storage_options)
        if self.cache_path is None:
            self.cache_path = os.path.join(os.path.expanduser("~"), ".cache", ".litdata-cache-index-pq")
            os.makedirs(self.cache_path, exist_ok=True)  # Ensure the directory exists

        import fsspec

        assert self.dir.url is not None

        _CLOUD_PROVIDER = ("s3", "gs")

        for provider in _CLOUD_PROVIDER:
            if self.dir.url.startswith(provider):
                # Initialize the cloud filesystem
                self.fs = fsspec.filesystem(provider, *self.storage_options)
                print(f"using provider: {provider}")
                break

    def __iter__(self) -> Generator[Tuple[str, str, str], None, None]:
        assert self.dir.url is not None
        assert self.cache_path is not None

        # List all files and directories in the top-level of the specified directory
        files = self.fs.ls(self.dir.url, detail=True)

        # Iterate through the items and check for Parquet files
        for file_info in files:
            if file_info["type"] == "file" and file_info["name"].endswith(".parquet"):
                file_name = os.path.basename(file_info["name"])
                assert self.cache_path is not None
                local_path = os.path.join(self.cache_path, file_name)

                # Download the file
                with self.fs.open(file_info["name"], "rb") as cloud_file, open(local_path, "wb") as local_file:
                    local_file.write(cloud_file.read())

                yield file_name, local_path, None

    def write_index(self, chunks_info: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        assert self.cache_path is not None
        assert self.dir.url is not None

        index_file_path = os.path.join(self.cache_path, _INDEX_FILENAME)
        cloud_index_path = os.path.join(self.dir.url, _INDEX_FILENAME)
        # write to index.json file
        with open(index_file_path, "w") as f:
            data = {"chunks": chunks_info, "config": config, "updated_at": str(time())}
            json.dump(data, f, sort_keys=True)

        # upload file to cloud
        with open(index_file_path, "rb") as local_file, self.fs.open(cloud_index_path, "wb") as cloud_file:
            cloud_file.write(local_file.read())

        print(f"Index file written to: {cloud_index_path}")


class HFParquetDir(ParquetDir):
    def __init__(
        self,
        dir_path: Optional[Union[str, Dir]],
        cache_path: Optional[str] = None,
        storage_options: Optional[Dict] = None,
    ):
        if not _DATATROVE_AVAILABLE:
            raise ModuleNotFoundError(
                "Support for Indexing HF depends on `datatrove.io`.", "Please, run: `pip install 'datatrove[io]>0.3.0'`"
            )

        super().__init__(dir_path, cache_path, storage_options)
        if self.cache_path is None:
            self.cache_path = os.path.join(os.path.expanduser("~"), ".cache", ".litdata-cache-index-pq")
            os.makedirs(self.cache_path, exist_ok=True)  # Ensure the directory exists

        assert self.dir.url is not None
        assert self.dir.url.startswith("hf")

    def __iter__(self) -> Generator[Tuple[str, str, str], None, None]:
        assert self.dir.url is not None
        assert self.cache_path is not None

        # List all files and directories in the top-level of the specified directory
        from datatrove.io import get_datafolder

        data_folder = get_datafolder(self.dir.url)
        filepaths = data_folder.get_shard(rank=0, world_size=1, recursive=True)  # get all files
        pq_files_data: List[Tuple[str, str]] = []

        for fp in filepaths:
            with data_folder.open(fp, "rb") as f:
                pq_files_data.append([fp, str(f.url())])

        # Iterate through the items and check for Parquet files
        for file_name, file_url in pq_files_data:
            if file_name.endswith(".parquet"):
                local_path = os.path.join(self.cache_path, file_name)
                try:
                    cmd = f"wget -q {file_url} -O {local_path}"
                    Popen(cmd, shell=True).wait()
                    yield file_name, local_path, file_url
                    if os.path.exists(local_path):
                        os.remove(local_path)
                except Exception as e:
                    print(e)
                    pass

    def write_index(self, chunks_info: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        assert self.cache_path is not None
        assert self.dir.url is not None

        index_file_path = os.path.join(self.cache_path, _INDEX_FILENAME)
        cloud_index_path = os.path.join(self.dir.url, _INDEX_FILENAME)
        # write to index.json file
        with open(index_file_path, "w") as f:
            data = {"chunks": chunks_info, "config": config, "updated_at": str(time())}
            json.dump(data, f, sort_keys=True)

        print(f"Index file written to: {cloud_index_path}")


_PARQUET_DIR: Dict[str, Type[ParquetDir]] = {
    "s3://": CloudParquetDir,
    "gs://": CloudParquetDir,
    "hf://": HFParquetDir,
    "local:": LocalParquetDir,
    "": LocalParquetDir,
}


def get_parquet_indexer_cls(
    dir_path: str, cache_path: Optional[str] = None, storage_options: Optional[Dict] = {}
) -> ParquetDir:
    for k, cls in _PARQUET_DIR.items():
        if str(dir_path).startswith(k):
            return cls(dir_path, cache_path, storage_options)
    raise ValueError(f"The provided `dir_path` {dir_path} doesn't have a ParquetDir class associated.")
