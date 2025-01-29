"""contains utility functions to return parquet files from local, s3, or gs."""

import json
import os
from abc import ABC, abstractmethod
from time import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union

from litdata.constants import _INDEX_FILENAME
from litdata.streaming.resolver import Dir, _resolve_dir
from litdata.utilities.dataset_utilities import _try_create_cache_dir


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
    def __iter__(self) -> Generator[Tuple[str, str], None, None]: ...

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

    def __iter__(self) -> Generator[Tuple[str, str], None, None]:
        assert self.dir.path is not None
        assert self.dir.path != "", "Dir path can't be empty"

        for file_name in os.listdir(self.dir.path):
            if file_name.endswith(".parquet"):
                file_path = os.path.join(self.dir.path, file_name)
                yield file_name, file_path

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
        super().__init__(dir_path, cache_path, storage_options)
        if self.cache_path is None:
            self.cache_path = _try_create_cache_dir(self.dir.path)

        import fsspec

        assert self.dir.url is not None

        _CLOUD_PROVIDER = ("s3", "gs")

        for provider in _CLOUD_PROVIDER:
            if self.dir.url.startswith(provider):
                # Initialize the cloud filesystem
                self.fs = fsspec.filesystem(provider, storage_options=self.storage_options)
                print(f"using provider: {provider}")
                break

    def __iter__(self) -> Generator[Tuple[str, str], None, None]:
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

                yield file_name, local_path

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


_PARQUET_DIR: Dict[str, Type[ParquetDir]] = {
    "s3://": CloudParquetDir,
    "gs://": CloudParquetDir,
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
