"""contains utility functions to return parquet files from local, s3, or gs."""

import hashlib
import json
import os
from abc import ABC, abstractmethod
from time import time
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Type, TypeVar, Union

from litdata.constants import (
    _FSSPEC_AVAILABLE,
    _HF_HUB_AVAILABLE,
    _INDEX_FILENAME,
    _TQDM_AVAILABLE,
)
from litdata.streaming.resolver import Dir, _resolve_dir

if _TQDM_AVAILABLE:
    from tqdm import tqdm
else:
    # hacky way for now to ignore tqdm import
    T = TypeVar("T")

    def tqdm(iterable: Iterable[T]) -> Iterable[T]:
        return iterable


class ParquetDir(ABC):
    def __init__(
        self,
        dir_path: Optional[Union[str, Dir]],
        cache_path: Optional[str] = None,
        storage_options: Optional[Dict] = {},
        remove_after_indexing: bool = True,
    ):
        self.dir = _resolve_dir(dir_path)
        self.cache_path = cache_path
        self.storage_options = storage_options
        self.remove_after_indexing = remove_after_indexing
        self.files = []

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
        remove_after_indexing: bool = True,
    ):
        super().__init__(dir_path, cache_path, storage_options, remove_after_indexing)
        self.files = os.listdir(self.dir.path)

    def __iter__(self) -> Generator[Tuple[str, str], None, None]:
        assert self.dir.path is not None
        assert self.dir.path != "", "Dir path can't be empty"

        for file_name in tqdm(self.files):
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
        remove_after_indexing: bool = True,
    ):
        if not _FSSPEC_AVAILABLE:
            raise ModuleNotFoundError(
                "Support for Indexing cloud parquet files depends on `fsspec`.", "Please, run: `pip install fsspec`"
            )

        super().__init__(dir_path, cache_path, storage_options, remove_after_indexing)

        assert self.dir.url is not None

        if self.cache_path is None:
            self.cache_path = default_cache_dir(self.dir.url)
            os.makedirs(self.cache_path, exist_ok=True)  # Ensure the directory exists

        import fsspec

        _CLOUD_PROVIDER = ("s3", "gs")

        for provider in _CLOUD_PROVIDER:
            if self.dir.url.startswith(provider):
                # Initialize the cloud filesystem
                self.fs = fsspec.filesystem(provider, *self.storage_options)
                print(f"using provider: {provider}")
                break

        # List all files and directories in the top-level of the specified directory
        self.files = self.fs.ls(self.dir.url, detail=True)

    def __iter__(self) -> Generator[Tuple[str, str], None, None]:
        assert self.dir.url is not None
        assert self.cache_path is not None

        # Iterate through the items and check for Parquet files
        for file_info in tqdm(self.files):
            if file_info["type"] == "file" and file_info["name"].endswith(".parquet"):
                file_name = os.path.basename(file_info["name"])
                assert self.cache_path is not None
                local_path = os.path.join(self.cache_path, file_name)

                # Download the file
                with self.fs.open(file_info["name"], "rb") as cloud_file, open(local_path, "wb") as local_file:
                    local_file.write(cloud_file.read())

                yield file_name, local_path
                if os.path.exists(local_path) and self.remove_after_indexing:
                    os.remove(local_path)

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
        remove_after_indexing: bool = True,
    ):
        if not _HF_HUB_AVAILABLE:
            raise ModuleNotFoundError(
                "Support for Indexing HF depends on `huggingface_hub`.", "Please, run: `pip install huggingface_hub"
            )

        super().__init__(dir_path, cache_path, storage_options, remove_after_indexing)

        assert self.dir.url is not None
        assert self.dir.url.startswith("hf")

        if self.cache_path is None:
            self.cache_path = default_cache_dir(self.dir.url)
            os.makedirs(self.cache_path, exist_ok=True)  # Ensure the directory exists

        # List all files and directories in the top-level of the specified directory
        from huggingface_hub import HfFileSystem

        self.fs = HfFileSystem()
        self.files = self.fs.ls(self.dir.url, detail=False)

    def __iter__(self) -> Generator[Tuple[str, str], None, None]:
        assert self.dir.url is not None
        assert self.cache_path is not None

        # Iterate through the items and check for Parquet files
        for file_name in tqdm(self.files):
            assert isinstance(file_name, str)
            if file_name.endswith(".parquet"):
                local_path = os.path.join(self.cache_path, file_name)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                temp_path = local_path + ".tmp"  # Avoid partial writes

                try:
                    with self.fs.open(file_name, "rb") as cloud_file, open(temp_path, "wb") as local_file:
                        data = cloud_file.read()
                        if isinstance(data, str):
                            raise ValueError(f"Expected parquet data in bytes format. But found str. {file_name}")
                        local_file.write(data)
                    os.rename(temp_path, local_path)  # Atomic move after successful write
                    yield file_name, local_path

                except Exception as e:
                    print(f"Error processing: {e}")

                finally:
                    # Ensure cleanup of temp file if an error occurs
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                if os.path.exists(local_path) and self.remove_after_indexing:
                    os.remove(local_path)

    def write_index(self, chunks_info: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        assert self.cache_path is not None
        assert self.dir.url is not None

        index_file_path = os.path.join(self.cache_path, _INDEX_FILENAME)
        # write to index.json file
        with open(index_file_path, "w") as f:
            data = {"chunks": chunks_info, "config": config, "updated_at": str(time())}
            json.dump(data, f, sort_keys=True)

        print(f"Index file written to: {index_file_path}")


_PARQUET_DIR: Dict[str, Type[ParquetDir]] = {
    "s3://": CloudParquetDir,
    "gs://": CloudParquetDir,
    "hf://": HFParquetDir,
    "local:": LocalParquetDir,
    "": LocalParquetDir,
}


def get_parquet_indexer_cls(
    dir_path: str,
    cache_path: Optional[str] = None,
    storage_options: Optional[Dict] = {},
    remove_after_indexing: bool = True,
) -> ParquetDir:
    for k, cls in _PARQUET_DIR.items():
        if str(dir_path).startswith(k):
            return cls(dir_path, cache_path, storage_options, remove_after_indexing)
    raise ValueError(f"The provided `dir_path` {dir_path} doesn't have a ParquetDir class associated.")


def default_cache_dir(url: str) -> str:
    # Hash the URL using SHA256 and take the first 16 characters for brevity
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    cache_path = os.path.join(os.path.expanduser("~"), ".cache", "litdata-cache-index-pq", url_hash)
    os.makedirs(cache_path, exist_ok=True)  # Ensure the directory exists
    return cache_path
