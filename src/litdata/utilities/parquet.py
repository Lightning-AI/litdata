"""contains utility functions to return parquet files from local, s3, or gs."""

import hashlib
import io
import json
import os
import sys
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from queue import Queue
from time import sleep, time
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from urllib import parse

from litdata.constants import _FSSPEC_AVAILABLE, _HF_HUB_AVAILABLE, _INDEX_FILENAME, _PYARROW_AVAILABLE
from litdata.streaming.resolver import Dir, _resolve_dir


def delete_thread(rmq: Queue) -> None:
    while True:
        file_path = rmq.get()
        if file_path is None:  # Sentinel value to exit
            break
        with suppress(FileNotFoundError):
            os.remove(file_path)

    # before exiting, just sleep for some time, to complete any pending deletions
    sleep(0.5)


class ParquetDir(ABC):
    def __init__(
        self,
        dir_path: Optional[Union[str, Dir]],
        cache_path: Optional[str] = None,
        storage_options: Optional[Dict] = {},
        remove_after_indexing: bool = True,
        num_workers: int = 4,
    ):
        self.dir = _resolve_dir(dir_path)
        self.cache_path = cache_path
        self.storage_options = storage_options
        self.remove_after_indexing = remove_after_indexing
        self.files: List[Any] = []
        self.process_queue: Queue = Queue()
        self.delete_queue: Queue = Queue()
        self.num_workers = num_workers

    def __iter__(self) -> Generator[Tuple[str, str], None, None]:
        # start worker in a separate thread, and then read values from `process_queue`, and yield

        self.is_delete_thread_running = self.dir.url is not None and self.remove_after_indexing

        if self.is_delete_thread_running:
            t = threading.Thread(
                target=delete_thread,
                name="delete_thread",
                args=(self.delete_queue,),
            )
            t.start()

        t_worker = threading.Thread(
            target=self.worker,
            name="worker_thread",
            args=(),
        )
        t_worker.start()

        # TODO: simplify this
        while True:
            file_name, file_path, order = self.process_queue.get()
            if file_name is None and file_path is None:  # Sentinel value to exit
                break
            yield file_name, file_path, order
            if self.is_delete_thread_running:
                self.delete_queue.put_nowait(file_path)

        if self.is_delete_thread_running:
            self.delete_queue.put_nowait(None)  # so that it doesn't hang indefinitely
            t.join()

        t_worker.join()  # should've completed by now. But, just to be on safe side.

    @abstractmethod
    def task(self, _file: Any, order: int) -> None: ...

    def worker(self) -> None:
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for order, _file in enumerate(self.files):
                executor.submit(self.task, _file, order)
        self.process_queue.put_nowait((None, None, None))

    @abstractmethod
    def write_index(self, chunks_info: List[Dict[str, Any]], config: Dict[str, Any]) -> None: ...


class LocalParquetDir(ParquetDir):
    def __init__(
        self,
        dir_path: Optional[Union[str, Dir]],
        cache_path: Optional[str] = None,
        storage_options: Optional[Dict] = {},
        remove_after_indexing: bool = True,
        num_workers: int = 4,
    ):
        super().__init__(dir_path, cache_path, storage_options, remove_after_indexing, num_workers)

        for _f in os.listdir(self.dir.path):
            if _f.endswith(".parquet"):
                self.files.append(_f)

    def task(self, _file: str, order: int) -> None:
        assert isinstance(_file, str)

        if _file.endswith(".parquet"):
            assert self.dir.path is not None
            assert self.dir.path != "", "Dir path can't be empty"

            file_path = os.path.join(self.dir.path, _file)
            self.process_queue.put_nowait((_file, file_path, order))

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
        num_workers: int = 4,
    ):
        if not _FSSPEC_AVAILABLE:
            raise ModuleNotFoundError(
                "Support for Indexing cloud parquet files depends on `fsspec`.", "Please, run: `pip install fsspec`"
            )

        super().__init__(dir_path, cache_path, storage_options, remove_after_indexing, num_workers)

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
        for _f in self.fs.ls(self.dir.url, detail=True):
            if _f["type"] == "file" and _f["name"].endswith(".parquet"):
                self.files.append(_f)

    def task(self, _file: Any, order: int) -> None:
        if _file["type"] == "file" and _file["name"].endswith(".parquet"):
            file_name = os.path.basename(_file["name"])
            assert self.cache_path is not None
            local_path = os.path.join(self.cache_path, file_name)
            temp_path = local_path + ".tmp"  # Avoid partial writes
            # if an existing temp file is present, means its corrupted
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Download the file
            with self.fs.open(_file["name"], "rb") as cloud_file, open(temp_path, "wb") as local_file:
                for chunk in iter(lambda: cloud_file.read(4096), b""):  # Read in 4KB chunks
                    local_file.write(chunk)

            os.rename(temp_path, local_path)  # Atomic move after successful write
            self.process_queue.put_nowait((file_name, local_path, order))

    def write_index(self, chunks_info: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        assert self.cache_path is not None
        assert self.dir.url is not None

        # clear any `.lock` or `.tmp` chunk file if left
        if self.is_delete_thread_running:
            for file in os.listdir(self.cache_path):
                if file != _INDEX_FILENAME:
                    with suppress(FileNotFoundError):
                        os.remove(file)

        index_file_path = os.path.join(self.cache_path, _INDEX_FILENAME)
        cloud_index_path = os.path.join(self.dir.url, _INDEX_FILENAME)
        # write to index.json file
        with open(index_file_path, "w") as f:
            data = {"chunks": chunks_info, "config": config, "updated_at": str(time())}
            json.dump(data, f, sort_keys=True)

        # upload file to cloud
        with open(index_file_path, "rb") as local_file, self.fs.open(cloud_index_path, "wb") as cloud_file:
            for chunk in iter(lambda: local_file.read(4096), b""):  # Read in 4KB chunks
                cloud_file.write(chunk)

        print(f"Index file written to: {cloud_index_path}")
        if os.path.exists(index_file_path):
            os.remove(index_file_path)


class HFParquetDir(ParquetDir):
    def __init__(
        self,
        dir_path: Optional[Union[str, Dir]],
        cache_path: Optional[str] = None,
        storage_options: Optional[Dict] = None,
        remove_after_indexing: bool = True,
        num_workers: int = 4,
    ):
        if not _HF_HUB_AVAILABLE:
            raise ModuleNotFoundError(
                "Support for Indexing HF depends on `huggingface_hub`.", "Please, run: `pip install huggingface_hub"
            )
        if not _PYARROW_AVAILABLE:
            raise ModuleNotFoundError("Please, run: `pip install pyarrow`")

        super().__init__(dir_path, cache_path, storage_options, remove_after_indexing, num_workers)

        assert self.dir.url is not None
        assert self.dir.url.startswith("hf")

        if self.cache_path is None:
            self.cache_path = default_cache_dir(self.dir.url)
            os.makedirs(self.cache_path, exist_ok=True)  # Ensure the directory exists

        # List all files and directories in the top-level of the specified directory
        from huggingface_hub import HfFileSystem

        self.fs = HfFileSystem()

        for _f in self.fs.ls(self.dir.url, detail=True):
            if _f["name"].endswith(".parquet"):
                self.files.append(_f)

    def task(self, _file: dict, order: int) -> None:
        """Extract metadata from a Parquet file on Hugging Face Hub without downloading the entire file.

        This method:
        1. Accesses only the footer section of the Parquet file, which contains schema information
        2. Parses the schema to extract data types and row count
        3. Passes the metadata to the processing queue

        Args:
            _file: Dictionary containing file metadata from HF filesystem
            order: Order of the file in the list
        """
        import pyarrow.parquet as pq

        # Validate inputs
        if not isinstance(_file, dict) or "name" not in _file or "size" not in _file:
            raise ValueError(f"Invalid file object: {_file}")

        assert self.cache_path is not None
        assert _file["name"].endswith(".parquet")

        file_name = os.path.basename(_file["name"])
        file_size = _file["size"]

        with self.fs.open(_file["name"], "rb") as f:
            # Read footer size (last 8 bytes: 4 for footer size + 4 for magic number)
            f.seek(file_size - 8)
            footer_size = int.from_bytes(f.read(4), "little")

            # seek to the start of the footer and read the footer data
            footer_start = file_size - footer_size - 8
            f.seek(footer_start)
            footer_data = f.read(file_size - footer_start)

            if len(footer_data) != file_size - footer_start:
                raise ValueError(f"Failed to read complete footer data from {file_name}")

        # Parse the footer data to extract schema information
        with io.BytesIO(footer_data) as footer_buffer:
            pq_file = pq.ParquetFile(footer_buffer)
            dtypes = [str(col.type) for col in pq_file.schema_arrow]
            num_rows = pq_file.metadata.num_rows

        file_object = {
            "chunk_size": num_rows,
            "chunk_bytes": file_size,
            "dtypes": dtypes,
        }

        # os.rename(temp_path, local_path)  # Atomic move after successful write
        self.process_queue.put_nowait((file_name, file_object, order))

    def write_index(self, chunks_info: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
        assert self.cache_path is not None
        assert self.dir.url is not None

        # clear any `.lock` or `.tmp` chunk file if left
        if self.is_delete_thread_running:
            for file in os.listdir(self.cache_path):
                if file != _INDEX_FILENAME:
                    with suppress(FileNotFoundError):
                        os.remove(file)

        index_file_path = os.path.join(self.cache_path, _INDEX_FILENAME)
        # write to index.json file
        with open(index_file_path, "w") as f:
            data = {"chunks": chunks_info, "config": config, "updated_at": str(time())}
            json.dump(data, f, sort_keys=True)

        print(f"Index file written to: {index_file_path}")


def get_parquet_indexer_cls(
    dir_path: str,
    cache_path: Optional[str] = None,
    storage_options: Optional[Dict] = {},
    remove_after_indexing: bool = True,
    num_workers: int = 4,
) -> ParquetDir:
    args = (dir_path, cache_path, storage_options, remove_after_indexing, num_workers)

    obj = parse.urlparse(dir_path)

    if obj.scheme in ("local", ""):
        return LocalParquetDir(*args)
    if obj.scheme in ("gs", "s3"):
        return CloudParquetDir(*args)
    if obj.scheme == "hf":
        return HFParquetDir(*args)

    if sys.platform == "win32":
        return LocalParquetDir(*args)

    raise ValueError(
        f"The provided `dir_path` {dir_path} doesn't have a ParquetDir class associated.",
        f"Found scheme => {obj.scheme}",
    )


def default_cache_dir(url: str) -> str:
    # Hash the URL using SHA256 and take the first 16 characters for brevity
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    cache_path = os.path.join(os.path.expanduser("~"), ".cache", "litdata-cache-index-pq", url_hash)
    os.makedirs(cache_path, exist_ok=True)  # Ensure the directory exists
    return cache_path
