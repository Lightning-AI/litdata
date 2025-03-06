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
import logging
import os
import time
import warnings
from contextlib import suppress
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Dict, List, Optional, Tuple, Union

from filelock import FileLock, Timeout

from litdata.constants import _DEBUG, _USE_RUST_IMPLEMENTATION
from litdata.streaming.config import ChunksConfig, Interval
from litdata.streaming.item_loader import BaseItemLoader, PyTreeLoader, TokensLoader
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.serializers import Serializer, _get_serializers
from litdata.utilities.encryption import Encryption
from litdata.utilities.env import _DistributedEnv, _WorkerEnv

warnings.filterwarnings("ignore", message=".*The given buffer is not writable.*")


logger = logging.getLogger(__name__)


_END_TOKEN = "END"  # noqa: S105

# Note: The timeout here should not be too short. We need to prevent the caller from aggressively
# querying the queue and consuming too many CPU cycles.
_DEFAULT_TIMEOUT = 0.1
_LONG_DEFAULT_TIMEOUT = 5


class PrepareChunksThread(Thread):
    """This thread is responsible to download the chunks associated to a given worker."""

    def __init__(
        self,
        config: ChunksConfig,
        item_loader: BaseItemLoader,
        distributed_env: _DistributedEnv,
        max_cache_size: Optional[int] = None,
        max_pre_download: int = 2,
        rank: Optional[int] = None,
    ) -> None:
        super().__init__(daemon=True)
        print(f"using rust implementation (StreamingChunksDownloader): {_USE_RUST_IMPLEMENTATION}")
        self._config = config
        self._item_loader = item_loader
        self._max_pre_download = max_pre_download
        self._pre_download_counter = 0
        self._distributed_env = distributed_env

        self._chunks_index_to_be_deleted: List[int] = []
        self._max_cache_size = max_cache_size
        self._parent_cache_dir = os.path.dirname(self._config._cache_dir)
        self._to_download_queue: Queue = Queue()
        self._to_delete_queue: Queue = Queue()
        self._force_stop_event = Event()

        # TODO: Find a real fix to this problem
        self._force_download_queue: Queue = Queue()

        self._rank = rank

        # Check whether a dataset slice fits on the node
        num_bytes_per_nodes = self._config.num_bytes // self._distributed_env.num_nodes
        self._delete_chunks_when_processed = num_bytes_per_nodes > max_cache_size if max_cache_size else False
        self._has_exited = False

    def download(self, chunk_indexes: List[int]) -> None:
        """Receive the list of the chunk indices to download for the current epoch."""
        for chunk_index in chunk_indexes:
            self._to_download_queue.put(chunk_index)

    def delete(self, chunk_indexes: List[int]) -> None:
        """Receive the list of the chunk indices to delete for the current epoch."""
        for chunk_index in chunk_indexes:
            self._to_delete_queue.put(chunk_index)

    def _remaining_locks(self, chunkpath: str) -> int:
        countpath = chunkpath + ".cnt"
        if not os.path.exists(countpath):
            return 0
        with open(countpath) as count_f:
            try:
                return int(count_f.read().strip())
            except Exception:
                return 1

    def _decrement_local_lock(self, chunk_index: int) -> int:
        """Remove a count from the local lock, return the remaining count."""
        chunk_filepath, _, _ = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]

        countpath = chunk_filepath + ".cnt"
        with suppress(Timeout), FileLock(countpath + ".lock", timeout=3):
            if not os.path.exists(countpath):
                return 0
            with open(countpath) as count_f:
                try:
                    curr_count = int(count_f.read().strip())
                except Exception:
                    curr_count = 1
            curr_count -= 1
            if curr_count <= 0:
                with contextlib.suppress(FileNotFoundError, PermissionError):
                    os.remove(countpath)

                with contextlib.suppress(FileNotFoundError, PermissionError):
                    os.remove(countpath + ".lock")
            else:
                with open(countpath, "w+") as count_f:
                    count_f.write(str(curr_count))
            return curr_count
        return 0

    def _apply_delete(self, chunk_index: int) -> None:
        """Inform the item loader of the chunk to delete."""
        # TODO: Fix the can_delete method
        can_delete_chunk = self._config.can_delete(chunk_index)
        chunk_filepath, _, _ = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]

        remaining_locks = self._remaining_locks(chunk_filepath)
        if remaining_locks > 0:  # Can't delete this, something has it
            if _DEBUG:
                print(f"Skip delete {chunk_filepath} by {self._rank or 0}, current lock count: {remaining_locks}")
            return

        self._item_loader.delete(chunk_index, chunk_filepath)

        if _DEBUG:
            print(f"Deleted {chunk_filepath} by {self._rank or 0}. Debug: {can_delete_chunk}")

        for lock_extension in [".lock", ".cnt.lock"]:
            try:
                locak_chunk_path = chunk_filepath + lock_extension
                if os.path.exists(locak_chunk_path):
                    os.remove(locak_chunk_path)
            except FileNotFoundError:
                pass

    def stop(self) -> None:
        """Receive the list of the chunk indices to download for the current epoch."""
        self._to_download_queue.put(_END_TOKEN)

    def force_stop(self) -> None:
        self._force_stop_event.set()

    def _maybe_delete_chunks(self) -> None:
        reached_pre_download = self._pre_download_counter == self._max_pre_download

        # we have already pre-downloaded some chunks, we just need to wait for them to be processed.
        chunk_index = _get_from_queue(
            self._to_delete_queue, timeout=_LONG_DEFAULT_TIMEOUT if reached_pre_download else _DEFAULT_TIMEOUT
        )

        if chunk_index is not None:
            self._pre_download_counter -= 1

            # Store the current chunk index
            self._chunks_index_to_be_deleted.append(chunk_index)

        # Get the current cache size and decide whether we need to start cleanup. Otherwise, keep track of it
        while self._max_cache_size and self._chunks_index_to_be_deleted and self._can_delete_chunk():
            # Delete the oldest chunk
            self._apply_delete(self._chunks_index_to_be_deleted.pop(0))

        return

    def _can_delete_chunk(self) -> bool:
        if self._delete_chunks_when_processed:
            return self._pre_download_counter >= self._max_pre_download - 1
        return (
            self._max_cache_size is not None
            and _get_folder_size(self._config._cache_dir, self._config) >= self._max_cache_size
        )

    def _pre_load_chunk(self, chunk_index: int) -> None:
        chunk_filepath, _, _ = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]
        self._item_loader.pre_load_chunk(chunk_index, chunk_filepath)

    def _force_download(self) -> None:
        chunk_index = _get_from_queue(self._force_download_queue)
        if chunk_index is not None:
            if _DEBUG:
                chunk_filepath, _, _ = self._config[ChunkedIndex(index=-1, chunk_index=chunk_index)]
                print(f"Requested force download for {chunk_filepath} by {self._rank}")

            self._config.download_chunk_from_index(chunk_index)

            # Preload item if possible to gain some time but only
            # if this is one of the pre-downloaded chunk
            if self._pre_download_counter > 0:
                self._pre_load_chunk(chunk_index)

            # Avoid downloading too many chunks in advance at the risk of over using the disk space
            self._pre_download_counter += 1

    def run(self) -> None:
        while True:
            if self._force_stop_event.is_set():
                self._has_exited = True
                return

            self._force_download()

            if self._pre_download_counter < self._max_pre_download:
                chunk_index = _get_from_queue(self._to_download_queue)
                if chunk_index == _END_TOKEN:
                    self._has_exited = True
                    return

                if chunk_index is not None:
                    self._config.download_chunk_from_index(chunk_index)

                    # Preload item if possible to gain some time but only
                    # if this is one of the pre-downloaded chunk
                    if self._pre_download_counter > 0:
                        self._pre_load_chunk(chunk_index)

                    # Avoid downloading too many chunks in advance at the risk of over using the disk space
                    self._pre_download_counter += 1

            if self._max_cache_size:
                self._maybe_delete_chunks()


class StreamingChunksDownloader(Thread):
    def __init__(self, config: ChunksConfig, start: bool = False) -> None:
        super().__init__(daemon=True)
        print(f"using rust implementation (StreamingChunksDownloader): {_USE_RUST_IMPLEMENTATION}")
        self._config = config
        self.download_next_k_items_queue: Queue = Queue()
        self._meow_meow = 0
        if _USE_RUST_IMPLEMENTATION and start:
            # print("going to call on_start method of rust")
            pre_downloaded_items = self._config.streaming_data_provider.on_start()
            for item in pre_downloaded_items:
                epoch, chunk_index, sample_index, data = item
                self._config.set_index_to_sample_data(epoch, chunk_index, sample_index, bytes(data))

    def trigger_to_download_next_k_items(self) -> None:
        """Receive the list of the chunk indices to download for the current epoch."""
        # queue fills with multiple True values causing program to crash
        # get_no_wait from queue and increase the counter
        try:
            curr_download_count = self.download_next_k_items_queue.get_nowait()
            self.download_next_k_items_queue.put(curr_download_count + 1)
        except Empty:
            self.download_next_k_items_queue.put(1)
        except Exception as e:
            print(f"Error in trigger_to_download_next_k_items: {e}")

    def run(self) -> None:
        while True:
            # print("meow meow")
            # queue_item_count = self.download_next_k_items_queue.get()
            # if queue_item_count is None:
            #     self._has_exited = True
            #     return
            # for _ in range(queue_item_count):
            k_items = self._config.streaming_data_provider.get_next_k_item()
            for item in k_items:
                epoch, chunk_index, sample_index, data = item
            self._config.set_index_to_sample_data(epoch, chunk_index, sample_index, bytes(data))


# The BinaryReader operates as the inverse of the data optimization process:
# 1. Loads raw bytes from chunks based on specific indices
# 2. Uses deserializers to convert bytes back into Python objects
# 3. Reconstructs the original data structure with the data_spec from index.json and using `tree_unflatten function`
# 4. Supports features like compression, encryption, and distributed reading
class BinaryReader:
    def __init__(
        self,
        cache_dir: str,
        subsampled_files: Optional[List[str]] = None,
        region_of_interest: Optional[List[Tuple[int, int]]] = None,
        max_cache_size: Optional[Union[int, str]] = None,
        remote_input_dir: Optional[str] = None,
        compression: Optional[str] = None,
        encryption: Optional[Encryption] = None,
        item_loader: Optional[BaseItemLoader] = None,
        serializers: Optional[Dict[str, Serializer]] = None,
        storage_options: Optional[dict] = {},
        max_pre_download: int = 2,
        epoch: Optional[int] = None,
        config: Optional[ChunksConfig] = None,
        on_start_pre_item_download_count: int = 100,
        get_next_k_item_count: int = 10,
    ) -> None:
        """The BinaryReader enables to read chunked dataset in an efficient way.

        Args:
            cache_dir: The path to cache folder.
            subsampled_files: List of subsampled chunk files loaded from `input_dir/index.json` file.
            region_of_interest: List of tuples of {start,end} of region of interest for each chunk.
            remote_input_dir: The path to a remote folder where the data are located.
                The scheme needs to be added to the path.
            compression: The algorithm to decompress the chunks.
            encryption: The algorithm to decrypt the chunks or samples.
            item_loader: The chunk sampler to create sub arrays from a chunk.
            max_cache_size: The maximum cache size used by the reader when fetching the chunks.
            serializers: Provide your own serializers.
            storage_options: Additional connection options for accessing storage services.
            max_pre_download: Maximum number of chunks that can be pre-downloaded by the reader.
            epoch: The epoch number.
            config: The config object.
            on_start_pre_item_download_count: The number of items to download on start.
            get_next_k_item_count: The number of items to download on get_next_k_item.
        """
        super().__init__()
        warnings.filterwarnings("ignore", message=".*The given buffer is not writable.*")

        self._cache_dir = cache_dir
        self._remote_input_dir = remote_input_dir

        if not os.path.exists(self._cache_dir):
            raise FileNotFoundError(f"The provided cache_dir `{self._cache_dir}` doesn't exist.")

        self._compression = compression
        self._encryption = encryption
        self._intervals: Optional[List[str]] = None
        self.subsampled_files = subsampled_files
        self.region_of_interest = region_of_interest
        self._serializers: Dict[str, Serializer] = _get_serializers(serializers)
        self._distributed_env = _DistributedEnv.detect()
        self._rank: Optional[int] = None
        self._config: Optional[ChunksConfig] = config
        self._streaming_chunks_downloader: Optional[StreamingChunksDownloader] = None
        self._prepare_thread: Optional[PrepareChunksThread] = None
        self._item_loader = item_loader or PyTreeLoader()
        self._last_chunk_index: Optional[int] = None
        self._max_cache_size = int(os.getenv("MAX_CACHE_SIZE", max_cache_size or 0))
        self._storage_options = storage_options
        self._max_pre_download = max_pre_download
        self._epoch = epoch
        self.should_start_streaming_chunks_downloader = config is None
        # print(f"reader got config: {config}")
        self.last_read_key: Optional[str] = None
        self.on_start_pre_item_download_count = on_start_pre_item_download_count
        self.get_next_k_item_count = get_next_k_item_count

    def _get_chunk_index_from_index(self, index: int) -> Tuple[int, int]:
        # Load the config containing the index
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        return self._config._get_chunk_index_from_index(index)  # type: ignore

    def _try_load_config(self) -> Optional[ChunksConfig]:
        """Try to load the chunks config if the index files are available."""
        if self._config is None:
            self._config = ChunksConfig.load(
                self._cache_dir,
                self._serializers,
                self._remote_input_dir,
                self._item_loader,
                self.subsampled_files,
                self.region_of_interest,
                self._storage_options,
                self._epoch,
                self.on_start_pre_item_download_count,
                self.get_next_k_item_count,
            )
        return self._config

    @property
    def config(self) -> ChunksConfig:
        if self._config is None:
            raise RuntimeError("The config should be defined.")
        return self._config

    @property
    def rank(self) -> int:
        """Returns the rank of the writer."""
        if self._rank is None:
            self._worker_env = _WorkerEnv.detect()
            self._rank = self._distributed_env.global_rank * self._worker_env.world_size + self._worker_env.rank
        return self._rank

    def read(self, index: ChunkedIndex) -> Any:
        """Read an item for the given from a chunk.

        If the chunk isn't available locally or in memory, it will be downloaded.

        Prefetching should reduce the wait time to be the batch available.

        """
        assert self._config is not None

        if not isinstance(index, ChunkedIndex):
            raise ValueError("The Reader.read(...) method expects a chunked Index.")

        # Load the config containing the index
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        if self._config and (self._config._remote_dir or self._config._compressor) and not _USE_RUST_IMPLEMENTATION:
            # Create and start the prepare chunks thread
            if self._prepare_thread is None and self._config:
                self._prepare_thread = PrepareChunksThread(
                    self._config,
                    self._item_loader,
                    self._distributed_env,
                    self._max_cache_size,
                    self._max_pre_download,
                    self._rank,
                )
                # Attach the force download queue
                self._item_loader._force_download_queue = self._prepare_thread._force_download_queue  # type: ignore
                self._prepare_thread.start()
                if index.chunk_indexes:
                    self._prepare_thread.download(index.chunk_indexes)

            # If the chunk_index is new, request for it to be downloaded.
            if index.chunk_index != self._last_chunk_index:
                assert self._prepare_thread
                self._prepare_thread.download([index.chunk_index])

            if self._last_chunk_index is None:
                self._last_chunk_index = index.chunk_index
        if _USE_RUST_IMPLEMENTATION and self._streaming_chunks_downloader is None:
            # print(f"{self.should_start_streaming_chunks_downloader}")
            self._streaming_chunks_downloader = StreamingChunksDownloader(
                self._config, start=self.should_start_streaming_chunks_downloader
            )
            self._streaming_chunks_downloader.start()

        # Fetch the element
        if _USE_RUST_IMPLEMENTATION:
            with contextlib.suppress(Exception):
                assert self.last_read_key is not None  # even if it fails, contexlib will suppress the exception
                del self._config.index_to_sample_data[self.last_read_key]
            key = self._config.get_key_from_index(self._epoch or 1, index.chunk_index, index.index)
            limit = 100  # wait for 600*0.1 = 60 seconds before raising an error
            while limit > 0:
                if key in self._config.index_to_sample_data:
                    break
                limit -= 1
                time.sleep(0.1)
            if limit == 0:
                raise Exception(
                    f"The item with key: {key} is not available.",
                    f"Current stored items: {self._config.index_to_sample_data.keys()}",
                )
            item = self._config.index_to_sample_data[key]
            self.last_read_key = key
            assert self._streaming_chunks_downloader is not None
            # self._streaming_chunks_downloader.trigger_to_download_next_k_items()
        else:
            chunk_filepath, begin, filesize_bytes = self.config[index]

            if isinstance(self._item_loader, PyTreeLoader):
                item = self._item_loader.load_item_from_chunk(
                    index.index, index.chunk_index, chunk_filepath, begin, filesize_bytes, self._encryption
                )
            else:
                item = self._item_loader.load_item_from_chunk(
                    index.index, index.chunk_index, chunk_filepath, begin, filesize_bytes
                )

            # We need to request deletion after the latest element has been loaded.
            # Otherwise, this could trigger segmentation fault error depending on the item loader used.
            if (
                self._config
                and (self._config._remote_dir or self._config._compressor)
                and index.chunk_index != self._last_chunk_index
            ):
                assert self._prepare_thread
                assert self._last_chunk_index is not None

                # inform the chunk has been completely consumed
                self._prepare_thread._decrement_local_lock(self._last_chunk_index)
                self._prepare_thread.delete([self._last_chunk_index])

            if index.chunk_index != self._last_chunk_index:
                # Close the memory-mapped file for the last chunk index
                if isinstance(self._item_loader, TokensLoader) and self._last_chunk_index is not None:
                    self._item_loader.close(self._last_chunk_index)

                # track the new chunk index as the latest one
                self._last_chunk_index = index.chunk_index

            if index.is_last_index and self._prepare_thread:
                # inform the thread it is time to stop
                self._prepare_thread._decrement_local_lock(index.chunk_index)
                self._prepare_thread.stop()
                self._prepare_thread = None
        if index.is_last_index and self._streaming_chunks_downloader:
            # inform the thread it is time to stop
            self._streaming_chunks_downloader.stop()
            self._streaming_chunks_downloader = None

        return item

    def get_length(self) -> int:
        """Get the number of samples across all chunks."""
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        return len(self.config)

    def get_chunk_intervals(self) -> List[Interval]:
        """Get the index interval of each chunk."""
        if self._config is None and self._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        return self.config.intervals

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch number."""
        self._epoch = epoch
        if self._config:
            self._config.set_epoch(epoch)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_prepare_thread"] = None
        return state

    def __del__(self) -> None:
        if self._prepare_thread and not self._prepare_thread._has_exited:
            self._prepare_thread.force_stop()
            self._prepare_thread = None


def _get_folder_size(path: str, config: ChunksConfig) -> int:
    """Collect the size of each files within a folder.

    This method is robust to file deletion races

    """
    size = 0
    for filename in os.listdir(path):
        if filename in config.filename_to_size_map:
            with contextlib.suppress(FileNotFoundError):
                size += config.filename_to_size_map[filename]
        elif ".bin" in filename:  # for temporary files
            with contextlib.suppress(FileNotFoundError):
                size += os.path.getsize(filename)
        elif not filename.endswith((".cnt", ".lock", ".json", ".zstd.bin", ".tmp")):
            # ignore .cnt, .lock, .json and .zstd files for warning
            logger.warning(
                f"Skipping {filename}: Not a valid chunk file. It will be excluded from cache size calculation."
            )
    return size


def _get_from_queue(queue: Queue, timeout: float = _DEFAULT_TIMEOUT) -> Optional[Any]:
    try:
        return queue.get(timeout=timeout)
    except Empty:
        pass
    except OSError as err:
        # handle closed queue before the thread terminates
        if "handle is closed" in str(err) or "Bad file descriptor" in str(err):
            logger.debug(err)
        else:
            raise err
    except EOFError as err:
        logger.debug(err)
    return None
