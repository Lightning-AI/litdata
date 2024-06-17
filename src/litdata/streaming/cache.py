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

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from litdata.constants import (
    _INDEX_FILENAME,
)
from litdata.streaming.reader import BinaryReader
from litdata.streaming.resolver import Dir, _resolve_dir
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.serializers import Serializer
from litdata.streaming.writer import BinaryWriter
from litdata.utilities.env import _DistributedEnv, _WorkerEnv
from litdata.utilities.format import _convert_bytes_to_int
from litdata.streaming.item_loader import BaseItemLoader, Interval

logger = logging.Logger(__name__)


class Cache:
    def __init__(
        self,
        input_dir: Optional[Union[str, Dir]],
        subsampled_files: Optional[List[str]] = None,
        region_of_interest: Optional[List[Tuple[int, int]]] = None,
        compression: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_bytes: Optional[Union[int, str]] = None,
        item_loader: Optional[BaseItemLoader] = None,
        max_cache_size: Union[int, str] = "100GB",
        serializers: Optional[Dict[str, Serializer]] = None,
    ):
        """The Cache enables to optimise dataset format for cloud training. This is done by grouping several elements
        together in order to accelerate fetching.

        Arguments:
            input_dir: The path to where the chunks will be or are stored.
            subsampled_files: List of subsampled chunk files loaded from `input_dir/index.json` file.
            region_of_interest: List of tuples of (start,end) of region of interest for each chunk.
            compression: The name of the algorithm to reduce the size of the chunks.
            chunk_bytes: The maximum number of bytes within a chunk.
            chunk_size: The maximum number of items within a chunk.
            item_loader: The object responsible to generate the chunk intervals and load an item froma chunk.
            max_cache_size: The maximum cache size used by the reader when fetching the chunks.
            serializers: Provide your own serializers.

        """
        super().__init__()
        input_dir = _resolve_dir(input_dir)
        self._cache_dir = input_dir.path
        assert self._cache_dir
        self._writer = BinaryWriter(
            self._cache_dir,
            chunk_size=chunk_size,
            chunk_bytes=chunk_bytes,
            compression=compression,
            serializers=serializers,
        )
        self._reader = BinaryReader(
            self._cache_dir,
            subsampled_files=subsampled_files,
            region_of_interest=region_of_interest,
            max_cache_size=_convert_bytes_to_int(max_cache_size) if isinstance(max_cache_size, str) else max_cache_size,
            remote_input_dir=input_dir.url,
            compression=compression,
            item_loader=item_loader,
            serializers=serializers,
        )
        self._is_done = False
        self._distributed_env = _DistributedEnv.detect()
        self._rank: Optional[int] = None

    @property
    def rank(self) -> int:
        """Returns the rank of the Cache."""
        if self._rank is None:
            self._worker_env = _WorkerEnv.detect()
            self._rank = self._distributed_env.global_rank * self._worker_env.world_size + self._worker_env.rank
        return self._rank

    @property
    def filled(self) -> bool:
        """Returns whether the caching phase is done."""
        if self._is_done:
            return True
        assert self._cache_dir
        self._is_done = os.path.exists(os.path.join(self._cache_dir, _INDEX_FILENAME))
        return self._is_done

    @property
    def cache_dir(self) -> str:
        assert self._cache_dir
        return self._cache_dir

    @property
    def checkpoint_dir(self) -> str:
        assert self._cache_dir
        checkpoint_dir = os.path.join(self._cache_dir, "checkpoints")
        return self._try_create(checkpoint_dir)

    def _try_create(self, path: str) -> str:
        os.makedirs(path, exist_ok=True)
        return path

    def __setitem__(self, index: int, data: Any) -> None:
        """Store an item in the writer."""
        self._writer[index] = data

    def _add_item(self, index: int, data: Any) -> Optional[str]:
        """Store an item in the writer and optionally return the chunk path."""
        return self._writer.add_item(index, data)

    def __getitem__(self, index: Union[int, ChunkedIndex]) -> Dict[str, Any]:
        """Read an item in the reader."""
        if isinstance(index, int):
            index = ChunkedIndex(index, self._get_chunk_index_from_index(index))
        return self._reader.read(index)

    def done(self) -> Optional[List[str]]:
        """Inform the writer the chunking phase is finished."""
        return self._writer.done()

    def merge(self, num_workers: int = 1, node_rank: Optional[int] = None) -> None:
        """Inform the writer the chunking phase is finished."""
        self._writer.merge(num_workers, node_rank=node_rank)

    def _merge_no_wait(self, node_rank: Optional[int] = None) -> None:
        """Inform the writer the chunking phase is finished."""
        self._writer._merge_no_wait(node_rank=node_rank)

    def __len__(self) -> int:
        return self._reader.get_length()

    def get_chunk_intervals(self) -> List[Interval]:
        return self._reader.get_chunk_intervals()

    def _get_chunk_index_from_index(self, index: int) -> int:
        return self._reader._get_chunk_index_from_index(index)
