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

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from litdata.constants import _INDEX_FILENAME
from litdata.streaming.compression import _COMPRESSORS, Compressor
from litdata.streaming.downloader import get_downloader_cls
from litdata.streaming.item_loader import BaseItemLoader, PyTreeLoader, TokensLoader
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.serializers import Serializer
from litdata.utilities._pytree import tree_unflatten, treespec_loads


class ChunksConfig:
    def __init__(
        self,
        cache_dir: str,
        serializers: Dict[str, Serializer],
        remote_dir: Optional[str],
        item_loader: Optional[BaseItemLoader] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
        region_of_interest: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        """The ChunksConfig reads the index files associated a chunked dataset and enables to map an index to its
        chunk.

        Arguments:
            cache_dir: The path to cache folder.
            serializers: The serializers used to serialize and deserialize the chunks.
            remote_dir: The path to a remote folder where the data are located.
                The scheme needs to be added to the path.
            chunks: The chunks that were read from `input_dir/index.json` file.
            region_of_interest: List of tuples of {start,end} of region of interest for each chunk.

        """
        self._cache_dir = cache_dir
        self._intervals: List[Tuple[int, int, int, int]] = []
        self._config = None
        self._chunks = chunks
        self._remote_dir = remote_dir
        self._item_loader = item_loader or PyTreeLoader()

        with open(os.path.join(self._cache_dir, _INDEX_FILENAME)) as f:
            data = json.load(f)
            _original_chunks = data["chunks"]
            self._config = data["config"]
            self._validate_item_loader()

            assert _original_chunks is not None

            if chunks is None:
                self._chunks = _original_chunks
            else:
                assert self._chunks is not None
                

        self._config["data_spec"] = treespec_loads(self._config["data_spec"])

        assert self._chunks is not None
        self._item_loader.setup(self._config, self._chunks, serializers, region_of_interest)
        self._intervals = self._item_loader.generate_intervals()
        self._length = self._intervals[-1][-1]
        self._downloader = None

        if remote_dir:
            self._downloader = get_downloader_cls(remote_dir, cache_dir, self._chunks)

        self._compressor_name = self._config["compression"]
        self._compressor: Optional[Compressor] = None

        if self._compressor_name in _COMPRESSORS:
            self._compressor = _COMPRESSORS[self._compressor_name]

        self._skip_chunk_indexes_deletion: Optional[List[int]] = None

    def can_delete(self, chunk_index: int) -> bool:
        if self._skip_chunk_indexes_deletion is None:
            return True
        return chunk_index not in self._skip_chunk_indexes_deletion

    @property
    def skip_chunk_indexes_deletion(self) -> Optional[List[int]]:
        return self._skip_chunk_indexes_deletion

    @skip_chunk_indexes_deletion.setter
    def skip_chunk_indexes_deletion(self, skip_chunk_indexes_deletion: List[int]) -> None:
        self._skip_chunk_indexes_deletion = skip_chunk_indexes_deletion

    def download_chunk_from_index(self, chunk_index: int) -> None:
        assert self._chunks is not None
        chunk_filename = self._chunks[chunk_index]["filename"]

        local_chunkpath = os.path.join(self._cache_dir, chunk_filename)

        if os.path.exists(local_chunkpath):
            self.try_decompress(local_chunkpath)
            return

        if self._downloader is None:
            return

        self._downloader.download_chunk_from_index(chunk_index)

        self.try_decompress(local_chunkpath)

    def try_decompress(self, local_chunkpath: str) -> None:
        if self._compressor is None:
            return

        target_local_chunkpath = local_chunkpath.replace(f".{self._compressor_name}", "")

        if os.path.exists(target_local_chunkpath):
            return

        with open(local_chunkpath, "rb") as f:
            data = f.read()

        # delete the files only if they were downloaded
        if self._downloader is not None:
            os.remove(local_chunkpath)

        data = self._compressor.decompress(data)

        with open(target_local_chunkpath, "wb") as f:
            f.write(data)

    @property
    def intervals(self) -> List[Tuple[int, int, int, int]]:
        if self._intervals is None:
            raise RuntimeError("The intervals should be defined.")
        return self._intervals

    @property
    def num_bytes(self) -> int:
        if self._config is None:
            raise RuntimeError("The config should be defined.")
        assert self._chunks is not None
        return sum(c["chunk_bytes"] for c in self._chunks)

    @property
    def data_format(self) -> Any:
        if self._config is None:
            raise RuntimeError("The config should be defined.")
        return self._config["data_format"]

    @property
    def data_format_unflattened(self) -> Any:
        if self._config is None:
            raise RuntimeError("The config should be defined.")
        return tree_unflatten(self._config["data_format"], self._config["data_spec"])

    @property
    def compression(self) -> Any:
        if self._config is None:
            raise RuntimeError("The config should be defined.")
        return self._config["compression"]

    @property
    def chunk_bytes(self) -> int:
        if self._config is None:
            raise RuntimeError("The config should be defined.")
        return self._config["chunk_bytes"]

    @property
    def config(self) -> Dict[str, Any]:
        if self._config is None:
            raise RuntimeError("The config should be defined.")
        return self._config

    def _get_chunk_index_from_index(self, index: int) -> int:
        for chunk_index, internal in enumerate(self._intervals):
            if internal[0] <= index < internal[-1]:
                return chunk_index
        raise ValueError(
            f"The provided index {index} didn't find a match within the chunk intervals {self._intervals}."
        )

    def __getitem__(self, index: ChunkedIndex) -> Tuple[str, int, int]:
        """Find the associated chunk metadata."""
        assert self._chunks is not None
        chunk = self._chunks[index.chunk_index]

        local_chunkpath = os.path.join(self._cache_dir, chunk["filename"])

        if self._compressor is not None:
            local_chunkpath = local_chunkpath.replace(f".{self._compressor_name}", "")

        begin = self._intervals[index.chunk_index][0]

        return local_chunkpath, begin, chunk["chunk_bytes"]

    def _get_chunk_index_from_filename(self, chunk_filename: str) -> int:
        """Retrieves the associated chunk_index for a given chunk filename."""
        assert self._chunks is not None
        for chunk_index, chunk in enumerate(self._chunks):
            if chunk["filename"] == chunk_filename:
                return chunk_index
        raise ValueError(f"The provided filename doesn't exist {chunk_filename}.")

    @classmethod
    def load(
        cls,
        cache_dir: str,
        serializers: Dict[str, Serializer],
        remote_dir: Optional[str] = None,
        item_loader: Optional[BaseItemLoader] = None,
        chunks: Optional[List[Any]] = None,
        region_of_interest: Optional[List[Tuple[int, int]]] = None,
    ) -> Optional["ChunksConfig"]:
        cache_index_filepath = os.path.join(cache_dir, _INDEX_FILENAME)

        if isinstance(remote_dir, str):
            downloader = get_downloader_cls(remote_dir, cache_dir, [])
            downloader.download_file(os.path.join(remote_dir, _INDEX_FILENAME), cache_index_filepath)

        if not os.path.exists(cache_index_filepath):
            return None

        return ChunksConfig(cache_dir, serializers, remote_dir, item_loader, chunks, region_of_interest)

    def __len__(self) -> int:
        return self._length

    def _validate_item_loader(self) -> None:
        assert self._config
        if (
            len(self._config["data_format"]) == 1
            and self._config["data_format"][0].startswith("no_header_tensor")
            and not isinstance(self._item_loader, TokensLoader)
        ):
            raise ValueError("Please, use Cache(..., item_loader=TokensLoader(block_size=...))")
