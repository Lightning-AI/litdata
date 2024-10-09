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

import functools
import os
from abc import ABC, abstractmethod
from collections import namedtuple
from copy import deepcopy
from io import BytesIO, FileIO
from time import sleep
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from litdata.constants import (
    _TORCH_DTYPES_MAPPING,
)
from litdata.streaming.serializers import Serializer
from litdata.utilities._pytree import PyTree, tree_unflatten
from litdata.utilities.encryption import Encryption, EncryptionLevel

Interval = namedtuple("Interval", ["chunk_start", "roi_start_idx", "roi_end_idx", "chunk_end"])


class BaseItemLoader(ABC):
    """The base item loader is responsible to decide how the items within a chunk are loaded."""

    def setup(
        self,
        config: Dict,
        chunks: List,
        serializers: Dict[str, Serializer],
        region_of_interest: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self._config = config
        self._chunks = chunks
        self._serializers = {**serializers}
        self._data_format = self._config["data_format"]
        self._shift_idx = len(self._data_format) * 4
        self.region_of_interest = region_of_interest

        # setup the serializers on restart
        for data_format in self._data_format:
            serializer = deepcopy(self._serializers[self._data_format_to_key(data_format)])
            serializer.setup(data_format)
            self._serializers[data_format] = serializer

    @functools.lru_cache(maxsize=128)
    def _data_format_to_key(self, data_format: str) -> str:
        if ":" in data_format:
            serialier, serializer_sub_type = data_format.split(":")
            if serializer_sub_type in self._serializers:
                return serializer_sub_type
            return serialier
        return data_format

    def state_dict(self) -> Dict:
        return {}

    @abstractmethod
    def generate_intervals(self) -> List[Interval]:
        """Returns a list of intervals.

        The structure is: [chunk_start, region_of_interest_start, region_of_interest_end, chunk_end]

        region_of_interest: indicates the indexes a chunk our StreamingDataset is allowed to read.
        """

    @abstractmethod
    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        """Logic to load the chunk in background to gain some time."""

    @abstractmethod
    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        chunk_bytes: int,
    ) -> Any:
        """Returns an item loaded from a chunk."""

    @abstractmethod
    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        """Delete a chunk from the local filesystem."""

    @abstractmethod
    def encode_data(self, data: List[bytes], sizes: List[int], flattened: List[Any]) -> Any:
        pass


class PyTreeLoader(BaseItemLoader):
    """The Pytree Loader is the default loader of the Cache object."""

    def __init__(self) -> None:
        self._chunk_filepaths: Dict[str, bool] = {}
        self._decrypted_chunks: Dict[int, bytes] = {}

    def generate_intervals(self) -> List[Interval]:
        intervals = []
        begin = 0
        end = 0
        for idx, curr_chunk in enumerate(self._chunks):
            end += curr_chunk["chunk_size"]
            start_idx, end_idx = begin, end
            if self.region_of_interest is not None:
                start_idx = begin + self.region_of_interest[idx][0]
                end_idx = begin + self.region_of_interest[idx][1]

            intervals.append(Interval(begin, start_idx, end_idx, end))
            begin += curr_chunk["chunk_size"]
        return intervals

    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        pass

    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        chunk_bytes: int,
        encryption: Optional[Encryption] = None,
    ) -> bytes:
        offset = (1 + (index - begin) if index >= begin else index + 1) * 4

        if chunk_filepath in self._chunk_filepaths and not os.path.isfile(chunk_filepath):
            del self._chunk_filepaths[chunk_filepath]

        if chunk_filepath not in self._chunk_filepaths:
            exists = os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size >= chunk_bytes

            while not exists:
                sleep(0.1)
                exists = os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size >= chunk_bytes

            self._chunk_filepaths[chunk_filepath] = True

        if self._config.get("encryption"):
            data = self._load_encrypted_data(chunk_filepath, chunk_index, offset, encryption)
        else:
            with open(chunk_filepath, "rb", 0) as fp:
                data = self._load_data(fp, offset)

        # check for mosaic mds format
        if "format" in self._config and self._config["format"] == "mds":
            return self.mds_deserialize(data, chunk_index)
        return self.deserialize(data)

    def _load_encrypted_data(
        self, chunk_filepath: str, chunk_index: int, offset: int, encryption: Optional[Encryption]
    ) -> bytes:
        """Load and decrypt data from chunk based on the encryption configuration."""
        # Validate the provided encryption object against the expected configuration.
        self._validate_encryption(encryption)

        # chunk-level decryption
        if self._config["encryption"]["level"] == EncryptionLevel.CHUNK:
            decrypted_data = self._decrypted_chunks.get(chunk_index, None)
            if decrypted_data is None:
                with open(chunk_filepath, "rb", 0) as fp:
                    encrypted_data = fp.read()
                    decrypted_data = encryption.decrypt(encrypted_data)  # type: ignore
                    # Store the decrypted chunk to avoid re-decryption,
                    # also allows to free the previous chunk from the memory
                    self._decrypted_chunks = {chunk_index: decrypted_data}
            data = self._load_data(BytesIO(decrypted_data), offset)

        # sample-level decryption
        elif self._config["encryption"]["level"] == EncryptionLevel.SAMPLE:
            with open(chunk_filepath, "rb", 0) as fp:
                data = self._load_data(fp, offset)
                data = encryption.decrypt(data)  # type: ignore

        else:
            raise ValueError("Invalid encryption level.")

        return data

    def _load_data(self, fp: Union[FileIO, BytesIO], offset: int) -> bytes:
        """Load the data from the file pointer."""
        fp.seek(offset)
        pair = fp.read(8)
        begin, end = np.frombuffer(pair, np.uint32)
        fp.seek(begin)
        return fp.read(end - begin)

    def mds_deserialize(self, raw_item_data: bytes, chunk_index: int) -> "PyTree":
        """Deserialize the mds raw bytes into their python equivalent."""
        idx = 0
        sizes = []
        column_sizes = self._chunks[chunk_index]["column_sizes"]
        # adapted from: MDSReader.deserialize : https://github.com/mosaicml/streaming/blob/main/streaming/base/format/mds/reader.py
        for size in column_sizes:
            if size:
                sizes.append(size)
            else:
                (size,) = np.frombuffer(raw_item_data[idx : idx + 4], np.uint32)
                sizes.append(size)
                idx += 4
        data = []
        for size, data_format in zip(sizes, self._data_format):
            serializer = self._serializers[data_format]
            data_bytes = raw_item_data[idx : idx + size]
            data.append(serializer.deserialize(data_bytes))
            idx += size
        return tree_unflatten(data, self._config["data_spec"])

    def deserialize(self, raw_item_data: bytes) -> "PyTree":
        """Deserialize the raw bytes into their python equivalent."""
        idx = self._shift_idx
        sizes = np.frombuffer(raw_item_data[:idx], np.uint32)
        data = []
        for size, data_format in zip(sizes, self._data_format):
            serializer = self._serializers[data_format]
            data_bytes = raw_item_data[idx : idx + size]
            data.append(serializer.deserialize(data_bytes))
            idx += size
        return tree_unflatten(data, self._config["data_spec"])

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        if os.path.exists(chunk_filepath):
            os.remove(chunk_filepath)

    def _validate_encryption(self, encryption: Optional[Encryption]) -> None:
        """Validate the encryption object."""
        if not encryption:
            raise ValueError("Data is encrypted but no encryption object was provided.")
        if encryption.algorithm != self._config["encryption"]["algorithm"]:
            raise ValueError("Encryption algorithm mismatch.")
        if encryption.level != self._config["encryption"]["level"]:
            raise ValueError("Encryption level mismatch.")

    @classmethod
    def encode_data(cls, data: List[bytes], sizes: List[int], flattened: List[Any]) -> Tuple[bytes, Optional[int]]:
        # Concatenante into a single byte array
        head = np.array(sizes, np.uint32).tobytes()
        body = b"".join(data)
        return head + body, None


class TokensLoader(BaseItemLoader):
    def __init__(self, block_size: Optional[int] = None):
        """The Tokens Loader is an optimizer item loader for NLP.

        Args:
            block_size: The context length to use during training.

        """
        super().__init__()
        self._block_size = block_size
        self._mmaps: Dict[int, np.memmap] = {}
        self._buffers: Dict[int, bytes] = {}
        self._dtype: Optional[torch.dtype] = None
        self._chunk_filepaths: Dict[str, bool] = {}

    def state_dict(self) -> Dict:
        assert self._block_size
        return {
            "block_size": self._block_size,
        }

    def setup(
        self,
        config: Dict,
        chunks: List,
        serializers: Dict[str, Serializer],
        region_of_interest: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        super().setup(config, chunks, serializers, region_of_interest)
        self._dtype = _TORCH_DTYPES_MAPPING[int(config["data_format"][0].split(":")[1])]
        if all(chunk["dim"] is None for chunk in self._chunks):
            raise ValueError("The provided chunks isn't properly setup.")

    def generate_intervals(self) -> List[Interval]:
        assert self._block_size
        intervals = []
        begin = 0
        end = 0
        for idx, chunk in enumerate(self._chunks):
            dim = chunk["dim"]
            num_blocks = dim // self._block_size
            end += num_blocks
            start_idx, end_idx = begin, end
            if self.region_of_interest is not None:
                start_idx = begin + self.region_of_interest[idx][0]
                end_idx = begin + self.region_of_interest[idx][1]
            intervals.append(Interval(begin, start_idx, end_idx, end))
            begin += num_blocks
        return intervals

    def _load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        if chunk_index in self._mmaps:
            return
        chunk = self._chunks[chunk_index]

        # Skip the header
        # The number of items + the number of offsets (number of items in the chunk + 1)
        # multiplied by the header encoding dtype (np.uint32)
        offset = (1 + chunk["chunk_size"] + 1) * 4
        mmap = np.memmap(chunk_filepath, mode="r", order="C", offset=offset)
        self._mmaps[chunk_index] = mmap
        self._buffers[chunk_index] = memoryview(mmap)  

    def pre_load_chunk(self, chunk_index: int, chunk_filepath: str) -> None:
        # This is called within the prepare chunks thread, so we overlap data loading with data reading.
        if chunk_filepath not in self._chunk_filepaths:
            self._chunk_filepaths[chunk_filepath] = True

        if os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size > 0:
            self._load_chunk(chunk_index, chunk_filepath)

    def load_item_from_chunk(
        self,
        index: int,
        chunk_index: int,
        chunk_filepath: str,
        begin: int,
        chunk_bytes: int,
    ) -> torch.Tensor:
        assert self._block_size

        if chunk_filepath in self._chunk_filepaths and not os.path.isfile(chunk_filepath):
            del self._chunk_filepaths[chunk_filepath]

        if chunk_filepath not in self._chunk_filepaths:
            exists = os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size > 0

            while not exists:
                sleep(0.1)
                exists = os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size > 0

            self._chunk_filepaths[chunk_filepath] = True

        self._load_chunk(chunk_index, chunk_filepath)
        assert self._dtype

        buffer: bytes = self._buffers[chunk_index]
        offset = self._dtype.itemsize * (index - begin) * self._block_size
        return torch.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)

    def delete(self, chunk_index: int, chunk_filepath: str) -> None:
        if os.path.exists(chunk_filepath):
            if chunk_index in self._buffers:
                del self._buffers[chunk_index]
            if chunk_index in self._mmaps:
                del self._mmaps[chunk_index]
            os.remove(chunk_filepath)

    def close(self, chunk_index: int) -> None:
        """Release the memory-mapped file for a specific chunk index."""
        if chunk_index in self._mmaps:
            self._mmaps[chunk_index]._mmap.close()
            del self._mmaps[chunk_index]
        if chunk_index in self._buffers:
            del self._buffers[chunk_index]

    @classmethod
    def encode_data(cls, data: List[bytes], _: List[int], flattened: List[Any]) -> Tuple[bytes, Optional[int]]:
        return data[0], flattened[0].shape[0]
