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

import concurrent.futures
import inspect
import os
from datetime import datetime
from functools import partial
from pathlib import Path
from types import FunctionType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from litdata.constants import _IS_IN_STUDIO
from litdata.processing.data_processor import DataChunkRecipe, DataProcessor, DataTransformRecipe
from litdata.processing.readers import BaseReader
from litdata.processing.utilities import optimize_dns_context
from litdata.streaming.dataloader import StreamingDataLoader
from litdata.streaming.resolver import (
    Dir,
    _assert_dir_has_index_file,
    _assert_dir_is_empty,
    _execute,
    _resolve_dir,
)
from litdata.utilities._pytree import tree_flatten


def _get_indexed_paths(data: Any) -> Dict[int, str]:
    flattened_item, _ = tree_flatten(data)

    return {
        index: element
        for index, element in enumerate(flattened_item)
        if isinstance(element, str) and os.path.exists(element)
    }


def _get_input_dir(inputs: Sequence[Any]) -> Optional[str]:
    indexed_paths = _get_indexed_paths(inputs[0])

    if len(indexed_paths) == 0:
        if len(inputs) < 2:
            return None
        # Check whether the second element has any input_path
        indexed_paths = _get_indexed_paths(inputs[1])
        if len(indexed_paths) == 0:
            return None

        # Every element should have filepaths if any contains one.
        raise ValueError(f"The provided item {inputs[0]} didn't contain any filepaths.")

    absolute_path = str(Path(list(indexed_paths.values())[0]).resolve())

    if "/.project" in absolute_path:
        return "/" + os.path.join(*str(list(indexed_paths.values())[0]).split("/")[:4])

    return "/" + os.path.join(*str(absolute_path).split("/")[:4])


def _get_default_num_workers() -> int:
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return os.cpu_count() or 1


class LambdaDataTransformRecipe(DataTransformRecipe):
    def __init__(self, fn: Callable[[str, Any], None], inputs: Sequence[Any]):
        super().__init__()
        self._fn = fn
        self._inputs = inputs
        self._device: Optional[str] = None

        _fn = self._fn if isinstance(self._fn, FunctionType) else self._fn.__call__  # type: ignore
        params = inspect.signature(_fn).parameters
        self._contains_device = "device" in params
        self._contains_is_last = "is_last" in params

    def prepare_structure(self, _: Optional[str]) -> Any:
        return self._inputs

    def prepare_item(self, item_metadata: Any, output_dir: str, is_last: bool) -> None:
        if self._contains_device and self._device is None:
            self._find_device()

        kwargs: Dict[str, Any] = {}

        if self._contains_device:
            kwargs["device"] = self._device

        if self._contains_is_last:
            kwargs["is_last"] = is_last

        if isinstance(self._fn, (FunctionType, partial)):
            self._fn(item_metadata, output_dir, **kwargs)

        elif callable(self._fn):
            self._fn.__call__(item_metadata, output_dir, **kwargs)  # type: ignore
        else:
            raise ValueError(f"The provided {self._fn} isn't supported.")

    def _find_device(self) -> None:
        global_rank = os.getenv("DATA_OPTIMIZER_GLOBAL_RANK", None)
        if torch.cuda.is_available() and global_rank:
            num_gpus = torch.cuda.device_count()
            device = int(global_rank) % num_gpus
            self._device = f"cuda:{device}"


class LambdaDataChunkRecipe(DataChunkRecipe):
    def __init__(
        self,
        fn: Callable[[Any], None],
        inputs: Sequence[Any],
        chunk_size: Optional[int],
        chunk_bytes: Optional[Union[int, str]],
        compression: Optional[str],
    ):
        super().__init__(chunk_size=chunk_size, chunk_bytes=chunk_bytes, compression=compression)
        self._fn = fn
        self._inputs = inputs
        self.is_generator = False

        self.check_fn()

        self.prepare_item = self._prepare_item_generator if self.is_generator else self._prepare_item

    def check_fn(self) -> None:
        if (
            isinstance(self._fn, (partial, FunctionType))
            and inspect.isgeneratorfunction(self._fn)
            or (callable(self._fn) and inspect.isgeneratorfunction(self._fn.__call__))
        ):
            self.is_generator = True

    def _prepare_item(self, item_metadata) -> Any:
        return self._fn(item_metadata)

    def _prepare_item_generator(self, item_metadata) -> Any:
        yield from self._fn(item_metadata)

    def prepare_structure(self, input_dir: Optional[str]) -> Any:
        return self._inputs

    def prepare_item(self, item_metadata: Any) -> Any:
        """This method is overriden dynamically."""


def map(
    fn: Callable[[str, Any], None],
    inputs: Sequence[Any],
    output_dir: Union[str, Dir],
    weights: Optional[List[int]] = None,
    num_workers: Optional[int] = None,
    fast_dev_run: Union[bool, int] = False,
    num_nodes: Optional[int] = None,
    machine: Optional[str] = None,
    num_downloaders: Optional[int] = None,
    num_uploaders: Optional[int] = None,
    reorder_files: bool = True,
    error_when_not_empty: bool = False,
    reader: Optional[BaseReader] = None,
    batch_size: Optional[int] = None,
) -> None:
    """This function map a callbable over a collection of files possibly in a distributed way.

    Arguments:
        fn: A function to be executed over each input element
        inputs: A sequence of input to be processed by the `fn` function.
            Each input should contain at least a valid filepath.
        output_dir: The folder where the processed data should be stored.
        weights: Provide an associated weight to each input. This is used to balance work among workers.
        num_workers: The number of workers to use during processing
        fast_dev_run: Whether to use process only a sub part of the inputs
        num_nodes: When doing remote execution, the number of nodes to use. Only supported on https://lightning.ai/.
        machine: When doing remote execution, the machine to use. Only supported on https://lightning.ai/.
        num_downloaders: The number of downloaders per worker.
        num_uploaders: The number of uploaders per workers.
        reorder_files: By default, reorders the files by file size to distribute work equally among all workers.
            Set this to ``False`` if the order in which samples are processed should be preserved.
        error_when_not_empty: Whether we should error if the output folder isn't empty.
        batch_size: Group the inputs into batches of batch_size length.

    """
    if isinstance(inputs, StreamingDataLoader) and batch_size is not None:
        raise ValueError("When providing a streaming dataloader, pass the batch_size to the dataloader directly.")

    if isinstance(inputs, StreamingDataLoader) and weights is not None:
        raise ValueError("When providing a streaming dataloader, weights isn't supported.")

    if not isinstance(inputs, (Sequence, StreamingDataLoader)):
        raise ValueError(f"The provided inputs should be non empty sequence or a streaming dataloader. Found {inputs}.")

    if len(inputs) == 0:
        raise ValueError(f"The provided inputs should be non empty. Found {inputs}.")

    if not _IS_IN_STUDIO and (machine is not None or num_nodes is not None):
        raise ValueError(
            "Only https://lightning.ai/ supports multiple nodes or selecting a machine."
            " Create an account to try it out."
        )

    if not _IS_IN_STUDIO:
        print(
            "Create an account on https://lightning.ai/ to transform your data faster using "
            "multiple nodes and large machines."
        )

    if num_nodes is None or int(os.getenv("DATA_OPTIMIZER_NUM_NODES", 0)) > 0:
        _output_dir: Dir = _resolve_dir(output_dir)

        if _output_dir.url and "cloudspaces" in _output_dir.url:
            raise ValueError(
                f"The provided `output_dir` isn't valid. Found {_output_dir.path if _output_dir else None}."
                " HINT: You can either use `/teamspace/s3_connections/...` or `/teamspace/datasets/...`."
            )

        if error_when_not_empty:
            _assert_dir_is_empty(_output_dir)

        if not isinstance(inputs, StreamingDataLoader):
            input_dir = _resolve_dir(_get_input_dir(inputs))

            if isinstance(batch_size, int) and batch_size > 1:
                inputs = [inputs[pos : pos + batch_size] for pos in range(0, len(inputs), batch_size)]
        else:
            input_dir = Dir()

        data_processor = DataProcessor(
            input_dir=input_dir,
            output_dir=_output_dir,
            num_workers=num_workers or _get_default_num_workers(),
            fast_dev_run=fast_dev_run,
            num_downloaders=num_downloaders,
            num_uploaders=num_uploaders,
            reorder_files=reorder_files,
            weights=weights,
            reader=reader,
        )
        with optimize_dns_context(True):
            return data_processor.run(LambdaDataTransformRecipe(fn, inputs))
    return _execute(
        f"litdata-map-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        num_nodes,
        machine,
    )


def optimize(
    fn: Callable[[Any], Any],
    inputs: Sequence[Any],
    output_dir: str,
    weights: Optional[List[int]] = None,
    chunk_size: Optional[int] = None,
    chunk_bytes: Optional[Union[int, str]] = None,
    compression: Optional[str] = None,
    num_workers: Optional[int] = None,
    fast_dev_run: bool = False,
    num_nodes: Optional[int] = None,
    machine: Optional[str] = None,
    num_downloaders: Optional[int] = None,
    num_uploaders: Optional[int] = None,
    reorder_files: bool = True,
    reader: Optional[BaseReader] = None,
    batch_size: Optional[int] = None,
) -> None:
    """This function converts a dataset into chunks possibly in a distributed way.

    Arguments:
        fn: A function to be executed over each input element
        inputs: A sequence of input to be processed by the `fn` function.
            Each input should contain at least a valid filepath.
        output_dir: The folder where the processed data should be stored.
        weights: Provide an associated weight to each input. This is used to balance work among workers.
        chunk_size: The maximum number of elements to hold within a chunk.
        chunk_bytes: The maximum number of bytes to hold within a chunk.
        compression: The compression algorithm to use over the chunks.
        num_workers: The number of workers to use during processing
        fast_dev_run: Whether to use process only a sub part of the inputs
        num_nodes: When doing remote execution, the number of nodes to use. Only supported on https://lightning.ai/.
        machine: When doing remote execution, the machine to use. Only supported on https://lightning.ai/.
        num_downloaders: The number of downloaders per worker.
        num_uploaders: The numbers of uploaders per worker.
        reorder_files: By default, reorders the files by file size to distribute work equally among all workers.
            Set this to ``False`` if the order in which samples are processed should be preserved.
        batch_size: Group the inputs into batches of batch_size length.

    """
    if isinstance(inputs, StreamingDataLoader) and batch_size is not None:
        raise ValueError("When providing a streaming dataloader, pass the batch_size to the dataloader directly.")

    if isinstance(inputs, StreamingDataLoader) and weights is not None:
        raise ValueError("When providing a streaming dataloader, weights isn't supported.")

    if not isinstance(inputs, (Sequence, StreamingDataLoader)):
        raise ValueError(f"The provided inputs should be non empty sequence or a streaming dataloader. Found {inputs}.")

    if len(inputs) == 0:
        raise ValueError(f"The provided inputs should be non empty. Found {inputs}.")

    if chunk_size is None and chunk_bytes is None:
        raise ValueError("Either `chunk_size` or `chunk_bytes` needs to be defined.")

    if not _IS_IN_STUDIO and (machine is not None or num_nodes is not None):
        raise ValueError(
            "Only https://lightning.ai/ supports multiple nodes or selecting a machine."
            "Create an account to try it out."
        )

    if not _IS_IN_STUDIO:
        print(
            "Create an account on https://lightning.ai/ to optimize your data faster "
            "using multiple nodes and large machines."
        )

    if num_nodes is None or int(os.getenv("DATA_OPTIMIZER_NUM_NODES", 0)) > 0:
        _output_dir: Dir = _resolve_dir(output_dir)

        if _output_dir.url is not None and "cloudspaces" in _output_dir.url:
            raise ValueError(
                f"The provided `output_dir` isn't valid. Found {_output_dir.path}."
                " HINT: You can either use `/teamspace/s3_connections/...` or `/teamspace/datasets/...`."
            )

        _assert_dir_has_index_file(_output_dir)

        if not isinstance(inputs, StreamingDataLoader):
            input_dir = _resolve_dir(_get_input_dir(inputs))

            if isinstance(batch_size, int) and batch_size > 1:
                inputs = [inputs[pos : pos + batch_size] for pos in range(0, len(inputs), batch_size)]
        else:
            input_dir = Dir()

        data_processor = DataProcessor(
            input_dir=input_dir,
            output_dir=_output_dir,
            num_workers=num_workers or _get_default_num_workers(),
            fast_dev_run=fast_dev_run,
            num_downloaders=num_downloaders,
            num_uploaders=num_uploaders,
            reorder_files=reorder_files,
            reader=reader,
        )

        with optimize_dns_context(True):
            data_processor.run(
                LambdaDataChunkRecipe(
                    fn,
                    inputs,
                    chunk_size=chunk_size,
                    chunk_bytes=chunk_bytes,
                    compression=compression,
                )
            )
        return None
    return _execute(
        f"litdata-optimize-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        num_nodes,
        machine,
    )


def _listdir(folder: str) -> Tuple[str, List[str]]:
    return folder, os.listdir(folder)


class walk:
    """This class is an optimized version of os.walk for listing files and folders from cloud filesystem.

    Note: The order of files and folders yielded aren't depth-first anymore due to the asynchronous listing call.

    """

    def __init__(self, folder: str, max_workers: Optional[int] = os.cpu_count()) -> None:
        self.folders = [folder]
        self.max_workers = max_workers or 1
        self.futures: List[concurrent.futures.Future] = []

        if not _IS_IN_STUDIO:
            print("This method is optimized to run on https://lightning.ai/. Don't use it otherwise.")

    def __iter__(self) -> Any:
        """This function queues the folders to perform listdir across multiple workers."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while len(self.folders):
                folder = self.folders.pop(0)
                future = executor.submit(_listdir, folder)
                self.futures.append(future)

            while self.futures:
                for future in concurrent.futures.as_completed(self.futures):
                    filenames = []
                    folders = []

                    folder, files_or_folders = future.result()
                    self.futures = [f for f in self.futures if f != future]

                    for file_or_folder in files_or_folders:
                        if os.path.isfile(os.path.join(folder, file_or_folder)):
                            filenames.append(file_or_folder)
                        else:
                            folders.append(file_or_folder)
                            self.folders.append(os.path.join(folder, file_or_folder))

                    yield folder, folders, filenames

                    while len(self.folders) and len(self.futures) <= self.max_workers * 2:
                        folder = self.folders.pop(0)
                        future = executor.submit(_listdir, folder)
                        self.futures.append(future)
        return
