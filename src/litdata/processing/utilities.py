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

import glob
import io
import json
import os
import shutil
import urllib
from contextlib import contextmanager
from subprocess import DEVNULL, Popen
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from litdata.constants import _INDEX_FILENAME, _IS_IN_STUDIO, _LIGHTNING_CLOUD_AVAILABLE

if _LIGHTNING_CLOUD_AVAILABLE:
    from lightning_cloud.openapi import (
        ProjectIdDatasetsBody,
    )
    from lightning_cloud.openapi.rest import ApiException
    from lightning_cloud.rest_client import LightningClient


def _create_dataset(
    input_dir: Optional[str],
    storage_dir: str,
    dataset_type: Any,
    empty: Optional[bool] = None,
    size: Optional[int] = None,
    num_bytes: Optional[str] = None,
    data_format: Optional[Union[str, Tuple[str]]] = None,
    compression: Optional[str] = None,
    num_chunks: Optional[int] = None,
    num_bytes_per_chunk: Optional[List[int]] = None,
    name: Optional[str] = None,
    version: Optional[int] = None,
) -> None:
    """Create a dataset with metadata information about its source and destination."""
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID", None)
    cluster_id = os.getenv("LIGHTNING_CLUSTER_ID", None)
    user_id = os.getenv("LIGHTNING_USER_ID", None)
    cloud_space_id = os.getenv("LIGHTNING_CLOUD_SPACE_ID", None)
    lightning_app_id = os.getenv("LIGHTNING_CLOUD_APP_ID", None)

    if project_id is None:
        return

    if not storage_dir:
        raise ValueError("The storage_dir should be defined.")

    client = LightningClient(retry=False)

    try:
        client.dataset_service_create_dataset(
            body=ProjectIdDatasetsBody(
                cloud_space_id=cloud_space_id if lightning_app_id is None else None,
                cluster_id=cluster_id,
                creator_id=user_id,
                empty=empty,
                input_dir=input_dir,
                lightning_app_id=lightning_app_id,
                name=name,
                size=size,
                num_bytes=num_bytes,
                data_format=str(data_format) if data_format else data_format,
                compression=compression,
                num_chunks=num_chunks,
                num_bytes_per_chunk=num_bytes_per_chunk,
                storage_dir=storage_dir,
                type=dataset_type,
                version=version,
            ),
            project_id=project_id,
        )
    except ApiException as ex:
        if "already exists" in str(ex.body):
            pass
        else:
            raise ex


def get_worker_rank() -> Optional[str]:
    return os.getenv("DATA_OPTIMIZER_GLOBAL_RANK")


def catch(func: Callable) -> Callable:
    def _wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, Optional[Exception]]:
        try:
            return func(*args, **kwargs), None
        except Exception as e:
            return None, e

    return _wrapper


# Credit to the https://github.com/rom1504/img2dataset Github repo
# The code was taken from there. It has a MIT License.


def make_request(
    url: str,
    timeout: int = 10,
    user_agent_token: str = "pytorch-lightning",  # noqa: S107
) -> io.BytesIO:
    """Download an image with urllib."""
    user_agent_string = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
    if user_agent_token:
        user_agent_string += f" (compatible; {user_agent_token}; +https://github.com/Lightning-AI/pytorch-lightning)"

    with urllib.request.urlopen(
        urllib.request.Request(url, data=None, headers={"User-Agent": user_agent_string}), timeout=timeout
    ) as r:
        return io.BytesIO(r.read())


@contextmanager
def optimize_dns_context(enable: bool) -> Any:
    optimize_dns(enable)
    try:
        yield
        optimize_dns(False)  # always disable the optimize DNS
    except Exception as e:
        optimize_dns(False)  # always disable the optimize DNS
        raise e


def optimize_dns(enable: bool) -> None:
    if not _IS_IN_STUDIO:
        return

    with open("/etc/resolv.conf") as f:
        lines = f.readlines()

    if (enable and any("127.0.0.53" in line for line in lines)) or (
        not enable and any("127.0.0.1" in line for line in lines)
    ):
        cmd = (
            f"sudo /home/zeus/miniconda3/envs/cloudspace/bin/python"
            f" -c 'from litdata.processing.utilities import _optimize_dns; _optimize_dns({enable})'"
        )
        Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL).wait()  # E501


def _optimize_dns(enable: bool) -> None:
    with open("/etc/resolv.conf") as f:
        lines = f.readlines()

    write_lines = []
    for line in lines:
        if "nameserver 127" in line:
            if enable:
                write_lines.append("nameserver 127.0.0.1\n")
            else:
                write_lines.append("nameserver 127.0.0.53\n")
        else:
            write_lines.append(line)

    with open("/etc/resolv.conf", "w") as f:
        for line in write_lines:
            f.write(line)


def _get_work_dir() -> str:
    # Provides the storage path associated to the current Lightning Work.
    bucket_name = os.getenv("LIGHTNING_BUCKET_NAME")
    project_id = os.getenv("LIGHTNING_CLOUD_PROJECT_ID")
    app_id = os.getenv("LIGHTNING_CLOUD_APP_ID")
    work_id = os.getenv("LIGHTNING_CLOUD_WORK_ID")
    assert bucket_name is not None
    assert project_id is not None
    assert work_id is not None
    return f"s3://{bucket_name}/projects/{project_id}/lightningapps/{app_id}/artifacts/{work_id}/content/"


def append_index_json(temp_index: Dict[str, Any], output_index: Dict[str, Any]) -> Dict[str, Any]:
    "Utility function to append the optimize utility to the output directory."
    if temp_index["config"] != output_index["config"]:
        raise ValueError("The config of the optimized dataset is different from the original one.")

    combined_chunks = output_index["chunks"] + temp_index["chunks"]
    combined_config = temp_index["config"]

    return {"chunks": combined_chunks, "config": combined_config}


def overwrite_index_json(temp_index: Dict[str, Any], output_index: Dict[str, Any]) -> Dict[str, Any]:
    "Utility function to overwrite the optimize utility to the output directory."
    if temp_index["config"] != output_index["config"]:
        raise ValueError("The config of the optimized dataset is different from the original one.")

    return {"chunks": temp_index["chunks"], "config": temp_index["config"]}


def optimize_mode_utility(temp_dir: str, output_dir: str, mode: Literal["append", "overwrite"]) -> None:
    "Utility function to append/overwrite new optimized data to the output directory."

    if mode not in ["append", "overwrite"]:
        raise ValueError(f"The provided mode {mode} isn't supported. Use `append`, or `overwrite`.")

    try:
        if not os.path.exists(os.path.join(output_dir, _INDEX_FILENAME)):
            # simply move `index.json` from the temp_dir to the output_dir, and delete the temp_dir
            move_files_between_dirs(temp_dir, output_dir, _INDEX_FILENAME)
        else:
            # read index.json from temp_dir and output_dir and merge/overwrite them
            with open(os.path.join(temp_dir, _INDEX_FILENAME)) as f:
                with open(os.path.join(output_dir, _INDEX_FILENAME)) as g:
                    temp_index = json.load(f)
                    output_index = json.load(g)

                    if mode == "append":
                        final_index = append_index_json(temp_index, output_index)
                    else:
                        final_index = overwrite_index_json(temp_index, output_index)

            # write the final data to a file (final_index.json)
            with open(os.path.join(output_dir, _INDEX_FILENAME), "w") as final_index_file:
                json.dump(final_index, final_index_file)

        # move all the `.bin` files from the temp_dir to the output_dir
        move_files_between_dirs(temp_dir, output_dir, ".bin")
    finally:
        # delete the temp_dir
        shutil.rmtree(temp_dir)


def move_files_between_dirs(source_dir: str, target_dir: str, file_extension: str) -> None:
    # Ensure target_dir exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # List all files in the source_dir
    for filename in os.listdir(source_dir):
        # Check if the file has the desired extension
        if filename.endswith(file_extension):
            # Construct full file path
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)
            # Move the file
            shutil.move(source_file, target_file)


def delete_files_with_extension(directory: str, extension: str) -> None:
    """Delete all files with the given extension in the specified directory.

    **Not** in the subdirectories.

    """
    # Construct the pattern for the files with the given extension in the specified directory
    pattern = os.path.join(directory, f"*.{extension}")

    # Use glob to find all files that match the pattern
    files_to_delete = glob.glob(pattern)

    # Iterate over the files and delete them
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
        except OSError:
            continue
