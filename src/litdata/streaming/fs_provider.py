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

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from urllib import parse

from litdata.constants import _GOOGLE_STORAGE_AVAILABLE
from litdata.streaming.client import S3Client


class FsProvider(ABC):
    @abstractmethod
    def upload_file(self, local_path: str, remote_path: str) -> None:
        raise NotImplementedError

    def download_file(self, remote_path: str, local_path: str) -> None:
        raise NotImplementedError

    def list_directory(self, path: str) -> List[str]:
        raise NotImplementedError

    def delete_file(self, path: str) -> None:
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        raise NotImplementedError


class GCPFsProvider(FsProvider):
    def __init__(self, storage_options: Optional[Dict] = {}):
        if not _GOOGLE_STORAGE_AVAILABLE:
            raise ModuleNotFoundError(str(_GOOGLE_STORAGE_AVAILABLE))
        from google.cloud import storage

        super().__init__()
        self.storage_options = storage_options
        self.client = storage.Client(**self.storage_options)

    def upload_file(self, local_path: str, remote_path: str) -> None:
        bucket_name, blob_path = get_bucket_and_path(remote_path, "gs")
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path)

    def download_file(self, remote_path: str, local_path: str) -> None:
        bucket_name, blob_path = get_bucket_and_path(remote_path, "gs")
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_path)

    def list_directory(self, path: str) -> List[str]:
        raise NotImplementedError

    def delete_file(self, path: str) -> None:
        bucket_name, blob_path = get_bucket_and_path(path, "gs")
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.delete()

    def exists(self, path: str) -> bool:
        bucket_name, blob_path = get_bucket_and_path(path, "gs")
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.exists()


class S3FsProvider(FsProvider):
    def __init__(self, storage_options: Optional[Dict] = {}):
        super().__init__()
        self.storage_options = storage_options
        self.client = S3Client(storage_options=storage_options)

    def upload_file(self, local_path: str, remote_path: str) -> None:
        bucket_name, blob_path = get_bucket_and_path(remote_path, "s3")
        self.client.upload_file(local_path, bucket_name, blob_path)

    def download_file(self, remote_path: str, local_path: str) -> None:
        bucket_name, blob_path = get_bucket_and_path(remote_path, "s3")
        with open(local_path, "wb") as f:
            self.client.download_fileobj(bucket_name, blob_path, f)

    def list_directory(self, path: str) -> List[str]:
        raise NotImplementedError

    def delete(self, path: str) -> None:
        """Delete the file or the directory."""
        import boto3

        s3 = boto3.resource("s3")
        bucket_name, blob_path = get_bucket_and_path(path, "s3")

        for obj in s3.Bucket(bucket_name).objects.filter(Prefix=blob_path):
            s3.Object(bucket_name, obj.key).delete()

    def exists(self, path: str) -> bool:
        import botocore

        bucket_name, blob_path = get_bucket_and_path(path, "s3")
        try:
            _ = self.client.head_object(Bucket=bucket_name, Key=blob_path)
            return True
        except botocore.exceptions.ClientError as e:
            if "the HeadObject operation: Not Found" in str(e):
                return False
            raise e
        except Exception as e:
            raise e


def get_bucket_and_path(remote_filepath: str, expected_scheme: str = "s3") -> Tuple[str, str]:
    """Parse the remote filepath and return the bucket name and the blob path.

    Args:
        remote_filepath (str): The remote filepath to parse.
        expected_scheme (str, optional): The expected scheme of the remote filepath. Defaults to "s3".

    Raises:
        ValueError: If the scheme of the remote filepath is not as expected.

    Returns:
        Tuple[str, str]: The bucket name and the blob_path.
    """
    obj = parse.urlparse(remote_filepath)

    if obj.scheme != expected_scheme:
        raise ValueError(
            f"Expected obj.scheme to be `{expected_scheme}`, instead, got {obj.scheme} for remote={remote_filepath}."
        )

    bucket_name = obj.netloc
    blob_path = obj.path
    # Remove the leading "/":
    if blob_path[0] == "/":
        blob_path = blob_path[1:]

    return bucket_name, blob_path
