import sys
from time import sleep, time
from unittest import mock

import pytest

from litdata.streaming import client


def test_s3_client_with_storage_options(monkeypatch):
    boto3_session = mock.MagicMock()
    boto3 = mock.MagicMock(Session=boto3_session)
    monkeypatch.setattr(client, "boto3", boto3)

    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)

    # Create S3Client with storage options
    storage_options = {
        "region_name": "us-west-2",
        "endpoint_url": "https://custom.endpoint",
        "config": botocore.config.Config(retries={"max_attempts": 100}),
    }
    s3_client = client.S3Client(storage_options=storage_options)

    assert s3_client.client

    boto3_session().client.assert_called_with(
        "s3",
        region_name="us-west-2",
        endpoint_url="https://custom.endpoint",
        config=botocore.config.Config(retries={"max_attempts": 100}),
    )

    # Create S3Client without storage options
    s3_client = client.S3Client()
    assert s3_client.client

    # Verify that boto3.Session().client was called with the default parameters
    boto3_session().client.assert_called_with(
        "s3",
        config=botocore.config.Config(retries={"max_attempts": 1000, "mode": "adaptive"}),
    )


def test_s3_client_without_cloud_space_id(monkeypatch):
    boto3_session = mock.MagicMock()
    boto3 = mock.MagicMock(Session=boto3_session)
    monkeypatch.setattr(client, "boto3", boto3)

    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)

    instance_metadata_provider = mock.MagicMock()
    monkeypatch.setattr(client, "InstanceMetadataProvider", instance_metadata_provider)

    instance_metadata_fetcher = mock.MagicMock()
    monkeypatch.setattr(client, "InstanceMetadataFetcher", instance_metadata_fetcher)

    s3 = client.S3Client(1)
    assert s3.client
    assert s3.client
    assert s3.client
    assert s3.client
    assert s3.client

    boto3_session().client.assert_called_once()


@pytest.mark.skipif(sys.platform == "win32", reason="not supported on windows")
@pytest.mark.parametrize("use_shared_credentials", [False, True, None])
def test_s3_client_with_cloud_space_id(use_shared_credentials, monkeypatch):
    boto3_session = mock.MagicMock()
    boto3 = mock.MagicMock(Session=boto3_session)
    monkeypatch.setattr(client, "boto3", boto3)

    botocore = mock.MagicMock()
    monkeypatch.setattr(client, "botocore", botocore)

    if isinstance(use_shared_credentials, bool):
        monkeypatch.setenv("LIGHTNING_CLOUD_SPACE_ID", "dummy")
        monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/.credentials/.aws_credentials")
        monkeypatch.setenv("AWS_CONFIG_FILE", "/.credentials/.aws_credentials")

    instance_metadata_provider = mock.MagicMock()
    monkeypatch.setattr(client, "InstanceMetadataProvider", instance_metadata_provider)

    instance_metadata_fetcher = mock.MagicMock()
    monkeypatch.setattr(client, "InstanceMetadataFetcher", instance_metadata_fetcher)

    s3 = client.S3Client(1)
    assert s3.client
    assert s3.client
    boto3_session().client.assert_called_once()
    sleep(1 - (time() - s3._last_time))
    assert s3.client
    assert s3.client
    assert len(boto3_session().client._mock_mock_calls) == 6
    sleep(1 - (time() - s3._last_time))
    assert s3.client
    assert s3.client
    assert len(boto3_session().client._mock_mock_calls) == 9

    assert instance_metadata_provider._mock_call_count == 0 if use_shared_credentials else 3
