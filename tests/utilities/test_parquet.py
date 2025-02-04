import os

import pytest

from litdata.helpers import get_hf_pq_file_download_cmd


# Test with token provided
@pytest.mark.parametrize(
    ("hf_token", "expected_cmd"),
    [
        (
            "provided_token",
            'wget -q --header="Authorization: Bearer provided_token" "https://huggingface.co/file" -O "/tmp/file.parquet"',  # noqa: E501
        ),
    ],
)
def test_with_token_provided(hf_token, expected_cmd):
    file_url = "https://huggingface.co/file"
    local_path = "/tmp/file.parquet"  # noqa: S108

    actual_cmd = get_hf_pq_file_download_cmd(file_url, local_path, hf_token)

    assert actual_cmd == expected_cmd


# Test with token from environment variable
@pytest.mark.parametrize(
    ("env_token", "expected_cmd"),
    [
        (
            "test_token",
            'wget -q --header="Authorization: Bearer test_token" "https://huggingface.co/file" -O "/tmp/file.parquet"',
        ),
    ],
)
def test_with_token_from_env(env_token, expected_cmd):
    file_url = "https://huggingface.co/file"
    local_path = "/tmp/file.parquet"  # noqa: S108

    # Set environment variable for the test
    os.environ["HF_TOKEN"] = env_token

    actual_cmd = get_hf_pq_file_download_cmd(file_url, local_path)

    assert actual_cmd == expected_cmd


# Test without token
@pytest.mark.parametrize(
    ("expected_cmd"),
    [
        ('wget -q "https://huggingface.co/file" -O "/tmp/file.parquet"'),
    ],
)
def test_without_token(expected_cmd):
    file_url = "https://huggingface.co/file"
    local_path = "/tmp/file.parquet"  # noqa: S108

    # Ensure environment variable is not set for this test
    if "HF_TOKEN" in os.environ:
        del os.environ["HF_TOKEN"]

    actual_cmd = get_hf_pq_file_download_cmd(file_url, local_path)

    assert actual_cmd == expected_cmd
