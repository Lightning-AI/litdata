import sys
import threading
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch.distributed
from litdata import CombinedStreamingDataset, StreamingDataset
from litdata.streaming.cache import Cache
from litdata.streaming.reader import PrepareChunksThread


@pytest.fixture(autouse=True)
def teardown_process_group():
    """Ensures distributed process group gets closed before the next test runs."""
    yield
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.fixture
def mosaic_mds_index_data():
    return {
        "shards": [
            {
                "column_encodings": ["int", "jpeg"],
                "column_names": ["class", "image"],
                "column_sizes": [8, None],
                "compression": "zstd",
                "format": "mds",
                "hashes": [],
                "raw_data": {"basename": "shard.00000.mds", "bytes": 125824, "hashes": {}},
                "samples": 100,
                "size_limit": 67108864,
                "version": 2,
                "zip_data": {"basename": "shard.00000.mds.zstd", "bytes": 63407, "hashes": {}},
            }
        ],
        "version": 2,
    }


@pytest.fixture
def google_mock(monkeypatch):
    google = ModuleType("google")
    monkeypatch.setitem(sys.modules, "google", google)
    google_cloud = ModuleType("cloud")
    monkeypatch.setitem(sys.modules, "google.cloud", google_cloud)
    google_cloud_storage = ModuleType("storage")
    monkeypatch.setitem(sys.modules, "google.cloud.storage", google_cloud_storage)
    google.cloud = google_cloud
    google.cloud.storage = google_cloud_storage
    return google


@pytest.fixture
def azure_mock(monkeypatch):
    azure = ModuleType("azure")
    monkeypatch.setitem(sys.modules, "azure", azure)
    azure_storage = ModuleType("storage")
    monkeypatch.setitem(sys.modules, "azure.storage", azure_storage)
    azure_storage_blob = ModuleType("storage")
    monkeypatch.setitem(sys.modules, "azure.storage.blob", azure_storage_blob)
    azure.storage = azure_storage
    azure.storage.blob = azure_storage_blob
    return azure


@pytest.fixture
def lightning_cloud_mock(monkeypatch):
    lightning_cloud = ModuleType("lightning_sdk.lightning_cloud")
    monkeypatch.setitem(sys.modules, "lightning_sdk.lightning_cloud", lightning_cloud)
    rest_client = ModuleType("rest_client")
    monkeypatch.setitem(sys.modules, "lightning_sdk.lightning_cloud.rest_client", rest_client)
    lightning_cloud.rest_client = rest_client
    rest_client.LightningClient = Mock()
    return lightning_cloud


@pytest.fixture
def lightning_sdk_mock(monkeypatch):
    lightning_sdk = ModuleType("lightning_sdk")
    monkeypatch.setitem(sys.modules, "lightning_sdk", lightning_sdk)
    return lightning_sdk


@pytest.fixture(autouse=True)
def _thread_police():
    """Attempts stopping left-over threads to avoid test interactions.

    Adapted from PyTorch Lightning.

    """
    active_threads_before = set(threading.enumerate())
    yield
    active_threads_after = set(threading.enumerate())

    for thread in active_threads_after - active_threads_before:
        if isinstance(thread, PrepareChunksThread):
            thread.force_stop()
            continue

        stop = getattr(thread, "stop", None) or getattr(thread, "exit", None)
        if thread.daemon and callable(stop):
            # A daemon thread would anyway be stopped at the end of a program
            # We do it preemptively here to reduce the risk of interactions with other tests that run after
            stop()
            assert not thread.is_alive()
        elif thread.name == "QueueFeederThread":
            thread.join(timeout=20)
        else:
            raise AssertionError(f"Test left zombie thread: {thread}")


@pytest.fixture()
def combined_dataset(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("data")
    datasets = [str(tmpdir.join(f"dataset_{i}")) for i in range(2)]
    for dataset in datasets:
        cache = Cache(input_dir=dataset, chunk_bytes="64MB")
        for i in range(50):
            cache[i] = i
        cache.done()
        cache.merge()

    dataset_1 = StreamingDataset(datasets[0], shuffle=True)
    dataset_2 = StreamingDataset(datasets[1], shuffle=True)
    return CombinedStreamingDataset(datasets=[dataset_1, dataset_2])
