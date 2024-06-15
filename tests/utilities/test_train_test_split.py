import pytest
from litdata import train_test_split
from litdata.constants import _ZSTD_AVAILABLE
from litdata.streaming.cache import Cache
from litdata.streaming.dataset import StreamingDataset


@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        pytest.param("zstd", marks=pytest.mark.skipif(condition=not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")),
    ],
)
def test_train_test_split(tmpdir, compression):
    cache = Cache(str(tmpdir), chunk_size=10, compression=compression)
    for i in range(100):
        cache[i] = i
    cache.done()
    cache.merge()

    my_streaming_dataset = StreamingDataset(input_dir=str(tmpdir))
    train_dataset, test_dataset = train_test_split(my_streaming_dataset, splits=[0.75, 0.25])

    assert len(train_dataset) == 75
    assert len(test_dataset) == 25


@pytest.mark.parametrize(
    "compression",
    [
        pytest.param(None),
        # pytest.param("zstd", marks=pytest.mark.skipif(condition=not _ZSTD_AVAILABLE, reason="Requires: ['zstd']")),
    ],
)
def test_split_a_subsampled_dataset(tmpdir, compression):
    cache = Cache(str(tmpdir), chunk_size=15, compression=compression)
    for i in range(1000):
        cache[i] = i
    cache.done()
    cache.merge()

    _sub_sampled_streaming_dataset = StreamingDataset(input_dir=str(tmpdir), subsample=0.3)

    assert len(_sub_sampled_streaming_dataset) == 300  # 1000 * 0.3

    _split_fraction = [0.2, 0.3, 0.4, 0.1]

    split_datasets = train_test_split(_sub_sampled_streaming_dataset, _split_fraction)

    assert all(len(split_datasets[i]) == int(300 * split) for i, split in enumerate(_split_fraction))
