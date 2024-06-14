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
