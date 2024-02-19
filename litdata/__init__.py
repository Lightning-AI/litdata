from lightning_utilities.core.imports import RequirementCache

from litdata.processing.functions import map, optimize, walk
from litdata.streaming.combined import CombinedStreamingDataset
from litdata.streaming.dataloader import StreamingDataLoader
from litdata.streaming.dataset import StreamingDataset

__all__ = [
    "LightningDataset",
    "StreamingDataset",
    "CombinedStreamingDataset",
    "StreamingDataLoader",
    "LightningIterableDataset",
    "map",
    "optimize",
    "walk",
]

if RequirementCache("lightning_sdk"):
    from lightning_sdk import Machine  # noqa: F401

    __all__.append("Machine")
