from lightning_utilities.core.imports import RequirementCache

from lightning_data.processing.functions import map, optimize, walk
from lightning_data.streaming.combined import CombinedStreamingDataset
from lightning_data.streaming.dataloader import StreamingDataLoader
from lightning_data.streaming.dataset import StreamingDataset

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
