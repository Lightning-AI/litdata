<div align="center">

<img alt="Lightning" src="https://pl-flash-data.s3.amazonaws.com/lightning_data_logo.png" width="800px" style="max-width: 100%;">

<br/>
<br/>

## Blazingly fast, distributed streaming of training data from cloud storage

</div>

# ⚡ Welcome to Lightning Data

With Lightning Data, users can transform and optimize their data in cloud environments with an intuitive approach and at any scale. 

Then, efficient distributed training becomes feasible regardless of the data's location, allowing users to effortlessly stream in the required data as needed.

Lightning Data supports **images, text, video, audio, geo-spatial, and multimodal data** types, is already adopted by frameworks such as [Lit-GPT](https://github.com/Lightning-AI/lit-gpt/blob/main/pretrain/tinyllama.py) to pretrain LLMs and integrates smoothly [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), [Lightning Fabric](https://lightning.ai/docs/open-source/fabric), and [PyTorch](https://pytorch.org/docs/stable/index.html).

### Table of Contents

- [Getting started](#getting-started)
    - [Installation](#installation)
    - [Quick Start](#quick-start)
        - [1. Prepare Your Data](#1-prepare-your-data)
        - [2. Upload Your Data to Cloud Storage](#2-upload-your-data-to-cloud-storage)
        - [3. Use StreamingDataset and DataLoader](#3-use-streamingdataset-and-dataloader)
- [Real World Examples](#real-world-examples)
- [Key Features](#key-features)
- [Benchmarks](#benchmarks)
- [Lightning AI Platform: Scale cloud data processing](#lightning-ai-platform-scale-cloud-data-processing)
- [Contributors](#-contributors)

# Getting Started

## Installation

Lightning Data can be installed with `pip`:

<!--pytest.mark.skip-->

```bash
pip install -U litdata
```

## Quick Start

### 1. Prepare Your Data

Convert your raw dataset into Lightning Streaming format using the `optimize` operator. More formats are coming...

<!--pytest.mark.skip-->

```python
import numpy as np
from litdata import optimize
from PIL import Image


# Store random images into the chunks
def random_images(index):
    data = {
        "index": index,
        "image": Image.fromarray(np.random.randint(0, 256, (32, 32, 3), np.uint8)),
        "class": np.random.randint(10),
    }
    return data # The data is serialized into bytes and stored into chunks by the optimize operator.

if __name__ == "__main__":
    optimize(
        fn=random_images,  # The function applied over each input.
        inputs=list(range(1000)),  # Provide any inputs. The fn is applied on each item.
        output_dir="my_dataset",  # The directory where the optimized data are stored.
        num_workers=4,  # The number of workers. The inputs are distributed among them.
        chunk_bytes="64MB"  # The maximum number of bytes to write into a chunk.
    )

```

The `optimize` operator supports any data structures and types. Serialize whatever you want.

### 2. Upload Your Data to Cloud Storage

Cloud providers such as [AWS](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), [Google Cloud](https://cloud.google.com/storage/docs/uploading-objects?hl=en#upload-object-cli), [Azure](https://learn.microsoft.com/en-us/azure/import-export/storage-import-export-data-to-files?tabs=azure-portal-preview), etc.. provide command line client to upload your data to their storage.

Here is an example with [AWS S3](https://aws.amazon.com/s3).

```bash
⚡ aws s3 cp --recursive my_dataset s3://my-bucket/my_dataset
```

### 3. Use StreamingDataset and DataLoader

```python
from litdata import StreamingDataset
from torch.utils.data import DataLoader

# Remote path where full dataset is persistently stored
input_dir = 's3://pl-flash-data/my_dataset'

# Create streaming dataset
dataset = StreamingDataset(input_dir, shuffle=True)

# Check any elements
sample = dataset[50]
img = sample['image']
cls = sample['class']

# Create PyTorch DataLoader
dataloader = DataLoader(dataset)
```

# Real World Examples

We have built end-to-end free [Studios](https://lightning.ai) showing all the steps to prepare the following datasets:

| Dataset                                                                                                                                      |      Data type      |                                                                                                                                  Studio |
| -------------------------------------------------------------------------------------------------------------------------------------------- | :-----------------: | --------------------------------------------------------------------------------------------------------------------------------------: |
| [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/)                                                                                  | Image & description |            [Use or explore LAION-400MILLION dataset](https://lightning.ai/lightning-ai/studios/use-or-explore-laion-400million-dataset) |
| [Chesapeake Roads Spatial Context](https://github.com/isaaccorley/chesapeakersc)                                                             |    Image & Mask     | [Convert GeoSpatial data to Lightning Streaming](https://lightning.ai/lightning-ai/studios/convert-spatial-data-to-lightning-streaming) |
| [Imagenet 1M](https://paperswithcode.com/sota/image-classification-on-imagenet?tag_filter=171)                                               |    Image & Label    |              [Benchmark cloud data-loading libraries](https://lightning.ai/lightning-ai/studios/benchmark-cloud-data-loading-libraries) |
| [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) & [StartCoder](https://huggingface.co/datasets/bigcode/starcoderdata) |        Text         |              [Prepare the TinyLlama 1T token dataset](https://lightning.ai/lightning-ai/studios/prepare-the-tinyllama-1t-token-dataset) |
| [Swedish Wikipedia](https://huggingface.co/datasets/wikipedia) |        Text         |              [Tokenize 2M Swedish Wikipedia Articles](https://lightning.ai/lightning-ai/studios/tokenize-2m-swedish-wikipedia-articles) |
| [English Wikepedia](https://huggingface.co/datasets/wikipedia)                                                                               |        Text         |            [Embed English Wikipedia under 5 dollars](https://lightning.ai/lightning-ai/studios/embed-english-wikipedia-under-5-dollars) |
| Generated                                                                                                                                    |    Parquet Files    |            [Convert parquets to Lightning Streaming](https://lightning.ai/lightning-ai/studios/convert-parquets-to-lightning-streaming) |

[Lightning Studios](https://lightning.ai) are fully reproducible cloud IDE with data, code, dependencies, etc...

# Key Features

- [Multi-GPU / Multi-Node](#multi-gpu--multi-node)
- [Easy Data Mixing](#easy-data-mixing)
- [Stateful StreamingDataLoader](#stateful-streamingdataloader)
- [Profiling](#profiling)
- [Random access](#random-access)
- [Use data transforms](#use-data-transforms)
- [Disk usage limits](#disk-usage-limits)
- [Support yield](#support-yield)
- [Network Drive On-Prem Support](#network-drive-on-prem-support)

## Multi-GPU / Multi-Node

The `StreamingDataset` and `StreamingDataLoader` takes care of everything for you. They automatically make sure each rank receives different batch of data. There is nothing for you to do if you use them.

![An illustration showing how the Streaming Dataset works with multi node.](https://pl-flash-data.s3.amazonaws.com/streaming_dataset.gif)

## Easy Data Mixing

You can easily experiment with dataset mixtures using the CombinedStreamingDataset.

```python
from litdata import StreamingDataset, CombinedStreamingDataset
from litdata.streaming.item_loader import TokensLoader
from tqdm import tqdm
import os
from torch.utils.data import DataLoader

train_datasets = [
    StreamingDataset(
        input_dir="s3://tinyllama-template/slimpajama/train/",
        item_loader=TokensLoader(block_size=2048 + 1), # Optimized loader for tokens used by LLMs 
        shuffle=True,
        drop_last=True,
    ),
    StreamingDataset(
        input_dir="s3://tinyllama-template/starcoder/",
        item_loader=TokensLoader(block_size=2048 + 1), # Optimized loader for tokens used by LLMs 
        shuffle=True,
        drop_last=True,
    ),
]

# Mix SlimPajama data and Starcoder data with these proportions:
weights = (0.693584, 0.306416)
combined_dataset = CombinedStreamingDataset(datasets=train_datasets, seed=42, weights=weights)

train_dataloader = DataLoader(combined_dataset, batch_size=8, pin_memory=True, num_workers=os.cpu_count())

# Iterate over the combined datasets
for batch in tqdm(train_dataloader):
    pass
```

## Stateful StreamingDataLoader

Lightning Data provides a stateful `StreamingDataLoader`. This simplifies resuming training over large datasets.

Note: The `StreamingDataLoader` is used by [Lit-GPT](https://github.com/Lightning-AI/lit-gpt/blob/main/pretrain/tinyllama.py) to pretrain LLMs. The statefulness still works when using a mixture of datasets with the `CombinedStreamingDataset`.

```python
import os
import torch
from litdata import StreamingDataset, StreamingDataLoader

dataset = StreamingDataset("s3://my-bucket/my-data", shuffle=True)
dataloader = StreamingDataLoader(dataset, num_workers=os.cpu_count(), batch_size=64)

# Restore the dataLoader state if it exists
if os.path.isfile("dataloader_state.pt"):
    state_dict = torch.load("dataloader_state.pt")
    dataloader.load_state_dict(state_dict)

# Iterate over the data
for batch_idx, batch in enumerate(dataloader):
  
    # Store the state every 1000 batches
    if batch_idx % 1000 == 0:
        torch.save(dataloader.state_dict(), "dataloader_state.pt")
```

## Profiling

The `StreamingDataLoader` supports profiling your data loading. Simply use the `profile_batches` argument as follows:

```python
from litdata import StreamingDataset, StreamingDataLoader

StreamingDataLoader(..., profile_batches=5)
```

This generates a Chrome trace called `result.json`. You can visualize this trace by opening Chrome browser at the `chrome://tracing` URL and load the trace inside.

## Random access

Access the data you need when you need it.

```python
from litdata import StreamingDataset

dataset = StreamingDataset(...)

print(len(dataset)) # display the length of your data

print(dataset[42]) # show the 42th element of the dataset
```

## Use data transforms

```python
from litdata import StreamingDataset, StreamingDataLoader
import torchvision.transforms.v2.functional as F

class ImagenetStreamingDataset(StreamingDataset):

    def __getitem__(self, index):
        image = super().__getitem__(index)
        return F.resize(image, (224, 224))

dataset = ImagenetStreamingDataset(...)
dataloader = StreamingDataLoader(dataset, batch_size=4)

for batch in dataloader:
    print(batch.shape)
    # Out: (4, 3, 224, 224)
```

## Disk usage limits

Limit the size of the cache holding the chunks.

```python
from litdata import StreamingDataset

dataset = StreamingDataset(..., max_cache_size="10GB")
```

## Support yield

When processing large files like compressed [parquet files](https://en.wikipedia.org/wiki/Apache_Parquet), you can use python yield to process and store one item at the time.

```python
from pathlib import Path
import pyarrow.parquet as pq
from litdata import optimize
from tokenizer import Tokenizer
from functools import partial

# 1. Define a function to convert the text within the parquet files into tokens
def tokenize_fn(filepath, tokenizer=None):
    parquet_file = pq.ParquetFile(filepath)
    # Process per batch to reduce RAM usage
    for batch in parquet_file.iter_batches(batch_size=8192, columns=["content"]):
        for text in batch.to_pandas()["content"]:
            yield tokenizer.encode(text, bos=False, eos=True)

# 2. Generate the inputs
input_dir = "/teamspace/s3_connections/tinyllama-template"
inputs = [str(file) for file in Path(f"{input_dir}/starcoderdata").rglob("*.parquet")]

# 3. Store the optimized data wherever you want under "/teamspace/datasets" or "/teamspace/s3_connections"
outputs = optimize(
    fn=partial(tokenize_fn, tokenizer=Tokenizer(f"{input_dir}/checkpoints/Llama-2-7b-hf")), # Note: You can use HF tokenizer or any others
    inputs=inputs,
    output_dir="/teamspace/datasets/starcoderdata",
    chunk_size=(2049 * 8012),
)
```

## Network Drive On-Prem Support

On-prem compute nodes can mount and use network drive. In order to reduce their network overload, the `StreamingDataset` supports `caching` the chunks.

```python
from lightning.data import StreamingDataset

dataset = StreamingDataset(input_dir="local:/data/shared-drive/some-data")
```

# Benchmarks

[Imagenet-1.2M](https://www.image-net.org/) is a commonly used dataset to compare computer vision models. Its training dataset contains `1,281,167 images`. 

In this benchmark, we measured the streaming speed (`images per second`) loaded from [AWS S3](https://aws.amazon.com/s3/) for several frameworks. 

Find the reproducible [Studio Benchmark](https://lightning.ai/lightning-ai/studios/benchmark-cloud-data-loading-libraries).

### Imagenet-1.2M Streaming from AWS S3

| Framework | Images / sec  1st Epoch (float32)  | Images / sec   2nd Epoch (float32) | Images / sec 1st Epoch (torch16) | Images / sec 2nd Epoch (torch16) |
|---|---|---|---|---|
| PL Data  | **5800.34** | **6589.98**  | **6282.17**  | **7221.88**  |
| Web Dataset  | 3134.42 | 3924.95 | 3343.40 | 4424.62 |
| Mosaic ML  | 2898.61 | 5099.93 | 2809.69 | 5158.98 |

Higher is better.

### Imagenet-1.2M Conversion

| Framework |Train Conversion Time | Val Conversion Time | Dataset Size | # Files |
|---|---|---|---|---|
| PL Data  |  **10:05 min** | **00:30 min** | **143.1 GB**  | 2.339  |
| Web Dataset  | 32:36 min | 01:22 min | 147.8 GB | 1.144 |
| Mosaic ML  | 49:49 min | 01:04 min | **143.1 GB** | 2.298 |

The dataset needs to be converted into an optimized format for cloud streaming. We measured how fast the 1.2 million images are converted.

Faster is better.

# Lightning AI Platform: Scale cloud data processing

To scale data processing, create a free account on [lightning.ai](https://lightning.ai/) platform. With the platform, the `optimize` and `map` can start multiple machines to make data processing drastically faster as follows:

```python
from litdata import optimize, Machine

optimize(
  ...
  num_nodes=32,
  machine=Machine.DATA_PREP, # You can select between dozens of optimized machines
)
```

OR

```python
from litdata import map, Machine

map(
  ...
  num_nodes=32,
  machine=Machine.DATA_PREP, # You can select between dozens of optimized machines
)
```

The Data Prep Job UI from the [LAION 400M Studio](https://lightning.ai/lightning-ai/studios/use-or-explore-laion-400million-dataset) where we used 32 machines with 32 CPU each to download 400 million images in only 2 hours.

<div align="center">

<img alt="Lightning" src="https://pl-flash-data.s3.amazonaws.com/data-prep.jpg" width="800px" style="max-width: 100%;">

</div> 

# ⚡ Contributors

We welcome any contributions, pull requests, or issues. If you use the Streaming Dataset for your own project, please reach out to us on Slack or Discord.
