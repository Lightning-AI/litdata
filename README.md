<div align="center">

<img alt="Lightning" src="https://pl-flash-data.s3.amazonaws.com/lightning_data_logo.png" width="800px" style="max-width: 100%;">

<br/>
<br/>

## Blazingly fast, distributed streaming of training data from any cloud storage

</div>

# ⚡ Welcome to Lightning Data

With Lightning Data, users can transform and optimize their data in cloud storage environments with an intuitive approach at any scale. 

Then, efficient distributed training becomes feasible regardless of the data's location, allowing users to effortlessly stream your data as needed.

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

```bash
pip install litdata
```

## Quick Start

### 1. Prepare Your Data

Convert your raw dataset into Lightning Streaming Format using the `optimize` operator.

Here is an example with some random images. 

```python
import numpy as np
from litdata import optimize
from PIL import Image


# Store random images into the data chunks
def random_images(index):
    data = {
        "index": index, # int data type
        "image": Image.fromarray(np.random.randint(0, 256, (32, 32, 3), np.uint8)), # PIL image data type
        "class": np.random.randint(10), # numpy array data type
    }
    # The data is serialized into bytes and stored into data chunks by the optimize operator.
    return data # The data is serialized into bytes and stored into data chunks by the optimize operator.

if __name__ == "__main__":
    optimize(
        fn=random_images,  # The function applied over each input.
        inputs=list(range(1000)),  # Provide any inputs. The fn is applied on each item.
        output_dir="my_optimized_dataset",  # The directory where the optimized data are stored.
        num_workers=4,  # The number of workers. The inputs are distributed among them.
        chunk_bytes="64MB"  # The maximum number of bytes to write into a chunk.
    )

```

The `optimize` operator supports any data structures and types. Serialize whatever you want. The optimized data are stored under the output directory `my_optimized_dataset`.

### 2. Upload Your Data to Cloud Storage

Cloud providers such as [AWS](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), [Google Cloud](https://cloud.google.com/storage/docs/uploading-objects?hl=en#upload-object-cli), [Azure](https://learn.microsoft.com/en-us/azure/import-export/storage-import-export-data-to-files?tabs=azure-portal-preview), etc.. provide command line client to upload your data to their storage.

Here is how to upload the optimized dataset using the [AWS CLI](https://aws.amazon.com/s3) to [AWS S3](https://aws.amazon.com/s3/).

```bash
⚡ aws s3 cp --recursive my_optimized_dataset s3://my-bucket/my_optimized_dataset
```

### 3. Use StreamingDataset

Then, the Streaming Dataset can read the data directly from [AWS S3](https://aws.amazon.com/s3/).

```python
from litdata import StreamingDataset
from torch.utils.data import DataLoader

# Remote path where full dataset is stored
input_dir = 's3://my-bucket/my_optimized_dataset'

# Create the Streaming Dataset
dataset = StreamingDataset(input_dir, shuffle=True)

# You can access any elements of the dataset
sample = dataset[50]
img = sample['image']
cls = sample['class']

# Create PyTorch DataLoader and iterate over it to train your AI models.
dataloader = DataLoader(dataset)
```

# Real World Examples

We have built end-to-end free [Studios](https://lightning.ai) showing all the steps to prepare several data types of data. The [Studios](https://lightning.ai) are fully reproducible cloud IDE with data, code, dependencies, e.g so you can re-do everything yourself with ease !

| Studio | Data type | Dataset |
| -------------------------------------------------------------------------------------------------------------------------------------------- | :-----------------: | --------------------------------------------------------------------------------------------------------------------------------------: |
| [Use or explore LAION-400MILLION dataset](https://lightning.ai/lightning-ai/studios/use-or-explore-laion-400million-dataset)                                                                                  | Image & description |[LAION-400M](https://laion.ai/blog/laion-400-open-dataset/) |
| [Convert GeoSpatial data to Lightning Streaming](https://lightning.ai/lightning-ai/studios/convert-spatial-data-to-lightning-streaming) [Chesapeake Roads Spatial Context](https://github.com/isaaccorley/chesapeakersc)                                                             |    Image & Mask     |  [Chesapeake Roads Spatial Context](https://github.com/isaaccorley/chesapeakersc) |
| [Benchmark cloud data-loading libraries](https://lightning.ai/lightning-ai/studios/benchmark-cloud-data-loading-libraries)                                               |    Image & Label    | [Imagenet 1M](https://paperswithcode.com/sota/image-classification-on-imagenet?tag_filter=171) |
| [Prepare the TinyLlama 1T token dataset](https://lightning.ai/lightning-ai/studios/prepare-the-tinyllama-1t-token-dataset) |        Text         |              [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) & [StartCoder](https://huggingface.co/datasets/bigcode/starcoderdata) |
| [Tokenize 2M Swedish Wikipedia Articles](https://lightning.ai/lightning-ai/studios/tokenize-2m-swedish-wikipedia-articles) |        Text         |              [Swedish Wikipedia](https://huggingface.co/datasets/wikipedia) |
| [Embed English Wikipedia under 5 dollars](https://lightning.ai/lightning-ai/studios/embed-english-wikipedia-under-5-dollars)                                                                               |        Text         |            [English Wikipedia](https://huggingface.co/datasets/wikipedia) |
| [Convert parquets to Lightning Streaming](https://lightning.ai/lightning-ai/studios/convert-parquets-to-lightning-streaming)                                                                                                                                    |    Parquet Files    | Randomly Generated data |

# Key Features

- [Multi-GPU / Multi-Node](#multi-gpu--multi-node)
- [Easy Data Mixing](#easy-data-mixing)
- [Stateful StreamingDataLoader](#stateful-streamingdataloader)
- [Profiling](#profiling)
- [Random access](#random-access)
- [Use data transforms](#use-data-transforms)
- [Disk usage limits](#disk-usage-limits)
- [Support the python yield keyword](#support-python-yield-keyword)
- [Network Drive Storage](#on-prem-storage-with-network-drive)
- [Map Operator](#map-operator)

## Multi-GPU / Multi-Node

The `StreamingDataset` and `StreamingDataLoader` take care of everything for you. They automatically make sure each rank receives the same quantity of varied batches of data. 

![An illustration showing how the Streaming Dataset works with multi node.](https://pl-flash-data.s3.amazonaws.com/streaming_dataset.gif)

## Easy Data Mixing with the Combined Streaming Dataset

You can easily experiment with dataset mixtures using the `CombinedStreamingDataset` class. 

As an example, this mixture of [Slimpajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) & [StarCoder](https://huggingface.co/datasets/bigcode/starcoderdata) was used in the [TinyLLAMA](https://github.com/jzhang38/TinyLlama) project to pretrain a 1.1B Llama model on 3 trillion tokens. 

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

## Stateful Streaming DataLoader

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

The `StreamingDataLoader` supports profiling your dataloading. Simply use the `profile_batches` argument to set how many batches to profile:

```python
from litdata import StreamingDataset, StreamingDataLoader

StreamingDataLoader(..., profile_batches=5)
```

This generates a Chrome trace called `result.json`. You can visualize this trace by opening Chrome browser at the `chrome://tracing` URL and load the trace inside.

## Random access

Access the data you need when you need it whenever they are stored.

```python
from litdata import StreamingDataset

dataset = StreamingDataset("s3://my-bucket/my-data") # data are stored in the cloud

print(len(dataset)) # display the length of your data

print(dataset[42]) # show the 42th element of the dataset
```

## Use data transforms

You can subclass the `StreamingDataset` and override its `__getitem__` method to add any extra data transformations.

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

Adapt the local caching limit of the `StreamingDataset`.

```python
from litdata import StreamingDataset

dataset = StreamingDataset(..., max_cache_size="10GB")
```

## Support python yield keyword

When processing large files like compressed [parquet files](https://en.wikipedia.org/wiki/Apache_Parquet), you can use the python yield keyword to process and store one item at the time, reducing the memory footprint of the entire program. 

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
    chunk_size=(2049 * 8012), # Number of tokens to store by chunks. This is roughly 64MB of tokens per chunk.
)
```

## On-Prem Storage with Network Drive

A network drive is a shared storage device on a local area network. On-prem compute nodes can mount and use network drive. In order to reduce their network overload, the `StreamingDataset` supports `caching` the chunks.

```python
from lightning.data import StreamingDataset

dataset = StreamingDataset(input_dir="local:/data/shared-drive/some-data")
```

## Map Operator

The `map` operator can be used to apply a function over a list of inputs.

Here is an example where the `map` operator is used to apply a `resize_image` function over a folder of large images.

```python
from lightning.data import map
from PIL import Image

input_dir = "my_large_images"
inputs = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

# The resize image takes one of the input (image_path) and the output directory. Files written to output_dir are persisted.
def resize_image(image_path, output_dir):
  output_image_path = os.path.join(output_dir, os.path.basename(image_path))
  Image.open(image_path).resize((224, 224)).save(output_image_path)
  
map(
    fn=resize_image,
    inputs=inputs, 
    output_dir="my_resized_images",
)
```

# Benchmarks

[Imagenet-1.2M](https://www.image-net.org/) is a commonly used dataset to compare computer vision models. Its training dataset contains `1,281,167 images`. 

In this benchmark, we measured the streaming speed (`images per second`) loaded from [AWS S3](https://aws.amazon.com/s3/) for several frameworks. 

You can reproduce this benchmark with this [Studio](https://lightning.ai/lightning-ai/studios/benchmark-cloud-data-loading-libraries). 

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

To scale data processing, create a free account on the [Lightning.ai](https://lightning.ai/) platform. 

With the platform, the `map` operator can start multiple machines to make data processing drastically faster as follows:

```python
from litdata import map, Machine

map(
  ...
  num_nodes=32,
  machine=Machine.DATA_PREP, # You can select between dozens of optimized machines
)
```

Also, the `optimize`operator  can do the same to make immense dataset streamable as follows:

```python
from litdata import optimize, Machine

optimize(
  ...
  num_nodes=32,
  machine=Machine.DATA_PREP, # You can select between dozens of optimized machines
)
```

Here is a screenshot from the [LAION 400M Studio](https://lightning.ai/lightning-ai/studios/use-or-explore-laion-400million-dataset) where we used 32 machines with 32 CPU each to download 400 million images in only 2 hours. You can run this entire Studio to reproduce on your own.

<div align="center">

<img alt="Lightning" src="https://pl-flash-data.s3.amazonaws.com/data-prep.jpg" width="800px" style="max-width: 100%;">

</div> 

# ⚡ Contributors

We welcome any contributions, pull requests, or issues. If you use the Streaming Dataset for your own project, please reach out to us on [Discord](https://discord.com/invite/XncpTy7DSt).
