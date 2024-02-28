<div align="center">

<img alt="Lightning" src="https://pl-flash-data.s3.amazonaws.com/lightning_data_logo.png" width="800px" style="max-width: 100%;">

<br/>
<br/>

## Blazingly fast, distributed streaming of training data from any cloud storage

</div>

# ⚡ Welcome to Lightning Data

With Lightning Data, users can transform and optimize their data in cloud storage environments with an intuitive approach at any scale. 

Then, efficient distributed training becomes feasible regardless of the data's location, allowing users to effortlessly stream their data as needed with one or many machines.

Lightning Data supports **images, text, video, audio, geo-spatial, and multimodal data** types, is already adopted by frameworks such as [Lit-GPT](https://github.com/Lightning-AI/lit-gpt/blob/main/pretrain/tinyllama.py) to pretrain LLMs and integrates smoothly with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), [Lightning Fabric](https://lightning.ai/docs/fabric/stable/), and [PyTorch](https://pytorch.org/docs/stable/index.html).

[Runnable templates](#runnable-templates) at the end are available, they are fully reproducible with 1-click.

### Table of Contents

- [Getting started](#getting-started)
    - [Installation](#installation)
    - [Quick Start](#quick-start)
        - [1. Prepare Your Data](#1-prepare-your-data)
        - [2. Upload Your Data to Cloud Storage](#2-upload-your-data-to-cloud-storage)
        - [3. Use StreamingDataset](#3-use-streamingdataset)
- [Key Features](#key-features)
- [Benchmarks](#benchmarks)
- [Runnable Templates](#runnable-templates)
- [Infinite cloud data processing](#infinite-cloud-data-processing)
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
        chunk_bytes="64MB"  # The maximum number of bytes to write into a data chunk.
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

# Access any elements of the dataset
sample = dataset[50]
img = sample['image']
cls = sample['class']

# Create PyTorch DataLoader and iterate over it to train your AI models.
dataloader = DataLoader(dataset)
```

# Key Features

- [Multi-GPU / Multi-Node Support](#built-in-multi-gpu--multi-node)
- [Access any item](#access-any-item)
- [Add any data transforms](#add-any-data-transforms)
- [The Map Operator](#the-map-operator)
- [Easy Data Mixing with the Combined Streaming Dataset](#easy-data-mixing-with-the-combined-streaming-dataset)
- [Continue Training From Previous Dataset State](#continue-training-from-previous-dataset-state)
- [Support Profiling](#support-profiling)
- [Reduce your memory footprint](#reduce-your-memory-footprint)
- [Configure Cache Size Limit](#configure-cache-size-limit)
- [On-Prem Optimizations](#on-prem-optimizations)

## Multi-GPU / Multi-Node Support

The `StreamingDataset` and `StreamingDataLoader` automatically make sure each rank receives the same quantity of varied batches of data, so it works out of the box with your favorites frameworks ([PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), [Lightning Fabric](https://lightning.ai/docs/fabric/stable/), or [PyTorch](https://pytorch.org/docs/stable/index.html)) to do distributed training. 

Here is an illustration showing how the Streaming Dataset works with multi node / multi gpu under the hood.

![An illustration showing how the Streaming Dataset works with multi node.](https://pl-flash-data.s3.amazonaws.com/streaming_dataset.gif)

## Access any item

Access the data you need when you need it whenever they are stored.

```python
from litdata import StreamingDataset

dataset = StreamingDataset("s3://my-bucket/my-data") # data are stored in the cloud

print(len(dataset)) # display the length of your data

print(dataset[42]) # show the 42th element of the dataset
```

## Add any data transforms

Subclass the `StreamingDataset` and override its `__getitem__` method to add any extra data transformations.

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

## The Map Operator

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

## Easy Data Mixing with the Combined Streaming Dataset

Easily experiment with dataset mixtures using the `CombinedStreamingDataset` class. 

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

## Continue Training From Previous Dataset State

Lightning Data provides a stateful `Streaming DataLoader` e.g. you can `pause` and `resume` your training whenever you want.

Info: The `Streaming DataLoader` was used by [Lit-GPT](https://github.com/Lightning-AI/lit-gpt/blob/main/pretrain/tinyllama.py) to pretrain LLMs. Restarting from an older checkpoint was critical to get to pretrain the full model due several failures (network, CUDA Errors, etc..).

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

## Support Profiling

The `StreamingDataLoader` supports profiling your dataloading. Simply use the `profile_batches` argument to set how many batches to profile:

```python
from litdata import StreamingDataset, StreamingDataLoader

StreamingDataLoader(..., profile_batches=5)
```

This generates a Chrome trace called `result.json`. Then, visualize this trace by opening Chrome browser at the `chrome://tracing` URL and load the trace inside.

## Reduce your memory footprint

When processing large files like compressed [parquet files](https://en.wikipedia.org/wiki/Apache_Parquet), use the python yield keyword to process and store one item at the time, reducing the memory footprint of the entire program. 

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
    fn=partial(tokenize_fn, tokenizer=Tokenizer(f"{input_dir}/checkpoints/Llama-2-7b-hf")), # Note: Use HF tokenizer or any others
    inputs=inputs,
    output_dir="/teamspace/datasets/starcoderdata",
    chunk_size=(2049 * 8012), # Number of tokens to store by chunks. This is roughly 64MB of tokens per chunk.
)
```

## Configure Cache Size Limit

Adapt the local caching limit of the `StreamingDataset`. This is useful to make sure the downloaded data chunks are deleted when used and the disk usage stays low.

```python
from litdata import StreamingDataset

dataset = StreamingDataset(..., max_cache_size="10GB")
```

## On-Prem Optimizations

On-prem compute nodes can mount and use a network drive. A network drive is a shared storage device on a local area network. In order to reduce their network overload, the `StreamingDataset` supports `caching` the data chunks.

```python
from lightning.data import StreamingDataset

dataset = StreamingDataset(input_dir="local:/data/shared-drive/some-data")
```

# Benchmarks

In order to measure the effectiveness of Lightning Data, we used a commonly used dataset for benchmarks: [Imagenet-1.2M](https://www.image-net.org/) where the training set contains `1,281,167 images`. 

To align with other benchmarks, we measured the streaming speed (`images per second`) loaded from [AWS S3](https://aws.amazon.com/s3/) for several frameworks. 

Reproduce our benchmark **by running** this [Studio](https://lightning.ai/lightning-ai/studios/benchmark-cloud-data-loading-libraries). 

### Imagenet-1.2M Streaming from AWS S3

We can observe Lightning Data is up to 85 % faster than the second best. Higher is better in the table below. 

| Framework | Images / sec  1st Epoch (float32)  | Images / sec   2nd Epoch (float32) | Images / sec 1st Epoch (torch16) | Images / sec 2nd Epoch (torch16) |
|---|---|---|---|---|
| PL Data  | **5800.34** | **6589.98**  | **6282.17**  | **7221.88**  |
| Web Dataset  | 3134.42 | 3924.95 | 3343.40 | 4424.62 |
| Mosaic ML  | 2898.61 | 5099.93 | 2809.69 | 5158.98 |

### Imagenet-1.2M Conversion

We measured how fast the 1.2 million images can converted into a streamable format. Faster is better in the table below.

| Framework |Train Conversion Time | Val Conversion Time | Dataset Size | # Files |
|---|---|---|---|---|
| PL Data  |  **10:05 min** | **00:30 min** | **143.1 GB**  | 2.339  |
| Web Dataset  | 32:36 min | 01:22 min | 147.8 GB | 1.144 |
| Mosaic ML  | 49:49 min | 01:04 min | **143.1 GB** | 2.298 |


# Runnable Templates

Fastest way to learn is with [Studios](https://lightning.ai/studios).  

[Studios](https://lightning.ai/studios) are reproducible cloud IDE with data, code, dependencies, e.g. so redo everything yourself with ease!

We've published [public templates](https://lightning.ai/studios) that demonstrates how best to use the Lightning Data framework at scale and with several data types.

Sign up [here](https://lightning.ai/) and run your first Studio for free.

| Studio | Data type | Dataset |
| -------------------------------------------------------------------------------------------------------------------------------------------- | :-----------------: | --------------------------------------------------------------------------------------------------------------------------------------: |
| [Use or explore LAION-400MILLION dataset](https://lightning.ai/lightning-ai/studios/use-or-explore-laion-400million-dataset)                                                                                  | Image & Text |[LAION-400M](https://laion.ai/blog/laion-400-open-dataset/) |
| [Convert GeoSpatial data to Lightning Streaming](https://lightning.ai/lightning-ai/studios/convert-spatial-data-to-lightning-streaming) |    Image & Mask     |  [Chesapeake Roads Spatial Context](https://github.com/isaaccorley/chesapeakersc) |
| [Benchmark cloud data-loading libraries](https://lightning.ai/lightning-ai/studios/benchmark-cloud-data-loading-libraries)                                               |    Image & Label    | [Imagenet 1M](https://paperswithcode.com/sota/image-classification-on-imagenet?tag_filter=171) |
| [Prepare the TinyLlama 1T token dataset](https://lightning.ai/lightning-ai/studios/prepare-the-tinyllama-1t-token-dataset) |        Text         |              [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) & [StarCoder](https://huggingface.co/datasets/bigcode/starcoderdata) |
| [Tokenize 2M Swedish Wikipedia Articles](https://lightning.ai/lightning-ai/studios/tokenize-2m-swedish-wikipedia-articles) |        Text         |              [Swedish Wikipedia](https://huggingface.co/datasets/wikipedia) |
| [Embed English Wikipedia under 5 dollars](https://lightning.ai/lightning-ai/studios/embed-english-wikipedia-under-5-dollars)                                                                               |        Text         |            [English Wikipedia](https://huggingface.co/datasets/wikipedia) |
| [Convert parquets to Lightning Streaming](https://lightning.ai/lightning-ai/studios/convert-parquets-to-lightning-streaming)                                                                                                                                    |    Parquet Files    | Randomly Generated data |

# Infinite cloud data processing

If you want to scale data processing, you typically need more machines and if you do this yourself, this becomes very tedious and can take a long time to get there.

Instead, create a free account on the [Lightning.ai](https://lightning.ai/) platform and use as many machines as you need from code.

On the platform, simply specify the number of nodes and the machine type you need as follows:

```python
from litdata import map, Machine

map(
  ...
  num_nodes=32,
  machine=Machine.DATA_PREP, # Select between dozens of optimized machines
)
```

Also, the `optimize` operator  can do the same to make immense dataset streamable as follows:

```python
from litdata import optimize, Machine

optimize(
  ...
  num_nodes=32,
  machine=Machine.DATA_PREP, # Select between dozens of optimized machines
)
```

Here is a screenshot from the [LAION 400 MILLION Studio](https://lightning.ai/lightning-ai/studios/use-or-explore-laion-400million-dataset) on the [Lightning.ai](https://lightning.ai/) platform where we used 32 machines with 32 CPU each to download 400 million images in only 2 hours. You can run this template yourself [here](https://lightning.ai/lightning-ai/studios/use-or-explore-laion-400million-dataset).

<div align="center">

<img alt="Lightning" src="https://pl-flash-data.s3.amazonaws.com/data-prep.jpg" width="800px" style="max-width: 100%;">

</div> 

# ⚡ Contributors

We welcome any contributions, pull requests, or issues. If you use the Streaming Dataset for your own project, please reach out to us on [Discord](https://discord.com/invite/XncpTy7DSt).
