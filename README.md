<div align="center">
<img src="https://pl-flash-data.s3.amazonaws.com/lit_data_logo.webp" alt="LitData" width="800px"/>

&nbsp;
&nbsp;

**Transform datasets at scale.    
Optimize data for fast AI model training.**


<pre>
Transform                              Optimize
  
✅ Parallelize data processing       ✅ Stream large cloud datasets          
✅ Create vector embeddings          ✅ Accelerate training by 20x           
✅ Run distributed inference         ✅ Pause and resume data streaming      
✅ Scrape websites at scale          ✅ Use remote data without local loading
</pre>

---

![PyPI](https://img.shields.io/pypi/v/litdata)
![Downloads](https://img.shields.io/pypi/dm/litdata)
![License](https://img.shields.io/github/license/Lightning-AI/litdata)
[![Discord](https://img.shields.io/discord/1077906959069626439?label=Get%20Help%20on%20Discord)](https://discord.gg/VptPCZkGNa)

<p align="center">
  <a href="https://lightning.ai/">Lightning AI</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#speed-up-model-training">Optimize data</a> •
  <a href="#transform-datasets">Transform data</a> •
  <a href="#key-features">Features</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#start-from-a-template">Templates</a> •
  <a href="#community">Community</a>
</p>

&nbsp;

<a target="_blank" href="https://lightning.ai/docs/overview/prep-data/optimize-datasets-for-model-training-speed">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/get-started-badge.svg" height="36px" alt="Get started"/>
</a>

</div>

&nbsp;

# Transform data at scale. Optimize for fast model training.
LitData scales [data processing tasks](#transform-datasets) (data scraping, image resizing, distributed inference, embedding creation) on local or cloud machines. It also enables [optimizing datasets](#speed-up-model-training) to accelerate AI model training and work with large remote datasets without local loading.

&nbsp;

# Quick start
First, install LitData:

```bash
pip install litdata
```

Choose your workflow:

🚀 [Speed up model training](#speed-up-model-training)    
🚀 [Transform datasets](#transform-datasets)

&nbsp;

<details>
  <summary>Advanced install</summary>

Install all the extras
```bash
pip install 'litdata[extras]'
```

</details>

&nbsp;

----

# Speed up model training
Accelerate model training (20x faster) by optimizing datasets for streaming directly from cloud storage. Work with remote data without local downloads with features like loading data subsets, accessing individual samples, and resumable streaming.

**Step 1: Optimize the data**
This step will format the dataset for fast loading. The data will be written in a chunked binary format.

```python
import numpy as np
from PIL import Image
import litdata as ld

def random_images(index):
    fake_images = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
    fake_labels = np.random.randint(10)

    # You can use any key:value pairs. Note that their types must not change between samples, and Python lists must
    # always contain the same number of elements with the same types.
    data = {"index": index, "image": fake_images, "class": fake_labels}

    return data

if __name__ == "__main__":
    # The optimize function writes data in an optimized format.
    ld.optimize(
        fn=random_images,                   # the function applied to each input
        inputs=list(range(1000)),           # the inputs to the function (here it's a list of numbers)
        output_dir="fast_data",             # optimized data is stored here
        num_workers=4,                      # The number of workers on the same machine
        chunk_bytes="64MB"                  # size of each chunk
    )
```

**Step 2: Put the data on the cloud**

Upload the data to a [Lightning Studio](https://lightning.ai) (backed by S3) or your own S3 bucket:
```bash
aws s3 cp --recursive fast_data s3://my-bucket/fast_data
```

**Step 3: Stream the data during training**

Load the data by replacing the PyTorch DataSet and DataLoader with the StreamingDataset and StreamingDataloader

```python
import litdata as ld

train_dataset = ld.StreamingDataset('s3://my-bucket/fast_data', shuffle=True, drop_last=True)
train_dataloader = ld.StreamingDataLoader(train_dataset)

for sample in train_dataloader:
    img, cls = sample['image'], sample['class']
```

**Key benefits:**

✅ Accelerate training:       Optimized datasets load 20x faster.      
✅ Stream cloud datasets:     Work with cloud data without downloading it.    
✅ Pytorch-first:             Works with PyTorch libraries like PyTorch Lightning, Lightning Fabric, Hugging Face.    
✅ Easy collaboration:        Share and access datasets in the cloud, streamlining team projects.     
✅ Scale across GPUs:         Streamed data automatically scales to all GPUs.      
✅ Flexible storage:          Use S3, GCS, Azure, or your own cloud account for data storage.    
✅ Compression:               Reduce your data footprint by using advanced compression algorithms.  
✅ Run local or cloud:        Run on your own machines or auto-scale to 1000s of cloud GPUs with Lightning Studios.         
✅ Enterprise security:       Self host or process data on your cloud account with Lightning Studios.  

&nbsp;

----

# Transform datasets
Accelerate data processing tasks (data scraping, image resizing, embedding creation, distributed inference) by parallelizing (map) the work across many machines at once.

Here's an example that resizes and crops a large image dataset:

```python
from PIL import Image
import litdata as ld

# use a local or S3 folder
input_dir = "my_large_images"     # or "s3://my-bucket/my_large_images"
output_dir = "my_resized_images"  # or "s3://my-bucket/my_resized_images"

inputs = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

# resize the input image
def resize_image(image_path, output_dir):
  output_image_path = os.path.join(output_dir, os.path.basename(image_path))
  Image.open(image_path).resize((224, 224)).save(output_image_path)

ld.map(
    fn=resize_image,
    inputs=inputs,
    output_dir="output_dir",
)
```

**Key benefits:**

✅ Parallelize processing:    Reduce processing time by transforming data across multiple machines simultaneously.    
✅ Scale to large data:       Increase the size of datasets you can efficiently handle.    
✅ Flexible usecases:         Resize images, create embeddings, scrape the internet, etc...    
✅ Run local or cloud:        Run on your own machines or auto-scale to 1000s of cloud GPUs with Lightning Studios.         
✅ Enterprise security:       Self host or process data on your cloud account with Lightning Studios.  

&nbsp;

----

# Key Features

## Features for optimizing and streaming datasets for model training

<details>
  <summary> ✅ Stream large cloud datasets</summary>
&nbsp;

Use data stored on the cloud without needing to download it all to your computer, saving time and space.

Imagine you're working on a project with a huge amount of data stored online. Instead of waiting hours to download it all, you can start working with the data almost immediately by streaming it.

Once you've optimized the dataset with LitData, stream it as follows:
```python
from litdata import StreamingDataset, StreamingDataLoader

dataset = StreamingDataset('s3://my-bucket/my-data', shuffle=True)
dataloader = StreamingDataLoader(dataset, batch_size=64)

for batch in dataloader:
    process(batch)  # Replace with your data processing logic

```


Additionally, you can inject client connection settings for [S3](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/core/session.html#boto3.session.Session.client) or GCP when initializing your dataset. This is useful for specifying custom endpoints and credentials per dataset.

```python
from litdata import StreamingDataset

storage_options = {
    "key": "your_access_key_id",
    "secret": "your_secret_access_key",
}

dataset = StreamingDataset('s3://my-bucket/my-data', storage_options=storage_options)
```

</details>

<details>
  <summary> ✅ Streams on multi-GPU, multi-node</summary>

&nbsp;

Data optimized and loaded with Lightning automatically streams efficiently in distributed training across GPUs or multi-node.

The `StreamingDataset` and `StreamingDataLoader` automatically make sure each rank receives the same quantity of varied batches of data, so it works out of the box with your favorite frameworks ([PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), [Lightning Fabric](https://lightning.ai/docs/fabric/stable/), or [PyTorch](https://pytorch.org/docs/stable/index.html)) to do distributed training.

Here you can see an illustration showing how the Streaming Dataset works with multi node / multi gpu under the hood.

```python
from litdata import StreamingDataset, StreamingDataLoader

# For the training dataset, don't forget to enable shuffle and drop_last !!! 
train_dataset = StreamingDataset('s3://my-bucket/my-train-data', shuffle=True, drop_last=True)
train_dataloader = StreamingDataLoader(train_dataset, batch_size=64)

for batch in train_dataloader:
    process(batch)  # Replace with your data processing logic

val_dataset = StreamingDataset('s3://my-bucket/my-val-data', shuffle=False, drop_last=False)
val_dataloader = StreamingDataLoader(val_dataset, batch_size=64)

for batch in val_dataloader:
    process(batch)  # Replace with your data processing logic
```

![An illustration showing how the Streaming Dataset works with multi node.](https://pl-flash-data.s3.amazonaws.com/streaming_dataset.gif)

</details>

<details>
  <summary> ✅ Stream from multiple cloud providers</summary>

&nbsp;

The StreamingDataset supports reading optimized datasets from common cloud providers.

```python
import os
import litdata as ld

# Read data from AWS S3
aws_storage_options={
    "key": os.environ['AWS_ACCESS_KEY_ID'],
    "secret": os.environ['AWS_SECRET_ACCESS_KEY'],
}
dataset = ld.StreamingDataset("s3://my-bucket/my-data", storage_options=aws_storage_options)

# Read data from GCS
gcp_storage_options={
    "token": {
      # dumped from cat ~/.config/gcloud/application_default_credentials.json 
        "account": "",
        "client_id": "your_client_id",
        "client_secret": "your_client_secret",
        "quota_project_id": "your_quota_project_id",
        "refresh_token": "your_refresh_token",
        "type": "authorized_user",
        "universe_domain": "googleapis.com",
    }
}
dataset = ld.StreamingDataset("gs://my-bucket/my-data", storage_options=gcp_storage_options)

# Read data from Azure
azure_storage_options={
    "account_name": "azure_account_name",
    "account_key": os.environ['AZURE_ACCOUNT_ACCESS_KEY']
}
dataset = ld.StreamingDataset("azure://my-bucket/my-data", storage_options=azure_storage_options)
```

- For more details on which storage options are supported, please refer to:
  - [AWS S3 storage options](https://github.com/fsspec/s3fs/blob/main/s3fs/core.py#L176)
  - [GCS storage options](https://github.com/fsspec/gcsfs/blob/main/gcsfs/core.py#L154)
  - [Azure storage options](https://github.com/fsspec/adlfs/blob/main/adlfs/spec.py#L124)

</details>  

<details>
  <summary> ✅ Pause, resume data streaming</summary>
&nbsp;

Stream data during long training, if interrupted, pick up right where you left off without any issues.

LitData provides a stateful `Streaming DataLoader` e.g. you can `pause` and `resume` your training whenever you want.

Info: The `Streaming DataLoader` was used by [Lit-GPT](https://github.com/Lightning-AI/lit-gpt/blob/main/pretrain/tinyllama.py) to pretrain LLMs. Restarting from an older checkpoint was critical to get to pretrain the full model due to several failures (network, CUDA Errors, etc..).

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

</details>


<details>
  <summary> ✅ LLM Pre-training </summary>
&nbsp;

LitData is highly optimized for LLM pre-training. First, we need to tokenize the entire dataset and then we can consume it.

```python
import json
from pathlib import Path
import zstandard as zstd
from litdata import optimize, TokensLoader
from tokenizer import Tokenizer
from functools import partial

# 1. Define a function to convert the text within the jsonl files into tokens
def tokenize_fn(filepath, tokenizer=None):
    with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
        for row in f:
            text = json.loads(row)["text"]
            if json.loads(row)["meta"]["redpajama_set_name"] == "RedPajamaGithub":
                continue  # exclude the GitHub data since it overlaps with starcoder
            text_ids = tokenizer.encode(text, bos=False, eos=True)
            yield text_ids

if __name__ == "__main__":
    # 2. Generate the inputs (we are going to optimize all the compressed json files from SlimPajama dataset )
    input_dir = "./slimpajama-raw"
    inputs = [str(file) for file in Path(f"{input_dir}/SlimPajama-627B/train").rglob("*.zst")]

    # 3. Store the optimized data wherever you want under "/teamspace/datasets" or "/teamspace/s3_connections"
    outputs = optimize(
        fn=partial(tokenize_fn, tokenizer=Tokenizer(f"{input_dir}/checkpoints/Llama-2-7b-hf")), # Note: You can use HF tokenizer or any others
        inputs=inputs,
        output_dir="./slimpajama-optimized",
        chunk_size=(2049 * 8012),
        # This is important to inform LitData that we are encoding contiguous 1D array (tokens). 
        # LitData skips storing metadata for each sample e.g all the tokens are concatenated to form one large tensor.
        item_loader=TokensLoader(),
    )
```

```python
import os
from litdata import StreamingDataset, StreamingDataLoader, TokensLoader
from tqdm import tqdm

# Increase by one because we need the next word as well
dataset = StreamingDataset(
  input_dir=f"./slimpajama-optimized/train",
  item_loader=TokensLoader(block_size=2048 + 1),
  shuffle=True,
  drop_last=True,
)

train_dataloader = StreamingDataLoader(dataset, batch_size=8, pin_memory=True, num_workers=os.cpu_count())

# Iterate over the SlimPajama dataset
for batch in tqdm(train_dataloader):
    pass
```

</details>

<details>
  <summary> ✅ Combine datasets</summary>
&nbsp;

Mix and match different sets of data to experiment and create better models.

Combine datasets with `CombinedStreamingDataset`.  As an example, this mixture of [Slimpajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) & [StarCoder](https://huggingface.co/datasets/bigcode/starcoderdata) was used in the [TinyLLAMA](https://github.com/jzhang38/TinyLlama) project to pretrain a 1.1B Llama model on 3 trillion tokens.

```python
from litdata import StreamingDataset, CombinedStreamingDataset, StreamingDataLoader, TokensLoader
from tqdm import tqdm
import os

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
combined_dataset = CombinedStreamingDataset(datasets=train_datasets, seed=42, weights=weights, iterate_over_all=False)

train_dataloader = StreamingDataLoader(combined_dataset, batch_size=8, pin_memory=True, num_workers=os.cpu_count())

# Iterate over the combined datasets
for batch in tqdm(train_dataloader):
    pass
```
</details>

<details>
  <summary> ✅ Split datasets for train, val, test</summary>

&nbsp;

Split a dataset into train, val, test splits with `train_test_split`.

```python
from litdata import StreamingDataset, train_test_split

dataset = StreamingDataset("s3://my-bucket/my-data") # data are stored in the cloud

print(len(dataset)) # display the length of your data
# out: 100,000

train_dataset, val_dataset, test_dataset = train_test_split(dataset, splits=[0.3, 0.2, 0.5])

print(train_dataset)
# out: 30,000

print(val_dataset)
# out: 20,000

print(test_dataset)
# out: 50,000
```

</details>

<details>
  <summary> ✅ Load a subset of the remote dataset</summary>

&nbsp;
Work on a smaller, manageable portion of your data to save time and resources.


```python
from litdata import StreamingDataset, train_test_split

dataset = StreamingDataset("s3://my-bucket/my-data", subsample=0.01) # data are stored in the cloud

print(len(dataset)) # display the length of your data
# out: 1000
```

</details>

<details>
  <summary> ✅ Easily modify optimized cloud datasets</summary>
&nbsp;

Add new data to an existing dataset or start fresh if needed, providing flexibility in data management.

LitData optimized datasets are assumed to be immutable. However, you can make the decision to modify them by changing the mode to either `append` or `overwrite`.

```python
from litdata import optimize, StreamingDataset

def compress(index):
    return index, index**2

if __name__ == "__main__":
    # Add some data
    optimize(
        fn=compress,
        inputs=list(range(100)),
        output_dir="./my_optimized_dataset",
        chunk_bytes="64MB",
    )

    # Later on, you add more data
    optimize(
        fn=compress,
        inputs=list(range(100, 200)),
        output_dir="./my_optimized_dataset",
        chunk_bytes="64MB",
        mode="append",
    )

    ds = StreamingDataset("./my_optimized_dataset")
    assert len(ds) == 200
    assert ds[:] == [(i, i**2) for i in range(200)]
```

The `overwrite` mode will delete the existing data and start from fresh.

</details>

<details>
  <summary> ✅ Use compression</summary>
&nbsp;

Reduce your data footprint by using advanced compression algorithms.

```python
import litdata as ld

def compress(index):
    return index, index**2

if __name__ == "__main__":
    # Add some data
    ld.optimize(
        fn=compress,
        inputs=list(range(100)),
        output_dir="./my_optimized_dataset",
        chunk_bytes="64MB",
        num_workers=1,
        compression="zstd"
    )
```

Using [zstd](https://github.com/facebook/zstd), you can achieve high compression ratio like 4.34x for this simple example.

| Without | With |
| -------- | -------- | 
| 2.8kb | 646b |


</details>

<details>
  <summary> ✅ Access samples without full data download</summary>
&nbsp;

Look at specific parts of a large dataset without downloading the whole thing or loading it on a local machine.

```python
from litdata import StreamingDataset

dataset = StreamingDataset("s3://my-bucket/my-data") # data are stored in the cloud

print(len(dataset)) # display the length of your data

print(dataset[42]) # show the 42th element of the dataset
```

</details>

<details>
  <summary> ✅ Use any data transforms</summary>
&nbsp;

Customize how your data is processed to better fit your needs.

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

</details>

<details>
  <summary> ✅ Profile data loading speed</summary>
&nbsp;

Measure and optimize how fast your data is being loaded, improving efficiency.

The `StreamingDataLoader` supports profiling of your data loading process. Simply use the `profile_batches` argument to specify the number of batches you want to profile:

```python
from litdata import StreamingDataset, StreamingDataLoader

StreamingDataLoader(..., profile_batches=5)
```

This generates a Chrome trace called `result.json`. Then, visualize this trace by opening Chrome browser at the `chrome://tracing` URL and load the trace inside.

</details>

<details>
  <summary> ✅ Reduce memory use for large files</summary>
&nbsp;

Handle large data files efficiently without using too much of your computer's memory.

When processing large files like compressed [parquet files](https://en.wikipedia.org/wiki/Apache_Parquet), use the Python yield keyword to process and store one item at the time, reducing the memory footprint of the entire program.

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

</details>

<details>
  <summary> ✅ Limit local cache space</summary>
&nbsp;

Limit the amount of disk space used by temporary files, preventing storage issues.

Adapt the local caching limit of the `StreamingDataset`. This is useful to make sure the downloaded data chunks are deleted when used and the disk usage stays low.

```python
from litdata import StreamingDataset

dataset = StreamingDataset(..., max_cache_size="10GB")
```

</details>

<details>
  <summary> ✅ Change cache directory path</summary>
&nbsp;

Specify the directory where cached files should be stored, ensuring efficient data retrieval and management. This is particularly useful for organizing your data storage and improving access times.

```python
from litdata import StreamingDataset
from litdata.streaming.cache import Dir

cache_dir = "/path/to/your/cache"
data_dir = "s3://my-bucket/my_optimized_dataset"

dataset = StreamingDataset(input_dir=Dir(path=cache_dir, url=data_dir))
```

</details>

<details>
  <summary> ✅ Optimize loading on networked drives</summary>
&nbsp;

Optimize data handling for computers on a local network to improve performance for on-site setups.

On-prem compute nodes can mount and use a network drive. A network drive is a shared storage device on a local area network. In order to reduce their network overload, the `StreamingDataset` supports `caching` the data chunks.

```python
from litdata import StreamingDataset

dataset = StreamingDataset(input_dir="local:/data/shared-drive/some-data")
```

</details>

<details>
  <summary> ✅ Optimize dataset in distributed environment</summary>
&nbsp;

Lightning can distribute large workloads across hundreds of machines in parallel. This can reduce the time to complete a data processing task from weeks to minutes by scaling to enough machines.

To apply the optimize operator across multiple machines, simply provide the num_nodes and machine arguments to it as follows:

```python
import os
from litdata import optimize, Machine

def compress(index):
    return (index, index ** 2)

optimize(
    fn=compress,
    inputs=list(range(100)),
    num_workers=2,
    output_dir="my_output",
    chunk_bytes="64MB",
    num_nodes=2,
    machine=Machine.DATA_PREP, # You can select between dozens of optimized machines
)
```

If the `output_dir` is a local path, the optimized dataset will be present in: `/teamspace/jobs/{job_name}/nodes-0/my_output`. Otherwise, it will be stored in the specified `output_dir`.

Read the optimized dataset:

```python
from litdata import StreamingDataset

output_dir = "/teamspace/jobs/litdata-optimize-2024-07-08/nodes.0/my_output"

dataset = StreamingDataset(output_dir)

print(dataset[:])
```

</details>

<details>
  <summary> ✅ Encrypt, decrypt data at chunk/sample level</summary>
&nbsp;

Secure data by applying encryption to individual samples or chunks, ensuring sensitive information is protected during storage.

This example shows how to use the `FernetEncryption` class for sample-level encryption with a data optimization function.

```python
from litdata import optimize
from litdata.utilities.encryption import FernetEncryption
import numpy as np
from PIL import Image

# Initialize FernetEncryption with a password for sample-level encryption
fernet = FernetEncryption(password="your_secure_password", level="sample")
data_dir = "s3://my-bucket/optimized_data"

def random_image(index):
    """Generate a random image for demonstration purposes."""
    fake_img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    return {"image": fake_img, "class": index}

# Optimize data while applying encryption
optimize(
    fn=random_image,
    inputs=list(range(5)),  # Example inputs: [0, 1, 2, 3, 4]
    num_workers=1,
    output_dir=data_dir,
    chunk_bytes="64MB",
    encryption=fernet,
)

# Save the encryption key to a file for later use
fernet.save("fernet.pem")
```

Load the encrypted data using the `StreamingDataset` class as follows:

```python
from litdata import StreamingDataset
from litdata.utilities.encryption import FernetEncryption

# Load the encryption key
fernet = FernetEncryption(password="your_secure_password", level="sample")
fernet.load("fernet.pem")

# Create a streaming dataset for reading the encrypted samples
ds = StreamingDataset(input_dir=data_dir, encryption=fernet)
```

Implement your own encryption method: Subclass the `Encryption` class and define the necessary methods:

```python
from litdata.utilities.encryption import Encryption

class CustomEncryption(Encryption):
    def encrypt(self, data):
        # Implement your custom encryption logic here
        return data

    def decrypt(self, data):
        # Implement your custom decryption logic here
        return data
```

This allows the data to remain secure while maintaining flexibility in the encryption method.
</details>

&nbsp;

## Features for transforming datasets

<details>
  <summary> ✅ Parallelize data transformations (map)</summary>
&nbsp;

Apply the same change to different parts of the dataset at once to save time and effort.

The `map` operator can be used to apply a function over a list of inputs.

Here is an example where the `map` operator is used to apply a `resize_image` function over a folder of large images.

```python
from litdata import map
from PIL import Image

# Note: Inputs could also refer to files on s3 directly.
input_dir = "my_large_images"
inputs = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

# The resize image takes one of the input (image_path) and the output directory.
# Files written to output_dir are persisted.
def resize_image(image_path, output_dir):
  output_image_path = os.path.join(output_dir, os.path.basename(image_path))
  Image.open(image_path).resize((224, 224)).save(output_image_path)

map(
    fn=resize_image,
    inputs=inputs,
    output_dir="s3://my-bucket/my_resized_images",
)
```

</details>

<details>
  <summary> ✅ Support S3-Compatible cloud object storage</summary>
&nbsp;

Use different cloud storage services, offering data storage flexibility and cost-saving options.

Integrate S3-compatible object storage servers like [MinIO](https://min.io/) with litdata, ideal for on-premises infrastructure setups. Configure the endpoint and credentials using environment variables or configuration files.

Set up the environment variables to connect to MinIO:

```bash
export AWS_ACCESS_KEY_ID=access_key
export AWS_SECRET_ACCESS_KEY=secret_key
export AWS_ENDPOINT_URL=http://localhost:9000  # MinIO endpoint
```

Alternatively, configure credentials and endpoint in `~/.aws/{credentials,config}`:

```bash
mkdir -p ~/.aws && \
cat <<EOL >> ~/.aws/credentials
[default]
aws_access_key_id = access_key
aws_secret_access_key = secret_key
EOL

cat <<EOL >> ~/.aws/config
[default]
endpoint_url = http://localhost:9000  # MinIO endpoint
EOL
```
Explore an example setup of litdata with MinIO in the [LitData with MinIO](https://github.com/bhimrazy/litdata-with-minio) repository for practical implementation details.

</details>

&nbsp;

----

# Benchmarks
In this section we show benchmarks for speed to optimize a dataset and the resulting streaming speed ([Reproduce the benchmark](https://lightning.ai/lightning-ai/studios/benchmark-cloud-data-loading-libraries)).

## Streaming speed

Data optimized and streamed with LitData achieves a 20x speed up over non optimized data and 2x speed up over other streaming solutions.

Speed to stream Imagenet 1.2M from AWS S3:

| Framework | Images / sec  1st Epoch (float32)  | Images / sec   2nd Epoch (float32) | Images / sec 1st Epoch (torch16) | Images / sec 2nd Epoch (torch16) |
|---|---|---|---|---|
| PL Data  | **5800** | **6589**  | **6282**  | **7221**  |
| Web Dataset  | 3134 | 3924 | 3343 | 4424 |
| Mosaic ML  | 2898 | 5099 | 2809 | 5158 |

<details>
  <summary> Benchmark details</summary>
&nbsp;

- [Imagenet-1.2M dataset](https://www.image-net.org/) contains `1,281,167 images`.
- To align with other benchmarks, we measured the streaming speed (`images per second`) loaded from [AWS S3](https://aws.amazon.com/s3/) for several frameworks.

</details>

&nbsp;

## Time to optimize data
LitData optimizes the Imagenet dataset for fast training 3-5x faster than other frameworks:

Time to optimize 1.2 million ImageNet images (Faster is better):
| Framework |Train Conversion Time | Val Conversion Time | Dataset Size | # Files |
|---|---|---|---|---|
| PL Data  |  **10:05 min** | **00:30 min** | **143.1 GB**  | 2.339  |
| Web Dataset  | 32:36 min | 01:22 min | 147.8 GB | 1.144 |
| Mosaic ML  | 49:49 min | 01:04 min | **143.1 GB** | 2.298 |

&nbsp;

----

# Parallelize transforms and data optimization on cloud machines
<div align="center">
<img alt="Lightning" src="https://pl-flash-data.s3.amazonaws.com/data-prep.jpg" width="700px">
</div>

## Parallelize data transforms

Transformations with LitData are linearly parallelizable across machines.

For example, let's say that it takes 56 hours to embed a dataset on a single A10G machine. With LitData,
this can be speed up by adding more machines in parallel

| Number of machines | Hours |
|-----------------|--------------|
| 1               | 56           |
| 2               | 28           |
| 4               | 14           |
| ...               | ...            |
| 64              | 0.875        |

To scale the number of machines, run the processing script on [Lightning Studios](https://lightning.ai/):

```python
from litdata import map, Machine

map(
  ...
  num_nodes=32,
  machine=Machine.DATA_PREP, # Select between dozens of optimized machines
)
```

## Parallelize data optimization
To scale the number of machines for data optimization, use [Lightning Studios](https://lightning.ai/):

```python
from litdata import optimize, Machine

optimize(
  ...
  num_nodes=32,
  machine=Machine.DATA_PREP, # Select between dozens of optimized machines
)
```

&nbsp;

Example: [Process the LAION 400 million image dataset in 2 hours on 32 machines, each with 32 CPUs](https://lightning.ai/lightning-ai/studios/use-or-explore-laion-400million-dataset).

&nbsp;

----

# Start from a template
Below are templates for real-world applications of LitData at scale.

## Templates: Transform datasets

| Studio | Data type | Time (minutes) | Machines | Dataset |
| ------------------------------------ | ----------------- | ----------------- | -------------- | -------------- |
| [Download LAION-400MILLION dataset](https://lightning.ai/lightning-ai/studios/use-or-explore-laion-400million-dataset) | Image & Text | 120 | 32 |[LAION-400M](https://laion.ai/blog/laion-400-open-dataset/) |
| [Tokenize 2M Swedish Wikipedia Articles](https://lightning.ai/lightning-ai/studios/tokenize-2m-swedish-wikipedia-articles) | Text | 7 | 4 | [Swedish Wikipedia](https://huggingface.co/datasets/wikipedia) |
| [Embed English Wikipedia under 5 dollars](https://lightning.ai/lightning-ai/studios/embed-english-wikipedia-under-5-dollars) | Text | 15 | 3 | [English Wikipedia](https://huggingface.co/datasets/wikipedia) |

## Templates: Optimize + stream data

| Studio | Data type | Time (minutes) | Machines | Dataset |
| -------------------------------- | ----------------- | ----------------- | -------------- | -------------- |
| [Benchmark cloud data-loading libraries](https://lightning.ai/lightning-ai/studios/benchmark-cloud-data-loading-libraries) | Image & Label | 10 | 1 | [Imagenet 1M](https://paperswithcode.com/sota/image-classification-on-imagenet?tag_filter=171) |
| [Optimize GeoSpatial data for model training](https://lightning.ai/lightning-ai/studios/convert-spatial-data-to-lightning-streaming) | Image & Mask | 120 | 32 | [Chesapeake Roads Spatial Context](https://github.com/isaaccorley/chesapeakersc) |
| [Optimize TinyLlama 1T dataset for training](https://lightning.ai/lightning-ai/studios/prepare-the-tinyllama-1t-token-dataset) | Text | 240 | 32 | [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) & [StarCoder](https://huggingface.co/datasets/bigcode/starcoderdata) |
| [Optimize parquet files for model training](https://lightning.ai/lightning-ai/studios/convert-parquets-to-lightning-streaming) | Parquet Files | 12 | 16 | Randomly Generated data |

&nbsp;

----

# Community
LitData is a community project accepting contributions -  Let's make the world's most advanced AI data processing framework.

💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)    
📋 [License: Apache 2.0](https://github.com/Lightning-AI/litdata/blob/main/LICENSE)
