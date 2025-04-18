.. container::

      

   **Transform datasets at scale.
   Optimize data for fast AI model training.**

   .. raw:: html

      <pre>
      Transform                              Optimize
        
      ✅ Parallelize data processing       ✅ Stream large cloud datasets          
      ✅ Create vector embeddings          ✅ Accelerate training by 20x           
      ✅ Run distributed inference         ✅ Pause and resume data streaming      
      ✅ Scrape websites at scale          ✅ Use remote data without local loading
      </pre>

   --------------

   |PyPI| |Downloads| |License|

   .. raw:: html

      <p align="center">

   Lightning AI • Quick start • Optimize data • Transform data •
   Features • Benchmarks • Templates • Community

   .. raw:: html

      </p>

    

 

Transform data at scale. Optimize for fast model training.
==========================================================

LitData scales `data processing tasks <#transform-datasets>`__ (data
scraping, image resizing, distributed inference, embedding creation) on
local or cloud machines. It also enables `optimizing
datasets <#speed-up-model-training>`__ to accelerate AI model training
and work with large remote datasets without local loading.

 

Quick start
===========

First, install LitData:

.. code:: bash

   pip install litdata

Choose your workflow:

| 🚀 `Speed up model training <#speed-up-model-training>`__
| 🚀 `Transform datasets <#transform-datasets>`__

 

.. raw:: html

   <details>

.. raw:: html

   <summary>

Advanced install

.. raw:: html

   </summary>

Install all the extras

.. code:: bash

   pip install 'litdata[extras]'

.. raw:: html

   </details>

 

--------------

Speed up model training
=======================

Accelerate model training (20x faster) by optimizing datasets for
streaming directly from cloud storage. Work with remote data without
local downloads with features like loading data subsets, accessing
individual samples, and resumable streaming.

**Step 1: Optimize the data** This step will format the dataset for fast
loading. The data will be written in a chunked binary format.

.. code:: python

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

**Step 2: Put the data on the cloud**

Upload the data to a `Lightning Studio <https://lightning.ai>`__ (backed
by S3) or your own S3 bucket:

.. code:: bash

   aws s3 cp --recursive fast_data s3://my-bucket/fast_data

**Step 3: Stream the data during training**

Load the data by replacing the PyTorch DataSet and DataLoader with the
StreamingDataset and StreamingDataloader

.. code:: python

   import litdata as ld

   train_dataset = ld.StreamingDataset('s3://my-bucket/fast_data', shuffle=True, drop_last=True)
   train_dataloader = ld.StreamingDataLoader(train_dataset)

   for sample in train_dataloader:
       img, cls = sample['image'], sample['class']

**Key benefits:**

| ✅ Accelerate training: Optimized datasets load 20x faster.
| ✅ Stream cloud datasets: Work with cloud data without downloading it.
| ✅ Pytorch-first: Works with PyTorch libraries like PyTorch Lightning,
  Lightning Fabric, Hugging Face.
| ✅ Easy collaboration: Share and access datasets in the cloud,
  streamlining team projects.
| ✅ Scale across GPUs: Streamed data automatically scales to all GPUs.
| ✅ Flexible storage: Use S3, GCS, Azure, or your own cloud account for
  data storage.
| ✅ Compression: Reduce your data footprint by using advanced
  compression algorithms.
| ✅ Run local or cloud: Run on your own machines or auto-scale to 1000s
  of cloud GPUs with Lightning Studios.
| ✅ Enterprise security: Self host or process data on your cloud
  account with Lightning Studios.

 

--------------

Transform datasets
==================

Accelerate data processing tasks (data scraping, image resizing,
embedding creation, distributed inference) by parallelizing (map) the
work across many machines at once.

Here’s an example that resizes and crops a large image dataset:

.. code:: python

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

**Key benefits:**

| ✅ Parallelize processing: Reduce processing time by transforming data
  across multiple machines simultaneously.
| ✅ Scale to large data: Increase the size of datasets you can
  efficiently handle.
| ✅ Flexible usecases: Resize images, create embeddings, scrape the
  internet, etc…
| ✅ Run local or cloud: Run on your own machines or auto-scale to 1000s
  of cloud GPUs with Lightning Studios.
| ✅ Enterprise security: Self host or process data on your cloud
  account with Lightning Studios.

 

--------------

Key Features
============

Features for optimizing and streaming datasets for model training
-----------------------------------------------------------------

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Stream large cloud datasets

.. raw:: html

   </summary>

 

Use data stored on the cloud without needing to download it all to your
computer, saving time and space.

Imagine you’re working on a project with a huge amount of data stored
online. Instead of waiting hours to download it all, you can start
working with the data almost immediately by streaming it.

Once you’ve optimized the dataset with LitData, stream it as follows:

.. code:: python

   from litdata import StreamingDataset, StreamingDataLoader

   dataset = StreamingDataset('s3://my-bucket/my-data', shuffle=True)
   dataloader = StreamingDataLoader(dataset, batch_size=64)

   for batch in dataloader:
       process(batch)  # Replace with your data processing logic

Additionally, you can inject client connection settings for
`S3 <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/core/session.html#boto3.session.Session.client>`__
or GCP when initializing your dataset. This is useful for specifying
custom endpoints and credentials per dataset.

.. code:: python

   from litdata import StreamingDataset

   # boto3 compatible storage options for a custom S3-compatible endpoint
   storage_options = {
       "endpoint_url": "your_endpoint_url",
       "aws_access_key_id": "your_access_key_id",
       "aws_secret_access_key": "your_secret_access_key",
   }

   dataset = StreamingDataset('s3://my-bucket/my-data', storage_options=storage_options)

   # s5cmd compatible storage options for a custom S3-compatible endpoint
   # Note: If s5cmd is installed, it will be used by default for S3 operations. If you prefer not to use s5cmd, you can disable it by setting the environment variable: `DISABLE_S5CMD=1`
   storage_options = {
       "AWS_ACCESS_KEY_ID": "your_access_key_id",
       "AWS_SECRET_ACCESS_KEY": "your_secret_access_key",
       "S3_ENDPOINT_URL": "your_endpoint_url",  # Required only for custom endpoints
   }


   dataset = StreamingDataset('s3://my-bucket/my-data', storage_options=storage_options)

Alternative: Using ``s5cmd`` for S3 Operations

Also, you can specify a custom cache directory when initializing your
dataset. This is useful when you want to store the cache in a specific
location.

.. code:: python

   from litdata import StreamingDataset

   # Initialize the StreamingDataset with the custom cache directory
   dataset = StreamingDataset('s3://my-bucket/my-data', cache_dir="/path/to/cache")

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Stream Hugging Face 🤗 datasets

.. raw:: html

   </summary>

 

To use your favorite Hugging Face dataset with LitData, simply pass its
URL to ``StreamingDataset``.

.. raw:: html

   <details>

.. raw:: html

   <summary>

How to get HF dataset URI?

.. raw:: html

   </summary>

https://github.com/user-attachments/assets/3ba9e2ef-bf6b-41fc-a578-e4b4113a0e72

.. raw:: html

   </details>

**Prerequisites:**

Install the required dependencies to stream Hugging Face datasets:

.. code:: sh

   pip install "litdata[extra]" huggingface_hub

   # Optional: To speed up downloads on high-bandwidth networks
   pip install hf_transfer
   export HF_HUB_ENABLE_HF_TRANSFER=1

**Stream Hugging Face dataset:**

.. code:: python

   import litdata as ld

   # Define the Hugging Face dataset URI
   hf_dataset_uri = "hf://datasets/leonardPKU/clevr_cogen_a_train/data"

   # Create a streaming dataset
   dataset = ld.StreamingDataset(hf_dataset_uri)

   # Print the first sample
   print("Sample", dataset[0])

   # Stream the dataset using StreamingDataLoader
   dataloader = ld.StreamingDataLoader(dataset, batch_size=4)
   for sample in dataloader:
       pass 

You don’t need to worry about indexing the dataset or any other setup.
**LitData** will **handle all the necessary steps automatically** and
``cache`` the ``index.json`` file, so you won’t have to index it again.

This ensures that the next time you stream the dataset, the indexing
step is skipped..

 

Indexing the HF dataset (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the Hugging Face dataset hasn’t been indexed yet, you can index it
first using the ``index_hf_dataset`` method, and then stream it using
the code above.

.. code:: python

   import litdata as ld

   hf_dataset_uri = "hf://datasets/leonardPKU/clevr_cogen_a_train/data"

   ld.index_hf_dataset(hf_dataset_uri)

- Indexing the Hugging Face dataset ahead of time will make streaming
  abit faster, as it avoids the need for real-time indexing during
  streaming.

- To use ``HF gated dataset``, ensure the ``HF_TOKEN`` environment
  variable is set.

**Note**: For HuggingFace datasets, ``indexing`` & ``streaming`` is
supported only for datasets in **``Parquet format``**.

 

Full Workflow for Hugging Face Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For full control over the cache
path(``where index.json file will be stored``) and other configurations,
follow these steps:

1. Index the Hugging Face dataset first:

.. code:: python

   import litdata as ld

   hf_dataset_uri = "hf://datasets/open-thoughts/OpenThoughts-114k/data"

   ld.index_parquet_dataset(hf_dataset_uri, "hf-index-dir")

2. To stream HF datasets now, pass the ``HF dataset URI``, the path
   where the ``index.json`` file is stored, and ``ParquetLoader`` as the
   ``item_loader`` to the **``StreamingDataset``**:

.. code:: python

   import litdata as ld
   from litdata.streaming.item_loader import ParquetLoader

   hf_dataset_uri = "hf://datasets/open-thoughts/OpenThoughts-114k/data"

   dataset = ld.StreamingDataset(hf_dataset_uri, item_loader=ParquetLoader(), index_path="hf-index-dir")

   for batch in ld.StreamingDataLoader(dataset, batch_size=4):
     pass

 

LitData ``Optimize`` v/s ``Parquet``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <!-- TODO: Update benchmark -->

Below is the benchmark for the ``Imagenet dataset (155 GB)``,
demonstrating that
**``optimizing the dataset using LitData is faster and results in smaller output size compared to raw Parquet files``**.

+------------------------+---------+--------------+-------------------+
| **Operation**          | **Size  | **Time       | **Throughput      |
|                        | (GB)**  | (seconds)**  | (images/sec)**    |
+========================+=========+==============+===================+
| LitData Optimize       | 45      | 283.17       | 4000-4700         |
| Dataset                |         |              |                   |
+------------------------+---------+--------------+-------------------+
| Parquet Optimize       | 51      | 465.96       | 3600-3900         |
| Dataset                |         |              |                   |
+------------------------+---------+--------------+-------------------+
| Index Parquet Dataset  | N/A     | 6            | N/A               |
| (overhead)             |         |              |                   |
+------------------------+---------+--------------+-------------------+

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Streams on multi-GPU, multi-node

.. raw:: html

   </summary>

 

Data optimized and loaded with Lightning automatically streams
efficiently in distributed training across GPUs or multi-node.

The ``StreamingDataset`` and ``StreamingDataLoader`` automatically make
sure each rank receives the same quantity of varied batches of data, so
it works out of the box with your favorite frameworks (`PyTorch
Lightning <https://lightning.ai/docs/pytorch/stable/>`__, `Lightning
Fabric <https://lightning.ai/docs/fabric/stable/>`__, or
`PyTorch <https://pytorch.org/docs/stable/index.html>`__) to do
distributed training.

Here you can see an illustration showing how the Streaming Dataset works
with multi node / multi gpu under the hood.

.. code:: python

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

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Stream from multiple cloud providers

.. raw:: html

   </summary>

 

The ``StreamingDataset`` provides support for reading optimized datasets
from common cloud storage providers like AWS S3, Google Cloud Storage
(GCS), and Azure Blob Storage. Below are examples of how to use
StreamingDataset with each cloud provider.

.. code:: python

   import os
   import litdata as ld

   # Read data from AWS S3 using s5cmd
   # Note: If s5cmd is installed, it will be used by default for S3 operations. If you prefer not to use s5cmd, you can disable it by setting the environment variable: `DISABLE_S5CMD=1`
   aws_storage_options={
       "AWS_ACCESS_KEY_ID": os.environ['AWS_ACCESS_KEY_ID'],
       "AWS_SECRET_ACCESS_KEY": os.environ['AWS_SECRET_ACCESS_KEY'],
       "S3_ENDPOINT_URL": os.environ['AWS_ENDPOINT_URL'],  # Required only for custom endpoints
   }
   dataset = ld.StreamingDataset("s3://my-bucket/my-data", storage_options=aws_storage_options)

   # Read Data from AWS S3 with Unsigned Request using s5cmd
   aws_storage_options={
     "AWS_NO_SIGN_REQUEST": "Yes" # Required for unsigned requests
     "S3_ENDPOINT_URL": os.environ['AWS_ENDPOINT_URL'],  # Required only for custom endpoints
   }
   dataset = ld.StreamingDataset("s3://my-bucket/my-data", storage_options=aws_storage_options)

   # Read data from AWS S3 using boto3
   os.environ["DISABLE_S5CMD"] = "1"
   aws_storage_options={
       "aws_access_key_id": os.environ['AWS_ACCESS_KEY_ID'],
       "aws_secret_access_key": os.environ['AWS_SECRET_ACCESS_KEY'],
   }
   dataset = ld.StreamingDataset("s3://my-bucket/my-data", storage_options=aws_storage_options)


   # Read data from GCS
   gcp_storage_options={
       "project": os.environ['PROJECT_ID'],
   }
   dataset = ld.StreamingDataset("gs://my-bucket/my-data", storage_options=gcp_storage_options)

   # Read data from Azure
   azure_storage_options={
       "account_url": f"https://{os.environ['AZURE_ACCOUNT_NAME']}.blob.core.windows.net",
       "credential": os.environ['AZURE_ACCOUNT_ACCESS_KEY']
   }
   dataset = ld.StreamingDataset("azure://my-bucket/my-data", storage_options=azure_storage_options)

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Pause, resume data streaming

.. raw:: html

   </summary>

 

Stream data during long training, if interrupted, pick up right where
you left off without any issues.

LitData provides a stateful ``Streaming DataLoader`` e.g. you can
``pause`` and ``resume`` your training whenever you want.

Info: The ``Streaming DataLoader`` was used by
`Lit-GPT <https://github.com/Lightning-AI/lit-gpt/blob/main/pretrain/tinyllama.py>`__
to pretrain LLMs. Restarting from an older checkpoint was critical to
get to pretrain the full model due to several failures (network, CUDA
Errors, etc..).

.. code:: python

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

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ LLM Pre-training

.. raw:: html

   </summary>

 

LitData is highly optimized for LLM pre-training. First, we need to
tokenize the entire dataset and then we can consume it.

.. code:: python

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

.. code:: python

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

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Filter illegal data

.. raw:: html

   </summary>

 

Sometimes, you have bad data that you don’t want to include in the
optimized dataset. With LitData, yield only the good data sample to
include.

.. code:: python

   from litdata import optimize, StreamingDataset

   def should_keep(index) -> bool:
     # Replace with your own logic
     return index % 2 == 0


   def fn(data):
       if should_keep(data):
           yield data

   if __name__ == "__main__":
       optimize(
           fn=fn,
           inputs=list(range(1000)),
           output_dir="only_even_index_optimized",
           chunk_bytes="64MB",
           num_workers=1
       )

       dataset = StreamingDataset("only_even_index_optimized")
       data = list(dataset)
       print(data)
       # [0, 2, 4, 6, 8, 10, ..., 992, 994, 996, 998]

You can even use try/expect.

.. code:: python

   from litdata import optimize, StreamingDataset

   def fn(data):
       try:
           yield 1 / data 
       except:
           pass

   if __name__ == "__main__":
       optimize(
           fn=fn,
           inputs=[0, 0, 0, 1, 2, 4, 0],
           output_dir="only_defined_ratio_optimized",
           chunk_bytes="64MB",
           num_workers=1
       )

       dataset = StreamingDataset("only_defined_ratio_optimized")
       data = list(dataset)
       # The 0 are filtered out as they raise a division by zero 
       print(data)
       # [1.0, 0.5, 0.25] 

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Combine datasets

.. raw:: html

   </summary>

 

Mix and match different sets of data to experiment and create better
models.

Combine datasets with ``CombinedStreamingDataset``. As an example, this
mixture of
`Slimpajama <https://huggingface.co/datasets/cerebras/SlimPajama-627B>`__
& `StarCoder <https://huggingface.co/datasets/bigcode/starcoderdata>`__
was used in the `TinyLLAMA <https://github.com/jzhang38/TinyLlama>`__
project to pretrain a 1.1B Llama model on 3 trillion tokens.

.. code:: python

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

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Merge datasets

.. raw:: html

   </summary>

 

Merge multiple optimized datasets into one.

.. code:: python

   import numpy as np
   from PIL import Image

   from litdata import StreamingDataset, merge_datasets, optimize


   def random_images(index):
       return {
           "index": index,
           "image": Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)),
           "class": np.random.randint(10),
       }


   if __name__ == "__main__":
       out_dirs = ["fast_data_1", "fast_data_2", "fast_data_3", "fast_data_4"]  # or ["s3://my-bucket/fast_data_1", etc.]"
       for out_dir in out_dirs:
           optimize(fn=random_images, inputs=list(range(250)), output_dir=out_dir, num_workers=4, chunk_bytes="64MB")

       merged_out_dir = "merged_fast_data" # or "s3://my-bucket/merged_fast_data"
       merge_datasets(input_dirs=out_dirs, output_dir=merged_out_dir)

       dataset = StreamingDataset(merged_out_dir)
       print(len(dataset))
       # out: 1000

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Split datasets for train, val, test

.. raw:: html

   </summary>

 

Split a dataset into train, val, test splits with ``train_test_split``.

.. code:: python

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

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Load a subset of the remote dataset

.. raw:: html

   </summary>

  Work on a smaller, manageable portion of your data to save time and
resources.

.. code:: python

   from litdata import StreamingDataset, train_test_split

   dataset = StreamingDataset("s3://my-bucket/my-data", subsample=0.01) # data are stored in the cloud

   print(len(dataset)) # display the length of your data
   # out: 1000

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Upsample from your source datasets

.. raw:: html

   </summary>

  Use to control the size of one iteration of a StreamingDataset using
repeats. Contains ``floor(N)`` possibly shuffled copies of the source
data, then a subsampling of the remainder.

.. code:: python

   from litdata import StreamingDataset

   dataset = StreamingDataset("s3://my-bucket/my-data", subsample=2.5, shuffle=True)

   print(len(dataset)) # display the length of your data
   # out: 250000

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Easily modify optimized cloud datasets

.. raw:: html

   </summary>

 

Add new data to an existing dataset or start fresh if needed, providing
flexibility in data management.

LitData optimized datasets are assumed to be immutable. However, you can
make the decision to modify them by changing the mode to either
``append`` or ``overwrite``.

.. code:: python

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

The ``overwrite`` mode will delete the existing data and start from
fresh.

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Stream parquet datasets

.. raw:: html

   </summary>

 

Stream Parquet datasets directly with LitData—no need to convert them
into LitData’s optimized binary format! If your dataset is already in
Parquet format, you can efficiently index and stream it using
``StreamingDataset`` and ``StreamingDataLoader``.

**Assumption:**

Your dataset directory contains one or more Parquet files.

**Prerequisites:**

Install the required dependencies to stream Parquet datasets from cloud
storage like **Amazon S3** or **Google Cloud Storage**:

.. code:: bash

   # For Amazon S3
   pip install "litdata[extra]" s3fs

   # For Google Cloud Storage
   pip install "litdata[extra]" gcsfs

**Index Your Dataset**:

Index your Parquet dataset to create an index file that LitData can use
to stream the dataset.

.. code:: python

   import litdata as ld

   # Point to your data stored in the cloud
   pq_dataset_uri = "s3://my-bucket/my-parquet-data"  # or "gs://my-bucket/my-parquet-data"

   ld.index_parquet_dataset(pq_dataset_uri)

**Stream the Dataset**

Use ``StreamingDataset`` with ``ParquetLoader`` to load and stream the
dataset efficiently:

.. code:: python

   import litdata as ld
   from litdata.streaming.item_loader import ParquetLoader

   # Specify your dataset location in the cloud
   pq_dataset_uri = "s3://my-bucket/my-parquet-data"  # or "gs://my-bucket/my-parquet-data"

   # Set up the streaming dataset
   dataset = ld.StreamingDataset(pq_dataset_uri, item_loader=ParquetLoader())

   # print the first sample
   print("Sample", dataset[0])

   # Stream the dataset using StreamingDataLoader
   dataloader = ld.StreamingDataLoader(dataset, batch_size=4)
   for sample in dataloader:
       pass

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Use compression

.. raw:: html

   </summary>

 

Reduce your data footprint by using advanced compression algorithms.

.. code:: python

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

Using `zstd <https://github.com/facebook/zstd>`__, you can achieve high
compression ratio like 4.34x for this simple example.

======= ====
Without With
======= ====
2.8kb   646b
======= ====

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Access samples without full data download

.. raw:: html

   </summary>

 

Look at specific parts of a large dataset without downloading the whole
thing or loading it on a local machine.

.. code:: python

   from litdata import StreamingDataset

   dataset = StreamingDataset("s3://my-bucket/my-data") # data are stored in the cloud

   print(len(dataset)) # display the length of your data

   print(dataset[42]) # show the 42th element of the dataset

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Use any data transforms

.. raw:: html

   </summary>

 

Customize how your data is processed to better fit your needs.

Subclass the ``StreamingDataset`` and override its ``__getitem__``
method to add any extra data transformations.

.. code:: python

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

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Profile data loading speed

.. raw:: html

   </summary>

 

Measure and optimize how fast your data is being loaded, improving
efficiency.

The ``StreamingDataLoader`` supports profiling of your data loading
process. Simply use the ``profile_batches`` argument to specify the
number of batches you want to profile:

.. code:: python

   from litdata import StreamingDataset, StreamingDataLoader

   StreamingDataLoader(..., profile_batches=5)

This generates a Chrome trace called ``result.json``. Then, visualize
this trace by opening Chrome browser at the ``chrome://tracing`` URL and
load the trace inside.

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Reduce memory use for large files

.. raw:: html

   </summary>

 

Handle large data files efficiently without using too much of your
computer’s memory.

When processing large files like compressed `parquet
files <https://en.wikipedia.org/wiki/Apache_Parquet>`__, use the Python
yield keyword to process and store one item at the time, reducing the
memory footprint of the entire program.

.. code:: python

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

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Limit local cache space

.. raw:: html

   </summary>

 

Limit the amount of disk space used by temporary files, preventing
storage issues.

Adapt the local caching limit of the ``StreamingDataset``. This is
useful to make sure the downloaded data chunks are deleted when used and
the disk usage stays low.

.. code:: python

   from litdata import StreamingDataset

   dataset = StreamingDataset(..., max_cache_size="10GB")

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Change cache directory path

.. raw:: html

   </summary>

 

Specify the directory where cached files should be stored, ensuring
efficient data retrieval and management. This is particularly useful for
organizing your data storage and improving access times.

.. code:: python

   from litdata import StreamingDataset
   from litdata.streaming.cache import Dir

   cache_dir = "/path/to/your/cache"
   data_dir = "s3://my-bucket/my_optimized_dataset"

   dataset = StreamingDataset(input_dir=Dir(path=cache_dir, url=data_dir))

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Optimize loading on networked drives

.. raw:: html

   </summary>

 

Optimize data handling for computers on a local network to improve
performance for on-site setups.

On-prem compute nodes can mount and use a network drive. A network drive
is a shared storage device on a local area network. In order to reduce
their network overload, the ``StreamingDataset`` supports ``caching``
the data chunks.

.. code:: python

   from litdata import StreamingDataset

   dataset = StreamingDataset(input_dir="local:/data/shared-drive/some-data")

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Optimize dataset in distributed environment

.. raw:: html

   </summary>

 

Lightning can distribute large workloads across hundreds of machines in
parallel. This can reduce the time to complete a data processing task
from weeks to minutes by scaling to enough machines.

To apply the optimize operator across multiple machines, simply provide
the num_nodes and machine arguments to it as follows:

.. code:: python

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

If the ``output_dir`` is a local path, the optimized dataset will be
present in: ``/teamspace/jobs/{job_name}/nodes-0/my_output``. Otherwise,
it will be stored in the specified ``output_dir``.

Read the optimized dataset:

.. code:: python

   from litdata import StreamingDataset

   output_dir = "/teamspace/jobs/litdata-optimize-2024-07-08/nodes.0/my_output"

   dataset = StreamingDataset(output_dir)

   print(dataset[:])

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Encrypt, decrypt data at chunk/sample level

.. raw:: html

   </summary>

 

Secure data by applying encryption to individual samples or chunks,
ensuring sensitive information is protected during storage.

This example shows how to use the ``FernetEncryption`` class for
sample-level encryption with a data optimization function.

.. code:: python

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

Load the encrypted data using the ``StreamingDataset`` class as follows:

.. code:: python

   from litdata import StreamingDataset
   from litdata.utilities.encryption import FernetEncryption

   # Load the encryption key
   fernet = FernetEncryption(password="your_secure_password", level="sample")
   fernet.load("fernet.pem")

   # Create a streaming dataset for reading the encrypted samples
   ds = StreamingDataset(input_dir=data_dir, encryption=fernet)

Implement your own encryption method: Subclass the ``Encryption`` class
and define the necessary methods:

.. code:: python

   from litdata.utilities.encryption import Encryption

   class CustomEncryption(Encryption):
       def encrypt(self, data):
           # Implement your custom encryption logic here
           return data

       def decrypt(self, data):
           # Implement your custom decryption logic here
           return data

This allows the data to remain secure while maintaining flexibility in
the encryption method.

.. raw:: html

   </details>

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Debug & Profile LitData with logs & Litracer

.. raw:: html

   </summary>

 

LitData comes with built-in logging and profiling capabilities to help
you debug and profile your data streaming workloads.

- e.g., with LitData Streaming

.. code:: python

   import litdata as ld
   from litdata.debugger import enable_tracer

   # WARNING: Remove existing trace `litdata_debug.log` file if it exists before re-tracing
   enable_tracer()

   if __name__ == "__main__":
       dataset = ld.StreamingDataset("s3://my-bucket/my-data", shuffle=True)
       dataloader = ld.StreamingDataLoader(dataset, batch_size=64)

       for batch in dataloader:
           print(batch)  # Replace with your data processing logic

1. Generate Debug Log:

   - Run your Python program and it’ll create a log file containing
     detailed debug information.

   .. code:: bash

        python main.py

2. Install `Litracer <https://github.com/deependujha/litracer/>`__:

   - Option 1: Using Go (recommended)

     - Install Go on your system.
     - Run the following command to install Litracer:

     .. code:: bash

          go install github.com/deependujha/litracer@latest

   - Option 2: Download Binary

     - Visit the `LitRacer GitHub
       Releases <https://github.com/deependujha/litracer/releases>`__
       page.
     - Download the appropriate binary for your operating system and
       follow the installation instructions.

3. Convert Debug Log to trace JSON:

   - Use litracer to convert the generated log file into a trace JSON
     file. This command uses 100 workers for conversion:

   .. code:: bash

        litracer litdata_debug.log -o litdata_trace.json -w 100

4. Visualize the trace:

   - Use either ``chrome://tracing`` in the Chrome browser or
     ``ui.perfetto.dev`` to view the ``litdata_trace.json`` file for
     in-depth performance insights. You can also use ``SQL queries`` to
     analyze the logs.
   - ``Perfetto`` is recommended over ``chrome://tracing`` for
     visualization & analyzing.

- Key Points:

  - For very large trace.json files (``> 2GB``), refer to the `Perfetto
    documentation <https://perfetto.dev/docs/visualization/large-traces>`__
    for using native accelerators.
  - If you are trying to connect Perfetto to the RPC server, it is
    recommended to use Chrome over Brave, as it has been observed that
    Perfetto in Brave does not autodetect the RPC server.

.. raw:: html

   </details>

 

Features for transforming datasets
----------------------------------

.. raw:: html

   <details>

.. raw:: html

   <summary>

✅ Parallelize data transformations (map)

.. raw:: html

   </summary>

 

Apply the same change to different parts of the dataset at once to save
time and effort.

The ``map`` operator can be used to apply a function over a list of
inputs.

Here is an example where the ``map`` operator is used to apply a
``resize_image`` function over a folder of large images.

.. code:: python

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

.. raw:: html

   </details>

 

--------------

Benchmarks
==========

In this section we show benchmarks for speed to optimize a dataset and
the resulting streaming speed (`Reproduce the
benchmark <https://lightning.ai/lightning-ai/studios/benchmark-cloud-data-loading-libraries>`__).

Streaming speed
---------------

Data optimized and streamed with LitData achieves a 20x speed up over
non optimized data and 2x speed up over other streaming solutions.

Speed to stream Imagenet 1.2M from AWS S3:

+-------------+-------------+-------------+-------------+-------------+
| Framework   | Images /    | Images /    | Images /    | Images /    |
|             | sec 1st     | sec 2nd     | sec 1st     | sec 2nd     |
|             | Epoch       | Epoch       | Epoch       | Epoch       |
|             | (float32)   | (float32)   | (torch16)   | (torch16)   |
+=============+=============+=============+=============+=============+
| LitData     | **5839**    | **6692**    | **6282**    | **7221**    |
+-------------+-------------+-------------+-------------+-------------+
| Web Dataset | 3134        | 3924        | 3343        | 4424        |
+-------------+-------------+-------------+-------------+-------------+
| Mosaic ML   | 2898        | 5099        | 2809        | 5158        |
+-------------+-------------+-------------+-------------+-------------+

.. raw:: html

   <details>

.. raw:: html

   <summary>

Benchmark details

.. raw:: html

   </summary>

 

- `Imagenet-1.2M dataset <https://www.image-net.org/>`__ contains
  ``1,281,167 images``.
- To align with other benchmarks, we measured the streaming speed
  (``images per second``) loaded from `AWS
  S3 <https://aws.amazon.com/s3/>`__ for several frameworks.

.. raw:: html

   </details>

 

Speed to stream Imagenet 1.2M from other cloud storage providers:

+-----------------+-----------------+-----------------+-----------------+
| Storage         | Framework       | Images / sec    | Images / sec    |
| Provider        |                 | 1st Epoch       | 2nd Epoch       |
|                 |                 | (float32)       | (float32)       |
+=================+=================+=================+=================+
| Cloudflare R2   | LitData         | **5335**        | **5630**        |
+-----------------+-----------------+-----------------+-----------------+

 

Time to optimize data
---------------------

LitData optimizes the Imagenet dataset for fast training 3-5x faster
than other frameworks:

Time to optimize 1.2 million ImageNet images (Faster is better): \|
Framework \|Train Conversion Time \| Val Conversion Time \| Dataset Size
\| # Files \| \|—\|—\|—\|—\|—\| \| LitData \| **10:05 min** \| **00:30
min** \| **143.1 GB** \| 2.339 \| \| Web Dataset \| 32:36 min \| 01:22
min \| 147.8 GB \| 1.144 \| \| Mosaic ML \| 49:49 min \| 01:04 min \|
**143.1 GB** \| 2.298 \|

 

--------------

Parallelize transforms and data optimization on cloud machines
==============================================================

.. container::

Parallelize data transforms
---------------------------

Transformations with LitData are linearly parallelizable across
machines.

For example, let’s say that it takes 56 hours to embed a dataset on a
single A10G machine. With LitData, this can be speed up by adding more
machines in parallel

================== =====
Number of machines Hours
================== =====
1                  56
2                  28
4                  14
…                  …
64                 0.875
================== =====

To scale the number of machines, run the processing script on `Lightning
Studios <https://lightning.ai/>`__:

.. code:: python

   from litdata import map, Machine

   map(
     ...
     num_nodes=32,
     machine=Machine.DATA_PREP, # Select between dozens of optimized machines
   )

Parallelize data optimization
-----------------------------

To scale the number of machines for data optimization, use `Lightning
Studios <https://lightning.ai/>`__:

.. code:: python

   from litdata import optimize, Machine

   optimize(
     ...
     num_nodes=32,
     machine=Machine.DATA_PREP, # Select between dozens of optimized machines
   )

 

Example: `Process the LAION 400 million image dataset in 2 hours on 32
machines, each with 32
CPUs <https://lightning.ai/lightning-ai/studios/use-or-explore-laion-400million-dataset>`__.

 

--------------

Start from a template
=====================

Below are templates for real-world applications of LitData at scale.

Templates: Transform datasets
-----------------------------

+-------------------------+-----------+-----------+---------+---------+
| Studio                  | Data type | Time      | M       | Dataset |
|                         |           | (minutes) | achines |         |
+=========================+===========+===========+=========+=========+
| `Download               | Image &   | 120       | 32      | `LAION  |
| LAION-400MILLION        | Text      |           |         | -400M < |
| da                      |           |           |         | https:/ |
| taset <https://lightnin |           |           |         | /laion. |
| g.ai/lightning-ai/studi |           |           |         | ai/blog |
| os/use-or-explore-laion |           |           |         | /laion- |
| -400million-dataset>`__ |           |           |         | 400-ope |
|                         |           |           |         | n-datas |
|                         |           |           |         | et/>`__ |
+-------------------------+-----------+-----------+---------+---------+
| `Tokenize 2M Swedish    | Text      | 7         | 4       | `       |
| Wikipedia               |           |           |         | Swedish |
| Ar                      |           |           |         | Wikiped |
| ticles <https://lightni |           |           |         | ia <htt |
| ng.ai/lightning-ai/stud |           |           |         | ps://hu |
| ios/tokenize-2m-swedish |           |           |         | ggingfa |
| -wikipedia-articles>`__ |           |           |         | ce.co/d |
|                         |           |           |         | atasets |
|                         |           |           |         | /wikipe |
|                         |           |           |         | dia>`__ |
+-------------------------+-----------+-----------+---------+---------+
| `Embed English          | Text      | 15        | 3       | `       |
| Wikipedia under 5       |           |           |         | English |
| do                      |           |           |         | Wikiped |
| llars <https://lightnin |           |           |         | ia <htt |
| g.ai/lightning-ai/studi |           |           |         | ps://hu |
| os/embed-english-wikipe |           |           |         | ggingfa |
| dia-under-5-dollars>`__ |           |           |         | ce.co/d |
|                         |           |           |         | atasets |
|                         |           |           |         | /wikipe |
|                         |           |           |         | dia>`__ |
+-------------------------+-----------+-----------+---------+---------+

Templates: Optimize + stream data
---------------------------------

+-----------------------+------------+------------+---------+---------+
| Studio                | Data type  | Time       | M       | Dataset |
|                       |            | (minutes)  | achines |         |
+=======================+============+============+=========+=========+
| `Benchmark cloud      | Image &    | 10         | 1       | `I      |
| data-loading          | Label      |            |         | magenet |
| libraries <           |            |            |         | 1M      |
| https://lightning.ai/ |            |            |         | <https: |
| lightning-ai/studios/ |            |            |         | //paper |
| benchmark-cloud-data- |            |            |         | swithco |
| loading-libraries>`__ |            |            |         | de.com/ |
|                       |            |            |         | sota/im |
|                       |            |            |         | age-cla |
|                       |            |            |         | ssifica |
|                       |            |            |         | tion-on |
|                       |            |            |         | -imagen |
|                       |            |            |         | et?tag_ |
|                       |            |            |         | filter= |
|                       |            |            |         | 171>`__ |
+-----------------------+------------+------------+---------+---------+
| `Optimize GeoSpatial  | Image &    | 120        | 32      | `Che    |
| data for model        | Mask       |            |         | sapeake |
| training <https       |            |            |         | Roads   |
| ://lightning.ai/light |            |            |         | Spatial |
| ning-ai/studios/conve |            |            |         | C       |
| rt-spatial-data-to-li |            |            |         | ontext  |
| ghtning-streaming>`__ |            |            |         | <https: |
|                       |            |            |         | //githu |
|                       |            |            |         | b.com/i |
|                       |            |            |         | saaccor |
|                       |            |            |         | ley/che |
|                       |            |            |         | sapeake |
|                       |            |            |         | rsc>`__ |
+-----------------------+------------+------------+---------+---------+
| `Optimize TinyLlama   | Text       | 240        | 32      | `Sl     |
| 1T dataset for        |            |            |         | imPajam |
| training <            |            |            |         | a <http |
| https://lightning.ai/ |            |            |         | s://hug |
| lightning-ai/studios/ |            |            |         | gingfac |
| prepare-the-tinyllama |            |            |         | e.co/da |
| -1t-token-dataset>`__ |            |            |         | tasets/ |
|                       |            |            |         | cerebra |
|                       |            |            |         | s/SlimP |
|                       |            |            |         | ajama-6 |
|                       |            |            |         | 27B>`__ |
|                       |            |            |         | &       |
|                       |            |            |         | `StarC  |
|                       |            |            |         | oder <h |
|                       |            |            |         | ttps:// |
|                       |            |            |         | hugging |
|                       |            |            |         | face.co |
|                       |            |            |         | /datase |
|                       |            |            |         | ts/bigc |
|                       |            |            |         | ode/sta |
|                       |            |            |         | rcoderd |
|                       |            |            |         | ata>`__ |
+-----------------------+------------+------------+---------+---------+
| `Optimize parquet     | Parquet    | 12         | 16      | R       |
| files for model       | Files      |            |         | andomly |
| training <h           |            |            |         | Ge      |
| ttps://lightning.ai/l |            |            |         | nerated |
| ightning-ai/studios/c |            |            |         | data    |
| onvert-parquets-to-li |            |            |         |         |
| ghtning-streaming>`__ |            |            |         |         |
+-----------------------+------------+------------+---------+---------+

 

--------------

Community
=========

LitData is a community project accepting contributions - Let’s make the
world’s most advanced AI data processing framework.

| 💬 `Get help on Discord <https://discord.com/invite/XncpTy7DSt>`__
| 📋 `License: Apache
  2.0 <https://github.com/Lightning-AI/litdata/blob/main/LICENSE>`__

--------------

Citation
--------

::

   @misc{litdata2023,
     author       = {Thomas Chaton and Lightning AI},
     title        = {LitData: Transform datasets at scale. Optimize datasets for fast AI model training.},
     year         = {2023},
     howpublished = {\url{https://github.com/Lightning-AI/litdata}},
     note         = {Accessed: 2025-04-09}
   }

--------------

Papers with LitData
-------------------

- `Towards Interpretable Protein Structure Prediction with Sparse
  Autoencoders <https://arxiv.org/pdf/2503.08764>`__ \|
  `Github <https://github.com/johnyang101/reticular-sae>`__ \| (Nithin
  Parsan, David J. Yang and John J. Yang)

--------------

Governance
==========

Maintainers
-----------

- Thomas Chaton (`tchaton <https://github.com/tchaton>`__)
- Luca Antiga (`lantiga <https://github.com/lantiga>`__)
- Justus Schock (`justusschock <https://github.com/justusschock>`__)
- Bhimraj Yadav (`bhimrazy <https://github.com/bhimrazy>`__)
- Deependu (`deependujha <https://github.com/deependujha>`__)
- Jirka Borda (`Borda <https://github.com/Borda>`__)

Emeritus Maintainers
--------------------

- Adrian Wälchli (`awaelchli <https://github.com/awaelchli>`__)

.. |PyPI| image:: https://img.shields.io/pypi/v/litdata
.. |Downloads| image:: https://img.shields.io/pypi/dm/litdata
.. |License| image:: https://img.shields.io/github/license/Lightning-AI/litdata
