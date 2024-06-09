import io
import os

from litdata import optimize
from PIL import Image
from pyarrow.parquet import ParquetFile

# from s3path import S3Path


def convert_parquet_to_lightning_data(parquet_file):
    try:
        df_bytes = parquet_file.read_bytes()
        str_df_bytes = io.BytesIO(df_bytes)
        parquet_file = ParquetFile(str_df_bytes)
    except:
        parquet_file = ParquetFile(parquet_file)

    for batch in parquet_file.iter_batches(batch_size=32):
        df = batch.to_pandas()
        df["page"] = df["page"].apply(lambda x: Image.open(io.BytesIO(x)))
        for row in df.itertuples():
            # 'numerical_id', 'text', 'page', 'label2'
            sample = (row.numerical_id, row.text, row.label2, row.page)
            yield sample  # -> encode the sample into binary chunks


def pl_optimize(input_dir, output_dir):
    # input_s3_dir = S3Path.from_uri(input_dir)
    # parquet_files = [file for file in input_s3_dir.rglob("*.parquet")]
    parquet_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".parquet")])
    print(parquet_files)
    # 3. Apply the optimize operator over the parquet files
    optimize(
        convert_parquet_to_lightning_data, parquet_files, output_dir, num_workers=os.cpu_count(), chunk_bytes="64MB"
    )


if __name__ == "__main__":
    mod = "local"
    if mod == "local":
        input_dir = "dataframe_data"
        output_dir = "lightning_data/version_0"
        pass
    else:
        # 1. List the parquet files
        input_dir = "/teamspace/studios/this_studio/dataframe_data"
        output_dir = "/teamspace/datasets/lightning_data/version_0"
        # parquet_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
    pl_optimize(input_dir, output_dir)
