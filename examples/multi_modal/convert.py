import io
import os

from litdata import optimize
from PIL import Image
from pyarrow.parquet import ParquetFile


def convert_parquet_to_lightning_data(parquet_file):
    parquet_file = ParquetFile(parquet_file)
    for batch in parquet_file.iter_batches(batch_size=32):
        df = batch.to_pandas()
        df["page"] = df["page"].apply(lambda x: Image.open(io.BytesIO(x)))
        for row in df.itertuples():
            # 'numerical_id': 'numerical id as int', 'text': 'text of the ocred document',
            # 'page': image of the document, 'label2': 'class label of the document'
            sample = (row.numerical_id, row.text, row.label2, row.page)
            yield sample


def pl_optimize(input_dir, output_dir):
    parquet_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".parquet")])
    print(parquet_files)
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
        input_dir = "/teamspace/studios/this_studio/dataframe_data"
        output_dir = "/teamspace/datasets/lightning_data/version_0"
    pl_optimize(input_dir, output_dir)
