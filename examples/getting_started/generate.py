import numpy as np
from PIL import Image

from litdata import optimize


# Store random images into the data chunks
def random_images(index):
    # The data is serialized into bytes and stored into data chunks by the optimize operator.
    return {
        "index": index,  # int data type
        "image": Image.fromarray(np.random.randint(0, 256, (32, 32, 3), np.uint8)),  # PIL image data type
        "class": np.random.randint(10),  # numpy array data type
    }


if __name__ == "__main__":
    optimize(
        fn=random_images,  # The function applied over each input.
        inputs=list(range(1000)),  # Provide any inputs. The fn is applied on each item.
        output_dir="my_optimized_dataset",  # The directory where the optimized data are stored.
        num_workers=4,  # The number of workers. The inputs are distributed among them.
        chunk_bytes="64MB",  # The maximum number of bytes to write into a data chunk.
    )
