import os

from config import HYPERPARAMETERS
from convert import pl_optimize
from create_labelencoder import create_labelencoder
from generate import generate_data_set
from train import lightning_training


def main():
    output_dir = "dataframe_data"
    os.makedirs(output_dir, exist_ok=True)
    generate_data_set(output_dir, num_pds=2, num_entries=10)
    create_labelencoder()
    input_dir = "dataframe_data"
    output_dir = "lightning_data/version_0"
    pl_optimize(input_dir, output_dir)
    HYPERPARAMETERS.update({"num_workers": 2})
    HYPERPARAMETERS.update({"max_epochs": 1})
    HYPERPARAMETERS.update({"limit_batches": 1})
    HYPERPARAMETERS.update({"batch_size": 1})
    lightning_training(model_dir="logs", hyperparameters=HYPERPARAMETERS)


if __name__ == "__main__":
    main()
