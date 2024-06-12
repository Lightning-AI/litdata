"""Train, validate and test the model."""

import logging
import os

import lightning as pl
from dataloader import MixedDataModule
from loop import LitModel

os.environ["TOKENIZERS_PARALLELISM"] = "true"


logger = logging.getLogger(__name__)


def lightning_training(model_dir: str, hyperparameters: dict) -> object:
    """Executes the training process. This involves creating the dataset and corresponding dataloader, initializing the
    model, and then training, validating, and testing the model.

    Args:
        model_dir (str): The path where the model output will be saved.
        hyperparameters (dict): A dictionary containing the hyperparameters for training.

    Returns:
        model: The trained model.

    """
    logger.debug(f"hyperparameters: {hyperparameters}, {type(hyperparameters)}")
    os.makedirs("lightning_logs", exist_ok=True)
    data_module = MixedDataModule(hyperparameters=hyperparameters)
    number_classes = hyperparameters["num_classes"]
    logger.info("Limit batches %s" % hyperparameters["limit_batches"])
    logger.debug("num_classes %s" % number_classes)

    model = LitModel(
        hyperparameters=hyperparameters,
    )
    trainer = pl.Trainer(
        max_epochs=hyperparameters["max_epochs"],
        accelerator="gpu",
        devices=hyperparameters["devices"],
        default_root_dir=model_dir,
        # strategy=DDPStrategy(find_unused_parameters=True),
        precision=hyperparameters["precision"],
        limit_train_batches=hyperparameters["limit_batches"],
        limit_test_batches=hyperparameters["limit_batches"],
        limit_val_batches=hyperparameters["limit_batches"],
        profiler=hyperparameters["profiler"],
        # fast_dev_run=True,
    )
    trainer.fit(model, data_module)
    logger.debug("trainer model %s" % trainer.model)
    trainer.save_checkpoint("trained_model", weights_only=True)

    if hyperparameters["val_mode"] == "on":
        logger.info("Validate Model")
        logger.debug("trainer_test model %s" % trainer.model)
        trainer.validate(
            model,
            data_module,
            ckpt_path=f"{hyperparameters['model_dir']}/{hyperparameters['model_filename']}.ckpt",
        )
    if hyperparameters["test_mode"] == "on":
        logger.info("Test Model")
        trainer.test(
            model,
            data_module,
            ckpt_path=f"{hyperparameters['model_dir']}/{hyperparameters['model_filename']}.ckpt",
        )
    return trainer


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    from examples.multi_modal.config import HYPERPARAMETERS

    lightning_training(model_dir="logs", hyperparameters=HYPERPARAMETERS)
