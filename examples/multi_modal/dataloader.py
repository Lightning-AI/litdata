"""Lightning dataloader for text and image data."""

import logging
import os
from typing import Any, Union

import joblib
import lightning as pl
import numpy as np
import torch
from examples.multi_modal.config import HYPERPARAMETERS
from lightning import seed_everything
from litdata import StreamingDataLoader, StreamingDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizerFast

pil_transform = transforms.Compose([transforms.PILToTensor()])


logger = logging.getLogger()


os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.manual_seed(21)
seed = seed_everything(21, workers=True)


class EncoderAndTokenizer:
    def __init__(self):
        self.hyperparameters = HYPERPARAMETERS

    def load_labelencoder(self):
        """
        Function to load the label encoder from s3
        Returns:
        """
        labelencoder = joblib.load(self.hyperparameters["label_encoder_name"])
        return labelencoder

    def load_tokenizer(self):
        """
        load the tokenizer files and the pre training model path from s3 spezified in the hyperparameters
        Returns: tokenizer
        """
        # Load Bert tokenizer
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        return tokenizer


class IHDataset(StreamingDataset):
    """Streaming dataset class."""

    def __init__(self, input_dir: Union[str, "Dir"], hyperparameters: Union[dict, Any] = None) -> None:
        super().__init__(input_dir, shuffle=True, max_cache_size=hyperparameters["max_cache_size"])
        self.hyperparameters = hyperparameters
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5], [0.5]),
            ]
        )
        EC = EncoderAndTokenizer()
        self.tokenizer = EC.load_tokenizer()
        self.labelencoder = EC.load_labelencoder()

    def tokenize_data(self, tokenizer, texts, max_length: int):
        """
        Tokenize the text
        Args:
            tokenizer:
            texts:
            max_length:
        Returns: input_ids, attention_masks
        """
        encoded_text = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded_text.input_ids
        attention_masks = encoded_text.attention_mask
        return input_ids, attention_masks

    def __getitem__(self, index):
        numerical_id, text, label2, page = super().__getitem__(index)
        text_ind, text_att = self.tokenize_data(self.tokenizer, text, 512)
        text_ind = np.squeeze(text_ind, axis=0)
        text_att = np.squeeze(text_att, axis=0)
        try:
            img = pil_transform(page.convert("RGB")).float()
        except:
            img = page.float()

        label = self.labelencoder.transform([label2])

        return {"ID": text_ind, "Att": text_att, "NID": numerical_id, "text": text, "GT": label[0], "IMG": img}


class MixedDataModule(pl.LightningDataModule):
    """Own DataModule form the pytorch lightning DataModule."""

    def __init__(self, hyperparameters: dict):
        """
        Init if the Data Module
        Args:
            data_path: dataframe with the data
            hyperparameters:  Hyperparameters
        """
        super().__init__()
        self.hyperparameters = hyperparameters
        self.batch_size = hyperparameters["batch_size"]
        self.hyperparameters = hyperparameters
        self.num_classes = hyperparameters["num_classes"]
        # Augmentation policy for training set
        self.augmentation = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=300, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5], [0.5]),
            ]
        )
        # Preprocessing steps applied to validation and test set.
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5], [0.5]),
            ]
        )

    def train_dataloader(self) -> DataLoader:
        """
        Define the training dataloader
        Returns:
            training dataloader
        """
        dataset_train = IHDataset(
            hyperparameters=self.hyperparameters,
            input_dir=self.hyperparameters["train_shards"],
        )

        dataset_train.image_transform = self.augmentation
        print(f"LEN OF DATASET Train: {len(dataset_train)}")
        return StreamingDataLoader(
            dataset_train,
            batch_size=self.batch_size,
            num_workers=self.hyperparameters["num_workers"],
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Define the validation dataloader
        Returns:
            validation dataloader
        """
        dataset_val = IHDataset(
            hyperparameters=self.hyperparameters,
            input_dir=self.hyperparameters["val_shards"],
        )
        dataset_val.image_transform = self.transform
        return StreamingDataLoader(
            dataset_val,
            batch_size=self.batch_size,
            num_workers=self.hyperparameters["num_workers"],
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Define the test dataloader
        Returns:
            test dataloader
        """
        dataset_test = IHDataset(
            hyperparameters=self.hyperparameters,
            input_dir=self.hyperparameters["test_shards"],
        )
        dataset_test.image_transform = self.transform
        print(f"LEN OF DATASET Test: {len(dataset_test)}")
        return StreamingDataLoader(
            dataset_test,
            batch_size=self.batch_size,
            num_workers=self.hyperparameters["num_workers"],
            pin_memory=False,
        )


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
