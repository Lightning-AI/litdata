"""Config file with the Hyperparameter dict."""

import datetime
import logging

current_time = datetime.datetime.now()
timestamp = current_time.strftime("%Y_%m_%d_%H_%M_%S")
logger = logging.getLogger(__name__)

OUT_URI = "output/"


HYPERPARAMETERS = {
    "train_shards": "lightning_data/version_0",
    "val_shards": "lightning_data/version_0",
    "test_shards": "lightning_data/version_0",  # "/teamspace/datasets/lightning_data/version_0"
    "max_cache_size": "1GB",
    "devices": -1,
    "profiler": "advanced",
    "test_mode": "on",
    "val_mode": "on",
    "limit_batches": 100,
    "precision": "16-mixed",
    "out_uri": OUT_URI,
    "batch_size": 4,
    "max_epochs": 1,
    "text_max_length": 512,
    "model_dir": "./artefacts",
    "learning_rate": 2e-5,
    "weight_decay": 1e-2,
    "label_encoder_name": "labelencoder.joblib",
    "resnet18_name": "resnet18.pth",
    "load_model": True,
    "model_path": " ",
    "patience": 1,
    "num_workers": 8,
    "num_classes": 3,
    "model_filename": "test_model",
    "img_size": 224,
    "dropout": 0.25,
    "continue": 0,
    "endpoint_mode": False,
    "pretrained_name": "test_model.ckpt",
    "selected_model": "bert_text_img",
    "input_dir": "lightning_data/",
}


