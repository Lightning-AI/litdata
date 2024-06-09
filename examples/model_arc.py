""" Architecure Bert & Resnet lightning """
import os

import torch
from lightning import seed_everything
from torch import Tensor, nn
from torchvision.models import resnet18
from transformers import AdamW, BertConfig, BertModel

import logging

logger = logging.getLogger(__name__)

torch.manual_seed(21)
seed = seed_everything(21, workers=True)


class ResNet18(nn.Module):
    """ResNet 18 cut of the last layer for latent space representation"""

    def __init__(self, num_classes: int, hyperparameters: dict, endpoint_mode: bool):
        super(ResNet18, self).__init__()
        self.model = resnet18(weights=None)
        logger.info(hyperparameters)
        logger.info(num_classes)
        self.hyperparameters = hyperparameters
        self.model.fc = nn.Identity()

    def forward(self, x):
        """forward step for resnet18"""
        return self.model(x)


class BertClassifier(nn.Module):
    """Bert Classifier Model"""

    def __init__(self, bert_model):
        super(BertClassifier, self).__init__()
        self.bert = bert_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Forward path, calculate the computational graph in the forward direction. Used for train, test and val.
        Args:
            input_ids
            attention_mask
        Returns:
            computional graph
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return cls_output


class BertResNetClassifier(nn.Module):
    """Bert Resnet Classifier Model"""

    def __init__(self, endpoint_mode: bool, hyperparameters: dict):
        super(BertResNetClassifier, self).__init__()
        self.endpoint_mode = endpoint_mode
        self.hyperparameters = hyperparameters
        self.num_classes = self.hyperparameters["num_classes"]
        self.text_module = self.get_bert_model()
        self.feature_extractor = ResNet18(
            self.num_classes,
            self.hyperparameters,
            endpoint_mode=self.endpoint_mode,
        )

        self.projection = nn.Linear(512, 768)
        self.classifier = nn.Linear(768, self.num_classes)
        self.dropout = nn.Dropout(self.hyperparameters["dropout"])

    def get_bert_model(self):
        """
        Load the pre trained bert model weigths
        Returns: model
        """
        prefix = self.hyperparameters.get("artefact_location_path", os.getcwd())
        #config = BertConfig.from_json_file(os.path.join(prefix, self.hyperparameters["model_config_name"]))
        #model = BertModel.from_pretrained(os.path.join(prefix, self.hyperparameters["model_path_name"]), config=config)
        model = BertModel.from_pretrained("bert-base-cased")
        text_classification_model = BertClassifier(model)
        return text_classification_model

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass to calculate the computational graph. This method is used during training, 
        testing, and validation.
        
        Args:
            x (torch.Tensor): Tensor with id tokesn
            y (torch.Tensor): Tensor with attention tokens.
            z (torch.Tensor): Tensor with iamge.
        
        Returns:
            torch.Tensor: The output tensor representing the computational graph.
        """
        text_y = self.text_module(x, y)
        img = self.feature_extractor(z)
        ret = self.projection(img)
        ret = text_y + ret
        ret = self.dropout(ret)
        ret = self.classifier(ret)
        return ret