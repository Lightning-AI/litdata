# ruff: noqa: RET504
import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

import litdata as ld


class SineModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.tanh(self.fc4(x))  # for output to be in -1 to 1
        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch["x"], batch["sine"]
        x = x.view(x.size(0), -1)
        x = self.forward(x)

        loss = F.mse_loss(x.squeeze(), y)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch["x"], batch["sine"]
        x = x.view(x.size(0), -1)
        x = self.forward(x)

        test_loss = F.mse_loss(x.squeeze(), y)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch["x"], batch["sine"]
        x = x.view(x.size(0), -1)
        x = self.forward(x)

        val_loss = F.mse_loss(x.squeeze(), y)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class SineDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        dataset = ld.StreamingDataset(self.data_dir)
        self.train_dataset, self.val_dataset, self.test_dataset = ld.train_test_split(dataset, splits=[0.7, 0.1, 0.1])

    def train_dataloader(self):
        return ld.StreamingDataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=7, persistent_workers=True
        )

    def val_dataloader(self):
        return ld.StreamingDataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=7, persistent_workers=True
        )

    def test_dataloader(self):
        return ld.StreamingDataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=7, persistent_workers=True
        )


# ======================================================


if __name__ == "__main__":
    model = SineModule()
    data = SineDataModule("example_optimize_dataset")

    trainer = L.Trainer(max_epochs=100, accelerator="cpu", precision="64-true")
    trainer.fit(model, data)
    trainer.test(model, data)
