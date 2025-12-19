from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class LightningClassifierModelWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        f1 = pl.metrics.classification.F1(num_classes=2)
        f1_score = f1(y_hat.argmax(1), y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)
        self.log("val_f1", f1_score, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return y_hat, y

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=0.01, momentum=0.9)
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
    
def modelling_choice(model_name = "resnet18"):
    #set seed for reproducibility
    pl.seed_everything(42, workers=True)
    #model selection
    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    elif model_name == "alexnet":
        model = torchvision.models.alexnet(weights="IMAGENET1K_V1")
    else:
        raise ValueError(f"Model {model_name} not supported.")
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    lightning_model = LightningClassifierModelWrapper(model)

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=pl.loggers.TensorBoardLogger("logs/", name="hurricane"),
        callbacks=[
            ModelCheckpoint(
                dirpath="checkpoints",
                filename=f"{model_name}-{{epoch}}-{{val_accuracy:.2f}}-{{val_f1:.2f}}",
                monitor="val_accuracy",
                mode="max"
            )
        ],
        deterministic=True,
    )
    return trainer, lightning_model

