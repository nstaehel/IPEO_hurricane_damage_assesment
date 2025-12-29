import torchvision.transforms as T
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pandas as pd

class LightningClassifierModelWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore="model")
        self.val_metrics_df = pd.DataFrame()
        self.train_metrics_df = pd.DataFrame()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        """Store training loss in DataFrame"""
        metrics_dict = {
            "epoch": self.current_epoch,
            "train_loss": self.trainer.callback_metrics.get("train_loss", None)
        }
        
        # Convert to numeric values
        numeric_dict = {}
        for key, value in metrics_dict.items():
            if value is not None:
                numeric_dict[key] = value.item() if hasattr(value, 'item') else value
            else:
                numeric_dict[key] = None
        
        # Append to DataFrame
        new_row = pd.DataFrame([numeric_dict])
        self.train_metrics_df = pd.concat([self.train_metrics_df, new_row], ignore_index=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        recall = torchmetrics.Recall(task="binary", num_classes=2).to(self.device)  
        recall_score = recall(y_hat.argmax(1), y)
        f1 = torchmetrics.F1Score(task="binary", num_classes=2).to(self.device)
        f1_score = f1(y_hat.argmax(1), y)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_accuracy", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_recall", recall_score, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", f1_score, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        """Store metrics in DataFrame"""
        metrics_dict = {
            "epoch": self.current_epoch,
            "val_loss": self.trainer.callback_metrics.get("val_loss", None),
            "val_accuracy": self.trainer.callback_metrics.get("val_accuracy", None),
            "val_recall": self.trainer.callback_metrics.get("val_recall", None),
            "val_f1": self.trainer.callback_metrics.get("val_f1", None)
        }
        
        # Convert to numeric values
        numeric_dict = {}
        for key, value in metrics_dict.items():
            if value is not None:
                numeric_dict[key] = value.item() if hasattr(value, 'item') else value
            else:
                numeric_dict[key] = None
        
        # Append to DataFrame
        new_row = pd.DataFrame([numeric_dict])
        self.val_metrics_df = pd.concat([self.val_metrics_df, new_row], ignore_index=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return y_hat, y

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=0.01, momentum=0.9)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        return optimizer
    
def modelling_choice(model_name = "resnet18", max_epochs=5, pretrained=True):
    #set seed for reproducibility
    pl.seed_everything(42, workers=True)
    #model selection
    if model_name == "resnet18":
        if pretrained:
            model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        else:
            model = torchvision.models.resnet18(weights=None)
    elif model_name == "alexnet":
        if pretrained:
            model = torchvision.models.alexnet(weights="IMAGENET1K_V1")
        else:           
            model = torchvision.models.alexnet(weights=None)
    else:
        raise ValueError(f"Model {model_name} not (yet) supported.")
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    lightning_model = LightningClassifierModelWrapper(model)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=pl_loggers.TensorBoardLogger("logs/", name="hurricane"),
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

def count_parameters(model):
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total