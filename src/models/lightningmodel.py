import torchvision.transforms as T
import numpy as np
import os

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.tuner import Tuner
import torchmetrics
from torchgeo.models import ResNet18_Weights, resnet18
from torchgeo.models import ResNet50_Weights, resnet50

from torchvision.transforms import ToTensor
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pandas as pd

class LightningClassifierModelWrapper(pl.LightningModule):
    """
    Lightning Wrapper to create the different models used. See pytorch lightning for more information.
    """

    def __init__(self, model, optimizer_name="sgd", lr=0.01, name="resnet18", train_is_augmented=False):
        super().__init__()
        self.model = model
        self.lr = lr
        self.model_name = name
        self.optimizer_name = optimizer_name
        self.save_hyperparameters(ignore="model")
        self.val_conf_matrix = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(self.device)
        # Initialize metrics
        self.val_conf_matrix = torchmetrics.ConfusionMatrix(task="binary", num_classes=2)
        self.val_recall = torchmetrics.Recall(task="binary", num_classes=2)
        self.val_precision = torchmetrics.Precision(task="binary", num_classes=2)
        self.val_f1 = torchmetrics.F1Score(task="binary", num_classes=2)
        # TEST metrics (separate instances)
        self.test_recall = torchmetrics.Recall(task="binary", num_classes=2)
        self.test_precision = torchmetrics.Precision(task="binary", num_classes=2)
        self.test_f1 = torchmetrics.F1Score(task="binary", num_classes=2)
        # Move metrics to device when needed
        self._metrics_to_device = False
        self.train_is_augmented = train_is_augmented

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
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        
        # Move metrics to device if not already done
        if not self._metrics_to_device:
            self.val_conf_matrix.to(self.device)
            self.val_recall.to(self.device)
            self.val_precision.to(self.device)
            self.val_f1.to(self.device)
            self._metrics_to_device = True
        
        # Update metrics
        preds = y_hat.argmax(1)
        self.val_conf_matrix.update(preds, y)
        self.val_recall.update(preds, y)
        self.val_precision.update(preds, y)
        self.val_f1.update(preds, y)
        
        # Log batch-level metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_accuracy", acc, prog_bar=True, on_step=False, on_epoch=True)
        
    def on_validation_epoch_end(self):
        # Compute and log epoch-level metrics
        recall_score = self.val_recall.compute()
        precision_score = self.val_precision.compute()
        f1_score = self.val_f1.compute()
        
        self.log("val_recall", recall_score, prog_bar=True)
        self.log("val_precision", precision_score, prog_bar=True)
        self.log("val_f1", f1_score, prog_bar=True)
        
        # Reset metrics for next epoch
        self.val_recall.reset()
        self.val_precision.reset()
        self.val_f1.reset()
        
        # Confusion matrix visualization (as before)
        conf_mat = self.val_conf_matrix.compute().cpu().numpy()
        self.val_conf_matrix.reset()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'Confusion Matrix Epoch {self.current_epoch}')
        plt.savefig(f"/home/nstaehel/IPEO_hurricane_assesment/logs/figures/val_conf_mat_{self.model_name}_{self.optimizer_name}_{self.train_is_augmented}_epoch_{self.current_epoch}")
        plt.close(fig)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return y_hat, y

    def configure_optimizers(self):
        # Access the choice from self.hparams
        if self.hparams.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        else:
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.hparams.lr, 
                momentum=0.9
            )
        return optimizer
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        # Get predictions
        preds = y_hat.argmax(1)
        
        # Compute accuracy for this batch (will be averaged automatically)
        acc = (preds == y).float().mean()
        
        # Log with aggregation
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_accuracy", acc, prog_bar=True, on_step=False, on_epoch=True)
        
        # Update test metrics
        self.test_recall.update(preds, y)
        self.test_precision.update(preds, y)
        self.test_f1.update(preds, y)    
        return loss

    def on_test_epoch_end(self):
        """Compute and log test metrics."""
        # Compute metrics
        recall = self.test_recall.compute()
        precision = self.test_precision.compute()
        f1 = self.test_f1.compute()
        
        # Log metrics
        self.log("test_recall", recall, prog_bar=True)
        self.log("test_precision", precision, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        
        # Reset metrics
        self.test_recall.reset()
        self.test_precision.reset()
        self.test_f1.reset()

class CNNModel_from_paper(nn.Module):
    """
    CNN model architecture based on the paper "Detecting Damaged Buildings on Post-Hurricane Satellite Imagery Based on
    Customized Convolutional Neural Networks", by Quoc Dung Cao, Youngjun Choe, https://arxiv.org/abs/1807.01688
    """
    def __init__(self, input_size=150, num_classes=2):
        super(CNNModel_from_paper, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the flattened size
        # After 4 pooling layers with stride 2: 150 -> 75 -> 37 -> 18 -> 9
        # So final spatial dimensions: 9x9
        self.flattened_size = 128 * 9 * 9
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self._calculate_flattened_size(input_size)

    def _calculate_flattened_size(self, input_size):
        # Simulate forward pass to get output size
        test_tensor = torch.randn(1, 3, input_size, input_size)
        x = F.relu(self.conv1(test_tensor))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        self.flattened_size = x.view(1, -1).size(1)
        
        # Update fc1 layer
        self.fc1 = nn.Linear(self.flattened_size, 512)

    def forward(self, x):
        # Convolutional blocks with ReLU activation
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer (sigmoid can be applied in loss function)
        
        return x

def modelling_choice(model_name = "resnet18", patience=5, max_epochs=50, pretrained=True, optimizer_name="sgd", lr=0.01, train_loader=None, layer_to_freeze=2, train_is_augmented=False):
    """
    Create the trainer and lightning model based on input parameters

    Args:
        model_name (str): Name of the model to chose (either "resnet18" or "alexnet" for now)
        max_epochs (int): Max number of epochs to train on
        pretrained (bool): if True, then use pretrained weights (for resnet18, SENTINEL2_RGB_MOCO from torchgeo, and for alexnet, IMAGENET1K_V1 from torchvision)
        optimizer_name (str): Optimizer to use for model, choose between "sgd" and "adam"
        lr (float): Learning rate for sgd optimizer to use a starting point
        train_loader (torch.utils.DataLoader): Dataloader to use for tuning of learning rate when optimizer_name is sgd
        layer_to_freeze (int): Number of the layer of the model up to, and including it, in which the parameters are frozen (check validity of number depending on model)

    Returns:
    pytorch_lightning.Trainer, LightningClassifierModelWrapper: Trainer and Lightning wrapper 
    """
    #set seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    #model selection
    if model_name == "resnet18":
        if pretrained:
            model = resnet18(ResNet18_Weights.SENTINEL2_RGB_MOCO) #cite https://arxiv.org/pdf/1512.03385 !!!!
            layers_to_freeze = ['conv1','bn1','act1']+["layer"+str(n) for n in range(1,layer_to_freeze+1)]
            # Freeze layers up to `layer3`
            for name, layer in model.named_children():
                if name in layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
        else:
            model = torchvision.models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    elif model_name == "alexnet":
        if pretrained:
            model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
            # Freeze all convolutional layers (features section)
            for param in model.features.parameters():
                param.requires_grad = False
        else:           
            model = torchvision.models.alexnet(weights=None)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
    elif model_name == "resnet50":
        if pretrained:
            model = resnet50(ResNet50_Weights.SENTINEL2_RGB_MOCO) #cite https://arxiv.org/abs/2211.07044 !!!!
            layers_to_freeze = ['conv1','bn1','relu','maxpool']+["layer"+str(n) for n in range(1,layer_to_freeze+1)]
            # Freeze layers up to `layer3`
            for name, layer in model.named_children():
                if name in layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
        else:
            model = torchvision.models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    elif model_name == "cnn_from_paper":
        model = CNNModel_from_paper()
    else:
        raise ValueError(f"Model {model_name} not (yet) supported.")
    

    lightning_model = LightningClassifierModelWrapper(model, optimizer_name=optimizer_name, lr=lr, name=model_name, train_is_augmented=train_is_augmented)
    
    early_stop_callback = EarlyStopping(
        monitor="val_f1",   # Metric to monitor
        patience=patience,           # Number of epochs with no improvement after which training will be stopped
        verbose=True, 
        mode="max"            # We want to maximize f1 score, if no improvement after too many epochs then it makes no sense to continue
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{model_name}-{optimizer_name}-{lr}-{{epoch}}-{{val_accuracy:.2f}}-{{val_f1:.2f}}",
        monitor="val_loss",
        mode="min",
        every_n_epochs=5,
        save_top_k=1
    )

    timer_callback = Timer(interval="epoch")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=pl_loggers.CSVLogger("logs/", name=f"csv-{model_name}-{optimizer_name}-{lr}"),
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            timer_callback
        ],
        deterministic=True,
    )
    
    if train_loader is not None:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(lightning_model, train_dataloaders=train_loader, attr_name="lr")
        new_lr = lr_finder.suggestion()
        lightning_model.lr = new_lr
        print(f"Optimal LR found: {new_lr}")
        
    # Set the model to use the best LR found
    lightning_model.lr = lr_finder.suggestion()
    return trainer, lightning_model

def count_parameters(model):
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total