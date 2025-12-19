"""
Training utilities using PyTorch Lightning
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics


class LightningClassifier(pl.LightningModule):
    """
    PyTorch Lightning wrapper for image classification
    
    Args:
        model: PyTorch model
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        optimizer_type: 'adam' or 'sgd'
        scheduler_type: 'cosine', 'step', or None
        class_weights: Optional class weights for loss (tensor of shape [num_classes])
    """
    
    def __init__(self, model, lr=1e-3, weight_decay=1e-4, optimizer_type='adam',
                 scheduler_type='cosine', class_weights=None, num_classes=2):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.class_weights = class_weights
        self.num_classes = num_classes
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task='binary' if num_classes == 2 else 'multiclass', 
                                               num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='binary' if num_classes == 2 else 'multiclass',
                                             num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task='binary' if num_classes == 2 else 'multiclass',
                                           num_classes=num_classes, average='macro')
        self.val_precision = torchmetrics.Precision(task='binary' if num_classes == 2 else 'multiclass',
                                                    num_classes=num_classes, average='macro')
        self.val_recall = torchmetrics.Recall(task='binary' if num_classes == 2 else 'multiclass',
                                              num_classes=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # Compute loss with optional class weights
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, y, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, y)
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        
        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # Compute loss
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, y, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, y)
        
        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        # Same as validation
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return logits, y

    def configure_optimizers(self):
        # Optimizer
        if self.optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.lr, 
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
        
        # Scheduler
        if self.scheduler_type is None:
            return optimizer
        
        if self.scheduler_type.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs,
                eta_min=1e-6
            )
        elif self.scheduler_type.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1
            )
        elif self.scheduler_type.lower() == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                }
            }
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_type}")
        
        return [optimizer], [scheduler]


def train_model(model, train_loader, val_loader, config):
    """
    Train a model using PyTorch Lightning
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Dictionary with training configuration
            - max_epochs: Maximum number of epochs
            - lr: Learning rate
            - weight_decay: Weight decay
            - optimizer_type: 'adam', 'adamw', or 'sgd'
            - scheduler_type: 'cosine', 'step', 'plateau', or None
            - class_weights: Optional class weights
            - save_dir: Directory to save checkpoints
            - experiment_name: Name for logging
            
    Returns:
        trainer: PyTorch Lightning Trainer
        lightning_model: Trained LightningClassifier
    """
    # Extract config
    max_epochs = config.get('max_epochs', 20)
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)
    optimizer_type = config.get('optimizer_type', 'adam')
    scheduler_type = config.get('scheduler_type', 'cosine')
    class_weights = config.get('class_weights', None)
    save_dir = config.get('save_dir', 'checkpoints')
    experiment_name = config.get('experiment_name', 'hurricane_detection')
    
    # Convert class weights to tensor if provided
    if class_weights is not None and not isinstance(class_weights, torch.Tensor):
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # Create Lightning module
    lightning_model = LightningClassifier(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        optimizer_type=optimizer_type,
        scheduler_type=scheduler_type,
        class_weights=class_weights,
        num_classes=2
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename=f'{experiment_name}-{{epoch:02d}}-{{val_acc:.3f}}',
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir='logs',
        name=experiment_name
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        deterministic=True
    )
    
    # Train
    trainer.fit(lightning_model, train_loader, val_loader)
    
    return trainer, lightning_model