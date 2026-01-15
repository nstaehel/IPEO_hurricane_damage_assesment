import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import time
import os
import pandas as pd
from src.models.lightningmodel import LightningClassifierModelWrapper, CNNModel_from_paper
import torch
from src.preprocessing.data_loader import get_dataloaders
from torchgeo.models import ResNet18_Weights, resnet18
from torchgeo.models import ResNet50_Weights, resnet50
import torchvision.models as models
import torch
import pytorch_lightning as pl

def evaluate_with_lightning(checkpoint_dir, test_loader):
    """Evaluate saved checkpoints using PyTorch Lightning and log results to CSV.
     Args:
        checkpoint_dir: Directory containing model checkpoints
        test_loader: DataLoader for the test dataset
    Returns:
        DataFrame with evaluation results
    """
    
    checkpoint_path_list = os.listdir(checkpoint_dir)
    checkpoint_path_list = [f for f in checkpoint_path_list if f.endswith('.ckpt')]
    
    results = []
    
    for checkpoint_file in checkpoint_path_list:
        architecture = checkpoint_file[:-5].split('-')[0]
        optimizer = checkpoint_file[:-5].split('-')[1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        
        print(f"\nEvaluating: {checkpoint_file}")
        
        try:
            if architecture == "resnet18":
                base_model = resnet18(weights=None, num_classes=2)
            elif architecture == "resnet50":
                base_model = resnet50(weights=None, num_classes=2)
            elif architecture == "alexnet":
                base_model = models.alexnet(weights=None, num_classes=2)
            elif architecture == "cnn_from_paper":
                base_model = CNNModel_from_paper(input_size=150, num_classes=2)
            else:
                raise ValueError(f"Unknown architecture: {architecture}")

            lightning_model = LightningClassifierModelWrapper.load_from_checkpoint(
                checkpoint_path,
                model=base_model  # Pass the model as a parameter
            )
            # Create trainer for testing
            trainer = pl.Trainer(
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=True,
                devices=1,
                accelerator="auto"
            )
            
            # Measure inference time
            
            start_time = time.time()
            
            # Run test
            test_results = trainer.test(
                model=lightning_model,
                dataloaders=test_loader,
                verbose=True
            )
            
            inference_time = time.time() - start_time
            
            # Extract results
            result = {
                'checkpoint_file': checkpoint_file,
                'model_name': lightning_model.hparams.get('name', 'unknown'),
                'optimizer_name': lightning_model.hparams.get('optimizer_name', 'unknown'),
                'learning_rate': lightning_model.hparams.get('lr', 'unknown'),
                'train_is_augmented': lightning_model.hparams.get('train_is_augmented', False),
                'inference_time': inference_time,
                **test_results[0]  # Unpack test metrics
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv("/home/nstaehel/IPEO_hurricane_assesment/lightning_test_results.csv", index=False)
    
    return df

# Example usage:
root_dir = "ipeo_hurricane_for_students"
mean = torch.load("src/preprocessing/mean.pt")
std = torch.load("src/preprocessing/std.pt")
train_loader, val_loader, test_loader = get_dataloaders(root_dir=root_dir, mean=mean, std=std, batch_size=100, num_workers=4)
df_results = evaluate_with_lightning("/home/nstaehel/IPEO_hurricane_assesment/checkpoints/", test_loader)