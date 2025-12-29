# %%
from src.models.lightningmodel import LightningClassifierModelWrapper, modelling_choice, count_parameters
from src.preprocessing.data_loader import GeoEye1, compute_dataset_statistics, get_transforms, get_dataloaders
import torch
import pandas as pd
import matplotlib.pyplot as plt

# %%
root_dir="ipeo_hurricane_for_students"
trainer, lightning_model = modelling_choice(model_name="resnet18", max_epochs=40, pretrained=True)
#mean, std = compute_dataset_statistics(root_dir, split="train", batch_size=1000)

# %%
mean = torch.load("src/preprocessing/mean.pt")
std = torch.load("src/preprocessing/std.pt")

# %%
train_loader, val_loader, test_loader = get_dataloaders(root_dir, mean=mean, std=std, batch_size=100)
trainer.fit(lightning_model, train_loader, val_loader)
trainable_params, total_params = count_parameters(lightning_model)
print(f"Trainable parameters: {trainable_params}, Total parameters: {total_params}")    

# %%
# Simple plotting for quick visualization of validation metrics
if not lightning_model.val_metrics_df.empty:
    df = lightning_model.val_metrics_df
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Define metrics to plot
    metrics_config = [
        {'col': 'val_loss', 'color': 'blue', 'title': 'Validation Loss', 'ylabel': 'Loss'},
        {'col': 'val_accuracy', 'color': 'green', 'title': 'Validation Accuracy', 'ylabel': 'Accuracy'},
        {'col': 'val_f1', 'color': 'red', 'title': 'Validation F1 Score', 'ylabel': 'F1 Score'}
    ]
    
    for ax, config in zip(axes, metrics_config):
        if config['col'] in df.columns and df[config['col']].notna().any():
            # Remove NaN values for plotting
            plot_data = df[['epoch', config['col']]].dropna()
            
            if not plot_data.empty:
                ax.plot(plot_data['epoch'], plot_data[config['col']], 
                       color=config['color'], linewidth=2, marker='o')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(config['ylabel'])
                ax.set_title(config['title'])
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()

    
    # Save the plots if needed
    fig.savefig('logs/figures/validation_metrics.png', dpi=300, bbox_inches='tight')
    print("Plots saved as 'validation_metrics.png'")

# %%
# Simple plotting for quick visualization of training metrics
if not lightning_model.train_metrics_df.empty:
    df = lightning_model.train_metrics_df
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 1, figsize=(15, 4))
    
    # Define metrics to plot
    metrics_config = [
        {'col': 'train_loss', 'color': 'blue', 'title': 'Training Loss', 'ylabel': 'Loss'},
    ]
    
    for ax, config in zip(axes, metrics_config):
        if config['col'] in df.columns and df[config['col']].notna().any():
            # Remove NaN values for plotting
            plot_data = df[['epoch', config['col']]].dropna()
            
            if not plot_data.empty:
                ax.plot(plot_data['epoch'], plot_data[config['col']], 
                       color=config['color'], linewidth=2, marker='o')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(config['ylabel'])
                ax.set_title(config['title'])
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plots if needed
    fig.savefig('logs/figures/training_metrics.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'training_metrics.png'")


