from src.models.lightningmodel import LightningClassifierModelWrapper, modelling_choice, count_parameters
from src.preprocessing.data_loader import GeoEye1, compute_dataset_statistics, get_transforms, get_dataloaders
import torch
import pandas as pd
import matplotlib.pyplot as plt

trainer, lightning_model = modelling_choice(model_name="resnet18", max_epochs=40, pretrained=True)

root_dir="ipeo_hurricane_for_students"
mean = torch.load("src/preprocessing/mean.pt")
std = torch.load("src/preprocessing/std.pt")

train_loader, val_loader, test_loader = get_dataloaders(root_dir, mean=mean, std=std, batch_size=100)
# Get a single batch from the dataloader
data_iter = iter(val_loader)
images, labels = next(data_iter)

# Take just the first image from the batch
single_image = images[0]  # Shape: [3, 150, 150]
single_label = labels[0]

# Add batch dimension if needed
single_image_batch = single_image.unsqueeze(0)  # Shape: [1, 3, 150, 150]

# disable randomness, dropout, etc...
lightning_model.eval()
activations = []
 
def hook(module, input, output):
    activations.append(output.detach())

handle = lightning_model.model.relu.register_forward_hook(hook)

# predict with the model
y_hat = lightning_model(single_image_batch)
handle.remove()
# Visualize the activations
num_activations = activations[0].shape[1]  # Number of feature maps
fig, axes = plt.subplots(1, num_activations, figsize=(15, 15))
for i in range(num_activations):
    axes[i].imshow(activations[0][0, i].cpu(), cmap='viridis')
    axes[i].axis('off')
plt.tight_layout()
plt.savefig('logs/figures/activations.png', dpi=300, bbox_inches='tight')
print("Activation maps saved as 'activations.png'")