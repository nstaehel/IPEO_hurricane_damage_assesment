# %%
from src.models.lightningmodel import LightningClassifierModelWrapper, modelling_choice, count_parameters
from src.preprocessing.data_loader import GeoEye1, compute_dataset_statistics, get_transforms, get_dataloaders
import torch
import pandas as pd

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
rs = trainer.predict(lightning_model, dataloaders=test_loader)
y_hat, y = list(map(list, zip(*rs)))
y_hat = torch.vstack(y_hat)
y = torch.hstack(y)
accuracy = (y_hat.argmax(1) == y).float().mean()
print(f"test accuracy {accuracy:.4f}")
trainable_params, total_params = count_parameters(lightning_model)
print(f"Trainable parameters: {trainable_params}, Total parameters: {total_params}")    


