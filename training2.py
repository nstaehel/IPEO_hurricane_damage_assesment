import pandas as pd
from src.models.lightningmodel import modelling_choice
import torch
from src.preprocessing.data_loader import get_dataloaders
from pytorch_lightning.callbacks import Timer
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Define your search space
arch = "cnn_from_paper"
optimizers = ["sgd", "adam"]
results_list = []
timer_list= {}
augment_train_list = [True, False]

root_dir="ipeo_hurricane_for_students"
mean = torch.load("src/preprocessing/mean.pt")
std = torch.load("src/preprocessing/std.pt")

max_epochs = 50

for augment_train in augment_train_list:
    train_loader, val_loader, _ = get_dataloaders(root_dir=root_dir, mean=mean, std=std, batch_size=100, num_workers=4, augment_train=augment_train)
    for opt in optimizers:
        print(f"\n Starting Experiment: Model={arch}, Optimizer={opt}, Max Epochs={max_epochs}, Augment Train={augment_train} \n")
            
        # 1. Initialize Trainer and Model
        trainer, lightning_model = modelling_choice(
            pretrained=True,
            patience=5,
            model_name=arch, 
            optimizer_name=opt, 
            max_epochs=max_epochs, # Early stopping will likely cut this short
            train_loader=train_loader,
            layer_to_freeze=2,
            train_is_augmented=augment_train
        )
        # 2. Train the model
        trainer.fit(lightning_model, train_loader, val_loader)
        
        # 3. Extract best metrics from callbacks or logger
        # We take the best score recorded by ModelCheckpoint
        best_score = trainer.checkpoint_callback.best_model_score.item()
        timer_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, Timer):
                timer_callback = callback
                break

        if timer_callback:
            timer = timer_callback.time_elapsed("train")
            timer_list[f"{arch}-{opt}-augment_train-{augment_train}"] = timer
        else:
            print("Timer callback not found in trainer.callbacks")
            timer = None
        results_list.append({
            "architecture": arch,
            "optimizer": opt,
            "augment_train": augment_train,
            "max_epochs": max_epochs,
            "training_time_sec": timer,
            "best_lr": lightning_model.lr,
            "best_val_loss": best_score,
            "epochs_run": trainer.current_epoch
        })

# 4. Create Leaderboard
df_results = pd.DataFrame(results_list)
df_results.to_csv("model_comparison_results_from_paper.csv", index=False)
print("\n--- Experiment Summary ---")
print(df_results.sort_values(by="best_val_loss", ascending=False))