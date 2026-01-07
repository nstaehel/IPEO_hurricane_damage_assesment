import pandas as pd
from src.models.lightningmodel import modelling_choice
import torch
from src.preprocessing.data_loader import get_dataloaders

# Define your search space
architectures = ["resnet18", "resnet50", "alexnet"]
optimizers = ["sgd", "adam"]
results_list = []

root_dir="ipeo_hurricane_for_students"
ckpt_path = "checkpoints/resnet18-sgd-0.01-epoch=18-val_accuracy=0.66-val_f1=0.47.ckpt"
mean = torch.load("src/preprocessing/mean.pt")
std = torch.load("src/preprocessing/std.pt")
train_loader, val_loader, _ = get_dataloaders(root_dir=root_dir, mean=mean, std=std, batch_size=100, num_workers=4)

for arch in architectures:
    for opt in optimizers:
        print(f"\n Starting Experiment: Model={arch}, Optimizer={opt}, Max Epochs=20")
        
        # 1. Initialize Trainer and Model
        trainer, lightning_model = modelling_choice(
            model_name=arch, 
            optimizer_name=opt, 
            max_epochs=20, # Early stopping will likely cut this short
            train_loader=train_loader
        )
        # Logic to resume ONLY if this is the resnet18/sgd run
        if arch == "resnet18" and opt == "sgd":
            print(f"Resuming {arch} from checkpoint...")
            trainer.fit(lightning_model, train_loader, val_loader, ckpt_path=ckpt_path)
        else:
            # 2. Train the model
            trainer.fit(lightning_model, train_loader, val_loader)
            
            # 3. Extract best metrics from callbacks or logger
            # We take the best score recorded by ModelCheckpoint
            best_score = trainer.checkpoint_callback.best_model_score.item()
            
            results_list.append({
                "architecture": arch,
                "optimizer": opt,
                "best_lr": lightning_model.lr,
                "best_val_acc": best_score,
                "epochs_run": trainer.current_epoch
            })

# 4. Create Leaderboard
df_results = pd.DataFrame(results_list)
df_results.to_csv("model_comparison_results.csv", index=False)
print("\n--- Experiment Summary ---")
print(df_results.sort_values(by="best_val_acc", ascending=False))