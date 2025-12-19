"""
Main script for Phase 1: Baseline Training
Hurricane Damage Detection with ResNet-18
"""
import os
import torch
import numpy as np
import random
from pathlib import Path

# Import custom modules
from data.data_loader import (
    GeoEye1, compute_dataset_statistics, 
    get_dataloaders
)
from models.resnet import get_resnet18, count_parameters
from training.train import train_model
from calibration.calibration import (
    TemperatureScaling, collect_logits_and_labels
)
from evaluation.metrics import compute_all_metrics
from evaluation.visualization import (
    plot_reliability_diagram, plot_confusion_matrix,
    plot_confidence_histogram, plot_calibration_comparison
)


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Configuration
    config = {
        'data_dir': 'ipeo_hurricane_for_students',
        'image_size': 150,
        'batch_size': 32,
        'num_workers': 4,
        'max_epochs': 20,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'optimizer_type': 'adam',
        'scheduler_type': 'cosine',
        'save_dir': 'checkpoints/baseline_resnet18',
        'experiment_name': 'baseline_resnet18',
        'use_class_weights': True,
    }
    
    # Create directories
    Path(config['save_dir']).mkdir(parents=True, exist_ok=True)
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PHASE 1: BASELINE TRAINING - Hurricane Damage Detection")
    print("=" * 80)
    
    # =========================================================================
    # STEP 1: Compute dataset statistics (or use hardcoded values)
    # =========================================================================
    print("\n[1/6] Computing dataset statistics...")
    
    # If you already computed these, you can hardcode them:
    # mean = torch.tensor([0.4353, 0.4570, 0.4156])
    # std = torch.tensor([0.2097, 0.1875, 0.1877])
    
    # Otherwise compute:
    mean, std = compute_dataset_statistics(
        root_dir=config['data_dir'],
        split='train',
        image_size=config['image_size'],
        batch_size=64,
        num_workers=config['num_workers']
    )
    
    print(f"Dataset Mean: {mean}")
    print(f"Dataset Std:  {std}")
    
    # =========================================================================
    # STEP 2: Create dataloaders
    # =========================================================================
    print("\n[2/6] Creating dataloaders...")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=config['data_dir'],
        mean=mean,
        std=std,
        image_size=config['image_size'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    
    # Analyze class distribution
    train_dataset = train_loader.dataset
    class_dist = train_dataset.get_class_distribution()
    print(f"\nClass distribution (train): {class_dist}")
    
    # Compute class weights for handling imbalance
    if config['use_class_weights']:
        total = sum(class_dist.values())
        class_weights = [total / (len(class_dist) * count) for count in class_dist.values()]
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        print(f"Class weights: {class_weights}")
        config['class_weights'] = class_weights
    else:
        config['class_weights'] = None
    
    # =========================================================================
    # STEP 3: Create and train model
    # =========================================================================
    print("\n[3/6] Creating ResNet-18 model...")
    
    model = get_resnet18(num_classes=2, pretrained=True, freeze_backbone=False)
    
    trainable, total = count_parameters(model)
    print(f"Model parameters: {total:,} total, {trainable:,} trainable")
    
    print("\nStarting training...")
    trainer, lightning_model = train_model(model, train_loader, val_loader, config)
    
    print("\nTraining complete!")
    print(f"Best model saved to: {config['save_dir']}")
    
    # =========================================================================
    # STEP 4: Evaluate on validation set
    # =========================================================================
    print("\n[4/6] Evaluating on validation set...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lightning_model.eval()
    lightning_model.to(device)
    
    # Collect predictions
    val_logits, val_labels = collect_logits_and_labels(
        lightning_model, val_loader, device
    )
    
    # Compute metrics
    class_names = ['No Damage', 'Damage']
    val_metrics = compute_all_metrics(val_logits, val_labels, class_names, n_bins=15)
    
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS (Before Calibration)")
    print("=" * 50)
    print(f"Accuracy:     {val_metrics['accuracy']:.4f}")
    print(f"Precision:    {val_metrics['precision']:.4f}")
    print(f"Recall:       {val_metrics['recall']:.4f}")
    print(f"F1-Score:     {val_metrics['f1']:.4f}")
    print(f"ROC-AUC:      {val_metrics['roc_auc']:.4f}")
    print(f"\nCalibration Metrics:")
    print(f"ECE:          {val_metrics['ece']:.4f}")
    print(f"MCE:          {val_metrics['mce']:.4f}")
    print(f"Brier Score:  {val_metrics['brier_score']:.4f}")
    print(f"NLL:          {val_metrics['nll']:.4f}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    import torch.nn.functional as F
    val_probs = F.softmax(val_logits, dim=1)
    
    # Reliability diagram
    plot_reliability_diagram(
        val_probs, val_labels, n_bins=15,
        title="Reliability Diagram (Before Calibration)",
        save_path='results/figures/reliability_before_calibration.png'
    )
    
    # Confusion matrix
    plot_confusion_matrix(
        val_metrics['confusion_matrix'], class_names,
        title="Confusion Matrix - Validation Set",
        save_path='results/figures/confusion_matrix_val.png'
    )
    
    # Confidence histogram
    plot_confidence_histogram(
        val_probs, val_labels,
        title="Confidence Distribution (Before Calibration)",
        save_path='results/figures/confidence_histogram_before.png'
    )
    
    # =========================================================================
    # STEP 5: Apply temperature scaling calibration
    # =========================================================================
    print("\n[5/6] Applying temperature scaling calibration...")
    
    temp_scaler = TemperatureScaling()
    temp_scaler.fit(val_logits.to(device), val_labels.to(device))
    
    # Get calibrated predictions
    val_logits_calibrated = temp_scaler(val_logits.to(device))
    val_probs_calibrated = F.softmax(val_logits_calibrated, dim=1)
    
    # Compute metrics after calibration
    val_metrics_cal = compute_all_metrics(
        val_logits_calibrated.cpu(), val_labels, class_names, n_bins=15
    )
    
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS (After Calibration)")
    print("=" * 50)
    print(f"Accuracy:     {val_metrics_cal['accuracy']:.4f}")
    print(f"F1-Score:     {val_metrics_cal['f1']:.4f}")
    print(f"\nCalibration Metrics:")
    print(f"ECE:          {val_metrics_cal['ece']:.4f} (before: {val_metrics['ece']:.4f})")
    print(f"MCE:          {val_metrics_cal['mce']:.4f} (before: {val_metrics['mce']:.4f})")
    print(f"Brier Score:  {val_metrics_cal['brier_score']:.4f} (before: {val_metrics['brier_score']:.4f})")
    print(f"NLL:          {val_metrics_cal['nll']:.4f} (before: {val_metrics['nll']:.4f})")
    
    # Visualizations after calibration
    plot_reliability_diagram(
        val_probs_calibrated.cpu(), val_labels, n_bins=15,
        title="Reliability Diagram (After Calibration)",
        save_path='results/figures/reliability_after_calibration.png'
    )
    
    plot_confidence_histogram(
        val_probs_calibrated.cpu(), val_labels,
        title="Confidence Distribution (After Calibration)",
        save_path='results/figures/confidence_histogram_after.png'
    )
    
    # Comparison plot
    plot_calibration_comparison(
        [
            {'name': 'Before Calibration', 'probs': val_probs, 'labels': val_labels},
            {'name': 'After Temperature Scaling', 'probs': val_probs_calibrated.cpu(), 'labels': val_labels}
        ],
        n_bins=15,
        save_path='results/figures/calibration_comparison.png'
    )
    
    # =========================================================================
    # STEP 6: Test set evaluation
    # =========================================================================
    print("\n[6/6] Evaluating on test set...")
    
    # Collect test predictions
    test_logits, test_labels = collect_logits_and_labels(
        lightning_model, test_loader, device
    )
    
    # Apply calibration
    test_logits_calibrated = temp_scaler(test_logits.to(device))
    
    # Compute metrics
    test_metrics = compute_all_metrics(
        test_logits_calibrated.cpu(), test_labels, class_names, n_bins=15
    )
    
    print("\n" + "=" * 50)
    print("TEST SET RESULTS (After Calibration)")
    print("=" * 50)
    print(f"Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"Precision:    {test_metrics['precision']:.4f}")
    print(f"Recall:       {test_metrics['recall']:.4f}")
    print(f"F1-Score:     {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:      {test_metrics['roc_auc']:.4f}")
    print(f"\nCalibration Metrics:")
    print(f"ECE:          {test_metrics['ece']:.4f}")
    print(f"MCE:          {test_metrics['mce']:.4f}")
    print(f"Brier Score:  {test_metrics['brier_score']:.4f}")
    print(f"NLL:          {test_metrics['nll']:.4f}")
    
    # Test set visualizations
    test_probs_calibrated = F.softmax(test_logits_calibrated, dim=1)
    
    plot_reliability_diagram(
        test_probs_calibrated.cpu(), test_labels, n_bins=15,
        title="Reliability Diagram - Test Set",
        save_path='results/figures/reliability_test.png'
    )
    
    plot_confusion_matrix(
        test_metrics['confusion_matrix'], class_names,
        title="Confusion Matrix - Test Set",
        save_path='results/figures/confusion_matrix_test.png'
    )
    
    # Save final results to file
    print("\nSaving results summary...")
    with open('results/baseline_results.txt', 'w') as f:
        f.write("BASELINE RESNET-18 RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("VALIDATION SET\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:     {val_metrics_cal['accuracy']:.4f}\n")
        f.write(f"Precision:    {val_metrics_cal['precision']:.4f}\n")
        f.write(f"Recall:       {val_metrics_cal['recall']:.4f}\n")
        f.write(f"F1-Score:     {val_metrics_cal['f1']:.4f}\n")
        f.write(f"ECE:          {val_metrics_cal['ece']:.4f}\n")
        f.write(f"Brier Score:  {val_metrics_cal['brier_score']:.4f}\n\n")
        
        f.write("TEST SET\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:     {test_metrics['accuracy']:.4f}\n")
        f.write(f"Precision:    {test_metrics['precision']:.4f}\n")
        f.write(f"Recall:       {test_metrics['recall']:.4f}\n")
        f.write(f"F1-Score:     {test_metrics['f1']:.4f}\n")
        f.write(f"ECE:          {test_metrics['ece']:.4f}\n")
        f.write(f"Brier Score:  {test_metrics['brier_score']:.4f}\n")
    
    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - Checkpoints: {config['save_dir']}/")
    print(f"  - Figures: results/figures/")
    print(f"  - Summary: results/baseline_results.txt")
    print(f"  - Logs: logs/{config['experiment_name']}/")


if __name__ == '__main__':
    main()