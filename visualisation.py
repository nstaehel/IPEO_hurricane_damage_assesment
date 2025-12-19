"""
Visualization utilities for results analysis
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from evaluation.metrics import get_calibration_bins


def plot_reliability_diagram(probs, labels, n_bins=15, title="Reliability Diagram", save_path=None):
    """
    Plot reliability diagram (calibration curve)
    
    Args:
        probs: Predicted probabilities (N, num_classes)
        labels: True labels (N,)
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save figure (optional)
    """
    bin_data = get_calibration_bins(probs, labels, n_bins)
    
    # Filter out None values
    valid_indices = [i for i, conf in enumerate(bin_data['confidences']) if conf is not None]
    confidences = [bin_data['confidences'][i] for i in valid_indices]
    accuracies = [bin_data['accuracies'][i] for i in valid_indices]
    counts = [bin_data['counts'][i] for i in valid_indices]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    ax1.plot(confidences, accuracies, 'o-', linewidth=2, markersize=8, label='Model')
    
    # Add error bars based on bin size
    for i, (conf, acc, count) in enumerate(zip(confidences, accuracies, counts)):
        # Standard error approximation
        stderr = np.sqrt(acc * (1 - acc) / count) if count > 0 else 0
        ax1.errorbar(conf, acc, yerr=stderr, fmt='none', ecolor='gray', alpha=0.5)
    
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Histogram of predictions per bin
    ax2.bar(range(len(counts)), counts, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Confidence Bin', fontsize=12)
    ax2.set_ylabel('Number of Predictions', fontsize=12)
    ax2.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confidence_histogram(probs, labels, title="Confidence Histogram", save_path=None):
    """
    Plot histogram of model confidence for correct and incorrect predictions
    
    Args:
        probs: Predicted probabilities (N, num_classes)
        labels: True labels (N,)
        title: Plot title
        save_path: Path to save figure
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu()
        labels = labels.detach().cpu()
    
    confidences, predictions = torch.max(probs, 1)
    correct = predictions.eq(labels)
    
    conf_correct = confidences[correct].numpy()
    conf_incorrect = confidences[~correct].numpy()
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(conf_correct, bins=30, alpha=0.6, label='Correct', color='green', edgecolor='black')
    plt.hist(conf_incorrect, bins=30, alpha=0.6, label='Incorrect', color='red', edgecolor='black')
    
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", normalize=False, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        normalize: Whether to normalize values
        save_path: Path to save figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(history_dict, save_path=None):
    """
    Plot training and validation loss/accuracy curves
    
    Args:
        history_dict: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history_dict['train_loss']) + 1)
    
    # Loss curves
    ax1.plot(epochs, history_dict['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history_dict['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, history_dict['train_acc'], 'b-o', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history_dict['val_acc'], 'r-s', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_calibration_comparison(models_data, n_bins=15, save_path=None):
    """
    Compare calibration of multiple models on same plot
    
    Args:
        models_data: List of dicts with 'name', 'probs', 'labels'
        n_bins: Number of bins
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))
    
    for i, model_data in enumerate(models_data):
        bin_data = get_calibration_bins(model_data['probs'], model_data['labels'], n_bins)
        
        valid_indices = [j for j, conf in enumerate(bin_data['confidences']) if conf is not None]
        confidences = [bin_data['confidences'][j] for j in valid_indices]
        accuracies = [bin_data['accuracies'][j] for j in valid_indices]
        
        plt.plot(confidences, accuracies, 'o-', linewidth=2, markersize=6, 
                color=colors[i], label=model_data['name'])
    
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Calibration Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_roc_curve(probs, labels, title="ROC Curve", save_path=None):
    """
    Plot ROC curve for binary classification
    
    Args:
        probs: Predicted probabilities (N, 2)
        labels: True labels (N,)
        title: Plot title
        save_path: Path to save figure
    """
    from sklearn.metrics import roc_curve, auc
    
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Get probabilities for positive class
    if probs.ndim == 2:
        probs = probs[:, 1]
    
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_predictions(images, labels, probs, class_names, num_samples=16, save_path=None):
    """
    Visualize sample predictions with confidence
    
    Args:
        images: Batch of images (N, C, H, W)
        labels: True labels (N,)
        probs: Predicted probabilities (N, num_classes)
        class_names: List of class names
        num_samples: Number of samples to show
        save_path: Path to save figure
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()
    
    num_samples = min(num_samples, len(images))
    
    # Create grid
    rows = int(np.sqrt(num_samples))
    cols = (num_samples + rows - 1) // rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for i in range(num_samples):
        # Denormalize image for display
        img = images[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        pred_class = probs[i].argmax().item()
        confidence = probs[i].max().item()
        true_class = labels[i].item()
        
        # Plot
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # Title with prediction
        color = 'green' if pred_class == true_class else 'red'
        title = f"True: {class_names[true_class]}\nPred: {class_names[pred_class]} ({confidence:.2f})"
        axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')
    
    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()