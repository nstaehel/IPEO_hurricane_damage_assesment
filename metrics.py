"""
Evaluation metrics for classification and calibration
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)


def compute_classification_metrics(logits, labels, class_names=None):
    """
    Compute comprehensive classification metrics
    
    Args:
        logits: Model logits (N, num_classes)
        labels: True labels (N,)
        class_names: List of class names for report
        
    Returns:
        metrics: Dictionary of metrics
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu()
        labels = labels.detach().cpu()
    
    # Get predictions and probabilities
    probs = F.softmax(logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    labels_np = labels.numpy()
    
    # Basic metrics
    accuracy = accuracy_score(labels_np, preds)
    precision = precision_score(labels_np, preds, average='macro', zero_division=0)
    recall = recall_score(labels_np, preds, average='macro', zero_division=0)
    f1 = f1_score(labels_np, preds, average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(labels_np, preds, average=None, zero_division=0)
    recall_per_class = recall_score(labels_np, preds, average=None, zero_division=0)
    f1_per_class = f1_score(labels_np, preds, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(labels_np, preds)
    
    # ROC-AUC (for binary or use OvR for multi-class)
    try:
        if probs.shape[1] == 2:
            roc_auc = roc_auc_score(labels_np, probs[:, 1])
        else:
            roc_auc = roc_auc_score(labels_np, probs, multi_class='ovr', average='macro')
    except:
        roc_auc = None
    
    # Classification report
    if class_names is None:
        class_names = [f"Class {i}" for i in range(probs.shape[1])]
    report = classification_report(labels_np, preds, target_names=class_names)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'classification_report': report
    }
    
    return metrics


def compute_ece(probs, labels, n_bins=15):
    """
    Expected Calibration Error (ECE)
    Measures average calibration error across confidence bins
    
    Args:
        probs: Predicted probabilities (N, num_classes)
        labels: True labels (N,)
        n_bins: Number of bins
        
    Returns:
        ece: Expected Calibration Error
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu()
        labels = labels.detach().cpu()
    
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # Check which samples fall in this bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()


def compute_mce(probs, labels, n_bins=15):
    """
    Maximum Calibration Error (MCE)
    Worst-case calibration error across bins
    
    Args:
        probs: Predicted probabilities (N, num_classes)
        labels: True labels (N,)
        n_bins: Number of bins
        
    Returns:
        mce: Maximum Calibration Error
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu()
        labels = labels.detach().cpu()
    
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    mce = 0.0
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            calibration_error = torch.abs(avg_confidence_in_bin - accuracy_in_bin).item()
            mce = max(mce, calibration_error)
    
    return mce


def compute_brier_score(probs, labels):
    """
    Brier Score - mean squared error of probabilistic predictions
    
    Args:
        probs: Predicted probabilities (N, num_classes)
        labels: True labels (N,)
        
    Returns:
        brier: Brier score
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # One-hot encode labels
    num_classes = probs.shape[1]
    labels_one_hot = np.eye(num_classes)[labels]
    
    # Compute Brier score
    brier = np.mean(np.sum((probs - labels_one_hot) ** 2, axis=1))
    
    return brier


def compute_nll(logits, labels):
    """
    Negative Log-Likelihood
    
    Args:
        logits: Model logits (N, num_classes)
        labels: True labels (N,)
        
    Returns:
        nll: Negative log-likelihood
    """
    if isinstance(logits, torch.Tensor):
        nll = F.cross_entropy(logits, labels).item()
    else:
        logits = torch.from_numpy(logits)
        labels = torch.from_numpy(labels)
        nll = F.cross_entropy(logits, labels).item()
    
    return nll


def get_calibration_bins(probs, labels, n_bins=15):
    """
    Get calibration data organized by bins for plotting
    
    Args:
        probs: Predicted probabilities (N, num_classes)
        labels: True labels (N,)
        n_bins: Number of bins
        
    Returns:
        bin_data: Dictionary with bin information
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu()
        labels = labels.detach().cpu()
    
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            count_in_bin = in_bin.sum().item()
            
            bin_confidences.append(avg_confidence_in_bin)
            bin_accuracies.append(accuracy_in_bin)
            bin_counts.append(count_in_bin)
        else:
            bin_confidences.append(None)
            bin_accuracies.append(None)
            bin_counts.append(0)
    
    return {
        'confidences': bin_confidences,
        'accuracies': bin_accuracies,
        'counts': bin_counts,
        'bin_boundaries': bin_boundaries.tolist()
    }


def compute_all_metrics(logits, labels, class_names=None, n_bins=15):
    """
    Compute all classification and calibration metrics
    
    Args:
        logits: Model logits
        labels: True labels
        class_names: Class names for report
        n_bins: Number of bins for calibration
        
    Returns:
        metrics: Dictionary with all metrics
    """
    # Get probabilities
    if isinstance(logits, torch.Tensor):
        probs = F.softmax(logits, dim=1)
    else:
        logits_torch = torch.from_numpy(logits)
        probs = F.softmax(logits_torch, dim=1)
    
    # Classification metrics
    clf_metrics = compute_classification_metrics(logits, labels, class_names)
    
    # Calibration metrics
    ece = compute_ece(probs, labels, n_bins)
    mce = compute_mce(probs, labels, n_bins)
    brier = compute_brier_score(probs, labels)
    nll = compute_nll(logits, labels)
    
    # Combine all metrics
    metrics = {
        **clf_metrics,
        'ece': ece,
        'mce': mce,
        'brier_score': brier,
        'nll': nll
    }
    
    return metrics