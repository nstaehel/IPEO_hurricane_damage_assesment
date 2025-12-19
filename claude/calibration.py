"""
Calibration methods for neural network classifiers
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling calibration method
    Simple and effective post-hoc calibration that learns a single temperature parameter
    
    Reference: Guo et al. "On Calibration of Modern Neural Networks" (2017)
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        """
        Apply temperature scaling to logits
        
        Args:
            logits: Raw model outputs (before softmax)
            
        Returns:
            Scaled logits
        """
        return logits / self.temperature
    
    def fit(self, logits, labels, lr=0.01, max_iter=100):
        """
        Learn optimal temperature on validation set
        
        Args:
            logits: Validation set logits
            labels: Validation set labels
            lr: Learning rate
            max_iter: Maximum iterations
        """
        # Move to same device as logits
        self.temperature = nn.Parameter(torch.ones(1, device=logits.device) * 1.5)
        
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self


class PlattScaling:
    """
    Platt Scaling (logistic regression on model outputs)
    More flexible than temperature scaling for binary classification
    """
    def __init__(self):
        self.lr_model = LogisticRegression()
    
    def fit(self, logits, labels):
        """
        Fit logistic regression on validation logits
        
        Args:
            logits: Validation set logits (N, num_classes)
            labels: Validation set labels (N,)
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Use max logit or probability as feature
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        max_probs = probs.max(axis=1).reshape(-1, 1)
        
        self.lr_model.fit(max_probs, labels)
        return self
    
    def predict(self, logits):
        """
        Return calibrated probabilities
        
        Args:
            logits: Model logits
            
        Returns:
            Calibrated probabilities
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        max_probs = probs.max(axis=1).reshape(-1, 1)
        
        calibrated = self.lr_model.predict_proba(max_probs)
        return torch.from_numpy(calibrated).float()


class IsotonicCalibration:
    """
    Isotonic Regression for calibration
    Non-parametric method, most flexible but needs more data
    """
    def __init__(self):
        self.iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    
    def fit(self, logits, labels):
        """
        Fit isotonic regression on validation set
        
        Args:
            logits: Validation set logits
            labels: Validation set labels
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu()
            labels = labels.detach().cpu()
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get confidence (max probability)
        confidences = torch.max(probs, dim=1)[0].numpy()
        
        # Binary correctness (1 if prediction correct, 0 otherwise)
        predictions = torch.argmax(probs, dim=1)
        correct = (predictions == labels).float().numpy()
        
        # Fit isotonic regression
        self.iso_reg.fit(confidences, correct)
        return self
    
    def predict(self, logits):
        """
        Return calibrated probabilities
        
        Args:
            logits: Model logits
            
        Returns:
            Calibrated probabilities
        """
        if isinstance(logits, torch.Tensor):
            device = logits.device
            logits = logits.detach().cpu()
        
        probs = F.softmax(logits, dim=1).numpy()
        
        # Calibrate each probability
        calibrated_probs = np.zeros_like(probs)
        for i in range(probs.shape[0]):
            for j in range(probs.shape[1]):
                calibrated_probs[i, j] = self.iso_reg.predict([probs[i, j]])[0]
        
        # Normalize to sum to 1
        calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
        
        result = torch.from_numpy(calibrated_probs).float()
        if 'device' in locals():
            result = result.to(device)
        return result


def collect_logits_and_labels(model, dataloader, device):
    """
    Collect all logits and labels from a dataloader
    
    Args:
        model: PyTorch model or Lightning module
        dataloader: DataLoader to collect from
        device: Device to run on
        
    Returns:
        logits: All logits (N, num_classes)
        labels: All labels (N,)
    """
    # Check if Lightning module
    if hasattr(model, 'model'):
        model = model.model
    
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    return torch.cat(all_logits), torch.cat(all_labels)