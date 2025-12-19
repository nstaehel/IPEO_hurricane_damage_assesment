import torch
import torch.nn.functional as F
import numpy as np
from sklearn.isotonic import IsotonicRegression

class IsotonicCalibration:
    """
    Isotonic Regression pour calibrer les probabilités du modèle
    """
    def __init__(self):
        self.iso_reg = IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip')
    
    def fit(self, logits, labels):
        """
        Entraîne le modèle de calibration sur le validation set
        """
        # Convertir logits en probabilités
        probs = F.softmax(logits, dim=1)
        
        # Prendre la probabilité de la classe prédite
        confidences = torch.max(probs, dim=1)[0].numpy()
        
        # Créer des targets binaires (1 si prédiction correcte, 0 sinon)
        predictions = torch.argmax(probs, dim=1)
        correct = (predictions == labels).float().numpy()
        
        # Fit isotonic regression
        self.iso_reg.fit(confidences, correct)
        return self
    
    def predict(self, logits):
        """
        Retourne les probabilités calibrées
        """
        probs = F.softmax(logits, dim=1).numpy()
        
        # Calibrer chaque probabilité
        calibrated_probs = np.zeros_like(probs)
        for i in range(probs.shape[0]):
            for j in range(probs.shape[1]):
                calibrated_probs[i, j] = self.iso_reg.predict([probs[i, j]])[0]
        
        # Normaliser pour que la somme = 1
        calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
        
        return torch.from_numpy(calibrated_probs).float()

def compute_ece(probs, labels, n_bins=15):
    """
    Expected Calibration Error
    Mesure la différence entre confiance et précision
    """
    confidences, predictions = torch.max(probs, 1)
    accuracies = predictions.eq(labels)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()

def collect_logits_and_labels(model, dataloader, device):
    """
    Collecte tous les logits et labels du validation set
    """
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