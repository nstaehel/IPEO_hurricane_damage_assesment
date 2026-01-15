from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import seaborn as sns

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibration:
    """
    Isotonic Regression pour calibrer les probabilités du modèle
    Pour classification binaire
    """
    def __init__(self):
        self.iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')

    def fit(self, logits, labels):
        """
        Entraîne le modèle de calibration sur le validation set
        """
        # Convertir logits en probabilités
        probs = F.softmax(logits, dim=1)

        # Pour classification binaire, on utilise la probabilité de la classe 1
        prob_class1 = probs[:, 1].numpy()
        labels_np = labels.numpy()

        # Fit isotonic regression: prob(class=1) -> actual class
        self.iso_reg.fit(prob_class1, labels_np)
        return self

    def predict(self, logits):
        """
        Retourne les probabilités calibrées
        """
        probs = F.softmax(logits, dim=1)
        prob_class1 = probs[:, 1].numpy()

        # Calibrer la probabilité de la classe 1
        calibrated_prob_class1 = self.iso_reg.predict(prob_class1)

        # Reconstruire les probabilités pour les deux classes
        calibrated_probs = np.stack([
            1 - calibrated_prob_class1,  # prob class 0
            calibrated_prob_class1       # prob class 1
        ], axis=1)

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


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This is the gap between confidence and accuracy).
    """
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class ModelCalibrator:
    def __init__(self, model):
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def set_temperature(self, valid_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Tunes the temperature of the model (using the validation set) with L-BFGS.
        """
        self.model.to(device)
        self.model.eval()

        # Recreate temperature parameter on the correct device
        self.temperature = nn.Parameter(torch.ones(1, device=device) * 1.5)

        logits_list = []
        labels_list = []

        # 1. Collect all logits and labels from validation set
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(device)
            labels = torch.cat(labels_list).to(device)

        # 2. Optimize the temperature T
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = ECELoss().to(device)

        # Calculate ECE before calibration
        before_temperature_ece = ece_criterion(logits, labels).item()
        print(f'Before temperature - ECE: {before_temperature_ece:.4f}')

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # Calculate ECE after calibration
        after_temperature_ece = ece_criterion(logits / self.temperature, labels).item()
        print(f'Optimal temperature: {self.temperature.item():.3f}')
        print(f'After temperature - ECE: {after_temperature_ece:.4f}')

        return self

    def calibrate_probs(self, logits):
        """Applies the learned temperature to new logits."""
        return F.softmax(logits / self.temperature, dim=1)
    
def plot_reliability_diagram(probs_list, labels, names, colors, n_bins=10, title="", save_path=None):
    """
    Plot multiple reliability diagrams on the same figure

    Args:
        probs_list: list of probability tensors
        labels: ground truth labels
        names: list of names for each curve
        colors: list of colors for each curve
        n_bins: number of bins
        title: plot title
        save_path: path to save the figure (optional)
    """
    plt.figure(figsize=(10, 8))

    for probs, name, color in zip(probs_list, names, colors):
        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(labels)

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        accuracies_in_bins = []
        confidences_in_bins = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracies_in_bins.append(accuracies[in_bin].float().mean().item())
                confidences_in_bins.append(confidences[in_bin].mean().item())

        plt.plot(confidences_in_bins, accuracies_in_bins, 's-', color=color, label=name, markersize=8)

    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

def run_discard_test(model, data_loader, device=None, num_fractions=10):
    """
    Performs the Discard Test to evaluate uncertainty estimates.

    Args:
        model: Trained PyTorch model.
        data_loader: DataLoader for the test set.
        device: 'cuda' or 'cpu'.
        num_fractions: Number of discard fractions (default 10).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    all_losses = []
    all_uncertainties = []
    all_correct = []

    # --- Step 1: Collect Predictions, Uncertainties, and Losses ---
    print("Collecting model predictions on test set...")
    with torch.no_grad():
        for batch in data_loader:
            # Handle cases where batch might be (images, labels) or a dict
            if isinstance(batch, (list, tuple)):
                images, labels = batch
            else: # Assuming dictionary if using custom dataset wrappers
                images = batch['image']
                labels = batch['label']

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)

            # 1. Calculate Error (Loss) per sample
            # reduction='none' gives us the loss for each individual image
            loss = F.cross_entropy(logits, labels, reduction='none')
            all_losses.append(loss.cpu().numpy())

            # 2. Calculate Uncertainty (Entropy) per sample
            probs = F.softmax(logits, dim=1)
            # Entropy = - sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            all_uncertainties.append(entropy.cpu().numpy())

            # 3. Determine Correct/Incorrect Classification
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels)
            all_correct.append(correct.cpu().numpy())

    # Concatenate all batches
    losses = np.concatenate(all_losses)
    uncertainties = np.concatenate(all_uncertainties)
    correct_mask = np.concatenate(all_correct)

    n_samples = len(losses)
    print(f"Total samples evaluated: {n_samples}")

    # --- Step 2: Discard Loop ---
    # Sort indices by uncertainty (Descending: Most uncertain first)
    sorted_indices = np.argsort(uncertainties)[::-1]

    sorted_losses = losses[sorted_indices]

    discard_fractions = np.linspace(0, 1, num_fractions + 1)[:-1] # 0.0 to 0.9
    errors = []

    for fraction in discard_fractions:
        # Calculate how many samples to discard
        n_discard = int(n_samples * fraction)

        # Keep the remaining samples (least uncertain)
        # Since we sorted descending, the "remaining" are at the end of the array
        if n_discard == 0:
            current_losses = sorted_losses
        else:
            current_losses = sorted_losses[n_discard:]

        # Error metric is the mean loss of remaining samples
        mean_loss = np.mean(current_losses)
        errors.append(mean_loss)

    # --- Step 3: Compute Metrics (MF and DI) ---
    # Convert errors to array for easy indexing
    epsilon = np.array(errors)
    N_f = len(discard_fractions)

    # Monotonicity Fraction (MF)
    mf_count = np.sum(epsilon[:-1] >= epsilon[1:])
    mf = mf_count / (N_f - 1)

    # Discard Improvement (DI)
    di_sum = np.sum(epsilon[:-1] - epsilon[1:])
    di = di_sum / (N_f - 1)

    print(f"\n--- Discard Test Results ---")
    print(f"Monotonicity Fraction (MF): {mf:.4f} (Target: 1.0)")
    print(f"Discard Improvement (DI):   {di:.4f}")

    # --- Step 4: Visualization ---
    plt.figure(figsize=(14, 6))

    # Plot 1: Discard Curve
    plt.subplot(1, 2, 1)
    plt.plot(discard_fractions, errors, marker='o', linestyle='-', linewidth=2)
    plt.title(f"Discard Test (Loss)\nMF: {mf:.2f} | DI: {di:.4f}")
    plt.xlabel("Discard Fraction")
    plt.ylabel("Model Error (Loss)")
    plt.grid(True, alpha=0.3)

    # Plot 2: Uncertainty Density Plots
    plt.subplot(1, 2, 2)

    # Separate uncertainties
    unc_correct = uncertainties[correct_mask]
    unc_incorrect = uncertainties[~correct_mask]

    # Calculate Medians
    med_correct = np.median(unc_correct) if len(unc_correct) > 0 else 0
    med_incorrect = np.median(unc_incorrect) if len(unc_incorrect) > 0 else 0

    # Plot Densities using Seaborn for KDE + Histogram
    sns.kdeplot(unc_correct, fill=True, label=f"Correct (Med: {med_correct:.2f})", color='green', alpha=0.3)
    sns.kdeplot(unc_incorrect, fill=True, label=f"Incorrect (Med: {med_incorrect:.2f})", color='red', alpha=0.3)

    # Add vertical lines for medians
    plt.axvline(med_correct, color='green', linestyle='--')
    plt.axvline(med_incorrect, color='red', linestyle='--')

    plt.title("Uncertainty Density (Entropy)")
    plt.xlabel("Uncertainty Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


import torchvision.transforms.functional as TF
from scipy import stats

def create_perturbed_loader(original_loader, perturbation_type='blur', **kwargs):
    """
    Create a new dataloader with perturbed images.
    """
    class PerturbedDataset(Dataset):
        def __init__(self, original_dataset, perturbation_type, **kwargs):
            self.dataset = original_dataset
            self.perturbation_type = perturbation_type
            self.kwargs = kwargs

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, label = self.dataset[idx]

            if self.perturbation_type == 'blur':
                kernel_size = self.kwargs.get('kernel_size', 3)
                sigma = self.kwargs.get('sigma', 1.0)
                image = TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

            elif self.perturbation_type == 'rotate':
                angle = self.kwargs.get('angle', 90)
                image = TF.rotate(image, angle)

            elif self.perturbation_type == 'noise':
                noise_std = self.kwargs.get('noise_std', 0.02)
                noise = torch.randn_like(image) * noise_std
                image = torch.clamp(image + noise, 0, 1)

            return image, label

    perturbed_dataset = PerturbedDataset(
        original_loader.dataset,
        perturbation_type,
        **kwargs
    )

    return DataLoader(
        perturbed_dataset,
        batch_size=original_loader.batch_size,
        shuffle=False,
        num_workers=0
    )


def compute_confidence_drift(probs_test, probs_retest):
    """
    Compute confidence drift metrics between test and retest.
    """
    conf_test, pred_test = torch.max(probs_test, 1)
    conf_retest, pred_retest = torch.max(probs_retest, 1)

    conf_diff = (conf_retest - conf_test).detach().numpy()
    pred_consistency = (pred_test == pred_retest).float().mean().item()

    return {
        'mean_conf_diff': np.mean(conf_diff),
        'std_conf_diff': np.std(conf_diff),
        'max_conf_diff': np.max(np.abs(conf_diff)),
        'pred_consistency': pred_consistency,
        'conf_test_mean': conf_test.mean().item(),
        'conf_retest_mean': conf_retest.mean().item(),
    }


def run_test_retest_analysis(model, test_loader, calibrator_iso, calibrator_temp, device):
    """
    Run complete test-retest confidence drift analysis.
    """
    perturbations = [
        ('blur_light', 'blur', {'kernel_size': 3, 'sigma': 0.3}),
        ('blur_medium', 'blur', {'kernel_size': 5, 'sigma': 0.7}),
        ('rotate_90', 'rotate', {'angle': 90}),
        ('rotate_180', 'rotate', {'angle': 180}),
        ('rotate_270', 'rotate', {'angle': 270}),
        ('noise_light', 'noise', {'noise_std': 0.005}),
        ('noise_medium', 'noise', {'noise_std': 0.01}),
    ]

    results = []
    rotate90_data = None  # Store data for reliability diagram

    # Collect original test logits
    print("Collecting original test predictions...")
    test_logits, test_labels = collect_logits_and_labels(model, test_loader, device)

    # Original probabilities
    probs_uncal = F.softmax(test_logits, dim=1)
    probs_iso = calibrator_iso.predict(test_logits)
    probs_temp = calibrator_temp.calibrate_probs(test_logits.to(device)).cpu()

    # ECE for original
    ece_uncal_orig = compute_ece(probs_uncal, test_labels)
    ece_iso_orig = compute_ece(probs_iso, test_labels)
    ece_temp_orig = compute_ece(probs_temp, test_labels)

    print(f"\nOriginal ECE - Uncalibrated: {ece_uncal_orig:.4f}, Isotonic: {ece_iso_orig:.4f}, Temperature: {ece_temp_orig:.4f}")

    for name, pert_type, params in perturbations:
        print(f"\nProcessing perturbation: {name}...")

        perturbed_loader = create_perturbed_loader(test_loader, pert_type, **params)
        retest_logits, retest_labels = collect_logits_and_labels(model, perturbed_loader, device)

        probs_uncal_retest = F.softmax(retest_logits, dim=1)
        probs_iso_retest = calibrator_iso.predict(retest_logits)
        probs_temp_retest = calibrator_temp.calibrate_probs(retest_logits.to(device)).cpu()

        ece_uncal_retest = compute_ece(probs_uncal_retest, retest_labels)
        ece_iso_retest = compute_ece(probs_iso_retest, retest_labels)
        ece_temp_retest = compute_ece(probs_temp_retest, retest_labels)

        drift_uncal = compute_confidence_drift(probs_uncal, probs_uncal_retest)
        drift_iso = compute_confidence_drift(probs_iso, probs_iso_retest)
        drift_temp = compute_confidence_drift(probs_temp, probs_temp_retest)

        # Store rotate_90 data for reliability diagram
        if name == 'rotate_90':
            rotate90_data = {
                'probs_uncal': probs_uncal_retest,
                'probs_iso': probs_iso_retest,
                'probs_temp': probs_temp_retest,
                'labels': retest_labels,
                'ece_uncal': ece_uncal_retest,
                'ece_iso': ece_iso_retest,
                'ece_temp': ece_temp_retest,
            }

        results.append({
            'perturbation': name,
            'uncal': {
                'ece_orig': ece_uncal_orig,
                'ece_retest': ece_uncal_retest,
                'ece_diff': ece_uncal_retest - ece_uncal_orig,
                **drift_uncal
            },
            'isotonic': {
                'ece_orig': ece_iso_orig,
                'ece_retest': ece_iso_retest,
                'ece_diff': ece_iso_retest - ece_iso_orig,
                **drift_iso
            },
            'temperature': {
                'ece_orig': ece_temp_orig,
                'ece_retest': ece_temp_retest,
                'ece_diff': ece_temp_retest - ece_temp_orig,
                **drift_temp
            }
        })

    return results, rotate90_data


def plot_test_retest_results(results):
    """
    Visualize test-retest confidence drift results (3 plots).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    perturbations = [r['perturbation'] for r in results]
    x = np.arange(len(perturbations))
    width = 0.25

    # Plot 1: ECE Difference
    ax1 = axes[0]
    ece_diff_uncal = [r['uncal']['ece_diff'] for r in results]
    ece_diff_iso = [r['isotonic']['ece_diff'] for r in results]
    ece_diff_temp = [r['temperature']['ece_diff'] for r in results]

    ax1.bar(x - width, ece_diff_uncal, width, label='Uncalibrated', color='red', alpha=0.7)
    ax1.bar(x, ece_diff_iso, width, label='Isotonic', color='blue', alpha=0.7)
    ax1.bar(x + width, ece_diff_temp, width, label='Temperature', color='green', alpha=0.7)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Perturbation')
    ax1.set_ylabel('ECE Difference (Retest - Test)')
    ax1.set_title('ECE Change After Perturbation')
    ax1.set_xticks(x)
    ax1.set_xticklabels(perturbations, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mean Confidence Drift
    ax2 = axes[1]
    conf_diff_uncal = [r['uncal']['mean_conf_diff'] for r in results]
    conf_diff_iso = [r['isotonic']['mean_conf_diff'] for r in results]
    conf_diff_temp = [r['temperature']['mean_conf_diff'] for r in results]

    ax2.bar(x - width, conf_diff_uncal, width, label='Uncalibrated', color='red', alpha=0.7)
    ax2.bar(x, conf_diff_iso, width, label='Isotonic', color='blue', alpha=0.7)
    ax2.bar(x + width, conf_diff_temp, width, label='Temperature', color='green', alpha=0.7)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Perturbation')
    ax2.set_ylabel('Mean Confidence Difference')
    ax2.set_title('Mean Confidence Drift')
    ax2.set_xticks(x)
    ax2.set_xticklabels(perturbations, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Prediction Consistency
    ax3 = axes[2]
    pred_cons_uncal = [r['uncal']['pred_consistency'] for r in results]
    pred_cons_iso = [r['isotonic']['pred_consistency'] for r in results]
    pred_cons_temp = [r['temperature']['pred_consistency'] for r in results]

    ax3.bar(x - width, pred_cons_uncal, width, label='Uncalibrated', color='red', alpha=0.7)
    ax3.bar(x, pred_cons_iso, width, label='Isotonic', color='blue', alpha=0.7)
    ax3.bar(x + width, pred_cons_temp, width, label='Temperature', color='green', alpha=0.7)

    ax3.set_xlabel('Perturbation')
    ax3.set_ylabel('Prediction Consistency')
    ax3.set_title('Prediction Stability (Higher = Better)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(perturbations, rotation=45, ha='right')
    ax3.set_ylim(0, 1.05)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/test_retest_drift.png", dpi=300, bbox_inches='tight')
    print("Figure saved to figures/test_retest_drift.png")
    plt.show()


def print_summary_table(results):
    """Print a summary table of results."""
    print("\n" + "=" * 90)
    print("TEST-RETEST CONFIDENCE DRIFT SUMMARY")
    print("=" * 90)

    print(f"\n{'Perturbation':<15} | {'Method':<12} | {'ECE Orig':<10} | {'ECE Retest':<10} | {'ΔECE':<10} | {'Pred Cons':<10}")
    print("-" * 90)

    for r in results:
        for method, label in [('uncal', 'Uncalibrated'), ('isotonic', 'Isotonic'), ('temperature', 'Temperature')]:
            data = r[method]
            print(f"{r['perturbation']:<15} | {label:<12} | {data['ece_orig']:<10.4f} | {data['ece_retest']:<10.4f} | {data['ece_diff']:>+10.4f} | {data['pred_consistency']:<10.4f}")
        print("-" * 90)



def run_discard_test_all_calib(model, data_loader, device=None, num_fractions=10, calibrator_iso=None, calibrator_temp=None):
    """
    Performs the Discard Test to evaluate uncertainty estimates.

    Args:
        model: Trained PyTorch model.
        data_loader: DataLoader for the test set.
        device: 'cuda' or 'cpu'.
        num_fractions: Number of discard fractions (default 10).
        calibrator_iso: IsotonicCalibration object (optional).
        calibrator_temp: ModelCalibrator object (optional).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    all_logits = []
    all_labels = []
    all_correct = []

    # --- Step 1: Collect Logits and Labels ---
    print("Collecting model predictions on test set...")
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                images, labels = batch
            else:
                images = batch['image']
                labels = batch['label']

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels)
            all_correct.append(correct.cpu().numpy())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    correct_mask = np.concatenate(all_correct)

    n_samples = len(labels)
    print(f"Total samples evaluated: {n_samples}")

    # --- Step 2: Prepare calibration methods ---
    methods = {
        'Uncalibrated': F.softmax(logits, dim=1)
    }
    
    if calibrator_iso is not None:
        methods['Isotonic'] = calibrator_iso.predict(logits)
    
    if calibrator_temp is not None:
        methods['Temperature'] = calibrator_temp.calibrate_probs(logits.to(device)).cpu()

    # --- Step 3: Discard Test for each method ---
    discard_fractions = np.linspace(0, 1, num_fractions + 1)[:-1]
    all_errors = {}
    all_metrics = {}

    for method_name, probs in methods.items():
        # Calculate loss using negative log-likelihood
        losses = -torch.log(probs.gather(1, labels.unsqueeze(1)).squeeze() + 1e-10).detach().numpy()
        
        # Calculate uncertainty (entropy)
        uncertainties = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).detach().numpy()
        
        # Sort by uncertainty (descending)
        sorted_indices = np.argsort(uncertainties)[::-1]
        sorted_losses = losses[sorted_indices]

        errors = []
        for fraction in discard_fractions:
            n_discard = int(n_samples * fraction)
            if n_discard == 0:
                current_losses = sorted_losses
            else:
                current_losses = sorted_losses[n_discard:]
            errors.append(np.mean(current_losses))

        all_errors[method_name] = errors

        # Compute metrics
        epsilon = np.array(errors)
        N_f = len(discard_fractions)
        mf = np.sum(epsilon[:-1] >= epsilon[1:]) / (N_f - 1)
        di = np.sum(epsilon[:-1] - epsilon[1:]) / (N_f - 1)
        
        all_metrics[method_name] = {'mf': mf, 'di': di, 'uncertainties': uncertainties}

    # --- Step 4: Print Results ---
    print(f"\n{'='*60}")
    print("DISCARD TEST RESULTS")
    print(f"{'='*60}")
    print(f"\n{'Method':<15} | {'MF (↑1.0)':<12} | {'DI (↑)':<12}")
    print("-" * 45)
    for method_name in methods.keys():
        m = all_metrics[method_name]
        print(f"{method_name:<15} | {m['mf']:<12.4f} | {m['di']:<12.4f}")

    # --- Step 5: Visualization ---
    colors = {'Uncalibrated': 'red', 'Isotonic': 'blue', 'Temperature': 'green'}

    # Figure 1: Discard Test Curves
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    for method_name, errors in all_errors.items():
        m = all_metrics[method_name]
        color = colors.get(method_name, 'gray')
        ax1.plot(discard_fractions, errors, marker='o', linestyle='-', linewidth=2,
                 color=color, label=f"{method_name} (MF:{m['mf']:.2f}, DI:{m['di']:.4f})")

    ax1.set_title("Discard Test - All Calibration Methods", fontsize=14)
    ax1.set_xlabel("Discard Fraction", fontsize=12)
    ax1.set_ylabel("Model Error (NLL Loss)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/discard_test_comparison.png", dpi=300, bbox_inches='tight')
    print("\nFigure saved to figures/discard_test_comparison.png")
    plt.show()

    # Figure 2: Uncertainty Density for each method
    n_methods = len(methods)
    fig2, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    
    # Handle case where there's only one method
    if n_methods == 1:
        axes = [axes]

    for idx, (method_name, _) in enumerate(methods.items()):
        ax = axes[idx]
        unc = all_metrics[method_name]['uncertainties']
        unc_correct = unc[correct_mask]
        unc_incorrect = unc[~correct_mask]

        med_correct = np.median(unc_correct) if len(unc_correct) > 0 else 0
        med_incorrect = np.median(unc_incorrect) if len(unc_incorrect) > 0 else 0

        color = colors.get(method_name, 'gray')
        
        sns.kdeplot(unc_correct, fill=True, label=f"Correct (Med: {med_correct:.2f})", 
                    color='green', alpha=0.3, ax=ax)
        sns.kdeplot(unc_incorrect, fill=True, label=f"Incorrect (Med: {med_incorrect:.2f})", 
                    color='red', alpha=0.3, ax=ax)

        ax.axvline(med_correct, color='green', linestyle='--', linewidth=1.5)
        ax.axvline(med_incorrect, color='red', linestyle='--', linewidth=1.5)
        ax.set_title(f"Uncertainty Density (Entropy)\n{method_name}", fontsize=12)
        ax.set_xlabel("Uncertainty Score", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/uncertainty_density_comparison.png", dpi=300, bbox_inches='tight')
    print("Figure saved to figures/uncertainty_density_comparison.png")
    plt.show()

    return all_metrics, all_errors
