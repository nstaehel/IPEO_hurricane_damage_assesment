"""
ResNet model implementations for hurricane damage detection
"""
import torch
import torch.nn as nn
import torchvision.models as models


def get_resnet18(num_classes=2, pretrained=True, freeze_backbone=False):
    """
    Get ResNet-18 model
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze all layers except final FC
        
    Returns:
        model: ResNet-18 model
    """
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


def get_resnet50(num_classes=2, pretrained=True, freeze_backbone=False):
    """
    Get ResNet-50 model
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze all layers except final FC
        
    Returns:
        model: ResNet-50 model
    """
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = models.resnet50(weights=None)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


def count_parameters(model):
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total