"""
Data loading utilities for Hurricane Damage Detection
"""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class GeoEye1(Dataset):
    """
    Dataset class for GeoEye-1 Hurricane Harvey satellite imagery
    
    Args:
        root_dir: Root directory containing train/validation/test folders
        split: One of 'train', 'validation', 'test'
        transforms: torchvision transforms to apply
    """
    
    # Mapping between label class names and indices
    LABEL_CLASSES = {
        'no_damage': 0,
        'damage': 1,
    }
    
    CLASS_NAMES = ['no_damage', 'damage']

    def __init__(self, root_dir="ipeo_hurricane_for_students", split='train', transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms

        # Path to split folder (train/test/validation)
        split_dir = os.path.join(root_dir, split)

        # List containing (image_path, label)
        self.data = []

        # For each class (damage / no_damage)
        for class_name, class_idx in self.LABEL_CLASSES.items():
            class_dir = os.path.join(split_dir, class_name)

            if not os.path.isdir(class_dir):
                continue

            if split == 'train':
            # Get all images from folder while avoiding duplicates in training set
                for img_name in list(set(os.listdir(class_dir))):
                    if img_name.lower().endswith((".jpeg", ".jpg", ".png")):
                        img_path = os.path.join(class_dir, img_name)
                        self.data.append((img_path, class_idx))
            else:
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith((".jpeg", ".jpg", ".png")):
                        img_path = os.path.join(class_dir, img_name)
                        self.data.append((img_path, class_idx))

        # Sort for reproducibility
        self.data.sort()
        
        print(f"Loaded {len(self.data)} images for {split} split")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)

        return img, label
    
    def get_class_distribution(self):
        """Returns class distribution as dict"""
        class_counts = {name: 0 for name in self.CLASS_NAMES}
        for _, label in self.data:
            class_counts[self.CLASS_NAMES[label]] += 1
        return class_counts


def compute_dataset_statistics(root_dir, split='train', image_size=150, batch_size=64, num_workers=4):
    """
    Compute mean and std of dataset for normalization
    
    Args:
        root_dir: Root directory of dataset
        split: Which split to compute stats on
        image_size: Size to resize images to
        batch_size: Batch size for computation
        num_workers: Number of workers for DataLoader
        
    Returns:
        mean, std: Tensors of shape (3,) for RGB channels
    """
    transforms_for_stats = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])

    dataset = GeoEye1(root_dir=root_dir, split=split, transforms=transforms_for_stats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"\nComputing dataset statistics on {split} split...")
    mean = 0.
    std = 0.
    total_images = 0

    for imgs, _ in loader:
        batch_size = imgs.size(0)
        imgs = imgs.view(batch_size, imgs.size(1), -1)
        mean += imgs.mean(2).sum(0)
        std += imgs.std(2).sum(0)
        total_images += batch_size

    mean /= total_images
    std /= total_images

    print(f"Mean: {mean}")
    print(f"Std:  {std}")
    
    return mean, std


def get_transforms(mean, std, image_size=150, augment=True):
    """
    Get train and validation transforms
    
    Args:
        mean: Mean for normalization
        std: Std for normalization
        image_size: Target image size
        augment: Whether to apply augmentations (for training)
        
    Returns:
        transforms: torchvision.transforms.Compose object
    """
    normalize = T.Normalize(mean, std)
    
    if augment:
        # Training transforms with augmentation
        transforms = T.Compose([
            T.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            normalize
        ])
    else:
        # Validation/test transforms without augmentation
        transforms = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            normalize
        ])
    
    return transforms


def get_dataloaders(root_dir, mean, std, image_size=150, batch_size=32, num_workers=4):
    """
    Create train, validation, and test dataloaders
    
    Args:
        root_dir: Root directory of dataset
        mean: Mean for normalization
        std: Std for normalization
        image_size: Target image size
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        train_loader, val_loader, test_loader
    """
    transforms_train = get_transforms(mean, std, image_size, augment=True)
    transforms_val = get_transforms(mean, std, image_size, augment=False)
    
    train_dataset = GeoEye1(root_dir, "train", transforms_train)
    val_dataset = GeoEye1(root_dir, "validation", transforms_val)
    test_dataset = GeoEye1(root_dir, "test", transforms_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader