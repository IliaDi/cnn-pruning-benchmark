import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

try:
    from config import BATCH_SIZE as DEFAULT_BATCH_SIZE
except ImportError:
    DEFAULT_BATCH_SIZE = 64

def get_cifar10_loaders(batch_size=None, limit_batches=None, use_augmentation=True, num_workers=0):
    """
    CIFAR-10 data loaders with transforms for ImageNet pretrained VGG-16.
    
    Images are resized to 224x224 to match ImageNet VGG-16 input size.
    Uses ImageNet normalization statistics for pretrained model compatibility.
    
    Args:
        batch_size: Batch size for data loaders (defaults to config.BATCH_SIZE if None)
        limit_batches: Limit number of batches (for debugging)
        use_augmentation: If True, apply data augmentation (random crop, horizontal flip) to training set
        num_workers: Number of worker processes for data loading (0 = single-threaded, 4+ recommended for full experiments)
    """
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    # Test/validation transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize CIFAR-10 (32x32) to 224x224 for ImageNet VGG-16
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Training transform (with augmentation if enabled)
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to slightly larger than target
            transforms.RandomCrop(224),  # Random crop to 224x224
            transforms.RandomHorizontalFlip(),  # Horizontal flip
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        train_transform = test_transform

    train_set = datasets.CIFAR10(
        root="./data", train=True,
        download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        root="./data", train=False,
        download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader
