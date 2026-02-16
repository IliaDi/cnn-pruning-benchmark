import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size=128, limit_batches=None):
    """
    CIFAR-10 data loaders with transforms for ImageNet pretrained VGG-16.
    
    Images are resized to 224x224 to match ImageNet VGG-16 input size.
    Uses ImageNet normalization statistics for pretrained model compatibility.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize CIFAR-10 (32x32) to 224x224 for ImageNet VGG-16
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_set = datasets.CIFAR10(
        root="./data", train=True,
        download=True, transform=transform
    )
    test_set = datasets.CIFAR10(
        root="./data", train=False,
        download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=0
    )

    return train_loader, test_loader
