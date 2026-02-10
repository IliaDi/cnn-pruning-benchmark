import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size=128, limit_batches=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
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
