import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

try:
    from config import BATCH_SIZE as DEFAULT_BATCH_SIZE
except ImportError:
    DEFAULT_BATCH_SIZE = 64

# Stratified held-out calibration: 1000 images per class (10k total); remainder for training (4k per class = 40k).
CALIB_PER_CLASS = 1000
TRAIN_PER_CLASS = 5000 - CALIB_PER_CLASS  # CIFAR-10 has 5000 train images per class
SPLIT_SEED = 42


def get_cifar10_loaders(
    batch_size=None,
    limit_batches=None,
    use_augmentation=True,
    num_workers=0,
):
    """
    CIFAR-10 data loaders with transforms for ImageNet pretrained VGG-16.

    Images are resized to 224x224 to match ImageNet VGG-16 input size.
    Uses ImageNet normalization statistics for pretrained model compatibility.

    Calibration is a held-out subset of the original training set: 1000 images
    per class (10k total), stratified. The remaining 4000 per class (40k) are
    used for training. Calibration uses test-style transforms (no augmentation).
    Test set (10k) is unchanged and used only for final evaluation.

    Args:
        batch_size: Batch size for data loaders (defaults to config.BATCH_SIZE if None)
        limit_batches: Limit number of batches (for debugging; not applied to calibration)
        use_augmentation: If True, apply data augmentation to training set
        num_workers: Number of worker processes for data loading
    """
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    # Test/validation transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        train_transform = test_transform

    # Full training data (50k): two copies for different transforms
    train_full_aug = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    train_full_no_aug = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=test_transform
    )
    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    # Stratified split: 1000 per class → calibration; 4000 per class → training
    targets = train_full_aug.targets
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    train_indices = []
    calib_indices = []
    for c in range(10):
        class_idx = [i for i, t in enumerate(targets) if t == c]
        perm = torch.randperm(len(class_idx), generator=generator).tolist()
        class_idx = [class_idx[i] for i in perm]
        train_indices.extend(class_idx[:TRAIN_PER_CLASS])
        calib_indices.extend(class_idx[TRAIN_PER_CLASS : TRAIN_PER_CLASS + CALIB_PER_CLASS])

    train_set = Subset(train_full_aug, train_indices)
    calib_set = Subset(train_full_no_aug, calib_indices)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    calib_loader = DataLoader(
        calib_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, calib_loader
