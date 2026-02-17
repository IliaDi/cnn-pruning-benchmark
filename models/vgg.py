import torch.nn as nn
import torchvision.models as models


def vgg16(num_classes=10, pretrained=True):
    """
    VGG-16 with ImageNet pretrained weights, adapted for CIFAR-10.
    
    Approach:
    1. Load ImageNet pretrained VGG-16 weights
    2. Replace classifier head for 10 classes (CIFAR-10)
    3. Fine-tune on CIFAR-10 to convergence
    
    Input images are resized to 224x224 to match ImageNet VGG-16 architecture.
    """
    # Load ImageNet pretrained VGG-16
    if pretrained:
        model = models.vgg16(weights='IMAGENET1K_V1')
    else:
        model = models.vgg16(weights=None)
    
    # Replace classifier head for CIFAR-10 (10 classes)
    # Original classifier: 25088 -> 4096 -> 4096 -> 1000
    # New classifier: 25088 -> 4096 -> 4096 -> 10
    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features,
        num_classes
    )
    
    return model
