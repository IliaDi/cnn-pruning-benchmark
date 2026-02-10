import torch.nn as nn
import torchvision.models as models


def vgg16(num_classes=10):
    """
    Standard VGG16 architecture adapted for CIFAR-10.
    Trained from scratch (no ImageNet pretraining).
    """
    model = models.vgg16(weights=None)

    # Replace classifier for CIFAR-10
    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features,
        num_classes
    )

    return model
