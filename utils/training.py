import torch
import torch.nn as nn
import torch.optim as optim

def train(model, dataloader, epochs=1, lr=0.001, limit_batches=None, fine_tune=True):
    """
    Train/fine-tune model on CIFAR-10.
    
    For fine-tuning ImageNet pretrained models:
    - Uses lower learning rate (0.001) for stable fine-tuning
    - Optionally uses different learning rates for features vs classifier
    - Trains for specified epochs until convergence
    """
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    
    if fine_tune:
        # Fine-tuning: lower LR for pretrained features, higher for new classifier
        # This is a common practice when fine-tuning pretrained models
        # torchvision VGG-16 has 'features' (conv layers) and 'classifier' (FC layers)
        if hasattr(model, 'features') and hasattr(model, 'classifier'):
            feature_params = list(model.features.parameters())
            classifier_params = list(model.classifier.parameters())
            
            optimizer = optim.SGD([
                {'params': feature_params, 'lr': lr * 0.1},  # Lower LR for pretrained features
                {'params': classifier_params, 'lr': lr}      # Higher LR for new classifier
            ], momentum=0.9, weight_decay=5e-4)
        else:
            # Fallback: use same LR for all parameters
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=5e-4
            )
    else:
        # Standard training (all parameters same LR)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4
        )

    model.train()
    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(dataloader):
            if limit_batches is not None and i >= limit_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
