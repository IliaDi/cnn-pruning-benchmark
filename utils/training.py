import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

def train(model, dataloader, epochs=1, lr=0.001, fine_tune=True, verbose=False):
    """
    Train/fine-tune model on CIFAR-10.
    
    For fine-tuning ImageNet pretrained models:
    - Uses lower learning rate (0.001) for stable fine-tuning
    - Optionally uses different learning rates for features vs classifier
    - Trains for specified epochs until convergence
    
    Args:
        model: Model to train
        dataloader: Training data loader
        epochs: Number of epochs
        lr: Learning rate
        fine_tune: If True, use fine-tuning LR schedule (lower LR for features)
        verbose: If True, print per-epoch and per-batch progress
    """
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    
    if fine_tune:
        # Fine-tuning: lower LR for pretrained features, higher for new classifier
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

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs) if epochs > 1 else None
    
    model.train()
    total_batches = len(dataloader)
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # Print progress: every 10% of batches, or every 5 batches if total < 50, or at the last batch
            print_freq = max(1, min(5, total_batches // 10)) if total_batches < 50 else max(1, total_batches // 10)
            if verbose and ((batch_idx + 1) % print_freq == 0 or (batch_idx + 1) == total_batches):
                print(f"    [Baseline] epoch {epoch+1}/{epochs}  batch {batch_idx+1}/{total_batches}  loss={loss.item():.4f}")
        
        if scheduler is not None:
            scheduler.step()
        
        if verbose:
            avg_loss = running_loss / total_batches
            print(f"    [Baseline] epoch {epoch+1}/{epochs}  done  avg_loss={avg_loss:.4f}  batches={total_batches}")


def fine_tune_post_pruning(model, train_loader, test_loader, baseline_accuracy, epochs=30, lr=0.001, momentum=0.9, weight_decay=5e-4, verbose=False):
    """
    Standardized post-pruning fine-tuning protocol applied identically to all methods.
    
    Protocol:
    - Optimizer: SGD with momentum 0.9
    - Learning rate: 0.001 (one order of magnitude below initial training LR)
    - LR schedule: Cosine decay
    - Epochs: 30 (fixed for all methods)
    - Weight decay: Same as baseline training (5e-4)
    - Batch size: Same as baseline training (from train_loader)
    - Data augmentation: Same as baseline training (from train_loader)
    
    Args:
        model: Pruned model to fine-tune
        train_loader: Training data loader (with augmentation)
        test_loader: Test data loader for accuracy tracking
        baseline_accuracy: Baseline accuracy to compare against for recovery tracking
        epochs: Number of fine-tuning epochs (default: 30)
        lr: Learning rate (default: 0.001)
        momentum: SGD momentum (default: 0.9)
        weight_decay: Weight decay (default: 5e-4)
        verbose: If True, print per-epoch and per-batch progress
    
    Returns:
        model: Fine-tuned model
        fine_tune_metrics: Dict with fine-tuning cost metrics:
            - epochs_to_recover: Epochs needed to recover baseline accuracy (or None if not recovered)
            - accuracy_per_epoch: List of accuracies after each epoch
            - total_epochs: Total epochs used
    """
    from utils.metrics import evaluate_accuracy
    
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    
    # SGD optimizer with specified parameters
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Track accuracy during fine-tuning
    accuracy_per_epoch = []
    epochs_to_recover = None
    recovery_threshold = 0.001  # Consider "recovered" if within 0.1% of baseline
    
    model.train()
    total_batches = len(train_loader)
    for epoch in range(epochs):
        running_loss = 0.0
        # Training phase
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # Print progress: every 10% of batches, or every 5 batches if total < 50, or at the last batch
            print_freq = max(1, min(5, total_batches // 10)) if total_batches < 50 else max(1, total_batches // 10)
            if verbose and ((batch_idx + 1) % print_freq == 0 or (batch_idx + 1) == total_batches):
                print(f"    [Fine-tune] epoch {epoch+1}/{epochs}  batch {batch_idx+1}/{total_batches}  loss={loss.item():.4f}")
        
        # Step scheduler at end of each epoch
        scheduler.step()
        
        # Evaluate accuracy after each epoch
        model.eval()
        current_accuracy = evaluate_accuracy(model, test_loader)
        accuracy_per_epoch.append(current_accuracy)
        model.train()
        
        if verbose:
            avg_loss = running_loss / total_batches
            print(f"    [Fine-tune] epoch {epoch+1}/{epochs}  done  avg_loss={avg_loss:.4f}  acc={100*current_accuracy:.2f}%  (baseline {100*baseline_accuracy:.2f}%)")
        
        # Check if accuracy has recovered
        if epochs_to_recover is None and current_accuracy >= (baseline_accuracy - recovery_threshold):
            epochs_to_recover = epoch + 1
    
    # If not recovered within max epochs, return None
    if epochs_to_recover is None:
        epochs_to_recover = None
    
    fine_tune_metrics = {
        'epochs_to_recover': epochs_to_recover,
        'accuracy_per_epoch': accuracy_per_epoch,
        'total_epochs': epochs,
        'final_accuracy': accuracy_per_epoch[-1] if accuracy_per_epoch else None,
        'baseline_accuracy': baseline_accuracy
    }
    
    return model, fine_tune_metrics
