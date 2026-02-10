import torch
from thop import profile

def evaluate_accuracy(model, dataloader, limit_batches=None):
    model.eval()
    correct = 0
    total = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if limit_batches is not None and i >= limit_batches:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return correct / total if total > 0 else 0.0


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def compute_flops(model, input_size=(1, 3, 32, 32)):
    dummy = torch.randn(input_size).to(
        next(model.parameters()).device
    )
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    return flops
