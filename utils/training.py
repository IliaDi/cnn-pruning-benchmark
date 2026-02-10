import torch
import torch.nn as nn
import torch.optim as optim

def train(model, dataloader, epochs=1, lr=0.01, limit_batches=None):
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9
    )

    model.train()
    for _ in range(epochs):
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
