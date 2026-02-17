import torch
import time
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


def compute_flops(model, input_size=(1, 3, 224, 224)):
    dummy = torch.randn(input_size).to(
        next(model.parameters()).device
    )
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    return flops


def measure_inference_latency(model, batch_size=1, num_warmup=10, num_iterations=100, input_size=(3, 224, 224)):
    """
    Measure inference latency (wall-clock time) on a fixed GPU.
    
    FLOPs ≠ actual speed due to hardware efficiency, memory access patterns, etc.
    This metric distinguishes methods that have similar FLOPs but different actual speeds.
    
    Args:
        model: Model to measure
        batch_size: Batch size for inference (default: 1 for single-image latency)
        num_warmup: Number of warmup iterations to stabilize GPU
        num_iterations: Number of iterations to average over
        input_size: Input tensor size (C, H, W)
    
    Returns:
        Average inference latency in milliseconds per sample
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Ensure device is a torch.device object for .type access
    if isinstance(device, str):
        device = torch.device(device)
    device_type = str(device.type) if hasattr(device, 'type') else str(device)
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    # Synchronize GPU before timing
    if device_type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure inference time
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if device_type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(dummy_input)
            
            if device_type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    # Average latency per sample
    avg_latency_ms = sum(times) / len(times) / batch_size
    
    return avg_latency_ms
