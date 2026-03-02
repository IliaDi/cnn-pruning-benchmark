import torch

try:
    from config import BATCH_SIZE
except ImportError:
    BATCH_SIZE = 64

def apply_pruning_method(
    model,
    method,
    scope,
    target_ratio,
    limit_batches=None,
    calib_loader=None
):
    """
    Apply structural pruning to the model.
    
    This must perform HARD structural pruning that physically removes
    filters and reshapes weight tensors. Soft masks (zeroing weights but keeping
    layer sizes) are NOT sufficient for accurate parameter/FLOPs reduction reporting.
    
    Args:
        model: PyTorch model to prune
        method: Pruning method name (e.g., "ApoZ", "ThiNet", etc.)
        scope: Pruning scope ("local" or "global")
        target_ratio: Fraction of filters/channels to REMOVE (0.3 = remove 30%, keep 70%)
        limit_batches: Optional limit on batches for calibration data (e.g. when using a separate quick-test script)
        calib_loader: Optional calibration dataloader. If None, creates a new test loader.
                     Should use test data with no augmentation for consistent APoZ scores.
    
    Returns:
        model: Structurally pruned model (architecture modified, not just weights masked)
    """
    # Use provided calibration loader, or create one if not provided
    if calib_loader is None:
        from utils.data import get_cifar10_loaders
        # Get a calibration dataloader for pruning computation
        # Use test loader (no augmentation) for consistent scores
        _, calib_loader = get_cifar10_loaders(batch_size=BATCH_SIZE, use_augmentation=False)
    
    if method == "ApoZ":
        from utils.apoz import apply_apoz_pruning
        
        # Apply APoZ pruning
        model = apply_apoz_pruning(
            model=model,
            dataloader=calib_loader,
            target_ratio=target_ratio,
            scope=scope,
            limit_batches=limit_batches
        )
        
        return model
    elif method == "DropNet":
        from utils.dropnet import apply_dropnet_pruning
        
        # Apply DropNet pruning
        device = next(model.parameters()).device
        model = apply_dropnet_pruning(
            model=model,
            dataloader=calib_loader,
            target_ratio=target_ratio,
            scope=scope,
            device=device,
            limit_batches=limit_batches
        )
        
        return model
    elif method == "Entropy":
        from utils.entropy import apply_entropy_pruning
        
        # Apply Entropy pruning
        device = next(model.parameters()).device
        model = apply_entropy_pruning(
            model=model,
            dataloader=calib_loader,
            target_ratio=target_ratio,
            scope=scope,
            device=device,
            limit_batches=limit_batches
        )
        
        return model
    elif method == "HRank":
        from utils.hrank import apply_hrank_pruning
        
        # Apply HRank pruning
        device = next(model.parameters()).device
        model = apply_hrank_pruning(
            model=model,
            dataloader=calib_loader,
            target_ratio=target_ratio,
            scope=scope,
            device=device,
            limit_batches=limit_batches
        )
        
        return model
    elif method == "CHIP":
        from utils.chip import apply_chip_pruning
        
        # Apply CHIP pruning
        device = next(model.parameters()).device
        model = apply_chip_pruning(
            model=model,
            dataloader=calib_loader,
            target_ratio=target_ratio,
            scope=scope,
            device=device,
            limit_batches=limit_batches
        )
        
        return model
    elif method == "LRMF":
        from utils.lrmf import apply_lrmf_pruning
        
        # Apply LRMF pruning
        device = next(model.parameters()).device
        model = apply_lrmf_pruning(
            model=model,
            dataloader=calib_loader,
            target_ratio=target_ratio,
            scope=scope,
            device=device,
            limit_batches=limit_batches
        )
        
        return model
    elif method == "NISP":
        from utils.nisp import apply_nisp_pruning
        
        # Apply NISP pruning (global by default)
        device = next(model.parameters()).device
        model = apply_nisp_pruning(
            model=model,
            dataloader=calib_loader,
            target_ratio=target_ratio,
            scope=scope,
            device=device,
            limit_batches=limit_batches
        )
        
        return model
    elif method == "ThiNet":
        from utils.thinet import apply_thinet_pruning
        
        # Apply ThiNet pruning (global / sequential)
        device = next(model.parameters()).device
        model = apply_thinet_pruning(
            model=model,
            dataloader=calib_loader,
            target_ratio=target_ratio,
            scope=scope,
            device=device,
            limit_batches=limit_batches
        )
        
        return model
    else:
        # TODO: Implement other pruning methods
        # For now, return model unchanged
        print(f"  Warning: Method '{method}' not yet implemented, skipping pruning.")
        return model


def get_layer_info(model):
    """
    Get layer-wise information about the model (filter counts, etc.).
    
    Useful for logging actual per-layer pruning ratios and diagnosing
    pruning behavior across different layers.
    
    Args:
        model: PyTorch model
    
    Returns:
        dict: Layer information including filter/channel counts per layer
    """
    layer_info = {}
    
    # Extract layer information from VGG-16 structure
    if hasattr(model, 'features'):
        # VGG-16 features (conv layers)
        for i, layer in enumerate(model.features):
            if isinstance(layer, torch.nn.Conv2d):
                layer_name = f"features.{i}"
                layer_info[layer_name] = {
                    "type": "Conv2d",
                    "in_channels": layer.in_channels,
                    "out_channels": layer.out_channels,
                    "kernel_size": layer.kernel_size,
                    "parameters": sum(p.numel() for p in layer.parameters())
                }
    
    if hasattr(model, 'classifier'):
        # VGG-16 classifier (FC layers)
        for i, layer in enumerate(model.classifier):
            if isinstance(layer, torch.nn.Linear):
                layer_name = f"classifier.{i}"
                layer_info[layer_name] = {
                    "type": "Linear",
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                    "parameters": sum(p.numel() for p in layer.parameters())
                }
    
    return layer_info
