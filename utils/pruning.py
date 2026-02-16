import torch


def apply_pruning_method(
    model,
    method,
    scope,
    target_ratio
):
    """
    Apply structural pruning to the model.
    
    CRITICAL: This must perform HARD structural pruning that physically removes
    filters and reshapes weight tensors. Soft masks (zeroing weights but keeping
    layer sizes) are NOT sufficient for accurate parameter/FLOPs reduction reporting.
    
    Args:
        model: PyTorch model to prune
        method: Pruning method name (e.g., "ApoZ", "ThiNet", etc.)
        scope: Pruning scope ("local" or "global")
        target_ratio: Fraction of filters/channels to REMOVE (0.3 = remove 30%, keep 70%)
    
    Returns:
        model: Structurally pruned model (architecture modified, not just weights masked)
    """
    # TODO: Replace with real structural pruning logic
    # For now, do nothing (identity) - this is a placeholder
    # 
    # IMPORTANT: When implementing, ensure:
    # 1. Filters/channels are physically removed (not just zeroed)
    # 2. Weight tensors are reshaped (e.g., Conv2d out_channels reduced)
    # 3. Subsequent layers' in_channels are adjusted accordingly
    # 4. Model architecture is permanently modified
    
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
