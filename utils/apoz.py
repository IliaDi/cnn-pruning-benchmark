"""
APoZ (Average Percentage of Zeros) Pruning
Based on: "Network Trimming: A Data-Driven Neuron Pruning Approach
           towards Efficient Deep Architectures" (Hu et al., 2016)

This module implements APoZ-based structural pruning that physically removes
filters/channels from the model architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class APoZCalculator:
    """
    Computes APoZ (Average Percentage of Zeros) for each channel/filter in the model.
    
    APoZ measures how often a neuron/channel outputs zero activations across the dataset.
    Higher APoZ indicates a less important channel that can be pruned.
    
    APoZ_c^(i) = Σ_k Σ_j 𝟙[O_c,j(k)==0] / (N × M)
    
    Where:
    - N = number of samples
    - M = spatial size (H×W for conv, 1 for linear)
    - O_c,j(k) = activation of channel c at spatial position j for sample k
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self._hooks = []
        self._stats = {}  # name → {"zeros": Tensor|None, "total": int}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on all Conv2d and Linear layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._stats[name] = {"zeros": None, "total": 0}
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)

    def _make_hook(self, name):
        """Create a forward hook that tracks zero activations."""
        def hook(module, inp, output):
            # Clamp to ensure non-negative (ReLU guard)
            act = torch.clamp(output.detach(), min=0.0)
            
            if act.dim() == 4:  # Conv2d: (B, C, H, W)
                B, C, H, W = act.shape
                # Count zeros per channel across batch and spatial dimensions
                zeros_ch = (act == 0).float().sum(dim=(0, 2, 3))  # (C,)
                total_ch = B * H * W
            else:  # Linear: (B, C)
                # Count zeros per channel across batch dimension
                zeros_ch = (act == 0).float().sum(dim=0)  # (C,)
                total_ch = act.size(0)

            # Accumulate statistics
            if self._stats[name]["zeros"] is None:
                self._stats[name]["zeros"] = zeros_ch.cpu()
            else:
                self._stats[name]["zeros"] += zeros_ch.cpu()
            self._stats[name]["total"] += total_ch
        
        return hook

    def reset(self):
        """Reset accumulated statistics."""
        for s in self._stats.values():
            s["zeros"] = None
            s["total"] = 0

    def compute(self, dataloader, limit_batches: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Compute APoZ scores for all layers by passing data through the model.
        
        Args:
            dataloader: DataLoader to compute APoZ over
            limit_batches: Optional limit on number of batches to process (for quick testing)
            
        Returns:
            Dictionary mapping layer names to APoZ scores (Tensor of shape [num_channels])
        """
        self.reset()
        self.model.eval()
        
        with torch.no_grad():
            for i, (imgs, _) in enumerate(dataloader):
                if limit_batches is not None and i >= limit_batches:
                    break
                self.model(imgs.to(self.device))

        # Compute APoZ = zeros / total
        apoz_dict = {}
        for name, s in self._stats.items():
            if s["zeros"] is not None and s["total"] > 0:
                apoz_dict[name] = s["zeros"] / s["total"]
        
        return apoz_dict

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def get_prune_mask(apoz: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Get pruning mask based on APoZ scores and threshold.
    
    Args:
        apoz: APoZ scores for channels (Tensor of shape [num_channels])
        threshold: Threshold value - channels with APoZ > threshold will be pruned
        
    Returns:
        Boolean tensor: True = keep, False = prune
    """
    mask = apoz <= threshold
    n_pruned = (~mask).sum().item()
    n_total = len(apoz)
    print(f"    threshold={threshold:.4f}  "
          f"pruned={n_pruned}/{n_total} ({100*n_pruned/n_total:.1f}%)")
    return mask


def compute_threshold(apoz: torch.Tensor, target_ratio: float, scope: str = "local") -> float:
    """
    Compute threshold to achieve target pruning ratio.
    
    Args:
        apoz: APoZ scores for channels (Tensor of shape [num_channels])
        target_ratio: Fraction of channels to REMOVE (0.3 = remove 30%, keep 70%)
        scope: "local" or "global" (for future use)
        
    Returns:
        Threshold value
    """
    if scope == "local":
        # For local pruning, use percentile-based threshold
        # target_ratio=0.3 means remove top 30% (highest APoZ = least important)
        percentile = (1.0 - target_ratio) * 100  # Keep bottom 70% = remove top 30%
        threshold = torch.quantile(apoz, percentile / 100.0).item()
    else:
        # Global scope would require aggregating across all layers
        # For now, same as local
        percentile = (1.0 - target_ratio) * 100
        threshold = torch.quantile(apoz, percentile / 100.0).item()
    
    return threshold


def get_prunable_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Get list of prunable layers (Conv2d and Linear) with their names.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of (name, module) tuples for prunable layers
    """
    prunable = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prunable.append((name, module))
    return prunable


def prune_conv2d_layer(module: nn.Conv2d, keep_channels: torch.Tensor, 
                      prev_keep_channels: Optional[torch.Tensor] = None) -> nn.Conv2d:
    """
    Structurally prune a Conv2d layer by removing output channels.
    Optionally also prune input channels if previous layer was pruned.
    
    Args:
        module: Conv2d module to prune
        keep_channels: Boolean tensor indicating which output channels to keep
        prev_keep_channels: Boolean tensor for previous layer's kept channels (for input pruning)
        
    Returns:
        New Conv2d module with pruned channels
    """
    keep_out_indices = torch.where(keep_channels)[0].tolist()
    num_out = len(keep_out_indices)
    
    # Determine input channels
    if prev_keep_channels is not None:
        keep_in_indices = torch.where(prev_keep_channels)[0].tolist()
        num_in = len(keep_in_indices)
    else:
        keep_in_indices = None
        num_in = module.in_channels
    
    # Create new Conv2d
    new_module = nn.Conv2d(
        in_channels=num_in,
        out_channels=num_out,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None
    )
    
    # Copy weights
    if keep_in_indices is not None:
        # Prune both input and output channels
        new_module.weight.data = module.weight.data[keep_out_indices][:, keep_in_indices].clone()
    else:
        # Only prune output channels
        new_module.weight.data = module.weight.data[keep_out_indices].clone()
    
    if module.bias is not None:
        new_module.bias.data = module.bias.data[keep_out_indices].clone()
    
    return new_module


def prune_linear_layer(module: nn.Linear, keep_channels: torch.Tensor,
                      prev_keep_channels: Optional[torch.Tensor] = None,
                      is_output: bool = False) -> nn.Linear:
    """
    Structurally prune a Linear layer.
    
    Args:
        module: Linear module to prune
        keep_channels: Boolean tensor indicating which output channels to keep
        prev_keep_channels: Boolean tensor for previous layer's kept channels (for input pruning)
        is_output: If True, this is the final output layer (don't prune inputs)
        
    Returns:
        New Linear module with pruned channels
    """
    keep_out_indices = torch.where(keep_channels)[0].tolist()
    num_out = len(keep_out_indices)
    
    # Determine input features
    if prev_keep_channels is not None and not is_output:
        # For hidden layers, use previous layer's kept channels
        # Note: For first Linear layer after features, prev_keep_channels represents
        # the flattened feature map channels
        keep_in_indices = torch.where(prev_keep_channels)[0].tolist()
        num_in = len(keep_in_indices)
    else:
        keep_in_indices = None
        num_in = module.in_features
    
    # Create new Linear layer
    new_module = nn.Linear(
        in_features=num_in,
        out_features=num_out,
        bias=module.bias is not None
    )
    
    # Copy weights
    if keep_in_indices is not None:
        # Prune both input and output features
        new_module.weight.data = module.weight.data[keep_out_indices][:, keep_in_indices].clone()
    else:
        # Only prune output features
        new_module.weight.data = module.weight.data[keep_out_indices].clone()
    
    if module.bias is not None:
        new_module.bias.data = module.bias.data[keep_out_indices].clone()
    
    return new_module


def apply_apoz_pruning(model: nn.Module, dataloader, target_ratio: float, 
                      scope: str = "local", device: Optional[torch.device] = None,
                      limit_batches: Optional[int] = None) -> nn.Module:
    """
    Apply APoZ-based structural pruning to the model.
    
    This function:
    1. Computes APoZ scores for all prunable layers
    2. Determines which channels to prune based on target_ratio
    3. Physically removes pruned channels from the model architecture
    
    Args:
        model: PyTorch model to prune
        dataloader: DataLoader for computing APoZ scores (typically validation/test set)
        target_ratio: Fraction of channels to REMOVE (0.3 = remove 30%, keep 70%)
        scope: "local" (per-layer) or "global" (across all layers) pruning
        device: Device to run computation on (if None, uses model's device)
        limit_batches: Optional limit on number of batches for APoZ computation (for quick testing)
        
    Returns:
        Structurally pruned model
    """
    if device is None:
        device = next(model.parameters()).device
    
    print(f"  Computing APoZ scores...")
    if limit_batches is not None:
        print(f"    (Limited to {limit_batches} batches for quick testing)")
    calculator = APoZCalculator(model, device)
    apoz_dict = calculator.compute(dataloader, limit_batches=limit_batches)
    calculator.remove_hooks()
    
    # Get prunable layers in forward order
    prunable_layers = get_prunable_layers(model)
    
    print(f"  Pruning layers (target ratio: {target_ratio:.1%} removal)...")
    
    # For each layer, compute threshold and create pruning mask
    masks = {}
    for name, module in prunable_layers:
        if name not in apoz_dict:
            continue
            
        apoz = apoz_dict[name]
        
        if scope == "local":
            # Local pruning: each layer prunes independently
            threshold = compute_threshold(apoz, target_ratio, scope="local")
            mask = get_prune_mask(apoz, threshold)
        else:
            # Global pruning: use global threshold across all layers
            # For now, same as local (can be enhanced later)
            threshold = compute_threshold(apoz, target_ratio, scope="local")
            mask = get_prune_mask(apoz, threshold)
        
        masks[name] = mask
    
    # Apply pruning masks to layers (in forward order)
    # Track kept channels from previous layer to update input channels
    prev_keep_channels = None
    
    for i, (name, module) in enumerate(prunable_layers):
        if name not in masks:
            continue
        
        mask = masks[name]
        keep_channels = mask
        
        # Skip if all channels are kept
        if keep_channels.all():
            prev_keep_channels = keep_channels
            continue
        
        # Parse layer name to navigate to module
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        layer_idx = int(parts[-1])
        
        # Prune the layer
        if isinstance(module, nn.Conv2d):
            # Check if we're in features or classifier
            is_in_features = parts[0] == "features"
            
            pruned_module = prune_conv2d_layer(module, keep_channels, prev_keep_channels)
            parent[layer_idx] = pruned_module
            
            prev_keep_channels = keep_channels
            
        elif isinstance(module, nn.Linear):
            # Check if this is the final output layer
            is_output = (i == len(prunable_layers) - 1)
            
            # For first Linear layer after Conv2d features, handle flattened feature map
            if i > 0 and isinstance(prunable_layers[i-1][1], nn.Conv2d):
                # Previous layer was Conv2d (last conv in features)
                # The Linear layer receives flattened feature map: (B, C, H, W) -> (B, C*H*W)
                prev_name, prev_module_original = prunable_layers[i-1]
                
                # Get the actual current module from model (may have been pruned)
                prev_parts = prev_name.split('.')
                prev_parent = model
                for part in prev_parts[:-1]:
                    prev_parent = getattr(prev_parent, part)
                prev_layer_idx = int(prev_parts[-1])
                prev_conv_current = prev_parent[prev_layer_idx]
                
                if prev_keep_channels is not None:
                    # Compute spatial size from original in_features
                    # Original: C * H * W = in_features
                    # After pruning: kept_C * H * W = new_in_features
                    num_kept_channels = prev_keep_channels.sum().item()
                    # Use original out_channels to compute spatial size (spatial dims don't change)
                    spatial_size = module.in_features // prev_module_original.out_channels  # H*W (e.g., 7*7=49)
                    new_in_features = num_kept_channels * spatial_size
                    
                    # Create expanded mask for flattened features
                    # For each kept channel, keep all spatial positions
                    keep_indices_flat = []
                    for c_idx in range(prev_module_original.out_channels):
                        if prev_keep_channels[c_idx]:
                            for s_idx in range(spatial_size):
                                keep_indices_flat.append(c_idx * spatial_size + s_idx)
                    keep_indices_flat = torch.tensor(keep_indices_flat, dtype=torch.long)
                    
                    # Create pruned Linear layer
                    keep_out_indices = torch.where(keep_channels)[0].tolist()
                    new_linear = nn.Linear(
                        in_features=new_in_features,
                        out_features=len(keep_out_indices),
                        bias=module.bias is not None
                    )
                    
                    # Copy weights: select output channels and input features
                    new_linear.weight.data = module.weight.data[keep_out_indices][:, keep_indices_flat].clone()
                    if module.bias is not None:
                        new_linear.bias.data = module.bias.data[keep_out_indices].clone()
                    
                    pruned_module = new_linear
                else:
                    # No previous pruning, just prune outputs
                    pruned_module = prune_linear_layer(module, keep_channels, None, is_output)
            else:
                # Regular Linear layer (not first after Conv2d)
                pruned_module = prune_linear_layer(module, keep_channels, prev_keep_channels, is_output)
            
            parent[layer_idx] = pruned_module
            
            if not is_output:
                prev_keep_channels = keep_channels
    
    print(f"  Pruning complete.")
    return model
