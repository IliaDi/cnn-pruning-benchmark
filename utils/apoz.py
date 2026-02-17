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
    n_kept = mask.sum().item()
    n_pruned = (~mask).sum().item()
    n_total = len(apoz)
    print(f"    threshold={threshold:.4f}  "
          f"kept={n_kept}/{n_total} ({100*n_kept/n_total:.1f}%)  "
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


def _get_module_at_path(model: nn.Module, name: str) -> nn.Module:
    """Return the module at the given path (e.g. 'classifier.0')."""
    m = model
    for p in name.split("."):
        m = getattr(m, p)
    return m


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
                      is_output: bool = False,
                      in_features_override: Optional[int] = None) -> nn.Linear:
    """
    Structurally prune a Linear layer.

    For non-output layers: prunes both input and output dimensions.
    For the output layer: keeps all output neurons (classes), but STILL updates
    in_features to match whatever the previous layer now emits, so shapes remain
    consistent after structural pruning of preceding layers.

    Args:
        module: Linear module to prune
        keep_channels: Boolean tensor indicating which output channels to keep.
                       For the output layer this is ignored (all outputs kept).
        prev_keep_channels: Boolean tensor for previous layer's kept channels
                            (used for input-dimension weight indexing).
        is_output: If True this is the final classification layer; do NOT prune
                   output neurons, but DO update in_features.
        in_features_override: If set, use this value as the new in_features
                              (e.g. from previous layer's actual out_features in
                              the live model after earlier replacements).

    Returns:
        New Linear module with correctly sized weight tensors.
    """
    if is_output:
        # ── Output layer ────────────────────────────────────────────────────
        # Keep ALL output neurons (classes), update in_features only.
        num_out = module.out_features          # e.g. 10 for CIFAR-10

        # Determine new in_features and which original input columns to keep
        if in_features_override is not None:
            num_in = in_features_override
            keep_in_indices = (
                torch.where(prev_keep_channels)[0].tolist()
                if prev_keep_channels is not None
                else None
            )
        elif prev_keep_channels is not None:
            keep_in_indices = torch.where(prev_keep_channels)[0].tolist()
            num_in = len(keep_in_indices)
        else:
            keep_in_indices = None
            num_in = module.in_features

        new_module = nn.Linear(num_in, num_out, bias=module.bias is not None)

        if keep_in_indices is not None:
            new_module.weight.data = module.weight.data[:, keep_in_indices].clone()
        else:
            new_module.weight.data = module.weight.data.clone()

        if module.bias is not None:
            new_module.bias.data = module.bias.data.clone()

        return new_module

    # ── Hidden layer ─────────────────────────────────────────────────────────
    keep_out_indices = torch.where(keep_channels)[0].tolist()
    num_out = len(keep_out_indices)

    # Determine input features: prefer override (from actual previous layer in model), else from prev mask
    if in_features_override is not None:
        num_in = in_features_override
        keep_in_indices = (
            torch.where(prev_keep_channels)[0].tolist()
            if prev_keep_channels is not None
            else None
        )
    elif prev_keep_channels is not None:
        keep_in_indices = torch.where(prev_keep_channels)[0].tolist()
        num_in = len(keep_in_indices)
    else:
        keep_in_indices = None
        num_in = module.in_features

    new_module = nn.Linear(num_in, num_out, bias=module.bias is not None)

    if keep_in_indices is not None:
        new_module.weight.data = module.weight.data[keep_out_indices][:, keep_in_indices].clone()
    else:
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
    
    # ── Apply pruning masks in forward order ─────────────────────────────────
    # prev_keep_channels: boolean mask of the *output* channels kept by the
    # most-recently-processed layer; used to update the *input* side of the
    # next layer so weight tensors stay consistent after structural removal.
    prev_keep_channels = None   # None → previous layer was not structurally changed

    n_layers = len(prunable_layers)
    for i, (name, module) in enumerate(prunable_layers):
        is_output_layer = (i == n_layers - 1)  # never prune outputs of the final classifier

        # ── Navigate to parent container ─────────────────────────────────────
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        layer_idx = int(parts[-1])

        # ── Conv2d ───────────────────────────────────────────────────────────
        if isinstance(module, nn.Conv2d):
            if name in masks:
                keep_channels = masks[name]
                pruned_module = prune_conv2d_layer(module, keep_channels, prev_keep_channels)
                parent[layer_idx] = pruned_module
                prev_keep_channels = keep_channels
            else:
                # Not pruned, but record full mask so next layer can update inputs
                prev_keep_channels = torch.ones(module.out_channels, dtype=torch.bool)

        # ── Linear ───────────────────────────────────────────────────────────
        elif isinstance(module, nn.Linear):
            # Decide whether we need to prune outputs of this layer.
            # The output (classification) layer NEVER has its output neurons pruned;
            # it only needs its input dimension updated if a preceding layer changed.
            if is_output_layer:
                # Build a keep-all mask over outputs so prune_linear_layer knows
                # we want every output neuron retained.
                keep_channels = torch.ones(module.out_features, dtype=torch.bool)
            elif name in masks:
                keep_channels = masks[name]
            else:
                keep_channels = torch.ones(module.out_features, dtype=torch.bool)

            # ── Determine new in_features ────────────────────────────────────
            # Case A: immediately follows a Conv2d — must account for H×W flattening.
            if i > 0 and isinstance(prunable_layers[i - 1][1], nn.Conv2d):
                prev_conv_name, prev_conv_orig = prunable_layers[i - 1]
                orig_out_ch = prev_conv_orig.out_channels   # channels *before* pruning
                spatial_size = module.in_features // orig_out_ch  # H×W (e.g. 49 for 7×7)

                if prev_keep_channels is not None:
                    # Build the index list that maps kept (channel, spatial) pairs
                    # in the original flattened feature vector to contiguous positions.
                    keep_ch_indices = torch.where(prev_keep_channels)[0].tolist()
                    keep_indices_flat = torch.tensor(
                        [c * spatial_size + s
                         for c in keep_ch_indices
                         for s in range(spatial_size)],
                        dtype=torch.long
                    )
                    new_in_features = len(keep_ch_indices) * spatial_size
                else:
                    keep_indices_flat = None
                    new_in_features = module.in_features

                if is_output_layer:
                    # Only update in_features; keep all output neurons.
                    new_module = nn.Linear(new_in_features, module.out_features,
                                           bias=module.bias is not None)
                    if keep_indices_flat is not None:
                        new_module.weight.data = module.weight.data[:, keep_indices_flat].clone()
                    else:
                        new_module.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        new_module.bias.data = module.bias.data.clone()
                    pruned_module = new_module
                else:
                    keep_out_indices = torch.where(keep_channels)[0].tolist()
                    new_module = nn.Linear(new_in_features, len(keep_out_indices),
                                           bias=module.bias is not None)
                    if keep_indices_flat is not None:
                        new_module.weight.data = module.weight.data[keep_out_indices][:, keep_indices_flat].clone()
                    else:
                        new_module.weight.data = module.weight.data[keep_out_indices].clone()
                    if module.bias is not None:
                        new_module.bias.data = module.bias.data[keep_out_indices].clone()
                    pruned_module = new_module

            # Case B: follows another Linear layer.
            else:
                # Fetch the *current* (already-replaced) previous Linear in the model
                # to get its true out_features after earlier structural pruning.
                prev_name = prunable_layers[i - 1][0]
                prev_layer_live = _get_module_at_path(model, prev_name)
                in_features_from_prev = prev_layer_live.out_features

                pruned_module = prune_linear_layer(
                    module, keep_channels, prev_keep_channels,
                    is_output=is_output_layer,
                    in_features_override=in_features_from_prev
                )

            parent[layer_idx] = pruned_module

            # Update prev_keep_channels for the NEXT layer's input dimension.
            # For the output layer there is no next layer, so this is moot;
            # for hidden layers we pass the output mask forward.
            if not is_output_layer:
                prev_keep_channels = keep_channels

    print(f"  Pruning complete.")
    return model
