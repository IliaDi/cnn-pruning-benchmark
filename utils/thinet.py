"""
thinet.py – ThiNet pruning for VGG-16 / CIFAR-10.

Luo, Wu & Lin (2017), "ThiNet: A Filter Level Pruning Method for Deep
Neural Network Compression" (ICCV 2017).

This implementation:
- For each consecutive Conv2d pair (i, i+1), samples per-channel
  contributions and uses a greedy objective (ThiNet Eq. 6) plus a
  least-squares rescaling step (Eq. 7).
- Applies a uniform target_ratio across conv layers; falls back to
  weight magnitude for the last convs and hidden linear layers.
- Runs as a **global / sequential** pruning method and uses the shared
  structural helpers from `utils.apoz`. Fine-tuning is done externally.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.apoz import (
    get_prunable_layers,
    prune_conv2d_layer,
    prune_linear_layer,
    _update_output_layer,
    _get_module_at_path,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data collection: sample (x_hat, y_hat) pairs per layer
# ─────────────────────────────────────────────────────────────────────────────

def _get_conv_layers_with_next(model: nn.Module) -> List[Tuple[str, nn.Conv2d, str, nn.Conv2d]]:
    """
    Get pairs of (layer_i, layer_i+1) consecutive Conv2d layers from VGG-16.
    ThiNet prunes layer i based on statistics from layer i+1.
    Returns list of (name_i, conv_i, name_i+1, conv_i+1).
    """
    conv_layers: List[Tuple[str, nn.Conv2d, str, nn.Conv2d]] = []
    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        children = list(model.features.named_children())
        conv_indices = [(name, mod) for name, mod in children if isinstance(mod, nn.Conv2d)]

        for k in range(len(conv_indices) - 1):
            name_i, conv_i = conv_indices[k]
            name_next, conv_next = conv_indices[k + 1]
            conv_layers.append(
                (f"features.{name_i}", conv_i, f"features.{name_next}", conv_next)
            )
    return conv_layers


def _collect_channel_contributions(
    model: nn.Module,
    conv_name: str,
    conv_module: nn.Conv2d,
    next_conv_name: str,
    next_conv_module: nn.Conv2d,
    dataloader,
    device: torch.device,
    limit_batches: Optional[int] = None,
    n_samples_per_image: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect (x_hat, y_hat) training examples for ThiNet channel selection.

    For each calibration image:
      - Get the input to layer i+1 (= output of layer i after ReLU): shape (B, C, H, W)
      - Sample random spatial locations from the next conv's output field
      - For each location, compute per-channel contributions x_hat_c
        and the total y_hat

    Returns
    -------
    x_hat_all : (M, C) tensor of per-channel contributions
    y_hat_all : (M,) tensor of target values
    """
    model.eval()

    # Hook to capture input to next_conv (= output of previous block)
    input_to_next: List[torch.Tensor] = []

    def _hook_input(module, inp, out):
        input_to_next.append(inp[0].detach())

    handle = next_conv_module.register_forward_hook(_hook_input)

    x_hat_list: List[torch.Tensor] = []
    y_hat_list: List[torch.Tensor] = []

    try:
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dataloader):
                if limit_batches is not None and batch_idx >= limit_batches:
                    break
                images = images.to(device)
                _ = model(images)

                if not input_to_next:
                    continue

                # input_to_next[-1]: (B, C, H_in, W_in) — input channels to next conv
                inp_tensor = input_to_next[-1]
                input_to_next.clear()

                B, C, H_in, W_in = inp_tensor.shape
                W_next = next_conv_module.weight.data  # (D, C, kH, kW)
                D, _, kH, kW = W_next.shape
                pad_h, pad_w = next_conv_module.padding
                stride_h, stride_w = next_conv_module.stride

                # Compute output spatial size
                H_out = (H_in + 2 * pad_h - kH) // stride_h + 1
                W_out = (W_in + 2 * pad_w - kW) // stride_w + 1
                if H_out <= 0 or W_out <= 0:
                    continue

                # Pad input
                if pad_h > 0 or pad_w > 0:
                    inp_padded = F.pad(
                        inp_tensor, [pad_w, pad_w, pad_h, pad_h]
                    )
                else:
                    inp_padded = inp_tensor

                # Sample random locations
                n_samples = min(n_samples_per_image, H_out * W_out)
                for b in range(B):
                    positions = torch.randperm(H_out * W_out, device=device)[:n_samples]
                    h_positions = (positions // W_out).tolist()
                    w_positions = (positions % W_out).tolist()

                    for h, w in zip(h_positions, w_positions):
                        d = torch.randint(0, D, (1,), device=device).item()

                        h_start = h * stride_h
                        w_start = w * stride_w
                        x_patch = inp_padded[
                            b, :, h_start : h_start + kH, w_start : w_start + kW
                        ]  # (C, kH, kW)

                        w_filter = W_next[d]  # (C, kH, kW)
                        x_hat = (w_filter * x_patch).sum(dim=(1, 2))  # (C,)
                        y_hat = x_hat.sum()

                        x_hat_list.append(x_hat.cpu())
                        y_hat_list.append(y_hat.cpu())
    finally:
        handle.remove()

    if not x_hat_list:
        return (
            torch.zeros(1, conv_module.out_channels),
            torch.zeros(1),
        )

    return torch.stack(x_hat_list), torch.stack(y_hat_list)


# ─────────────────────────────────────────────────────────────────────────────
# Greedy channel selection (with incremental updates)
# ─────────────────────────────────────────────────────────────────────────────

def _greedy_channel_selection_fast(
    x_hat: torch.Tensor,
    n_channels_to_remove: int,
) -> List[int]:
    """
    Faster greedy channel selection using incremental objective updates.

    Objective (Eq. 6): min_T sum_i (sum_{j in T} x_hat_{i,j})^2
    """
    M, C = x_hat.shape
    n_channels_to_remove = min(n_channels_to_remove, max(C - 1, 1))

    T: List[int] = []
    remaining = list(range(C))
    current_sum = torch.zeros(M, device=x_hat.device)

    channel_sq_norms = (x_hat ** 2).sum(dim=0)  # (C,)

    for _ in range(n_channels_to_remove):
        if not remaining:
            break
        cross_terms = x_hat[:, remaining].t() @ current_sum  # (len(remaining),)
        marginal = 2 * cross_terms + channel_sq_norms[remaining]
        best_local = int(torch.argmin(marginal).item())
        best_idx = remaining[best_local]

        T.append(best_idx)
        current_sum = current_sum + x_hat[:, best_idx]
        remaining.remove(best_idx)

    return T


# ─────────────────────────────────────────────────────────────────────────────
# Least squares weight adjustment (paper Eq. 7)
# ─────────────────────────────────────────────────────────────────────────────

def _least_squares_rescale(
    next_conv: nn.Conv2d,
    keep_channels: List[int],
    x_hat: torch.Tensor,
    y_hat: torch.Tensor,
) -> None:
    """
    Apply least-squares scaling to the kept channels' filters to minimize
    reconstruction error (paper Eq. 7).
    """
    x_kept = x_hat[:, keep_channels]  # (M, C_kept)
    if x_kept.numel() == 0:
        return

    try:
        result = torch.linalg.lstsq(x_kept, y_hat.unsqueeze(1))
        w_hat = result.solution.squeeze()  # (C_kept,)
        with torch.no_grad():
            for local_idx, global_idx in enumerate(keep_channels):
                if local_idx < w_hat.numel():
                    next_conv.weight.data[:, global_idx, :, :] *= w_hat[
                        local_idx
                    ].item()
    except Exception:
        # If lstsq fails, skip rescaling
        return


# ─────────────────────────────────────────────────────────────────────────────
# Per-layer ThiNet channel selection
# ─────────────────────────────────────────────────────────────────────────────

def _thinet_select_channels(
    model: nn.Module,
    conv_name: str,
    conv_module: nn.Conv2d,
    next_conv_name: str,
    next_conv_module: nn.Conv2d,
    dataloader,
    device: torch.device,
    target_ratio: float,
    limit_batches: Optional[int] = None,
    n_samples_per_image: int = 10,
) -> torch.Tensor:
    """
    Run ThiNet channel selection for one layer pair.

    Returns a boolean keep-mask of shape (C,) for conv_module's output channels.
    """
    C = conv_module.out_channels
    n_remove = max(0, int(target_ratio * C))
    n_keep = max(1, C - n_remove)

    if n_remove == 0:
        return torch.ones(C, dtype=torch.bool)

    x_hat, y_hat = _collect_channel_contributions(
        model,
        conv_name,
        conv_module,
        next_conv_name,
        next_conv_module,
        dataloader,
        device,
        limit_batches,
        n_samples_per_image,
    )

    T = _greedy_channel_selection_fast(x_hat.to(device), n_remove)

    keep = torch.ones(C, dtype=torch.bool)
    for idx in T:
        keep[idx] = False

    if keep.sum() == 0:
        keep[0] = True

    keep_channels = torch.where(keep)[0].tolist()
    _least_squares_rescale(
        next_conv_module, keep_channels, x_hat.to(device), y_hat.to(device)
    )

    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Structural pruning (shared surgery from apoz.py)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_structural_pruning(
    model: nn.Module, masks: Dict[str, torch.Tensor]
) -> nn.Module:
    """Apply structural channel removal using shared helpers from apoz.py."""
    prunable = get_prunable_layers(model)
    prev_keep: Optional[torch.Tensor] = None

    for i, (name, module) in enumerate(prunable):
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        layer_idx = int(parts[-1])

        if isinstance(module, nn.Conv2d):
            keep_out = masks.get(
                name, torch.ones(module.out_channels, dtype=torch.bool)
            )
            parent[layer_idx] = prune_conv2d_layer(
                module, keep_out, keep_in=prev_keep
            )
            prev_keep = keep_out

        elif isinstance(module, nn.Linear):
            keep_out = masks.get(
                name, torch.ones(module.out_features, dtype=torch.bool)
            )
            prev_name, prev_orig = prunable[i - 1] if i > 0 else (None, None)
            follows_conv = prev_orig is not None and isinstance(prev_orig, nn.Conv2d)

            if follows_conv and prev_keep is not None:
                spatial = module.in_features // prev_orig.out_channels
                kept_ch = torch.where(prev_keep)[0].tolist()
                flat_idx = torch.tensor(
                    [c * spatial + s for c in kept_ch for s in range(spatial)],
                    dtype=torch.long,
                )
                out_idx = torch.where(keep_out)[0].tolist()
                new_module = nn.Linear(
                    len(kept_ch) * spatial,
                    len(out_idx),
                    bias=module.bias is not None,
                )
                new_module.weight.data = (
                    module.weight.data[out_idx][:, flat_idx].clone()
                )
                if module.bias is not None:
                    new_module.bias.data = module.bias.data[out_idx].clone()
            else:
                in_feat_live = None
                if prev_name is not None:
                    in_feat_live = _get_module_at_path(model, prev_name).out_features
                new_module = prune_linear_layer(
                    module,
                    keep_out,
                    keep_in=prev_keep,
                    in_features_override=in_feat_live,
                )

            parent[layer_idx] = new_module
            prev_keep = keep_out

    all_linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if all_linears:
        out_name = all_linears[-1]
        out_parts = out_name.split(".")
        out_par = model
        for p in out_parts[:-1]:
            out_par = getattr(out_par, p)
        out_idx_int = int(out_parts[-1])
        out_mod = out_par[out_idx_int]
        new_in = None
        if prunable:
            new_in = _get_module_at_path(model, prunable[-1][0]).out_features
        out_par[out_idx_int] = _update_output_layer(
            out_mod, prev_keep, in_features_override=new_in
        )

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def apply_thinet_pruning(
    model: nn.Module,
    dataloader,
    target_ratio: float,
    scope: str = "global",
    device: Optional[torch.device] = None,
    limit_batches: Optional[int] = None,
) -> nn.Module:
    """
    Apply ThiNet-based structural filter pruning.

    Parameters
    ----------
    model        : VGG-16 PyTorch model (torchvision-style).
    dataloader   : Calibration DataLoader (no augmentation).
    target_ratio : Fraction of filters to REMOVE (0.3 → remove 30%).
    scope        : Included for API symmetry; ThiNet is inherently global
                   and ignores this argument.
    device       : Compute device; inferred if None.
    limit_batches: Cap on calibration batches.
    """
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")

    if device is None:
        device = next(model.parameters()).device

    print(f"  ThiNet: starting channel selection (target={target_ratio:.0%} removed)...")

    conv_pairs = _get_conv_layers_with_next(model)
    masks: Dict[str, torch.Tensor] = {}
    processed_layers = set()

    # Process each conv pair
    for name_i, conv_i, name_next, conv_next in conv_pairs:
        C = conv_i.out_channels
        n_remove = max(0, int(target_ratio * C))
        n_keep = max(1, C - n_remove)

        if n_remove == 0:
            masks[name_i] = torch.ones(C, dtype=torch.bool)
            processed_layers.add(name_i)
            print(f"    [{name_i}] kept={C}/{C} (100.0%)  pruned=0/{C} (0.0%)")
            continue

        print(f"    [{name_i}] collecting samples via next layer '{name_next}'...")
        keep_mask = _thinet_select_channels(
            model,
            name_i,
            conv_i,
            name_next,
            conv_next,
            dataloader,
            device,
            target_ratio,
            limit_batches,
            n_samples_per_image=10,
        )
        masks[name_i] = keep_mask
        processed_layers.add(name_i)
        n_kept = int(keep_mask.sum().item())
        print(
            f"    [{name_i}] kept={n_kept}/{C} ({100*n_kept/C:.1f}%)  "
            f"pruned={C-n_kept}/{C} ({100*(C-n_kept)/C:.1f}%)"
        )

    # Last convs not in any pair: fallback to weight magnitude
    prunable_convs = [
        (n, m) for n, m in get_prunable_layers(model) if isinstance(m, nn.Conv2d)
    ]
    for name, module in prunable_convs:
        if name in processed_layers:
            continue
        C = module.out_channels
        n_remove = max(0, int(target_ratio * C))
        n_keep = max(1, C - n_remove)
        if n_remove == 0:
            masks[name] = torch.ones(C, dtype=torch.bool)
        else:
            importance = module.weight.data.abs().sum(dim=(1, 2, 3))
            _, top_idx = torch.topk(importance, n_keep)
            keep = torch.zeros(C, dtype=torch.bool)
            keep[top_idx] = True
            masks[name] = keep
        n_kept = int(masks[name].sum().item())
        print(
            f"    [{name}] (fallback) kept={n_kept}/{C} ({100*n_kept/C:.1f}%)  "
            f"pruned={C-n_kept}/{C} ({100*(C-n_kept)/C:.1f}%)"
        )

    # Hidden linears: simple magnitude-based pruning
    prunable_linears = [
        (n, m) for n, m in get_prunable_layers(model) if isinstance(m, nn.Linear)
    ]
    for name, module in prunable_linears:
        if name in masks:
            continue
        D = module.out_features
        n_remove = max(0, int(target_ratio * D))
        n_keep = max(1, D - n_remove)
        importance = module.weight.data.abs().sum(dim=1)
        _, top_idx = torch.topk(importance, n_keep)
        keep = torch.zeros(D, dtype=torch.bool)
        keep[top_idx] = True
        masks[name] = keep
        n_kept = int(keep.sum().item())
        print(
            f"    [{name}] (linear fallback) kept={n_kept}/{D} ({100*n_kept/D:.1f}%)  "
            f"pruned={D-n_kept}/{D} ({100*(D-n_kept)/D:.1f}%)"
        )

    print("  ThiNet: applying structural pruning...")
    model = _apply_structural_pruning(model, masks)
    print("  ThiNet pruning complete.")
    return model

