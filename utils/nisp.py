"""
nisp.py – NISP (Neuron Importance Score Propagation) pruning.

Yu et al. (2018), "NISP: Pruning Networks using Neuron Importance Score
Propagation".

This implementation:
- Identifies the **Final Response Layer (FRL)**, ranks its neurons by a
  simple variance × mean-|activation| criterion, then propagates
  importance backward through Linear/Conv2d layers.
- Uses the propagated scores to build global/local keep-masks for conv
  and hidden linear layers.
- Applies **structural** pruning using helpers from `utils.apoz`; all
  fine-tuning is done externally by the shared training code.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from utils.apoz import (
    get_prunable_layers,
    apply_structural_pruning,
    _get_module_at_path,
)


# ─────────────────────────────────────────────────────────────────────────────
# FRL identification and activation collection
# ─────────────────────────────────────────────────────────────────────────────

def _identify_frl(model: nn.Module) -> Tuple[str, nn.Module]:
    """
    Identify the Final Response Layer (FRL) — the second-to-last Linear
    layer before the classification output.
    """
    all_linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if len(all_linears) < 2:
        raise ValueError("NISP requires at least 2 Linear layers (FRL + output)")
    return all_linears[-2]


def _collect_frl_activations(
    model: nn.Module,
    frl_name: str,
    frl_module: nn.Module,
    dataloader,
    device: torch.device,
    limit_batches: Optional[int] = None,
) -> torch.Tensor:
    """
    Collect post-ReLU activations from the FRL across calibration data.
    Returns (N, D) tensor where N = total samples, D = FRL output features.
    """
    model.eval()
    activations = []

    # Find the ReLU immediately following the FRL Linear
    hook_target = None
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        frl_idx = int(frl_name.split(".")[-1])
        for j in range(frl_idx + 1, min(frl_idx + 3, len(model.classifier))):
            if isinstance(model.classifier[j], nn.ReLU):
                hook_target = model.classifier[j]
                break
    if hook_target is None:
        hook_target = frl_module

    captured = []

    def _hook_fn(module, inp, out):
        captured.append(out.detach().cpu())

    handle = hook_target.register_forward_hook(_hook_fn)
    try:
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dataloader):
                if limit_batches is not None and batch_idx >= limit_batches:
                    break
                images = images.to(device)
                _ = model(images)
                if not captured:
                    continue
                activations.append(captured[-1])
                captured.clear()
    finally:
        handle.remove()

    if not activations:
        raise RuntimeError("NISP: no FRL activations were captured.")
    return torch.cat(activations, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Feature ranking on FRL (importance scores for FRL neurons)
# ─────────────────────────────────────────────────────────────────────────────

def _rank_frl_neurons(activations: torch.Tensor) -> torch.Tensor:
    """
    Compute importance scores for each neuron in the FRL.

    Uses variance × mean-absolute-activation as a practical surrogate
    for feature selection. Variance captures discriminative spread;
    mean-abs captures overall activation strength.

    Parameters
    ----------
    activations : (N, D) Tensor of FRL activations.

    Returns
    -------
    importance : (D,) Tensor, non-negative importance scores.
    """
    var = activations.var(dim=0)
    mean_abs = activations.abs().mean(dim=0)
    return (var * mean_abs).clamp(min=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Importance score propagation
# ─────────────────────────────────────────────────────────────────────────────

def _propagate_importance(
    model: nn.Module,
    frl_importance: torch.Tensor,
    frl_name: str,
) -> Dict[str, torch.Tensor]:
    """
    Propagate importance scores from the FRL backward through the network.

    For Conv2d: W (out_ch, in_ch, kH, kW) → sum abs over spatial dims →
    effective (out_ch, in_ch) matrix, then s_k = W^T @ s_{k+1}.

    For Linear: W (out_features, in_features) → s_k = |W|^T @ s_{k+1}.

    When propagating from Linear to preceding Conv2d, the Linear's
    in_features = C × H × W (flattened). We reshape and sum over spatial
    positions to get per-channel importance.
    """
    all_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            all_layers.append((name, module))

    frl_idx = None
    for i, (n, _) in enumerate(all_layers):
        if n == frl_name:
            frl_idx = i
            break
    if frl_idx is None:
        raise ValueError(f"NISP: FRL '{frl_name}' not found in model layers")

    importance: Dict[str, torch.Tensor] = {frl_name: frl_importance.clone()}
    current_importance = frl_importance.clone()

    for idx in range(frl_idx, 0, -1):
        layer_name, layer_module = all_layers[idx]
        prev_name, prev_module = all_layers[idx - 1]
        W_device = layer_module.weight.data.device

        if isinstance(layer_module, nn.Linear):
            W = layer_module.weight.data.abs()  # (out, in)
            prev_importance = W.t() @ current_importance.to(W_device)
            if isinstance(prev_module, nn.Conv2d):
                C_out = prev_module.out_channels
                spatial = layer_module.in_features // C_out
                if spatial > 0:
                    prev_importance = prev_importance.view(C_out, spatial).sum(dim=1)

        elif isinstance(layer_module, nn.Conv2d):
            W = layer_module.weight.data.abs().sum(dim=(2, 3))  # (out, in)
            prev_importance = W.t() @ current_importance.to(W_device)
        else:
            continue

        prev_importance = prev_importance.cpu().clamp(min=0.0)
        importance[prev_name] = prev_importance
        current_importance = prev_importance

    return importance


# ─────────────────────────────────────────────────────────────────────────────
# Mask building (global only — NISP is a global-ranking method)
# ─────────────────────────────────────────────────────────────────────────────

def _build_masks_global(
    scores: Dict[str, torch.Tensor],
    pruning_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    Keep-masks using global ranking across all Conv2d layers.

    Per-layer min-max normalization is applied to correct for the
    systematic scale differences introduced by backward importance
    propagation through weight matrices (early layers accumulate higher
    absolute scores than late layers).  Unlike rank normalization, min-max
    preserves each layer's distribution shape, so layers with many
    low-importance filters still contribute disproportionately to the
    prune pool — maintaining genuine global budget allocation.
    """
    normalized: Dict[str, torch.Tensor] = {}
    for name, s in scores.items():
        s_min = s.min()
        s_max = s.max()
        if (s_max - s_min) > 1e-12:
            normalized[name] = (s - s_min) / (s_max - s_min)
        else:
            normalized[name] = torch.ones_like(s)

    layer_names = list(normalized.keys())
    all_scores = torch.cat([normalized[n] for n in layer_names])
    n_total = len(all_scores)
    n_prune = max(1, int(pruning_ratio * n_total))

    prune_set = set(
        torch.argsort(all_scores, descending=False)[:n_prune].tolist()
    )

    masks: Dict[str, torch.Tensor] = {}
    offset = 0
    for name in layer_names:
        s = normalized[name]
        n_out = len(s)
        keep = torch.tensor(
            [(offset + i) not in prune_set for i in range(n_out)],
            dtype=torch.bool,
        )
        if keep.sum() == 0:
            keep[int(torch.argmax(s).item())] = True
        masks[name] = keep
        n_kept = keep.sum().item()
        print(
            f"    [{name}] kept={n_kept}/{n_out} ({100*n_kept/n_out:.1f}%)  "
            f"pruned={n_out-n_kept}/{n_out} ({100*(n_out-n_kept)/n_out:.1f}%)"
        )
        offset += n_out

    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def apply_nisp_pruning(
    model: nn.Module,
    dataloader,
    target_ratio: float,
    scope: str = "global",
    device: Optional[torch.device] = None,
    limit_batches: Optional[int] = None,
) -> nn.Module:
    """
    Apply NISP-based structural filter pruning.

    Pipeline (Yu et al., 2018):
    1. Identify the Final Response Layer (FRL).
    2. Collect FRL activations from calibration data.
    3. Rank FRL neurons by importance (variance-based feature ranking).
    4. Propagate importance backward.
    5. Build keep-masks from propagated importance scores.
    6. Structurally remove pruned channels and update classifier input.
    """
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")

    if device is None:
        device = next(model.parameters()).device

    frl_name, frl_module = _identify_frl(model)
    print(f"  NISP: FRL identified as '{frl_name}'")

    print("  NISP: Collecting FRL activations...")
    frl_activations = _collect_frl_activations(
        model,
        frl_name,
        frl_module,
        dataloader,
        device,
        limit_batches=limit_batches,
    )
    print(f"  NISP: Collected activations from {frl_activations.shape[0]} samples")

    print("  NISP: Computing FRL neuron importance scores...")
    frl_importance = _rank_frl_neurons(frl_activations)

    print("  NISP: Propagating importance scores through network...")
    all_importance = _propagate_importance(model, frl_importance, frl_name)

    # Only build masks for Conv2d layers (FC layers are not output-pruned).
    prunable = get_prunable_layers(model)
    prunable_importance: Dict[str, torch.Tensor] = {
        n: all_importance[n]
        for n, m in prunable
        if n in all_importance and isinstance(m, nn.Conv2d)
    }

    if not prunable_importance:
        print("  NISP WARNING: No importance scores computed. Returning model unchanged.")
        return model

    print(
        f"  NISP: Building pruning masks (global, "
        f"target={target_ratio:.0%} removed)..."
    )
    masks = _build_masks_global(prunable_importance, target_ratio)

    print("  NISP: Applying structural pruning...")
    model = apply_structural_pruning(model, masks)

    print("  NISP pruning complete.")
    return model

