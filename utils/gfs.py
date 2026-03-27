"""
gfs.py – GFS (Greedy Forward Selection) pruning.

Ye et al. (2020), "Good Subnetworks Provably Exist: Pruning via Greedy
Forward Selection" (ICML 2020).

This implementation:
- Implements greedy forward selection by iteratively growing a set of
  retained filters for each Conv2d layer; each step selects the candidate
  filter whose inclusion yields the lowest loss on a calibration subset.
  To keep runtime tractable for wide layers, candidate evaluation is
  restricted to a random subset per greedy step.
- Builds local/global masks from these importance scores; linear layers
  use a weight-magnitude fallback.
- Applies **structural** pruning using helpers from `utils.apoz`; all
  fine-tuning remains in the shared training utilities.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

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
# Calibration data collection
# ─────────────────────────────────────────────────────────────────────────────

def _get_calibration_batch(
    dataloader,
    device: torch.device,
    limit_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect a calibration dataset from the dataloader.

    Returns concatenated (images, labels) tensors, limited by limit_batches.
    """
    imgs_list, lbls_list = [], []
    for i, (imgs, lbls) in enumerate(dataloader):
        if limit_batches is not None and i >= limit_batches:
            break
        imgs_list.append(imgs.to(device))
        lbls_list.append(lbls.to(device))
    if not imgs_list:
        raise RuntimeError("GFS: calibration dataloader produced no batches.")
    return torch.cat(imgs_list, dim=0), torch.cat(lbls_list, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Core: loss evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute cross-entropy loss on a batch (no grad)."""
    model.eval()
    with torch.no_grad():
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
    return float(loss.item())


def _score_filters_by_greedy_forward_selection(
    model: nn.Module,
    layer_name: str,
    layer_module: nn.Conv2d,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    n_keep: int,
    eval_batch_size: int = 256,
) -> torch.Tensor:
    """
    Greedy forward selection scoring for one Conv2d layer.

    Builds a retained set S iteratively. At greedy step t, it considers
    candidates k and picks the one with lowest loss when using
    exactly S U {k} active output channels (scaled by |S| normalization).

    Importance is the reverse selection order: filters picked earlier get
    higher scores.
    """
    N_h = layer_module.out_channels
    n_keep = max(1, min(int(n_keep), N_h))

    # Restrict evaluated candidates per greedy step for runtime.
    num_evaluate = min(20, N_h)

    # Random subset of images for tractable loss evaluation.
    if len(images) > eval_batch_size:
        perm = torch.randperm(len(images))[:eval_batch_size]
        eval_imgs = images[perm]
        eval_lbls = labels[perm]
    else:
        eval_imgs = images
        eval_lbls = labels

    orig_weight = layer_module.weight.data.clone()
    orig_bias = layer_module.bias.data.clone() if layer_module.bias is not None else None

    importance = torch.zeros(N_h, dtype=torch.float32, device="cpu")
    selected = torch.zeros(N_h, dtype=torch.bool, device="cpu")

    for step in range(n_keep):
        remaining = torch.where(~selected)[0].tolist()
        if not remaining:
            break

        selected_idxs = torch.where(selected)[0].tolist()
        if len(remaining) > num_evaluate:
            perm = torch.randperm(len(remaining))[:num_evaluate].tolist()
            candidates = [remaining[i] for i in perm]
        else:
            candidates = remaining

        best_k = None
        best_loss = float("inf")

        for k in candidates:
            candidate = selected_idxs + [k]
            scale = float(N_h) / float(len(candidate))

            # Mask weights to keep only candidate output channels
            layer_module.weight.data.zero_()
            layer_module.weight.data[candidate] = orig_weight[candidate]
            layer_module.weight.data.mul_(scale)

            if orig_bias is not None:
                layer_module.bias.data.zero_()
                layer_module.bias.data[candidate] = orig_bias[candidate]
                layer_module.bias.data.mul_(scale)

            loss = _compute_loss(model, eval_imgs, eval_lbls)
            if loss < best_loss:
                best_loss = loss
                best_k = k

        if best_k is None:
            break

        selected[best_k] = True
        importance[best_k] = float(n_keep - step)

    # Restore original weights
    layer_module.weight.data = orig_weight
    if orig_bias is not None:
        layer_module.bias.data = orig_bias

    return importance.detach().cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Mask building
# ─────────────────────────────────────────────────────────────────────────────

def _build_masks_from_importance(
    scores: Dict[str, torch.Tensor],
    pruning_ratio: float,
    scope: str = "global",
) -> Dict[str, torch.Tensor]:
    """
    Build keep-masks from importance scores.

    For scope="global": pool all scores across layers, find global threshold.
    For scope="local": per-layer threshold.
    """
    if scope == "global":
        layer_names = list(scores.keys())
        all_scores = torch.cat([scores[n] for n in layer_names])
        n_total = len(all_scores)
        n_prune = max(1, int(pruning_ratio * n_total))
        prune_set = set(
            torch.argsort(all_scores, descending=False)[:n_prune].tolist()
        )

        masks: Dict[str, torch.Tensor] = {}
        offset = 0
        for name in layer_names:
            s = scores[name]
            n_out = len(s)
            keep = torch.tensor(
                [(offset + i) not in prune_set for i in range(n_out)],
                dtype=torch.bool,
            )
            if keep.sum() == 0:
                keep[int(torch.argmax(s).item())] = True
            masks[name] = keep
            n_kept = int(keep.sum().item())
            print(
                f"    [{name}] kept={n_kept}/{n_out} ({100*n_kept/n_out:.1f}%)  "
                f"pruned={n_out-n_kept}/{n_out} ({100*(n_out-n_kept)/n_out:.1f}%)"
            )
            offset += n_out
        return masks

    else:  # local
        masks = {}
        for name, s in scores.items():
            n_out = len(s)
            n_prune = max(0, int(pruning_ratio * n_out))
            n_keep = max(1, n_out - n_prune)
            _, top_idx = torch.topk(s, n_keep)
            keep = torch.zeros(n_out, dtype=torch.bool)
            keep[top_idx] = True
            masks[name] = keep
            n_kept = int(keep.sum().item())
            print(
                f"    [{name}] kept={n_kept}/{n_out} ({100*n_kept/n_out:.1f}%)  "
                f"pruned={n_out-n_kept}/{n_out} ({100*(n_out-n_kept)/n_out:.1f}%)"
            )
        return masks


# ─────────────────────────────────────────────────────────────────────────────
# Structural pruning (shared surgery from apoz.py)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_structural_pruning(model: nn.Module, masks: Dict[str, torch.Tensor]) -> nn.Module:
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
            keep_out = masks.get(name, torch.ones(module.out_channels, dtype=torch.bool))
            parent[layer_idx] = prune_conv2d_layer(module, keep_out, keep_in=prev_keep)
            prev_keep = keep_out

        elif isinstance(module, nn.Linear):
            keep_out = masks.get(name, torch.ones(module.out_features, dtype=torch.bool))
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
                new_module.weight.data = module.weight.data[out_idx][:, flat_idx].clone()
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

    # Update output (classification) layer
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

def apply_gfs_pruning(
    model: nn.Module,
    dataloader,
    target_ratio: float,
    scope: str = "global",
    device: Optional[torch.device] = None,
    limit_batches: Optional[int] = None,
) -> nn.Module:
    """
    Apply GFS-based structural filter pruning.

    Parameters
    ----------
    model        : VGG-16 PyTorch model (torchvision-style).
    dataloader   : Calibration DataLoader (no augmentation).
    target_ratio : Fraction of filters to REMOVE (0.3 → remove 30%).
    scope        : "global" (default, pool scores across all layers) or "local".
    device       : Compute device; inferred if None.
    limit_batches: Cap on calibration batches.
    """
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")

    if device is None:
        device = next(model.parameters()).device

    print(f"  GFS: starting greedy forward-selection pruning (target={target_ratio:.0%} removed)...")

    # Step 1: collect calibration data
    print("  GFS: collecting calibration data...")
    calib_images, calib_labels = _get_calibration_batch(
        dataloader, device, limit_batches=limit_batches
    )
    print(f"  GFS: calibration set size = {calib_images.shape[0]} samples")

    # Step 2: score filters in prunable layers
    prunable = get_prunable_layers(model)
    all_importance: Dict[str, torch.Tensor] = {}

    for name, module in prunable:
        if isinstance(module, nn.Conv2d):
            print(f"    [{name}] scoring {module.out_channels} conv filters...")
            # Build a full per-layer ranking by selecting all filters in order.
            n_keep = module.out_channels
            importance = _score_filters_by_greedy_forward_selection(
                model,
                name,
                module,
                calib_images,
                calib_labels,
                device,
                n_keep=n_keep,
                eval_batch_size=64,
            )
            all_importance[name] = importance

    # Step 3: hidden Linear layers — weight magnitude fallback
    for name, module in prunable:
        if isinstance(module, nn.Linear):
            importance = module.weight.data.abs().sum(dim=1).cpu()
            all_importance[name] = importance
            print(f"    [{name}] linear layer — using weight magnitude fallback")

    if not all_importance:
        print("  GFS WARNING: no importance scores computed; returning model unchanged.")
        return model

    # Step 4: build keep masks
    print(f"  GFS: building pruning masks (scope={scope}, target={target_ratio:.0%} removed)...")
    if scope == "global":
        # Global ranking mixes heterogeneous score sources (conv greedy-loss
        # scores and linear weight-magnitude fallback). Normalize per-layer
        # so each layer contributes comparably.
        normalized: Dict[str, torch.Tensor] = {}
        for name, s in all_importance.items():
            s_min = s.min()
            s_max = s.max()
            if (s_max - s_min) > 1e-12:
                normalized[name] = (s - s_min) / (s_max - s_min)
            else:
                normalized[name] = torch.ones_like(s)
        masks = _build_masks_from_importance(normalized, target_ratio, scope=scope)
    else:
        masks = _build_masks_from_importance(all_importance, target_ratio, scope=scope)

    # Step 5: structural pruning
    print("  GFS: applying structural pruning...")
    model = _apply_structural_pruning(model, masks)

    print("  GFS pruning complete.")
    return model

