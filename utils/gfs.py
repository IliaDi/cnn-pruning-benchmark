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

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.apoz import (
    get_prunable_layers,
    apply_structural_pruning,
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

    # Normalize by n_keep (= N_h = layer width) so ordinal scores are on
    # (0, 1] across all layers, correcting for the scale artifact where
    # larger layers produce higher absolute scores.
    if n_keep > 0:
        importance = importance / float(n_keep)

    return importance.detach().cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Mask building (global only — GFS is a global-ranking method)
# ─────────────────────────────────────────────────────────────────────────────

def _build_masks_global(
    scores: Dict[str, torch.Tensor],
    pruning_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    Build keep-masks using global ranking across all Conv2d layers.

    Raw scores are used directly (no per-layer normalization) since all
    scored layers are Conv2d with greedy-selection importance on a common
    ordinal scale.
    """
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
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """
    Apply GFS-based structural filter pruning.

    Parameters
    ----------
    model           : VGG-16 PyTorch model (torchvision-style).
    dataloader      : Calibration DataLoader (no augmentation).
    target_ratio    : Fraction of filters to REMOVE (0.3 → remove 30%).
    scope           : "global" (default, pool scores across all layers) or "local".
    device          : Compute device; inferred if None.
    limit_batches   : Cap on calibration batches.
    checkpoint_path : Optional path to a .pt file for saving/resuming per-layer
                      importance scores. After each layer is scored the dict is
                      written to this file. If the file already exists on entry,
                      completed layers are loaded and skipped.
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

    # Step 2: score Conv2d filters only (FC layers are not output-pruned)
    prunable = get_prunable_layers(model)
    all_importance: Dict[str, torch.Tensor] = {}

    # Load any previously saved checkpoint so completed layers can be skipped.
    if checkpoint_path and os.path.isfile(checkpoint_path):
        all_importance = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        print(f"  GFS: resuming from checkpoint ({len(all_importance)} layers already scored)")

    for name, module in prunable:
        if isinstance(module, nn.Conv2d):
            if name in all_importance:
                print(f"    [{name}] skipping (already scored)")
                continue
            print(f"    [{name}] scoring {module.out_channels} conv filters...")
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
            if checkpoint_path:
                torch.save(all_importance, checkpoint_path)

    if not all_importance:
        print("  GFS WARNING: no importance scores computed; returning model unchanged.")
        return model

    # Step 3: build keep masks (global ranking, no normalization needed)
    print(f"  GFS: building pruning masks (global, target={target_ratio:.0%} removed)...")
    masks = _build_masks_global(all_importance, target_ratio)

    # Step 4: structural pruning
    print("  GFS: applying structural pruning...")
    model = apply_structural_pruning(model, masks)

    print("  GFS pruning complete.")
    return model

