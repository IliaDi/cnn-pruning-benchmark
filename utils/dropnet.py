"""
dropnet.py  –  DropNet pruning for VGG-16 / CIFAR-10 benchmark

Tan & Motani (2020). "DropNet: Reducing Neural Network Complexity via
Iterative Pruning."  ICML 2020.  arXiv:2207.06646

Method summary
--------------
DropNet scores each filter by its *expected absolute post-activation value*
across all training samples:

    E(f_i) = (1/T) * sum_j  mean_{H,W} |ReLU(conv_i(x_j))|
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Reuse the structural weight-surgery helpers that are already tested in the
# project.  This guarantees identical channel-removal behaviour across methods.
from utils.apoz import (
    get_prunable_layers,
    apply_structural_pruning,
)


# ─────────────────────────────────────────────────────────────────────────────
# Score computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_dropnet_scores(
    model: nn.Module,
    calib_loader,
    device: torch.device,
    limit_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-filter DropNet importance scores for all Conv2d layers.

    For each Conv2d we hook the immediately following nn.ReLU (or the Conv2d
    itself as a fallback) and accumulate:

        score[c] = (1/N) * sum_n  mean_{H,W} |act_n[c, :, :]|

    where N is the total number of calibration samples processed.

    Parameters
    ----------
    model         : VGG-16 model (torchvision-style, ``.features`` Sequential).
    calib_loader  : DataLoader; calibration set (no augmentation recommended).
    device        : Compute device.
    limit_batches : Stop after this many batches if set (quick-test mode).

    Returns
    -------
    scores : dict[layer_name -> 1-D float Tensor of shape (C_out,)]
        Higher score  →  more active  →  more important.
    """
    model.eval()

    # ── Map each conv-layer name to the module we should hook ────────────────────
    # For VGG-16 the pattern is  features.{i}:Conv2d  →  features.{i+1}:ReLU
    hook_targets: Dict[str, nn.Module] = {}

    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        children = list(model.features.children())
        for idx, child in enumerate(children):
            if isinstance(child, nn.Conv2d):
                lname = f"features.{idx}"
                # Search for the next ReLU within the same Sequential block
                target = child          # fallback: hook conv output directly
                for j in range(idx + 1, min(idx + 4, len(children))):
                    if isinstance(children[j], nn.ReLU):
                        target = children[j]
                        break
                hook_targets[lname] = target
    else:
        # Generic fallback for non-standard VGG variants
        named = list(model.named_modules())
        for idx, (name, mod) in enumerate(named):
            if isinstance(mod, nn.Conv2d):
                parent_prefix = ".".join(name.split(".")[:-1])
                target = mod
                for j in range(idx + 1, min(idx + 4, len(named))):
                    cname, cmod = named[j]
                    if (isinstance(cmod, nn.ReLU)
                            and ".".join(cname.split(".")[:-1]) == parent_prefix):
                        target = cmod
                        break
                hook_targets[name] = target

    # ── Accumulate weighted-mean absolute activations ───────────────────────────
    accum: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    handles: List = []

    def make_hook(layer_name: str):
        def hook_fn(module, input, output):
            with torch.no_grad():
                if output.dim() == 4:                          # (N, C, H, W)
                    val = output.abs().mean(dim=(0, 2, 3)).cpu()
                else:                                          # (N, C)
                    val = output.abs().mean(dim=0).cpu()
            n = output.shape[0]
            if layer_name not in accum:
                accum[layer_name] = val * n
                counts[layer_name] = n
            else:
                accum[layer_name] += val * n
                counts[layer_name] += n
        return hook_fn

    for lname, mod in hook_targets.items():
        handles.append(mod.register_forward_hook(make_hook(lname)))

    try:
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(calib_loader):
                if limit_batches is not None and batch_idx >= limit_batches:
                    break
                model(images.to(device))
    finally:
        for h in handles:
            h.remove()

    return {
        lname: accum[lname] / counts[lname]
        for lname in accum
        if counts.get(lname, 0) > 0
    }


# ─────────────────────────────────────────────────────────────────────────────
# Keep-mask construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_keep_masks_layer(
    scores: Dict[str, torch.Tensor],
    pruning_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    Layer-wise (``min_layer``) ranking.

    Within each layer independently, mark the bottom ``pruning_ratio``
    fraction of filters for removal.  Returns boolean keep-masks
    (True = keep this filter).  Always keeps ≥ 1 filter per layer.
    """
    masks: Dict[str, torch.Tensor] = {}
    for lname, s in scores.items():
        n_out = len(s)
        n_prune = max(1, int(pruning_ratio * n_out))
        n_keep = max(1, n_out - n_prune)
        sorted_idx = torch.argsort(s, descending=False)
        keep_idx = torch.sort(sorted_idx[n_out - n_keep:]).values
        mask = torch.zeros(n_out, dtype=torch.bool)
        mask[keep_idx] = True
        masks[lname] = mask
        
        n_kept = mask.sum().item()
        n_pruned = (~mask).sum().item()
        print(f"    [{lname}] kept={n_kept}/{n_out} ({100*n_kept/n_out:.1f}%)  "
              f"pruned={n_pruned}/{n_out} ({100*n_pruned/n_out:.1f}%)")
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def apply_dropnet_pruning(
    model: nn.Module,
    dataloader,
    target_ratio: float,
    scope: str = "local",
    device: Optional[torch.device] = None,
    limit_batches: Optional[int] = None,
) -> nn.Module:
    """
    Apply DropNet-based structural filter pruning to *model*.

    Pipeline
    --------
    1. Forward-pass calibration data  →  mean absolute post-ReLU activations
       per filter  (DropNet importance score).
    2. Build per-layer keep-masks via layer-wise or global ranking.
    3. Structurally remove pruned channels in forward order, propagating
       dimension changes so consecutive layers remain consistent.
    4. Resize the final classification layer's input dimension only
       (output neurons / num_classes are never changed).

    Parameters
    ----------
    model        : VGG-16 PyTorch model (torchvision-style).
    dataloader   : Calibration DataLoader (test/val split; no augmentation).
    target_ratio : Fraction of filters to *remove* (e.g. 0.3 → remove 30%).
    scope        : ``"local"``  = layer-wise / ``min_layer`` in the paper
                                  (recommended for VGG-style models).
                   ``"global"`` = joint ranking / ``min`` in the paper.
    device       : Compute device; inferred from model parameters if None.
    limit_batches: Cap calibration batches (quick-test mode).

    Returns
    -------
    model : Structurally pruned ``nn.Module`` with physically removed channels.
            No fine-tuning is applied here — that is handled externally by
            ``training.fine_tune_post_pruning()`` using the standardised
            protocol common to all 12 benchmark methods.
    """
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")

    if device is None:
        device = next(model.parameters()).device

    # ── Step 1: Scores ────────────────────────────────────────────────────────
    print("  Computing DropNet scores (mean absolute post-ReLU activations)...")
    scores = compute_dropnet_scores(model, dataloader, device, limit_batches)

    # ── Step 2: Keep-masks ────────────────────────────────────────────────────
    print(f"  Building pruning masks "
          f"(target: {target_ratio:.0%} removed, scope: {scope})...")

    masks = _build_keep_masks_layer(scores, target_ratio)

    # ── Step 3: Structural weight surgery ─────────────────────────────────────
    model = apply_structural_pruning(model, masks)

    print("  DropNet pruning complete.")
    return model
