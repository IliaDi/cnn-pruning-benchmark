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
    prune_conv2d_layer,
    prune_linear_layer,
    _update_output_layer,
    _get_module_at_path,
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


def _build_keep_masks_global(
    scores: Dict[str, torch.Tensor],
    pruning_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    Global (``min``) ranking.

    Rank all filters across all layers jointly and prune the globally
    lowest-scoring ``pruning_ratio`` fraction.  Always keeps ≥ 1 filter
    per layer (guard against collapsing any layer to zero width).
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
    for lname in layer_names:
        s = scores[lname]
        n_out = len(s)
        keep = torch.tensor(
            [(offset + i) not in prune_set for i in range(n_out)],
            dtype=torch.bool,
        )
        if keep.sum() == 0:                      # guard: keep best filter
            keep[int(torch.argmax(s).item())] = True
        masks[lname] = keep
        
        n_kept = keep.sum().item()
        n_pruned = (~keep).sum().item()
        print(f"    [{lname}] kept={n_kept}/{n_out} ({100*n_kept/n_out:.1f}%)  "
              f"pruned={n_pruned}/{n_out} ({100*n_pruned/n_out:.1f}%)")
        offset += n_out
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
    if limit_batches is not None:
        print(f"    (Limited to {limit_batches} calibration batches)")

    scores = compute_dropnet_scores(model, dataloader, device, limit_batches)

    # ── Step 2: Keep-masks ────────────────────────────────────────────────────
    print(f"  Building pruning masks "
          f"(target: {target_ratio:.0%} removed, scope: {scope})...")

    if scope == "local":
        masks = _build_keep_masks_layer(scores, target_ratio)
    elif scope == "global":
        masks = _build_keep_masks_global(scores, target_ratio)
    else:
        raise ValueError(f"scope must be 'local' or 'global', got '{scope}'")

    # ── Step 3: Structural weight surgery (forward order) ─────────────────────
    # ``get_prunable_layers`` returns (Conv2d + hidden Linear) layers,
    # excluding the final classification layer.
    prunable = get_prunable_layers(model)
    prev_keep: Optional[torch.Tensor] = None

    for i, (name, module) in enumerate(prunable):
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        layer_idx = int(parts[-1])

        # ── Conv2d ────────────────────────────────────────────────────────
        if isinstance(module, nn.Conv2d):
            keep_out = masks.get(name, torch.ones(module.out_channels, dtype=torch.bool))
            parent[layer_idx] = prune_conv2d_layer(module, keep_out, keep_in=prev_keep)
            prev_keep = keep_out

        # ── Hidden Linear ─────────────────────────────────────────────────
        elif isinstance(module, nn.Linear):
            keep_out = masks.get(name, torch.ones(module.out_features, dtype=torch.bool))

            prev_name, prev_orig = prunable[i - 1] if i > 0 else (None, None)
            follows_conv = (prev_orig is not None and isinstance(prev_orig, nn.Conv2d))

            if follows_conv and prev_keep is not None:
                # Expand channel-level mask to cover all H×W spatial positions
                spatial = module.in_features // prev_orig.out_channels
                kept_ch = torch.where(prev_keep)[0].tolist()
                flat_idx = torch.tensor(
                    [c * spatial + s for c in kept_ch for s in range(spatial)],
                    dtype=torch.long,
                )
                out_idx = torch.where(keep_out)[0].tolist()
                new_module = nn.Linear(
                    len(kept_ch) * spatial, len(out_idx),
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
                    module, keep_out,
                    keep_in=prev_keep,
                    in_features_override=in_feat_live,
                )

            parent[layer_idx] = new_module
            prev_keep = keep_out

    # ── Step 4: Final classification layer (input dim only) ───────────────────
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

    print("  DropNet pruning complete.")
    return model
