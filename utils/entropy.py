"""
entropy.py  –  Entropy-based pruning for VGG-16 / CIFAR-10 benchmark

Luo & Wu (2017). "An Entropy-based Pruning Method for CNN Compression."
arXiv:1706.05791

Method summary
--------------
Each filter is scored by the Shannon entropy of its post-activation output
distribution across a calibration set.  Filters whose outputs carry little
information (low entropy) are pruned first.

For each Conv2d layer i:
  1. Run calibration images through the network.
  2. Hook post-ReLU activations: shape (B, C, H, W).
  3. Apply global average pooling → one scalar per (image, channel): (N, C).
  4. For each channel j, bin the N scalar values into m bins and compute:

         H_j = - sum_i  p_i * log(p_i)          (Shannon entropy, Eq. 1)

  5. Rank channels by H_j (ascending).  Remove the bottom `target_ratio`
     fraction — lowest entropy = least informative = most prunable.

Structural surgery (weight removal) is handled by the shared helpers from
utils.apoz, identical to APoZ and DropNet.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np

# Reuse the structural weight-surgery helpers already tested in the project.
# This guarantees identical channel-removal behaviour across all methods.
from utils.apoz import (
    apply_structural_pruning,
)


# ─────────────────────────────────────────────────────────────────────────────
# Entropy score computation
# ─────────────────────────────────────────────────────────────────────────────

def _shannon_entropy(values: np.ndarray, n_bins: int = 256) -> float:
    """
    Compute the Shannon entropy of a 1-D array of scalar values.

    Paper Eq. 1:  H_j = - sum_i  p_i * log(p_i)

    Implementation notes
    --------------------
    - We use numpy.histogram to bin the values, matching the paper's
      description of dividing the distribution into m bins.
    - Zero-probability bins contribute 0 to the sum (0 * log 0 = 0).
    - Natural log is used (consistent with information-theoretic convention
      and the paper's formulation).

    Parameters
    ----------
    values : 1-D float array of N scalar activation scores for one channel.
    n_bins : Number of histogram bins m (default 256, paper uses unspecified m;
             256 is standard and gives stable estimates with typical dataset sizes).

    Returns
    -------
    H : float  Shannon entropy ≥ 0.  Higher = more informative channel.
    """
    if len(values) == 0:
        return 0.0

    counts, _ = np.histogram(values, bins=n_bins)
    probs = counts / counts.sum()                   # normalise to probabilities
    probs = probs[probs > 0]                        # drop zero bins (0*log0 = 0)
    return float(-np.sum(probs * np.log(probs)))    # H = -sum p*log(p)


def compute_entropy_scores(
    model: nn.Module,
    calib_loader,
    device: torch.device,
    limit_batches: Optional[int] = None,
    n_bins: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-filter entropy importance scores for all Conv2d layers.

    Algorithm (paper Sec. 3.2)
    --------------------------
    For each Conv2d layer i:
      - Hook the immediately following nn.ReLU (fall back to hooking the Conv2d
        output directly if no ReLU is found within 3 positions).
      - Accumulate the GLOBAL AVERAGE POOLED output for each image:
            gap[n, c] = mean_{H,W}( act[n, c, :, :] )        shape: (N, C)
      - After all calibration batches, compute Shannon entropy of gap[:, c]
        for each channel c across all N images.

    This matches the paper exactly:
      "we first use global average pooling to convert the output of layer i,
       which is a c×h×w tensor, into a 1×c vector." (Sec. 3.2)

    Parameters
    ----------
    model         : VGG-16 model (torchvision-style, ``.features`` Sequential).
    calib_loader  : DataLoader; calibration set (no augmentation recommended).
    device        : Compute device.
    limit_batches : Stop after this many batches (quick-test mode).
    n_bins        : Histogram bins for entropy estimation (default 256).

    Returns
    -------
    scores : dict[layer_name -> 1-D float Tensor of shape (C_out,)]
        Higher score  →  more informative  →  more important (keep).
        Lower score   →  less informative  →  candidate for removal.
    """
    model.eval()

    # ── Identify which module to hook for each Conv2d layer ──────────────────
    # For VGG-16 the pattern is:  features.{i}:Conv2d  →  features.{i+1}:ReLU
    hook_targets: Dict[str, nn.Module] = {}   # layer_name → module to hook

    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        children = list(model.features.children())
        for idx, child in enumerate(children):
            if isinstance(child, nn.Conv2d):
                lname = f"features.{idx}"
                target = child                          # fallback: hook conv itself
                for j in range(idx + 1, min(idx + 4, len(children))):
                    if isinstance(children[j], nn.ReLU):
                        target = children[j]            # preferred: post-ReLU
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

    # ── Accumulate global-average-pooled activations: list of (N, C) chunks ──
    # We store all GAP vectors so we can compute the full per-channel
    # distribution across ALL calibration images before binning.
    gap_accum: Dict[str, List[torch.Tensor]] = {k: [] for k in hook_targets}
    handles: List = []

    def make_hook(layer_name: str):
        def hook_fn(module, input, output):
            with torch.no_grad():
                if output.dim() == 4:               # (B, C, H, W) — Conv2d
                    # Global average pool over spatial dimensions → (B, C)
                    gap = output.mean(dim=(2, 3)).cpu()
                else:                               # (B, C) — Linear fallback
                    gap = output.cpu()
            gap_accum[layer_name].append(gap)
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

    # ── Compute Shannon entropy per channel ───────────────────────────────────
    # Concatenate all batches → (N_total, C), then histogram each column.
    scores: Dict[str, torch.Tensor] = {}
    for lname, chunks in gap_accum.items():
        if not chunks:
            continue

        M = torch.cat(chunks, dim=0).numpy()        # (N_total, C)
        n_channels = M.shape[1]
        H = np.zeros(n_channels, dtype=np.float32)

        for c in range(n_channels):
            H[c] = _shannon_entropy(M[:, c], n_bins=n_bins)

        scores[lname] = torch.from_numpy(H)         # (C,) float32

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Keep-mask construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_keep_masks_local(
    scores: Dict[str, torch.Tensor],
    pruning_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    Layer-wise ranking (local scope).

    Within each layer independently, mark the bottom ``pruning_ratio``
    fraction of filters (lowest entropy = least informative) for removal.
    Always keeps ≥ 1 filter per layer to avoid collapsing any layer to width 0.

    Returns boolean keep-masks (True = keep this filter).
    """
    masks: Dict[str, torch.Tensor] = {}
    for lname, s in scores.items():
        n_out = len(s)
        n_prune = max(1, int(pruning_ratio * n_out))
        n_keep  = max(1, n_out - n_prune)

        # Sort ascending by entropy; lowest entropy → prune first
        sorted_idx = torch.argsort(s, descending=False)
        keep_idx   = torch.sort(sorted_idx[n_out - n_keep:]).values  # top-k by entropy

        mask = torch.zeros(n_out, dtype=torch.bool)
        mask[keep_idx] = True
        masks[lname] = mask

        n_kept   = mask.sum().item()
        n_pruned = (~mask).sum().item()
        print(f"    [{lname}] kept={n_kept}/{n_out} ({100*n_kept/n_out:.1f}%)  "
              f"pruned={n_pruned}/{n_out} ({100*n_pruned/n_out:.1f}%)")
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def apply_entropy_pruning(
    model: nn.Module,
    dataloader,
    target_ratio: float,
    scope: str = "local",
    device: Optional[torch.device] = None,
    limit_batches: Optional[int] = None,
    n_bins: int = 256,
) -> nn.Module:
    """
    Apply Entropy-based structural filter pruning to *model*.

    Implements Luo & Wu (2017), Sec. 3.2, adapted for the thesis's standardised
    fixed-ratio benchmarking protocol.

    Pipeline
    --------
    1. Forward-pass calibration data through the network.
    2. For each Conv2d layer, collect global-average-pooled post-ReLU
       activations across all calibration images → matrix M of shape (N, C).
    3. Compute Shannon entropy H_j for each channel j over M[:, j] (Eq. 1).
    4. Build keep-masks: remove the ``target_ratio`` fraction with lowest H_j
       (least informative channels), either layer-wise or globally.
    5. Structurally remove pruned channels in forward order, propagating
       dimension changes so all consecutive layers remain consistent.
    6. Resize the final classification layer's input dimension only
       (output neurons / num_classes are never changed).

    Parameters
    ----------
    model        : VGG-16 PyTorch model (torchvision-style).
    dataloader   : Calibration DataLoader (test/val split; no augmentation).
    target_ratio : Fraction of filters to *remove* (e.g. 0.3 → remove 30%).
    scope        : ``"local"``  = layer-wise ranking (recommended for VGG).
                   ``"global"`` = joint ranking across all layers.
    device       : Compute device; inferred from model parameters if None.
    limit_batches: Cap calibration batches (quick-test mode).
    n_bins       : Histogram bins for entropy estimation (default 256).

    Returns
    -------
    model : Structurally pruned ``nn.Module`` with physically removed channels.
            No fine-tuning applied — handled externally by
            ``training.fine_tune_post_pruning()`` using the standardised
            protocol common to all 12 benchmark methods.
    """
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")

    if device is None:
        device = next(model.parameters()).device

    # ── Step 1–3: Entropy scores ──────────────────────────────────────────────
    print("  Computing Entropy scores (Shannon entropy of GAP activations)...")
    scores = compute_entropy_scores(
        model, dataloader, device,
        limit_batches=limit_batches,
        n_bins=n_bins,
    )

    # ── Step 4: Keep-masks ────────────────────────────────────────────────────
    print(f"  Building pruning masks "
          f"(target: {target_ratio:.0%} removed, scope: {scope})...")

    masks = _build_keep_masks_local(scores, target_ratio)

    # ── Step 5–6: Structural weight surgery ──────────────────────────────────
    model = apply_structural_pruning(model, masks)

    print("  Entropy pruning complete.")
    return model
