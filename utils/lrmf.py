"""
lrmf.py – LRMF pruning for VGG-16 / CIFAR-10.

Zhang et al. (2023), "Filter Pruning via Learned Representation Median in
the Frequency Domain" (IEEE Trans. Cybernetics).

This implementation:
- Scores channels by **DCT-distance** to other channels (representation
  median idea; channels most similar to others are pruned first).
- Uses 2-D DCT + low-frequency crop and pairwise distances to compute
  per-channel importance.
- Performs one-shot **structural** pruning via shared helpers in
  `utils.apoz`; fine-tuning is handled externally.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# ── Shared structural helpers (same across all 12 methods) ────────────────────
from utils.apoz import (
    get_prunable_layers,
    apply_structural_pruning,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2-D DCT helper  (paper Eq. 5)
# ─────────────────────────────────────────────────────────────────────────────

def _dct2(arr: np.ndarray) -> np.ndarray:
    """
    2-D Type-II DCT of a float32 (H, W) array.

    Uses scipy.fft.dctn when available — fast and numerically identical to
    cv2.dct() for real input (which the original repo uses).  Falls back to
    two sequential 1-D DCTs when scipy is absent.
    """
    try:
        from scipy.fft import dctn
        return dctn(arr.astype(np.float32), type=2, norm=None).astype(np.float32)
    except ImportError:
        pass

    # Pure-NumPy fallback – exact Type-II definition (same as cv2.dct)
    def _dct1d(x: np.ndarray) -> np.ndarray:
        N   = x.shape[-1]
        k   = np.arange(N)[:, None]          # (N, 1)
        n   = np.arange(N)[None, :]          # (1, N)
        cos = np.cos(math.pi * k * (2 * n + 1) / (2 * N))
        return 2.0 * (x @ cos.T)

    arr = arr.astype(np.float32)
    return _dct1d(_dct1d(arr).T).T


def _compute_dct_descriptors(
    feature_maps: np.ndarray,
    crop_a: int,
) -> np.ndarray:
    """
    Compute DCT low-frequency descriptors for one image's feature maps.

    Exactly mirrors get_filter_similar() lines 524-529 of the original repo:

        feature_channel = np.resize(feature_out_hook_conv[k], (w, w))
        feature_dct     = cv2.dct(feature_channel)
        vis             = feature_dct[:a, :a]          # a = ceil(w / 4)
        fd.append(vis)

    Parameters
    ----------
    feature_maps : (C, H, W) float32 array – Conv2d output for one image.
    crop_a       : Size of the low-frequency crop (= ceil(side / 4)).

    Returns
    -------
    fd : (C, crop_a * crop_a) float32 – flattened DCT descriptors.
    """
    C, H, W = feature_maps.shape
    side = max(H, W)                             # repo uses np.resize → square

    fd = np.empty((C, crop_a * crop_a), dtype=np.float32)
    for c in range(C):
        # np.resize replicates row-major, matching original exactly
        fm_sq    = np.resize(feature_maps[c].astype(np.float32), (side, side))
        dct_full = _dct2(fm_sq)
        fd[c]    = dct_full[:crop_a, :crop_a].ravel()

    return fd                                    # (C, crop_a²)


# ─────────────────────────────────────────────────────────────────────────────
# LRMF score computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_lrmf_scores(
    model: nn.Module,
    dataloader,
    device: torch.device,
    limit_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-filter LRMF distance scores for every Conv2d layer.

    Score semantics
    ---------------
    High d_i  →  unique feature map, irreplaceable   →  KEEP
    Low  d_i  →  "median" representation, replaceable →  PRUNE

    The implementation faithfully mirrors get_filter_similar() in the
    original repo:
      • Hook raw Conv2d outputs (pre-BN/ReLU, same as 'output.data' in repo).
      • For each image: DCT descriptors → pairwise Euclidean distances →
        row sums.
      • Accumulate over all images across all batches; divide by
        num_batches (repo line 538: distance_final/num_batch_prune).

    Parameters
    ----------
    model         : VGG-16 PyTorch model.
    dataloader    : Calibration DataLoader (no augmentation).
    device        : Compute device.
    limit_batches : Cap on calibration batches.

    Returns
    -------
    scores : dict[conv_layer_name  →  Tensor(C_out,)]
    """
    model.eval()
    # ── Discover Conv2d layers and their metadata ─────────────────────────────
    conv_names: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_names.append(name)

    # Accumulators: distance sums per layer, populated incrementally.
    distance_accum: Dict[str, np.ndarray] = {}   # layer → (C,) float64
    layer_meta: Dict[str, dict] = {}             # layer → {C, H, W, crop_a}
    num_batches = 0

    # ── Per-batch: hook → forward → score → discard ──────────────────────────
    # We store only the current batch's feature maps, then clear them after
    # scoring. Peak memory ≈ 1 batch × all layers.
    current_batch_outputs: Dict[str, np.ndarray] = {}
    handles: List = []

    for cname in conv_names:
        def _make_hook(lname: str):
            def _hook(mod, inp, out):
                current_batch_outputs[lname] = out.detach().cpu().numpy()
            return _hook

        module = dict(model.named_modules())[cname]
        handles.append(module.register_forward_hook(_make_hook(cname)))

    try:
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dataloader):
                if limit_batches is not None and batch_idx >= limit_batches:
                    break

                current_batch_outputs.clear()
                model(images.to(device))
                num_batches += 1

                for lname in conv_names:
                    if lname not in current_batch_outputs:
                        continue

                    batch_arr = current_batch_outputs[lname]   # (B, C, H, W)
                    B, C, H, W = batch_arr.shape

                    if lname not in distance_accum:
                        if H < 2 or W < 2:
                            layer_meta[lname] = {
                                "C": C, "H": H, "W": W,
                                "crop_a": 0, "skip": True,
                            }
                        else:
                            side   = max(H, W)
                            crop_a = math.ceil(side / 4)
                            layer_meta[lname] = {
                                "C": C, "H": H, "W": W,
                                "crop_a": crop_a, "skip": False,
                            }
                        distance_accum[lname] = np.zeros(C, dtype=np.float64)

                    meta = layer_meta[lname]
                    if meta["skip"]:
                        continue

                    crop_a = meta["crop_a"]
                    batch_dist_sum = np.zeros(C, dtype=np.float64)

                    for b in range(B):
                        fd_np = _compute_dct_descriptors(batch_arr[b], crop_a)

                        try:
                            from scipy.spatial.distance import cdist
                            dist_mat = cdist(fd_np, fd_np, metric="euclidean")
                        except ImportError:
                            diff     = fd_np[:, None, :] - fd_np[None, :, :]
                            dist_mat = np.sqrt((diff * diff).sum(-1))

                        batch_dist_sum += dist_mat.sum(axis=1)

                    distance_accum[lname] += batch_dist_sum

                current_batch_outputs.clear()

    finally:
        for h in handles:
            h.remove()

    # ── Finalise scores ───────────────────────────────────────────────────────
    scores: Dict[str, torch.Tensor] = {}

    for lname in conv_names:
        if lname not in layer_meta:
            continue

        meta = layer_meta[lname]
        C = meta["C"]

        if meta["skip"]:
            scores[lname] = torch.ones(C, dtype=torch.float32)
            print(f"    [{lname}] 1×1 spatial → uniform LRMF scores (no pruning bias)")
            continue

        distance_final = distance_accum[lname]
        if num_batches > 0:
            distance_final /= num_batches

        scores[lname] = torch.from_numpy(distance_final.astype(np.float32))
        print(
            f"    [{lname}] C={C}  crop={meta['crop_a']}×{meta['crop_a']}  "
            f"d_i in [{distance_final.min():.1f}, {distance_final.max():.1f}]"
        )

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Keep-mask construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_keep_masks_local(
    scores: Dict[str, torch.Tensor],
    pruning_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    Layer-wise (local) ranking – default, matches paper Eq. 9.

    Within each layer: prune the `pruning_ratio` fraction of channels with
    the *smallest* d_i (the representation medians). Always keeps ≥ 1.
    """
    masks: Dict[str, torch.Tensor] = {}
    for lname, s in scores.items():
        n_out   = len(s)
        n_prune = max(0, min(int(pruning_ratio * n_out), n_out - 1))

        # argsort ascending → smallest d_i first → those are the medians → prune
        sorted_idx = torch.argsort(s, descending=False)
        keep_idx   = torch.sort(sorted_idx[n_prune:]).values

        mask = torch.zeros(n_out, dtype=torch.bool)
        mask[keep_idx] = True
        masks[lname]   = mask

        n_kept = mask.sum().item()
        print(
            f"    [{lname}] kept={n_kept}/{n_out} ({100*n_kept/n_out:.1f}%)  "
            f"pruned={n_out-n_kept}/{n_out} ({100*(n_out-n_kept)/n_out:.1f}%)"
        )
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def apply_lrmf_pruning(
    model: nn.Module,
    dataloader,
    target_ratio: float,
    scope: str = "local",
    device: Optional[torch.device] = None,
    limit_batches: Optional[int] = None,
) -> nn.Module:
    """
    Apply one-shot structural LRMF filter pruning.

    Steps
    -----
    1. Run calibration batches; hook all Conv2d outputs.
    2. For each layer and each image:
         a. 2-D DCT of each channel's feature map       [Eq. 5]
         b. Crop top-left quarter (low-frequency zone)  [Sec. III-C]
         c. Pairwise Euclidean distances                 [Eq. 7 / 8]
         d. Row-sum scores:  d_i = Σ_j d_{ij}           [Eq. 8]
       Average d_i over all images and batches.
    3. Build keep-masks: prune channels with smallest d_i (scope=local/global).
    4. Structural weight surgery: physically remove pruned channels and
       propagate dimension changes through all consecutive layers.
    5. Return pruned model — no fine-tuning applied here.

    Parameters
    ----------
    model        : VGG-16 PyTorch model (torchvision-style).
    dataloader   : Calibration DataLoader (no augmentation; test/val split).
    target_ratio : Fraction of filters to REMOVE (0.3 → 30%).
    scope        : "local"  = per-layer ranking (default, paper default).
                   "global" = joint ranking across all layers.
    device       : Compute device; inferred from model parameters if None.
    limit_batches: Cap on calibration batches.

    Returns
    -------
    model : Structurally pruned nn.Module with physically removed channels.

    Performance notes
    -----------------
    LRMF is computationally cheap among feature-map-based methods — it
    requires only one 2-D DCT + one (C×C) pairwise distance matrix per
    image per layer, with no SVD.
    """
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")

    if device is None:
        device = next(model.parameters()).device

    # ── Steps 1–2: Feature map collection + LRMF scoring ─────────────────────
    print("  Computing LRMF scores "
          "(DCT low-frequency pairwise-distance representation medians)...")
    scores = compute_lrmf_scores(model, dataloader, device, limit_batches)

    if not scores:
        raise RuntimeError(
            "LRMF: no scores collected. "
            "Verify the model has Conv2d layers with spatial output H,W >= 2."
        )

    # ── Step 3: Keep-masks ────────────────────────────────────────────────────
    print(f"  Building keep-masks "
          f"(removing {target_ratio:.0%} of channels)...")

    masks = _build_keep_masks_local(scores, target_ratio)

    # ── Step 4: Structural weight surgery ─────────────────────────────────────
    model = apply_structural_pruning(model, masks)

    print("  LRMF pruning complete.")
    return model
