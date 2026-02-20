"""
lrmf.py  –  LRMF filter pruning for the VGG-16 / CIFAR-10 benchmark

Reference
---------
Zhang et al. (2023). "Filter Pruning via Learned Representation Median in
the Frequency Domain." IEEE Transactions on Cybernetics, Vol. 53, No. 5.
DOI: 10.1109/TCYB.2021.3124284
GitHub: https://github.com/zhangxin-xd/LRMF

Core idea
---------
LRMF scores each output channel of a Conv2d layer by how *dissimilar* its
feature map is from all others in the same layer, measured after mapping the
representations into the frequency domain via DCT.

  1. For each channel k in layer l, compute the 2-D DCT of the output
     feature map M^l_k.                                          [Eq. 5]
  2. Crop to the top-left quarter (low-frequency region only):
         T^s(M^l_k) = T(M^l_k)[:ceil(side/4), :ceil(side/4)]   [Sec. III-C]
  3. Build the C×C pairwise Euclidean distance matrix D:
         d_{ij} = ||T^s(M^l_i) − T^s(M^l_j)||_2                [Eq. 8]
  4. Row-sum each channel:
         d_i = Σ_j d_{ij}                                       [Eq. 8]
  5. Channels with the *smallest* d_i are the "representation medians" —
     most replaceable by neighbours → prune.
     Channels with the *largest*  d_i carry unique information   → keep.
         s_i = 0  if i ∈ argsort(d_i)[:α·C]                    [Eq. 9]
  6. Scores are accumulated and averaged across all calibration images.

Adaptation for this project's benchmarking protocol
----------------------------------------------------
The original LRMF repo (pruning_cifar10.py) uses iterative soft-masking:
it zeroes filter weights every `epoch_prune` epochs during training, with
the gradient also masked at every step.  This bakes fine-tuning deep into
the pruning loop.

For fair comparison across all 12 methods we use one-shot STRUCTURAL
pruning (physical channel removal) + the standardised external fine-tuning
protocol (identical for all methods).  We therefore:
  • Compute LRMF scores identically to the repo (same DCT, same quarter-
    crop, same pairwise Euclidean distance, same per-image averaging).
  • Apply structural weight surgery via the shared helpers from utils.apoz,
    exactly as APoZ, DropNet, HRank and CHIP do.
  • Return the pruned model with no fine-tuning applied.

This ensures the importance criterion is faithfully reproduced while the
fine-tuning advantage of the original iterative scheme is neutralised,
enabling a fair controlled comparison.

Thesis note
-----------
Paper results for VGG-16 on CIFAR-10 use non-uniform per-layer ratios
(supplementary Sec. 7).  Our uniform target_ratio will produce slightly
different compression numbers — this is correct and expected under the
standardised benchmarking protocol.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

# ── Shared structural helpers (same across all 12 methods) ────────────────────
from utils.apoz import (
    _get_module_at_path,
    get_prunable_layers,
    prune_conv2d_layer,
    prune_linear_layer,
    _update_output_layer,
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

    # ── Register hooks on every Conv2d ────────────────────────────────────────
    layer_outputs: Dict[str, List[np.ndarray]] = {}
    handles: List = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layer_outputs[name] = []

            def _make_hook(lname: str):
                def _hook(mod, inp, out):
                    layer_outputs[lname].append(out.detach().cpu().numpy())
                return _hook

            handles.append(module.register_forward_hook(_make_hook(name)))

    try:
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(dataloader):
                if limit_batches is not None and batch_idx >= limit_batches:
                    break
                model(images.to(device))
    finally:
        for h in handles:
            h.remove()

    # ── Compute per-layer LRMF scores ─────────────────────────────────────────
    scores: Dict[str, torch.Tensor] = {}

    for lname, batch_list in layer_outputs.items():
        if not batch_list:
            continue

        num_batches = len(batch_list)
        C = batch_list[0].shape[1]
        H = batch_list[0].shape[2]
        W = batch_list[0].shape[3]

        if H < 2 or W < 2:
            # 1×1 spatial: DCT is trivial; assign uniform scores → no pruning bias
            scores[lname] = torch.ones(C, dtype=torch.float32)
            print(f"    [{lname}] 1×1 spatial → uniform LRMF scores (no pruning bias)")
            continue

        side   = max(H, W)
        crop_a = math.ceil(side / 4)             # a = ceil(w/4), matches repo

        # Accumulate distance sums across all batches × images.
        # Divided by num_batches at the end — matches repo lines 537-539.
        distance_final = np.zeros(C, dtype=np.float64)

        for batch_arr in batch_list:             # shape: (B, C, H, W)
            B = batch_arr.shape[0]
            batch_dist_sum = np.zeros(C, dtype=np.float64)

            for b in range(B):
                # fd_np: (C, crop_a²) – DCT descriptors for one image
                fd_np = _compute_dct_descriptors(batch_arr[b], crop_a)

                # Pairwise Euclidean distance matrix D ∈ R^{C×C}  [Eq. 8]
                try:
                    from scipy.spatial.distance import cdist
                    dist_mat = cdist(fd_np, fd_np, metric='euclidean')
                except ImportError:
                    diff     = fd_np[:, None, :] - fd_np[None, :, :]
                    dist_mat = np.sqrt((diff * diff).sum(-1))

                # Row-sum: d_i = Σ_j d_{ij}   [Eq. 8]
                batch_dist_sum += dist_mat.sum(axis=1)

            distance_final += batch_dist_sum

        # Average over batches
        distance_final /= num_batches

        scores[lname] = torch.from_numpy(distance_final.astype(np.float32))
        print(
            f"    [{lname}] C={C}  crop={crop_a}×{crop_a}  "
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


def _build_keep_masks_global(
    scores: Dict[str, torch.Tensor],
    pruning_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    Global joint ranking – prune the lowest `pruning_ratio` fraction of
    all channels across the entire network. Always keeps ≥ 1 per layer.
    """
    layer_names = list(scores.keys())
    all_scores  = torch.cat([scores[n] for n in layer_names])
    n_prune     = max(1, int(pruning_ratio * len(all_scores)))

    prune_set = set(
        torch.argsort(all_scores, descending=False)[:n_prune].tolist()
    )

    masks: Dict[str, torch.Tensor] = {}
    offset = 0
    for lname in layer_names:
        s     = scores[lname]
        n_out = len(s)
        keep  = torch.tensor(
            [(offset + i) not in prune_set for i in range(n_out)],
            dtype=torch.bool,
        )
        if keep.sum() == 0:                      # guard: always keep ≥ 1
            keep[int(torch.argmax(s).item())] = True
        masks[lname] = keep

        n_kept = keep.sum().item()
        print(
            f"    [{lname}] kept={n_kept}/{n_out} ({100*n_kept/n_out:.1f}%)  "
            f"pruned={n_out-n_kept}/{n_out} ({100*(n_out-n_kept)/n_out:.1f}%)"
        )
        offset += n_out
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
          f"(scope={scope}, removing {target_ratio:.0%} of channels)...")

    if scope == "local":
        masks = _build_keep_masks_local(scores, target_ratio)
    elif scope == "global":
        masks = _build_keep_masks_global(scores, target_ratio)
    else:
        raise ValueError(f"scope must be 'local' or 'global', got '{scope}'")

    # ── Step 4: Structural weight surgery ─────────────────────────────────────
    # Mirrors apply_apoz_pruning() exactly.  We iterate `prunable` in forward
    # order; prev_keep propagates the output mask of layer i as the input
    # mask of layer i+1.
    prunable  = get_prunable_layers(model)
    prev_keep: Optional[torch.Tensor] = None

    for i, (name, module) in enumerate(prunable):

        # Navigate to parent container + integer index
        parts  = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        layer_idx = int(parts[-1])

        # ── Conv2d ────────────────────────────────────────────────────────────
        if isinstance(module, nn.Conv2d):
            keep_out = masks.get(
                name, torch.ones(module.out_channels, dtype=torch.bool)
            )
            parent[layer_idx] = prune_conv2d_layer(module, keep_out, keep_in=prev_keep)
            prev_keep = keep_out

        # ── Hidden Linear ─────────────────────────────────────────────────────
        elif isinstance(module, nn.Linear):
            keep_out = masks.get(
                name, torch.ones(module.out_features, dtype=torch.bool)
            )

            prev_name, prev_orig = prunable[i - 1] if i > 0 else (None, None)
            follows_conv = prev_orig is not None and isinstance(prev_orig, nn.Conv2d)

            if follows_conv and prev_keep is not None:
                # in_features = C_prev × spatial; expand channel mask accordingly
                spatial  = module.in_features // prev_orig.out_channels
                kept_ch  = torch.where(prev_keep)[0].tolist()
                flat_idx = torch.tensor(
                    [c * spatial + s for c in kept_ch for s in range(spatial)],
                    dtype=torch.long,
                )
                out_idx    = torch.where(keep_out)[0].tolist()
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
                # Linear → Linear: use live out_features from already-replaced
                # previous layer to guarantee consistent dimensions
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

    # ── Step 5: Resize final classification layer's input (outputs intact) ────
    all_linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if all_linears:
        out_name  = all_linears[-1]
        out_parts = out_name.split(".")
        out_par   = model
        for p in out_parts[:-1]:
            out_par = getattr(out_par, p)
        out_idx_int = int(out_parts[-1])
        out_mod     = out_par[out_idx_int]

        new_in = None
        if prunable:
            new_in = _get_module_at_path(model, prunable[-1][0]).out_features

        out_par[out_idx_int] = _update_output_layer(
            out_mod, prev_keep, in_features_override=new_in
        )

    print("  LRMF pruning complete.")
    return model
