"""
chip.py  –  CHIP filter pruning for VGG-16 / CIFAR-10 benchmark

Sui et al. (2021). "CHIP: CHannel Independence-based Pruning for Compact
Neural Networks."  NeurIPS 2021.  arXiv:2110.13981

Method summary
--------------
CHIP scores each filter by the *channel independence* (CI) of the feature
map it produces — an inter-channel metric that measures how linearly
dependent one channel's feature map is on all other channels in the same
layer.

Given the full set of feature maps for layer l, matricized into
    A^l  ∈  R^{C × hw}     (C channels, each flattened to hw)

the CI score for channel i is:

    CI(A^l_i)  =  ||A^l||_*  −  ||M^l_i ⊙ A^l||_*           (Eq. 3)

where  ||·||_*  is the nuclear norm (ℓ1-norm of singular values),
M^l_i  is a row-mask matrix that zeros out row i, and ⊙ is Hadamard
product (element-wise multiplication).

Intuitively:
  - High CI  →  removing channel i causes a large nuclear-norm drop
              →  channel carries unique information  →  KEEP
  - Low CI   →  channel is nearly linearly dependent on others
              →  its information is already encoded elsewhere  →  PRUNE

Key properties (paper, Sec. 3, Q3 & Q4):
  • CI is stable across input batches (Pearson r > 0.85 across 5 batches).
    5 batches of 128 images is sufficient for reliable estimation.
  • One-shot calculation is enough; further mask adjustment via learning
    does not improve results (paper, Sec. 3 Q4 & Sec. 6.5).

Integration notes
-----------------
* The original repo separates (a) feature-map extraction → .npy files,
  (b) CI calculation from .npy files, (c) model pruning with saved CI.
  Here we fuse all three steps into a single in-memory pipeline matching
  the APoZ / DropNet / HRank contract used throughout this project.
* Structural weight surgery reuses the shared helpers from utils.apoz for
  identical, tested channel-removal behaviour across all 12 methods.
* Fine-tuning is NOT performed here; it is handled externally by
  training.fine_tune_post_pruning() using the thesis's standardised
  protocol (identical for all 12 methods).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Reuse the structural weight-surgery helpers shared across all methods.
from utils.apoz import (
    get_prunable_layers,
    prune_conv2d_layer,
    prune_linear_layer,
    _update_output_layer,
    _get_module_at_path,
)


# ─────────────────────────────────────────────────────────────────────────────
# CI score computation
# ─────────────────────────────────────────────────────────────────────────────

def _nuclear_norm(mat: torch.Tensor) -> float:
    """
    Nuclear norm of a 2-D float tensor  =  sum of singular values.

    Uses torch.linalg.svdvals which is more efficient than full SVD when
    only singular values (not vectors) are needed.
    """
    return torch.linalg.svdvals(mat.float()).sum().item()


def _ci_scores_single_sample(feature_maps: torch.Tensor) -> torch.Tensor:
    """
    Compute per-channel CI scores for ONE input sample's feature maps.

    Parameters
    ----------
    feature_maps : Tensor of shape (C, H, W)
        Output of one Conv2d layer for a single image.

    Returns
    -------
    ci : Tensor of shape (C,)
        CI(A_i) = ||A||_* − ||M_i ⊙ A||_*  for each channel i.

    Algorithm (paper Eq. 3 + Algorithm 1, steps 2–4)
    -------------------------------------------------
    1. Matricize: A = reshape(feature_maps, [C, H*W])
    2. original_norm = ||A||_*
    3. For each channel i:
           zero row i  →  A_masked
           CI[i] = original_norm − ||A_masked||_*
    """
    C, H, W = feature_maps.shape
    # Matricize: A ∈ R^{C × hw}
    A = feature_maps.reshape(C, H * W).float()           # (C, hw)
    original_norm = _nuclear_norm(A)

    ci = torch.zeros(C, dtype=torch.float32)
    for i in range(C):
        A_masked = A.clone()
        A_masked[i, :] = 0.0                              # zero out row i
        ci[i] = original_norm - _nuclear_norm(A_masked)

    return ci                                             # shape (C,)


def compute_chip_scores(
    model: nn.Module,
    calib_loader,
    device: torch.device,
    limit_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-filter CHIP channel-independence scores for all Conv2d layers.

    Following the paper's calibration scheme (Sec. 4.1): we hook the ReLU
    immediately following each Conv2d (i.e., post-activation feature maps)
    to obtain  A^l, matching how the original CHIP code captures outputs via
    `calculate_feature_maps.py` (which hooks the ReLU layer in relucfg).

    For each layer we accumulate the average CI across all calibration
    samples:

        CI_avg[i] = (1/N) * Σ_t  CI(A^l_i(t))           (paper Alg. 1, step 7)

    Parameters
    ----------
    model         : VGG-16 (torchvision-style, `.features` Sequential).
    calib_loader  : DataLoader; calibration set (no augmentation recommended).
    device        : Compute device.
    limit_batches : Stop after this many batches (quick-test mode).
                    Paper uses 5 batches of 128 images → limit_batches=5.

    Returns
    -------
    scores : dict[layer_name -> 1-D float Tensor of shape (C_out,)]
        Higher CI score → more independent → more important → KEEP.
        Lower CI score  → more dependent  → less important → PRUNE.
    """
    model.eval()

    # ── Identify which module to hook for each Conv2d ─────────────────────────
    # For VGG-16 (torchvision style) the pattern inside model.features is:
    #   Conv2d → BatchNorm2d → ReLU   (indices i, i+1, i+2)
    # We hook the ReLU output to get post-activation feature maps, exactly
    # as the original CHIP code uses relucfg to hook ReLU layers.
    hook_targets: Dict[str, nn.Module] = {}   # conv_name → module to hook

    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        children = list(model.features.children())
        for idx, child in enumerate(children):
            if isinstance(child, nn.Conv2d):
                lname = f"features.{idx}"
                target = child                             # fallback: hook conv
                for j in range(idx + 1, min(idx + 4, len(children))):
                    if isinstance(children[j], nn.ReLU):
                        target = children[j]
                        break
                hook_targets[lname] = target
    else:
        # Generic fallback
        named = list(model.named_modules())
        for idx, (name, mod) in enumerate(named):
            if isinstance(mod, nn.Conv2d):
                parent = ".".join(name.split(".")[:-1])
                target = mod
                for j in range(idx + 1, min(idx + 4, len(named))):
                    cname, cmod = named[j]
                    if (isinstance(cmod, nn.ReLU) and
                            ".".join(cname.split(".")[:-1]) == parent):
                        target = cmod
                        break
                hook_targets[name] = target

    # ── Accumulate CI scores across calibration batches ───────────────────────
    # We keep a running sum of per-image CI vectors and a sample count.
    ci_accum: Dict[str, torch.Tensor] = {}    # layer_name → Tensor (C,)
    sample_counts: Dict[str, int] = {}         # layer_name → N images seen
    handles: List = []

    def make_hook(layer_name: str):
        def hook_fn(module: nn.Module, input, output: torch.Tensor):
            if output.dim() != 4:
                return
            B, C, H, W = output.shape
            if H < 2 or W < 2:
                # 1×1 spatial — nuclear norm change is trivially small;
                # assign a uniform score so the layer is skipped gracefully.
                if layer_name not in ci_accum:
                    ci_accum[layer_name] = torch.ones(C, dtype=torch.float32)
                    sample_counts[layer_name] = 1
                return

            # Compute CI per image in the batch (CPU; SVD is more stable on CPU)
            maps_cpu = output.detach().cpu()
            batch_ci = torch.zeros(C, dtype=torch.float32)
            for b in range(B):
                batch_ci += _ci_scores_single_sample(maps_cpu[b])  # (C,)

            if layer_name not in ci_accum:
                ci_accum[layer_name] = batch_ci
                sample_counts[layer_name] = B
            else:
                ci_accum[layer_name] += batch_ci
                sample_counts[layer_name] += B

        return hook_fn

    for lname, target_mod in hook_targets.items():
        handles.append(target_mod.register_forward_hook(make_hook(lname)))

    try:
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(calib_loader):
                if limit_batches is not None and batch_idx >= limit_batches:
                    break
                model(images.to(device))
    finally:
        for h in handles:
            h.remove()

    # Normalise by total number of samples seen
    return {
        name: ci_accum[name] / sample_counts[name]
        for name in ci_accum
        if sample_counts.get(name, 0) > 0
    }


# ─────────────────────────────────────────────────────────────────────────────
# Keep-mask construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_keep_masks_local(
    scores: Dict[str, torch.Tensor],
    pruning_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    Layer-wise (local) ranking — default for CHIP.

    Within each layer independently, the bottom ``pruning_ratio`` fraction
    of filters (lowest CI → most dependent → least important) are marked
    for removal.  Always keeps ≥ 1 filter per layer.

    Returns boolean keep-masks (True = keep this filter).

    This matches the original paper's Algorithm 1, step 8–9:
    "Sort {CI(A^l_i)} in ascending order; prune c^l − κ^l filters with
    the c^l − κ^l smallest CI."
    """
    masks: Dict[str, torch.Tensor] = {}
    for lname, s in scores.items():
        n_out = len(s)
        n_prune = max(0, min(int(pruning_ratio * n_out), n_out - 1))
        n_keep  = n_out - n_prune

        # argsort ascending: lowest CI first → those are pruned
        sorted_idx = torch.argsort(s, descending=False)
        keep_idx   = torch.sort(sorted_idx[n_prune:]).values  # top n_keep

        mask = torch.zeros(n_out, dtype=torch.bool)
        mask[keep_idx] = True
        masks[lname] = mask

        n_kept   = mask.sum().item()
        n_pruned = (~mask).sum().item()
        avg_ci_kept   = s[mask].mean().item()  if mask.any()  else 0.0
        avg_ci_pruned = s[~mask].mean().item() if (~mask).any() else 0.0
        print(f"    [{lname}] kept={n_kept}/{n_out} ({100*n_kept/n_out:.1f}%)  "
              f"pruned={n_pruned}/{n_out} ({100*n_pruned/n_out:.1f}%)  "
              f"avg_CI kept={avg_ci_kept:.4f} pruned={avg_ci_pruned:.4f}")
    return masks


def _build_keep_masks_global(
    scores: Dict[str, torch.Tensor],
    pruning_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    Global joint ranking across all layers.

    Rank all filters jointly by CI and prune the globally lowest-scoring
    ``pruning_ratio`` fraction.  Always keeps ≥ 1 filter per layer.
    """
    layer_names = list(scores.keys())
    all_scores  = torch.cat([scores[n] for n in layer_names])
    n_total  = len(all_scores)
    n_prune  = max(1, int(pruning_ratio * n_total))

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
        if keep.sum() == 0:                      # guard: always keep highest-CI
            keep[int(torch.argmax(s).item())] = True
        masks[lname] = keep

        n_kept   = keep.sum().item()
        n_pruned = (~keep).sum().item()
        print(f"    [{lname}] kept={n_kept}/{n_out} ({100*n_kept/n_out:.1f}%)  "
              f"pruned={n_pruned}/{n_out} ({100*n_pruned/n_out:.1f}%)")
        offset += n_out
    return masks


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def apply_chip_pruning(
    model: nn.Module,
    dataloader,
    target_ratio: float,
    scope: str = "local",
    device: Optional[torch.device] = None,
    limit_batches: Optional[int] = None,
) -> nn.Module:
    """
    Apply CHIP-based structural filter pruning to *model*.

    Pipeline (paper Algorithm 1, fused into one in-memory pass)
    -----------------------------------------------------------
    1. Forward-pass calibration data  →  post-ReLU feature maps per layer.
    2. Compute per-channel CI scores:
           CI(A^l_i) = ||A^l||_* − ||M^l_i ⊙ A^l||_*     (paper Eq. 3)
       Averaged over all calibration samples (paper Alg. 1, step 7).
    3. Build keep-masks: lowest-CI filters pruned first.
       (layer-wise or global ranking depending on ``scope``).
    4. Structurally remove pruned channels in forward order, propagating
       dimension changes so consecutive layers remain consistent.
    5. Resize the final classification layer's input dimension only
       (output neurons / num_classes are never changed).

    Differences from the original CHIP repo
    ----------------------------------------
    The original code uses a multi-step offline pipeline:
      calculate_feature_maps.py  →  .npy files per Conv2d layer
      calculate_ci.py            →  CI .npy files per layer
      prune_finetune_cifar.py    →  loads CI scores + custom sparse model

    Here we fuse these into a single in-memory pass, operating directly on
    the torchvision-style VGG-16 used throughout this project.  The
    mathematical computation is identical to the paper.

    Parameters
    ----------
    model        : VGG-16 PyTorch model (torchvision-style).
    dataloader   : Calibration DataLoader (test/val split, no augmentation).
    target_ratio : Fraction of filters to *remove* (e.g. 0.3 → remove 30%).
    scope        : ``"local"``  = layer-wise ranking (default, matches paper).
                   ``"global"`` = joint ranking across all layers.
    device       : Compute device; inferred from model parameters if None.
    limit_batches: Cap calibration batches (quick-test mode).
                   Paper uses 5 batches of 128 images → limit_batches=5.

    Returns
    -------
    model : Structurally pruned nn.Module with physically removed channels.
            No fine-tuning applied — handled externally by
            training.fine_tune_post_pruning() under the standardised protocol.

    Notes on computational cost
    ---------------------------
    CI computation requires one nuclear norm calculation per channel per
    image sample, each involving SVD on a (C × hw) submatrix.  For VGG-16
    at 224×224 input, early layers produce large feature maps (e.g.
    features.0: 64 channels × 224×224 → each norm call is SVD of 64×50176).
    This is expensive.  Use limit_batches=5 for full runs (matching the
    paper's 5-batch calibration) and limit_batches=1 for quick tests.

    Memory note: CI is computed per image on CPU to avoid OOM on GPU for
    large feature maps.  Feature maps are transferred to CPU inside the hook.
    """
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")

    if device is None:
        device = next(model.parameters()).device

    # ── Step 1–2: CHIP CI scores ──────────────────────────────────────────────
    print("  Computing CHIP scores (channel independence via nuclear norm)...")
    if limit_batches is not None:
        print(f"    (Limited to {limit_batches} calibration batches — "
              f"paper uses 5 batches of 128 images)")
    else:
        print("    (No batch limit — this may be slow for large feature maps; "
              "consider limit_batches=5)")

    scores = compute_chip_scores(model, dataloader, device, limit_batches)

    if not scores:
        raise RuntimeError(
            "CHIP: no CI scores were collected. "
            "Check that the model has Conv2d layers with spatial output (H, W ≥ 2)."
        )

    # ── Step 3: Keep-masks ────────────────────────────────────────────────────
    print(f"  Building pruning masks "
          f"(target: {target_ratio:.0%} removed, scope: {scope})...")

    if scope == "local":
        masks = _build_keep_masks_local(scores, target_ratio)
    elif scope == "global":
        masks = _build_keep_masks_global(scores, target_ratio)
    else:
        raise ValueError(f"scope must be 'local' or 'global', got '{scope}'")

    # ── Step 4: Structural weight surgery (forward order) ─────────────────────
    # get_prunable_layers returns (Conv2d + hidden Linear) excluding the final
    # classification layer — consistent with APoZ / DropNet / HRank.
    prunable = get_prunable_layers(model)
    prev_keep: Optional[torch.Tensor] = None

    for i, (name, module) in enumerate(prunable):
        parts  = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        layer_idx = int(parts[-1])

        # ── Conv2d ────────────────────────────────────────────────────────
        if isinstance(module, nn.Conv2d):
            keep_out = masks.get(
                name, torch.ones(module.out_channels, dtype=torch.bool)
            )
            parent[layer_idx] = prune_conv2d_layer(module, keep_out, keep_in=prev_keep)
            prev_keep = keep_out

        # ── Hidden Linear ─────────────────────────────────────────────────
        elif isinstance(module, nn.Linear):
            keep_out = masks.get(
                name, torch.ones(module.out_features, dtype=torch.bool)
            )

            prev_name, prev_orig = prunable[i - 1] if i > 0 else (None, None)
            follows_conv = (prev_orig is not None and isinstance(prev_orig, nn.Conv2d))

            if follows_conv and prev_keep is not None:
                # Expand channel-level mask to cover all H×W spatial positions
                # (VGG-16 AdaptiveAvgPool2d → flatten before first FC layer)
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
                    module, keep_out,
                    keep_in=prev_keep,
                    in_features_override=in_feat_live,
                )

            parent[layer_idx] = new_module
            prev_keep = keep_out

    # ── Step 5: Final classification layer (input dim only) ───────────────────
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

    print("  CHIP pruning complete.")
    return model
