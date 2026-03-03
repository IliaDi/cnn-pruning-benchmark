"""
chip.py – CHIP pruning for VGG-16 / CIFAR-10.

Sui et al. (2021), "CHIP: CHannel Independence-based Pruning for Compact
Neural Networks" (NeurIPS 2021).

This implementation:
- Scores channels by **channel independence** (nuclear-norm drop when a
  channel is masked out).
- Pools feature maps to `_POOL_SIZE×_POOL_SIZE` before SVD for efficiency
  on 224×224 inputs.
- Supports local/global ranking and performs **structural** pruning using
  the shared helpers from `utils.apoz`.
Fine-tuning is handled externally by the standard training protocol.
"""

from __future__ import annotations

from typing import Dict, List, Optional

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
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Target spatial size before SVD.  8×8 = 64 elements per channel.
# Matches the paper's implicit scale (32×32 CIFAR-10 → 8×8 at layer 3).
_POOL_SIZE: int = 8


# ─────────────────────────────────────────────────────────────────────────────
# Spatial pooling helper
# ─────────────────────────────────────────────────────────────────────────────

def _pool_feature_maps(maps: torch.Tensor) -> torch.Tensor:
    """
    Adaptive-average-pool a (B, C, H, W) tensor to (B, C, _POOL_SIZE, _POOL_SIZE).
    Returns the tensor unchanged if H and W are already ≤ _POOL_SIZE.
    """
    B, C, H, W = maps.shape
    if H <= _POOL_SIZE and W <= _POOL_SIZE:
        return maps
    return F.adaptive_avg_pool2d(maps, _POOL_SIZE)


# ─────────────────────────────────────────────────────────────────────────────
# CI score computation
# ─────────────────────────────────────────────────────────────────────────────

def _nuclear_norm(mat: torch.Tensor) -> float:
    """Nuclear norm = sum of singular values of a 2-D float tensor."""
    return torch.linalg.svdvals(mat.float()).sum().item()


def _ci_scores_single_sample(feature_maps: torch.Tensor) -> torch.Tensor:
    """
    Compute per-channel CI scores for ONE sample's (pooled) feature maps.

    Parameters
    ----------
    feature_maps : Tensor of shape (C, H, W)
        Already pooled to _POOL_SIZE × _POOL_SIZE; H*W ≤ 64.

    Returns
    -------
    ci : Tensor of shape (C,)
        CI(A_i) = ||A||_* − ||M_i ⊙ A||_*  for each channel i.

    The loop over C SVD calls is fast because hw ≤ 64 after pooling,
    so each call operates on a (C × 64) matrix at most.
    """
    C, H, W = feature_maps.shape
    A = feature_maps.reshape(C, H * W).float()   # (C, hw)
    original_norm = _nuclear_norm(A)

    ci = torch.zeros(C, dtype=torch.float32)
    for i in range(C):
        A_masked = A.clone()
        A_masked[i, :] = 0.0
        ci[i] = original_norm - _nuclear_norm(A_masked)

    return ci   # (C,)


def compute_chip_scores(
    model: nn.Module,
    calib_loader,
    device: torch.device,
    limit_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-filter CHIP channel-independence scores for all Conv2d layers.

    Hooks the ReLU immediately after each Conv2d (post-activation feature
    maps), pools them to _POOL_SIZE × _POOL_SIZE, then computes CI via
    nuclear-norm differencing averaged over all calibration samples.

    Parameters
    ----------
    model         : VGG-16 (torchvision-style).
    calib_loader  : DataLoader; no augmentation recommended.
    device        : Compute device.
    limit_batches : Stop after this many batches.

    Returns
    -------
    scores : dict[layer_name → Tensor(C_out,)]
        Higher CI → more important → KEEP.
        Lower CI  → more dependent → PRUNE.
    """
    model.eval()

    # ── Identify hook target: ReLU after each Conv2d, or Conv2d itself ───────
    hook_targets: Dict[str, nn.Module] = {}

    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        children = list(model.features.children())
        for idx, child in enumerate(children):
            if isinstance(child, nn.Conv2d):
                lname  = f"features.{idx}"
                target = child                             # fallback
                for j in range(idx + 1, min(idx + 4, len(children))):
                    if isinstance(children[j], nn.ReLU):
                        target = children[j]
                        break
                hook_targets[lname] = target
    else:
        named = list(model.named_modules())
        for idx, (name, mod) in enumerate(named):
            if isinstance(mod, nn.Conv2d):
                parent_prefix = ".".join(name.split(".")[:-1])
                target = mod
                for j in range(idx + 1, min(idx + 4, len(named))):
                    cname, cmod = named[j]
                    if (isinstance(cmod, nn.ReLU) and
                            ".".join(cname.split(".")[:-1]) == parent_prefix):
                        target = cmod
                        break
                hook_targets[name] = target

    # ── Accumulate CI scores across calibration batches ───────────────────────
    ci_sum:    Dict[str, torch.Tensor] = {}
    n_samples: Dict[str, int]          = {}
    handles:   List                    = []

    def make_hook(layer_name: str):
        def hook_fn(module, inp, output: torch.Tensor):
            if output.dim() != 4:
                return
            B, C, H, W = output.shape

            if H < 2 or W < 2:
                # 1×1 spatial: CI trivially uniform; skip to avoid degenerate SVD
                if layer_name not in ci_sum:
                    ci_sum[layer_name]    = torch.ones(C, dtype=torch.float32)
                    n_samples[layer_name] = 1
                return

            # Pool to small spatial size on GPU, then move to CPU for SVD
            pooled = _pool_feature_maps(output.detach())   # (B, C, ps, ps) on device
            pooled = pooled.cpu()

            batch_ci = torch.zeros(C, dtype=torch.float32)
            for b in range(B):
                batch_ci += _ci_scores_single_sample(pooled[b])

            if layer_name not in ci_sum:
                ci_sum[layer_name]    = batch_ci
                n_samples[layer_name] = B
            else:
                ci_sum[layer_name]    += batch_ci
                n_samples[layer_name] += B

        return hook_fn

    for lname, target_mod in hook_targets.items():
        handles.append(target_mod.register_forward_hook(make_hook(lname)))

    try:
        print(f"    Pooling feature maps to {_POOL_SIZE}×{_POOL_SIZE} before SVD "
              f"(adapts paper's 32×32 scale to our 224×224 pipeline).")
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(calib_loader):
                if limit_batches is not None and batch_idx >= limit_batches:
                    break
                print(
                    f"    Batch {batch_idx + 1}"
                    + (f"/{limit_batches}" if limit_batches else "")
                    + f"  ({images.shape[0]} images)...",
                    flush=True,
                )
                model(images.to(device))
    finally:
        for h in handles:
            h.remove()

    scores = {
        name: ci_sum[name] / n_samples[name]
        for name in ci_sum
        if n_samples.get(name, 0) > 0
    }

    for lname, s in scores.items():
        print(
            f"    [{lname}] C={len(s)}  "
            f"CI in [{s.min():.4f}, {s.max():.4f}]  "
            f"mean={s.mean():.4f}"
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
    Layer-wise (local) ranking — default for CHIP (paper Alg. 1, step 8–9).
    Prune the lowest-CI ``pruning_ratio`` fraction per layer. Keeps ≥ 1.
    """
    masks: Dict[str, torch.Tensor] = {}
    for lname, s in scores.items():
        n_out   = len(s)
        n_prune = max(0, min(int(pruning_ratio * n_out), n_out - 1))

        sorted_idx = torch.argsort(s, descending=False)   # lowest CI first → prune
        keep_idx   = torch.sort(sorted_idx[n_prune:]).values

        mask = torch.zeros(n_out, dtype=torch.bool)
        mask[keep_idx] = True
        masks[lname]   = mask

        n_kept = mask.sum().item()
        avg_kept   = s[mask].mean().item()
        avg_pruned = s[~mask].mean().item() if (~mask).any() else float("nan")
        print(
            f"    [{lname}] kept={n_kept}/{n_out} ({100*n_kept/n_out:.1f}%)  "
            f"pruned={n_out-n_kept}/{n_out}  "
            f"avg_CI kept={avg_kept:.4f}  pruned={avg_pruned:.4f}"
        )
    return masks


def _build_keep_masks_global(
    scores: Dict[str, torch.Tensor],
    pruning_ratio: float,
) -> Dict[str, torch.Tensor]:
    """Global joint ranking — prune lowest-CI fraction across all layers."""
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
        if keep.sum() == 0:
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
    1. Forward-pass calibration data → post-ReLU feature maps per Conv2d.
    2. Pool feature maps to 8×8 spatial resolution (see performance note).
    3. Compute per-channel CI scores:
           CI(A^l_i) = ||A^l||_* − ||M^l_i ⊙ A^l||_*     (paper Eq. 3)
       Averaged over all calibration samples (paper Alg. 1, step 7).
    4. Build keep-masks: lowest-CI filters pruned first.
    5. Structurally remove pruned channels, propagating dimension changes.
    6. Resize classification layer input (outputs never changed).

    Parameters
    ----------
    model        : VGG-16 PyTorch model (torchvision-style).
    dataloader   : Calibration DataLoader (test/val split, no augmentation).
    target_ratio : Fraction of filters to REMOVE (e.g. 0.3 → remove 30%).
    scope        : "local"  = layer-wise ranking (default, matches paper).
                   "global" = joint ranking across all layers.
    device       : Compute device; inferred from model parameters if None.
    limit_batches: Cap on calibration batches.

    Returns
    -------
    model : Structurally pruned nn.Module with physically removed channels.
    """
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")

    if device is None:
        device = next(model.parameters()).device

    # ── Steps 1–3: feature maps → CI scores ───────────────────────────────────
    print("  Computing CHIP scores (channel independence via nuclear norm)...")
    scores = compute_chip_scores(model, dataloader, device, limit_batches)

    if not scores:
        raise RuntimeError(
            "CHIP: no CI scores collected. "
            "Check the model has Conv2d layers with spatial output H,W >= 2."
        )

    # ── Step 4: Keep-masks ────────────────────────────────────────────────────
    print(f"  Building keep-masks "
          f"(scope={scope}, removing {target_ratio:.0%} of channels)...")

    if scope == "local":
        masks = _build_keep_masks_local(scores, target_ratio)
    elif scope == "global":
        masks = _build_keep_masks_global(scores, target_ratio)
    else:
        raise ValueError(f"scope must be 'local' or 'global', got '{scope}'")

    # ── Step 5: Structural weight surgery ─────────────────────────────────────
    prunable  = get_prunable_layers(model)
    prev_keep: Optional[torch.Tensor] = None

    for i, (name, module) in enumerate(prunable):
        parts  = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        layer_idx = int(parts[-1])

        if isinstance(module, nn.Conv2d):
            keep_out = masks.get(
                name, torch.ones(module.out_channels, dtype=torch.bool)
            )
            parent[layer_idx] = prune_conv2d_layer(module, keep_out, keep_in=prev_keep)
            prev_keep = keep_out

        elif isinstance(module, nn.Linear):
            keep_out = masks.get(
                name, torch.ones(module.out_features, dtype=torch.bool)
            )
            prev_name, prev_orig = prunable[i - 1] if i > 0 else (None, None)
            follows_conv = prev_orig is not None and isinstance(prev_orig, nn.Conv2d)

            if follows_conv and prev_keep is not None:
                spatial  = module.in_features // prev_orig.out_channels
                kept_ch  = torch.where(prev_keep)[0].tolist()
                flat_idx = torch.tensor(
                    [c * spatial + s for c in kept_ch for s in range(spatial)],
                    dtype=torch.long,
                )
                out_idx    = torch.where(keep_out)[0].tolist()
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

    # ── Step 6: Resize classification layer input ──────────────────────────────
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
