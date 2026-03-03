"""
aofp.py – AOFP (Approximated Oracle Filter Pruning) for VGG-16 / CIFAR-10.

Ding et al. (2019), "Approximated Oracle Filter Pruning for Destructive CNN
Width Optimization" (ICML 2019).

This implementation:
- Scores conv filters via **Damage Isolation**: how much the next conv layer’s
  output changes when a filter is randomly ablated.
- Uses global or per-layer ranking across conv and hidden linear layers
  (with a weight‑magnitude fallback when there is no next conv).
- Applies one‑shot **structural** pruning using helpers from `utils.apoz`;
  fine‑tuning is handled externally by the shared training utilities.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from utils.apoz import (
    get_prunable_layers,
    prune_conv2d_layer,
    prune_linear_layer,
    _update_output_layer,
    _get_module_at_path,
)


# ─────────────────────────────────────────────────────────────────────────────
# Conv layer pairs (for Damage Isolation)
# ─────────────────────────────────────────────────────────────────────────────

def _get_conv_layer_pairs(model: nn.Module) -> List[Tuple[str, nn.Conv2d, str, nn.Conv2d]]:
    """Return consecutive (conv_i, next_conv) pairs inside `model.features`."""
    pairs: List[Tuple[str, nn.Conv2d, str, nn.Conv2d]] = []
    if not hasattr(model, "features") or not isinstance(model.features, nn.Sequential):
        return pairs

    children = list(model.features.named_children())
    convs = [(name, mod) for name, mod in children if isinstance(mod, nn.Conv2d)]
    for (name_i, conv_i), (name_next, conv_next) in zip(convs[:-1], convs[1:]):
        pairs.append((f"features.{name_i}", conv_i, f"features.{name_next}", conv_next))
    return pairs


def _forward_from_conv_to_next(
    x: torch.Tensor,
    model: nn.Module,
    conv_name: str,
    next_conv_name: str,
) -> torch.Tensor:
    """Forward `x` from just after conv_name up to and including next_conv_name."""
    conv_idx = int(conv_name.split(".")[-1])
    next_idx = int(next_conv_name.split(".")[-1])

    out = x
    for idx in range(conv_idx + 1, next_idx + 1):
        out = model.features[idx](out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Damage Isolation scoring
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _score_filters_damage_isolation(
    model: nn.Module,
    dataloader,
    device: torch.device,
    n_ablation_rounds: int = 20,
    limit_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute AOFP Damage Isolation scores for each conv filter.

    For each conv layer i:
      - Collect conv_i outputs on calibration data.
      - In each ablation round, randomly zero half of the channels, forward
        to the next conv, and measure normalised output deviation.
      - Average damage per filter over all rounds.
    """
    model.eval()
    pairs = _get_conv_layer_pairs(model)
    if not pairs:
        return {}

    damage_sum: Dict[str, torch.Tensor] = {}
    damage_count: Dict[str, torch.Tensor] = {}
    for name_i, conv_i, _, _ in pairs:
        C = conv_i.out_channels
        damage_sum[name_i] = torch.zeros(C, device="cpu")
        damage_count[name_i] = torch.zeros(C, device="cpu")

    conv_outputs: Dict[str, torch.Tensor] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    for name_i, conv_i, _, _ in pairs:
        def make_hook(layer_name: str):
            def hook_fn(module, inp, out):
                conv_outputs[layer_name] = out.detach()
            return hook_fn
        hooks.append(conv_i.register_forward_hook(make_hook(name_i)))

    try:
        for batch_idx, (images, _) in enumerate(dataloader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break

            images = images.to(device)
            _ = model(images)

            for name_i, conv_i, name_next, _ in pairs:
                if name_i not in conv_outputs:
                    continue
                feat = conv_outputs[name_i]  # (B, C, H, W)
                B, C, _, _ = feat.shape

                ref = _forward_from_conv_to_next(feat, model, name_i, name_next)
                ref_norm_sq = (ref ** 2).sum(dim=(1, 2, 3)).clamp(min=1e-10)  # (B,)

                for _round in range(n_ablation_rounds):
                    n_ablate = max(1, C // 2)
                    idxs = torch.randperm(C, device=feat.device)[:n_ablate]

                    ablated = feat.clone()
                    ablated[:, idxs, :, :] = 0.0
                    ref_abl = _forward_from_conv_to_next(ablated, model, name_i, name_next)

                    diff_sq = ((ref - ref_abl) ** 2).sum(dim=(1, 2, 3))  # (B,)
                    t = (diff_sq / ref_norm_sq).mean().item()

                    for j in idxs.tolist():
                        damage_sum[name_i][j] += t
                        damage_count[name_i][j] += 1
    finally:
        for h in hooks:
            h.remove()

    scores: Dict[str, torch.Tensor] = {}
    for name_i, conv_i, _, _ in pairs:
        C = conv_i.out_channels
        count = damage_count[name_i].clamp(min=1)
        s = (damage_sum[name_i] / count).reshape(C)
        scores[name_i] = s
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Fallback scoring and masks
# ─────────────────────────────────────────────────────────────────────────────

def _score_by_weight_magnitude(module: nn.Module) -> torch.Tensor:
    """Simple L1 magnitude scoring for conv/linear layers."""
    if isinstance(module, nn.Conv2d):
        return module.weight.data.abs().sum(dim=(1, 2, 3)).cpu()
    if isinstance(module, nn.Linear):
        return module.weight.data.abs().sum(dim=1).cpu()
    raise TypeError(f"AOFP: unsupported module type {type(module)}")


def _build_masks_local(scores: Dict[str, torch.Tensor], pruning_ratio: float) -> Dict[str, torch.Tensor]:
    masks: Dict[str, torch.Tensor] = {}
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


def _build_masks_global(scores: Dict[str, torch.Tensor], pruning_ratio: float) -> Dict[str, torch.Tensor]:
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
# Structural pruning (reuse APoZ helpers)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_structural_pruning(model: nn.Module, masks: Dict[str, torch.Tensor]) -> nn.Module:
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

    # Resize final classification layer input
    all_linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if all_linears:
        out_name = all_linears[-1]
        out_parts = out_name.split(".")
        out_par = model
        for p in out_parts[:-1]:
            out_par = getattr(out_par, p)
        out_idx = int(out_parts[-1])
        out_mod = out_par[out_idx]

        new_in = None
        if prunable:
            new_in = _get_module_at_path(model, prunable[-1][0]).out_features

        out_par[out_idx] = _update_output_layer(out_mod, prev_keep, in_features_override=new_in)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def apply_aofp_pruning(
    model: nn.Module,
    dataloader,
    target_ratio: float,
    scope: str = "global",
    device: Optional[torch.device] = None,
    limit_batches: Optional[int] = None,
    n_ablation_rounds: int = 20,
) -> nn.Module:
    """AOFP Damage Isolation + structural pruning."""
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")

    if device is None:
        device = next(model.parameters()).device

    print("  AOFP: computing Damage Isolation scores...")
    di_scores = _score_filters_damage_isolation(
        model,
        dataloader,
        device,
        n_ablation_rounds=n_ablation_rounds,
        limit_batches=limit_batches,
    )

    prunable = get_prunable_layers(model)
    all_scores: Dict[str, torch.Tensor] = {}
    for name, module in prunable:
        if name in di_scores:
            all_scores[name] = di_scores[name]
        else:
            all_scores[name] = _score_by_weight_magnitude(module)

    if not all_scores:
        print("  AOFP WARNING: no scores computed; returning model unchanged.")
        return model

    print(f"  AOFP: building masks (scope={scope}, target={target_ratio:.0%} removed)...")
    if scope == "global":
        masks = _build_masks_global(all_scores, pruning_ratio=target_ratio)
    else:
        masks = _build_masks_local(all_scores, pruning_ratio=target_ratio)

    print("  AOFP: applying structural pruning...")
    model = _apply_structural_pruning(model, masks)
    print("  AOFP pruning complete.")
    return model

