"""
aofp.py – AOFP (Approximated Oracle Filter Pruning) for VGG-16 / CIFAR-10.

Ding et al. (2019), "Approximated Oracle Filter Pruning for Destructive CNN
Width Optimization" (ICML 2019).

This implementation:
- Scores conv filters via **Damage Isolation**: how much the next layer’s
  output changes when a filter is randomly ablated.
- For the last conv layer (no following conv), DI is extended by forwarding
  through MaxPool + AvgPool + first FC layer to measure damage at the
  classifier input — faithful to the paper’s "immediately following layer".
- Uses global ranking across all Conv2d layers (no normalization needed
  since all scores are DI-based on a common scale).
- Applies one‑shot **structural** pruning using helpers from `utils.apoz`;
  fine‑tuning is handled externally by the shared training utilities.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from utils.apoz import (
    get_prunable_layers,
    apply_structural_pruning,
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


def _forward_from_last_conv_to_fc(
    x: torch.Tensor,
    model: nn.Module,
    conv_name: str,
) -> torch.Tensor:
    """
    Forward `x` from just after the last Conv2d through remaining features
    layers (MaxPool), avgpool, flatten, and the first FC layer.

    This extends Damage Isolation to the last conv layer by measuring
    damage at the immediately following layer in the network, which is
    the classifier's first Linear layer (via pool + flatten).
    """
    conv_idx = int(conv_name.split(".")[-1])

    out = x
    # Forward through remaining features layers (e.g. MaxPool after last conv)
    for idx in range(conv_idx + 1, len(model.features)):
        out = model.features[idx](out)

    # avgpool + flatten + first FC layer
    out = model.avgpool(out)
    out = torch.flatten(out, 1)
    out = model.classifier[0](out)
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

    For each conv layer i with a following conv layer:
      - Collect conv_i outputs on calibration data.
      - In each ablation round, randomly zero half of the channels, forward
        to the next conv, and measure normalised output deviation.
      - Average damage per filter over all rounds.

    For the last conv layer (no following conv), DI is extended by
    forwarding through MaxPool + AvgPool + first FC layer, measuring
    damage at the classifier input.
    """
    model.eval()
    pairs = _get_conv_layer_pairs(model)
    if not pairs:
        return {}

    # Identify the last conv layer in features (may not be in any pair as first)
    all_convs = [
        (f"features.{name}", mod)
        for name, mod in model.features.named_children()
        if isinstance(mod, nn.Conv2d)
    ]
    last_conv_name, last_conv_mod = all_convs[-1]
    paired_first_names = {name_i for name_i, _, _, _ in pairs}

    # Layers to score via DI: all paired first-conv layers + the last conv
    score_targets: Dict[str, nn.Conv2d] = {}
    for name_i, conv_i, _, _ in pairs:
        score_targets[name_i] = conv_i
    if last_conv_name not in paired_first_names:
        score_targets[last_conv_name] = last_conv_mod

    damage_sum: Dict[str, torch.Tensor] = {}
    damage_count: Dict[str, torch.Tensor] = {}
    for name, conv in score_targets.items():
        C = conv.out_channels
        damage_sum[name] = torch.zeros(C, device="cpu")
        damage_count[name] = torch.zeros(C, device="cpu")

    conv_outputs: Dict[str, torch.Tensor] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    for name, conv in score_targets.items():
        def make_hook(layer_name: str):
            def hook_fn(module, inp, out):
                conv_outputs[layer_name] = out.detach()
            return hook_fn
        hooks.append(conv.register_forward_hook(make_hook(name)))

    try:
        for batch_idx, (images, _) in enumerate(dataloader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break

            images = images.to(device)
            _ = model(images)

            # Score paired conv layers (forward to next conv)
            for name_i, conv_i, name_next, _ in pairs:
                if name_i not in conv_outputs:
                    continue
                feat = conv_outputs[name_i]  # (B, C, H, W)
                B, C, _, _ = feat.shape

                ref = _forward_from_conv_to_next(feat, model, name_i, name_next)
                ref_norm_sq = (ref ** 2).sum(dim=(1, 2, 3)).clamp(min=1e-10)

                for _round in range(n_ablation_rounds):
                    n_ablate = max(1, C // 2)
                    idxs = torch.randperm(C, device=feat.device)[:n_ablate]

                    ablated = feat.clone()
                    ablated[:, idxs, :, :] = 0.0
                    ref_abl = _forward_from_conv_to_next(ablated, model, name_i, name_next)

                    diff_sq = ((ref - ref_abl) ** 2).sum(dim=(1, 2, 3))
                    t = (diff_sq / ref_norm_sq).mean().item()

                    for j in idxs.tolist():
                        damage_sum[name_i][j] += t
                        damage_count[name_i][j] += 1

            # Score last conv layer (forward through classifier)
            if last_conv_name not in paired_first_names and last_conv_name in conv_outputs:
                feat = conv_outputs[last_conv_name]
                B, C, _, _ = feat.shape

                ref = _forward_from_last_conv_to_fc(feat, model, last_conv_name)
                # ref is (B, D) where D = first FC output features
                ref_norm_sq = (ref ** 2).sum(dim=1).clamp(min=1e-10)

                for _round in range(n_ablation_rounds):
                    n_ablate = max(1, C // 2)
                    idxs = torch.randperm(C, device=feat.device)[:n_ablate]

                    ablated = feat.clone()
                    ablated[:, idxs, :, :] = 0.0
                    ref_abl = _forward_from_last_conv_to_fc(ablated, model, last_conv_name)

                    diff_sq = ((ref - ref_abl) ** 2).sum(dim=1)
                    t = (diff_sq / ref_norm_sq).mean().item()

                    for j in idxs.tolist():
                        damage_sum[last_conv_name][j] += t
                        damage_count[last_conv_name][j] += 1
    finally:
        for h in hooks:
            h.remove()

    scores: Dict[str, torch.Tensor] = {}
    for name, conv in score_targets.items():
        C = conv.out_channels
        count = damage_count[name].clamp(min=1)
        s = (damage_sum[name] / count).reshape(C)
        scores[name] = s
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Fallback scoring and global mask building
# ─────────────────────────────────────────────────────────────────────────────

def _score_by_weight_magnitude(module: nn.Conv2d) -> torch.Tensor:
    """Simple L1 magnitude scoring for conv layers without a DI pair."""
    return module.weight.data.abs().sum(dim=(1, 2, 3)).cpu()


def _build_masks_global(
    scores: Dict[str, torch.Tensor],
    pruning_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    Build keep-masks using global ranking across all Conv2d layers.

    Raw scores are used directly (no per-layer normalization) since all
    scored layers are Conv2d.
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

    # Only score Conv2d layers (FC layers are not output-pruned).
    prunable = get_prunable_layers(model)
    all_scores: Dict[str, torch.Tensor] = {}
    for name, module in prunable:
        if isinstance(module, nn.Conv2d):
            if name in di_scores:
                all_scores[name] = di_scores[name]
            else:
                all_scores[name] = _score_by_weight_magnitude(module)

    if not all_scores:
        print("  AOFP WARNING: no scores computed; returning model unchanged.")
        return model

    print(f"  AOFP: building masks (global, target={target_ratio:.0%} removed)...")
    masks = _build_masks_global(all_scores, pruning_ratio=target_ratio)

    print("  AOFP: applying structural pruning...")
    model = apply_structural_pruning(model, masks)
    print("  AOFP pruning complete.")
    return model

