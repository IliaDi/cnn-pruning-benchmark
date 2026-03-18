"""
dcp.py – DCP (Discrimination-aware Channel Pruning) for VGG-16 / CIFAR-10.

Zhuang et al. (2018), "Discrimination-aware Channel Pruning for Deep Neural
Networks" (NeurIPS 2018).

This implementation follows the thesis one-shot pruning protocol:
- Attach auxiliary classifiers at intermediate conv layers.
- Compute a joint loss L = L_f + lambda_disc * L_aux and collect gradients.
- Use per-channel gradient magnitudes ||∂L/∂W_j||_F as importance scores.
- Build keep-masks (local or global).
- Apply hard structural pruning using shared helpers from `utils.apoz`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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


class AuxClassifier(nn.Module):
    """
    Auxiliary discrimination head attached to an intermediate conv output.

    BN → ReLU → AdaptiveAvgPool(1) → Flatten → Linear(C, num_classes)
    """

    def __init__(self, in_channels: int, num_classes: int = 10):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def _get_aux_attach_points(
    model: nn.Module,
    n_stages: int = 2,
) -> List[Tuple[str, nn.Conv2d]]:
    """
    Select evenly spaced conv layers in `model.features` to attach aux heads.
    Returns a list of (module_name, conv_module) pairs, where names match
    VGG feature indexing (e.g. "features.9").
    """
    if not hasattr(model, "features") or not isinstance(model.features, nn.Sequential):
        return []

    convs: List[Tuple[str, nn.Conv2d]] = []
    for idx, child in enumerate(model.features):
        if isinstance(child, nn.Conv2d):
            convs.append((f"features.{idx}", child))

    if len(convs) < n_stages:
        return []

    attach: List[Tuple[str, nn.Conv2d]] = []
    for i in range(1, n_stages + 1):
        pos = int(len(convs) * i / (n_stages + 1))
        pos = min(pos, len(convs) - 1)
        attach.append(convs[pos])

    return attach


def _compute_gradient_importance(
    model: nn.Module,
    dataloader,
    device: torch.device,
    num_classes: int = 10,
    n_stages: int = 2,
    lambda_disc: float = 1.0,
    limit_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-output-channel gradient magnitude importance scores.

    Importance per output channel j:
        score[j] = ||∂L/∂W_j||_F
    where L = L_final + lambda_disc * L_aux over aux classifier heads.
    """
    attach_points = _get_aux_attach_points(model, n_stages=n_stages)
    if not attach_points:
        print("  DCP WARNING: no attachment points found.")
        return {}

    attach_names = [name for name, _ in attach_points]

    aux_classifiers: Dict[str, AuxClassifier] = {}
    for name, conv in attach_points:
        aux_classifiers[name] = AuxClassifier(conv.out_channels, num_classes).to(device)

    conv_layers: List[Tuple[str, nn.Conv2d]] = []
    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        for idx, child in enumerate(model.features):
            if isinstance(child, nn.Conv2d):
                conv_layers.append((f"features.{idx}", child))

    if not conv_layers:
        return {}

    grad_accum: Dict[str, torch.Tensor] = {
        name: torch.zeros(conv.out_channels, device="cpu") for name, conv in conv_layers
    }

    n_batches_processed = 0
    model.train()

    for batch_idx, (images, labels) in enumerate(dataloader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        for aux in aux_classifiers.values():
            aux.zero_grad(set_to_none=True)

        captured: Dict[str, torch.Tensor] = {}
        hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Hook intermediate conv outputs that feed the auxiliary heads.
        for name in attach_names:
            idx = int(name.split(".")[-1])
            mod = model.features[idx]

            def make_hook(layer_name: str):
                def hook_fn(module, inp, out):
                    captured[layer_name] = out
                return hook_fn

            hooks.append(mod.register_forward_hook(make_hook(name)))

        logits = model(images)

        for h in hooks:
            h.remove()

        # Final classification loss
        loss_final = F.cross_entropy(logits, labels)

        # Auxiliary discrimination loss
        loss_aux = torch.tensor(0.0, device=device)
        for name in attach_names:
            if name not in captured:
                continue
            aux_logits = aux_classifiers[name](captured[name])
            loss_aux = loss_aux + F.cross_entropy(aux_logits, labels)

        if len(attach_names) > 0:
            loss_aux = loss_aux / len(attach_names)

        loss = loss_final + lambda_disc * loss_aux
        loss.backward()

        # Collect per-channel gradient magnitudes for all conv layers.
        for name, conv in conv_layers:
            if conv.weight.grad is None:
                continue
            grad = conv.weight.grad.data
            per_filter_grad = grad.view(grad.size(0), -1).norm(dim=1)  # (C_out,)
            grad_accum[name] += per_filter_grad.detach().cpu()

        n_batches_processed += 1

    # Average over batches
    if n_batches_processed == 0:
        return {}

    return {name: acc / n_batches_processed for name, acc in grad_accum.items()}


def _build_masks_local(scores: Dict[str, torch.Tensor], pruning_ratio: float) -> Dict[str, torch.Tensor]:
    masks: Dict[str, torch.Tensor] = {}
    for name, s in scores.items():
        n_out = len(s)
        n_prune = max(0, int(pruning_ratio * n_out))
        n_keep = max(1, n_out - n_prune)
        _, top_idx = torch.topk(s, n_keep)  # higher grad magnitude => keep
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
    # Per-layer min-max normalization prevents heterogeneous score sources
    # (this is important even when DCP scores come from gradients).
    normalized: Dict[str, torch.Tensor] = {}
    for name, s in scores.items():
        s_min = s.min()
        s_max = s.max()
        if (s_max - s_min) > 1e-12:
            normalized[name] = (s - s_min) / (s_max - s_min)
        else:
            normalized[name] = torch.ones_like(s)

    layer_names = list(normalized.keys())
    all_scores = torch.cat([normalized[n] for n in layer_names])
    n_total = len(all_scores)
    n_prune = max(1, int(pruning_ratio * n_total))

    prune_set = set(torch.argsort(all_scores, descending=False)[:n_prune].tolist())

    masks: Dict[str, torch.Tensor] = {}
    offset = 0
    for name in layer_names:
        s = normalized[name]
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


def _apply_structural_pruning(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
) -> nn.Module:
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

    # Update classification output layer input dim only
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

        out_par[out_idx_int] = _update_output_layer(out_mod, prev_keep, in_features_override=new_in)

    return model


def apply_dcp_pruning(
    model: nn.Module,
    dataloader,
    target_ratio: float,
    scope: str = "global",
    device: Optional[torch.device] = None,
    limit_batches: Optional[int] = None,
    num_classes: int = 10,
    n_stages: int = 2,
    lambda_disc: float = 1.0,
) -> nn.Module:
    """DCP gradient-magnitude importance + one-shot structural pruning."""
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")

    if device is None:
        device = next(model.parameters()).device

    print(f"  DCP: computing gradient-based importance scores (lambda={lambda_disc}, stages={n_stages})...")
    scores = _compute_gradient_importance(
        model,
        dataloader,
        device,
        num_classes=num_classes,
        n_stages=n_stages,
        lambda_disc=lambda_disc,
        limit_batches=limit_batches,
    )
    if not scores:
        print("  DCP WARNING: no scores computed; returning model unchanged.")
        return model

    # Filter to prunable names (conv + hidden linear)
    prunable = get_prunable_layers(model)
    all_scores: Dict[str, torch.Tensor] = {}
    for name, module in prunable:
        if name in scores:
            all_scores[name] = scores[name]
        elif isinstance(module, nn.Linear):
            # Fallback for hidden linear layers when DI gradients are unavailable
            all_scores[name] = module.weight.data.abs().sum(dim=1).cpu()

    if not all_scores:
        print("  DCP WARNING: no prunable scores; returning model unchanged.")
        return model

    print(f"  DCP: building keep-masks (scope={scope}, removing {target_ratio:.0%} of channels)...")
    if scope == "global":
        masks = _build_masks_global(all_scores, pruning_ratio=target_ratio)
    else:
        masks = _build_masks_local(all_scores, pruning_ratio=target_ratio)

    print("  DCP: applying structural pruning...")
    model = _apply_structural_pruning(model, masks)
    print("  DCP pruning complete.")
    return model

