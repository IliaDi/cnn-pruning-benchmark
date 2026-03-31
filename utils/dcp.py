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
    apply_structural_pruning,
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


def _build_masks_local(
    scores: Dict[str, torch.Tensor],
    pruning_ratio: float,
) -> Dict[str, torch.Tensor]:
    """Per-layer keep-masks for DCP (local ranking)."""
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


def apply_dcp_pruning(
    model: nn.Module,
    dataloader,
    target_ratio: float,
    scope: str = "local",
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

    # Only Conv2d scores (FC layers are not output-pruned).
    all_scores: Dict[str, torch.Tensor] = {
        name: s for name, s in scores.items()
    }

    if not all_scores:
        print("  DCP WARNING: no prunable scores; returning model unchanged.")
        return model

    print(f"  DCP: building keep-masks (local, removing {target_ratio:.0%} of channels)...")
    masks = _build_masks_local(all_scores, pruning_ratio=target_ratio)

    print("  DCP: applying structural pruning...")
    model = apply_structural_pruning(model, masks)
    print("  DCP pruning complete.")
    return model

