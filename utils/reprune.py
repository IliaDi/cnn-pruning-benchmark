"""
reprune.py – REPrune (Channel Pruning via Kernel Representative Selection)
for VGG-16 / CIFAR-10.

Park et al. (2024), "REPrune: Channel Pruning via Kernel Representative
Selection" (AAAI 2024).

This implementation:
- For each input channel j in each conv layer l, performs agglomerative
  clustering (Ward's method) on the kernel set K^l_j to identify groups
  of similar kernels.
- Selects filters that maximise coverage of kernel cluster representatives
  via a greedy Maximum Cluster Coverage Problem (MCP) solver.
- Applies **structural** pruning using helpers from `utils.apoz`;
  fine-tuning is handled externally by the shared training protocol.

Adaptation for our benchmark:
  The full REPrune framework integrates pruning with training via a
  concurrent training-pruning paradigm using BN scaling factors. We
  adapt it to our standardised one-shot protocol by performing kernel
  clustering and filter selection as a single scoring step on the
  pretrained baseline model's weight tensors (no calibration data needed).
  This ensures REPrune is compared on equal footing with all other methods.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
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
# Agglomerative clustering (Ward's method)
# ─────────────────────────────────────────────────────────────────────────────

def _ward_agglomerative_clustering(
    kernels: np.ndarray,
    n_clusters: int,
) -> Tuple[List[List[int]], float]:
    """
    Perform agglomerative clustering with Ward's linkage on a set of kernels.

    Parameters
    ----------
    kernels   : (N, D) array where N is the number of kernels (= number of
                filters) and D = kH * kW is the flattened kernel dimension.
    n_clusters: Target number of clusters to form.

    Returns
    -------
    clusters      : List of lists, each containing kernel indices in that cluster.
    max_linkage   : The Ward's linkage distance at the final merge step.
    """
    N = kernels.shape[0]
    if N <= n_clusters or n_clusters < 1:
        return [[i] for i in range(N)], 0.0

    try:
        from scipy.cluster.hierarchy import linkage, fcluster

        # Ward's method: monotonically non-decreasing linkage distances
        Z = linkage(kernels, method="ward")

        # fcluster with maxclust gives exactly n_clusters clusters
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")

        clusters: List[List[int]] = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            clusters[label - 1].append(i)

        clusters = [c for c in clusters if len(c) > 0]

        merge_step = N - n_clusters
        if merge_step > 0 and merge_step <= len(Z):
            max_linkage = float(Z[merge_step - 1, 2])
        else:
            max_linkage = 0.0

        return clusters, max_linkage

    except ImportError:
        return _ward_manual(kernels, n_clusters)


def _ward_manual(
    kernels: np.ndarray,
    n_clusters: int,
) -> Tuple[List[List[int]], float]:
    """
    Manual Ward's agglomerative clustering fallback (no scipy dependency).

    Starts with singleton clusters, merges the pair with the smallest Ward's
    linkage distance.
    """
    N = kernels.shape[0]
    active_clusters: Dict[int, List[int]] = {i: [i] for i in range(N)}
    centroids: Dict[int, np.ndarray] = {i: kernels[i].copy() for i in range(N)}
    next_id = N
    max_linkage = 0.0

    while len(active_clusters) > n_clusters:
        best_dist = float("inf")
        best_pair = (-1, -1)
        ids = list(active_clusters.keys())

        for ii in range(len(ids)):
            for jj in range(ii + 1, len(ids)):
                a, b = ids[ii], ids[jj]
                na = len(active_clusters[a])
                nb = len(active_clusters[b])
                diff = centroids[a] - centroids[b]
                ward_dist = (na * nb / (na + nb)) * np.dot(diff, diff)
                if ward_dist < best_dist:
                    best_dist = ward_dist
                    best_pair = (a, b)

        a, b = best_pair
        max_linkage = best_dist

        merged = active_clusters[a] + active_clusters[b]
        na = len(active_clusters[a])
        nb = len(active_clusters[b])
        new_centroid = (na * centroids[a] + nb * centroids[b]) / (na + nb)

        del active_clusters[a]
        del active_clusters[b]
        del centroids[a]
        del centroids[b]

        active_clusters[next_id] = merged
        centroids[next_id] = new_centroid
        next_id += 1

    clusters = list(active_clusters.values())
    return clusters, max_linkage


# ─────────────────────────────────────────────────────────────────────────────
# Per-channel clustering
# ─────────────────────────────────────────────────────────────────────────────

def _cluster_kernels_per_channel(
    weight: torch.Tensor,
    target_n_clusters: int,
) -> Tuple[List[List[List[int]]], float]:
    """
    For a conv layer weight tensor (C_out, C_in, kH, kW), perform
    agglomerative clustering on each input channel's kernel set.

    Returns
    -------
    all_clusters : List of length C_in; each element is a list of clusters
                   (each cluster is a list of filter indices).
    h_l          : Layer-wise linkage cut-off (max over channels).
    """
    W = weight.detach().cpu().numpy()
    C_out, C_in, kH, kW = W.shape
    D = kH * kW

    all_clusters: List[List[List[int]]] = []
    max_linkage_distances: List[float] = []

    for j in range(C_in):
        kernels_j = W[:, j, :, :].reshape(C_out, D)
        clusters_j, linkage_dist_j = _ward_agglomerative_clustering(
            kernels_j, n_clusters=target_n_clusters
        )
        all_clusters.append(clusters_j)
        max_linkage_distances.append(linkage_dist_j)

    h_l = max(max_linkage_distances) if max_linkage_distances else 0.0
    return all_clusters, h_l


# ─────────────────────────────────────────────────────────────────────────────
# Greedy Maximum Cluster Coverage (MCP) solver
# ─────────────────────────────────────────────────────────────────────────────

def _greedy_mcp_filter_selection(
    all_clusters: List[List[List[int]]],
    n_filters: int,
    n_keep: int,
) -> List[int]:
    """
    Greedy solver for the Maximum Cluster Coverage Problem.

    Returns selected filters in selection order.
    """
    C_in = len(all_clusters)

    # Map: filter_i -> list of covered cluster identifiers (j, cluster_idx).
    filter_to_clusters: Dict[int, List[Tuple[int, int]]] = {
        i: [] for i in range(n_filters)
    }
    for j in range(C_in):
        for cluster_idx, cluster_members in enumerate(all_clusters[j]):
            for filter_i in cluster_members:
                filter_to_clusters[filter_i].append((j, cluster_idx))

    covered: Set[Tuple[int, int]] = set()
    selected: List[int] = []
    selected_set: Set[int] = set()

    for _ in range(n_keep):
        best_filter = -1
        best_coverage = -1

        for i in range(n_filters):
            if i in selected_set:
                continue
            coverage = sum(1 for cid in filter_to_clusters[i] if cid not in covered)
            if coverage > best_coverage:
                best_coverage = coverage
                best_filter = i

        if best_filter == -1:
            for i in range(n_filters):
                if i not in selected_set:
                    best_filter = i
                    break
            if best_filter == -1:
                break

        selected.append(best_filter)
        selected_set.add(best_filter)

        for cid in filter_to_clusters[best_filter]:
            covered.add(cid)

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# REPrune score computation (weights only)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_reprune_scores(
    model: nn.Module,
    target_ratio: float,
) -> Dict[str, torch.Tensor]:
    """
    Compute REPrune importance scores for all Conv2d layers.

    For each layer:
      1) Cluster kernels per input channel with Ward linkage into
         (C_out - 1) clusters (max meaningful split).
      2) Run greedy MCP for C_out steps to get a full selection order.
      3) Importance: filters selected earlier get higher scores.
    """
    scores: Dict[str, torch.Tensor] = {}

    if not hasattr(model, "features") or not isinstance(model.features, nn.Sequential):
        return scores

    for idx, child in enumerate(model.features):
        if not isinstance(child, nn.Conv2d):
            continue

        name = f"features.{idx}"
        C_out = child.out_channels
        n_clusters = max(1, C_out - 1)
        n_keep = C_out

        print(
            f"    [{name}] clustering {C_out} filters into {n_clusters} "
            f"representatives per input-channel..."
        )

        all_clusters, h_l = _cluster_kernels_per_channel(
            child.weight, target_n_clusters=n_clusters
        )

        selected = _greedy_mcp_filter_selection(
            all_clusters, n_filters=C_out, n_keep=n_keep
        )

        importance = torch.zeros(C_out, dtype=torch.float32)
        for rank, filter_i in enumerate(selected):
            importance[filter_i] = float(n_keep - rank)

        scores[name] = importance

        n_covered_total = sum(
            len(cl) for cl_list in all_clusters for cl in cl_list
        )
        print(
            f"    [{name}] h^l={h_l:.4f}, total cluster slots={n_covered_total}, "
            f"selected={len(selected)}/{C_out}"
        )

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Keep-mask construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_masks_local(
    scores: Dict[str, torch.Tensor],
    pruning_ratio: float,
) -> Dict[str, torch.Tensor]:
    """Per-layer keep-masks for REPrune (local ranking)."""
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


def _build_masks_global(
    scores: Dict[str, torch.Tensor],
    pruning_ratio: float,
) -> Dict[str, torch.Tensor]:
    """Global ranking with per-layer min-max normalization."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Structural pruning (shared surgery from apoz.py)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_structural_pruning(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
) -> nn.Module:
    """Apply structural channel removal using shared surgery helpers."""
    prunable = get_prunable_layers(model)
    prev_keep: Optional[torch.Tensor] = None

    for i, (name, module) in enumerate(prunable):
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        layer_idx = int(parts[-1])

        if isinstance(module, nn.Conv2d):
            keep_out = masks.get(
                name, torch.ones(module.out_channels, dtype=torch.bool)
            )
            parent[layer_idx] = prune_conv2d_layer(
                module, keep_out, keep_in=prev_keep
            )
            prev_keep = keep_out

        elif isinstance(module, nn.Linear):
            keep_out = masks.get(
                name, torch.ones(module.out_features, dtype=torch.bool)
            )
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

    # Update output (classification) layer input dim only.
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

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def apply_reprune_pruning(
    model: nn.Module,
    dataloader,
    target_ratio: float,
    scope: str = "global",
    device: Optional[torch.device] = None,
    limit_batches: Optional[int] = None,
) -> nn.Module:
    """
    Apply REPrune-based structural filter pruning.

    Note: REPrune scores from weight structure only, so `dataloader` is
    accepted for API compatibility but is not used.
    """
    if not 0.0 < target_ratio < 1.0:
        raise ValueError(f"target_ratio must be in (0, 1), got {target_ratio}")

    if device is None:
        device = next(model.parameters()).device

    print(
        f"  REPrune: kernel clustering + MCP filter selection "
        f"(scope={scope}, target={target_ratio:.0%} removed)..."
    )

    conv_scores = _compute_reprune_scores(model, target_ratio)
    if not conv_scores:
        print("  REPrune WARNING: no scores computed; returning model unchanged.")
        return model

    # Hidden Linear layers: weight-magnitude fallback.
    prunable = get_prunable_layers(model)
    all_scores: Dict[str, torch.Tensor] = {}
    for name, module in prunable:
        if name in conv_scores:
            all_scores[name] = conv_scores[name]
        elif isinstance(module, nn.Linear):
            importance = module.weight.data.abs().sum(dim=1).cpu()
            all_scores[name] = importance
            print(f"    [{name}] linear layer — using weight magnitude fallback")

    if not all_scores:
        print("  REPrune WARNING: no prunable scores; returning model unchanged.")
        return model

    print(
        f"  REPrune: building masks (scope={scope}, target={target_ratio:.0%} removed)..."
    )
    if scope == "global":
        masks = _build_masks_global(all_scores, pruning_ratio=target_ratio)
    else:
        masks = _build_masks_local(all_scores, pruning_ratio=target_ratio)

    print("  REPrune: applying structural pruning...")
    model = _apply_structural_pruning(model, masks)
    print("  REPrune pruning complete.")
    return model

