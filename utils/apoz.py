"""
APoZ (Average Percentage of Zeros) Pruning
Based on: "Network Trimming: A Data-Driven Neuron Pruning Approach
           towards Efficient Deep Architectures" (Hu et al., 2016)

"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_output_layer(name: str, model: nn.Module) -> bool:
    """True iff `name` is the last nn.Linear in the model."""
    all_linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    return bool(all_linears) and name == all_linears[-1]


def _get_module_at_path(model: nn.Module, name: str) -> nn.Module:
    """Return the live module at dotted path (e.g. 'classifier.3')."""
    m = model
    for part in name.split("."):
        m = getattr(m, part)
    return m


def _find_next_relu(
    named_list: List[Tuple[str, nn.Module]],
    current_idx: int,
) -> Tuple[Optional[str], Optional[nn.Module]]:
    """
    Scan forward from current_idx for the first nn.ReLU that is a sibling
    (same parent prefix) of the current module.  Looks at most 3 steps ahead.
    Returns (name, module) or (None, None).
    """
    parent_prefix = ".".join(named_list[current_idx][0].split(".")[:-1])
    for j in range(current_idx + 1, min(current_idx + 4, len(named_list))):
        cand_name, cand_mod = named_list[j]
        cand_prefix = ".".join(cand_name.split(".")[:-1])
        if isinstance(cand_mod, nn.ReLU) and cand_prefix == parent_prefix:
            return cand_name, cand_mod
    return None, None


def get_prunable_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Return (name, module) pairs for all Conv2d and hidden Linear layers in
    forward order, excluding the final classification layer.

    Paper: APoZ is computed (and pruning applied) to all layers "except for
    the last one" (Table 1 footnote).
    """
    prunable = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if _is_output_layer(name, model):
                continue
            prunable.append((name, module))
    return prunable


# ─────────────────────────────────────────────────────────────────────────────
# APoZ computation
# ─────────────────────────────────────────────────────────────────────────────

class APoZCalculator:
    """
    Computes per-channel APoZ scores (Hu et al. 2016, Eq. 1).

    Measurement is taken from POST-ReLU activations by hooking the nn.ReLU
    module that immediately follows each Conv2d or hidden Linear layer.
    The final classification layer is excluded.

    For each hooked layer we accumulate:
        zeros[c] — cumulative count of zero activations for channel c
        total    — total (samples × spatial positions) observations

    APoZ[c] = zeros[c] / total  ∈ [0, 1]
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self._hooks: List = []
        self._stats: Dict[str, Dict] = {}   # layer_name → {zeros, total}
        self._register_hooks()

    # ── Hook registration ─────────────────────────────────────────────────────

    def _register_hooks(self):
        named = list(self.model.named_modules())
        for idx, (name, module) in enumerate(named):
            if isinstance(module, nn.Conv2d):
                self._attach(named, idx, name, is_conv=True)
            elif isinstance(module, nn.Linear) and not _is_output_layer(name, self.model):
                self._attach(named, idx, name, is_conv=False)

    def _attach(self, named, idx, layer_name, is_conv):
        """Try to hook the downstream ReLU; fall back to hooking the layer itself."""
        if layer_name in self._stats:
            return  # already registered
        self._stats[layer_name] = {"zeros": None, "total": 0}

        relu_name, relu_mod = _find_next_relu(named, idx)
        if relu_mod is not None:
            # Hook post-ReLU output — exact match to paper's definition
            def make_relu_hook(lname, conv):
                def hook(mod, inp, out):
                    self._accumulate(lname, out.detach(), is_conv=conv)
                return hook
            h = relu_mod.register_forward_hook(make_relu_hook(layer_name, is_conv))
        else:
            # Fallback: hook the layer itself and clamp
            # ReLU(x) == 0  ⟺  x ≤ 0, so clamp(x, min=0) gives same zero-count
            def make_direct_hook(lname, conv):
                def hook(mod, inp, out):
                    clamped = torch.clamp(out.detach(), min=0.0)
                    self._accumulate(lname, clamped, is_conv=conv)
                return hook
            h = named[idx][1].register_forward_hook(make_direct_hook(layer_name, is_conv))

        self._hooks.append(h)

    # ── Statistics accumulation ───────────────────────────────────────────────

    def _accumulate(self, layer_name: str, act: torch.Tensor, is_conv: bool):
        """
        Update zero counts for `layer_name`.

        Conv tensors: shape (B, C, H, W) — sum zeros over B, H, W → per-channel.
        Linear tensors: shape (B, C)     — sum zeros over B          → per-neuron.

        'total' counts the denominator N×M from Eq. 1:
            Conv:   M = H×W spatial positions, accumulated over N samples.
            Linear: M = 1,   accumulated over N samples.
        """
        s = self._stats[layer_name]
        if is_conv and act.dim() == 4:
            B, C, H, W = act.shape
            zeros_ch = (act == 0).float().sum(dim=(0, 2, 3))  # (C,)
            total    = B * H * W
        else:
            zeros_ch = (act == 0).float().sum(dim=0)           # (C,)
            total    = act.size(0)

        if s["zeros"] is None:
            s["zeros"] = zeros_ch.cpu()
        else:
            s["zeros"] += zeros_ch.cpu()
        s["total"] += total

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self):
        for s in self._stats.values():
            s["zeros"] = None
            s["total"] = 0

    def compute(
        self,
        dataloader,
        limit_batches: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward-pass data and return per-layer APoZ score tensors.

        Args:
            dataloader:    Calibration data (validation / test split).
            limit_batches: Cap on batches processed (quick-test mode).

        Returns:
            Dict[layer_name, Tensor of shape [num_channels]] with values in [0,1].
        """
        self.reset()
        self.model.eval()
        with torch.no_grad():
            for i, (imgs, _) in enumerate(dataloader):
                if limit_batches is not None and i >= limit_batches:
                    break
                self.model(imgs.to(self.device))

        return {
            name: s["zeros"] / s["total"]
            for name, s in self._stats.items()
            if s["zeros"] is not None and s["total"] > 0
        }

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Threshold & mask
# ─────────────────────────────────────────────────────────────────────────────

def compute_threshold(
    apoz: torch.Tensor,
    target_ratio: float,
    scope: str = "local",
) -> float:
    """
    Compute the APoZ threshold that removes exactly `target_ratio` of channels
    per layer (thesis standardised-ratio protocol).

    Args:
        apoz:         Per-channel APoZ scores, shape [num_channels].
        target_ratio: Fraction of channels to REMOVE (0.3 → remove 30%).
        scope:        "local" = per-layer independent threshold (default).

    Returns:
        Scalar threshold T; channels with APoZ > T are pruned.
    """
    percentile = 1.0 - target_ratio   # e.g. target_ratio=0.3 → 70th percentile
    return torch.quantile(apoz, percentile).item()


def compute_threshold_paper(apoz: torch.Tensor) -> float:
    """
    Paper-original threshold (Hu et al. 2016, Sec 3.2):
        T = mean(APoZ) + std(APoZ)

    Kept for ablation / reference.  NOT used in the main benchmarking loop
    because it does not guarantee a fixed compression ratio.
    """
    return (apoz.mean() + apoz.std()).item()


def get_prune_mask(
    apoz: torch.Tensor,
    threshold: float,
    layer_name: str = "",
) -> torch.Tensor:
    """
    Build a boolean keep-mask aligned with the paper's criterion.

    → channels with APoZ > threshold are pruned (strict inequality).
    → channels with APoZ ≤ threshold are kept.

    Args:
        apoz:       APoZ scores, shape [num_channels].
        threshold:  Scalar threshold.
        layer_name: Optional string for console logging.

    Returns:
        Boolean tensor of shape [num_channels]; True = keep.
    """
    mask    = apoz <= threshold
    n_kept  = mask.sum().item()
    n_total = len(apoz)
    tag     = f"[{layer_name}] " if layer_name else ""
    print(
        f"    {tag}threshold={threshold:.4f}  "
        f"kept={n_kept}/{n_total} ({100*n_kept/n_total:.1f}%)  "
        f"pruned={n_total-n_kept}/{n_total} ({100*(n_total-n_kept)/n_total:.1f}%)"
    )
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Structural weight surgery
# ─────────────────────────────────────────────────────────────────────────────

def prune_conv2d_layer(
    module: nn.Conv2d,
    keep_out: torch.Tensor,
    keep_in: Optional[torch.Tensor] = None,
) -> nn.Conv2d:
    """
    Return a new Conv2d with output channels restricted to `keep_out` and,
    optionally, input channels restricted to `keep_in` (propagated from the
    previous layer's output mask).
    """
    out_idx = torch.where(keep_out)[0].tolist()
    in_idx  = torch.where(keep_in)[0].tolist() if keep_in is not None else None

    new = nn.Conv2d(
        in_channels  = len(in_idx) if in_idx is not None else module.in_channels,
        out_channels = len(out_idx),
        kernel_size  = module.kernel_size,
        stride       = module.stride,
        padding      = module.padding,
        dilation     = module.dilation,
        groups       = module.groups,
        bias         = module.bias is not None,
    )
    w = module.weight.data[out_idx]               # (kept_out, in, kH, kW)
    new.weight.data = (w[:, in_idx] if in_idx is not None else w).clone()
    if module.bias is not None:
        new.bias.data = module.bias.data[out_idx].clone()
    return new


def prune_linear_layer(
    module: nn.Linear,
    keep_out: torch.Tensor,
    keep_in: Optional[torch.Tensor] = None,
    in_features_override: Optional[int] = None,
) -> nn.Linear:
    """
    Return a new hidden Linear with output neurons restricted to `keep_out`
    and input features trimmed to match the previous (already-replaced) layer.

    Args:
        module:              Original Linear.
        keep_out:            Boolean [out_features] — which outputs to keep.
        keep_in:             Boolean [in_features]  — which inputs were kept
                             in the previous layer (for weight column selection).
        in_features_override: Exact in_features of the replacement layer when
                              the previous layer was already structurally updated.
    """
    out_idx = torch.where(keep_out)[0].tolist()
    in_idx  = torch.where(keep_in)[0].tolist() if keep_in is not None else None
    num_in  = (in_features_override if in_features_override is not None
               else (len(in_idx) if in_idx is not None else module.in_features))

    new = nn.Linear(num_in, len(out_idx), bias=module.bias is not None)
    w   = module.weight.data[out_idx]             # (kept_out, original_in)
    new.weight.data = (w[:, in_idx] if in_idx is not None else w).clone()
    if module.bias is not None:
        new.bias.data = module.bias.data[out_idx].clone()
    return new


def _update_output_layer(
    module: nn.Linear,
    keep_in: Optional[torch.Tensor],
    in_features_override: Optional[int] = None,
) -> nn.Linear:
    """
    Resize the output (classification) layer's INPUT dimension only.

    Paper: the final layer's output neurons (num_classes) are NEVER pruned.
    We only update in_features so the model remains runnable after the
    preceding hidden layer was structurally shrunk.
    """
    in_idx = torch.where(keep_in)[0].tolist() if keep_in is not None else None
    num_in = (in_features_override if in_features_override is not None
              else (len(in_idx) if in_idx is not None else module.in_features))

    new = nn.Linear(num_in, module.out_features, bias=module.bias is not None)
    w   = module.weight.data                      # (num_classes, original_in)
    new.weight.data = (w[:, in_idx] if in_idx is not None else w.clone())
    if module.bias is not None:
        new.bias.data = module.bias.data.clone()
    return new


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def apply_apoz_pruning(
    model: nn.Module,
    dataloader,
    target_ratio: float,
    scope: str = "local",
    device: Optional[torch.device] = None,
    limit_batches: Optional[int] = None,
) -> nn.Module:
    """
    Apply APoZ-based structural filter pruning to *model*.

    Implements the pipeline from Hu et al. (2016), Sec 3.2, adapted for the
    thesis's standardised fixed-ratio benchmarking protocol:

      1. Calibration forward pass → accumulate post-ReLU zero-activation stats.
      2. Compute per-layer APoZ scores (Eq. 1).
      3. Per-layer threshold = (1 − target_ratio) quantile of APoZ scores.
      4. Build boolean keep-masks (APoZ > threshold → prune).
      5. Structurally remove pruned channels in forward order, propagating
         dimension changes so consecutive layers remain consistent.
      6. Resize the final classification layer's input (no output neuron pruning).

    Args:
        model:         PyTorch model (VGG-16).
        dataloader:    Calibration loader (test/val; no augmentation).
        target_ratio:  Fraction of channels to remove per layer (0.3 → 30%).
        scope:         "local" = independent per-layer threshold (default).
        device:        Computation device; inferred from model parameters if None.
        limit_batches: Cap on calibration batches (quick-test mode).

    Returns:
        The structurally pruned model (modified in-place and returned).
    """
    if device is None:
        device = next(model.parameters()).device

    # ── Steps 1–2: APoZ scores ────────────────────────────────────────────────
    print("  Computing APoZ scores...")
    calc     = APoZCalculator(model, device)
    apoz_dict = calc.compute(dataloader, limit_batches=limit_batches)
    calc.remove_hooks()

    # ── Steps 3–4: thresholds + keep-masks ───────────────────────────────────
    prunable = get_prunable_layers(model)   # excludes final classifier layer
    print(f"  Building pruning masks (target: {target_ratio:.0%} removed per layer)...")

    masks: Dict[str, torch.Tensor] = {}
    for name, module in prunable:
        if name not in apoz_dict:
            print(f"    [{name}] WARNING: no APoZ scores — skipping this layer.")
            continue
        thresh     = compute_threshold(apoz_dict[name], target_ratio, scope)
        masks[name] = get_prune_mask(apoz_dict[name], thresh, layer_name=name)

    # ── Steps 5–6: structural weight surgery in forward order ─────────────────
    #
    # prev_keep tracks the boolean keep-mask for the OUTPUTS of the most recently
    # processed layer.  It is used to trim the INPUT side of the next layer.
    prev_keep: Optional[torch.Tensor] = None

    for i, (name, module) in enumerate(prunable):

        # Navigate to parent container and integer index
        parts  = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        layer_idx = int(parts[-1])

        # ── Conv2d ────────────────────────────────────────────────────────────
        if isinstance(module, nn.Conv2d):
            keep_out = masks.get(name, torch.ones(module.out_channels, dtype=torch.bool))
            parent[layer_idx] = prune_conv2d_layer(module, keep_out, keep_in=prev_keep)
            prev_keep = keep_out

        # ── Hidden Linear ─────────────────────────────────────────────────────
        elif isinstance(module, nn.Linear):
            keep_out  = masks.get(name, torch.ones(module.out_features, dtype=torch.bool))

            # Does this Linear immediately follow a Conv2d?
            # Input is a flattened spatial feature map: in_features = C × H × W.
            # The channel-level prev_keep mask must be expanded to all H×W positions.
            prev_name, prev_orig = prunable[i - 1] if i > 0 else (None, None)
            follows_conv = prev_orig is not None and isinstance(prev_orig, nn.Conv2d)

            if follows_conv and prev_keep is not None:
                # prev_orig.out_channels = channel count BEFORE pruning
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
                new_module.weight.data = module.weight.data[out_idx][:, flat_idx].clone()
                if module.bias is not None:
                    new_module.bias.data = module.bias.data[out_idx].clone()

            else:
                # Linear → Linear: fetch live out_features from the already-replaced
                # previous layer so dimensions are guaranteed to match.
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

    # ── Step 6: update output (classification) layer's input dimension ─────────
    # The output layer is excluded from `prunable`, so we handle it separately.
    # We ONLY update in_features; all num_classes output neurons are kept intact.
    all_linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if all_linears:
        out_name  = all_linears[-1]
        out_parts = out_name.split(".")
        out_par   = model
        for p in out_parts[:-1]:
            out_par = getattr(out_par, p)
        out_idx_int = int(out_parts[-1])
        out_mod     = out_par[out_idx_int]

        # in_features = out_features of the last prunable layer (already replaced)
        new_in = None
        if prunable:
            new_in = _get_module_at_path(model, prunable[-1][0]).out_features

        out_par[out_idx_int] = _update_output_layer(out_mod, prev_keep,
                                                     in_features_override=new_in)

    print("  Pruning complete.")
    return model
