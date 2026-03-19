import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import csv
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

from config import (
    METHODS, PRUNING_RATIOS, RESULTS_DIR, SEED, BASELINE_TRAIN_EPOCHS,
    POST_PRUNING_FINE_TUNE_EPOCHS, POST_PRUNING_FINE_TUNE_LR,
    POST_PRUNING_FINE_TUNE_MOMENTUM, POST_PRUNING_FINE_TUNE_WEIGHT_DECAY
)

from utils.metrics import (
    evaluate_accuracy,
    count_parameters,
    compute_flops,
    measure_inference_latency
)
from utils.pruning import apply_pruning_method, get_layer_info
from utils.training import train, fine_tune_post_pruning
from utils.data import get_cifar10_loaders
from models.vgg import vgg16


# Experiment Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _Tee:
    """Write to multiple file-like objects (e.g. stdout and a log file)."""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_metrics(path, metrics):
    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(path, "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)


def extract_pruning_mask(baseline_state, pruned_model):
    """
    Extract which conv output channels were kept vs removed by matching
    pruned conv biases/weights back to the baseline snapshot.
    """
    layers = {}

    for name, mod in pruned_model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue

        weight_key = f"{name}.weight"
        bias_key = f"{name}.bias"
        if weight_key not in baseline_state:
            continue

        n_orig = baseline_state[weight_key].shape[0]
        n_pruned = mod.out_channels

        if n_pruned >= n_orig:
            kept = list(range(n_orig))
        elif mod.bias is not None and bias_key in baseline_state:
            bias_orig = baseline_state[bias_key]
            bias_pruned = mod.bias.data.cpu()
            kept = []
            used = set()
            for j in range(n_pruned):
                diffs = (bias_orig - bias_pruned[j]).abs()
                for u in used:
                    diffs[u] = float("inf")
                best = int(torch.argmin(diffs).item())
                kept.append(best)
                used.add(best)
        else:
            w_orig = baseline_state[weight_key].cpu().reshape(n_orig, -1)
            w_pruned = mod.weight.data.cpu().reshape(n_pruned, -1)
            norm_orig = w_orig.norm(dim=1)
            norm_pruned = w_pruned.norm(dim=1)
            kept = []
            used = set()
            for j in range(n_pruned):
                diffs = (norm_orig - norm_pruned[j]).abs()
                for u in used:
                    diffs[u] = float("inf")
                best = int(torch.argmin(diffs).item())
                kept.append(best)
                used.add(best)

        removed = sorted(set(range(n_orig)) - set(kept))
        layers[name] = {
            "total_filters": n_orig,
            "kept_indices": sorted(kept),
            "removed_indices": removed,
            "kept_count": len(kept),
            "removed_count": len(removed),
        }

    return layers


def save_pruning_mask(exp_dir, method, scope, ratio, mask_layers):
    result = {
        "method": method,
        "scope": scope,
        "target_pruning_ratio_removed": ratio,
        "layers": mask_layers,
    }
    with open(os.path.join(exp_dir, "pruning_mask.json"), "w") as f:
        json.dump(result, f, indent=2)


def backfill_pruning_mask(
    method,
    scope,
    ratio,
    exp_dir,
    baseline_model_path,
    calib_loader,
    baseline_dir,
):
    """
    Backfill pruning_mask.json for an already-completed experiment.
    Re-runs pruning only (no fine-tuning), deterministic given SEED.
    """
    print("  Backfilling pruning_mask.json (pruning only)...")
    set_seed(SEED)

    model = vgg16(num_classes=10, pretrained=False).to(DEVICE)
    ckpt = torch.load(baseline_model_path, map_location=DEVICE)
    model_keys = set(model.state_dict().keys())
    ckpt_filtered = {k: v for k, v in ckpt.items() if k in model_keys}
    if len(ckpt_filtered) < len(ckpt):
        extra = set(ckpt.keys()) - model_keys
        print(f"  (Dropping non-model keys: {extra})")
    model.load_state_dict(ckpt_filtered, strict=True)

    # Snapshot conv params only (used to recover kept indices).
    baseline_state = {
        k: v.cpu().clone()
        for k, v in model.state_dict().items()
        if k.startswith("features.") and (k.endswith(".weight") or k.endswith(".bias"))
    }

    apply_pruning_method(
        model=model,
        method=method,
        scope=scope,
        target_ratio=ratio,
        calib_loader=calib_loader,
    )

    mask_layers = extract_pruning_mask(baseline_state, model)
    save_pruning_mask(exp_dir, method, scope, ratio, mask_layers)
    print(f"  Backfilled pruning_mask.json ({len(mask_layers)} layers)")

    del model, baseline_state


def run():
    """Run full pruning experiments."""
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Redirect all prints to a log file (and keep stdout)
    log_name = f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(RESULTS_DIR, log_name)
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_file)
    try:
        _run_impl(log_path)
    finally:
        sys.stdout = original_stdout
        log_file.close()
        print(f"Log written to {log_path}")


def _run_impl(log_path):
    """Actual experiment logic (stdout is already tee'd to log file)."""
    print(f"Log file: {log_path}")
    print("=" * 60)
    print("FULL EXPERIMENTS")
    print(f"  Device: {DEVICE}")
    print(f"  Baseline: {BASELINE_TRAIN_EPOCHS} epoch(s)")
    print(f"  Fine-tune: {POST_PRUNING_FINE_TUNE_EPOCHS} epoch(s)  |  Ratios: {PRUNING_RATIOS}")
    print("=" * 60)

    print("\n[1/4] Building data loaders...")
    import platform
    num_workers = 0 if platform.system() == "Darwin" else 4
    train_loader, test_loader, calib_loader = get_cifar10_loaders(use_augmentation=True, num_workers=num_workers)
    print(f"  Train: {len(train_loader)} batches (40k, held out from calib)")
    print(f"  Test:  {len(test_loader)} batches")
    print(f"  Calib: {len(calib_loader)} batches (10k held-out, 1k per class)")

    baseline_dir = os.path.join(RESULTS_DIR, "baseline")
    baseline_model_path = os.path.join(baseline_dir, "model.pth")
    baseline_metrics_path = os.path.join(baseline_dir, "metrics.json")
    os.makedirs(baseline_dir, exist_ok=True)

    print("\n[2/4] PHASE 1 — Baseline model")
    if os.path.exists(baseline_model_path) and os.path.exists(baseline_metrics_path):
        print("  Baseline already exists; loading metrics from disk (skip training).")
        with open(baseline_metrics_path) as f:
            baseline_metrics = json.load(f)
        baseline_acc = baseline_metrics["accuracy"]
        baseline_params = baseline_metrics["parameters"]
        baseline_flops = baseline_metrics["flops"]
        baseline_latency_ms = baseline_metrics["inference_latency_ms"]
        baseline_params_m = baseline_params / 1e6
        baseline_flops_m = baseline_flops / 1e6
        print(f"  Loaded: acc={100*baseline_acc:.2f}%  params={baseline_params/1e6:.2f}M  FLOPs={baseline_flops/1e6:.0f}M  latency={baseline_latency_ms:.3f}ms")
    else:
        print("  Loading VGG-16 (ImageNet pretrained, 10 classes)...")
        model = vgg16(num_classes=10, pretrained=True).to(DEVICE)
        print(f"  Training baseline for {BASELINE_TRAIN_EPOCHS} epoch(s)...")
        train(model, train_loader, epochs=BASELINE_TRAIN_EPOCHS, fine_tune=True, verbose=True)
        print("  Baseline training done.")

        print("  Evaluating baseline on test set...")
        baseline_acc = evaluate_accuracy(model, test_loader)
        baseline_params = count_parameters(model)
        # Save state_dict before compute_flops — thop.profile() adds total_ops/total_params to the model
        print("  Saving baseline model...")
        torch.save(model.state_dict(), baseline_model_path)
        baseline_flops = compute_flops(model)
        print(f"  Baseline results: acc={100*baseline_acc:.2f}%  params={baseline_params/1e6:.2f}M  FLOPs={baseline_flops/1e6:.0f}M")
        print("  Measuring baseline inference latency...")
        baseline_latency_ms = measure_inference_latency(model, batch_size=1)
        print(f"  Baseline inference latency: {baseline_latency_ms:.3f} ms per sample")

        print("  Saving baseline metrics...")
        baseline_metrics = {
            "accuracy": baseline_acc,
            "parameters": baseline_params,
            "flops": baseline_flops,
            "inference_latency_ms": baseline_latency_ms
        }
        save_metrics(baseline_dir, baseline_metrics)

        # Convert baseline to millions for display
        baseline_params_m = baseline_params / 1e6
        baseline_flops_m = baseline_flops / 1e6

    summary_rows = []
    total_experiments = sum(len(methods) * len(PRUNING_RATIOS) for _, methods in METHODS.items() if methods)
    exp_counter = [0]

    print(f"\n[3/4] PHASE 2 — Pruning experiments (total: {total_experiments})")
    for scope, methods in METHODS.items():
        for method in methods:
            for ratio in PRUNING_RATIOS:
                exp_counter[0] += 1
                n = exp_counter[0]
                removal_pct = ratio * 100
                retention_pct = (1 - ratio) * 100
                print(f"\n  --- Experiment {n}/{total_experiments}: {method} | {scope} | remove {removal_pct:.0f}% (retain {retention_pct:.0f}%) ---")

                exp_dir = os.path.join(
                    RESULTS_DIR, method, f"ratio_{ratio}"
                )
                exp_metrics_path = os.path.join(exp_dir, "metrics.json")
                exp_mask_path = os.path.join(exp_dir, "pruning_mask.json")
                os.makedirs(exp_dir, exist_ok=True)

                if os.path.exists(exp_metrics_path):
                    if os.path.exists(exp_mask_path):
                        print("  Experiment complete (metrics + pruning_mask exist); skipping.")
                    else:
                        print("  Experiment has metrics but no pruning_mask; backfilling mask only...")
                    with open(exp_metrics_path) as f:
                        metrics = json.load(f)
                    summary_rows.append(metrics)
                    if summary_rows:
                        with open(os.path.join(RESULTS_DIR, "summary.csv"), "w", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
                            writer.writeheader()
                            writer.writerows(summary_rows)
                    if not os.path.exists(exp_mask_path):
                        backfill_pruning_mask(
                            method=method,
                            scope=scope,
                            ratio=ratio,
                            exp_dir=exp_dir,
                            baseline_model_path=baseline_model_path,
                            calib_loader=calib_loader,
                            baseline_dir=baseline_dir,
                        )
                    print(f"  Skipped {n}/{total_experiments} (summary.csv updated)")
                    continue

                # Re-seed for reproducibility (each experiment should start from same random state)
                set_seed(SEED)

                print("  Loading baseline model...")
                # Load fine-tuned baseline model (filter out any keys added by thop.profile in old checkpoints)
                model = vgg16(num_classes=10, pretrained=False).to(DEVICE)
                ckpt = torch.load(os.path.join(baseline_dir, "model.pth"), map_location=DEVICE)
                model_keys = set(model.state_dict().keys())
                ckpt_filtered = {k: v for k, v in ckpt.items() if k in model_keys}
                if len(ckpt_filtered) < len(ckpt):
                    extra = set(ckpt.keys()) - model_keys
                    print(f"  (Dropping non-model keys: {extra})")
                model.load_state_dict(ckpt_filtered, strict=True)
                print("  Model loaded.")

                print("  Gathering pre-pruning layer info...")
                # Measure pre-pruning layer structure for logging
                pre_pruning_layer_info = get_layer_info(model)

                # Snapshot conv parameters before pruning so we can recover
                # which filters were kept/removed.
                baseline_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                    if k.startswith("features.")
                    and (k.endswith(".weight") or k.endswith(".bias"))
                }

                print(f"  Applying {method} pruning (scope={scope}, ratio={ratio})...")
                # Calibration uses held-out set (10k); test set used only for final evaluation.
                apply_pruning_method(
                    model=model,
                    method=method,
                    scope=scope,
                    target_ratio=ratio,
                    calib_loader=calib_loader
                )
                print("  Pruning applied.")

                print("  Extracting pruning mask...")
                mask_layers = extract_pruning_mask(baseline_state, model)
                save_pruning_mask(exp_dir, method, scope, ratio, mask_layers)
                del baseline_state

                print("  Gathering post-pruning layer info...")
                # Measure post-pruning layer structure for logging
                post_pruning_layer_info = get_layer_info(model)

                print("  Evaluating accuracy after pruning (before fine-tune)...")
                # Measure accuracy immediately after pruning (before fine-tuning)
                # This isolates the quality of each method's importance criterion
                pre_finetune_acc = evaluate_accuracy(model, test_loader)
                print(f"  Accuracy after pruning: {100*pre_finetune_acc:.2f}%")

                print(f"  Fine-tuning pruned model for {POST_PRUNING_FINE_TUNE_EPOCHS} epoch(s)...")
                model, fine_tune_metrics = fine_tune_post_pruning(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    baseline_accuracy=baseline_acc,
                    epochs=POST_PRUNING_FINE_TUNE_EPOCHS,
                    lr=POST_PRUNING_FINE_TUNE_LR,
                    momentum=POST_PRUNING_FINE_TUNE_MOMENTUM,
                    weight_decay=POST_PRUNING_FINE_TUNE_WEIGHT_DECAY,
                    verbose=True
                )
                print("  Fine-tuning done.")
                
                # Check if accuracy recovered
                if fine_tune_metrics['epochs_to_recover'] is not None:
                    print(f"  Accuracy recovered after {fine_tune_metrics['epochs_to_recover']} epochs")
                else:
                    print(f"  Accuracy did not fully recover (final: {fine_tune_metrics['final_accuracy']:.4f}, baseline: {baseline_acc:.4f})")

                print("  Computing final metrics...")
                pruned_params = count_parameters(model)
                pruned_flops = compute_flops(model)
                print("  Measuring inference latency...")
                inference_latency_ms = measure_inference_latency(model, batch_size=1)
                print(f"  Inference latency: {inference_latency_ms:.3f} ms per sample")

                pruned_acc = evaluate_accuracy(model, test_loader)
                print(f"  Pruned model: acc={100*pruned_acc:.2f}%  params={pruned_params/1e6:.2f}M  FLOPs={pruned_flops/1e6:.0f}M")

                # Convert to millions
                pruned_params_m = pruned_params / 1e6
                pruned_flops_m = pruned_flops / 1e6

                # Calculate drops
                accuracy_drop_pct = 100.0 * (baseline_acc - pruned_acc)
                accuracy_drop_before_finetune_pct = 100.0 * (baseline_acc - pre_finetune_acc)
                params_drop_pct = 100.0 * (1.0 - pruned_params / baseline_params)
                flops_drop_pct = 100.0 * (1.0 - pruned_flops / baseline_flops)

                metrics = {
                    "method": method,
                    "scope": scope,
                    "target_pruning_ratio_removed": ratio,  # Fraction REMOVED (0.3 = 30% removed, 70% retained)
                    "target_pruning_ratio_retained": 1.0 - ratio,  # Fraction RETAINED (0.7 = 70% retained)
                    # Note: Many papers use "pruning ratio" to mean fraction retained, hence both fields for clarity
                    # Baseline metrics
                    "baseline_accuracy": baseline_acc,
                    "baseline_flops_m": baseline_flops_m,
                    "baseline_parameters_m": baseline_params_m,
                    "baseline_inference_latency_ms": baseline_latency_ms,
                    # Pruned metrics (after fine-tuning)
                    "accuracy_after_pruning": pruned_acc,
                    "flops_after_pruning_m": pruned_flops_m,
                    "parameters_after_pruning_m": pruned_params_m,
                    # Pre-fine-tuning metrics (isolates pruning quality from fine-tuning recovery)
                    "accuracy_before_finetune": pre_finetune_acc,
                    "accuracy_drop_before_finetune_pct": accuracy_drop_before_finetune_pct,
                    # Drop metrics (after fine-tuning)
                    "accuracy_drop_pct": accuracy_drop_pct,
                    "flops_drop_pct": flops_drop_pct,
                    "parameters_drop_pct": params_drop_pct,
                    # Fine-tuning info
                    "fine_tune_epochs": POST_PRUNING_FINE_TUNE_EPOCHS,
                    "fine_tune_lr": POST_PRUNING_FINE_TUNE_LR,
                    # Fine-tuning cost metrics
                    "epochs_to_recover_accuracy": fine_tune_metrics['epochs_to_recover'],
                    "fine_tune_final_accuracy": fine_tune_metrics['final_accuracy'],
                    # Inference latency (FLOPs ≠ speed)
                    "inference_latency_ms": inference_latency_ms,
                    "inference_speedup": baseline_latency_ms / inference_latency_ms if (baseline_latency_ms and inference_latency_ms and inference_latency_ms > 0) else None,
                    # Raw values (for precision)
                    "accuracy": pruned_acc,
                    "parameters": pruned_params,
                    "flops": pruned_flops,
                    "timestamp": datetime.now().isoformat()
                }

                print("  Saving pruned model and metrics...")
                torch.save(
                    model.state_dict(),
                    os.path.join(exp_dir, "model.pth")
                )
                save_metrics(exp_dir, metrics)
                
                # Save layer-wise pruning information (useful for analysis)
                layer_info = {
                    "pre_pruning": pre_pruning_layer_info,
                    "post_pruning": post_pruning_layer_info,
                    "method": method,
                    "scope": scope,
                    "target_ratio_removed": ratio
                }
                with open(os.path.join(exp_dir, "layer_info.json"), "w") as f:
                    json.dump(layer_info, f, indent=2)
                
                # Save fine-tuning curve (accuracy per epoch) as CSV for easy plotting
                # CSV format: epoch,accuracy,baseline_accuracy
                with open(os.path.join(exp_dir, "fine_tune_curve.csv"), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["epoch", "accuracy", "baseline_accuracy"])
                    for epoch, acc in enumerate(fine_tune_metrics['accuracy_per_epoch'], start=1):
                        writer.writerow([epoch, acc, fine_tune_metrics['baseline_accuracy']])
                
                summary_rows.append(metrics)
                # Update summary CSV after each experiment so partial results are saved if a later run fails
                with open(os.path.join(RESULTS_DIR, "summary.csv"), "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(summary_rows)
                print(f"  Experiment {n}/{total_experiments} saved to {exp_dir} (summary.csv updated)")

    print("\n[4/4] Writing summary...")
    if summary_rows:
        with open(os.path.join(RESULTS_DIR, "summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print("\n" + "=" * 60)
        print("FULL EXPERIMENTS COMPLETE")
        print(f"  Results dir: {RESULTS_DIR}/")
        print(f"  Summary CSV: {RESULTS_DIR}/summary.csv")
        print("=" * 60)
    else:
        print("No experiments ran — nothing to write to summary.csv")


if __name__ == "__main__":
    run()
