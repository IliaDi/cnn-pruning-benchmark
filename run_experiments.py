import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import csv
import torch
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


def run():
    """Run full pruning experiments."""
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_loader, test_loader = get_cifar10_loaders(use_augmentation=True)
    baseline_dir = os.path.join(RESULTS_DIR, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    model = vgg16(num_classes=10, pretrained=True).to(DEVICE)
    print(f"Fine-tuning ImageNet pretrained VGG-16 on CIFAR-10 for {BASELINE_TRAIN_EPOCHS} epochs...")
    train(model, train_loader, epochs=BASELINE_TRAIN_EPOCHS, fine_tune=True)
    print("Fine-tuning complete.")

    baseline_acc = evaluate_accuracy(model, test_loader)
    baseline_params = count_parameters(model)
    # Save state_dict before compute_flops — thop.profile() adds total_ops/total_params to the model
    torch.save(model.state_dict(), os.path.join(baseline_dir, "model.pth"))
    baseline_flops = compute_flops(model)
    print("Measuring baseline inference latency...")
    baseline_latency_ms = measure_inference_latency(model, batch_size=1)
    print(f"Baseline inference latency: {baseline_latency_ms:.3f} ms per sample")

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

    for scope, methods in METHODS.items():
        for method in methods:
            for ratio in PRUNING_RATIOS:
                removal_pct = ratio * 100
                retention_pct = (1 - ratio) * 100
                print(f"[{method} | {scope}] Removing {removal_pct:.0f}% of filters (retaining {retention_pct:.0f}%)")

                exp_dir = os.path.join(
                    RESULTS_DIR, method, f"ratio_{ratio}"
                )
                os.makedirs(exp_dir, exist_ok=True)

                # Re-seed for reproducibility (each experiment should start from same random state)
                set_seed(SEED)

                # Load fine-tuned baseline model (filter out any keys added by thop.profile in old checkpoints)
                model = vgg16(num_classes=10, pretrained=False).to(DEVICE)
                ckpt = torch.load(os.path.join(baseline_dir, "model.pth"), map_location=DEVICE)
                model_keys = set(model.state_dict().keys())
                ckpt_filtered = {k: v for k, v in ckpt.items() if k in model_keys}
                model.load_state_dict(ckpt_filtered, strict=True)

                # Measure pre-pruning layer structure for logging
                pre_pruning_layer_info = get_layer_info(model)

                # Apply structural pruning (must physically remove filters, not just mask them)
                apply_pruning_method(
                    model=model,
                    method=method,
                    scope=scope,
                    target_ratio=ratio
                )

                # Measure post-pruning layer structure for logging
                post_pruning_layer_info = get_layer_info(model)

                # Measure accuracy immediately after pruning (before fine-tuning)
                # This isolates the quality of each method's importance criterion
                pre_finetune_acc = evaluate_accuracy(model, test_loader)

                print(f"  Fine-tuning pruned model for {POST_PRUNING_FINE_TUNE_EPOCHS} epochs...")
                model, fine_tune_metrics = fine_tune_post_pruning(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    baseline_accuracy=baseline_acc,
                    epochs=POST_PRUNING_FINE_TUNE_EPOCHS,
                    lr=POST_PRUNING_FINE_TUNE_LR,
                    momentum=POST_PRUNING_FINE_TUNE_MOMENTUM,
                    weight_decay=POST_PRUNING_FINE_TUNE_WEIGHT_DECAY
                )
                print(f"  Fine-tuning complete.")
                
                # Check if accuracy recovered
                if fine_tune_metrics['epochs_to_recover'] is not None:
                    print(f"  Accuracy recovered after {fine_tune_metrics['epochs_to_recover']} epochs")
                else:
                    print(f"  Accuracy did not fully recover (final: {fine_tune_metrics['final_accuracy']:.4f}, baseline: {baseline_acc:.4f})")

                pruned_params = count_parameters(model)
                pruned_flops = compute_flops(model)
                print("  Measuring inference latency...")
                inference_latency_ms = measure_inference_latency(model, batch_size=1)
                print(f"  Inference latency: {inference_latency_ms:.3f} ms per sample")

                pruned_acc = evaluate_accuracy(model, test_loader)

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
                
                summary_rows.append(metrics)

    # Global Summary
    if summary_rows:
        with open(os.path.join(RESULTS_DIR, "summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        print("No experiments ran — nothing to write to summary.csv")


if __name__ == "__main__":
    run()
