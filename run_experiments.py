import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import csv
import torch
import numpy as np
from datetime import datetime

from config import METHODS, PRUNING_RATIOS, RESULTS_DIR, SEED

from utils.metrics import (
    evaluate_accuracy,
    count_parameters,
    compute_flops
)
from utils.pruning import apply_pruning_method
from utils.training import train
from utils.data import get_cifar10_loaders
from models.vgg import vgg16


# Experiment Configuration
FINE_TUNE = False          # Explicitly disabled for primary benchmark
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
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_loader, test_loader = get_cifar10_loaders()

    # Baseline Model
    baseline_dir = os.path.join(RESULTS_DIR, "baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    model = vgg16(num_classes=10).to(DEVICE)
    train(model, train_loader)

    baseline_acc = evaluate_accuracy(model, test_loader)
    baseline_params = count_parameters(model)
    baseline_flops = compute_flops(model)

    torch.save(model.state_dict(), os.path.join(baseline_dir, "model.pth"))

    baseline_metrics = {
        "accuracy": baseline_acc,
        "parameters": baseline_params,
        "flops": baseline_flops
    }
    save_metrics(baseline_dir, baseline_metrics)

    summary_rows = []

    # Pruning Experiments
    for scope, methods in METHODS.items():
        for method in methods:
            for ratio in PRUNING_RATIOS:
                print(f"[{method} | {scope}] Target pruning ratio: {ratio}")

                exp_dir = os.path.join(
                    RESULTS_DIR, method, f"ratio_{ratio}"
                )
                os.makedirs(exp_dir, exist_ok=True)

                model = vgg16(num_classes=10).to(DEVICE)
                model.load_state_dict(
                    torch.load(os.path.join(baseline_dir, "model.pth")),
                    strict=False
                )

                apply_pruning_method(
                    model=model,
                    method=method,
                    scope=scope,
                    target_ratio=ratio
                )

                pruned_params = count_parameters(model)
                pruned_flops = compute_flops(model)

                pruned_acc = evaluate_accuracy(model, test_loader)

                metrics = {
                    "method": method,
                    "scope": scope,
                    "target_pruning_ratio": ratio,
                    "accuracy": pruned_acc,
                    "accuracy_drop": baseline_acc - pruned_acc,
                    "params_removed_pct":
                        100.0 * (1.0 - pruned_params / baseline_params),
                    "flops_reduction_pct":
                        100.0 * (1.0 - pruned_flops / baseline_flops),
                    "parameters": pruned_params,
                    "flops": pruned_flops,
                    "timestamp": datetime.now().isoformat()
                }

                torch.save(
                    model.state_dict(),
                    os.path.join(exp_dir, "model.pth")
                )
                save_metrics(exp_dir, metrics)
                summary_rows.append(metrics)

    # Global Summary
    with open(os.path.join(RESULTS_DIR, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=summary_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    run()
