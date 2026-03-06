# CNN Pruning Benchmark

Master's thesis codebase for benchmarking activation-based pruning methods on VGG-16 (CIFAR-10, resized to 224×224).

## Quick Start

- **Full experiments:** `python run_experiments.py`  
  Uses `config.py` (30 baseline epochs, 10 fine-tune epochs, batch size 128, ratios [0.3, 0.5, 0.7]). Results in `results/`. Skips baseline if `results/baseline/model.pth` and `metrics.json` exist; skips each (method, ratio) if that run’s `metrics.json` exists.

## Configuration

- **`config.py`** – Methods (local/global), pruning ratios, epochs, LR, batch size, seed, `RESULTS_DIR`.

## Data

- **Training:** 40k images (4000 per class), held out from calibration. Augmentation: random crop, horizontal flip, ImageNet normalization.
- **Calibration:** 10k images (1000 per class), stratified held-out from the original training set. No augmentation. Used only for pruning (activation statistics); test set is never used for pruning.
- **Test:** Standard CIFAR-10 test set (10k). Used only for final accuracy and eval.

Loaders: `utils/data.py` → `get_cifar10_loaders()` returns `(train_loader, test_loader, calib_loader)`.

## Results Layout

```
results/
├── baseline/           # model.pth, metrics.json
├── <method>/ratio_<r>/  # e.g. ApoZ/ratio_0.3  → model.pth, metrics.json, layer_info.json, fine_tune_curve.csv
└── summary.csv        # One row per (method, ratio)
```

## Pruning Methods

All methods perform **structural** (hard) pruning; fine-tuning is applied externally with the same protocol (10 epochs, SGD 0.001, cosine annealing).

| Method | Paper | Idea |
|--------|--------|------|
| **ApoZ** | Hu et al. 2016 | Prune channels with highest % zeros (post-ReLU). |
| **DropNet** | Tan & Motani 2020 | Score by mean absolute post-ReLU activation; prune lowest. |
| **Entropy** | Luo & Wu 2017 | Score by Shannon entropy of activations; prune lowest. |
| **HRank** | Lin et al. 2020 | Score by average rank of feature maps (SVD); prune lowest. |
| **CHIP** | Sui et al. 2021 | Score by channel independence (nuclear norm); prune lowest. |
| **LRMF** | Zhang et al. 2023 | Score by DCT low-freq pairwise distance; prune “median” channels. |
| **NISP** | Yu et al. 2018 | Propagate importance from final response layer backward; prune lowest. |
| **ThiNet** | Luo, Wu & Lin 2017 | Next-layer reconstruction; greedy channel selection. |
| **GFS** | Ye et al. 2020 | Score by single-filter loss contribution; prune lowest. |
| **AOFP** | Ding et al. 2019 | Damage isolation (ablation + next-layer deviation); prune lowest. |

Implementation: `utils/pruning.py` dispatches to `utils/<method>.py`. Structural removal uses shared helpers in `utils/apoz.py`.

## Metrics 

- **Accuracy:** Baseline and pruned (before and after fine-tuning). Accuracy drop %.
- **Compression:** Parameters and FLOPs (counts and reduction %).
- **Inference latency:** Wall-clock ms per sample (warmup + average over 50 runs).
- **Fine-tuning:** Epochs to recover baseline (within 0.1%), curve in `fine_tune_curve.csv`.

Details: `utils/metrics.py`, `utils/training.py`.

## Project Layout

- **Scripts:** `run_experiments.py`, `run_quick_test.py`
- **Config:** `config.py`
- **Model:** `models/vgg.py` (VGG-16, 10 classes)
- **Utils:** `data.py`, `training.py`, `metrics.py`, `pruning.py`, and per-method modules under `utils/`
