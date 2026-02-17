# CNN Pruning Benchmark

Master's thesis codebase for benchmarking activation-based pruning methods on CNNs using VGG-16 and CIFAR-10.

## Project Structure

### Main Scripts

- **`run_experiments.py`** – Full experiment runner for production runs
  - Trains baseline model, runs all pruning experiments, saves comprehensive metrics
  - Uses configuration from `config.py` (100 baseline epochs, 30 fine-tune epochs, ratios [0.3, 0.5, 0.7])
  - Results saved to `results/` directory

- **`run_quick_test.py`** – Lightweight validation script for rapid iteration (not tracked)
  - Same pipeline as `run_experiments.py` but with reduced epochs and data subsets
  - 2 baseline epochs, 2 fine-tune epochs, 5 batches/epoch, 3 APoZ calibration batches
  - Results saved to `results_quick_test/` directory
  - Useful for testing code changes without running full experiments

### Configuration

- **`config.py`** – Centralized experiment configuration
  - Pruning methods (local/global scope)
  - Pruning ratios (fraction of channels to remove)
  - Training hyperparameters (epochs, learning rates, etc.)
  - Random seed for reproducibility

### Model Definitions

- **`models/vgg.py`** – VGG-16 model definition
  - ImageNet pretrained VGG-16 adapted for CIFAR-10 (10 classes)
  - Handles model loading and initialization

### Utilities (`utils/`)

- **`data.py`** – CIFAR-10 data loading and preprocessing
  - Resizes CIFAR-10 (32×32) to 224×224 for ImageNet VGG-16 compatibility
  - Training transforms: random crop, horizontal flip, ImageNet normalization
  - Test transforms: resize, ImageNet normalization (no augmentation)
  - Configurable `num_workers` for data loading (4 for full experiments, 0 for quick test)

- **`training.py`** – Training and fine-tuning functions
  - `train()`: Baseline fine-tuning with differential LR (features: 0.1×, classifier: 1×)
  - `fine_tune_post_pruning()`: Standardized post-pruning fine-tuning protocol
    - Uniform LR for all parameters (pruned layers are no longer original pretrained weights)
    - Tracks accuracy per epoch and recovery time
    - Supports verbose logging for progress monitoring

- **`metrics.py`** – Evaluation and measurement utilities
  - `evaluate_accuracy()`: Model accuracy on test set
  - `count_parameters()`: Total trainable parameters
  - `compute_flops()`: FLOPs count using `thop` library
  - `measure_inference_latency()`: Wall-clock inference time per sample
    - Handles both CPU and GPU (CUDA synchronization)
    - Warmup iterations and averaging for stable measurements

- **`pruning.py`** – Pruning method dispatcher
  - `apply_pruning_method()`: Routes to specific pruning implementations
  - Accepts calibration data loader to avoid redundant data loading
  - `get_layer_info()`: Extracts layer-wise structure information

- **`apoz.py`** – APoZ (Average Percentage of Zeros) pruning implementation
  - Implements Hu et al. (2016) "Network Trimming" method
  - Hooks ReLU modules to measure post-activation zeros (matches paper definition)
  - Excludes final classification layer from pruning (only updates input dimension)
  - Uses quantile-based threshold for fixed compression ratios (thesis protocol)
  - Performs structural (hard) pruning by physically removing channels

## Results Structure

Results are organized hierarchically by method and pruning ratio:

```
results/
├── baseline/
│   ├── model.pth              # Trained baseline model state_dict
│   ├── metrics.json           # Baseline metrics (JSON format)
│   └── metrics.csv            # Baseline metrics (CSV format, single row)
│
├── <method>/                  # e.g. ApoZ, ThiNet, etc.
│   └── ratio_<ratio>/         # e.g. ratio_0.3, ratio_0.5, ratio_0.7
│       ├── model.pth          # Pruned and fine-tuned model state_dict
│       ├── metrics.json       # Comprehensive experiment metrics (JSON)
│       ├── metrics.csv         # Same metrics in CSV format (single row)
│       ├── layer_info.json    # Pre/post pruning layer structure details
│       └── fine_tune_curve.csv # Fine-tuning accuracy per epoch (time-series)
│
└── summary.csv                # Aggregated results from all experiments
```

## Approach

### Baseline Model

- **ImageNet pretrained VGG-16** adapted for CIFAR-10 (10 classes)
- Fine-tuned on CIFAR-10 to convergence (100 epochs)
- Provides strong pretrained features while being fully grounded in CIFAR-10
- CIFAR-10 images resized to 224×224 to match ImageNet VGG-16 input size
- Differential learning rates: features at 0.1× LR, classifier at 1× LR

### Multiple Pruning Ratios

Tests 30%, 50%, and 70% filter removal to cover the compression spectrum:
- Allows drawing accuracy-vs-compression curves for each method
- Methods behave differently across ratios (a method good at moderate pruning may fail at aggressive pruning)
- Results can be sorted by FLOPs of pruned model rather than fixing a single ratio

### Standardized Post-Pruning Fine-Tuning

Applied identically to all methods for fair comparison:
- **30 epochs** of fine-tuning
- **SGD optimizer** with momentum 0.9
- **Learning rate**: 0.001 (one order of magnitude below baseline LR)
- **Uniform LR** for all parameters (pruned layers are no longer original pretrained weights)
- **Cosine annealing** LR schedule
- **Weight decay**: 5e-4 (same as baseline)
- Same batch size and data augmentation (random crop, horizontal flip) as baseline

This ensures fair comparison - some methods (e.g., AOFP, ThiNet) bake fine-tuning into their pruning loop, so using their "default parameters" would give systematic advantage to more complex methods.

## Metrics

### Accuracy
- Baseline and pruned model accuracy on test set
- Accuracy drop percentage (after fine-tuning)
- Pre-fine-tuning accuracy (isolates pruning quality from fine-tuning recovery)

### Compression
- **Parameters**: Baseline and pruned parameter counts, reduction percentage
- **FLOPs**: Baseline and pruned FLOPs, reduction percentage
- Structural pruning physically removes channels, so parameter/FLOPs counts reflect actual compression

### Speed
- **Inference Latency**: Wall-clock time per sample on fixed GPU
  - Measured with warmup iterations and averaged over multiple runs
  - Captures hardware efficiency, memory access patterns, etc.
  - FLOPs ≠ actual speed due to hardware efficiency differences
- **Speedup**: Ratio of baseline latency to pruned latency

### Fine-Tuning Cost
- **Epochs to recover**: Number of epochs needed to recover baseline accuracy after pruning
- Directly relevant for comparing lightweight local methods vs expensive global ones (e.g., NISP, DCP)
- Tracks accuracy recovery during fine-tuning to measure computational cost
- Saved as `fine_tune_curve.csv` for detailed analysis

### Data Loading

- **Full experiments**: `num_workers=4` for parallel data loading (faster)
- **Quick test**: `num_workers=0` for simplicity and debugging
- Calibration data loader passed to pruning methods to avoid redundant data loading
