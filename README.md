# CNN Pruning Benchmark

Master's thesis codebase for benchmarking activation-based pruning methods on CNNs using VGG-16 and CIFAR-10.

## Project Structure

### Main Scripts

- **`run_experiments.py`** – Full experiment runner for production runs
  - Trains baseline model, runs all pruning experiments, saves comprehensive metrics
  - Uses configuration from `config.py` (50 baseline epochs, 25 fine-tune epochs, batch size 64, ratios [0.3, 0.5, 0.7])
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
  - Training hyperparameters (epochs, learning rates, batch size, etc.)
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
  - Batch size defaults to `config.BATCH_SIZE` (64)
  - Configurable `num_workers` for data loading (auto-detects macOS and uses 0, Linux uses 4)

- **`training.py`** – Training and fine-tuning functions
  - `train()`: Baseline fine-tuning with differential learning rates
    - Base LR: 0.001
    - Features (Conv2d): 0.0001 (0.1× base) - preserves pretrained ImageNet weights
    - Classifier (Linear): 0.001 (1× base) - adapts new 10-class head
    - Cosine annealing scheduler
  - `fine_tune_post_pruning()`: Standardized post-pruning fine-tuning protocol
    - Uniform LR: 0.001 for all parameters (pruned layers are no longer original pretrained weights)
    - Cosine annealing scheduler
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

### Pruning Methods (`utils/`)

- **`apoz.py`** – APoZ (Average Percentage of Zeros) pruning
  - **Paper**: Hu et al. (2016) "Network Trimming: A Data-Driven Neuron Pruning Approach"
  - **Method**: Measures how often channels output zeros (post-ReLU activations)
  - **Scoring**: APoZ = (zero activations) / (total activations) per channel
  - **Pruning**: Channels with highest APoZ (most frequently zero) are pruned first
  - **Implementation details**:
    - Hooks ReLU modules to measure post-activation zeros (matches paper definition)
    - Excludes final classification layer from pruning (only updates input dimension)
    - Uses quantile-based threshold for fixed compression ratios (thesis protocol)
    - Performs structural (hard) pruning by physically removing channels

- **`dropnet.py`** – DropNet pruning
  - **Paper**: Tan & Motani (2020) "DropNet: Reducing Neural Network Complexity via Iterative Pruning"
  - **Method**: Scores filters by expected absolute post-activation value
  - **Scoring**: E(f_i) = mean |ReLU(conv_i(x))| across calibration samples
  - **Pruning**: Filters with lowest scores (least active) are pruned first
  - **Implementation details**:
    - Supports both layer-wise (`scope="local"`, recommended) and global (`scope="global"`) ranking
    - Single-shot pruning (no iterative retraining) for fair benchmark comparison
    - Reuses APoZ structural pruning helpers for consistent channel removal

- **`entropy.py`** – Entropy-based pruning
  - **Paper**: Luo & Wu (2017) "An Entropy-based Pruning Method for CNN Compression"
  - **Method**: Scores filters by Shannon entropy of post-activation output distribution
  - **Scoring**: 
    1. Global average pooling: (B, C, H, W) → (B, C)
    2. Compute Shannon entropy H_j = -Σ p_i log(p_i) for each channel j
    3. Bin activation values into 256 bins to estimate probability distribution
  - **Pruning**: Channels with lowest entropy (least informative) are pruned first
  - **Implementation details**:
    - Supports both layer-wise (`scope="local"`) and global (`scope="global"`) ranking
    - Uses 256 histogram bins for entropy estimation (standard practice)
    - Reuses APoZ structural pruning helpers for consistent channel removal

- **`hrank.py`** – HRank filter pruning
  - **Paper**: Lin et al. (2020) "HRank: Filter Pruning using High-Rank Feature Map" (CVPR 2020)
  - **Method**: Scores filters by average numerical rank of their 2-D feature maps
  - **Scoring**: 
    1. Hook Conv2d outputs (pre-ReLU feature maps): shape (B, C, H, W)
    2. For each filter j, compute matrix rank of each spatial feature map (H×W) via SVD
    3. Average rank across all calibration images: rank_score[j] = (1/g) * Σ Rank(feature_map[t, j])
  - **Pruning**: Filters with lowest average rank (least informative feature maps) are pruned first
  - **Implementation details**:
    - Uses `torch.linalg.matrix_rank` (SVD-based) for rank computation
    - Supports both layer-wise (`scope="local"`, default) and global (`scope="global"`) ranking
    - More computationally expensive than APoZ/DropNet due to per-image SVD operations
    - Paper uses ~500 images (4 batches of 128) for VGG-16 calibration
    - Reuses APoZ structural pruning helpers for consistent channel removal

- **`chip.py`** – CHIP (Channel Independence-based Pruning)
  - **Paper**: Sui et al. (2021) "CHIP: CHannel Independence-based Pruning for Compact Neural Networks" (NeurIPS 2021)
  - **Method**: Scores filters by channel independence (CI) — how much removing a channel drops the nuclear norm of the layer's feature map matrix
  - **Scoring** (paper Eq. 3):
    1. Hook post-ReLU feature maps per layer: shape (B, C, H, W) → matricize to A ∈ R^{C×hw}
    2. CI(A_i) = ‖A‖_* − ‖M_i ⊙ A‖_* (nuclear norm of full matrix minus nuclear norm with row i zeroed)
    3. Average CI over all calibration samples
  - **Pruning**: Channels with lowest CI (most linearly dependent on others) are pruned first
  - **Implementation details**:
    - Uses `torch.linalg.svdvals` for nuclear norm (sum of singular values)
    - Supports both layer-wise (`scope="local"`, default) and global (`scope="global"`) ranking
    - Computationally expensive (SVD per channel per image); paper uses 5 batches of 128 images
    - Reuses APoZ structural pruning helpers for consistent channel removal

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
- Fine-tuned on CIFAR-10 to convergence (50 epochs)
- Provides strong pretrained features while being fully grounded in CIFAR-10
- CIFAR-10 images resized to 224×224 to match ImageNet VGG-16 input size
- **Differential learning rates**:
  - Features (Conv2d): 0.0001 (10× lower) - preserves pretrained ImageNet weights
  - Classifier (Linear): 0.001 - adapts new 10-class head from scratch
- **Cosine annealing scheduler**: LR decays smoothly from initial value to 0 over training
- Batch size: 64

### Multiple Pruning Ratios

Tests 30%, 50%, and 70% filter removal to cover the compression spectrum:
- Allows drawing accuracy-vs-compression curves for each method
- Methods behave differently across ratios (a method good at moderate pruning may fail at aggressive pruning)
- Results can be sorted by FLOPs of pruned model rather than fixing a single ratio

### Standardized Post-Pruning Fine-Tuning

Applied identically to all methods for fair comparison:
- **25 epochs** of fine-tuning
- **SGD optimizer** with momentum 0.9
- **Learning rate**: 0.001 (uniform for all parameters - pruned layers are no longer original pretrained weights)
- **Cosine annealing scheduler**: LR decays smoothly from 0.001 to 0 over 25 epochs
- **Weight decay**: 5e-4 (same as baseline)
- Same batch size (64) and data augmentation (random crop, horizontal flip) as baseline

**Learning Rate vs Scheduler:**
- **Learning Rate (LR)**: The initial step size for gradient updates (e.g., 0.001)
- **Scheduler**: Dynamically adjusts LR during training (e.g., CosineAnnealingLR starts at 0.001 and decays to 0)
- Cosine annealing provides smooth decay: high LR early (fast learning), low LR later (fine-tuning)

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
- **Inference Latency**: Wall-clock time per sample (pure inference, no training)
  - **Measurement process** (`utils/metrics.py:measure_inference_latency()`):
    1. **Warmup** (10 iterations): Runs inference without timing to stabilize device
       - GPU: Initializes CUDA kernels, loads weights into GPU memory
       - CPU: Warms up CPU cache, initializes operations
       - First few runs are slower, so we exclude them from timing
    2. **Timing** (50 iterations): Times each forward pass
       - Creates dummy input tensor (batch_size=1, 224×224×3)
       - Records wall-clock time: `start_time` → `model(dummy_input)` → `end_time`
       - GPU: Synchronizes CUDA before/after each run (ensures accurate timing)
       - CPU: Uses standard Python timing (no synchronization needed)
    3. **Average**: Sums all 50 timings, divides by 50, then by batch_size
       - Result: milliseconds per sample
  - **What's measured**: Pure forward pass only (no data loading, no training, no backward pass, no epochs)
- **Baseline vs Pruned**: Both measured identically after training/fine-tuning completes
    - Baseline: Measured after baseline training (50 epochs) completes
    - Pruned: Measured after pruning + fine-tuning (25 epochs) completes
- **Speedup**: Ratio of baseline latency to pruned latency (e.g., 2.0× means pruned model is 2× faster)

### Fine-Tuning Cost
- **Epochs to recover**: Number of epochs needed to recover baseline accuracy after pruning
  - **Recovery definition**: Accuracy within 0.1% of baseline (e.g., baseline 85.0% → recovered if ≥84.9%)
  - **Tracking**: After each fine-tuning epoch, checks if `current_accuracy >= (baseline_accuracy - 0.001)`
  - **If recovered**: Records the epoch number (e.g., "recovered after 12 epochs")
  - **If NOT recovered**: Returns `None` (accuracy never reached baseline level within 25 epochs)
    - This is common - aggressive pruning may permanently reduce accuracy
    - Final accuracy is still saved (e.g., "final: 82.5%, baseline: 85.0%")
- **Accuracy curve**: Saved as `fine_tune_curve.csv` showing accuracy after each epoch
  - Useful for analyzing recovery dynamics and comparing methods

### Data Loading

- **Full experiments**: Auto-detects OS - uses `num_workers=0` on macOS (multiprocessing compatibility), `num_workers=4` on Linux (faster parallel loading)
- **Quick test**: Always uses `num_workers=0` for simplicity and debugging
- Calibration data loader passed to pruning methods to avoid redundant data loading
