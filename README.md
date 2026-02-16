# CNN Pruning Benchmark

Master's thesis codebase for benchmarking activation-based pruning methods
on CNNs using VGG16 and CIFAR-10.

## Structure
- `run_experiments.py` – main experiment runner
- `models/` – model definitions
- `utils/` – training, metrics, pruning utilities

## Approach
- **Baseline Model**: ImageNet pretrained VGG-16, adapted classifier head for 10 classes, fine-tuned on CIFAR-10 to convergence
- This provides strong pretrained features while being fully grounded in CIFAR-10
- CIFAR-10 images are resized to 224x224 to match ImageNet VGG-16 input size
- **Multiple Pruning Ratios**: Tests 30%, 50%, and 70% filter removal to cover the compression spectrum
  - Allows drawing accuracy-vs-compression curves for each method
  - Methods behave differently across ratios (a method good at moderate pruning may fail at aggressive pruning)
  - Results can be sorted by FLOPs of pruned model rather than fixing a single ratio
- **Standardized Post-Pruning Fine-Tuning**: Applied identically to all 12 methods for fair comparison
  - 30 epochs, SGD with momentum 0.9, LR 0.001 (one order below baseline LR)
  - Cosine annealing LR schedule, weight decay 5e-4 (same as baseline)
  - Same batch size and data augmentation (random crop, horizontal flip) as baseline
  - Ensures fair comparison - some methods bake fine-tuning into their pruning loop, so using their "default parameters" would give systematic advantage

## Metrics

- **Accuracy**: Baseline and pruned model accuracy, accuracy drop percentage
- **Parameters**: Baseline and pruned parameter counts, reduction percentage
- **FLOPs**: Baseline and pruned FLOPs, reduction percentage
- **Inference Latency**: Wall-clock time per sample on fixed GPU (distinguishes from FLOPs - FLOPs ≠ actual speed)
  - Measured with warmup iterations and averaged over multiple runs
  - Captures hardware efficiency, memory access patterns, etc.
- **Fine-Tuning Cost**: Epochs needed to recover baseline accuracy after pruning
  - Directly relevant for comparing lightweight local methods vs expensive global ones (e.g., NISP, DCP)
  - Tracks accuracy recovery during fine-tuning to measure computational cost