# CNN Pruning Benchmark

Master's thesis codebase for benchmarking activation-based pruning methods
on CNNs using VGG16 and CIFAR-10.

## Structure
- `run_experiments.py` – main experiment runner
- `models/` – model definitions
- `utils/` – training, metrics, pruning utilities

## Notes
- Models are trained from scratch on CIFAR-10
- No post-pruning fine-tuning is applied by default
- Debug/smoke-test scripts are excluded from the repository