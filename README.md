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
- No post-pruning fine-tuning is applied by default