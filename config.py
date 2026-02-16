METHODS = {
    "local": [
        "ApoZ", "DropNet", "Entropy", "HRank", "CHIP", "LRMF"
    ],
    "global": [
        "NISP", "ThiNet", "AOFP", "GFS", "DCP", "REPrune"
    ]
}

PRUNING_RATIOS = [0.3, 0.5, 0.7]
SEED = 42
# Fine-tuning epochs for ImageNet pretrained VGG-16 on CIFAR-10
# Train until convergence (typically 50-100 epochs)
BASELINE_TRAIN_EPOCHS = 100

# Standardized post-pruning fine-tuning protocol (applied identically to all methods)
# This ensures fair comparison - some methods bake fine-tuning into their pruning loop,
# so using their "default parameters" would give systematic advantage
POST_PRUNING_FINE_TUNE_EPOCHS = 30
POST_PRUNING_FINE_TUNE_LR = 0.001  # One order of magnitude below initial training LR
POST_PRUNING_FINE_TUNE_MOMENTUM = 0.9
POST_PRUNING_FINE_TUNE_WEIGHT_DECAY = 5e-4  # Same as baseline training

RESULTS_DIR = "results"
