METHODS = {
    "local": [
        "ApoZ", "DropNet", "Entropy", "HRank", "CHIP", "LRMF"
    ],
    "global": [
        "NISP", "ThiNet", "AOFP", "GFS", "DCP", "REPrune"
    ]
}

PRUNING_RATIOS = [0.1, 0.3, 0.5, 0.7]
SEED = 42
# Fine-tuning epochs for ImageNet pretrained VGG-16 on CIFAR-10
# Train until convergence (typically 50-100 epochs)
BASELINE_TRAIN_EPOCHS = 100
FINE_TUNE_EPOCHS = 10  # For post-pruning fine-tuning (if enabled)
RESULTS_DIR = "results"
