METHODS = {
    "local": [
        "ApoZ",
        "DropNet",
        "Entropy",
        "HRank",
        "LRMF",
        "CHIP",
    ],
    "global": [
        "NISP",
        "ThiNet",
        "AOFP",
        "GFS",
        # "DCP",       # TODO: not yet implemented
        # "REPrune",   # TODO: not yet implemented
    ]
}

PRUNING_RATIOS = [0.3, 0.5, 0.7]
SEED = 42
BATCH_SIZE = 128
BASELINE_TRAIN_EPOCHS = 30
POST_PRUNING_FINE_TUNE_EPOCHS = 10
POST_PRUNING_FINE_TUNE_LR = 0.001
POST_PRUNING_FINE_TUNE_MOMENTUM = 0.9
POST_PRUNING_FINE_TUNE_WEIGHT_DECAY = 5e-4
RESULTS_DIR = "results"
