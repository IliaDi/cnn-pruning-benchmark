METHODS = {
    "local": [
        "ApoZ",
        "DropNet",
        "Entropy",
        "HRank",
        # "CHIP",      # TODO: not yet implemented
        # "LRMF",      # TODO: not yet implemented
    ],
    "global": [
        # "NISP",      # TODO: not yet implemented
        # "ThiNet",    # TODO: not yet implemented
        # "AOFP",      # TODO: not yet implemented
        # "GFS",       # TODO: not yet implemented
        # "DCP",       # TODO: not yet implemented
        # "REPrune",   # TODO: not yet implemented
    ]
}

PRUNING_RATIOS = [0.3, 0.5, 0.7]
SEED = 42
BATCH_SIZE = 64
BASELINE_TRAIN_EPOCHS = 50
POST_PRUNING_FINE_TUNE_EPOCHS = 25
POST_PRUNING_FINE_TUNE_LR = 0.001
POST_PRUNING_FINE_TUNE_MOMENTUM = 0.9
POST_PRUNING_FINE_TUNE_WEIGHT_DECAY = 5e-4
RESULTS_DIR = "results"
