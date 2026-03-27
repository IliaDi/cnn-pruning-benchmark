METHODS = {
    "local": [
        "ApoZ",
        "DropNet",
        "Entropy",
        "HRank",
        "LRMF",
        "CHIP",
        "ThiNet",    # per-layer sequential scoring (ignores scope anyway)
        "DCP",       # per-layer gradient scoring
    ],
    "global": [
        "REPrune",   # global mask from per-layer MCP-based full rankings
        "NISP",      # the only genuinely global method
        "AOFP",      # DI scoring + global rank-based normalization
        "GFS",       # global mask from per-layer greedy-selection rankings
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