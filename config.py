METHODS = {
    "local": [
        "ApoZ",
        "DropNet",
        "Entropy",
        "HRank",
        "LRMF",
        "CHIP",
        "ThiNet",      # per-layer sequential scoring (ignores scope anyway)
        "DCP",         # per-layer gradient scoring
        "REPrune",     # ordinal MCP ranks / layer-width → uniform distribution → effectively local
        "NISP_local",  # NISP importance scoring, per-layer uniform budget allocation
        "AOFP_local",  # AOFP damage-isolation scoring, per-layer uniform budget allocation
        "GFS",         # ordinal greedy-selection ranks / layer-width → uniform distribution → effectively local
    ],
    "global": [
        "NISP",   # propagated importance + min-max normalisation → genuine non-uniform allocation
        "AOFP",   # damage-isolation fractions, inherently comparable → genuine non-uniform allocation
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