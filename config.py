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
FINE_TUNE_EPOCHS = 10
RESULTS_DIR = "results"
