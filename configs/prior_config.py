import math


DEFAULT_FIXED_HP = {
    # MixPrior
    "mix_prior_types": ["gmires", "gm"],
    "mix_prior_probas": [0.6, 0.4],
    "max_categories": float("inf"),
    "scale_by_max_features": False,
    "permute_features": True,
    # GaussianMixturePrior (MixSim)
    "PiLow": 1e-1,  # Minimum mixing proportion
    "int": [-1.0, 1.0],    # Internal parameter - feature range [min, max]
    "ecc": 0.9,     # Eccentricity
    # GMIResPrior: Zeus-style GMM with optional transformations
    "gmires_min_distance": 0.5,  # Minimum Wasserstein-2 distance between Gaussians
    "gmires_start_distance": 1.0,  # If not None, distance is uniform between min_distance and start_distance
    "gmires_p1": 0.005,  # Lower bound for covariance eigenvalues
    "gmires_p2": 0.5,  # Upper bound for covariance eigenvalues
    # Dirichlet concentration parameters for mixture weights
    # Higher alpha -> more balanced cluster sizes; Lower alpha -> allows imbalanced clusters
    "dirichlet_alpha_binary": 2.0,  # For binary classification (num_classes=2), allows slight imbalance
    "dirichlet_alpha_multiclass": 2.0,  # For multi-class (num_classes>2), more balanced
    "resample_weights": True,  # Whether to resample weights using Dirichlet distribution (True) or use MixSim's original weights (False)
}

DEFAULT_SAMPLED_HP = {
    # GaussianMixturePrior (MixSim)
    # MaxOmega: Maximum overlap, dynamically adjusted based on num_features
    # In high dimensions, higher MaxOmega values are harder to achieve, so we reduce the upper bound
    # Using inverse relationship: higher dimensions -> lower MaxOmega bounds
    "MaxOmega": {
        "distribution": "meta_uniform",
        "min_fn": lambda ctx: 0.01,
        "max_fn": lambda ctx: max(0.01, min(0.8, 1.5 / math.pow(ctx.get("num_features"), 0.82))),
    },
    "hom": {"distribution": "meta_choice", "choice_values": [True, False]},
    "sph": {"distribution": "meta_choice", "choice_values": [True, False]},
    # GMIResPrior: Sampled hyperparameters
    "gmires_mode": {"distribution": "meta_choice", "choice_values": ["gaussian", "transformed"]},  # Generation mode: gaussian (standard) or transformed (with random network)
    "gmires_h": {"distribution": "uniform", "min": 0.1, "max": 0.9},  # SpectralNormBlock h parameter (for transformed mode)
    "gmires_num_blocks": {  # Number of blocks in RandomNetwork (for transformed mode)
        "distribution": "meta_trunc_norm_log_scaled",
        "max_mean": 8,
        "min_mean": 3,
        "round": True,
        "lower_bound": 3,
    },
}
