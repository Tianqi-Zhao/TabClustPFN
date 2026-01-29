"""
GMIRes-based prior class for synthetic dataset generation.

This module contains the GMIResPrior class that generates
synthetic datasets using the Zeus-style Gaussian Mixture
approach with optional random network transformations.
"""

from __future__ import annotations

import torch
from torch import Tensor
from typing import Dict, Tuple, Optional, Any
import torch.nn.functional as F
import logging

from .base import StructuredPrior
from .gmires import GMIRes
from .categorical_features import CategoricalFeatureGenerator
from .hp_sampling import HpSamplerList
from .utils import outlier_removing, standard_scaling
from configs.prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP
from .utils import sample_gaussian_parameters  # Import shared function
import numpy as np

logger = logging.getLogger(__name__)


class GMIResPrior(StructuredPrior):
    """
    Generates synthetic datasets using Zeus-style Gaussian Mixture with
    optional random network transformations.
    
    The data generation process follows a hierarchical structure similar to other priors:
    1. Generate a list of parameters for each dataset, respecting group/subgroup sharing.
    2. Process the parameter list to generate datasets using GMIRes.
    
    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch
    
    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics
    
    batch_size_per_subgp : int, default=None
        Number of datasets per subgroup, with more similar causal structures
        If None, defaults to batch_size_per_gp
    
    min_features : int, default=2
        Minimum number of features per dataset
    
    max_features : int, default=100
        Maximum number of features per dataset
    
    max_classes : int, default=10
        Maximum number of target classes
    
    min_seq_len : int, default=None
        Minimum samples per dataset. If None, uses max_seq_len directly.
    
    max_seq_len : int, default=1024
        Maximum samples per dataset
    
    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution
    
    seq_len_per_gp : bool = False
        If True, sample sequence length per group, allowing variable-sized datasets
    
    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets
    
    fixed_hp : dict, default=DEFAULT_FIXED_HP
        Fixed structural configuration parameters
    
    sampled_hp : dict, default=DEFAULT_SAMPLED_HP
        Parameters sampled during generation
    
    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors).
    
    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation
    
    device : str, default="cpu"
        Computation device ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: Optional[int] = None,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: Optional[int] = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        replay_small: bool = False,
        fixed_hp: Dict[str, Any] = DEFAULT_FIXED_HP,
        sampled_hp: Dict[str, Any] = DEFAULT_SAMPLED_HP,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
    ):
        super().__init__(
            batch_size=batch_size,
            batch_size_per_gp=batch_size_per_gp,
            batch_size_per_subgp=batch_size_per_subgp,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes,
            min_seq_len=min_seq_len,
            max_seq_len=max_seq_len,
            log_seq_len=log_seq_len,
            seq_len_per_gp=seq_len_per_gp,
            replay_small=replay_small,
            n_jobs=n_jobs,
            num_threads_per_generate=num_threads_per_generate,
            device=device,
        )
        
        self._fixed_hp = fixed_hp
        self._sampled_hp = sampled_hp
    
    @property
    def fixed_hp(self) -> Dict[str, Any]:
        """Return fixed hyperparameters for GMIRes."""
        return {
            "gmires_min_distance": self._fixed_hp["gmires_min_distance"],
            "gmires_start_distance": self._fixed_hp["gmires_start_distance"],
            "gmires_p1": self._fixed_hp["gmires_p1"],
            "gmires_p2": self._fixed_hp["gmires_p2"],
            "dirichlet_alpha_binary": self._fixed_hp["dirichlet_alpha_binary"],
            "dirichlet_alpha_multiclass": self._fixed_hp["dirichlet_alpha_multiclass"],
            "permute_features": self._fixed_hp["permute_features"],
            "scale_by_max_features": self._fixed_hp["scale_by_max_features"],
            "max_categories": self._fixed_hp["max_categories"],  # Reuse existing parameter
        }
    
    @property
    def sampled_hp(self) -> Dict[str, Any]:
        """Return sampled hyperparameters for GMIRes."""
        return {
            "gmires_mode": self._sampled_hp["gmires_mode"],
            "gmires_h": self._sampled_hp["gmires_h"],
            "gmires_num_blocks": self._sampled_hp["gmires_num_blocks"],  # Sampled per subgroup
        }
    
    def hp_sampling(self, generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
        """
        Sample hyperparameters for dataset generation.
        
        Returns
        -------
        dict
            Dictionary with sampled hyperparameters
        """
        hp_sampler = HpSamplerList(self.sampled_hp, device=self.device, generator=generator)
        return hp_sampler.sample()
    
    def _sample_feature_split(
        self,
        num_features: int,
        max_categories: int,
        generator: torch.Generator,
    ) -> Tuple[int, list]:
        """
        Sample feature split between continuous and categorical features.
        
        Note: Categorical features use label representation (not one-hot),
        so each categorical feature occupies 1 dimension.
        Total: num_features = continuous_dim + len(categorical_dims)
        
        Parameters
        ----------
        num_features : int
            Total number of features to generate
        only_categorical : bool
            If True, all features are categorical
        max_categories : int or float
            Maximum number of categories per categorical feature
        generator : torch.Generator
            Random generator for reproducibility
        
        Returns
        -------
        tuple
            (continuous_dim, categorical_dims)
            - continuous_dim: number of continuous features
            - categorical_dims: list of category counts for each categorical feature
        """
        seed = torch.empty((), dtype=torch.int64, device='cpu').random_(generator=generator).item()
        rng = np.random.default_rng(seed)
        
        # Randomly split features between continuous and categorical
        # Always ensure at least 2 continuous features (never only categorical)
        max_continuous = num_features
        continuous_dim = rng.integers(2, max(3, max_continuous + 1))
        num_categorical_features = num_features - continuous_dim
        
        # Sample categorical feature configuration
        # Generate exactly num_categorical_features categorical features
        categorical_dims = []
        if num_categorical_features > 0:
            for _ in range(num_categorical_features):
                if max_categories != float("inf"):
                    max_cats = int(max_categories)
                else:
                    max_cats = 5  # Reasonable default
                n_categories = rng.integers(2, max_cats + 1)
                categorical_dims.append(n_categories)
        
        return continuous_dim, categorical_dims
    
    @torch.no_grad()
    def generate_dataset(self, params: Dict[str, Any]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Generates a single valid dataset based on the provided parameters.
        
        Parameters
        ----------
        params : dict
            Hyperparameters for generating this specific dataset, including seq_len,
            num_features, num_classes, device, etc.
        
        Returns
        -------
        tuple
            (X, y, d) where:
            - X: Features tensor of shape (seq_len, max_features)
            - y: Labels tensor of shape (seq_len,)
            - d: Number of active features after filtering (scalar Tensor)
        """
        generator = params["generator"]
        attempt_count = 0
        
        while True:
            attempt_count += 1
            
            try:
                # Sample feature split between continuous and categorical
                continuous_dim, categorical_dims = self._sample_feature_split(
                    num_features=params["num_features"],
                    max_categories=params["max_categories"],
                    generator=generator,
                )
                
                
                # Sample Gaussian parameters if we have continuous features
                if continuous_dim > 0:
                    weights, means, covariances = sample_gaussian_parameters(
                        num_classes=params["num_classes"],
                        continuous_dim=continuous_dim,
                        min_distance=params["gmires_min_distance"],
                        p1=params["gmires_p1"],
                        p2=params["gmires_p2"],
                        device=self.device,
                        generator=generator,
                        start_distance=params["gmires_start_distance"],
                        alpha_binary=params["dirichlet_alpha_binary"],
                        alpha_multiclass=params["dirichlet_alpha_multiclass"],
                    )
                else:
                    # Only categorical features - use dummy Gaussian parameters
                    weights = torch.ones(params["num_classes"], device=self.device) / params["num_classes"]
                    means = torch.zeros(params["num_classes"], 1, device=self.device)
                    covariances = torch.eye(1, device=self.device).unsqueeze(0).repeat(params["num_classes"], 1, 1)
                    continuous_dim = 1  # Dummy dimension
                
                # Create GMIRes instance for continuous features
                gmires = GMIRes(
                    seq_len=params["seq_len"],
                    num_classes=params["num_classes"],
                    continuous_dim=continuous_dim,
                    weights=weights,
                    means=means,
                    covariances=covariances,
                    mode=params["gmires_mode"],
                    num_blocks=params["gmires_num_blocks"],
                    h=params["gmires_h"],
                    device=self.device,
                    generator=generator,
                )
                
                # Generate continuous features
                X_continuous, y = gmires()
                
                # Generate categorical features separately (as labels, not one-hot)
                if len(categorical_dims) > 0:
                    cat_gen = CategoricalFeatureGenerator(
                        seq_len=params["seq_len"],
                        num_classes=params["num_classes"],
                        categorical_dims=categorical_dims,
                        device=self.device,
                        generator=generator,
                    )
                    X_categorical_labels = cat_gen(y)  # Shape: (seq_len, num_cat_features)
                    
                    # Convert categorical labels to float for concatenation with continuous features
                    X_categorical = X_categorical_labels.float()
                    
                    # Combine continuous and categorical features
                    X_raw = torch.cat([X_continuous, X_categorical], dim=1)
                else:
                    X_raw = X_continuous
                
                # Record actual number of features before processing
                # Label representation: total = continuous features + number of categorical features
                actual_features = continuous_dim + len(categorical_dims)
                
                # Verify: actual_features should equal num_features
                assert actual_features == params["num_features"], \
                    f"Feature mismatch: {actual_features} != {params['num_features']}"
                
                # Step 2: Apply feature processing
                X = self._process_features(X_raw, params, actual_features=actual_features)
                
                # Add batch dim for single dataset to be compatible with delete_unique_features
                X, y = X.unsqueeze(0), y.unsqueeze(0)
                d = torch.tensor([params["num_features"]], device=self.device, dtype=torch.long)
                
                # Step 3: Only keep valid datasets with sufficient features
                X, d = self.delete_unique_features(X, d)
                
                if (d > 0).all():
                    # Only return if valid data is successfully generated
                    return X.squeeze(0), y.squeeze(0), d.squeeze(0)
                
                # If validation fails (all features filtered out), log warning
                logger.info(
                    "Validation failed after %d attempts (all features filtered). "
                    "Prior: %s | Params: seq_len=%s, num_features=%s, num_classes=%s",
                    attempt_count,
                    self.__class__.__name__,
                    params['seq_len'],
                    params['num_features'],
                    params['num_classes']
                )
            
            # Catch all exceptions during generation
            except Exception as e:
                logger.warning(
                    "GMIResPrior dataset generation failed (attempt %d): %s | "
                    "params={num_classes:%s, num_features:%s, seq_len:%s}",
                    attempt_count,
                    e,
                    params.get("num_classes"),
                    params.get("num_features"),
                    params.get("seq_len"),
                )
            
            # Common retry logic for both validation failures and exceptions
            # Advance generator to ensure next retry uses different random seed
            _ = torch.empty((), dtype=torch.int64, device='cpu').random_(generator=generator).item()
            
            # Prevent infinite loop by setting retry limit
            if attempt_count >= 5:
                raise RuntimeError(
                    f"Failed to generate a valid dataset after {attempt_count} attempts. "
                    f"Params: seq_len={params['seq_len']}, num_features={params['num_features']}, "
                    f"num_classes={params['num_classes']}, prior_type={params.get('prior_type', 'gmires')}"
                )
    
    def _process_features(self, X: Tensor, params: Dict[str, Any], actual_features: int) -> Tensor:
        """Process features through outlier removal, scaling, and padding to max features.
        
        Parameters
        ----------
        X : Tensor
            Feature tensor of shape (T, H).
        params : dict
            Hyperparameters containing max_features and other settings.
        actual_features : int
            Actual number of features (before padding)
        
        Returns
        -------
        Tensor
            Processed feature tensor of shape (T, max_features).
        """
        
        num_features = X.shape[1]
        max_features = params["max_features"]
        
        # Only process actual features (not padding)
        if actual_features < num_features:
            X_actual = X[:, :actual_features]
            X_padding = X[:, actual_features:]
        else:
            X_actual = X
            X_padding = None
        
        # Outlier removal and standardization on actual features
        X_actual = outlier_removing(X_actual, threshold=4)
        X_actual = standard_scaling(X_actual)
        
        # Reconstruct with padding
        if X_padding is not None:
            X = torch.cat([X_actual, X_padding], dim=1)
        else:
            X = X_actual
        
        # Permute features if specified (only actual features)
        if params.get("permute_features", True):
            generator = params["generator"]
            perm = torch.randperm(actual_features, device=X.device, generator=generator)
            # Create full permutation including padding
            full_perm = torch.cat([perm, torch.arange(actual_features, num_features, device=X.device)])
            X = X[:, full_perm]
        
        # Scale by the proportion of features used relative to max features
        if params.get("scale_by_max_features", False):
            scaling_factor = actual_features / max_features
            X[:, :actual_features] = X[:, :actual_features] / scaling_factor
        
        # Add empty features if needed to match max features
        if num_features < max_features:
            X = F.pad(X, (0, max_features - num_features), mode="constant", value=0.0)
        
        return X
    
    def get_prior(self, generator: Optional[torch.Generator] = None) -> str:
        """
        Return the prior type name.
        
        Returns
        -------
        str
            The prior type name: 'gmires'
        """
        return "gmires"
