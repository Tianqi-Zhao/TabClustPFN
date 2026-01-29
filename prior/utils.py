from __future__ import annotations

import random
import numpy as np
import time
from scipy.linalg import sqrtm

import torch
from torch import nn
from typing import Optional, Tuple
from torch import Tensor
import warnings


class GaussianNoise(nn.Module):
    def __init__(self, std, generator: Optional[torch.Generator] = None):
        super().__init__()
        self.std = std
        self.generator = generator

    def forward(self, X):
        return X + torch.normal(torch.zeros_like(X), self.std, generator=self.generator)


def sample_gaussian_parameters(
    num_classes: int,
    continuous_dim: int,
    min_distance: float,
    p1: float,
    p2: float,
    device: str,
    generator: torch.Generator,
    start_distance: Optional[float] = None,
    alpha: Optional[float] = None,
    alpha_binary: float = 1.5,
    alpha_multiclass: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample Gaussian mixture parameters (weights, means, covariances) following Zeus approach.
    
    Parameters
    ----------
    num_classes : int
        Number of Gaussian components
    continuous_dim : int
        Dimensionality of continuous features
    min_distance : float
        Minimum Wasserstein-2 distance between Gaussians
    p1 : float
        Lower bound for covariance eigenvalues
    p2 : float
        Upper bound for covariance eigenvalues
    device : str
        Computation device ('cpu' or 'cuda')
    generator : torch.Generator
        Random generator
    start_distance : float, optional
        If provided, distance is sampled between min_distance and start_distance
    alpha : float, optional
        Dirichlet concentration parameter for mixture weights.
        If provided, overrides alpha_binary/alpha_multiclass logic.
    alpha_binary : float, default=1.5
        Dirichlet alpha for binary classification (num_classes=2).
        Lower value allows more imbalanced cluster sizes.
    alpha_multiclass : float, default=5.0
        Dirichlet alpha for multi-class (num_classes>2).
        Higher value produces more balanced cluster sizes.
    
    Returns
    -------
    tuple
        (weights, means, covariances) as torch tensors
    """
    # Create numpy RNG
    # Get the device of the generator to avoid device mismatch
    gen_device = generator.device if hasattr(generator, 'device') else 'cpu'
    seed = torch.empty((), dtype=torch.int64, device=gen_device).random_(generator=generator).item()
    rng = np.random.default_rng(seed)
    
    means_list = []
    covs_list = []
    
    # Generate each Gaussian component
    for gaussian_idx in range(num_classes):
        # Determine distance threshold for this component
        cur_min_distance = min_distance if start_distance is None \
            else rng.uniform(min_distance, start_distance)
        
        # Generate covariance matrix with controlled eigenvalues
        random_matrix = rng.standard_normal((continuous_dim, continuous_dim))
        covariance_matrix = np.dot(random_matrix, random_matrix.T)
        
        cur_p2 = p2
        cur_p1 = p1 if rng.random() > 0.75 else p2 - p1
        
        U, S, U_T = np.linalg.svd(covariance_matrix)
        S_new = rng.random(S.shape[0]) * (cur_p2 - cur_p1) + cur_p1
        
        # Occasionally zero out some dimensions for diversity
        if rng.random() < 0.2:
            zero_out_dims = max(1, S_new.shape[0] // 3)
            chosen_idx = rng.integers(zero_out_dims) + 1
            S_new[:chosen_idx] = rng.random(chosen_idx) * p1
        S_new = np.sort(S_new)[::-1]
        
        cov = U @ np.diag(S_new) @ U_T
        sqrt_cov = sqrtm(cov)
        
        # Compute Wasserstein-2 distances to existing Gaussians
        traces = [
            np.trace(cov + cov_j - 2 * sqrtm(sqrt_cov @ cov_j @ sqrt_cov))
            for cov_j in covs_list
        ]
        
        # Find mean with minimum distance constraint
        found_mean = False
        retry_steps = 1
        while not found_mean:
            mean = np.zeros(continuous_dim)
            mean_dir = rng.uniform(-1, 1, size=continuous_dim)
            mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-8)
            eps = 0.003 * max(1, (50 - continuous_dim + 1)) * retry_steps * cur_min_distance
            
            for _ in range(500):
                mean += eps * mean_dir
                if all(
                    trace + np.linalg.norm(mean - m) ** 2 > cur_min_distance
                    for m, trace in zip(means_list, traces)
                ):
                    found_mean = True
                    break
            
            retry_steps *= 2  # Exponential growth like Zeus
        
        means_list.append(mean)
        covs_list.append(cov)
    
    # Convert to tensors
    means = torch.tensor(np.array(means_list), dtype=torch.float32, device=device)
    covariances = torch.tensor(np.array(covs_list), dtype=torch.float32, device=device)
    
    # Sample weights from Dirichlet distribution
    # This is independent of seq_len and is a cluster property
    # Higher alpha -> more balanced; Lower alpha -> allows imbalance
    # Use provided alpha, or select based on num_classes
    if alpha is not None:
        effective_alpha = alpha
    else:
        effective_alpha = alpha_binary if num_classes == 2 else alpha_multiclass
    
    weights_np = rng.dirichlet(alpha=[effective_alpha] * num_classes)
    weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
    
    return weights, means, covariances


def resample_weights_dirichlet(
    num_classes: int,
    generator: torch.Generator,
    alpha_binary: float = 1.5,
    alpha_multiclass: float = 5.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Resample mixture weights using Dirichlet distribution based on num_classes.
    
    This function generates balanced or imbalanced weights depending on the
    number of classes, useful for post-processing MixSim weights or generating
    new weights for GMM-based priors.
    
    Parameters
    ----------
    num_classes : int
        Number of mixture components
    generator : torch.Generator
        Random generator for reproducibility
    alpha_binary : float, default=1.5
        Dirichlet alpha for binary classification (num_classes=2).
        Lower value allows more imbalanced cluster sizes.
    alpha_multiclass : float, default=5.0
        Dirichlet alpha for multi-class (num_classes>2).
        Higher value produces more balanced cluster sizes.
    device : str, default="cpu"
        Device to place output tensor on
        
    Returns
    -------
    torch.Tensor
        Mixture weights of shape (num_classes,), summing to 1.0
    """
    gen_device = generator.device if hasattr(generator, 'device') else 'cpu'
    seed = torch.empty((), dtype=torch.int64, device=gen_device).random_(generator=generator).item()
    rng = np.random.default_rng(seed)
    
    # Select alpha based on num_classes
    effective_alpha = alpha_binary if num_classes == 2 else alpha_multiclass
    
    weights_np = rng.dirichlet(alpha=[effective_alpha] * num_classes)
    return torch.tensor(weights_np, dtype=torch.float32, device=device)


# rpy2 imports for MixSim usage
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import os
import logging
from typing import Dict, Tuple, Any

# Import R_HOME configuration
from configs.env_config import R_HOME_PATH

# Set R_HOME environment variable if not set, using the path from configuration
if 'R_HOME' not in os.environ:
    os.environ['R_HOME'] = R_HOME_PATH

# Global, process-local cache for the MixSim object
_mixsim_r = None

# Configure logger for MixSim operations
logger = logging.getLogger(__name__)

def generate_mixsim_params(
    num_classes: int,
    num_features: int,
    hom: bool,
    sph: bool,
    PiLow: float,
    int_vec: list[float],
    MaxOmega: float,
    ecc: float,
    device: str,
    generator: torch.Generator,
    resN: int = 50,
    max_retries: int = 5,
    enable_maxomega_adjustment: bool = True,
    maxomega_step: float = 0.02,
    maxomega_min: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates Gaussian mixture model parameters (weights, means, covariances) using the MixSim R package.

    This function uses MixSim's internal retry mechanism (resN parameter) to control the number
    of attempts for finding valid parameters. If MixSim fails after internal retries, this function
    can optionally adjust MaxOmega and retry with different seeds.

    Parameters
    ----------
    num_classes : int
        Number of Gaussian components.
    num_features : int
        Number of features for each sample.
    hom : bool
        Whether to generate homogeneous or heterogeneous covariance matrices.
    sph : bool
        Whether to generate spherical or ellipsoidal covariance matrices.
    PiLow : float
        Lower bound for mixture weights.
    int_vec : list[float]
        Interval for the uniform distribution for overlap.
    MaxOmega : float
        Maximum overlap value.
    ecc : float
        Eccentricity of the ellipsoidal clusters.
    device : str
        The computing device ('cpu' or 'cuda') where tensors will be allocated.
    generator : torch.Generator
        The random number generator to use for seeding.
    resN : int, default=50
        Number of search attempts within MixSim (MixSim's internal retry parameter).
        Higher values allow MixSim more time to find valid parameters.
    max_retries : int, default=5
        Maximum number of outer retries (with different seeds and potentially adjusted MaxOmega).
    enable_maxomega_adjustment : bool, default=True
        If True, gradually decrease MaxOmega on failure to improve success rate in higher dimensions.
    maxomega_step : float, default=0.02
        Amount to decrease MaxOmega on each retry (only if enable_maxomega_adjustment=True).
    maxomega_min : float, default=0.01
        Minimum allowed MaxOmega value when adjusting (only if enable_maxomega_adjustment=True).

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        A tuple containing:
        - weights (Tensor): Mixture weights for the Gaussian components.
        - means (Tensor): Mean vectors for each Gaussian component.
        - covariances (Tensor): Covariance matrices for each Gaussian component.
        
    Raises
    ------
    RuntimeError
        If unable to generate valid parameters after all retries.
        The caller should catch this exception and resample hyperparameters.
    """
    global _mixsim_r
    # Lazy import: Import MixSim only once per worker process
    if _mixsim_r is None:
        _mixsim_r = importr('MixSim')
    
    # Track current MaxOmega (may be adjusted during retries)
    current_maxomega = MaxOmega
    original_maxomega = MaxOmega

    def adjust_maxomega(cur_maxomega: float, retry_attempt: int) -> float:
        """Decrease MaxOmega for next retry; no-op if below min or adjustment disabled."""
        if not enable_maxomega_adjustment:
            return cur_maxomega
        if retry_attempt >= max_retries - 1:
            return cur_maxomega
        if cur_maxomega <= maxomega_min:
            return cur_maxomega
        new_maxomega = max(cur_maxomega - maxomega_step, maxomega_min)
        logger.debug(
            f"Adjusting MaxOmega downward: {cur_maxomega:.3f} -> {new_maxomega:.3f}"
        )
        return new_maxomega
    
    # Retry loop: try with potentially adjusted parameters
    for retry_attempt in range(max_retries):
        try:
            # Generate unique seed for each attempt
            gen_device = generator.device if hasattr(generator, 'device') else 'cpu'
            seed_64bit = torch.empty((), dtype=torch.int64, device=gen_device).random_(generator=generator).item()
            seed_32bit = seed_64bit % (2**32)
            if seed_32bit > (2**31 - 1):
                seed_32bit -= (2**32)
            
            robjects.r(f'set.seed({seed_32bit})')
            
            # Prepare MixSim arguments (use current_maxomega which may be adjusted)
            mixsim_kwargs = {
                "K": int(num_classes),
                "p": int(num_features),
                "hom": hom,
                "sph": sph,
                "PiLow": PiLow,
                "int": robjects.FloatVector(int_vec),
                "MaxOmega": current_maxomega,
                "ecc": ecc,
                "resN": int(resN),  # MixSim's internal retry parameter
            }
            
            # Call MixSim directly (MixSim handles timeout internally via resN)
            mixsim_params = _mixsim_r.MixSim(**mixsim_kwargs)
            
            # Check result
            if mixsim_params is not None and mixsim_params != robjects.NULL:
                # Successfully generated parameters
                if current_maxomega != original_maxomega:
                    logger.info(
                        f"MixSim (num_features: {num_features}) succeeded with adjusted MaxOmega, {original_maxomega:.3f} -> {current_maxomega:.3f} "
                        f"after {retry_attempt + 1} attempts"
                    )
                
                # Convert R objects to Python
                with localconverter(robjects.default_converter) as cv:
                    pi_py = np.array(mixsim_params.rx2('Pi'))
                    mu_py = np.array(mixsim_params.rx2('Mu'))
                    s_py = np.array(mixsim_params.rx2('S'))
                
                # Convert to torch tensors
                weights = torch.tensor(pi_py, device=device, dtype=torch.float32)
                means = torch.tensor(mu_py, device=device, dtype=torch.float32)
                covariances = torch.tensor(s_py.transpose(2, 0, 1), device=device, dtype=torch.float32)
                
                return weights, means, covariances
            else:
                # MixSim returned NULL (failed after resN internal attempts)
                logger.info(
                    f"MixSim returned NULL (retry {retry_attempt + 1}/{max_retries}, "
                    f"MaxOmega={current_maxomega:.3f}, num_features={num_features}, resN={resN})"
                )
                
                # Adjust MaxOmega for next retry if enabled (decrease for high-dim overlap difficulty)
                current_maxomega = adjust_maxomega(current_maxomega, retry_attempt)
                
        except Exception as e:
            logger.info(
                f"MixSim exception (retry {retry_attempt + 1}/{max_retries}, "
                f"MaxOmega={current_maxomega:.3f}, num_features={num_features}): {str(e)}"
            )
            
            # Adjust MaxOmega for next retry if enabled (decrease for high-dim overlap difficulty)
            current_maxomega = adjust_maxomega(current_maxomega, retry_attempt)
            
            continue
    
    # All retries failed
    error_msg = (
        f"MixSim failed to generate valid parameters after {max_retries} retries "
        f"(resN={resN} internal attempts per retry). "
        f"Params: K={num_classes}, p={num_features}, "
        f"MaxOmega={current_maxomega:.3f} , original={original_maxomega:.3f}). "
        f"Suggest: resample hyperparameters and retry."
    )
    logger.warning(error_msg)
    raise RuntimeError(error_msg)


def torch_nanstd(input, dim=None, keepdim=False, ddof=0, *, dtype=None) -> Tensor:
    """Calculates the standard deviation of a tensor, ignoring NaNs, using NumPy internally.

    Parameters
    ----------
    input : Tensor
        The input tensor.

    dim : int or tuple[int], optional
        The dimension or dimensions to reduce. Defaults to None (reduce all dimensions).

    keepdim : bool, optional
        Whether the output tensor has `dim` retained or not. Defaults to False.

    ddof : int, optional
        Delta Degrees of Freedom.

    dtype : torch.dtype, optional
        The desired data type of returned tensor. Defaults to None.

    Returns
    -------
    Tensor
        The standard deviation.
    """
    device = input.device
    # Detach tensor from computation graph before converting to numpy
    np_input = input.detach().cpu().numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        std = np.nanstd(np_input, axis=dim, dtype=dtype, keepdims=keepdim, ddof=ddof)

    return torch.from_numpy(std).to(dtype=torch.float, device=device)


def standard_scaling(input: Tensor, clip_value: float = 100) -> Tensor:
    """Standardizes features by removing the mean and scaling to unit variance.

    NaNs are ignored in mean/std calculation.

    Parameters
    ----------
    input : Tensor
        Input tensor of shape (T, H), where T is sequence length, H is features.

    clip_value : float, optional, default=100
        The value to clip the standardized input to, preventing extreme outliers.

    Returns
    -------
    Tensor
        The standardized input, clipped between -clip_value and clip_value.
    """
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
    scaled_input = (input - mean) / std

    return torch.clip(scaled_input, min=-clip_value, max=clip_value)


def outlier_removing(input: Tensor, threshold: float = 4.0) -> Tensor:
    """Clamps outliers in the input tensor based on a specified number of standard deviations (threshold).

    Parameters
    ----------
    input : Tensor
        Input tensor of shape (T, H).

    threshold : float, optional, default=4.0
        Number of standard deviations to use as the cutoff.

    Returns
    -------
    Tensor
        The tensor with outliers clamped.
    """
    # First stage: Identify outliers using initial statistics
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
    cut_off = std * threshold
    lower, upper = mean - cut_off, mean + cut_off

    # Create mask for non-outlier, non-NaN values
    mask = (lower <= input) & (input <= upper) & ~torch.isnan(input)

    # Second pass using only non-outlier values for mean/std
    masked_input = torch.where(mask, input, torch.nan)
    masked_mean = torch.nanmean(masked_input, dim=0)
    masked_std = torch_nanstd(masked_input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)

    # Handle cases where a column had <= 1 valid value after masking -> std is NaN or 0
    masked_mean = torch.where(torch.isnan(masked_mean), mean, masked_mean)
    masked_std = torch.where(torch.isnan(masked_std), torch.zeros_like(std), masked_std)

    # Recalculate cutoff with robust estimates
    cut_off = masked_std * threshold
    lower, upper = masked_mean - cut_off, masked_mean + cut_off

    # Replace NaN bounds with +/- inf
    lower = torch.nan_to_num(lower, nan=-torch.inf)
    upper = torch.nan_to_num(upper, nan=torch.inf)

    # Clamp the input to remove outliers
    result = input.clamp(min=lower, max=upper)
    
    return result



def permute_classes(input: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
    """Label encoding and permute classes.

    Parameters
    ----------
    input : Tensor
        Target of shape (T,) containing class labels.

    Returns
    -------
    Tensor
        Target with potentially permuted labels (T,).
    """
    unique_vals, _ = torch.unique(input, return_inverse=True)
    num_classes = len(unique_vals)

    if num_classes <= 1:  # No permutation needed for single class
        return input

    # Ensure labels are encoded from 0 to num_classes-1
    indices = unique_vals.argsort()
    mapped = indices[torch.searchsorted(unique_vals, input)]

    # Randomly permute classes
    perm = torch.randperm(num_classes, device=input.device, generator=generator)
    permuted = perm[mapped]

    return permuted

