"""
GMIRes (Gaussian Mixture with Inverse Representation Sampling)

This module implements the Zeus-style data generation approach, which combines:
1. Gaussian Mixture Model with categorical features
2. Optional random network transformation (SpectralNormBlock-based)

Based on the original Zeus codebase (zeus/datasets.py).
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from typing import Tuple, Optional, Dict, Any
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA


class SpectralNormBlock(nn.Module):
    """A spectral-normalized transformation block with controlled singular values.
    
    This block implements: f(x) = x + h * A * ReLU(B * x) / std(output)
    where A and B have controlled spectral properties through SVD initialization.
    
    Parameters
    ----------
    dim : int
        Input and output dimension
    h : float, default=0.5
        Step size parameter for the transformation
    generator : torch.Generator, optional
        Random number generator for reproducibility
    """
    
    def __init__(self, dim: int, h: float = 0.5, generator: Optional[torch.Generator] = None):
        super().__init__()
        self.h = h
        self.dim = dim
        self.generator = generator

        # Matrix B: input -> hidden (4x expansion)
        hid_dim = 4 * dim
        B = torch.randn((dim, hid_dim), generator=generator)
        U, S, U_T = torch.linalg.svd(B, full_matrices=False)
        # Singular values between 0.5 and 1.0
        S_new = torch.rand(*S.shape, generator=generator) * 0.5 + 0.5
        S_new = torch.sort(S_new)[0].flip(dims=[0])
        self.B = nn.Parameter(U @ torch.diag(S_new) @ U_T, requires_grad=False)

        # Matrix A: hidden -> output
        A = torch.randn((hid_dim, dim), generator=generator)
        U, S, U_T = torch.linalg.svd(A, full_matrices=False)
        # Singular values between 0.75 and 1.0
        S_new = torch.rand(*S.shape, generator=generator) * 0.25 + 0.75
        S_new = torch.sort(S_new)[0].flip(dims=[0])
        self.A = nn.Parameter(U @ torch.diag(S_new) @ U_T, requires_grad=False)

    def forward(self, x):
        """Apply the spectral-normalized transformation."""
        out = x @ self.B
        out = torch.nn.functional.relu(out)
        out = out @ self.A
        return x + out / (torch.max(torch.std(out, dim=0, keepdim=True)) + 1e-8)


class RandomNetwork(nn.Module):
    """A random neural network for transforming Gaussian mixture data.
    
    Applies a sequence of SpectralNormBlocks with normalization between layers.
    
    Parameters
    ----------
    dim : int
        Continuous feature dimension
    n_components : int
        Number of Gaussian components (for one-hot encoding)
    num_blocks : int, default=3
        Number of transformation blocks
    h : float, default=0.5
        Step size parameter
    generator : torch.Generator, optional
        Random number generator
    """
    
    def __init__(
        self,
        dim: int,
        n_components: int,
        num_blocks: int = 3,
        h: float = 0.5,
        generator: Optional[torch.Generator] = None
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            SpectralNormBlock(dim + n_components, h=h, generator=generator)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        """Apply the random network transformation."""
        for block in self.blocks:
            x = block(x)
            # Normalize after each block
            x = (x - torch.mean(x, dim=0, keepdim=True)) / (torch.std(x, dim=0, keepdim=True) + 1e-8)
        return x


class GMIRes(nn.Module):
    """Gaussian Mixture with Inverse Representation Sampling (Zeus-style continuous feature generator).
    
    This class generates continuous features from a Gaussian mixture model, with optional
    random network transformation. It follows the same design pattern as GaussianMixture:
    parameters are provided at initialization, and forward() generates data.
    
    Note: This class only generates continuous features. Use CategoricalFeatureGenerator
    separately for categorical features, then combine in the Prior.
    
    Parameters
    ----------
    seq_len : int
        Number of samples to generate
    num_classes : int
        Number of Gaussian components/clusters
    continuous_dim : int
        Number of continuous dimensions
    weights : torch.Tensor
        Mixture weights of shape (num_classes,)
    means : torch.Tensor
        Gaussian means of shape (num_classes, continuous_dim)
    covariances : torch.Tensor
        Gaussian covariances of shape (num_classes, continuous_dim, continuous_dim)
    mode : str, default="gaussian"
        Generation mode: "gaussian" or "transformed"
    num_blocks : int, default=5
        Number of blocks in RandomNetwork (for transformed mode)
    h : float, default=0.5
        SpectralNormBlock step size parameter (for transformed mode)
    device : str, default="cpu"
        Computing device
    generator : torch.Generator, optional
        Random number generator for reproducibility
    **kwargs
        Additional unused parameters
    """
    
    def __init__(
        self,
        seq_len: int,
        num_classes: int,
        continuous_dim: int,
        weights: torch.Tensor,
        means: torch.Tensor,
        covariances: torch.Tensor,
        mode: str = "gaussian",
        num_blocks: int = 5,
        h: float = 0.5,
        device: str = "cpu",
        generator: Optional[torch.Generator] = None,
        **kwargs: Dict[str, Any],
    ):
        super(GMIRes, self).__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.continuous_dim = continuous_dim
        self.mode = mode
        self.num_blocks = num_blocks
        self.h = h
        self.device = device
        self.generator = generator if generator is not None else torch.Generator(device=device)
        
        # Set mixture weights
        assert weights.shape == (num_classes,), f"weights must have shape ({num_classes},)"
        self.weights = weights.to(device)
        
        # Set cluster means
        assert means.shape == (num_classes, continuous_dim), \
            f"means must have shape ({num_classes}, {continuous_dim})"
        self.means = means.to(device)
        
        # Set cluster covariances
        assert covariances.shape == (num_classes, continuous_dim, continuous_dim), \
            f"covariances must have shape ({num_classes}, {continuous_dim}, {continuous_dim})"
        self.covariances = covariances.to(device)
        
    def _generate_gaussian_mixture(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate continuous features from Gaussian mixture using pre-sampled parameters.
        
        Returns
        -------
        X : torch.Tensor
            Generated continuous features (seq_len, continuous_dim)
        y : torch.Tensor
            Cluster labels (seq_len,)
        """
        # Deterministically allocate samples per cluster based on weights
        # This ensures total samples = seq_len exactly
        samples_per_cluster = (self.weights * self.seq_len).long()
        
        # Handle rounding errors: add remaining samples to first cluster
        remaining = self.seq_len - samples_per_cluster.sum()
        if remaining > 0:
            samples_per_cluster[0] += remaining
        elif remaining < 0:
            # Should not happen, but handle it just in case
            samples_per_cluster[0] += remaining
        
        # Generate cluster assignments
        cluster_indices = torch.empty(self.seq_len, dtype=torch.long, device=self.device)
        current_idx = 0
        for cluster in range(self.num_classes):
            num_points = samples_per_cluster[cluster].item()
            cluster_indices[current_idx:current_idx + num_points] = cluster
            current_idx += num_points
        
        # Shuffle cluster assignments to avoid ordering bias
        perm = torch.randperm(self.seq_len, generator=self.generator, device=self.device)
        cluster_indices = cluster_indices[perm]
        
        # Generate continuous features from Gaussian mixture
        X_continuous = torch.empty(self.seq_len, self.continuous_dim, device=self.device)
        for cluster in range(self.num_classes):
            mask = (cluster_indices == cluster)
            num_points = mask.sum().item()
            if num_points > 0 and self.continuous_dim > 0:
                # Use Cholesky decomposition for sampling
                L = torch.linalg.cholesky(self.covariances[cluster])
                standard_normal = torch.randn(
                    num_points, self.continuous_dim, device=self.device, generator=self.generator
                )
                samples = self.means[cluster] + (L @ standard_normal.unsqueeze(-1)).squeeze(-1)
                X_continuous[mask] = samples
        
        # Standardize continuous features using pure tensor operations
        if self.continuous_dim > 0:
            # StandardScaler: (X - mean) / std
            mean = X_continuous.mean(dim=0, keepdim=True)
            std = X_continuous.std(dim=0, keepdim=True)
            X_continuous = (X_continuous - mean) / (std + 1e-8)
        
        return X_continuous, cluster_indices
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate continuous features using pre-sampled Gaussian parameters.
        
        Returns
        -------
        X : torch.Tensor
            Generated continuous features of shape (seq_len, continuous_dim)
        y : torch.Tensor
            Cluster labels of shape (seq_len,)
        """
        # Generate base Gaussian mixture (continuous features only)
        X, y = self._generate_gaussian_mixture()
        
        # Apply transformation if in transformed mode
        if self.mode == "transformed" and self.continuous_dim > 0:
            # Normalize continuous features again for transformation using tensor ops
            min_val = X.min(dim=0, keepdim=True)[0]
            max_val = X.max(dim=0, keepdim=True)[0]
            X = 2 * (X - min_val) / (max_val - min_val + 1e-8) - 1
            
            # Build and apply random network
            net = RandomNetwork(
                self.continuous_dim, self.num_classes, 
                num_blocks=self.num_blocks, h=self.h, generator=self.generator
            )
            net = net.to(self.device)
            
            # Create one-hot encoding of labels
            y_one_hot = torch.zeros(y.shape[0], self.num_classes, device=self.device)
            y_one_hot[torch.arange(y.shape[0]), y] = 1
            
            # Apply random network transformation
            X_with_classes = torch.cat((X, y_one_hot), dim=1)
            X_transformed = net(X_with_classes)
            
            # Apply PCA to maintain dimensionality (still need sklearn for PCA)
            pca = PCA(n_components=self.continuous_dim)
            X_transformed = pca.fit_transform(X_transformed.cpu().numpy())
            X = torch.tensor(X_transformed, dtype=torch.float32, device=self.device)
            
            # Re-standardize using pure tensor operations
            # StandardScaler: (X - mean) / std
            mean = X.mean(dim=0, keepdim=True)
            std = X.std(dim=0, keepdim=True)
            X = (X - mean) / (std + 1e-8)
        
        return X, y
