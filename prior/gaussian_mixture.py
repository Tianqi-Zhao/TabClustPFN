from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Tuple, Optional


class GaussianMixture(nn.Module):
    """Generates synthetic tabular datasets using a Gaussian Mixture Model.

    This class generates samples from a mixture of Gaussian distributions,
    similar to MLPSCM but using probabilistic clustering instead of causal models.

    Parameters
    ----------
    seq_len : int, default=1024
        The number of samples (rows) to generate for the dataset.

    num_features : int, default=100
        The number of features (dimensions) for each sample.

    num_classes : int, default=5
        The number of Gaussian clusters in the mixture model.

    device : str, default="cpu"
        The computing device ('cpu' or 'cuda') where tensors will be allocated.

    **kwargs : dict
        Unused hyperparameters passed from parent configurations.
    """

    def __init__(
        self,
        seq_len: int = 1024,
        num_features: int = 100,
        num_classes: int = 5,
        weights: torch.Tensor = None,
        means: torch.Tensor = None,
        covariances: torch.Tensor = None,
        device: str = "cpu",
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        super(GaussianMixture, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device
        self.generator = generator

        # Set mixture weights
        if weights is None:
            self.weights = torch.ones(num_classes, device=device) / num_classes
        else:
            assert weights.shape == (num_classes,), f"weights must have shape ({num_classes},)"
            self.weights = weights.to(device)

        # Set cluster means
        if means is None:
            self.means = torch.randn(num_classes, num_features, device=device, generator=self.generator) * 2.0
        else:
            assert means.shape == (num_classes, num_features), f"means must have shape ({num_classes}, {num_features})"
            self.means = means.to(device)

        # Set cluster covariances
        if covariances is None:
            cov = torch.randn(num_classes, num_features, num_features, device=device, generator=self.generator) * 0.1
            self.covariances = torch.matmul(cov, cov.transpose(-2, -1)) + torch.eye(num_features, device=device) * 0.1
        else:
            assert covariances.shape == (num_classes, num_features, num_features), f"covariances must have shape ({num_classes}, {num_features}, {num_features})"
            self.covariances = covariances.to(device)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates synthetic data samples from the Gaussian mixture model.

        Returns
        -------
        X : torch.Tensor
            Generated samples of shape (seq_len, num_features)
        y : torch.Tensor
            Cluster assignments of shape (seq_len,)
        """
        # Sample cluster assignments
        cluster_indices = torch.multinomial(self.weights, self.seq_len, replacement=True, generator=self.generator)

        # Generate samples for each cluster
        X = torch.empty(self.seq_len, self.num_features, device=self.device)
        for cluster in range(self.num_classes):
            mask = (cluster_indices == cluster)
            num_points = mask.sum().item()
            if num_points > 0:
                # Manually sample from multivariate normal to use the generator
                # Decompose covariance matrix using Cholesky factorization
                L = torch.linalg.cholesky(self.covariances[cluster])
                
                # Sample from standard normal distribution
                standard_normal_samples = torch.randn(
                    num_points, self.num_features, device=self.device, generator=self.generator
                )
                
                # Transform samples to the target distribution: mean + L @ standard_normal
                samples = self.means[cluster] + (L @ standard_normal_samples.unsqueeze(-1)).squeeze(-1)
                X[mask] = samples

        return X, cluster_indices