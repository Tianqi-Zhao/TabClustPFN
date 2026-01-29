"""
Categorical feature generator for synthetic datasets.

This module provides a standalone categorical feature generator that can be
combined with continuous feature generators.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Optional, Tuple


class CategoricalFeatureGenerator(nn.Module):
    """Generates categorical features for each cluster in a mixture model.
    
    This module generates categorical feature labels where each cluster
    has different probability distributions over categories, creating cluster-specific
    categorical patterns.
    
    Parameters
    ----------
    seq_len : int
        Number of samples to generate
    num_classes : int
        Number of clusters/classes
    categorical_dims : list of int
        List of category counts for each categorical feature (e.g., [3, 5, 2])
    device : str, default="cpu"
        Computing device
    generator : torch.Generator, optional
        Random number generator for reproducibility
    
    Examples
    --------
    >>> # Generate categorical features for 3 categorical variables with 3, 5, 2 categories
    >>> cat_gen = CategoricalFeatureGenerator(
    ...     seq_len=100,
    ...     num_classes=3,
    ...     categorical_dims=[3, 5, 2],
    ...     device="cpu"
    ... )
    >>> cluster_labels = torch.randint(0, 3, (100,))  # Cluster assignments
    >>> X_cat = cat_gen(cluster_labels)  # Shape: (100, 3) - category labels for 3 features
    """
    
    def __init__(
        self,
        seq_len: int,
        num_classes: int,
        categorical_dims: list,
        device: str = "cpu",
        generator: Optional[torch.Generator] = None,
    ):
        super(CategoricalFeatureGenerator, self).__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.categorical_dims = categorical_dims  # e.g., [3, 5, 2]
        self.num_categorical_features = len(categorical_dims)  # Number of categorical features
        self.device = device
        self.generator = generator if generator is not None else torch.Generator(device=device)
        
        # Pre-sample category probability distributions for each cluster
        # This ensures consistent categorical patterns for each cluster
        self._sample_cluster_probs()
    
    def _sample_cluster_probs(self):
        """Pre-sample Dirichlet distributions for each cluster and categorical feature.
        
        Each cluster has different biases towards certain categories, creating
        cluster-specific categorical patterns.
        """
        # Create numpy RNG for Dirichlet sampling
        seed = torch.empty((), dtype=torch.int64, device='cpu').random_(generator=self.generator).item()
        rng = np.random.default_rng(seed)
        
        # Store probability distributions for each (cluster, categorical_feature) pair
        # Shape: list of length num_categorical_features, each element is (num_classes, n_categories)
        self.cluster_probs = []
        
        for n_categories in self.categorical_dims:
            # Sample probability distribution for each cluster
            probs_for_clusters = []
            for _ in range(self.num_classes):
                # Use Dirichlet with alpha=0.5 to create diverse probability distributions
                # Lower alpha creates more peaked distributions (some categories strongly preferred)
                probs = rng.dirichlet(alpha=[0.5] * n_categories)
                probs_for_clusters.append(probs)
            
            self.cluster_probs.append(np.array(probs_for_clusters))
    
    def forward(self, cluster_labels: torch.Tensor) -> torch.Tensor:
        """Generate categorical features based on cluster assignments.
        
        Parameters
        ----------
        cluster_labels : torch.Tensor
            Cluster assignments of shape (seq_len,)
        
        Returns
        -------
        torch.Tensor
            Categorical feature labels of shape (seq_len, num_categorical_features)
            Each column contains category labels (integers) for one categorical feature
        """
        if len(self.categorical_dims) == 0:
            # No categorical features
            return torch.empty(self.seq_len, 0, dtype=torch.long, device=self.device)
        
        # Create numpy RNG for categorical sampling
        seed = torch.empty((), dtype=torch.int64, device='cpu').random_(generator=self.generator).item()
        rng = np.random.default_rng(seed)
        
        categorical_features = []
        
        for cat_idx, n_categories in enumerate(self.categorical_dims):
            # Generate categorical labels for this feature
            cat_labels = torch.zeros(self.seq_len, dtype=torch.long, device=self.device)
            
            for cluster in range(self.num_classes):
                mask = (cluster_labels == cluster)
                num_points = mask.sum().item()
                
                if num_points > 0:
                    # Use pre-sampled probability distribution for this cluster
                    probs = self.cluster_probs[cat_idx][cluster]
                    
                    # Sample categorical values
                    cat_values = rng.choice(n_categories, size=num_points, p=probs)
                    
                    # Store as labels (integers)
                    cat_labels[mask] = torch.tensor(cat_values, dtype=torch.long, device=self.device)
            
            categorical_features.append(cat_labels.unsqueeze(1))  # Add feature dimension
        
        # Concatenate all categorical features
        return torch.cat(categorical_features, dim=1)
    
    def get_feature_info(self) -> dict:
        """Get information about the categorical features.
        
        Returns
        -------
        dict
            Dictionary containing:
            - num_features: number of categorical features
            - dims: list of category counts (max value for each feature)
            - dtype: torch.long (categorical labels)
        """
        return {
            "num_features": len(self.categorical_dims),
            "dims": self.categorical_dims,
            "dtype": torch.long,
        }
