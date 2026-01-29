"""K Decision Modules for predicting optimal number of clusters.

This module contains different strategies for predicting k (number of clusters)
from clustering logits. Each module implements the KDecisionModule interface.

To create a new K decision strategy:
1. Subclass KDecisionModule
2. Implement __init__ to define learnable parameters
3. Implement forward to compute k_logits from all_logits
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor


class KDecisionModule(nn.Module, ABC):
    """Abstract base class for K decision modules.
    
    K decision modules predict the optimal number of clusters (k) from
    a list of clustering logits for different k values.
    
    Parameters
    ----------
    max_classes : int
        Maximum number of classes supported (k ranges from 2 to max_classes)
    """
    
    def __init__(self, max_classes: int):
        super().__init__()
        self.max_classes = max_classes
    
    @abstractmethod
    def forward(
        self, all_logits: list, return_g_matrices: bool = False
    ) -> Tensor | tuple[Tensor, list]:
        """Compute k_logits from clustering logits.
        
        Parameters
        ----------
        all_logits : list
            List of logits for k=2,...,max_classes. 
            Each logits[i] has shape (B, T, k) where k = i + 2
        return_g_matrices : bool, default=False
            If True, also return intermediate representations (e.g., G matrices)
            
        Returns
        -------
        Tensor or tuple
            k_logits: (B, max_classes - 1) - logits for k=2,...,max_classes
            intermediate: list of intermediate representations (if return_g_matrices=True)
        """
        pass



class GMatrixFingerprintKDecision(KDecisionModule):
    """K decision module using sorted G matrix (P^T @ P) elements and MLP.
    
    Similar to GMatrixKDecision, but instead of flattening the entire G matrix,
    this module extracts diagonal and off-diagonal elements separately, sorts
    them, and concatenates.
    
    For each k in {2, ..., max_classes}:
    1. Compute G_k = P_k^T @ P_k (cluster similarity matrix, shape k x k)
    2. Extract diagonal elements (k elements), sort descending -> D_k
    3. Extract upper triangular off-diagonal elements (k*(k-1)/2 elements), sort descending -> O_k
    4. Concatenate: g_k = [D_k; O_k] (k + k*(k-1)/2 = k*(k+1)/2 elements)
    
    Then concatenate all g_k and use MLP to predict optimal k.
    
    Parameters
    ----------
    max_classes : int
        Maximum number of classes supported (k ranges from 2 to max_classes)
    hidden_dim : int, default=256
        Hidden dimension for the MLP
    dropout : float, default=0.0
        Dropout probability
    activation : str, default="gelu"
        Activation function ("gelu" or "relu")
    """
    
    def __init__(
        self,
        max_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__(max_classes)
        self.hidden_dim = hidden_dim
        
        # Total input dimension: sum of k*(k+1)/2 for k=2,...,max_classes
        # For each k: k diagonal elements + k*(k-1)/2 off-diagonal elements = k*(k+1)/2
        input_dim = sum(k * (k + 1) // 2 for k in range(2, max_classes + 1))
        
        # LayerNorm to normalize concatenated features before MLP
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Build MLP
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_classes - 1),  # Output: logits for k=2,...,max_classes
        )
    
    def forward(
        self, all_logits: list, return_g_matrices: bool = False, detach: bool = True
    ) -> Tensor | tuple[Tensor, list]:
        """Compute sorted G matrix elements and predict k using MLP.
        
        Parameters
        ----------
        all_logits : list
            List of logits for k=2,...,max_classes. 
            Each logits[i] has shape (B, T, k) where k = i + 2
        return_g_matrices : bool, default=False
            If True, also return the list of G matrices
        detach : bool, default=True
            If True, detach logits so K prediction loss only optimizes the MLP,
            not the cluster_learner. If False, gradients will propagate back
            through the cluster_learner.
            
        Returns
        -------
        Tensor or tuple
            k_logits: (B, max_classes - 1) - logits for k=2,...,max_classes
            g_matrices: list of G matrices (if return_g_matrices=True)
        """
        B = all_logits[0].shape[0]
        device = all_logits[0].device
        
        g_matrices = []
        features_list = []
        
        for logits_k in all_logits:
            # Optionally detach logits so K prediction loss only optimizes the MLP,
            # not the cluster_learner. This decouples the two tasks.
            logits_k_processed = logits_k.detach() if detach else logits_k
            
            # Compute P = softmax(logits_k)
            P = torch.softmax(logits_k_processed, dim=-1)  # (B, T, k)
            T = P.shape[1]
            k = P.shape[-1]
            
            # Compute G = P^T @ P / T: (B, k, T) @ (B, T, k) -> (B, k, k)
            # Divide by T to normalize, keeping G elements in [0, 1] regardless of T
            G = torch.bmm(P.transpose(1, 2), P) / T  # (B, k, k)
            
            if return_g_matrices:
                g_matrices.append(G)
            
            # Extract diagonal elements and sort descending
            diag = torch.diagonal(G, dim1=1, dim2=2)  # (B, k)
            diag_sorted, _ = torch.sort(diag, dim=-1, descending=True)  # (B, k)
            
            # Extract upper triangular (off-diagonal) elements and sort descending
            triu_indices = torch.triu_indices(k, k, offset=1, device=device)
            off_diag = G[:, triu_indices[0], triu_indices[1]]  # (B, k*(k-1)/2)
            off_diag_sorted, _ = torch.sort(off_diag, dim=-1, descending=True)  # (B, k*(k-1)/2)
            
            # Concatenate sorted diagonal and off-diagonal
            g_k = torch.cat([diag_sorted, off_diag_sorted], dim=-1)  # (B, k*(k+1)/2)
            features_list.append(g_k)
        
        # Concatenate all features
        features_concat = torch.cat(features_list, dim=-1)  # (B, sum(k*(k+1)/2))
        
        # Apply LayerNorm before MLP
        features_normalized = self.layer_norm(features_concat)
        
        # Predict k using MLP
        k_logits = self.mlp(features_normalized)  # (B, max_classes - 1)
        
        if return_g_matrices:
            return k_logits, g_matrices
        return k_logits