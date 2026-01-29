from __future__ import annotations

from typing import Optional, List, Tuple
import torch
from torch import nn, Tensor

from .embedding import ColEmbedding
from .interaction import RowInteraction
from .learning import ClusterLearningIMAB
from .decision import GMatrixFingerprintKDecision
from configs.inference_config import InferenceConfig


class TabClusterIMAB(nn.Module):
    """A Tabular In-Context Learning Foundation Model.

    TabCluster is a transformer-based architecture for in-context learning on tabular data to make
    predictions without fine-tuning. It processes tabular data through three sequential stages:

    1. Column-wise embedding creates distribution-aware embeddings
    2. Row-wise interaction captures interactions between features within each row
    3. Cluster learning with induced decoder encoder and attention-based classification

    Parameters
    ----------
    max_classes : int, default=5
        Number of classes that the model supports natively. 

    embed_dim : int, default=128
        Model dimension used in the column / row embedding transformers. For the cluster
        learning, the dimension is this value multiplied by the number of CLS tokens.

    col_num_blocks : int, default=3
        Number of induced self-attention blocks in the column embedding transformer

    col_nhead : int, default=4
        Number of attention heads in the column embedding transformer

    col_num_inds : int, default=128
        Number of inducing points in the column embedding transformer

    row_num_blocks : int, default=3
        Number of attention blocks in the row interaction transformer

    row_nhead : int, default=8
        Number of attention heads in the row interaction transformer

    row_num_cls : int, default=4
        Number of learnable CLS tokens used to aggregate feature information per row

    row_rope_base : float, default=100000
        Base scaling factor for rotary position encoding in the row interaction transformer

    cluster_num_blocks : int, default=6
        Number of induced decoder blocks in the cluster learning module

    cluster_nhead : int, default=4
        Number of attention heads in the cluster learning module

    cluster_use_rope_cross_attn : bool, default=False
        If True, use rotary positional encoding in cross-attention when ind_vectors_hidden queries src

    cluster_rope_base : float, default=100000
        Base scaling factor for rotary position encoding in cluster learning module (only used if cluster_use_rope_cross_attn=True)

    cluster_use_representation_self_att : bool, default=False
        If True, Stage 2 uses decoder block (self-attention + cross-attention);
        If False, Stage 2 uses only cross-attention (more efficient)

    ff_factor : int, default=2
        Expansion factor for feedforward networks across all components

    dropout : float, default=0.0
        Dropout probability across all components

    activation : str or unary callable, default="gelu"
        Activation function used throughout the model

    norm_first : bool, default=True
        If True, uses pre-norm architecture across all components
    """

    def __init__(
        self,
        max_classes: int = 10,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        col_nhead: int = 4,
        col_num_inds: int = 128,
        row_num_blocks: int = 3,
        row_nhead: int = 8,
        row_num_cls: int = 4,
        row_rope_base: float = 100000,
        cluster_num_blocks: int = 6,
        cluster_nhead: int = 4,
        cluster_use_rope_cross_attn: bool = False,
        cluster_rope_base: float = 100000,
        cluster_use_representation_self_att: bool = False,
        ff_factor: int = 2,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.max_classes = max_classes
        self.embed_dim = embed_dim
        self.col_num_blocks = col_num_blocks
        self.col_nhead = col_nhead
        self.col_num_inds = col_num_inds
        self.row_num_blocks = row_num_blocks
        self.row_nhead = row_nhead
        self.row_num_cls = row_num_cls
        self.row_rope_base = row_rope_base
        self.cluster_num_blocks = cluster_num_blocks
        self.cluster_nhead = cluster_nhead
        self.cluster_use_rope_cross_attn = cluster_use_rope_cross_attn
        self.cluster_rope_base = cluster_rope_base
        self.cluster_use_representation_self_att = cluster_use_representation_self_att
        self.ff_factor = ff_factor
        self.dropout = dropout
        self.activation = activation
        self.norm_first = norm_first

        self.col_embedder = ColEmbedding(
            embed_dim=embed_dim,
            num_blocks=col_num_blocks,
            nhead=col_nhead,
            num_inds=col_num_inds,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            reserve_cls_tokens=row_num_cls,
        )

        self.row_interactor = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=row_num_blocks,
            nhead=row_nhead,
            num_cls=row_num_cls,
            rope_base=row_rope_base,
            dim_feedforward=embed_dim * ff_factor,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )

        # Cluster learning with induced decoder encoder
        dataset_dim = embed_dim * row_num_cls  # CLS tokens are concatenated
        self.cluster_learner = ClusterLearningIMAB(
            max_classes=max_classes,
            d_model=dataset_dim,
            num_blocks=cluster_num_blocks,
            nhead=cluster_nhead,
            dim_feedforward=dataset_dim * ff_factor,
            num_inds=max_classes,  # num_inds must equal max_classes
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            use_rope_cross_attn=cluster_use_rope_cross_attn,
            rope_base=cluster_rope_base,
            use_representation_self_att=cluster_use_representation_self_att,
        )
        
        # K decision module: predicts number of clusters from logits
        self.k_decision = GMatrixFingerprintKDecision(
            max_classes=max_classes,
            activation=activation if isinstance(activation, str) else "gelu",
        )
    
    def _extract_logits_from_all_k(
        self,
        all_k_logits: list,
        all_query_norms: Optional[list],
        all_key_norms: Optional[list],
        num_classes: Tensor,
        row_representations: Tensor,
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Extract logits for specified num_classes from pre-computed all_k_logits.
        
        Optimization for inference: reuse already computed logits instead of recomputing.
        
        Returns: (logits, query_norm, key_norm)
        """
        B, T = row_representations.shape[0], row_representations.shape[1]
        device = row_representations.device
        
        logits = torch.full((B, T, self.max_classes), float('-inf'), 
                           dtype=row_representations.dtype, device=device)
        
        return_hidden = all_query_norms is not None
        query_norm_list = [] if return_hidden else None
        key_norm_list = [] if return_hidden else None
        
        for b in range(B):
            k = num_classes[b].item()
            k_idx = k - 2  # all_k_logits[0] -> k=2
            logits[b, :, :k] = all_k_logits[k_idx][b, :, :]
            if return_hidden:
                query_norm_list.append(all_query_norms[k_idx][b])
                key_norm_list.append(all_key_norms[k_idx][b])
        
        query_norm = torch.stack(query_norm_list, dim=0) if return_hidden else None
        key_norm = torch.stack(key_norm_list, dim=0) if return_hidden else None
        
        return logits, query_norm, key_norm

    def _compute_all_k_logits(
        self, 
        row_representations: Tensor,
        mgr_config = None,
        return_hidden: bool = False,
    ) -> list | tuple[list, list, list]:
        """Compute logits for all possible num_classes values (2 to max_classes).
        
        This method computes logits for each k value using eval mode to ensure:
        1. Consistent outputs across different k values (no dropout randomness)
        2. Deterministic K prediction during both training and inference
        
        The logits are computed with torch.no_grad() since they will be detached
        anyway when computing G matrices for K prediction.
        
        Parameters
        ----------
        row_representations : Tensor
            Row representations of shape (B, T, D)
        mgr_config : MgrConfig, optional
            Configuration for InferenceManager (used in inference mode)
        return_hidden : bool, default=False
            If True, also return query_norm and key_norm for each k value.
            
        Returns
        -------
        list or tuple[list, list, list]
            If return_hidden is False:
                List of logits for k=2,...,max_classes.
                Each logits[i] has shape (B, T, k) where k = i + 2
            If return_hidden is True:
                Tuple of (all_logits, all_query_norms, all_key_norms) where:
                - all_logits: List of logits for k=2,...,max_classes
                - all_query_norms: List of query_norm for k=2,...,max_classes
                - all_key_norms: List of key_norm for k=2,...,max_classes
        """
        B = row_representations.shape[0]
        device = row_representations.device
        
        all_logits = []
        all_query_norms = [] if return_hidden else None
        all_key_norms = [] if return_hidden else None
        
        # Save original training state to restore later
        was_training = self.cluster_learner.training
        
        # Use eval mode for consistent K prediction (disable dropout)
        self.cluster_learner.eval()
        
        with torch.no_grad():
            # Iterate over k = 2, 3, ..., max_classes
            for k in range(2, self.max_classes + 1):
                # Create uniform num_classes tensor for this k
                num_classes_k = torch.full((B,), k, dtype=torch.long, device=device)
                
                # Call cluster_learner in eval mode
                if return_hidden:
                    logits_k, query_norm_k, key_norm_k = self.cluster_learner(
                        row_representations, 
                        num_classes=num_classes_k,
                        mgr_config=mgr_config,
                        return_hidden=True, 
                    )
                    all_query_norms.append(query_norm_k)
                    all_key_norms.append(key_norm_k)
                else:
                    logits_k = self.cluster_learner(
                        row_representations, 
                        num_classes=num_classes_k,
                        mgr_config=mgr_config,
                        return_hidden=False, 
                    )
                
                # Only keep the first k columns of logits (others are -inf anyway)
                # logits_k shape: (B, T, max_classes) -> (B, T, k)
                logits_k = logits_k[:, :, :k]
                all_logits.append(logits_k)

        # Restore original training state
        if was_training:
            self.cluster_learner.train()
        
        if return_hidden:
            return all_logits, all_query_norms, all_key_norms
        return all_logits

    def _train_forward(
        self, X: Tensor, d: Optional[Tensor] = None, num_classes: Optional[Tensor] = None, 
        return_hidden: bool = False, predict_k: bool = True
    ) -> Tensor | tuple[Tensor, ...]:
        """Column-wise embedding -> row-wise interaction -> cluster learning.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H)

        d : Optional[Tensor], default=None
            The number of features per dataset.

        num_classes : Optional[Tensor], default=None
            Number of valid classes for each sample, shape (B,). When provided:
            - Creates mask to exclude invalid inducing vectors from attention
            - Sets logits for invalid classes to -inf

        return_hidden : bool, default=False
            If True, return logits, query_norm, key_norm.
            When predict_k is True, also returns all_query_norms, all_key_norms for each k.

        predict_k : bool, default=True
            If True, compute all k logits and predict k using MLP.
            This requires running the encoder max_classes-1 times.

        Returns
        -------
        Tensor or tuple[Tensor, ...]
            When predict_k is True:
                - return_hidden=False: (logits, k_logits, all_k_logits)
                - return_hidden=True: (logits, k_logits, all_k_logits, query_norm, key_norm, all_query_norms, all_key_norms)
            When predict_k is False:
                Original behavior based on return_hidden
        """
        B, T, H = X.shape
        if d is not None and len(d.unique()) == 1 and d[0] == H:
            d = None

        row_representations = self.row_interactor(self.col_embedder(X, d=d), d=d)
        
        if predict_k:
            # Compute logits for all k values
            # When return_hidden=True, also get query_norm and key_norm for each k
            if return_hidden:
                all_k_logits, all_query_norms, all_key_norms = self._compute_all_k_logits(
                    row_representations, return_hidden=True
                )
            else:
                all_k_logits = self._compute_all_k_logits(row_representations, return_hidden=False)
            
            # Compute G matrices and k_logits
            k_logits = self.k_decision(all_k_logits)  # (B, max_classes - 1)
            
            # Get logits for the true num_classes (for ARI loss)
            # Also run the encoder with true num_classes to ensure correct masking
            if return_hidden:
                logits, query_norm, key_norm = self.cluster_learner(
                    row_representations, num_classes=num_classes, 
                    return_hidden=True
                )
                return logits, k_logits, all_k_logits, query_norm, key_norm, all_query_norms, all_key_norms
            else:
                logits = self.cluster_learner(
                    row_representations, num_classes=num_classes, 
                    return_hidden=False
                )
                return logits, k_logits, all_k_logits
        else:
            # Original behavior without k prediction
            return self.cluster_learner(
                row_representations, num_classes=num_classes, 
                return_hidden=return_hidden
            )

    def _inference_forward(
        self,
        X: Tensor,
        num_classes: Optional[Tensor] = None,
        feature_shuffles: Optional[List[List[int]]] = None,
        inference_config: InferenceConfig = None,
        return_hidden: bool = False,
        predict_k: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        """Column-wise embedding -> row-wise interaction -> cluster learning.
        
        Parameters
        ----------
        predict_k : bool, default=False
            If True, compute all k logits and predict k using MLP.
            Uses InferenceManager for each k value for efficient batch splitting.
        
        Returns
        -------
        Tensor or tuple[Tensor, ...]
            When predict_k is False:
                If return_hidden is False:
                    logits: (B, T, max_classes)
                If return_hidden is True:
                    tuple of (logits, query_norm, key_norm)
            When predict_k is True:
                - return_hidden=False: (logits, k_logits, all_k_logits)
                - return_hidden=True: (logits, k_logits, all_k_logits, query_norm, key_norm, all_query_norms, all_key_norms)
        """
        if inference_config is None:
            inference_config = InferenceConfig()

        row_representations = self.row_interactor(
            self.col_embedder(X, feature_shuffles=feature_shuffles, mgr_config=inference_config.COL_CONFIG),
            mgr_config=inference_config.ROW_CONFIG,
        )
        
        if predict_k:
            # Compute logits for all k values
            # When return_hidden=True, also get query_norm and key_norm for each k
            if return_hidden:
                all_k_logits, all_query_norms, all_key_norms = self._compute_all_k_logits(
                    row_representations, mgr_config=inference_config.ROW_CONFIG, return_hidden=True
                )
            else:
                all_k_logits = self._compute_all_k_logits(
                    row_representations, mgr_config=inference_config.ROW_CONFIG, return_hidden=False
                )
            
            # Compute G matrices and k_logits
            k_logits = self.k_decision(all_k_logits)  # (B, max_classes - 1)
            
            # Optimization: Reuse logits from all_k_logits instead of recomputing
            # (Only for inference, as _compute_all_k_logits uses torch.no_grad)
            if return_hidden:
                # Extract from cached all_k_logits, all_query_norms, all_key_norms
                logits, query_norm, key_norm = self._extract_logits_from_all_k(
                    all_k_logits, all_query_norms, all_key_norms, num_classes, row_representations
                )
                return logits, k_logits, all_k_logits, query_norm, key_norm, all_query_norms, all_key_norms
            else:
                # Extract from cached all_k_logits
                logits, _, _ = self._extract_logits_from_all_k(
                    all_k_logits, None, None, num_classes, row_representations
                )
                return logits, k_logits, all_k_logits
        else:
            return self.cluster_learner(
                row_representations, num_classes=num_classes, 
                mgr_config=inference_config.ROW_CONFIG, 
                return_hidden=return_hidden
            )

    def forward(
        self,
        X: Tensor,
        d: Optional[Tensor] = None,
        num_classes: Optional[Tensor] = None,
        feature_shuffles: Optional[List[List[int]]] = None,
        inference_config: InferenceConfig = None,
        return_hidden: bool = False,
        predict_k: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        """Column-wise embedding -> row-wise interaction -> cluster learning.
        
        Parameters
        ----------
        X : Tensor
            Input tensor of shape (B, T, H)
            
        d : Optional[Tensor], default=None
            The number of features per dataset. Used only in training mode.
            
        num_classes : Optional[Tensor], default=None
            Number of valid classes for each sample, shape (B,). When provided:
            - Creates mask to exclude invalid inducing vectors from attention
            - Sets logits for invalid classes to -inf
            
        feature_shuffles : Optional[List[List[int]]], default=None
            A list of feature shuffle patterns. Used only in inference mode.
            
        inference_config : InferenceConfig, default=None
            Configuration for inference. Used only in inference mode.
            
        return_hidden : bool, default=False
            If True, return logits, query_norm, key_norm.
            
        predict_k : bool, default=False
            If True, compute all k logits and predict k using MLP.
            Uses InferenceManager for efficient batch splitting during K prediction.
        
        Returns
        -------
        Tensor or tuple[Tensor, ...]
            When predict_k is False:
                If return_hidden is False:
                    logits: (B, T, max_classes) - invalid classes have -inf when num_classes provided
                If return_hidden is True:
                    tuple of (logits, query_norm, key_norm, [hidden_states])
            When predict_k is True:
                Base: (logits, k_logits, all_k_logits)
                + return_hidden: (..., query_norm, key_norm)
        """
        if self.training:
            return self._train_forward(
                X, d=d, num_classes=num_classes, 
                return_hidden=return_hidden,
                predict_k=predict_k
            )
        else:
            return self._inference_forward(
                X, num_classes=num_classes, 
                feature_shuffles=feature_shuffles, inference_config=inference_config, 
                return_hidden=return_hidden,
                predict_k=predict_k
            )
