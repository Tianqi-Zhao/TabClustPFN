import torch
from torch import nn, Tensor
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional

from .encoders import InducedDecoderEncoder

from .inference import InferenceManager
from configs.inference_config import MgrConfig

class ClusterLearningIMAB(nn.Module):
    """Cluster Learning Module using Induced Decoder Encoder with attention-based logits.

    This module uses InducedDecoderEncoder to process input representations, then
    computes attention between the output and inducing vectors to generate logits.
    The attention map serves as the classification logits.
    
    When num_classes is provided (shape: (B,)), the module dynamically masks invalid
    inducing vectors (indices >= num_classes[b] for each sample b):
    - Invalid ind_vectors won't participate in self-attention
    - src won't attend to invalid ind_vectors in cross-attention
    - Output logits for invalid classes are set to -inf

    Parameters
    ----------
    max_classes : int
        Number of classes (should equal num_inds for proper attention mapping)

    d_model : int
        Model dimension

    num_blocks : int
        Number of blocks used in the induced decoder encoder

    nhead : int
        Number of attention heads

    dim_feedforward : int
        Dimension of the feedforward network

    num_inds : int
        Number of inducing points (K), determines the number of output classes

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network

    norm_first : bool, default=True
        If True, uses pre-norm architecture

    use_rope_cross_attn : bool, default=False
        If True, use rotary positional encoding in cross-attention when ind_vectors_hidden queries src

    rope_base : float, default=100000
        Base scaling factor for rotary position encoding (only used if use_rope_cross_attn=True)

    use_representation_self_att : bool, default=False
        If True, Stage 2 uses decoder block (self-attention + cross-attention);
        If False, Stage 2 uses only cross-attention (more efficient)
    """

    def __init__(
        self,
        max_classes: int,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        use_rope_cross_attn: bool = False,
        rope_base: float = 100000,
        use_representation_self_att: bool = False,
    ):
        super().__init__()

        self.max_classes = max_classes
        self.num_inds = num_inds
        self.d_model = d_model
        self.norm_first = norm_first

        # Check that max_classes matches num_inds
        if max_classes != num_inds:
            raise ValueError(f"max_classes ({max_classes}) must equal num_inds ({num_inds})")

        # Initialize the encoder with InducedDecoderEncoder
        self.tf_dataset = InducedDecoderEncoder(
            num_blocks=num_blocks,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_inds=num_inds,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            use_rope_cross_attn=use_rope_cross_attn,
            rope_base=rope_base,
            use_representation_self_att=use_representation_self_att,
        )

        if self.norm_first:
            self.ln_src = nn.LayerNorm(d_model)
            self.ln_hidden = nn.LayerNorm(d_model)

        shared_hidden_dim = 2 * d_model
        self.shared_proj = nn.Sequential(
            nn.Linear(d_model, shared_hidden_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden_dim, d_model)
        )
        
        self.log_temperature = nn.Parameter(torch.tensor(2.66))  # exp(2.66) ≈ 14.3 ≈ 1/0.07

        # InferenceManager for single output
        self.inference_mgr = InferenceManager(
            enc_name="tf_dataset", 
            out_dim=max_classes
        )
        
        self.inference_mgr_multi = InferenceManager(
            enc_name="tf_dataset",
            out_dim=max_classes,
            out_dims=[max_classes, d_model, (num_inds, d_model)],  # [logits: (..., T, max_classes), query_norm: (..., T, d_model), key_norm: (..., num_inds, d_model)]
            out_no_seq=[False, False, True]  # logits and query_norm have seq_len, key_norm doesn't
        )

    def _create_ind_key_padding_mask(self, num_classes: Tensor) -> Tensor:
        """Create mask for inducing vectors based on num_classes.
        
        Parameters
        ----------
        num_classes : Tensor
            Number of valid classes for each sample, shape (B,)
            
        Returns
        -------
        Tensor
            Boolean mask of shape (B, max_classes) where True indicates invalid positions
        """
        B = num_classes.shape[0]
        # Create indices: (max_classes,)
        indices = torch.arange(self.max_classes, device=num_classes.device)
        # Expand for broadcasting: (B, max_classes)
        # mask[b, k] = True if k >= num_classes[b] (invalid position)
        mask = indices.unsqueeze(0) >= num_classes.unsqueeze(1)
        return mask

    def __cluster(
        self, 
        R: Tensor, 
        num_classes: Optional[Tensor] = None,
        return_hidden: bool = False, 
    ) -> Tensor | tuple[Tensor, ...]:
        """Cluster the representations R into class logits using attention mechanism.
        
        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D)
        num_classes : Optional[Tensor], default=None
            Number of valid classes for each sample, shape (B,). When provided:
            - Creates mask to exclude invalid inducing vectors from attention
            - Sets logits for invalid classes to -inf
        return_hidden : bool, default=False
            If True, return logits, query_norm, key_norm. Otherwise, return only logits.
        
        Returns
        -------
        Tensor or tuple[Tensor, ...]
            If return_hidden is False:
                logits: (B, T, max_classes)
            If return_hidden is True:
                tuple of (logits, query_norm, key_norm) where:
                - logits: (B, T, max_classes)
                - query_norm: (B, T, d_model)
                - key_norm: (B, K, d_model)
        """
        # Create mask for inducing vectors if num_classes is provided
        ind_key_padding_mask = None
        if num_classes is not None:
            ind_key_padding_mask = self._create_ind_key_padding_mask(num_classes)
        
        # Get output and inducing vectors from encoder
        encoder_output = self.tf_dataset(R, ind_key_padding_mask=ind_key_padding_mask)
        
        src, ind_vectors_hidden = encoder_output  # src: (B, T, d_model), ind_vectors_hidden: (B, K, d_model)

        # Apply layer normalization to both src and ind_vectors_hidden
        if self.norm_first:
            src = self.ln_src(src)
            ind_vectors_hidden = self.ln_hidden(ind_vectors_hidden)
        
        # Compute cross-attention: src attends to ind_vectors_hidden
        # Query from src, Key from ind_vectors_hidden (both use shared projection)
        query = self.shared_proj(src)  # (B, T, d_model)
        key = self.shared_proj(ind_vectors_hidden)  # (B, K, d_model)

        
        # Normalize query and key for consistent cosine similarity computation
        # This ensures numerical stability and semantic correctness
        query_norm = F.normalize(query, p=2, dim=-1, eps=1e-8)  # (B, T, d_model) - L2 normalized
        key_norm = F.normalize(key, p=2, dim=-1, eps=1e-8)  # (B, K, d_model) - L2 normalized
        
        # Compute attention scores using cosine similarity: (B, T, d_model) @ (B, d_model, K) -> (B, T, K)
        # Cosine similarity ranges in [-1, 1], learnable log_temperature amplifies for sharper softmax
        logit_scale = self.log_temperature.clamp(min=-2.3, max=4.605).exp()
        logits = torch.matmul(query_norm, key_norm.transpose(-2, -1)) * logit_scale  # (B, T, K)
        
        # Apply mask to logits: set invalid classes to -inf
        if ind_key_padding_mask is not None:
            # ind_key_padding_mask: (B, K) -> (B, 1, K) for broadcasting
            logits = logits.masked_fill(ind_key_padding_mask.unsqueeze(1), float('-inf'))

        if return_hidden:
            return logits, query_norm, key_norm
        return logits
    
    def _predict_standard(
        self,
        R: Tensor,
        num_classes: Optional[Tensor] = None,
        auto_batch: bool = True,
        return_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        """Generate predictions for standard classification.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        num_classes : Optional[Tensor], default=None
            Number of valid classes for each dataset, shape (B,). Will be split along with R
            when auto_batch is used.

        auto_batch : bool, default=True
            Whether to use InferenceManager to automatically split inputs into smaller batches.
            
        return_hidden : bool, default=False
            If True, return logits, query_norm, key_norm. Otherwise, return only logits.

        Returns
        -------
        Tensor or tuple[Tensor, ...]
            If return_hidden is False:
                Logits of shape (B, T, max_classes)
            If return_hidden is True:
                tuple of (logits, query_norm, key_norm)
        """

        # Create a wrapper function that passes return_hidden to __cluster
        def cluster_wrapper(R: Tensor, num_classes: Optional[Tensor] = None):
            return self.__cluster(R, num_classes=num_classes, return_hidden=return_hidden)
        
        # Build inputs dict - num_classes can be split along with R since they share the batch dimension
        inputs = OrderedDict([("R", R)])
        if num_classes is not None:
            inputs["num_classes"] = num_classes
        
        # Use appropriate InferenceManager based on return_hidden
        if return_hidden:
            outputs = self.inference_mgr_multi(
                cluster_wrapper, inputs=inputs, auto_batch=auto_batch
            )
            return outputs
        else:
            logits = self.inference_mgr(
                cluster_wrapper, inputs=inputs, auto_batch=auto_batch
            )
            return logits

    def _inference_forward(
        self,
        R: Tensor,
        num_classes: Optional[Tensor] = None,
        mgr_config: MgrConfig = None,
        return_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        """In-context learning based on learned row representations for inference.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        num_classes : Optional[Tensor], default=None
            Number of valid classes for each sample, shape (B,)

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager
            
        return_hidden : bool, default=False
            If True, return logits, query_norm, key_norm. Otherwise, return only logits.

        Returns
        -------
        Tensor or tuple[Tensor, ...]
            If return_hidden is False:
                Logits of shape (B, T, max_classes)
            If return_hidden is True:
                tuple of (logits, query_norm, key_norm)
        """
        # Configure inference parameters
        if mgr_config is None:
            mgr_config = MgrConfig(
                min_batch_size=1,
                safety_factor=0.8,
                offload=False,
                auto_offload_pct=0.5,
                device=None,
                use_amp=True,
                verbose=False,
            )
        self.inference_mgr.configure(**mgr_config)
        self.inference_mgr_multi.configure(**mgr_config)

        out = self._predict_standard(R, num_classes=num_classes, return_hidden=return_hidden)
        
        return out


    def forward(
        self,
        R: Tensor,
        num_classes: Optional[Tensor] = None,
        mgr_config: MgrConfig = None,
        return_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        """Cluster learning based on learned row representations.

        Parameters
        ----------
        R : Tensor
            Row representations of shape (B, T, D) where:
             - B is the number of tables
             - T is the number of samples (rows)
             - D is the dimension of row representations

        num_classes : Optional[Tensor], default=None
            Number of valid classes for each sample, shape (B,). When provided:
            - Creates mask to exclude invalid inducing vectors from attention
            - Sets logits for invalid classes to -inf

        mgr_config : MgrConfig, default=None
            Configuration for InferenceManager. Used only in inference mode.
            
        return_hidden : bool, default=False
            If True, return logits, query_norm, key_norm. Otherwise, return only logits.
        

        Returns
        -------
        Tensor or tuple[Tensor, ...]
            For training mode:
              If return_hidden is False:
                Raw logits of shape (B, T, max_classes)
              If return_hidden is True:
                tuple of (logits, query_norm, key_norm) where:
                - logits: (B, T, max_classes) - invalid classes have -inf
                - query_norm: (B, T, d_model)
                - key_norm: (B, K, d_model)

            For inference mode:
              Same as training mode.
        """

        if self.training:
            out = self.__cluster(R, num_classes=num_classes, return_hidden=return_hidden)
        else:
            out = self._inference_forward(R, num_classes=num_classes, mgr_config=mgr_config, return_hidden=return_hidden)

        return out
