from __future__ import annotations

from typing import Optional
import torch
from torch import nn, Tensor

from .layers import MultiheadAttentionBlock, InducedSelfAttentionBlock, InducedDecoderAttentionBlock
from .rope import RotaryEmbedding


class Encoder(nn.Module):
    """Stack of multihead attention blocks.

    Parameters
    ----------
    num_blocks : int
        Number of multihead attention blocks in the stack

    d_model : int
        Model dimension

    nhead : int
        Number of attention heads and should be a divisor of d_model

    dim_feedforward : int
        Dimension of the feedforward network in each block

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)

    use_rope : bool, default=False
        Whether to use rotary positional encoding

    rope_base : int, default=100000
        A base scaling factor for rotary position encoding
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        use_rope: bool = False,
        rope_base: int = 100000,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.blocks = nn.ModuleList(
            [
                MultiheadAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                )
                for _ in range(num_blocks)
            ]
        )

        self.rope = RotaryEmbedding(dim=d_model // nhead, theta=rope_base) if use_rope else None

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor | int] = None,
    ) -> Tensor:
        """Process input through the stacked blocks.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model)

        key_padding_mask : Optional[Tensor], default=None
            Mask of shape (..., src_len) that identifies padding elements
            in the key sequence to be ignored:
              - For binary masks: True values indicate positions to ignore
              - For float masks: Values are directly added to attention scores

        attn_mask : Optional[Tensor | int], default=None
            Controls attention pattern in two possible ways:
            1. When provided as Tensor: Traditional mask preventing attention to certain positions
              - Shape: (tgt_len, src_len) or (..., num_heads, tgt_len, src_len)
            2. When provided as integer: Creates a split attention pattern where:
              - The first `attn_mask` tokens perform self-attention only (attend to themselves)
              - The remaining tokens attend only to the first `attn_mask` tokens

        Returns
        -------
        Tensor
            Output tensor of shape (..., seq_len, d_model)
        """
        out = src
        for block in self.blocks:
            out = block(q=out, key_padding_mask=key_padding_mask, attn_mask=attn_mask, rope=self.rope)

        return out


class SetTransformer(nn.Module):
    """Stack of induced self-attention blocks.

    A set transformer uses induced self-attention mechanism to efficiently
    process variable-sized sets while maintaining permutation invariance.

    Parameters
    ----------
    num_blocks : int
        Number of induced self-attention blocks in the stack

    d_model : int
        Model dimension

    nhead : int
        Number of attention heads and should be a divisor of d_model

    dim_feedforward : int
        Dimension of the feedforward network in each block

    num_inds : int, default=16
        Number of inducing points used in self-attention blocks

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)

    References
    ----------
    .. [1] Lee et al. "Set Transformer: A Framework for Attention-based
           Permutation-Invariant Neural Networks", ICML 2019
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int = 16,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.blocks = nn.ModuleList(
            [
                InducedSelfAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    num_inds=num_inds,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, src: Tensor) -> Tensor:
        """Process input through the stacked blocks.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model)

        train_size : Optional[int], default=None
            Position to split the input into training and test data. When provided,
            inducing points will only attend to training data in the first attention
            stage of induced self-attention blocks to prevent information leakage.

        Returns
        -------
        Tensor
            Output tensor of shape (..., seq_len, d_model)
        """
        out = src
        for block in self.blocks:
            out = block(out)

        return out


class InducedDecoderEncoder(nn.Module):
    """Stack of induced decoder attention blocks with learnable inducing vectors.

    This encoder uses a decoder-based induced attention mechanism where learnable
    inducing points first perform self-attention, then attend to the source, and
    finally the source attends back to the processed inducing points.

    The inducing vectors are shared across all blocks and updated through each layer,
    allowing them to accumulate and refine information progressively.

    Parameters
    ----------
    num_blocks : int
        Number of induced decoder attention blocks in the stack

    d_model : int
        Model dimension

    nhead : int
        Number of attention heads and should be a divisor of d_model

    dim_feedforward : int
        Dimension of the feedforward network in each block

    num_inds : int, default=16
        Number of inducing points (K in the documentation)

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)

    use_rope_cross_attn : bool, default=False
        If True, use rotary positional encoding in cross-attention when ind_vectors_hidden queries src

    rope_base : float, default=100000
        Base scaling factor for rotary position encoding (only used if use_rope_cross_attn=True)

    use_representation_self_att : bool, default=False
        If True, Stage 2 uses decoder block (self-attention + cross-attention);
        If False, Stage 2 uses only cross-attention (more efficient)

    Notes
    -----
    The inducing vectors act as a bottleneck that captures global information
    from the input sequence. Each block refines these vectors through:
    1. Self-attention among inducing vectors
    2. Cross-attention from inducing vectors to source
    3. Cross-attention from source to refined inducing vectors
    """

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int = 16,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
        use_rope_cross_attn: bool = False,
        rope_base: float = 100000,
        use_representation_self_att: bool = False,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.num_inds = num_inds
        self.d_model = d_model

        # Learnable inducing vectors (initial state)
        self.ind_vectors = nn.Parameter(torch.empty(num_inds, d_model))
        nn.init.trunc_normal_(self.ind_vectors, std=0.02)

        # Stack of induced decoder attention blocks
        self.blocks = nn.ModuleList(
            [
                InducedDecoderAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                    use_rope_cross_attn=use_rope_cross_attn,
                    rope_base=rope_base,
                    use_representation_self_att=use_representation_self_att,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self, 
        src: Tensor, 
        ind_key_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict, dict]:
        """Process input through the stacked blocks.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (B, seq_len, d_model)
        
        ind_key_padding_mask : Optional[Tensor], default=None
            Mask of shape (B, num_inds) that identifies invalid inducing vectors to ignore.
            True values indicate positions to ignore. When provided:
            - Invalid ind_vectors won't participate in self-attention
            - src won't attend to invalid ind_vectors in cross-attention

        Returns
        -------
        out : Tensor
            Output tensor of shape (B, seq_len, d_model)
        ind_vectors_hidden : Tensor
            Final inducing vectors of shape (B, num_inds, d_model)
        """
        B = src.shape[0]
        
        # Expand inducing vectors for the batch
        ind_vectors_hidden = self.ind_vectors.unsqueeze(0).expand(B, self.num_inds, self.d_model)
        
        out = src
        for block in self.blocks:
            # Each block updates both the source and the inducing vectors
            out, ind_vectors_hidden = block(out, ind_vectors_hidden, ind_key_padding_mask=ind_key_padding_mask)
        
        return out, ind_vectors_hidden
