from __future__ import annotations
from typing import List, Optional

from torch import nn, Tensor
import torch.nn.functional as F
import torch
from .rope import RotaryEmbedding
from .attention import multi_head_attention_forward


class SkippableLinear(nn.Linear):
    """Linear layer that handles inputs where all values equal `skip_value`.

    First applies the linear transformation to all inputs, then replaces outputs for inputs
    where all values equal `skip_value` with the `skip_value`.

    Parameters
    ----------
    in_features : int
        Size of each input sample

    out_features : int
        Size of each output sample

    bias : bool, default=True
        If set to False, the layer will not learn an additive bias

    skip_value : float, default=-100.0
        Value used to mark inputs that should be skipped
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, skip_value: float = -100.0):
        super().__init__(in_features, out_features, bias)
        self.skip_value = skip_value

    def forward(self, src: Tensor) -> Tensor:
        """Forward pass that handles inputs flagged with `skip_value`.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., in_features)

        Returns
        -------
        Tensor
            Output tensor of shape (..., out_features) where rows corresponding
            to skipped inputs are filled with `skip_value`
        """

        out = F.linear(src, self.weight, self.bias)
        skip_mask = (src == self.skip_value).all(dim=-1)
        if skip_mask.any():
            out[skip_mask] = self.skip_value

        return out


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture.

    Parameters
    ----------
    in_dim : int
        Input feature dimension

    out_dim : Optional[int], default=None
        Output dimension. If None, uses the last hidden dimension

    hidden_dims : List[int], default=[256, 256, 256]
        Dimensions of hidden layers

    activation : str, default='gelu'
        Activation function: 'relu', 'gelu', 'leaky_relu', or 'tanh'

    bias : bool, default=True
        Whether to include bias terms in linear layers
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        hidden_dims: List[int] = [256, 256, 256],
        activation: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()
        # Build network architecture
        act_fn = self.get_activation(activation)
        layers = []

        # Create hidden layers with activations
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=bias))
            layers.append(act_fn())
            prev_dim = hidden_dim

        # Optional output projection
        if out_dim is not None:
            layers.append(nn.Linear(prev_dim, out_dim, bias=bias))

        self.net = nn.Sequential(*layers)

    @staticmethod
    def get_activation(activation: str) -> nn.Module:
        """Get activation function class from string name.

        Parameters
        ----------
        activation : str
            Name of activation function

        Returns
        -------
        class
            PyTorch activation function class
        """

        activation_map = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }

        if activation not in activation_map:
            raise ValueError(f"Unknown activation: {activation}. Supported: {list(activation_map.keys())}")

        return activation_map[activation]

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass through the MLP.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (..., in_dim)

        Returns
        -------
        Tensor
            Output tensor of shape (..., out_dim or last_hidden_dim)
        """
        return self.net(X)


class MultiheadAttention(nn.MultiheadAttention):
    """Enhanced multi-head attention with rotary positional embedding support.

    This extends PyTorch's MultiheadAttention to support rotary position embeddings (RoPE)
    and specialized attention masking when `attn_mask` is an integer. The implementation always
    uses `batch_first=True`, meaning all input tensors have shape (..., seq_len, embed_dim).

    Parameters
    ----------
    embed_dim : int
        Model dimension (total size of each attention head combined)

    num_heads : int
        Number of attention heads

    dropout : float, default=0.0
        Dropout probability applied to attention weights

    References
    ----------
    .. [1] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
           https://arxiv.org/abs/2104.09864
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__(embed_dim, num_heads, dropout, batch_first=True)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor | int] = None,
        rope: Optional[RotaryEmbedding] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Compute multi-head attention with support for rotary positional encoding.

        Parameters
        ----------
        query : Tensor
            Query tensor of shape (..., tgt_len, embed_dim)

        key : Tensor
            Key tensor of shape (..., src_len, embed_dim)

        value : Tensor
            Value tensor of shape (..., src_len, embed_dim)

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

        rope : Optional[RotaryEmbedding]
            Rotary positional encoding

        Returns
        -------
        Tensor or Tuple[Tensor, Tensor]
            Attention output of shape (..., tgt_len, embed_dim)
        """

        if isinstance(attn_mask, int):
            assert key_padding_mask is None, "key_padding_mask is not supported with attn_mask as int"
            assert rope is None, "Rotary position embedding is not supported with attn_mask as int"
        else:
            # Convert masks to correct dtype for compatibility (same as MultiheadAttentionBlock)
            key_padding_mask = F._canonical_mask(
                mask=key_padding_mask,
                mask_name="key_padding_mask",
                other_type=F._none_or_dtype(attn_mask),
                other_name="attn_mask",
                target_type=query.dtype,
            )
            attn_mask = F._canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=query.dtype,
                check_other=False,
            )

        return multi_head_attention_forward(
            query,
            key,
            value,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            rope=rope,
        )


class MultiheadAttentionBlock(nn.TransformerEncoderLayer):
    """Attention block supporting rotary positional encoding.

    Parameters
    ----------
    d_model : int
        Model dimension

    nhead : int
        Number of attention heads

    dim_feedforward : int
       Dimension of the feedforward network

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
    ):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, norm_first=norm_first, batch_first=True)
        del self.self_attn
        self.attn = MultiheadAttention(d_model, nhead, dropout)
        self.init_weights()

    def init_weights(self):
        """Initialize projection layers to zero for stable training."""
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor | int] = None,
        rope: Optional[RotaryEmbedding] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Process input through attention with optional rotary positional encoding.

        Parameters
        ----------
        q : Tensor
            Query tensor of shape (..., tgt_len, d_model)

        k : Tensor
            Key tensor of shape (..., src_len, d_model)
            If None, uses q for self-attention.

        v : Tensor
            Value tensor of shape (..., src_len, d_model)
            If None, uses q for self-attention.

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

        rope : Optional[RotaryEmbedding]
            Rotary positional encoding


        Returns
        -------
        Tensor or Tuple[Tensor, Tensor]
            Output tensor of shape (..., tgt_len, d_model)
        """

        if isinstance(attn_mask, int):
            assert key_padding_mask is None, "key_padding_mask is not supported with attn_mask as int"
            assert rope is None, "Rotary position embedding is not supported with attn_mask as int"
        # Note: mask conversion (F._canonical_mask) is handled in MultiheadAttention.forward()

        # Use q as k,v if not provided
        k = q if k is None else k
        v = q if v is None else v

        # Apply layer depending on normalization order
        x = q
        if self.norm_first:
            # Pre-norm: normalize before attention and FFN
            attn = self._attn_block(self.norm1(q), self.norm1(k), self.norm1(v), key_padding_mask, attn_mask, rope)
            x = x + attn
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-norm: normalize after attention and FFN
            attn = self._attn_block(q, k, v, key_padding_mask, attn_mask, rope)
            x = self.norm1(x + attn)
            x = self.norm2(x + self._ff_block(x))

        return x

    def _attn_block(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor | int],
        rope: Optional[RotaryEmbedding],
    ) -> Tensor | tuple[Tensor, Tensor]:

        attn = self.attn(q, k, v, key_padding_mask, attn_mask, rope)
        return self.dropout1(attn)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class InducedSelfAttentionBlock(nn.Module):
    """Induced Self-Attention for efficient O(n) attention on large sets.

    This module implements a bottleneck attention mechanism using a small set of
    learned inducing points that mediate interactions between input elements.
    The complexity is reduced from O(n²) to O(n) by:

    1. Projecting inputs onto inducing points (size m << n)
    2. Propagating information through these inducing points
    3. Projecting back to the original sequence

    Parameters
    ----------
    d_model : int
        Model dimension

    nhead : int
        Number of attention heads

    dim_feedforward : int
        Dimension of the feedforward network

    num_inds : int
        Number of inducing points (controls capacity vs. efficiency)

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)

    skip_value : float, default=-100.0
        Value used to mark inputs that should be skipped

    References
    ----------
    .. [1] Lee et al. "Set Transformer: A Framework for Attention-based
           Permutation-Invariant Neural Networks", ICML 2019
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        skip_value: float = -100.0,
    ):
        super().__init__()
        self.skip_value = skip_value

        # Two-stage attention mechanism
        self.multihead_attn1 = MultiheadAttentionBlock(d_model, nhead, dim_feedforward, dropout, activation, norm_first)
        self.multihead_attn2 = MultiheadAttentionBlock(d_model, nhead, dim_feedforward, dropout, activation, norm_first)

        # Learnable inducing points
        self.num_inds = num_inds
        self.ind_vectors = nn.Parameter(torch.empty(num_inds, d_model))
        nn.init.trunc_normal_(self.ind_vectors, std=0.02)

    def induced_attention(self, src: Tensor) -> Tensor:
        """Apply induced self-attention to input sequence.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model)

        Returns
        -------
        Tensor
            Output tensor with same shape as input
        """

        *batch_shape, _, d_model = src.shape
        ind_vectors = self.ind_vectors.expand(*batch_shape, self.num_inds, d_model)

        hidden = self.multihead_attn1(ind_vectors, src, src)

        out = self.multihead_attn2(src, hidden, hidden)

        return out

    def forward(self, src: Tensor) -> Tensor:
        """Apply induced self-attention to input sequence.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model)

        Returns
        -------
        Tensor
            Output tensor with same shape as input
        """

        skip_mask = (src == self.skip_value).all(dim=(-2, -1))  # batch shape
        if skip_mask.any():
            if skip_mask.all():
                out = torch.full_like(src, self.skip_value)
            else:
                out = torch.empty_like(src)
                out[~skip_mask] = self.induced_attention(src[~skip_mask])
                out[skip_mask] = self.skip_value
        else:
            out = self.induced_attention(src)

        return out



class DecoderAttentionBlock(nn.Module):
    """Decoder block with self-attention, cross-attention, and feedforward.

    This block implements a standard transformer decoder layer with:
    - Self-attention on the target sequence
    - Cross-attention from target to memory (encoder output)
    - Feedforward network

    Parameters
    ----------
    d_model : int
        Model dimension

    nhead : int
        Number of attention heads and should be a divisor of d_model

    dim_feedforward : int
        Dimension of the feedforward network

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        # Self-attention
        self.self_attn = MultiheadAttention(d_model, nhead, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)  # For query (decoder output)
        # self.norm_memory = nn.LayerNorm(d_model)  # For key/value (memory from encoder)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # Activation function
        if isinstance(activation, str):
            if activation == "relu":
                self.activation = nn.ReLU()
            elif activation == "gelu":
                self.activation = nn.GELU()
            else:
                raise ValueError(f"Unknown activation: {activation}")
        else:
            self.activation = activation()

        self.norm_first = norm_first
        self.init_weights()

    def init_weights(self):
        """Initialize projection layers to zero for stable training."""
        nn.init.zeros_(self.self_attn.out_proj.weight)
        nn.init.zeros_(self.self_attn.out_proj.bias)
        nn.init.zeros_(self.cross_attn.out_proj.weight)
        nn.init.zeros_(self.cross_attn.out_proj.bias)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def _ff_block(self, x: Tensor) -> Tensor:
        """Feedforward block."""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_attn_mask: Optional[Tensor | int] = None,
        memory_attn_mask: Optional[Tensor] = None,
        rope: Optional[RotaryEmbedding] = None,
        cross_rope: Optional[RotaryEmbedding] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Process input through self-attention, cross-attention, and feedforward.

        Parameters
        ----------
        tgt : Tensor
            Target tensor of shape (..., tgt_len, d_model)

        memory : Tensor
            Memory tensor from encoder of shape (..., src_len, d_model)

        tgt_key_padding_mask : Optional[Tensor], default=None
            Mask of shape (..., tgt_len) that identifies padding elements
            in the target sequence to be ignored

        memory_key_padding_mask : Optional[Tensor], default=None
            Mask of shape (..., src_len) that identifies padding elements
            in the memory sequence to be ignored

        tgt_attn_mask : Optional[Tensor | int], default=None
            Attention mask for self-attention. See MultiheadAttention for details.

        memory_attn_mask : Optional[Tensor], default=None
            Attention mask for cross-attention of shape (tgt_len, src_len) or
            (..., num_heads, tgt_len, src_len)

        rope : Optional[RotaryEmbedding], default=None
            Rotary positional encoding (applied to self-attention only)

        cross_rope : Optional[RotaryEmbedding], default=None
            Rotary positional encoding (applied to cross-attention)

        Returns
        -------
        Tensor or Tuple[Tensor, Tensor]
            Output tensor of shape (..., tgt_len, d_model)
        """
        x = tgt

        if self.norm_first:
            # Pre-norm: normalize before each operation
            # Self-attention
            norm_x = self.norm1(x)
            attn = self.self_attn(norm_x, norm_x, norm_x, tgt_key_padding_mask, tgt_attn_mask, rope)
            x = x + self.dropout1(attn)

            # Cross-attention (normalize both query and key/value)
            norm_x = self.norm2(x)
            # norm_memory = self.norm_memory(memory)
            attn = self.cross_attn(norm_x, memory, memory, memory_key_padding_mask, memory_attn_mask, cross_rope)
            x = x + self.dropout2(attn)

            # Feedforward
            x = x + self._ff_block(self.norm3(x))
        else:
            # Post-norm: normalize after each operation
            # Self-attention
            attn = self.self_attn(x, x, x, tgt_key_padding_mask, tgt_attn_mask, rope)
            x = self.norm1(x + self.dropout1(attn))

            attn = self.cross_attn(x, memory, memory, memory_key_padding_mask, memory_attn_mask, cross_rope)
            x = self.norm2(x + self.dropout2(attn))

            # Feedforward
            x = self.norm3(x + self._ff_block(x))

        return x


class InducedDecoderAttentionBlock(nn.Module):
    """Induced attention with decoder structure for both stages.

    This module uses decoder structures for both stages:
    - Stage 1: Inducing points perform self-attention, then cross-attention to source
    - Stage 2: Source performs self-attention, then cross-attention to processed inducing points

    The architecture consists of:
    1. Stage 1 (Decoder Block):
       - Self-attention on inducing points
       - Cross-attention: inducing points attend to source
    2. Stage 2 (Decoder Block):
       - Self-attention on source
       - Cross-attention: source attends to processed inducing points

    This allows both inducing points and source to communicate internally before
    cross-attention, potentially capturing more complex patterns.

    Parameters
    ----------
    d_model : int
        Model dimension

    nhead : int
        Number of attention heads

    dim_feedforward : int
        Dimension of the feedforward network

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

    use_representation_self_att : bool, default=True
        If True, Stage 2 uses decoder block (self-attention + cross-attention);
        If False, Stage 2 uses only cross-attention (more efficient)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        use_rope_cross_attn: bool = False,
        rope_base: float = 100000,
        use_representation_self_att: bool = False,
    ):
        super().__init__()

        # First stage: decoder block (self-attention + cross-attention)
        self.decoder_block = DecoderAttentionBlock(d_model, nhead, dim_feedforward, dropout, activation, norm_first)
        
        # Second stage: controlled by use_representation_self_att parameter
        self.use_representation_self_att = use_representation_self_att
        if use_representation_self_att:
            # Use decoder block (self-attention on src + cross-attention to hidden)
            self.decoder_block2 = DecoderAttentionBlock(d_model, nhead, dim_feedforward, dropout, activation, norm_first)
        else:
            # Use only cross-attention (more efficient, no self-attention on src)
            self.cross_attn_block = MultiheadAttentionBlock(d_model, nhead, dim_feedforward, dropout, activation, norm_first)

        # Rotary positional encoding for self-attention (ind_vectors_hidden self-attention)
        # Note: cross-attention (ind_vectors_hidden queries src) does not use RoPE
        self.use_rope_cross_attn = use_rope_cross_attn
        if use_rope_cross_attn:
            self.rope_self_attn = RotaryEmbedding(dim=d_model // nhead, theta=rope_base)
        else:
            self.rope_self_attn = None

    def forward(
        self, 
        src: Tensor, 
        ind_vectors_hidden: Tensor, 
        ind_key_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        """Apply induced attention with decoder structure to input sequence.

        This method applies two stages:
        1. Stage 1: ind_vectors_hidden performs self-attention, then cross-attention to src
        2. Stage 2: Depends on use_representation_self_att parameter:
           - If True: src performs self-attention, then cross-attention to processed hidden
           - If False: src only performs cross-attention to processed hidden (no self-attention)

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (B, seq_len, d_model)

        ind_vectors_hidden : Tensor
            Inducing vectors of shape (B, K, d_model) where K is the number of inducing points

        ind_key_padding_mask : Optional[Tensor], default=None
            Mask of shape (B, K) that identifies invalid inducing vectors to ignore.
            True values indicate positions to ignore. When provided:
            - In Stage 1 self-attention: invalid ind_vectors won't attend to each other
            - In Stage 2 cross-attention: src won't attend to invalid ind_vectors

        Returns
        -------
        tuple[Tensor, Tensor] or tuple[Tensor, Tensor, Tensor, Tensor]
            - out: Output tensor with same shape as src (B, seq_len, d_model)
            - hidden: Processed inducing points of shape (B, K, d_model)
        """

        # Stage 1: Decoder structure - self-attention on ind_vectors, then cross-attention to src
        # Pass rope for self-attention only (cross_rope=None means no RoPE in cross-attention)
        # Use ind_key_padding_mask in self-attention to prevent invalid ind_vectors from attending
        hidden = self.decoder_block(
            tgt=ind_vectors_hidden, 
            memory=src,
            tgt_key_padding_mask=ind_key_padding_mask,  # Mask for ind_vectors self-attention
            rope=self.rope_self_attn,  # Used for self-attention
            cross_rope=None,  # Explicitly disable RoPE in cross-attention
        )

        # Stage 2: Controlled by use_representation_self_att parameter
        # Use ind_key_padding_mask in cross-attention to prevent src from attending to invalid ind_vectors
        if self.use_representation_self_att:
            out = self.decoder_block2(
                tgt=src,
                memory=hidden,
                memory_key_padding_mask=ind_key_padding_mask,  # Mask for cross-attention to hidden
                rope=None,  # No RoPE for src self-attention (can be added if needed)
                cross_rope=None,  # No RoPE for cross-attention
            )
        else:
            # Use only cross-attention (src queries hidden)
            out = self.cross_attn_block(
                src, hidden, hidden, 
                key_padding_mask=ind_key_padding_mask,  # Mask for cross-attention to hidden
            )

        return out, hidden
