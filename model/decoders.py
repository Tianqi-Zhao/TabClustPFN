from __future__ import annotations

from typing import Optional
from torch import nn, Tensor

from .layers import DecoderAttentionBlock
from .rope import RotaryEmbedding


class Decoder(nn.Module):
    """Stack of decoder blocks with cross-attention.

    Parameters
    ----------
    num_blocks : int
        Number of decoder blocks in the stack

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
        Whether to use rotary positional encoding for self-attention

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
                DecoderAttentionBlock(
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
        tgt: Tensor,
        memory: Tensor,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_attn_mask: Optional[Tensor | int] = None,
        memory_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Process input through the stacked decoder blocks.

        Parameters
        ----------
        tgt : Tensor
            Target tensor of shape (..., tgt_len, d_model)

        memory : Tensor
            Memory tensor from encoder of shape (..., src_len, d_model)

        tgt_key_padding_mask : Optional[Tensor], default=None
            Mask of shape (..., tgt_len) that identifies padding elements
            in the target sequence to be ignored:
              - For binary masks: True values indicate positions to ignore
              - For float masks: Values are directly added to attention scores

        memory_key_padding_mask : Optional[Tensor], default=None
            Mask of shape (..., src_len) that identifies padding elements
            in the memory sequence to be ignored:
              - For binary masks: True values indicate positions to ignore
              - For float masks: Values are directly added to attention scores

        tgt_attn_mask : Optional[Tensor | int], default=None
            Controls attention pattern for self-attention in two possible ways:
            1. When provided as Tensor: Traditional mask preventing attention to certain positions
              - Shape: (tgt_len, tgt_len) or (..., num_heads, tgt_len, tgt_len)
            2. When provided as integer: Creates a split attention pattern where:
              - The first `tgt_attn_mask` tokens perform self-attention only (attend to themselves)
              - The remaining tokens attend only to the first `tgt_attn_mask` tokens

        memory_attn_mask : Optional[Tensor], default=None
            Attention mask for cross-attention of shape (tgt_len, src_len) or
            (..., num_heads, tgt_len, src_len)

        Returns
        -------
        Tensor
            Output tensor of shape (..., tgt_len, d_model)
        """
        out = tgt
        for block in self.blocks:
            out = block(
                tgt=out,
                memory=memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_attn_mask=tgt_attn_mask,
                memory_attn_mask=memory_attn_mask,
                rope=self.rope,
            )

        return out

