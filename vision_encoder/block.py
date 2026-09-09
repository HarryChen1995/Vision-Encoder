"""The Transformer block: attention + MLP, each wrapped in a residual.

    x = x + LayerScale(DropPath(Attention(LayerNorm(x))))
    x = x + LayerScale(DropPath(Mlp(LayerNorm(x))))

Two sub-layers, each following the same template: normalize, transform, scale,
maybe drop the branch, add back to the input. Stack L of these and you have the
entire encoder — a ViT has no pooling pyramid, no downsampling, no convolution
after the patch stem. Every block sees the full sequence at full resolution.

PRE-NORM vs POST-NORM (this file uses pre-norm)
----------------------------------------------
The original 2017 Transformer normalized *after* the residual add:
`x = LN(x + Attn(x))`. That places a LayerNorm directly on the residual path,
so gradients flowing back from the loss are rescaled at every one of L layers.
Deep post-norm stacks therefore need a careful learning-rate warmup to avoid
diverging in the first few hundred steps.

Pre-norm — `x = x + Attn(LN(x))` — normalizes only the *branch input*. The
residual path from input to output is then a clean sum of terms with nothing
multiplicative in the way, so gradients reach early layers undistorted. This is
what makes 24+ layer Transformers train reliably, and it is why essentially all
modern ViTs (and LLMs) are pre-norm. The trade-off: activations grow as blocks
keep adding to the stream, so a final LayerNorm after the last block is
required to renormalize before the head.

WHY RESIDUALS AT ALL
--------------------
`x + f(x)` gives every block an identity shortcut. Its Jacobian is `I + f'(x)`,
so even when `f'` is small the gradient still flows through the `I` term
unattenuated. Blocks learn a *correction* to the running representation rather
than a replacement — the "residual stream" view: each block reads from a shared
channel bus, computes something, and writes its result back by addition.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .layers import DropPath, LayerScale, Mlp


class Block(nn.Module):
    """One pre-norm Transformer block.

    Args:
        dim: Model width `D`.
        num_heads: Attention heads.
        mlp_ratio: Hidden width of the MLP as a multiple of `dim` (4.0 typical).
        qkv_bias: Bias on the fused QKV projection.
        qk_norm: LayerNorm on queries/keys (stability at scale).
        drop: Dropout on the MLP and on attention's output projection.
        attn_drop: Dropout on attention probabilities.
        init_values: If set, wrap both branches in LayerScale with this init.
        drop_path: Stochastic-depth probability for this block's branches.
        act_layer: MLP activation.
        norm_layer: Normalization class (LayerNorm by default).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        # ---- Sub-layer 1: token mixing -------------------------------------
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # ---- Sub-layer 2: per-token computation ----------------------------
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`(B, N, D) -> (B, N, D)`. `attn_mask` is additive, e.g. a causal mask."""
        # Note the argument to each branch is norm(x), but what we add to is the
        # un-normalized x. That is the defining property of pre-norm.
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
