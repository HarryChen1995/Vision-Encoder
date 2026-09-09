"""Multi-head self-attention — the layer that actually mixes information.

THE ONE-LINE VERSION
--------------------
Every token proposes a *query* ("what am I looking for?"), advertises a *key*
("what do I contain?") and carries a *value* ("what I'll hand over"). Token i
compares its query against all keys by dot product, softmaxes those scores into
weights that sum to 1, and takes the weighted average of the values. That is
the only place in the whole network where tokens see each other; the MLP that
follows is strictly per-token.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_head)) V

WHY DIVIDE BY sqrt(d_head)
--------------------------
If query/key components are roughly independent with unit variance, their dot
product over d_head dimensions has variance ~d_head, so the raw scores grow
with head size. Large scores push softmax into a near one-hot regime where the
gradient vanishes. Scaling by 1/sqrt(d_head) keeps score variance ~1 and the
softmax in a usable range, whatever the head dimension.

WHY MULTIPLE HEADS
------------------
One softmax produces one averaging pattern per token — a single "relationship"
per layer. Splitting the width D into `h` heads of size D/h runs `h` independent
attention patterns in parallel (one head can track texture continuity, another
object extent) and concatenates them. It costs the same FLOPs as one big head
because each operates on D/h channels, so multiple heads are essentially free
expressiveness. `out_proj` then mixes the concatenated heads back together —
without it the heads would never talk.

SHAPES (B=batch, N=tokens, D=width, h=heads, d=D/h)
---------------------------------------------------
    x        (B, N, D)
    qkv      (B, N, 3D)  -> reshape/permute -> 3 x (B, h, N, d)
    scores   (B, h, N, N)      <-- the O(N^2) term; the memory bottleneck
    out      (B, h, N, d) -> merge heads -> (B, N, D)

COST
----
FLOPs are ~4*N*D^2 (the four projections) + ~2*N^2*D (the score matmuls). Short
sequences are projection-bound; long ones are dominated by the quadratic term.

IMPLEMENTATION NOTE
-------------------
We call `F.scaled_dot_product_attention`, which dispatches to FlashAttention or
a memory-efficient kernel when available. Those kernels never materialize the
(B, h, N, N) score matrix in HBM — they tile the computation and recompute in
SRAM — turning attention's memory from quadratic to linear. Mathematically
identical, just far faster and lighter. A readable manual fallback is included
below for reference and for exporting attention maps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Standard MHSA with optional QK-normalization.

    Args:
        dim: Model width `D`. Must be divisible by `num_heads`.
        num_heads: Number of parallel attention heads.
        qkv_bias: Whether the fused QKV projection has a bias term.
        qk_norm: Apply LayerNorm to queries and keys before the dot product.
            This bounds the score magnitude regardless of activation growth and
            is a cheap, effective fix for the attention-logit blowup that
            destabilizes large/long training runs. Off by default to match the
            original ViT.
        attn_drop: Dropout on the attention weights.
        proj_drop: Dropout on the output projection.
        norm_layer: Normalization class used when `qk_norm=True`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5  # the 1/sqrt(d_head) above

        # One fused projection for Q, K and V. Three separate Linears would be
        # mathematically identical but launch three kernels over the same input;
        # fusing them is a straight speedup.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Prefer the fused kernel; fall back on very old torch builds.
        self.fused_attn = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Args: `x` is `(B, N, D)`. Returns `(B, N, D)`.

        `attn_mask` is an additive float mask broadcastable to `(B, h, N, N)`
        (use `-inf` to forbid a pair). `return_attn=True` forces the manual
        path and also returns the `(B, h, N, N)` attention probabilities, which
        is what you want for visualization or attention-rollout analysis.
        """
        b, n, d = x.shape

        # (B, N, 3D) -> (3, B, h, N, d). The permute puts the head axis next to
        # the batch axis so every head is an independent matmul in the batch.
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, h, N, d)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn and not return_attn:
            # Flash / memory-efficient kernel. Applies the 1/sqrt(d) scaling,
            # the softmax and the value matmul internally, never storing the
            # full N x N matrix.
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop_p if self.training else 0.0,
            )
            attn = None
        else:
            # Explicit reference path — same math, written out.
            attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, h, N, N)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)   # each row sums to 1: a distribution
            attn = self.attn_drop(attn)
            x = attn @ v                  # (B, h, N, d) weighted average of values

        # Merge heads: (B, h, N, d) -> (B, N, h, d) -> (B, N, D).
        # `transpose` makes the tensor non-contiguous, so reshape needs a real
        # copy; `.contiguous()` makes that explicit and cheap.
        x = x.transpose(1, 2).contiguous().reshape(b, n, d)

        x = self.proj_drop(self.proj(x))

        if return_attn:
            return x, attn
        return x
