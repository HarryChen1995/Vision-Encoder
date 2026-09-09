"""Supporting layers: the feed-forward network, DropPath and LayerScale.

Attention moves information *between* tokens. Everything in this file operates
*within* a token (or on the residual branch as a whole). Together with
attention these are all the pieces a Transformer block needs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Mlp(nn.Module):
    """Position-wise feed-forward network: Linear -> activation -> Linear.

    WHAT IT'S FOR
    -------------
    Self-attention is, per head, a *linear* operation on the values once the
    weights are decided — the only nonlinearity is inside the softmax that
    produces those weights. That is not enough to compute rich per-token
    features. The MLP supplies the network's real nonlinear capacity: it
    expands each token from D to `mlp_ratio * D` channels (4x by convention),
    applies GELU, and projects back to D. Every token goes through the same
    weights independently, so this layer costs O(N), not O(N^2).

    Note that most of a ViT's *parameters* live here, not in attention: per
    block, attention holds 4*D^2 weights while a 4x MLP holds 8*D^2. A common
    reading is that attention decides *what to gather* and the MLP decides
    *what to compute* with it (it behaves much like a key-value memory).

    GELU vs ReLU
    ------------
    GELU(x) = x * Phi(x) gates each input by the probability that a standard
    normal falls below it. It is smooth everywhere, unlike ReLU's kink at 0, so
    it never fully zeroes a unit's gradient. Transformers train slightly better
    with it, and it is the ViT default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic Depth: randomly zero an entire residual branch per sample.

    Dropout removes individual activations. DropPath removes the whole branch
    for a given sample, so that sample's block reduces to the identity
    `x = x + 0`. Across a batch, different samples effectively traverse
    networks of different depths, which regularizes deep stacks strongly and is
    standard for ViTs beyond ~12 layers.

    The surviving samples are divided by `keep_prob` ("inverted dropout") so the
    branch's expected value is unchanged and inference — where nothing is
    dropped — needs no rescaling.
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob
    # Shape (B, 1, 1, ...): one Bernoulli draw per *sample*, broadcast over
    # every token and channel, so the branch is kept or dropped as a whole.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        mask.div_(keep_prob)
    return x * mask


class DropPath(nn.Module):
    """Module wrapper around `drop_path`."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class LayerScale(nn.Module):
    """Per-channel learnable gain on a residual branch (CaiT).

    Multiplies the branch output by a learned vector `gamma` initialized to a
    tiny value (1e-5 .. 1e-4). At step 0 every block is therefore almost exactly
    the identity, and the network starts out shallow-but-stable; each block then
    learns how much to contribute. This is one of the simplest reliable fixes
    for divergence in deep ViTs, and it costs D parameters per branch.
    """

    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
