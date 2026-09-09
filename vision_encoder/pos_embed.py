"""Positional embeddings: telling a permutation-invariant model where things are.

THE PROBLEM
-----------
Self-attention is a weighted sum over tokens. If you shuffle the input tokens,
the outputs shuffle with them but are otherwise unchanged — attention has no
built-in notion of order or 2D location. Without extra information a ViT
literally cannot tell a patch in the top-left from one in the bottom-right, so
we *add* a position-dependent vector to every token before the first block.

TWO FLAVORS IMPLEMENTED HERE
----------------------------
1. Learnable (`nn.Parameter`, the original ViT choice): a free vector per
   position, trained by gradient descent. Maximally flexible, needs data to
   learn, and is tied to one grid size (we fix that with interpolation below).

2. 2D sin-cos (fixed, used by MAE/DINO-style setups): positions are encoded
   with sinusoids of geometrically spaced frequencies. Nothing is learned, it
   generalizes to unseen grid sizes for free, and it gives a small but real
   head start early in training. This is the 2D generalization of the original
   Transformer's 1D sinusoidal encoding: encode the row with half the channels,
   the column with the other half, and concatenate.

WHY SINUSOIDS WORK
------------------
For frequency w, the pair (sin(p*w), cos(p*w)) traces a point on a circle whose
angle is linear in position p. A shift p -> p + k is then a fixed 2D rotation
of that pair, independent of p. Stacking many frequencies gives the model a
representation in which relative offsets are simple linear maps — easy for the
attention's dot products to exploit.

RESOLUTION CHANGES
------------------
A model pretrained at 224px has a 14x14 grid of position embeddings. Fine-tuning
at 384px needs 24x24. `interpolate_pos_embed` reshapes the learned table back
to its 2D grid, bicubically resizes it, and flattens it again — the standard
recipe, and the reason ViTs transfer across resolutions at all.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def build_1d_sincos_embedding(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    """Classic sinusoidal encoding for a 1D list of positions.

    Args:
        embed_dim: Output channels; must be even (half sin, half cos).
        positions: Float tensor of arbitrary shape `(M,)` holding coordinates.

    Returns:
        `(M, embed_dim)` embedding.
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even for sin-cos, got {embed_dim}")

    # Frequencies spaced geometrically from 1 down to 1/10000, giving the model
    # both fast-varying channels (fine position) and slow ones (coarse region).
    omega = torch.arange(embed_dim // 2, dtype=torch.float64)
    omega = 1.0 / (10000.0 ** (omega / (embed_dim / 2.0)))

    # Outer product: every position against every frequency -> (M, embed_dim/2).
    angles = positions.reshape(-1).to(torch.float64)[:, None] * omega[None, :]

    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


def build_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int | tuple[int, int],
    cls_token: bool = False,
) -> torch.Tensor:
    """Fixed 2D sin-cos table for a `gh x gw` patch grid.

    Half the channels encode the row index, half encode the column index; the
    two halves are concatenated so the model can read out either coordinate
    with a linear projection.

    Returns:
        `(1, N (+1), embed_dim)`, with a leading zero row when `cls_token=True`
        (the class token has no spatial location, so it gets no position).
    """
    if embed_dim % 4 != 0:
        raise ValueError(
            f"embed_dim must be divisible by 4 for 2D sin-cos, got {embed_dim}"
        )

    gh, gw = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size

    rows = torch.arange(gh, dtype=torch.float64)
    cols = torch.arange(gw, dtype=torch.float64)
    # indexing="ij" then flattening row-major must match PatchEmbed's
    # `flatten(2)` ordering, or every token gets the wrong position.
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")

    emb_r = build_1d_sincos_embedding(embed_dim // 2, grid_r)  # (N, D/2)
    emb_c = build_1d_sincos_embedding(embed_dim // 2, grid_c)  # (N, D/2)
    pos_embed = torch.cat([emb_r, emb_c], dim=1)               # (N, D)

    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim, dtype=torch.float64), pos_embed])

    return pos_embed.unsqueeze(0).float()


@torch.no_grad()
def interpolate_pos_embed(
    pos_embed: torch.Tensor,
    new_grid_size: int | tuple[int, int],
    num_prefix_tokens: int = 1,
    mode: str = "bicubic",
) -> torch.Tensor:
    """Resize a positional-embedding table to a different patch grid.

    Args:
        pos_embed: `(1, num_prefix + N_old, D)` table to resize.
        new_grid_size: Target `(gh, gw)` (or an int for a square grid).
        num_prefix_tokens: Leading non-spatial tokens (CLS, registers) that are
            carried over untouched rather than interpolated.
        mode: Any `F.interpolate` mode; bicubic is the community default.

    Returns:
        `(1, num_prefix + gh*gw, D)`.
    """
    gh, gw = (
        (new_grid_size, new_grid_size)
        if isinstance(new_grid_size, int)
        else new_grid_size
    )

    prefix = pos_embed[:, :num_prefix_tokens]
    spatial = pos_embed[:, num_prefix_tokens:]

    num_old = spatial.shape[1]
    old_side = int(math.sqrt(num_old))
    if old_side * old_side != num_old:
        raise ValueError(
            f"cannot infer a square source grid from {num_old} spatial tokens"
        )
    if (old_side, old_side) == (gh, gw):
        return pos_embed

    dim = spatial.shape[-1]
    # (1, N, D) -> (1, D, gh_old, gw_old) so F.interpolate sees a real image.
    spatial = spatial.reshape(1, old_side, old_side, dim).permute(0, 3, 1, 2)
    spatial = F.interpolate(spatial, size=(gh, gw), mode=mode, align_corners=False)
    # ...and back to a token sequence.
    spatial = spatial.permute(0, 2, 3, 1).reshape(1, gh * gw, dim)

    return torch.cat([prefix, spatial], dim=1)
