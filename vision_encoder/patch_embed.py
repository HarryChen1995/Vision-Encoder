"""Patch embedding: turning an image into a sequence of tokens.

WHY THIS EXISTS
---------------
A Transformer consumes a *sequence* of vectors, but an image is a 2D grid of
pixels. The Vision Transformer (ViT) bridges that gap with the simplest
possible idea: chop the image into a grid of non-overlapping square patches,
flatten each patch, and linearly project it to the model width. Each patch
becomes one "token", exactly analogous to a word-piece in an NLP Transformer.

    image (3, 224, 224)  --patch 16x16-->  14 x 14 = 196 patches
    each patch is 16*16*3 = 768 raw numbers  --Linear-->  768-d token
    result: a (196, 768) sequence, ready for the Transformer

THE CONV TRICK
--------------
"Cut into patches, flatten, then apply a shared Linear layer" is *mathematically
identical* to a single Conv2d whose kernel size and stride both equal the patch
size. A convolution slides a kernel over the image; when stride == kernel_size
the windows tile the image without overlap, and each output pixel is a dot
product of one patch with the kernel weights. So one Conv2d does the cutting,
the flattening and the projection in a single fused, GPU-friendly op. That is
why every ViT implementation uses a Conv2d here even though the model has no
"real" convolutional stage.

COST NOTE
---------
Patch size controls the whole compute budget. Attention is O(N^2) in the number
of tokens N, and N = (H/P) * (W/P), so N scales as 1/P^2 and attention cost as
1/P^4. Halving the patch size (16 -> 8) makes the sequence 4x longer and
self-attention roughly 16x more expensive. Small patches see finer detail;
large patches are cheap. That trade-off is the single most important knob here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _as_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize `7` -> `(7, 7)` so callers may pass square sizes as an int."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"expected a length-2 sequence, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


class PatchEmbed(nn.Module):
    """Image -> sequence of patch tokens.

    Args:
        img_size: Expected input resolution. Only used to precompute
            `grid_size`/`num_patches` (handy for sizing positional embeddings).
            The layer itself works on any resolution divisible by `patch_size`
            when `strict_img_size=False`.
        patch_size: Side length of each square patch, in pixels.
        in_chans: Input image channels (3 for RGB, 1 for grayscale).
        embed_dim: Model width `D`; every token is a D-dimensional vector.
        norm_layer: Optional normalization applied to the tokens right after
            projection. Normalizing here measurably stabilizes early training.
        flatten: If True return `(B, N, D)`. If False return the spatial map
            `(B, D, H/P, W/P)`, which is what dense-prediction decoders
            (segmentation, detection) want.
        strict_img_size: If True, reject inputs whose resolution differs from
            `img_size`. Off by default so the same weights can run at other
            resolutions (paired with positional-embedding interpolation).
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: type[nn.Module] | None = None,
        flatten: bool = True,
        strict_img_size: bool = False,
    ) -> None:
        super().__init__()
        self.img_size = _as_pair(img_size)
        self.patch_size = _as_pair(patch_size)

        if self.img_size[0] % self.patch_size[0] or self.img_size[1] % self.patch_size[1]:
            raise ValueError(
                f"img_size {self.img_size} must be divisible by patch_size {self.patch_size}"
            )

        # Token grid: how many patches fit along each spatial axis.
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim
        self.flatten = flatten
        self.strict_img_size = strict_img_size

        # THE core op. kernel_size == stride == patch_size means the kernel
        # tiles the image with no overlap, so each output position is exactly
        # one patch projected by a shared weight matrix of shape
        # (embed_dim, in_chans * P * P).
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, N, D), or (B, D, gh, gw) when `flatten=False`."""
        _, _, h, w = x.shape

        if self.strict_img_size and (h, w) != self.img_size:
            raise ValueError(f"input size {(h, w)} != expected {self.img_size}")
        if h % self.patch_size[0] or w % self.patch_size[1]:
            raise ValueError(
                f"input size {(h, w)} is not divisible by patch_size {self.patch_size}"
            )

        # (B, C, H, W) -> (B, D, H/P, W/P): one vector per patch, still on a grid.
        x = self.proj(x)

        if self.flatten:
            # (B, D, gh, gw) -> (B, D, N) -> (B, N, D).
            # The flatten walks the grid in row-major order, so token index
            # i corresponds to grid cell (i // gw, i % gw). Positional
            # embeddings must use that same ordering to stay aligned.
            x = x.flatten(2).transpose(1, 2)

        return self.norm(x)

    def extra_repr(self) -> str:
        return (
            f"img_size={self.img_size}, patch_size={self.patch_size}, "
            f"num_patches={self.num_patches}"
        )
