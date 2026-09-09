"""A from-scratch Vision Transformer encoder and CLIP, in PyTorch.

Modules
-------
`patch_embed`  image -> patch tokens (the Conv2d tiling trick)
`pos_embed`    learnable / 2D sin-cos positions, plus resolution interpolation
`attention`    multi-head self-attention (fused SDPA + a readable fallback)
`layers`       MLP, DropPath (stochastic depth), LayerScale
`block`        pre-norm Transformer block
`encoder`      the ViT itself + model presets
`text_encoder` causal Transformer for text
`tokenizer`    dependency-free tokenizer for the text tower
`clip`         dual-encoder contrastive model + symmetric InfoNCE loss
`utils`        optimizer param groups, LR schedule, EMA, mixup/cutmix, metrics
"""

from .attention import MultiHeadSelfAttention
from .block import Block
from .clip import (
    CLIP,
    CLIPConfig,
    ClipLoss,
    available_clip_models,
    create_clip,
)
from .encoder import (
    VisionEncoderConfig,
    VisionTransformer,
    available_models,
    count_parameters,
    create_vision_encoder,
)
from .layers import DropPath, LayerScale, Mlp
from .patch_embed import PatchEmbed
from .pos_embed import build_2d_sincos_pos_embed, interpolate_pos_embed
from .text_encoder import TextTransformer, build_causal_mask
from .tokenizer import SimpleTokenizer

__version__ = "0.1.0"

__all__ = [
    "CLIP",
    "Block",
    "CLIPConfig",
    "ClipLoss",
    "DropPath",
    "LayerScale",
    "Mlp",
    "MultiHeadSelfAttention",
    "PatchEmbed",
    "SimpleTokenizer",
    "TextTransformer",
    "VisionEncoderConfig",
    "VisionTransformer",
    "available_clip_models",
    "available_models",
    "build_2d_sincos_pos_embed",
    "build_causal_mask",
    "count_parameters",
    "create_clip",
    "create_vision_encoder",
    "interpolate_pos_embed",
]
