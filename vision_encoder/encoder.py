"""The Vision Transformer encoder — the full model, assembled.

END-TO-END DATA FLOW
--------------------
    image (B, 3, H, W)
      |  PatchEmbed: Conv2d(k=P, s=P) tiles the image into N = (H/P)*(W/P)
      v  patches and projects each to D channels
    tokens (B, N, D)
      |  prepend a learned [CLS] token (+ optional register tokens)
      v
    tokens (B, 1+N, D)
      |  add positional embeddings (learned or fixed 2D sin-cos)
      v
    [ Block x L ]   each: x = x + Attn(LN(x)); x = x + MLP(LN(x))
      v
    final LayerNorm  (required by pre-norm: activations grow down the stack)
      v
    pool: take the CLS token, or average all patch tokens
      v
    features (B, D)  ->  optional Linear head  ->  logits (B, num_classes)

WHY A [CLS] TOKEN
-----------------
It is a learned vector, identical for every image, prepended to the sequence.
It owns no pixels, so it is free to act purely as an accumulator: through L
rounds of attention it queries whichever patches matter and builds a summary of
the image. Reading it out gives a single global vector without any hand-designed
pooling. The alternative — mean-pooling the patch tokens — works about as well
for classification (sometimes better) and is what `global_pool="avg"` does.

REGISTER TOKENS
---------------
Trained ViTs tend to hijack a few low-information background patches and use
them as scratch space for global computation, which shows up as bright
artifacts in attention maps and hurts dense downstream tasks. Adding a handful
of extra learnable tokens with no spatial meaning ("registers") gives the model
dedicated scratch space and cleans the maps up. They are dropped at pooling time.

WHAT AN ENCODER GIVES YOU
-------------------------
`forward_features` returns the full token sequence — the general-purpose output.
Classification pools it; segmentation/detection reshape the patch tokens back to
a (gh, gw) grid and attach a decoder; CLIP projects the pooled vector into a
shared image-text space (see `clip.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint_utils

from .block import Block
from .patch_embed import PatchEmbed
from .pos_embed import build_2d_sincos_pos_embed, interpolate_pos_embed


@dataclass
class VisionEncoderConfig:
    """Every architectural knob in one place.

    The width/depth/head triples below follow the standard ViT family: width
    grows with depth, and `head_dim = dim / num_heads` is held at 64, which is
    the sweet spot for tensor-core matmuls.
    """

    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_norm: bool = False
    num_classes: int = 1000
    global_pool: Literal["cls", "avg"] = "cls"
    class_token: bool = True
    num_register_tokens: int = 0
    pos_embed_type: Literal["learnable", "sincos", "none"] = "learnable"
    drop_rate: float = 0.0            # dropout on MLP / attn output projection
    attn_drop_rate: float = 0.0       # dropout on attention probabilities
    drop_path_rate: float = 0.0       # max stochastic-depth rate (linearly ramped)
    init_values: float | None = None  # LayerScale init; None disables it
    head_drop_rate: float = 0.0       # dropout right before the classifier
    extra: dict = field(default_factory=dict)


class VisionTransformer(nn.Module):
    """ViT-style image encoder with an optional classification head."""

    def __init__(self, config: VisionEncoderConfig | None = None, **overrides) -> None:
        super().__init__()
        cfg = config or VisionEncoderConfig()
        if overrides:
            cfg = VisionEncoderConfig(**{**cfg.__dict__, **overrides})
        self.config = cfg

        if cfg.global_pool == "cls" and not cfg.class_token:
            raise ValueError("global_pool='cls' requires class_token=True")

        self.embed_dim = cfg.embed_dim
        self.num_classes = cfg.num_classes
        self.global_pool = cfg.global_pool
        # CLS + registers sit in front of the patch tokens and carry no position.
        self.num_prefix_tokens = (1 if cfg.class_token else 0) + cfg.num_register_tokens
        self.grad_checkpointing = False

        # ---- 1. Patchify ---------------------------------------------------
        self.patch_embed = PatchEmbed(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            embed_dim=cfg.embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # ---- 2. Special tokens ---------------------------------------------
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, cfg.embed_dim)) if cfg.class_token else None
        )
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, cfg.num_register_tokens, cfg.embed_dim))
            if cfg.num_register_tokens > 0
            else None
        )

        # ---- 3. Positional information -------------------------------------
        num_pos = num_patches + self.num_prefix_tokens
        if cfg.pos_embed_type == "learnable":
            # A free parameter per position, learned like any other weight.
            self.pos_embed = nn.Parameter(torch.zeros(1, num_pos, cfg.embed_dim))
        elif cfg.pos_embed_type == "sincos":
            # Fixed table: registered as a buffer so it moves with .to(device)
            # and is saved in the state_dict, but receives no gradient.
            pos = build_2d_sincos_pos_embed(
                cfg.embed_dim, self.patch_embed.grid_size, cls_token=False
            )
            prefix = torch.zeros(1, self.num_prefix_tokens, cfg.embed_dim)
            self.register_buffer("pos_embed", torch.cat([prefix, pos], dim=1), persistent=True)
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(cfg.drop_rate)

        # ---- 4. The Transformer stack --------------------------------------
        # Stochastic depth is ramped linearly from 0 at the first block to
        # `drop_path_rate` at the last. Early layers compute features everything
        # else depends on, so dropping them is disproportionately harmful; late
        # layers are more redundant and tolerate heavier regularization.
        dpr = torch.linspace(0, cfg.drop_path_rate, cfg.depth).tolist()
        self.blocks = nn.ModuleList(
            Block(
                dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                qk_norm=cfg.qk_norm,
                drop=cfg.drop_rate,
                attn_drop=cfg.attn_drop_rate,
                init_values=cfg.init_values,
                drop_path=dpr[i],
            )
            for i in range(cfg.depth)
        )

        # ---- 5. Output normalization + head --------------------------------
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head_drop = nn.Dropout(cfg.head_drop_rate)
        self.head = (
            nn.Linear(cfg.embed_dim, cfg.num_classes)
            if cfg.num_classes > 0
            else nn.Identity()
        )

        self.init_weights()

    # ------------------------------------------------------------------ init
    def init_weights(self) -> None:
        """Truncated-normal init, std=0.02 — the BERT/ViT convention.

        Truncating at +/-2 std removes the rare large draws that a plain normal
        produces. With ~800 weights feeding each unit, one outlier can saturate
        an activation and stall that unit early in training. The special tokens
        get a smaller std because they are added to a normalized stream, and the
        classifier head starts at zero so initial logits are uniform (loss
        starts at exactly ln(num_classes), a useful sanity check).
        """
        if isinstance(self.pos_embed, nn.Parameter):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=1e-6)

        self.apply(self._init_module)

        if isinstance(self.head, nn.Linear):
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            # Patch projection: fan-in scaling keeps output variance ~1.
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        """Parameters that must be excluded from weight decay.

        Decaying a positional embedding or a CLS token shrinks a *coordinate*,
        not a redundant connection weight — it destroys information rather than
        regularizing. Same for LayerNorm gains/biases and LayerScale gammas:
        pulling a gain toward 0 silences the branch. The rule of thumb is to
        decay matrices, never vectors. See `param_groups_weight_decay`.
        """
        return {"pos_embed", "cls_token", "register_tokens"}

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Trade compute for memory: recompute block activations in backward.

        Activations for the backward pass dominate ViT memory. With
        checkpointing, blocks are re-run during backward instead of being
        stored, cutting activation memory to roughly O(1) per block for ~30%
        extra compute. This is how large batch sizes fit on small GPUs.
        """
        self.grad_checkpointing = enable

    # -------------------------------------------------------------- internals
    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend special tokens and add positional embeddings.

        Positions are interpolated on the fly when the input resolution differs
        from the one the table was built for, so a 224-trained model runs at 384
        without any code change.
        """
        b = x.shape[0]

        if self.pos_embed is not None:
            pos_embed = self.pos_embed
            expected = self.patch_embed.num_patches
            actual = x.shape[1]
            if actual != expected:
                side = int(actual**0.5)
                pos_embed = interpolate_pos_embed(
                    pos_embed, (side, side), num_prefix_tokens=self.num_prefix_tokens
                )
                pos_embed = pos_embed.to(x.dtype).to(x.device)
        else:
            pos_embed = None

        prefix = []
        if self.cls_token is not None:
            # `expand` is a view, not a copy: the same learned vector is shared
            # across the batch and gets a single accumulated gradient.
            prefix.append(self.cls_token.expand(b, -1, -1))
        if self.register_tokens is not None:
            prefix.append(self.register_tokens.expand(b, -1, -1))
        if prefix:
            x = torch.cat(prefix + [x], dim=1)

        if pos_embed is not None:
            x = x + pos_embed

        return self.pos_drop(x)

    # ------------------------------------------------------------------ API
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> full normalized token sequence `(B, num_prefix + N, D)`.

        This is the encoder's real output. Anything downstream (classification,
        retrieval, segmentation, CLIP) is a read-out of these tokens.
        """
        x = self.patch_embed(x)
        x = self._pos_embed(x)

        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint_utils.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        return self.norm(x)

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token sequence -> one vector per image `(B, D)`."""
        if self.global_pool == "cls":
            return tokens[:, 0]
        # "avg": mean over patch tokens only — prefix tokens are not spatial
        # and including them would bias the average.
        return tokens[:, self.num_prefix_tokens:].mean(dim=1)

    def forward_head(self, tokens: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        pooled = self.head_drop(self.pool(tokens))
        return pooled if pre_logits else self.head(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> logits `(B, num_classes)` (or pooled features if no head)."""
        return self.forward_head(self.forward_features(x))

    @torch.no_grad()
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int | list[int] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
    ) -> list[torch.Tensor]:
        """Extract hidden states from intermediate blocks.

        Useful because the last layer is specialized toward the training
        objective, while middle layers often carry better general-purpose or
        spatially-localized features. `reshape=True` returns each layer's patch
        tokens as a `(B, D, gh, gw)` feature map for dense decoders.
        """
        take = list(range(len(self.blocks) - n, len(self.blocks))) if isinstance(n, int) else n

        x = self.patch_embed(x)
        gh = gw = int(x.shape[1] ** 0.5)
        x = self._pos_embed(x)

        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in take:
                outputs.append(self.norm(x))

        results = []
        for out in outputs:
            prefix, patches = out[:, : self.num_prefix_tokens], out[:, self.num_prefix_tokens :]
            if reshape:
                patches = patches.reshape(x.shape[0], gh, gw, -1).permute(0, 3, 1, 2)
            results.append((patches, prefix) if return_prefix_tokens else patches)
        return results

    def reset_classifier(self, num_classes: int, global_pool: str | None = None) -> None:
        """Swap the head for transfer learning to a new label set."""
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        if isinstance(self.head, nn.Linear):
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)


# --------------------------------------------------------------------------
# Model factory. Widths/depths follow the ViT paper; head_dim stays at 64.
# --------------------------------------------------------------------------
_PRESETS: dict[str, dict] = {
    "vit_tiny":  dict(embed_dim=192,  depth=12, num_heads=3),
    "vit_small": dict(embed_dim=384,  depth=12, num_heads=6),
    "vit_base":  dict(embed_dim=768,  depth=12, num_heads=12),
    "vit_large": dict(embed_dim=1024, depth=24, num_heads=16),
    "vit_huge":  dict(embed_dim=1280, depth=32, num_heads=16),
}


def create_vision_encoder(name: str = "vit_small", **kwargs) -> VisionTransformer:
    """Build a preset model, e.g. `create_vision_encoder("vit_base", img_size=224)`."""
    if name not in _PRESETS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(_PRESETS)}")
    return VisionTransformer(VisionEncoderConfig(**{**_PRESETS[name], **kwargs}))


def available_models() -> list[str]:
    return sorted(_PRESETS)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
