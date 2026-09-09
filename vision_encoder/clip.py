"""CLIP — Contrastive Language-Image Pre-training (Radford et al., 2021).

THE IDEA
--------
Train two encoders — the ViT from `encoder.py` and the causal text Transformer
from `text_encoder.py` — to place an image and its caption at the *same point*
of a shared unit hypersphere. No class labels are needed, only (image, text)
pairs, of which the web supplies hundreds of millions. The supervision signal
comes from the batch itself: given N images and N captions, the model must
match them up.

    image  --ViT-->  h_i  --W_i-->  u_i  --normalize-->  z^I_i  in S^{d-1}
    text   --Txt-->  h_t  --W_t-->  u_t  --normalize-->  z^T_i  in S^{d-1}

THE LOSS (symmetric InfoNCE)
----------------------------
For a batch of N pairs, form the similarity matrix

    S_ij = (z^I_i . z^T_j) / tau

Row i is a distribution over captions for image i; column j is a distribution
over images for caption j. The correct answer is always the diagonal, so this
is just cross-entropy against `targets = [0, 1, ..., N-1]` in both directions:

    L = 1/2 * [ CE(S, targets) + CE(S^T, targets) ]

Each direction is InfoNCE with one positive and N-1 in-batch negatives.
Minimizing it maximizes a lower bound on the mutual information between the two
views, and the bound is capped at log N — which is the formal reason CLIP needs
enormous batches (32,768 in the paper). Negatives are free: every other caption
in the batch serves as one, so batch size *is* the negative count.

WHY NORMALIZE, AND WHY A TEMPERATURE
------------------------------------
Projecting to the unit sphere makes the dot product a cosine, bounded in
[-1, 1]. That removes vector magnitude — a degenerate axis the model would
otherwise inflate to reduce loss — but it also squashes all logits into a range
where softmax is nearly uniform and gradients are tiny. The temperature tau
rescales them back. CLIP learns tau rather than tuning it, parameterizing
`logit_scale = log(1/tau)` so the exponential keeps 1/tau positive under
unconstrained gradient descent, and clamping it at 100 because runaway
sharpening is a known divergence mode.

ZERO-SHOT CLASSIFICATION
------------------------
Because the text tower can embed *any* string, a trained CLIP classifies
without a classifier: embed the prompts "a photo of a {class}" for every class,
and assign each image to the nearest prompt embedding. The text encoder has
effectively synthesized the weights of a linear classifier from language alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import VisionEncoderConfig, VisionTransformer
from .text_encoder import TextTransformer


@dataclass
class CLIPConfig:
    """Configuration for both towers plus the joint space."""

    # Vision tower
    img_size: int = 224
    patch_size: int = 16
    vision_width: int = 768
    vision_depth: int = 12
    vision_heads: int = 12
    vision_pool: str = "cls"

    # Text tower
    vocab_size: int = 30000
    context_length: int = 77
    text_width: int = 512
    text_depth: int = 12
    text_heads: int = 8
    causal_text: bool = True

    # Joint space
    embed_dim: int = 512
    init_logit_scale: float = math.log(1 / 0.07)  # CLIP's initial temperature
    max_logit_scale: float = math.log(100.0)
    extra: dict = field(default_factory=dict)


class CLIP(nn.Module):
    """Dual-encoder contrastive image-text model."""

    def __init__(self, config: CLIPConfig | None = None, **overrides) -> None:
        super().__init__()
        cfg = config or CLIPConfig()
        if overrides:
            cfg = CLIPConfig(**{**cfg.__dict__, **overrides})
        self.config = cfg

        # ---- Vision tower --------------------------------------------------
        # num_classes=0 removes the classification head: we want the pooled
        # feature vector, not logits, and then apply our own projection.
        self.visual = VisionTransformer(
            VisionEncoderConfig(
                img_size=cfg.img_size,
                patch_size=cfg.patch_size,
                embed_dim=cfg.vision_width,
                depth=cfg.vision_depth,
                num_heads=cfg.vision_heads,
                global_pool=cfg.vision_pool,
                class_token=True,
                num_classes=0,
            )
        )
        self.visual_projection = nn.Linear(cfg.vision_width, cfg.embed_dim, bias=False)

        # ---- Text tower ----------------------------------------------------
        self.text = TextTransformer(
            vocab_size=cfg.vocab_size,
            context_length=cfg.context_length,
            width=cfg.text_width,
            depth=cfg.text_depth,
            num_heads=cfg.text_heads,
            output_dim=cfg.embed_dim,
            causal=cfg.causal_text,
        )

        # ---- Learnable temperature ----------------------------------------
        # Stored in log space so that exp() is always positive; a plain
        # parameter could go negative and flip the sign of every similarity.
        self.logit_scale = nn.Parameter(torch.tensor(cfg.init_logit_scale))
        self.max_logit_scale = cfg.max_logit_scale

        nn.init.normal_(self.visual_projection.weight, std=cfg.vision_width**-0.5)

    # -------------------------------------------------------------- encoding
    def encode_image(self, image: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """`(B, 3, H, W)` -> `(B, embed_dim)` on the unit sphere."""
        features = self.visual.forward_head(self.visual.forward_features(image), pre_logits=True)
        embedding = self.visual_projection(features)
        return F.normalize(embedding, dim=-1) if normalize else embedding

    def encode_text(
        self, text: torch.Tensor, eot_id: int | None = None, normalize: bool = True
    ) -> torch.Tensor:
        """`(B, L)` token ids -> `(B, embed_dim)` on the unit sphere."""
        embedding = self.text(text, eot_id=eot_id)
        return F.normalize(embedding, dim=-1) if normalize else embedding

    def clamp_logit_scale(self) -> None:
        """Call after each optimizer step: keeps 1/tau <= 100 as in the paper."""
        with torch.no_grad():
            self.logit_scale.clamp_(0, self.max_logit_scale)

    # --------------------------------------------------------------- forward
    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        eot_id: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Returns the two embedding sets and both logit matrices.

        `logits_per_image[i, j]` scores image i against caption j; its
        transpose scores captions against images. Both are needed because the
        loss is symmetric.
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(text, eot_id=eot_id)

        logit_scale = self.logit_scale.exp()  # = 1/tau
        logits_per_image = logit_scale * image_features @ text_features.t()

        return {
            "image_features": image_features,
            "text_features": text_features,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_image.t(),
            "logit_scale": logit_scale,
        }

    # ------------------------------------------------------------- zero-shot
    @torch.no_grad()
    def build_zeroshot_classifier(
        self,
        class_prompts: list[list[str]],
        tokenizer,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Turn text prompts into classifier weights `(embed_dim, num_classes)`.

        `class_prompts[c]` holds several phrasings for class c ("a photo of a
        cat", "a blurry photo of a cat", ...). Averaging their embeddings and
        renormalizing — "prompt ensembling" — reduces the variance from any one
        phrasing and is worth a few points of accuracy for free.
        """
        device = device or next(self.parameters()).device
        weights = []
        for prompts in class_prompts:
            tokens = tokenizer(prompts).to(device)
            embeddings = self.encode_text(tokens, eot_id=getattr(tokenizer, "eot_id", None))
            mean = F.normalize(embeddings.mean(dim=0), dim=-1)
            weights.append(mean)
        return torch.stack(weights, dim=1)

    @torch.no_grad()
    def zero_shot_predict(self, image: torch.Tensor, classifier: torch.Tensor) -> torch.Tensor:
        """`(B, 3, H, W)` + `(embed_dim, C)` -> `(B, C)` logits."""
        return self.logit_scale.exp() * self.encode_image(image) @ classifier


class ClipLoss(nn.Module):
    """Symmetric InfoNCE over an in-batch similarity matrix.

    The targets are the identity permutation: element i of the image batch
    belongs with element i of the text batch. Cross-entropy over rows teaches
    image->text retrieval, over columns teaches text->image, and CLIP averages
    the two so neither modality is privileged.
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(
        self, logits_per_image: torch.Tensor, logits_per_text: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        n = logits_per_image.shape[0]
        targets = torch.arange(n, device=logits_per_image.device)

        loss_i = F.cross_entropy(logits_per_image, targets, label_smoothing=self.label_smoothing)
        loss_t = F.cross_entropy(logits_per_text, targets, label_smoothing=self.label_smoothing)
        loss = 0.5 * (loss_i + loss_t)

        with torch.no_grad():
            # Retrieval accuracy on the diagonal: a far more readable progress
            # signal than the loss value itself.
            acc_i = (logits_per_image.argmax(dim=1) == targets).float().mean()
            acc_t = (logits_per_text.argmax(dim=1) == targets).float().mean()

        return {
            "loss": loss,
            "loss_image": loss_i.detach(),
            "loss_text": loss_t.detach(),
            "acc_image": acc_i,
            "acc_text": acc_t,
        }


_CLIP_PRESETS: dict[str, dict] = {
    "clip_tiny":  dict(vision_width=192, vision_depth=6,  vision_heads=3,
                       text_width=192, text_depth=4, text_heads=3, embed_dim=192),
    "clip_small": dict(vision_width=384, vision_depth=12, vision_heads=6,
                       text_width=384, text_depth=6, text_heads=6, embed_dim=384),
    "clip_base":  dict(vision_width=768, vision_depth=12, vision_heads=12,
                       text_width=512, text_depth=12, text_heads=8, embed_dim=512),
}


def create_clip(name: str = "clip_small", **kwargs) -> CLIP:
    """Build a preset CLIP, e.g. `create_clip("clip_base", vocab_size=49408)`."""
    if name not in _CLIP_PRESETS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(_CLIP_PRESETS)}")
    return CLIP(CLIPConfig(**{**_CLIP_PRESETS[name], **kwargs}))


def available_clip_models() -> list[str]:
    return sorted(_CLIP_PRESETS)
