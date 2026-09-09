"""Training utilities: parameter groups, LR schedule, EMA, mixup, metrics.

These are the pieces that sit *around* the model. Each one is small, but the
recipe matters at least as much as the architecture — a ViT trained with a CNN
recipe (step LR, no warmup, light augmentation) underperforms badly, and most
of the reported gains between ViT (2020) and DeiT (2021) came from exactly
these knobs rather than from any change to the network.
"""

from __future__ import annotations

import math
import random
from typing import Iterable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Optimizer parameter groups
# ---------------------------------------------------------------------------
def param_groups_weight_decay(
    model: nn.Module,
    weight_decay: float = 0.05,
    no_weight_decay_list: Iterable[str] = (),
) -> list[dict]:
    """Split parameters into decayed and non-decayed groups.

    L2/AdamW decay is meant to shrink redundant *connection* weights. Applying
    it to 1D parameters — LayerNorm gains and biases, all biases, positional
    embeddings, the CLS token, LayerScale gammas, the CLIP temperature — does
    something different and harmful: it pulls a scale or a coordinate toward
    zero, silencing a branch or erasing position information. The standard
    heuristic, used by every modern ViT recipe, is therefore:

        decay if p.ndim >= 2, else no decay.

    Skipping this typically costs ~0.5-1% top-1 on ImageNet-scale runs.
    """
    no_decay_set = set(no_weight_decay_list)
    decay, no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name in no_decay_set or name.endswith(".gamma"):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------
class CosineWithWarmup:
    """Linear warmup followed by cosine decay, stepped per iteration.

    WARMUP. At initialization the model's predictions are arbitrary, so the
    first gradients are large and poorly correlated with any useful direction.
    Adam's second-moment estimate `v` also starts at 0 and needs a few hundred
    steps to become a reliable normalizer, during which effective step sizes
    are erratic. A full-size LR here reliably breaks attention layers: scores
    saturate the softmax, gradients vanish, and the run plateaus. Ramping
    linearly from ~0 over the first few epochs avoids all of that.

        eta_t = eta_max * t / T_warm,                              t < T_warm

    COSINE DECAY. After warmup the rate follows a half cosine down to
    `min_lr`:

        eta_t = eta_min + 0.5 (eta_max - eta_min) (1 + cos(pi * p)),
        p = (t - T_warm) / (T_total - T_warm)

    It decays slowly at first (keeping exploration alive while the loss surface
    is still being mapped out) and slowly again at the end (letting the
    iterate settle into a minimum), with the fast drop in between. It
    outperforms step schedules for Transformers and has no extra
    hyperparameters to tune.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-8,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.step_count = 0
        # Per-group multipliers let one schedule drive groups with different
        # base rates (e.g. a 10x smaller rate on a pretrained backbone).
        self.lr_scales = [g.get("lr_scale", 1.0) for g in optimizer.param_groups]

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            progress = step / max(1, self.warmup_steps)
            return self.warmup_start_lr + progress * (self.base_lr - self.warmup_start_lr)

        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    def step(self) -> float:
        lr = self.get_lr(self.step_count)
        for group, scale in zip(self.optimizer.param_groups, self.lr_scales):
            group["lr"] = lr * scale
        self.step_count += 1
        return lr

    def state_dict(self) -> dict:
        return {"step_count": self.step_count}

    def load_state_dict(self, state: dict) -> None:
        self.step_count = state["step_count"]


# ---------------------------------------------------------------------------
# Exponential moving average of weights
# ---------------------------------------------------------------------------
class ModelEma:
    """Keep a slowly-updated copy of the weights: `theta' <- d*theta' + (1-d)*theta`.

    SGD with a finite learning rate never settles at a minimum; it oscillates
    in a bowl around one. Averaging the trajectory approximates the center of
    that bowl, which is both a flatter and a better-generalizing solution. The
    EMA copy usually evaluates 0.1-0.5% better than the raw weights at no
    training cost, and is what you should ship.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998, device: str | None = None) -> None:
        import copy

        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        self.device = device
        if device is not None:
            self.module.to(device)
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v.detach().to(ema_v.device), alpha=1 - self.decay)
            else:
                # Integer buffers (e.g. num_batches_tracked) are copied, not averaged.
                ema_v.copy_(model_v)


# ---------------------------------------------------------------------------
# Mixup / CutMix
# ---------------------------------------------------------------------------
class MixupCutmix:
    """Label-mixing augmentation, essential for training ViTs on small data.

    Transformers have far weaker inductive biases than CNNs — no locality, no
    translation equivariance — so they overfit small datasets badly and lean on
    aggressive augmentation instead.

    MIXUP interpolates two images and their labels:
        x = lam * x_a + (1-lam) * x_b,   y = lam * y_a + (1-lam) * y_b
    CUTMIX pastes a rectangle of image b into image a, with lam set to the
    *area* ratio so the label mix matches the pixel evidence.

    Both draw lam ~ Beta(alpha, alpha) and pair each sample with a reversed
    copy of the batch, so no extra data loading is required. The soft targets
    also act as label smoothing, discouraging overconfidence.
    """

    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        label_smoothing: float = 0.1,
        num_classes: int = 1000,
    ) -> None:
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def _one_hot(self, target: torch.Tensor) -> torch.Tensor:
        off = self.label_smoothing / self.num_classes
        on = 1.0 - self.label_smoothing + off
        return torch.full(
            (target.size(0), self.num_classes), off, device=target.device
        ).scatter_(1, target.unsqueeze(1), on)

    def _rand_bbox(self, h: int, w: int, lam: float) -> tuple[int, int, int, int]:
        """Random box covering a (1-lam) fraction of the image area."""
        ratio = math.sqrt(1.0 - lam)
        cut_h, cut_w = int(h * ratio), int(w * ratio)
        cy, cx = random.randint(0, h - 1), random.randint(0, w - 1)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)
        return y1, y2, x1, x2

    def __call__(self, x: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = self._one_hot(target)
        if random.random() > self.prob:
            return x, target

        use_cutmix = random.random() < self.switch_prob and self.cutmix_alpha > 0
        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        if alpha <= 0:
            return x, target
        lam = float(torch.distributions.Beta(alpha, alpha).sample())

        # Pair sample i with sample (B-1-i): a free permutation.
        flipped_x, flipped_target = x.flip(0), target.flip(0)

        if use_cutmix:
            y1, y2, x1, x2 = self._rand_bbox(x.size(2), x.size(3), lam)
            x = x.clone()
            x[:, :, y1:y2, x1:x2] = flipped_x[:, :, y1:y2, x1:x2]
            # Recompute lam from the *actual* box area (clipping changes it).
            lam = 1.0 - ((y2 - y1) * (x2 - x1) / (x.size(2) * x.size(3)))
        else:
            x = lam * x + (1.0 - lam) * flipped_x

        target = lam * target + (1.0 - lam) * flipped_target
        return x, target


class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy against soft (mixed) targets: -sum_c y_c log p_c."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sum(-target * torch.log_softmax(logits, dim=-1), dim=-1).mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class AverageMeter:
    """Running mean of a scalar."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


@torch.no_grad()
def accuracy(logits: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)) -> list[float]:
    """Top-k accuracy in percent."""
    maxk = min(max(topk), logits.size(1))
    batch_size = target.size(0)

    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    return [correct[:, :k].reshape(-1).float().sum().mul_(100.0 / batch_size).item() for k in topk]
