#!/usr/bin/env python3
"""Supervised training for the Vision Transformer encoder.

WHAT THIS SCRIPT IMPLEMENTS
---------------------------
A complete, single-GPU (or CPU/MPS) ViT recipe of the DeiT family:

  * AdamW with decoupled weight decay, applied only to >=2D parameters
  * linear warmup -> cosine decay, stepped every iteration
  * mixup + cutmix with soft-target cross-entropy
  * stochastic depth (ramped by block index) and label smoothing
  * mixed precision (bf16/fp16) with gradient scaling where needed
  * gradient clipping, gradient accumulation, EMA weights
  * resumable checkpointing and best-model tracking

WHY THE RECIPE MATTERS AS MUCH AS THE MODEL
-------------------------------------------
A CNN encodes locality and translation equivariance in its architecture. A ViT
encodes almost nothing: any patch can attend to any other from layer 1, so the
model must *learn* that nearby patches are related. That flexibility is what
makes ViTs scale better than CNNs on large corpora, and exactly what makes them
overfit badly on small ones. Everything above exists to inject, through data
and regularization, the priors the architecture does not have.

Rule of thumb: below ~10M images, regularize hard (this script's defaults).
Above that, dial augmentation down and let the data do the work.

USAGE
-----
    # smoke test, no dataset needed
    python train.py --dummy --epochs 1 --model vit_tiny --img-size 32 --patch-size 4

    # CIFAR-10 (downloads on first run; needs torchvision)
    python train.py --dataset cifar10 --model vit_tiny --img-size 32 \
        --patch-size 4 --epochs 100 --batch-size 128 --lr 1e-3

    # ImageNet-style folder layout: <root>/train/<class>/*.jpg, <root>/val/...
    python train.py --dataset folder --data-dir /path/to/imagenet \
        --model vit_base --img-size 224 --patch-size 16 --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from vision_encoder import VisionEncoderConfig, VisionTransformer, count_parameters
from vision_encoder.utils import (
    AverageMeter,
    CosineWithWarmup,
    MixupCutmix,
    ModelEma,
    SoftTargetCrossEntropy,
    accuracy,
    param_groups_weight_decay,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed every RNG we touch.

    Note this does not make CUDA bit-wise deterministic — cuDNN autotuning and
    atomic reductions still introduce nondeterminism. Add
    `torch.use_deterministic_algorithms(True)` if you need exact repeats and
    can accept the slowdown.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class SyntheticDataset(Dataset):
    """Random images with random labels — for shape/throughput smoke tests.

    Useful precisely because it cannot be learned: if training loss on this
    falls well below ln(num_classes) the model is memorizing, which is a quick
    way to confirm the optimizer and gradient flow are wired up correctly.
    """

    def __init__(self, size: int, img_size: int, num_classes: int, in_chans: int = 3) -> None:
        self.size = size
        self.img_size = img_size
        self.num_classes = num_classes
        self.in_chans = in_chans

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        generator = torch.Generator().manual_seed(idx)
        image = torch.randn(self.in_chans, self.img_size, self.img_size, generator=generator)
        label = torch.randint(0, self.num_classes, (1,), generator=generator).item()
        return image, label


def build_datasets(args) -> tuple[Dataset, Dataset, int]:
    """Return `(train_set, val_set, num_classes)`.

    AUGMENTATION NOTES (the torchvision paths)
    ------------------------------------------
    RandomResizedCrop is the workhorse: sampling a random scale and aspect
    ratio before resizing teaches scale invariance, which a ViT has no
    architectural way to acquire. RandAugment layers on photometric and
    geometric ops of random magnitude. RandomErasing masks a rectangle after
    normalization, forcing the model to use distributed rather than local
    evidence. Together with mixup/cutmix (applied on-GPU in the training loop)
    this is the DeiT augmentation stack.
    """
    if args.dummy:
        num_classes = args.num_classes or 10
        train = SyntheticDataset(args.dummy_size, args.img_size, num_classes)
        val = SyntheticDataset(max(64, args.dummy_size // 8), args.img_size, num_classes)
        return train, val, num_classes

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "torchvision is required for real datasets. Install it, or pass "
            "--dummy to run on synthetic data."
        ) from exc

    if args.dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        num_classes = 10
    elif args.dataset == "cifar100":
        mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        num_classes = 100
    else:  # ImageNet statistics for generic folder datasets
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        num_classes = args.num_classes or 1000

    train_tf = [
        transforms.RandomResizedCrop(args.img_size, scale=(args.min_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if args.randaug:
        train_tf.append(transforms.RandAugment(num_ops=2, magnitude=args.randaug_magnitude))
    train_tf += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if args.random_erasing > 0:
        train_tf.append(transforms.RandomErasing(p=args.random_erasing))
    train_tf = transforms.Compose(train_tf)

    # Eval: resize to 256/224 of the target, center crop. Deterministic, so the
    # validation number is comparable across epochs and runs.
    eval_tf = transforms.Compose([
        transforms.Resize(int(args.img_size * 256 / 224)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if args.dataset in ("cifar10", "cifar100"):
        cls = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
        train = cls(args.data_dir, train=True, download=True, transform=train_tf)
        val = cls(args.data_dir, train=False, download=True, transform=eval_tf)
    else:
        train = datasets.ImageFolder(os.path.join(args.data_dir, "train"), train_tf)
        val = datasets.ImageFolder(os.path.join(args.data_dir, "val"), eval_tf)
        num_classes = len(train.classes)

    return train, val, num_classes


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler, device, epoch, args, mixup, ema
) -> dict:
    model.train()
    loss_meter, batch_time = AverageMeter(), AverageMeter()
    start = time.time()

    optimizer.zero_grad(set_to_none=True)

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup is not None:
            images, targets = mixup(images, targets)

        # Mixed precision: matmuls run in bf16/fp16 (roughly 2x throughput and
        # half the activation memory on tensor cores) while reductions and the
        # master weights stay fp32.
        with torch.autocast(device_type=device.type, dtype=args.amp_dtype, enabled=args.amp):
            logits = model(images)
            loss = criterion(logits, targets)
            # Accumulation: scale so that the sum over `accum_steps` micro
            # batches equals the mean over one large batch.
            loss = loss / args.accum_steps

        if not math.isfinite(loss.item()):
            raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}; lower the LR")

        # fp16 has ~5 exponent bits, so small gradients underflow to zero. The
        # scaler multiplies the loss by a large factor before backward and
        # divides it out before the step. bf16 has fp32's exponent range and
        # needs no scaling.
        scaler.scale(loss).backward()

        if (step + 1) % args.accum_steps == 0:
            if args.clip_grad > 0:
                # Unscale first, or we would clip the *scaled* gradients.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr = scheduler.step()
            if ema is not None:
                ema.update(model)
        else:
            lr = optimizer.param_groups[0]["lr"]

        loss_meter.update(loss.item() * args.accum_steps, images.size(0))
        batch_time.update(time.time() - start)
        start = time.time()

        if step % args.log_interval == 0:
            print(
                f"epoch {epoch:3d} | {step:5d}/{len(loader)} | "
                f"loss {loss_meter.avg:.4f} | lr {lr:.2e} | "
                f"{images.size(0) / max(batch_time.avg, 1e-8):.0f} img/s",
                flush=True,
            )

    return {"train_loss": loss_meter.avg, "lr": lr}


@torch.no_grad()
def evaluate(model, loader, device, args) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_meter, top1_meter, top5_meter = AverageMeter(), AverageMeter(), AverageMeter()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=args.amp_dtype, enabled=args.amp):
            logits = model(images)
            loss = criterion(logits, targets)

        acc1, acc5 = accuracy(logits.float(), targets, topk=(1, 5))
        loss_meter.update(loss.item(), images.size(0))
        top1_meter.update(acc1, images.size(0))
        top5_meter.update(acc5, images.size(0))

    return {"val_loss": loss_meter.avg, "top1": top1_meter.avg, "top5": top5_meter.avg}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ViT encoder training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "folder"])
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--dummy", action="store_true", help="train on synthetic data (no download)")
    p.add_argument("--dummy-size", type=int, default=512)
    p.add_argument("--num-classes", type=int, default=0, help="0 = infer from the dataset")
    p.add_argument("--workers", type=int, default=4)

    # model
    p.add_argument("--model", default="vit_tiny",
                   choices=["vit_tiny", "vit_small", "vit_base", "vit_large", "vit_huge"])
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--global-pool", default="cls", choices=["cls", "avg"])
    p.add_argument("--num-registers", type=int, default=0)
    p.add_argument("--pos-embed", default="learnable", choices=["learnable", "sincos", "none"])
    p.add_argument("--qk-norm", action="store_true", help="LayerNorm on Q/K (stability)")
    p.add_argument("--layer-scale", type=float, default=0.0, help="LayerScale init (0 disables)")
    p.add_argument("--grad-checkpointing", action="store_true")

    # regularization
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--attn-drop", type=float, default=0.0)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.8)
    p.add_argument("--cutmix", type=float, default=1.0)
    p.add_argument("--no-mixup", action="store_true")
    p.add_argument("--randaug", action="store_true", default=True)
    p.add_argument("--randaug-magnitude", type=int, default=9)
    p.add_argument("--random-erasing", type=float, default=0.25)
    p.add_argument("--min-scale", type=float, default=0.65)

    # optimization
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999])
    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.9998)

    # runtime
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="./checkpoints")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--resume", default="")
    p.add_argument("--eval-only", action="store_true")

    return p.parse_args()


def main() -> None:
    args = get_args()
    set_seed(args.seed)

    device = pick_device(args.device)
    # bf16 needs Ampere+ on CUDA; fall back rather than crash mid-run.
    if args.amp_dtype == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("[warn] bf16 unsupported on this GPU, falling back to fp16")
        args.amp_dtype = "fp16"
    if device.type == "cpu" and args.amp:
        print("[warn] disabling AMP on CPU")
        args.amp = False
    args.amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | amp: {args.amp} ({args.amp_dtype})")

    # ---- data ------------------------------------------------------------
    train_set, val_set, num_classes = build_datasets(args)
    # pin_memory + non_blocking overlaps the host->device copy with compute;
    # only meaningful on CUDA.
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=pin, drop_last=True, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers,
        pin_memory=pin, persistent_workers=args.workers > 0,
    )
    print(f"train: {len(train_set)} | val: {len(val_set)} | classes: {num_classes}")

    # ---- model -----------------------------------------------------------
    from vision_encoder.encoder import _PRESETS

    config = VisionEncoderConfig(
        **_PRESETS[args.model],
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_classes=num_classes,
        global_pool=args.global_pool,
        num_register_tokens=args.num_registers,
        pos_embed_type=args.pos_embed,
        qk_norm=args.qk_norm,
        drop_rate=args.drop,
        attn_drop_rate=args.attn_drop,
        drop_path_rate=args.drop_path,
        init_values=args.layer_scale or None,
    )
    model = VisionTransformer(config).to(device)
    if args.grad_checkpointing:
        model.set_grad_checkpointing(True)

    print(f"model: {args.model} | params: {count_parameters(model) / 1e6:.2f}M "
          f"| tokens: {model.patch_embed.num_patches} + {model.num_prefix_tokens}")

    ema = ModelEma(model, decay=args.ema_decay) if args.ema else None

    # ---- loss ------------------------------------------------------------
    use_mixup = not args.no_mixup and (args.mixup > 0 or args.cutmix > 0)
    if use_mixup:
        mixup = MixupCutmix(args.mixup, args.cutmix, label_smoothing=args.label_smoothing,
                            num_classes=num_classes)
        # Mixup already produces soft targets that encode the smoothing, so the
        # criterion must consume distributions, not integer labels.
        criterion = SoftTargetCrossEntropy()
    else:
        mixup = None
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ---- optimizer + schedule -------------------------------------------
    # AdamW, not Adam: it applies decay directly to the weights rather than
    # adding an L2 term to the gradient. With adaptive per-parameter step
    # sizes those are NOT equivalent — L2 gets divided by sqrt(v), so
    # frequently-updated weights get decayed less. Decoupling fixes that and is
    # what makes weight decay work at all for Transformers.
    param_groups = param_groups_weight_decay(model, args.weight_decay, model.no_weight_decay())
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=tuple(args.betas))

    steps_per_epoch = max(1, len(train_loader) // args.accum_steps)
    scheduler = CosineWithWarmup(
        optimizer,
        warmup_steps=args.warmup_epochs * steps_per_epoch,
        total_steps=args.epochs * steps_per_epoch,
        base_lr=args.lr,
        min_lr=args.min_lr,
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=args.amp and args.amp_dtype == torch.float16
    )

    # ---- resume ----------------------------------------------------------
    start_epoch, best_acc = 0, 0.0
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        if ema is not None and ckpt.get("ema"):
            ema.module.load_state_dict(ckpt["ema"])
        start_epoch, best_acc = ckpt["epoch"] + 1, ckpt.get("best_acc", 0.0)
        print(f"resumed from {args.resume} at epoch {start_epoch} (best {best_acc:.2f}%)")

    if args.eval_only:
        print(json.dumps(evaluate(model, val_loader, device, args), indent=2))
        return

    # ---- training loop ---------------------------------------------------
    history = []
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, epoch, args, mixup, ema,
        )
        val_stats = evaluate(model, val_loader, device, args)
        if ema is not None:
            ema_stats = evaluate(ema.module, val_loader, device, args)
            val_stats["ema_top1"] = ema_stats["top1"]

        stats = {"epoch": epoch, **train_stats, **val_stats}
        history.append(stats)
        print(f"[epoch {epoch}] " + " | ".join(f"{k} {v:.4f}" for k, v in stats.items() if k != "epoch"))

        current = max(val_stats["top1"], val_stats.get("ema_top1", 0.0))
        is_best = current > best_acc
        best_acc = max(best_acc, current)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.module.state_dict() if ema is not None else None,
            "epoch": epoch,
            "best_acc": best_acc,
            "config": config.__dict__,
            "args": vars(args) | {"amp_dtype": str(args.amp_dtype)},
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if is_best:
            torch.save(checkpoint, output_dir / "best.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"done. best top-1: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
