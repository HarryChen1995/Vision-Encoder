#!/usr/bin/env python3
"""Contrastive image-text pre-training (CLIP).

WHAT THIS SCRIPT IMPLEMENTS
---------------------------
The training loop from Radford et al. (2021), section 2.3:

    1. sample a batch of N (image, caption) pairs
    2. embed both modalities and L2-normalize onto the unit sphere
    3. form the N x N cosine-similarity matrix, scaled by a learned 1/tau
    4. cross-entropy against the diagonal, in both directions, averaged
    5. clamp the temperature so 1/tau <= 100

DATA FORMAT
-----------
A JSON/JSONL file, or a CSV, with one record per pair:

    [{"image": "images/001.jpg", "caption": "a dog running on grass"}, ...]

Relative paths resolve against `--image-root`. Pass `--dummy` to train on
synthetic pairs with no files at all — useful for verifying the loop end to end.
The tokenizer vocabulary is fit on the training captions and saved next to the
checkpoints so inference reproduces it exactly.

THE BATCH-SIZE CAVEAT
---------------------
InfoNCE with N in-batch negatives bounds the mutual information it can recover
at log N, so contrastive quality scales with batch size in a way ordinary
supervised training does not. OpenAI used N = 32,768 across 256 GPUs. On one
device you will be far below that, so treat single-GPU runs as mechanism
demonstrations, not accuracy reproductions. `--accum-steps` does NOT help here:
gradient accumulation makes the *gradient* larger but each micro-batch still
only sees its own negatives. Real fixes are (a) more devices with an all-gather
over the embeddings before the loss, or (b) a memory bank / MoCo-style queue.

USAGE
-----
    python train_clip.py --dummy --epochs 2 --model clip_tiny --img-size 32 --patch-size 4
    python train_clip.py --data pairs.json --image-root ./images \
        --model clip_base --img-size 224 --patch-size 16 --batch-size 256 --epochs 30
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vision_encoder import CLIP, CLIPConfig, ClipLoss, SimpleTokenizer, count_parameters
from vision_encoder.clip import _CLIP_PRESETS
from vision_encoder.utils import AverageMeter, CosineWithWarmup, param_groups_weight_decay


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_pairs(path: str) -> list[dict]:
    """Read (image, caption) records from .json, .jsonl or .csv."""
    p = Path(path)
    if p.suffix == ".jsonl":
        records = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    elif p.suffix == ".json":
        records = json.loads(p.read_text())
    elif p.suffix == ".csv":
        with p.open() as fh:
            records = list(csv.DictReader(fh))
    else:
        raise ValueError(f"unsupported data file {path!r}; use .json, .jsonl or .csv")

    out = []
    for rec in records:
        image = rec.get("image") or rec.get("image_path") or rec.get("filepath")
        caption = rec.get("caption") or rec.get("text") or rec.get("title")
        if image and caption:
            out.append({"image": str(image), "caption": str(caption)})
    if not out:
        raise ValueError(f"no usable image/caption pairs found in {path!r}")
    return out


class ImageCaptionDataset(Dataset):
    """Pairs on disk. Returns `(image_tensor, token_ids)`.

    A note on augmentation: CLIP deliberately uses only random-resized-crop.
    Heavier photometric augmentation risks breaking the image-text
    correspondence — recolor the image and "a red car" becomes a false caption,
    so the positive pair the loss is built around is no longer true.
    """

    def __init__(self, records, tokenizer, image_root=".", img_size=224, train=True) -> None:
        from torchvision import transforms  # imported lazily: only this path needs it

        self.records = records
        self.tokenizer = tokenizer
        self.image_root = Path(image_root)

        norm = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),  # CLIP's own statistics
            (0.26862954, 0.26130258, 0.27577711),
        )
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ]
            if train
            else [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                norm,
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        from PIL import Image

        rec = self.records[idx]
        path = self.image_root / rec["image"]
        image = Image.open(path).convert("RGB")
        tokens = self.tokenizer(rec["caption"])[0]
        return self.transform(image), tokens


class SyntheticPairDataset(Dataset):
    """Synthetic pairs whose captions are *predictable* from the image.

    Each sample picks a concept c, draws an image whose channel statistics
    depend on c, and emits a caption naming c. The mapping is therefore
    learnable, so contrastive accuracy should climb above chance (1/N) within a
    few hundred steps — a real end-to-end check of the loss, not just of shapes.
    """

    CONCEPTS = ["red square", "blue circle", "green triangle", "yellow star",
                "purple cross", "orange ring", "black grid", "white blob"]
    TEMPLATES = ["a photo of a {}", "an image of a {}", "a picture showing a {}", "{}"]

    def __init__(self, size: int, tokenizer, img_size: int = 32) -> None:
        self.size = size
        self.tokenizer = tokenizer
        self.img_size = img_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(idx)
        concept = idx % len(self.CONCEPTS)

        # Concept-dependent channel means -> the image genuinely carries the label.
        image = torch.randn(3, self.img_size, self.img_size, generator=g) * 0.3
        image[concept % 3] += 1.5 * (1 + concept // 3)

        caption = random.choice(self.TEMPLATES).format(self.CONCEPTS[concept])
        return image, self.tokenizer(caption)[0]


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device, epoch, args) -> dict:
    model.train()
    meters = {k: AverageMeter() for k in ("loss", "acc_i", "acc_t")}
    start = time.time()

    for step, (images, texts) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        texts = texts.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=args.amp_dtype, enabled=args.amp):
            out = model(images, texts, eot_id=args.eot_id)
            # The similarity matrix is computed in fp32 regardless: softmax over
            # scaled cosines is sensitive, and a 1/tau up to 100 pushes logits
            # near fp16's range.
            stats = loss_fn(out["logits_per_image"].float(), out["logits_per_text"].float())

        loss = stats["loss"]
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"non-finite loss at epoch {epoch} step {step}")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if args.clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        lr = scheduler.step()

        # Keep 1/tau bounded — an unclamped temperature is a known divergence mode.
        model.clamp_logit_scale()

        meters["loss"].update(loss.item(), images.size(0))
        meters["acc_i"].update(stats["acc_image"].item(), images.size(0))
        meters["acc_t"].update(stats["acc_text"].item(), images.size(0))

        if step % args.log_interval == 0:
            elapsed = max(time.time() - start, 1e-8)
            print(
                f"epoch {epoch:3d} | {step:5d}/{len(loader)} | loss {meters['loss'].avg:.4f} | "
                f"i2t {meters['acc_i'].avg:.3f} | t2i {meters['acc_t'].avg:.3f} | "
                f"1/tau {out['logit_scale'].item():.2f} | lr {lr:.2e} | "
                f"{(step + 1) * images.size(0) / elapsed:.0f} img/s",
                flush=True,
            )

    return {
        "train_loss": meters["loss"].avg,
        "train_i2t": meters["acc_i"].avg,
        "train_t2i": meters["acc_t"].avg,
    }


@torch.no_grad()
def evaluate_retrieval(model, loader, device, args) -> dict:
    """Batch-level retrieval accuracy and Recall@K.

    Caveat: these numbers are computed *within each batch*, so they depend on
    the batch size — chance is 1/N. Full-corpus retrieval (embed everything,
    then rank) is the number papers report; this is the cheap proxy you watch
    during training.
    """
    model.eval()
    loss_fn = ClipLoss()
    meters = {k: AverageMeter() for k in ("loss", "i2t", "t2i", "r5")}

    for images, texts in loader:
        images = images.to(device, non_blocking=True)
        texts = texts.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=args.amp_dtype, enabled=args.amp):
            out = model(images, texts, eot_id=args.eot_id)

        logits = out["logits_per_image"].float()
        stats = loss_fn(logits, logits.t())
        n = logits.size(0)
        targets = torch.arange(n, device=logits.device)

        k = min(5, n)
        top5 = logits.topk(k, dim=1).indices
        recall5 = (top5 == targets.unsqueeze(1)).any(dim=1).float().mean().item()

        meters["loss"].update(stats["loss"].item(), n)
        meters["i2t"].update(stats["acc_image"].item(), n)
        meters["t2i"].update(stats["acc_text"].item(), n)
        meters["r5"].update(recall5, n)

    return {
        "val_loss": meters["loss"].avg,
        "val_i2t": meters["i2t"].avg,
        "val_t2i": meters["t2i"].avg,
        "val_r@5": meters["r5"].avg,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("CLIP training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--data", default="", help="json/jsonl/csv of image-caption pairs")
    p.add_argument("--image-root", default=".")
    p.add_argument("--val-data", default="")
    p.add_argument("--val-split", type=float, default=0.05)
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--dummy-size", type=int, default=1024)
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--model", default="clip_small", choices=list(_CLIP_PRESETS))
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--context-length", type=int, default=77)
    p.add_argument("--max-vocab", type=int, default=30000)
    p.add_argument("--embed-dim", type=int, default=0, help="0 = use the preset")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.2, help="CLIP uses a large decay")
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.98],
                   help="beta2=0.98 is CLIP's; it reacts faster to gradient-scale changes")
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)

    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="./checkpoints_clip")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--resume", default="")

    return p.parse_args()


def main() -> None:
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from train import pick_device  # reuse the same device logic

    device = pick_device(args.device)
    if args.amp_dtype == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        args.amp_dtype = "fp16"
    if device.type == "cpu" and args.amp:
        print("[warn] disabling AMP on CPU")
        args.amp = False
    args.amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | amp: {args.amp}")

    # ---- tokenizer + data ------------------------------------------------
    if args.dummy:
        captions = [
            t.format(c)
            for c in SyntheticPairDataset.CONCEPTS
            for t in SyntheticPairDataset.TEMPLATES
        ]
        tokenizer = SimpleTokenizer.build(captions, context_length=args.context_length)
        train_set = SyntheticPairDataset(args.dummy_size, tokenizer, args.img_size)
        val_set = SyntheticPairDataset(max(128, args.dummy_size // 8), tokenizer, args.img_size)
    else:
        if not args.data:
            raise SystemExit("pass --data <pairs.json> or --dummy")
        records = load_pairs(args.data)
        random.shuffle(records)

        # Fit the vocabulary on training captions only — building it from the
        # validation split too would leak information.
        if args.val_data:
            train_recs, val_recs = records, load_pairs(args.val_data)
        else:
            cut = max(1, int(len(records) * (1 - args.val_split)))
            train_recs, val_recs = records[:cut], records[cut:]

        tokenizer = SimpleTokenizer.build(
            [r["caption"] for r in train_recs],
            max_vocab=args.max_vocab,
            context_length=args.context_length,
        )
        train_set = ImageCaptionDataset(train_recs, tokenizer, args.image_root, args.img_size, True)
        val_set = ImageCaptionDataset(val_recs, tokenizer, args.image_root, args.img_size, False)

    tokenizer.save(output_dir / "tokenizer.json")
    args.eot_id = tokenizer.eot_id
    print(f"train: {len(train_set)} | val: {len(val_set)} | vocab: {tokenizer.vocab_size}")

    pin = device.type == "cuda"
    # drop_last=True matters here: the loss assumes a square N x N matrix, and a
    # short final batch would quietly change the number of negatives.
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin, drop_last=True)

    # ---- model -----------------------------------------------------------
    preset = dict(_CLIP_PRESETS[args.model])
    if args.embed_dim:
        preset["embed_dim"] = args.embed_dim
    config = CLIPConfig(
        **preset,
        img_size=args.img_size,
        patch_size=args.patch_size,
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
    )
    model = CLIP(config).to(device)
    print(f"model: {args.model} | params: {count_parameters(model) / 1e6:.2f}M "
          f"| joint dim: {config.embed_dim}")

    loss_fn = ClipLoss(label_smoothing=args.label_smoothing)

    # `logit_scale` is a scalar (ndim=0), so param_groups_weight_decay already
    # routes it to the no-decay group — decaying the temperature toward 1 would
    # fight the objective directly.
    param_groups = param_groups_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, betas=tuple(args.betas), eps=args.eps
    )
    scheduler = CosineWithWarmup(
        optimizer,
        warmup_steps=args.warmup_epochs * len(train_loader),
        total_steps=args.epochs * len(train_loader),
        base_lr=args.lr,
        min_lr=args.min_lr,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and args.amp_dtype == torch.float16)

    start_epoch, best = 0, 0.0
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best = ckpt["epoch"] + 1, ckpt.get("best", 0.0)
        print(f"resumed at epoch {start_epoch}")

    # ---- loop ------------------------------------------------------------
    history = []
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, scaler, device, epoch, args
        )
        val_stats = evaluate_retrieval(model, val_loader, device, args)
        stats = {"epoch": epoch, **train_stats, **val_stats}
        history.append(stats)
        print(f"[epoch {epoch}] " + " | ".join(f"{k} {v:.4f}" for k, v in stats.items() if k != "epoch"))

        score = 0.5 * (val_stats["val_i2t"] + val_stats["val_t2i"])
        is_best = score > best
        best = max(best, score)

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best": best,
            "config": config.__dict__,
            "vocab_size": tokenizer.vocab_size,
        }
        torch.save(ckpt, output_dir / "last.pt")
        if is_best:
            torch.save(ckpt, output_dir / "best.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"done. best mean retrieval accuracy: {best:.4f}")


if __name__ == "__main__":
    main()
