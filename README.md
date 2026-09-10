# Vision Encoder: A Vision Transformer and CLIP, from scratch in PyTorch

A single-file-per-concept, heavily annotated reimplementation of the Vision
Transformer (ViT) encoder [1] and of Contrastive Language–Image Pre-training
(CLIP) [2]. Every module is written to be read: the comments derive the math,
justify the design choices, and flag the failure modes that the equations alone
do not reveal.

The goal is not another `timm` — it is a reference implementation whose
correctness you can verify line by line, with the training recipe (which
matters as much as the architecture) written out explicitly rather than hidden
behind a config system.

```
image ─▶ patchify ─▶ +[CLS] ─▶ +pos ─▶ [pre-norm block]×L ─▶ LN ─▶ pool ─▶ head
                                          │
text  ─▶ tokenize ─▶ +pos ─▶ [causal block]×L ─▶ LN ─▶ [EOT] ──┴──▶ InfoNCE (CLIP)
```

---

## Contents

1. [Installation](#installation)
2. [Quickstart](#quickstart)
3. [Repository layout](#repository-layout)
4. [Part I — The Vision Transformer](#part-i--the-vision-transformer)
   - [Patch embedding](#1-patch-embedding)
   - [Positional embeddings](#2-positional-embeddings)
   - [Multi-head self-attention](#3-multi-head-self-attention)
   - [The feed-forward network](#4-the-feed-forward-network)
   - [Pre-norm residual blocks](#5-pre-norm-residual-blocks)
   - [Pooling and read-out](#6-pooling-and-read-out)
   - [Complexity analysis](#7-complexity-analysis)
5. [Part II — CLIP](#part-ii--clip)
   - [The contrastive objective](#1-the-contrastive-objective)
   - [Temperature and the mutual-information bound](#2-temperature-and-the-mutual-information-bound)
   - [Zero-shot transfer](#3-zero-shot-transfer)
6. [Part III — The training recipe](#part-iii--the-training-recipe)
7. [Model presets](#model-presets)
8. [Testing](#testing)
9. [Extending this code](#extending-this-code)
10. [References](#references)

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch>=2.0` is the only hard dependency (needed for the fused
`scaled_dot_product_attention` kernels). `torchvision` and `pillow` are required
only for the real-dataset paths; every training loop runs end to end on
synthetic data with `--dummy` if they are absent.

## Quickstart

```bash
# 1. Verify the whole stack on synthetic data (no downloads, runs on CPU)
python train.py --dummy --epochs 2 --model vit_tiny --img-size 32 --patch-size 4

# 2. Train a ViT on CIFAR-10
python train.py --dataset cifar10 --model vit_tiny --img-size 32 --patch-size 4 \
    --epochs 200 --batch-size 128 --lr 1e-3 --drop-path 0.1

# 3. Train on an ImageNet-style folder tree (<root>/train/<class>/*.jpg)
python train.py --dataset folder --data-dir /path/to/data --model vit_base \
    --img-size 224 --patch-size 16 --batch-size 256 --epochs 300 --ema

# 4. Contrastive image–text pre-training
python train_clip.py --dummy --epochs 6 --model clip_tiny --img-size 32 --patch-size 4
python train_clip.py --data pairs.json --image-root ./images --model clip_base \
    --img-size 224 --batch-size 256 --epochs 30
```

Using the library directly:

```python
import torch
from vision_encoder import create_vision_encoder, create_clip, SimpleTokenizer

model = create_vision_encoder("vit_base", img_size=224, patch_size=16, num_classes=1000)
logits = model(torch.randn(2, 3, 224, 224))            # (2, 1000)
tokens = model.forward_features(torch.randn(2, 3, 224, 224))   # (2, 197, 768)

# Dense features for segmentation/detection decoders, as a (B, D, gh, gw) map
maps = model.get_intermediate_layers(torch.randn(1, 3, 224, 224), n=4, reshape=True)

# Zero-shot classification with CLIP
tok = SimpleTokenizer.build(["a photo of a cat", "a photo of a dog"])
clip = create_clip("clip_base", vocab_size=tok.vocab_size)
W = clip.build_zeroshot_classifier([["a photo of a cat"], ["a photo of a dog"]], tok)
preds = clip.zero_shot_predict(torch.randn(4, 3, 224, 224), W).argmax(-1)
```

## Repository layout

| Path | Contents |
|---|---|
| `vision_encoder/patch_embed.py` | Image → patch tokens; the Conv2d tiling identity |
| `vision_encoder/pos_embed.py` | Learnable and 2D sin–cos positions; resolution interpolation |
| `vision_encoder/attention.py` | Multi-head self-attention (fused SDPA + reference path) |
| `vision_encoder/layers.py` | MLP, DropPath (stochastic depth), LayerScale |
| `vision_encoder/block.py` | Pre-norm Transformer block |
| `vision_encoder/encoder.py` | `VisionTransformer`, config dataclass, model presets |
| `vision_encoder/text_encoder.py` | Causal Transformer text tower |
| `vision_encoder/tokenizer.py` | Dependency-free tokenizer with byte-level backoff |
| `vision_encoder/clip.py` | Dual-encoder CLIP + symmetric InfoNCE loss |
| `vision_encoder/utils.py` | Param groups, LR schedule, EMA, mixup/cutmix, metrics |
| `train.py` | Supervised classification training |
| `train_clip.py` | Contrastive image–text training |
| `tests/test_models.py` | 27 shape, invariance and gradient tests |

---

# Part I — The Vision Transformer

Notation. $B$ batch size, $H \times W$ input resolution, $C$ input channels,
$P$ patch size, $N = HW/P^2$ patch tokens, $D$ model width, $L$ depth,
$h$ heads, $d = D/h$ head dimension.

## 1. Patch embedding

An image is a grid; a Transformer eats a sequence. ViT bridges the two in the
crudest way that works: tile the image into non-overlapping $P \times P$
patches and linearly project each one.

```math
\mathbf{z}_0 = [\, \mathbf{x}_\text{cls} \;;\; \mathbf{x}^1_p \mathbf{E} \;;\;
\mathbf{x}^2_p \mathbf{E} \;;\; \dots \;;\; \mathbf{x}^N_p \mathbf{E} \,] + \mathbf{E}_\text{pos},
\qquad
\mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}
```

**The Conv2d identity.** Extracting patches, flattening them and applying a
shared $\mathbf{E}$ is *exactly* a convolution with `kernel_size = stride = P`:
when the stride equals the kernel size the windows tile the plane without
overlap, and each output position is one patch's dot product with the kernel.
One fused op replaces three. This equivalence is asserted numerically in
`test_patch_embed_equals_unfold_plus_linear`.

**Why $P$ is the most important hyperparameter.** Since $N \propto P^{-2}$ and
attention costs $O(N^2)$, attention compute scales as $P^{-4}$. Going from
$P=16$ to $P=8$ lengthens the sequence $4\times$ and makes attention $16\times$
more expensive. Small patches resolve fine detail; large patches are cheap.

## 2. Positional embeddings

Self-attention is permutation-equivariant: for any permutation matrix
$\mathbf{\Pi}$,

```math
\text{Attn}(\mathbf{\Pi}\mathbf{X}) = \mathbf{\Pi}\,\text{Attn}(\mathbf{X}).
```

The model therefore cannot distinguish a patch's location from its content —
a property this repository tests explicitly
(`test_attention_is_permutation_equivariant`). Position must be *added* to the
tokens. Two schemes are implemented.

**Learnable** (`pos_embed_type="learnable"`, the original ViT): a free
parameter $\mathbf{E}_\text{pos} \in \mathbb{R}^{(N+1) \times D}$.

**Fixed 2D sin–cos** (`"sincos"`, used by MAE/DINO): for a patch at grid
position $(r, c)$, half the channels encode the row and half the column,

```math
\text{PE}(p, 2i) = \sin\!\left(\frac{p}{10000^{2i/(D/2)}}\right),
\qquad
\text{PE}(p, 2i{+}1) = \cos\!\left(\frac{p}{10000^{2i/(D/2)}}\right),
```

```math
\mathbf{E}_\text{pos}(r,c) = \big[\,\text{PE}(r) \;\Vert\; \text{PE}(c)\,\big].
```

The reason sinusoids help is that a shift $p \mapsto p + k$ acts on each
frequency pair as a **fixed rotation independent of $p$**:

```math
\begin{bmatrix}\sin(\omega(p{+}k))\\ \cos(\omega(p{+}k))\end{bmatrix}
= \begin{bmatrix}\cos \omega k & \sin \omega k\\ -\sin \omega k & \cos \omega k\end{bmatrix}
\begin{bmatrix}\sin(\omega p)\\ \cos(\omega p)\end{bmatrix},
```

so relative offsets are linear maps — precisely the structure the attention
dot product can exploit.

**Changing resolution.** A model pretrained at 224px has a $14\times14$
position grid; fine-tuning at 384px needs $24\times24$. `interpolate_pos_embed`
reshapes the table to 2D, resizes it bicubically, and flattens it back. This
happens automatically in `forward`, so a 32px-trained model accepts 64px input
without any code change (`test_vit_handles_new_resolution`).

## 3. Multi-head self-attention

Each token emits a query, a key and a value; the output is a similarity-weighted
average of values.

```math
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad
\mathbf{V} = \mathbf{X}\mathbf{W}_V,
```

```math
\boxed{\;
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})
= \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}} + \mathbf{M}\right)\mathbf{V}
\;}
```

with $\mathbf{M}$ an optional additive mask ($-\infty$ forbids a pair, since
$e^{-\infty} = 0$). Multi-head attention runs $h$ of these in parallel on
$d = D/h$ channels each:

```math
\text{MHSA}(\mathbf{X}) = \big[\text{head}_1 \Vert \cdots \Vert \text{head}_h\big]\mathbf{W}_O,
\qquad
\text{head}_i = \text{Attention}(\mathbf{X}\mathbf{W}_Q^i, \mathbf{X}\mathbf{W}_K^i, \mathbf{X}\mathbf{W}_V^i).
```

**Why $1/\sqrt{d}$.** If $q_j, k_j$ are independent with zero mean and unit
variance, then

```math
\text{Var}\!\left(\mathbf{q}\cdot\mathbf{k}\right)
= \text{Var}\!\left(\sum_{j=1}^{d} q_j k_j\right) = d .
```

Un-normalized scores therefore grow like $\sqrt{d}$, pushing softmax toward a
one-hot distribution where its Jacobian
$\text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top \to \mathbf{0}$ and
gradients vanish. Dividing by $\sqrt{d}$ restores unit score variance.

**Why several heads.** A single softmax yields one averaging pattern per token
— one relation per layer. Splitting into $h$ heads gives $h$ independent
relations at identical FLOP cost, since each operates on $D/h$ channels. The
output projection $\mathbf{W}_O$ is what lets heads combine; without it they
would never interact.

**QK-normalization** (`--qk-norm`) applies LayerNorm to $\mathbf{Q}$ and
$\mathbf{K}$ before the product, bounding logits regardless of activation
growth. It is a cheap and reliable fix for the attention-logit divergence seen
in large-scale runs [10].

**Implementation.** The forward pass calls `F.scaled_dot_product_attention`,
which dispatches to FlashAttention [8]. Flash never materializes the
$B{\times}h{\times}N{\times}N$ score matrix in HBM — it tiles the computation
and recomputes in SRAM, converting attention's *memory* from $O(N^2)$ to
$O(N)$ while computing an identical result. A readable manual path is retained
for attention-map extraction and is tested against the fused one
(`test_fused_and_manual_attention_agree`).

## 4. The feed-forward network

```math
\text{MLP}(\mathbf{x}) = \mathbf{W}_2\,\sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2,
\qquad
\mathbf{W}_1 \in \mathbb{R}^{rD \times D},\; r = 4 .
```

Attention is, once its weights are fixed, a *linear* map on the values — the
only nonlinearity lives inside the softmax that produces the weights. The MLP
supplies the network's actual nonlinear capacity, applied per token and
therefore costing $O(N)$, not $O(N^2)$.

The activation is GELU [11], $\sigma(x) = x\,\Phi(x)$, which gates an input by
the probability that a standard normal falls below it. It is smooth everywhere,
unlike ReLU's kink at the origin.

Note where the parameters live: per block, attention holds $4D^2$ weights and
the $4\times$ MLP holds $8D^2$. **Two thirds of a ViT's parameters are in the
MLPs.** A useful reading is that attention decides *what to gather* while the
MLP decides *what to compute* with it — the latter behaving much like a
key–value memory [12].

## 5. Pre-norm residual blocks

```math
\begin{aligned}
\mathbf{z}'_\ell &= \mathbf{z}_{\ell-1} + \text{DropPath}\big(\gamma_1 \odot \text{MHSA}(\text{LN}(\mathbf{z}_{\ell-1}))\big), \\
\mathbf{z}_\ell  &= \mathbf{z}'_\ell + \text{DropPath}\big(\gamma_2 \odot \text{MLP}(\text{LN}(\mathbf{z}'_\ell))\big).
\end{aligned}
```

**Pre-norm vs post-norm.** The original Transformer [3] normalized *after* the
residual add, $\mathbf{z} = \text{LN}(\mathbf{z} + f(\mathbf{z}))$, placing a
LayerNorm directly on the residual path so that gradients are rescaled at each
of $L$ layers. Pre-norm normalizes only the branch *input*, leaving the
residual path a clean sum. Its Jacobian is

```math
\frac{\partial \mathbf{z}_\ell}{\partial \mathbf{z}_{\ell-1}} = \mathbf{I} + \frac{\partial f}{\partial \mathbf{z}_{\ell-1}},
```

so gradient reaches early layers through the identity term even when $f' \to 0$.
This is what makes 24+ layer stacks trainable without delicate warmup [4]. The
price: activations grow monotonically down the stack, which is why a **final
LayerNorm before the head is mandatory**, not decorative.

**Stochastic depth** [5] drops an entire residual branch per sample with
probability $p_\ell$, so that sample's block reduces to the identity. Surviving
samples are divided by $1 - p_\ell$ to keep the branch's expectation intact.
The rate is ramped linearly with depth,
$p_\ell = \frac{\ell}{L-1} \cdot p_\text{max}$, because early layers compute
features everything downstream depends on, whereas late layers are more
redundant.

**LayerScale** [6] multiplies each branch by a learned per-channel gain
$\gamma$ initialized to $\sim 10^{-5}$, so at step 0 every block is nearly the
identity and the network starts effectively shallow, then deepens itself.

## 6. Pooling and read-out

Two options, both supported:

```math
\mathbf{f}_\text{cls} = \mathbf{z}_L^{(0)},
\qquad
\mathbf{f}_\text{avg} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{z}_L^{(i + n_\text{prefix})}.
```

The `[CLS]` token is a learned vector, identical for every image, that owns no
pixels and is therefore free to act as an accumulator across $L$ rounds of
attention. Mean pooling over patch tokens works about as well — sometimes
better — and is selected with `--global-pool avg`. Prefix tokens are excluded
from the mean, since they carry no spatial meaning.

**Register tokens** [7] (`--num-registers`) add a few extra learnable tokens
with no position. Trained ViTs otherwise hijack low-information background
patches as scratch space for global computation, producing high-norm artifacts
that corrupt attention maps and hurt dense downstream tasks. Registers give the
model dedicated scratch space instead.

## 7. Complexity analysis

Per block, with $T = N + n_\text{prefix}$ tokens:

| Component | FLOPs (MACs) | Parameters | Scaling |
|---|---|---|---|
| QKV projection | $3TD^2$ | $3D^2$ | $O(N)$ |
| Attention scores $\mathbf{Q}\mathbf{K}^\top$ | $T^2D$ | — | $O(N^2)$ |
| Attention $\times \mathbf{V}$ | $T^2D$ | — | $O(N^2)$ |
| Output projection | $TD^2$ | $D^2$ | $O(N)$ |
| MLP | $2rTD^2$ | $2rD^2$ | $O(N)$ |
| **Total** | $\mathbf{(4{+}2r)TD^2 + 2T^2D}$ | $\mathbf{(4{+}2r)D^2}$ | |

The crossover where attention overtakes the projections is at $T \approx (2+r)D$
— for ViT-B ($D{=}768$, $r{=}4$) that is $\approx 4600$ tokens, far beyond the
196 used at 224px. **At standard resolutions a ViT is projection-bound, not
attention-bound**, which is why the quadratic term causes less trouble than its
reputation suggests, and why efficient-attention variants buy little at 224px:

| Model | Resolution | Tokens | GMACs | Attention share |
|---|---|---|---|---|
| ViT-B/16 | 224 | 196 | 17.6 | 4.1% |
| ViT-B/16 | 384 | 576 | 55.5 | 11.1% |
| ViT-B/16 | 512 | 1024 | 107.0 | 18.1% |

Memory is the other half of the story: the score matrix is
$O(B h N^2)$, which is what FlashAttention eliminates and what gradient
checkpointing (`--grad-checkpointing`, ~30% extra compute for near-constant
activation memory) attacks from the other side.

---

# Part II — CLIP

Two encoders — the ViT above and a causal text Transformer — are trained to map
an image and its caption to the *same point* on a unit hypersphere. Supervision
comes from pairing alone, so the training set is the web rather than a labeled
corpus.

```math
\mathbf{z}^I_i = \frac{\mathbf{W}_I\, f_I(\mathbf{x}_i)}{\lVert \mathbf{W}_I\, f_I(\mathbf{x}_i)\rVert_2},
\qquad
\mathbf{z}^T_i = \frac{\mathbf{W}_T\, f_T(\mathbf{t}_i)}{\lVert \mathbf{W}_T\, f_T(\mathbf{t}_i)\rVert_2}.
```

## 1. The contrastive objective

For a batch of $N$ pairs, form the scaled cosine-similarity matrix

```math
\mathbf{S}_{ij} = \frac{1}{\tau}\, \mathbf{z}^I_i \cdot \mathbf{z}^T_j .
```

The correct match is always the diagonal, so the loss is symmetric
cross-entropy against the identity permutation:

```math
\boxed{\;
\mathcal{L} = \frac{1}{2N}\sum_{i=1}^{N}
\left[
-\log \frac{e^{\mathbf{S}_{ii}}}{\sum_{j=1}^{N} e^{\mathbf{S}_{ij}}}
-\log \frac{e^{\mathbf{S}_{ii}}}{\sum_{j=1}^{N} e^{\mathbf{S}_{ji}}}
\right]
\;}
```

The first term is image→text retrieval, the second text→image; averaging keeps
neither modality privileged. Each is InfoNCE [9] with one positive and $N-1$
**in-batch negatives** — negatives are free, and *batch size is the negative
count*.

Two sanity values follow immediately and are asserted in the test suite: at
initialization $\mathcal{L} = \log N$, and for perfect matching
$\mathcal{L} \to 0$.

## 2. Temperature and the mutual-information bound

$\ell_2$-normalization confines similarities to $[-1, 1]$, removing vector
magnitude as a degenerate axis the model would otherwise inflate. But it also
compresses all logits into a range where softmax is nearly uniform. The
temperature $\tau$ rescales them; CLIP *learns* it, parameterizing

```math
s = \log(1/\tau), \qquad 1/\tau = e^{s}, \qquad e^{s} \le 100 ,
```

so that $1/\tau$ stays positive under unconstrained gradient descent. The clamp
matters: an unbounded temperature sharpens without limit and is a documented
divergence mode. `--dummy` training shows $1/\tau$ starting at $1/0.07 = 14.3$
as in the paper.

The InfoNCE objective lower-bounds the mutual information between the two views,

```math
I(\mathbf{z}^I; \mathbf{z}^T) \;\ge\; \log N - \mathcal{L},
```

which is the formal reason CLIP needs enormous batches: **the bound itself is
capped at $\log N$.** OpenAI used $N = 32{,}768$ across 256 GPUs.

> **Practical consequence.** Gradient accumulation does *not* substitute for
> batch size here: it enlarges the gradient but each micro-batch still sees only
> its own negatives. The real remedies are multi-device training with an
> all-gather over embeddings before the loss, or a MoCo-style memory queue.
> Treat single-GPU runs as mechanism demonstrations, not accuracy reproductions.

## 3. Zero-shot transfer

Because the text tower embeds *arbitrary strings*, a trained CLIP classifies
without a classifier. Embed a prompt per class and take the nearest:

```math
\hat{y} = \arg\max_c \; \mathbf{z}^I \cdot \mathbf{w}_c,
\qquad
\mathbf{w}_c = \frac{\sum_{k} \mathbf{z}^T(\text{prompt}_k(c))}{\lVert \sum_{k} \mathbf{z}^T(\text{prompt}_k(c)) \rVert_2}.
```

The text encoder has synthesized the weights of a linear classifier from
language alone. Averaging several phrasings per class before renormalizing —
*prompt ensembling* — reduces phrasing variance and is worth a few points of
accuracy at no training cost. See `CLIP.build_zeroshot_classifier`.

**Text tower details.** It is architecturally the same Transformer, with two
differences: 1D positional embeddings, and a causal mask $\mathbf{M}_{ij} = 0$
for $j \le i$ and $-\infty$ otherwise. Causality is not required by the
contrastive loss — CLIP uses it so the same weights can double as a language
model, which CoCa and BLIP exploit by adding a captioning term. It also
provides one canonical read-out position: under causal masking the final
`[EOT]` token is the only one whose receptive field spans the whole caption.

---

# Part III — The training recipe

A ViT trained with a CNN recipe underperforms badly. Most of the reported gain
between ViT (2020) and DeiT (2021) [13] came from the recipe, not the
architecture — so it is written out here rather than buried.

**Why ViTs need this.** A CNN hard-codes locality and translation equivariance.
A ViT hard-codes almost nothing: any patch may attend to any other from layer 1,
so it must *learn* that neighboring patches are related. That flexibility is
exactly why ViTs scale better on large corpora and overfit worse on small ones.
Every item below injects, through data and regularization, a prior the
architecture lacks.

### Optimizer

**AdamW** [14], not Adam. Adam's L2 penalty enters through the gradient and is
then divided by $\sqrt{\hat{v}}$, so frequently-updated weights get decayed
*less* — the coupling makes weight decay behave unpredictably. AdamW decouples
it:

```math
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right).
```

**Parameter groups matter.** Decay is applied only to tensors with
$\text{ndim} \ge 2$. Decaying a LayerNorm gain, a bias, a positional embedding,
the `[CLS]` token, a LayerScale $\gamma$ or CLIP's temperature does not
regularize — it erases a coordinate or silences a branch. The rule is *decay
matrices, never vectors*; skipping it typically costs 0.5–1% top-1
(`param_groups_weight_decay`, tested in `test_no_weight_decay_covers_1d_params`).

### Learning-rate schedule

```math
\eta_t =
\begin{cases}
\dfrac{t}{T_w}\,\eta_{\max}, & t < T_w \\[2ex]
\eta_{\min} + \dfrac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\dfrac{\pi (t - T_w)}{T - T_w}\right), & t \ge T_w
\end{cases}
```

**Warmup is not optional for Transformers.** At initialization, predictions are
arbitrary and gradients are large and poorly correlated with any useful
direction; meanwhile Adam's second moment starts at $\mathbf{0}$ and needs
hundreds of steps to become a reliable normalizer. A full-size LR here
saturates the attention softmax, gradients vanish, and the run plateaus without
ever throwing an error. **Cosine decay** then decays slowly at both ends —
keeping exploration alive early, letting the iterate settle late — with the
fast drop in between, and adds no hyperparameters.

### Regularization and augmentation

| Technique | Default | Why |
|---|---|---|
| RandomResizedCrop | scale $(0.65, 1)$ | Teaches scale invariance the architecture lacks |
| RandAugment [15] | $m = 9$, 2 ops | Photometric/geometric diversity |
| Mixup [16] | $\alpha = 0.8$ | Linear interpolation of inputs *and* labels |
| CutMix [17] | $\alpha = 1.0$ | Regional pasting; label mixed by area |
| RandomErasing [18] | $p = 0.25$ | Forces distributed rather than local evidence |
| Label smoothing | $0.1$ | Caps target confidence, calibrates logits |
| Stochastic depth [5] | $0.1$, ramped | Regularizes depth itself |
| Weight decay | $0.05$ (CLIP: $0.2$) | On matrices only |
| Grad clipping | $\lVert g \rVert \le 1.0$ | Bounds rare gradient spikes |
| EMA | $\delta = 0.9998$ | Averages the SGD trajectory |

Mixup and CutMix both draw $\lambda \sim \text{Beta}(\alpha, \alpha)$ and pair
each sample with a reversed copy of the batch:

```math
\tilde{\mathbf{x}} = \lambda \mathbf{x}_a + (1-\lambda)\mathbf{x}_b,
\qquad
\tilde{y} = \lambda y_a + (1-\lambda) y_b .
```

For CutMix, $\lambda$ is recomputed from the *realized* box area after
clipping, so the label mix always matches the pixel evidence. Because the
targets become distributions, the criterion switches to
`SoftTargetCrossEntropy`.

**EMA** keeps $\theta' \leftarrow \delta\theta' + (1-\delta)\theta$. SGD at a
finite learning rate never settles into a minimum — it oscillates in a bowl
around one — so averaging the trajectory approximates the bowl's center, a
flatter and better-generalizing solution. It typically evaluates 0.1–0.5%
higher at zero training cost, and is what you should ship.

### Mixed precision

`bf16` is preferred over `fp16`: it has fp32's 8-bit exponent, so it needs no
loss scaling. With `fp16` (5 exponent bits), small gradients underflow to zero,
and `torch.amp.GradScaler` multiplies the loss by a large factor before backward
and divides it out before the step — note that gradients must be **unscaled
before clipping**, or you would clip the scaled values. The script selects
automatically and falls back if the GPU lacks bf16 support.

In `train_clip.py` the similarity matrix is cast back to fp32 before the loss:
a softmax over cosines scaled by up to $1/\tau = 100$ is numerically delicate.

### Verifying a run

The zero-initialized classifier head gives an exact, checkable starting point:

```math
\mathcal{L}_0 = \log C \quad (\text{2.303 for CIFAR-10}),
\qquad
\mathcal{L}^\text{CLIP}_0 = \log N .
```

If your first loss is not that number, something is wrong before training even
begins. The `--dummy` path is the second check: random labels are unlearnable
in general, so training loss falling well below $\log C$ on it confirms the
optimizer and gradient flow are wired correctly (the model is memorizing, which
is the point). Running the synthetic CLIP task for 6 epochs takes seconds and
should drive batch retrieval accuracy from 0% to the theoretical ceiling of
$1/4$ — 8 distinct concepts spread over 32 batch slots means 4 equally valid
matches per row.

---

## Model presets

Encoder-only (no head), at 224px with $P = 16$. Widths follow [1]; head
dimension is held at $d = 64$ throughout, the sweet spot for tensor-core
matmuls.

| Preset | $D$ | $L$ | $h$ | Params | GMACs |
|---|---|---|---|---|---|
| `vit_tiny` | 192 | 12 | 3 | 5.5M | 1.25 |
| `vit_small` | 384 | 12 | 6 | 21.7M | 4.60 |
| `vit_base` | 768 | 12 | 12 | 85.8M | 17.56 |
| `vit_large` | 1024 | 24 | 16 | 303.3M | 61.55 |
| `vit_huge` | 1280 | 32 | 16 | 630.9M | 127.31 |

CLIP presets (vocabulary 30k, 224px, $P=16$; both towers plus projections):

| Preset | Vision | Text | Joint dim | Params |
|---|---|---|---|---|
| `clip_tiny` | $D{=}192$, $L{=}6$ | $D{=}192$, $L{=}4$ | 192 | 10.5M |
| `clip_small` | $D{=}384$, $L{=}12$ | $D{=}384$, $L{=}6$ | 384 | 44.2M |
| `clip_base` | $D{=}768$, $L{=}12$ | $D{=}512$, $L{=}12$ | 512 | 139.7M |

For small images, override the patch size — CIFAR at 32px with $P=4$ gives
64 tokens, a sensible sequence length:

```bash
python train.py --dataset cifar10 --model vit_tiny --img-size 32 --patch-size 4
```

## Testing

```bash
pytest -q                 # 27 tests, ~1.5s on CPU
```

The suite targets the bugs that do not raise. Transformer code fails *silently*
— a transpose that mismatches heads and channels, positional embeddings
misaligned with the patch raster, a causal mask that leaks the future — and
each one shows up only as a slightly worse final number. Pinned properties
include: the Conv2d patch embedding equals explicit unfold + Linear; attention
rows are probability distributions; the fused and reference attention paths
agree; attention is permutation-equivariant without positional embeddings;
sin–cos positions are mutually distinguishable; interpolation preserves prefix
tokens; the text encoder cannot see the future; the InfoNCE loss equals
$\log N$ at chance and $0$ at perfect matching; and gradients reach every
parameter of both towers.

## Extending this code

- **Masked autoencoding (MAE)** [19]: drop 75% of patch tokens after
  `_pos_embed`, encode the visible ones, and attach a shallow decoder that
  reconstructs pixels. The encoder here already returns full token sequences,
  which is all that is required.
- **Self-distillation (DINO)** [20]: run two augmented views, add a projection
  head, and train a student against a momentum-EMA teacher — `ModelEma` is
  already implemented.
- **Dense prediction**: use `get_intermediate_layers(..., reshape=True)` to get
  $(B, D, g_h, g_w)$ feature maps from several depths and feed a decoder.
- **Multi-GPU CLIP**: wrap in `DistributedDataParallel` and all-gather
  embeddings before the loss, so negatives scale with the world size rather
  than the per-device batch.
- **Better tokenization**: replace `SimpleTokenizer` with a real BPE
  (`open_clip.get_tokenizer`, or HuggingFace `CLIPTokenizer`). `encode_text`
  only needs a `(B, context_length)` integer tensor, so nothing else changes.

## References

1. Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
2. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
3. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
4. Xiong et al. *On Layer Normalization in the Transformer Architecture.* ICML 2020. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
5. Huang et al. *Deep Networks with Stochastic Depth.* ECCV 2016. [arXiv:1603.09382](https://arxiv.org/abs/1603.09382)
6. Touvron et al. *Going Deeper with Image Transformers (CaiT).* ICCV 2021. [arXiv:2103.17239](https://arxiv.org/abs/2103.17239)
7. Darcet et al. *Vision Transformers Need Registers.* ICLR 2024. [arXiv:2309.16588](https://arxiv.org/abs/2309.16588)
8. Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
9. van den Oord et al. *Representation Learning with Contrastive Predictive Coding.* 2018. [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
10. Dehghani et al. *Scaling Vision Transformers to 22 Billion Parameters.* ICML 2023. [arXiv:2302.05442](https://arxiv.org/abs/2302.05442)
11. Hendrycks & Gimpel. *Gaussian Error Linear Units (GELUs).* 2016. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)
12. Geva et al. *Transformer Feed-Forward Layers Are Key-Value Memories.* EMNLP 2021. [arXiv:2012.14913](https://arxiv.org/abs/2012.14913)
13. Touvron et al. *Training data-efficient image transformers & distillation through attention (DeiT).* ICML 2021. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
14. Loshchilov & Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
15. Cubuk et al. *RandAugment: Practical automated data augmentation.* NeurIPS 2020. [arXiv:1909.13719](https://arxiv.org/abs/1909.13719)
16. Zhang et al. *mixup: Beyond Empirical Risk Minimization.* ICLR 2018. [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
17. Yun et al. *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.* ICCV 2019. [arXiv:1905.04899](https://arxiv.org/abs/1905.04899)
18. Zhong et al. *Random Erasing Data Augmentation.* AAAI 2020. [arXiv:1708.04896](https://arxiv.org/abs/1708.04896)
19. He et al. *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
20. Caron et al. *Emerging Properties in Self-Supervised Vision Transformers (DINO).* ICCV 2021. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)

## License

Released under the [MIT License](LICENSE).
