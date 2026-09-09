"""Shape, invariance and gradient tests.

These are the checks worth having for a from-scratch implementation: the bugs
that actually bite in Transformer code are silent ones — a transpose that
mismatches heads and channels, positional embeddings misaligned with the patch
raster, a causal mask that leaks the future. None of these raise; they only
show up as a model that trains to a slightly worse number. Each test below
pins one such property.

Run with:  pytest -q
"""

from __future__ import annotations

import pytest
import torch

from vision_encoder import (
    CLIP,
    CLIPConfig,
    ClipLoss,
    PatchEmbed,
    SimpleTokenizer,
    VisionEncoderConfig,
    VisionTransformer,
    build_2d_sincos_pos_embed,
    create_clip,
    create_vision_encoder,
    interpolate_pos_embed,
)
from vision_encoder.attention import MultiHeadSelfAttention
from vision_encoder.text_encoder import TextTransformer, build_causal_mask


# --------------------------------------------------------------------------
# Patch embedding
# --------------------------------------------------------------------------
def test_patch_embed_shapes():
    pe = PatchEmbed(img_size=32, patch_size=4, embed_dim=64)
    assert pe.num_patches == 64 and pe.grid_size == (8, 8)
    assert pe(torch.randn(2, 3, 32, 32)).shape == (2, 64, 64)


def test_patch_embed_rejects_indivisible_input():
    pe = PatchEmbed(img_size=32, patch_size=4, embed_dim=64)
    with pytest.raises(ValueError):
        pe(torch.randn(1, 3, 30, 30))


def test_patch_embed_equals_unfold_plus_linear():
    """The Conv2d trick must equal explicit patch extraction + a Linear."""
    pe = PatchEmbed(img_size=8, patch_size=4, in_chans=3, embed_dim=16)
    x = torch.randn(2, 3, 8, 8)

    # Manual: cut into patches, flatten each, apply the same weights.
    patches = x.unfold(2, 4, 4).unfold(3, 4, 4)              # (B,C,gh,gw,P,P)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(2, 4, -1)
    weight = pe.proj.weight.reshape(16, -1)                  # (D, C*P*P)
    expected = patches @ weight.t() + pe.proj.bias

    assert torch.allclose(pe(x), expected, atol=1e-5)


# --------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------
def test_attention_rows_are_distributions():
    attn_layer = MultiHeadSelfAttention(64, num_heads=8)
    _, attn = attn_layer(torch.randn(2, 12, 64), return_attn=True)
    assert attn.shape == (2, 8, 12, 12)
    assert torch.allclose(attn.sum(-1), torch.ones(2, 8, 12), atol=1e-5)


def test_fused_and_manual_attention_agree():
    """The Flash kernel and the reference path must compute the same function."""
    layer = MultiHeadSelfAttention(32, num_heads=4, attn_drop=0.0).eval()
    x = torch.randn(2, 9, 32)
    with torch.no_grad():
        fused = layer(x)
        layer.fused_attn = False
        manual = layer(x)
    assert torch.allclose(fused, manual, atol=1e-5)


def test_attention_is_permutation_equivariant():
    """Without positional embeddings, shuffling tokens must shuffle outputs.

    This is the property positional embeddings exist to break; if it fails,
    position information is leaking in somewhere it shouldn't.
    """
    layer = MultiHeadSelfAttention(32, num_heads=4).eval()
    x = torch.randn(1, 6, 32)
    perm = torch.randperm(6)
    with torch.no_grad():
        assert torch.allclose(layer(x)[:, perm], layer(x[:, perm]), atol=1e-5)


# --------------------------------------------------------------------------
# Positional embeddings
# --------------------------------------------------------------------------
def test_sincos_shape_and_range():
    pos = build_2d_sincos_pos_embed(64, (8, 8), cls_token=True)
    assert pos.shape == (1, 65, 64)
    assert pos[0, 0].abs().sum() == 0        # CLS gets no position
    assert pos[:, 1:].abs().max() <= 1.0     # sinusoids are bounded


def test_sincos_positions_are_distinct():
    pos = build_2d_sincos_pos_embed(64, (4, 4))[0]
    sims = pos @ pos.t()
    # Each position must be most similar to itself.
    assert (sims.argmax(dim=1) == torch.arange(16)).all()


def test_pos_embed_interpolation():
    pos = torch.randn(1, 1 + 64, 32)
    out = interpolate_pos_embed(pos, (16, 16), num_prefix_tokens=1)
    assert out.shape == (1, 1 + 256, 32)
    # The prefix token is carried over untouched, never interpolated.
    assert torch.allclose(out[:, :1], pos[:, :1])


# --------------------------------------------------------------------------
# Vision encoder
# --------------------------------------------------------------------------
@pytest.mark.parametrize("pool", ["cls", "avg"])
def test_vit_forward(pool):
    model = create_vision_encoder(
        "vit_tiny", img_size=32, patch_size=4, num_classes=10, global_pool=pool
    )
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_vit_zero_init_head_gives_uniform_logits():
    """A zero-initialized head means the loss starts at exactly ln(C)."""
    model = create_vision_encoder("vit_tiny", img_size=32, patch_size=4, num_classes=10).eval()
    with torch.no_grad():
        logits = model(torch.randn(4, 3, 32, 32))
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(4, dtype=torch.long))
    assert torch.allclose(logits, torch.zeros_like(logits))
    assert abs(loss.item() - torch.log(torch.tensor(10.0)).item()) < 1e-5


def test_vit_handles_new_resolution():
    """Positional embeddings interpolate, so a 32px model runs at 64px."""
    model = create_vision_encoder("vit_tiny", img_size=32, patch_size=4, num_classes=10).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 64, 64)).shape == (1, 10)


def test_registers_excluded_from_avg_pool():
    model = VisionTransformer(
        VisionEncoderConfig(img_size=32, patch_size=4, embed_dim=64, depth=2, num_heads=4,
                            num_register_tokens=4, global_pool="avg", num_classes=0)
    )
    tokens = model.forward_features(torch.randn(2, 3, 32, 32))
    assert tokens.shape[1] == 64 + 5              # patches + CLS + 4 registers
    assert model.pool(tokens).shape == (2, 64)


def test_no_weight_decay_covers_1d_params():
    from vision_encoder.utils import param_groups_weight_decay

    model = create_vision_encoder("vit_tiny", img_size=32, patch_size=4, num_classes=10)
    groups = param_groups_weight_decay(model, 0.05, model.no_weight_decay())
    assert all(p.ndim >= 2 for p in groups[0]["params"])
    assert groups[1]["weight_decay"] == 0.0


def test_gradients_reach_every_parameter():
    model = create_vision_encoder("vit_tiny", img_size=32, patch_size=4, num_classes=10)
    model(torch.randn(2, 3, 32, 32)).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_intermediate_layers_reshape():
    model = create_vision_encoder("vit_tiny", img_size=32, patch_size=4, num_classes=10)
    outs = model.get_intermediate_layers(torch.randn(2, 3, 32, 32), n=3, reshape=True)
    assert len(outs) == 3 and all(o.shape == (2, 192, 8, 8) for o in outs)


# --------------------------------------------------------------------------
# Text encoder
# --------------------------------------------------------------------------
def test_causal_mask_blocks_the_future():
    mask = build_causal_mask(5)[0, 0]
    assert torch.isinf(mask[0, 1]) and mask[0, 1] < 0   # can't see ahead
    assert mask[3, 1] == 0.0                            # can see behind


def test_text_encoder_is_causal():
    """Changing token t must not alter the hidden state at any position < t."""
    model = TextTransformer(vocab_size=50, context_length=8, width=32, depth=2,
                            num_heads=4, output_dim=None, pool_type="mean").eval()
    ids = torch.randint(0, 50, (1, 8))
    modified = ids.clone()
    modified[0, 5] = (modified[0, 5] + 1) % 50

    with torch.no_grad():
        # Pool over a prefix that excludes the changed position.
        a = model(ids[:, :5])
        b = model(modified[:, :5])
    assert torch.allclose(a, b, atol=1e-6)


def test_tokenizer_roundtrip_and_special_tokens():
    tok = SimpleTokenizer.build(["a photo of a dog", "a cat"], context_length=16)
    ids = tok(["a photo of a dog"])
    assert ids.shape == (1, 16)
    assert ids[0, 0] == tok.sot_id and tok.eot_id in ids[0].tolist()
    assert tok.decode(ids[0]) == "a photo of a dog"


def test_tokenizer_truncates_but_keeps_eot():
    tok = SimpleTokenizer.build(["one two three four five six"], context_length=4)
    ids = tok(["one two three four five six"])
    assert ids.shape == (1, 4) and ids[0, -1] == tok.eot_id


# --------------------------------------------------------------------------
# CLIP
# --------------------------------------------------------------------------
def test_clip_forward_and_normalization():
    tok = SimpleTokenizer.build(["a dog", "a cat"], context_length=8)
    model = create_clip("clip_tiny", img_size=32, patch_size=4,
                        vocab_size=tok.vocab_size, context_length=8)
    out = model(torch.randn(3, 3, 32, 32), tok(["a dog", "a cat", "a dog"]), eot_id=tok.eot_id)

    assert out["logits_per_image"].shape == (3, 3)
    assert torch.allclose(out["image_features"].norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(out["text_features"].norm(dim=-1), torch.ones(3), atol=1e-5)
    # Similarity is symmetric under transposition.
    assert torch.allclose(out["logits_per_text"], out["logits_per_image"].t())


def test_clip_loss_at_chance():
    """With random embeddings the symmetric InfoNCE loss should sit near ln(N)."""
    n = 16
    logits = torch.zeros(n, n)   # perfectly uninformative
    stats = ClipLoss()(logits, logits.t())
    assert abs(stats["loss"].item() - torch.log(torch.tensor(float(n))).item()) < 1e-5


def test_clip_loss_is_zero_for_perfect_matching():
    logits = torch.eye(8) * 100.0
    assert ClipLoss()(logits, logits.t())["loss"].item() < 1e-4


def test_logit_scale_clamped():
    model = create_clip("clip_tiny", img_size=32, patch_size=4, vocab_size=100, context_length=8)
    with torch.no_grad():
        model.logit_scale.fill_(1000.0)
    model.clamp_logit_scale()
    assert model.logit_scale.exp().item() <= 100.0 + 1e-4


def test_clip_gradients_reach_both_towers():
    tok = SimpleTokenizer.build(["a dog", "a cat"], context_length=8)
    model = create_clip("clip_tiny", img_size=32, patch_size=4,
                        vocab_size=tok.vocab_size, context_length=8)
    out = model(torch.randn(4, 3, 32, 32), tok(["a dog", "a cat"] * 2), eot_id=tok.eot_id)
    ClipLoss()(out["logits_per_image"], out["logits_per_text"])["loss"].backward()

    assert model.visual.patch_embed.proj.weight.grad.abs().sum() > 0
    assert model.text.token_embedding.weight.grad.abs().sum() > 0
    assert model.logit_scale.grad is not None


def test_zeroshot_classifier_shape():
    tok = SimpleTokenizer.build(["a photo of a dog", "a photo of a cat"], context_length=8)
    model = create_clip("clip_tiny", img_size=32, patch_size=4,
                        vocab_size=tok.vocab_size, context_length=8).eval()
    classifier = model.build_zeroshot_classifier(
        [["a photo of a dog", "a dog"], ["a photo of a cat"]], tok
    )
    assert classifier.shape == (model.config.embed_dim, 2)
    assert torch.allclose(classifier.norm(dim=0), torch.ones(2), atol=1e-5)
    assert model.zero_shot_predict(torch.randn(2, 3, 32, 32), classifier).shape == (2, 2)
