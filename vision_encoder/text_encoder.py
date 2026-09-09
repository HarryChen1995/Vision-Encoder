"""Causal Transformer text encoder — CLIP's second tower.

STRUCTURE
---------
Architecturally this is the *same* Transformer as the vision tower; only the
tokenization and the masking differ.

    token ids (B, L)
      |  token embedding lookup: a learned vector per vocabulary entry
      v
    (B, L, W)
      |  + learned positional embedding (1D: text is a line, not a grid)
      v
    [ Block x L_txt ]   with a CAUSAL attention mask
      v
    final LayerNorm
      |  read out the position of the [EOT] token
      v
    (B, W)  -> linear projection -> shared image-text space

WHY CAUSAL MASKING
------------------
Position i may attend only to positions <= i. We enforce this by adding a mask
M with M[i,j] = 0 for j <= i and -inf otherwise to the pre-softmax scores:
exp(-inf) = 0, so forbidden pairs receive exactly zero weight.

CLIP's text tower is causal even though contrastive learning does not require
it. Two reasons: it lets the same weights double as an autoregressive language
model (which CoCa, BLIP and friends exploit by adding a captioning loss), and it
gives one canonical "the whole sentence has been read" position — the last
token. A bidirectional encoder would need a pooling choice instead.

WHY READ OUT AT [EOT]
---------------------
Under causal masking, the final token is the only position whose receptive field
covers the entire caption. Its hidden state is therefore the natural sentence
summary — the text-side analogue of the vision tower's [CLS] token.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .block import Block


def build_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Additive `(1, 1, L, L)` mask: 0 on/below the diagonal, -inf above."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask.triu_(diagonal=1)  # keep -inf strictly above the diagonal
    return mask.view(1, 1, seq_len, seq_len)


class TextTransformer(nn.Module):
    """Causal Transformer that maps token ids to a single embedding.

    Args:
        vocab_size: Size of the token vocabulary.
        context_length: Fixed sequence length `L` (77 in CLIP).
        width: Model width `W`.
        depth: Number of Transformer blocks.
        num_heads: Attention heads.
        mlp_ratio: MLP expansion factor.
        output_dim: Dimension of the shared image-text space. If None, no
            projection is applied and `width` is returned.
        causal: Apply the causal mask (True reproduces CLIP).
        pool_type: `"eot"` reads the [EOT] position; `"mean"` averages
            non-padding tokens (only sensible when `causal=False`).
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int = 77,
        width: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        output_dim: int | None = 512,
        drop_path_rate: float = 0.0,
        causal: bool = True,
        pool_type: str = "eot",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.width = width
        self.causal = causal
        self.pool_type = pool_type

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.zeros(1, context_length, width))

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            Block(dim=width, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=dpr[i])
            for i in range(depth)
        )
        self.ln_final = nn.LayerNorm(width)

        # Projection into the joint space. No bias: the space is only ever used
        # through cosine similarity, where a shared offset is meaningless.
        self.text_projection = (
            nn.Linear(width, output_dim, bias=False) if output_dim else nn.Identity()
        )
        self.output_dim = output_dim or width

        # The mask depends only on the sequence length, so build it once.
        # persistent=False keeps this derived constant out of the state_dict.
        self.register_buffer(
            "causal_mask", build_causal_mask(context_length), persistent=False
        )

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.apply(self._init_module)
        if isinstance(self.text_projection, nn.Linear):
            # Scaling by width^-0.5 keeps the projected vectors' norm ~O(1)
            # before normalization, which keeps the initial logits well-scaled.
            nn.init.normal_(self.text_projection.weight, std=self.width**-0.5)

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"positional_embedding", "token_embedding.weight"}

    def forward(self, text: torch.Tensor, eot_id: int | None = None) -> torch.Tensor:
        """`(B, L)` token ids -> `(B, output_dim)` sentence embeddings.

        `eot_id` locates the read-out position explicitly. Without it we fall
        back to `argmax`, which works because [EOT] is by construction the
        highest-numbered special token present... only when ids are laid out
        that way, so passing it is strongly preferred.
        """
        b, seq_len = text.shape
        if seq_len > self.context_length:
            raise ValueError(f"sequence length {seq_len} exceeds {self.context_length}")

        x = self.token_embedding(text)                     # (B, L, W)
        x = x + self.positional_embedding[:, :seq_len]

        mask = self.causal_mask[:, :, :seq_len, :seq_len] if self.causal else None
        for block in self.blocks:
            x = block(x, attn_mask=mask)

        x = self.ln_final(x)

        if self.pool_type == "eot":
            # Index of the [EOT] token in each row -> gather that hidden state.
            if eot_id is not None:
                is_eot = text == eot_id
                # `float().argmax()` returns the FIRST True, i.e. the real end
                # of the caption rather than any padding that follows.
                idx = is_eot.float().argmax(dim=-1)
                # Rows with no EOT (shouldn't happen) fall back to the last slot.
                idx = torch.where(is_eot.any(dim=-1), idx, torch.full_like(idx, seq_len - 1))
            else:
                idx = text.argmax(dim=-1)
            pooled = x[torch.arange(b, device=x.device), idx]
        else:
            pooled = x.mean(dim=1)

        return self.text_projection(pooled)
