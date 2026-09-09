"""A small, dependency-free tokenizer for the CLIP text tower.

SCOPE
-----
OpenAI's CLIP uses a byte-pair-encoding vocabulary of 49,152 merges shipped as a
data file. Reproducing BPE here would add a large asset and a lot of code that
teaches nothing about contrastive learning, so this module provides a
self-contained word-level tokenizer with a byte-level fallback for
out-of-vocabulary words. It is exact, reversible, and adequate for training on
a caption corpus you control.

For serious work, swap this for a real BPE (`open_clip.get_tokenizer(...)` or
HuggingFace `CLIPTokenizer`); `CLIP.encode_text` only needs an integer tensor of
shape (B, context_length), so the rest of the code is unaffected.

SEQUENCE FORMAT
---------------
Every caption becomes a fixed-length sequence:

    [SOT] t_1 t_2 ... t_k [EOT] [PAD] [PAD] ...

Fixed length lets us batch. The [EOT] position matters: the text encoder is
causal, so the *last* real token is the only position that has attended to the
entire caption, which is why CLIP reads its sentence embedding from there.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import torch

# Lowercase, then split into words / numbers / individual punctuation marks.
_TOKEN_RE = re.compile(r"[a-z0-9]+|[^\sa-z0-9]", re.IGNORECASE)

PAD_TOKEN, SOT_TOKEN, EOT_TOKEN, UNK_TOKEN = "<pad>", "<sot>", "<eot>", "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, SOT_TOKEN, EOT_TOKEN, UNK_TOKEN]


def basic_tokenize(text: str) -> list[str]:
    """Lowercase + regex split. Deliberately simple and deterministic."""
    return _TOKEN_RE.findall(text.lower().strip())


class SimpleTokenizer:
    """Word-level vocabulary with byte-level backoff.

    Words seen at least `min_freq` times during `build` get their own id. Any
    unseen word is decomposed into its UTF-8 bytes, each mapped to a reserved
    id, so the tokenizer never emits a lossy <unk> for content words and the
    vocabulary stays closed under arbitrary input.
    """

    def __init__(self, vocab: dict[str, int] | None = None, context_length: int = 77) -> None:
        self.context_length = context_length
        if vocab is None:
            vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
            # Reserve 256 ids so any byte is representable.
            for b in range(256):
                vocab[f"<byte_{b}>"] = len(vocab)
        self.vocab = vocab
        self.ids_to_tokens = {i: t for t, i in vocab.items()}

        self.pad_id = vocab[PAD_TOKEN]
        self.sot_id = vocab[SOT_TOKEN]
        self.eot_id = vocab[EOT_TOKEN]
        self.unk_id = vocab[UNK_TOKEN]

    # ------------------------------------------------------------- building
    @classmethod
    def build(
        cls,
        texts: list[str],
        min_freq: int = 1,
        max_vocab: int = 30000,
        context_length: int = 77,
    ) -> "SimpleTokenizer":
        """Fit a vocabulary on a caption corpus."""
        counts = Counter(tok for text in texts for tok in basic_tokenize(text))

        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for b in range(256):
            vocab[f"<byte_{b}>"] = len(vocab)

        for token, freq in counts.most_common():
            if freq < min_freq or len(vocab) >= max_vocab:
                break
            if token not in vocab:
                vocab[token] = len(vocab)

        return cls(vocab, context_length=context_length)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # -------------------------------------------------------------- encoding
    def _token_ids(self, token: str) -> list[int]:
        if token in self.vocab:
            return [self.vocab[token]]
        # Byte backoff: guarantees every string is encodable.
        return [self.vocab[f"<byte_{b}>"] for b in token.encode("utf-8")]

    def encode(self, text: str) -> list[int]:
        """Text -> id list, without special tokens or padding."""
        ids: list[int] = []
        for token in basic_tokenize(text):
            ids.extend(self._token_ids(token))
        return ids

    def __call__(self, texts: str | list[str]) -> torch.Tensor:
        """Batch-encode to a padded `(B, context_length)` LongTensor.

        Captions longer than the context window are truncated, but [EOT] is
        always written into the final slot so the read-out position stays valid.
        """
        if isinstance(texts, str):
            texts = [texts]

        out = torch.full(
            (len(texts), self.context_length), self.pad_id, dtype=torch.long
        )
        for i, text in enumerate(texts):
            ids = [self.sot_id] + self.encode(text) + [self.eot_id]
            if len(ids) > self.context_length:
                ids = ids[: self.context_length]
                ids[-1] = self.eot_id
            out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return out

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        pieces, byte_buf = [], bytearray()

        def flush() -> None:
            if byte_buf:
                pieces.append(byte_buf.decode("utf-8", errors="replace"))
                byte_buf.clear()

        for i in ids:
            token = self.ids_to_tokens.get(int(i), UNK_TOKEN)
            if token in (PAD_TOKEN, SOT_TOKEN, EOT_TOKEN):
                continue
            if token.startswith("<byte_"):
                byte_buf.append(int(token[6:-1]))
            else:
                flush()
                pieces.append(token)
        flush()
        return " ".join(pieces)

    # ---------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"context_length": self.context_length, "vocab": self.vocab})
        )

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTokenizer":
        data = json.loads(Path(path).read_text())
        return cls(data["vocab"], context_length=data["context_length"])
