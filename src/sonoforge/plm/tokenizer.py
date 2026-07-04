"""Amino-acid tokenizer for the SonoForge protein language model.

Vocabulary = 4 special tokens (PAD, BOS, EOS, MASK) + the 20 canonical amino
acids + UNK. Pure-python / numpy so it imports and runs without torch.
"""

from __future__ import annotations

import numpy as np

from sonoforge.data.types import AA_ALPHABET

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<mask>", "<unk>")


class ProteinTokenizer:
    """Maps amino-acid sequences to integer token ids and back."""

    def __init__(self) -> None:
        self.itos = list(SPECIAL_TOKENS) + list(AA_ALPHABET)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.mask_id = self.stoi["<mask>"]
        self.unk_id = self.stoi["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, sequence: str, *, add_special: bool = True) -> list[int]:
        ids = [self.stoi.get(aa, self.unk_id) for aa in sequence.upper()]
        if add_special:
            ids = [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        skip = {self.pad_id, self.bos_id, self.eos_id, self.mask_id}
        return "".join(self.itos[i] for i in ids if i not in skip)

    def encode_batch(self, sequences: list[str], *, max_len: int | None = None) -> np.ndarray:
        """Encode a batch to a padded (B, L) int array."""
        encoded = [self.encode(s) for s in sequences]
        length = max_len or (max((len(e) for e in encoded), default=0))
        out = np.full((len(encoded), length), self.pad_id, dtype=np.int64)
        for i, e in enumerate(encoded):
            e = e[:length]
            out[i, : len(e)] = e
        return out

    def is_amino_acid(self, token_id: int) -> bool:
        return token_id >= len(SPECIAL_TOKENS)
