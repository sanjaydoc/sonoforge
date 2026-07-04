"""Unified scoring/embedding interface for the SonoForge PLM.

Two interchangeable backends behind one interface:

- :class:`ProfileScorer` — a numpy unigram model + interpretable features. No
  torch required; used as the default so the DBTL loop and CI run anywhere.
- :class:`SSMScorer` — wraps a trained :class:`~sonoforge.plm.model.ProteinSSM`
  for pseudo-log-likelihood scoring and learned embeddings when torch is present.

Both expose ``embed`` and ``pseudo_log_likelihood`` so downstream code (Bayesian
optimization, ranking) is agnostic to which one is active.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from sonoforge.data.featurize import SequenceFeaturizer
from sonoforge.data.types import AA_ALPHABET
from sonoforge.plm.tokenizer import ProteinTokenizer


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class ProfileScorer:
    """Numpy unigram language model + descriptor embeddings (torch-free fallback)."""

    def __init__(self) -> None:
        self._featurizer = SequenceFeaturizer()
        self._logfreq = np.log(np.full(len(AA_ALPHABET), 1.0 / len(AA_ALPHABET)))
        self._idx = {aa: i for i, aa in enumerate(AA_ALPHABET)}
        self.fitted = False

    def fit(self, sequences: list[str]) -> ProfileScorer:
        counts = np.ones(len(AA_ALPHABET))  # Laplace smoothing
        for s in sequences:
            for aa in s.upper():
                j = self._idx.get(aa)
                if j is not None:
                    counts[j] += 1.0
        self._logfreq = np.log(counts / counts.sum())
        self.fitted = True
        return self

    def pseudo_log_likelihood(self, sequence: str) -> float:
        seq = [aa for aa in sequence.upper() if aa in self._idx]
        if not seq:
            return -math.inf
        return float(np.mean([self._logfreq[self._idx[aa]] for aa in seq]))

    def embed(self, sequences: list[str]) -> np.ndarray:
        return self._featurizer.featurize_many(sequences)

    @property
    def embed_dim(self) -> int:
        return self._featurizer.n_features


class SSMScorer:
    """Trained-ProteinSSM backend (requires torch)."""

    def __init__(self, model, tokenizer: ProteinTokenizer | None = None) -> None:
        import torch

        self._torch = torch
        self.model = model.eval()
        self.tokenizer = tokenizer or ProteinTokenizer()

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> SSMScorer:
        import torch

        from sonoforge.plm.model import ProteinSSM

        ckpt = torch.load(path, map_location="cpu")
        tok = ProteinTokenizer()
        model = ProteinSSM(vocab_size=tok.vocab_size, pad_id=tok.pad_id, **ckpt.get("config", {}))
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, tok)

    def pseudo_log_likelihood(self, sequence: str) -> float:
        torch = self._torch
        tokens = torch.as_tensor(self.tokenizer.encode_batch([sequence]))
        with torch.no_grad():
            logits = self.model(tokens)["logits"][0]
            logprobs = torch.log_softmax(logits, dim=-1)
        ids = tokens[0]
        total, n = 0.0, 0
        for pos in range(ids.shape[0]):
            tid = int(ids[pos])
            if self.tokenizer.is_amino_acid(tid):
                total += float(logprobs[pos, tid])
                n += 1
        return total / n if n else -math.inf

    def embed(self, sequences: list[str]):
        torch = self._torch
        tokens = torch.as_tensor(self.tokenizer.encode_batch(sequences))
        with torch.no_grad():
            return self.model.pooled(tokens).cpu().numpy()


def make_scorer(sequences: list[str] | None = None, checkpoint: str | Path | None = None):
    """Return the best available scorer: trained SSM if possible, else ProfileScorer."""
    if checkpoint is not None and torch_available():
        return SSMScorer.from_checkpoint(checkpoint)
    scorer = ProfileScorer()
    if sequences:
        scorer.fit(sequences)
    return scorer
