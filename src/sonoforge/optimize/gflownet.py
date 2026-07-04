"""GFlowNet proposer — amortized, diversity-seeking sequence generation.

A GFlowNet learns a forward policy that samples objects with probability
*proportional to a reward*, rather than collapsing onto a single mode the way a
reward-greedy RL policy does — exactly what you want when proposing a diverse
library for the next DBTL round. Here the policy is a GRU that builds a sequence
left-to-right; it is trained with the **Trajectory Balance** objective against a
reward from a cheap surrogate fit to the current archive (features → scalarized
objective). Sampling the trained policy yields the proposed library.

Requires PyTorch (``sonoforge[ml]``); the loop falls back to NSGA-II without it.
"""

from __future__ import annotations

import random

import numpy as np
from sklearn.linear_model import Ridge

from sonoforge.data.featurize import SequenceFeaturizer
from sonoforge.data.types import AA_ALPHABET, Candidate
from sonoforge.optimize.base import Evaluated


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class _Surrogate:
    """Ridge model mapping sequence features -> scalarized (mean) objective in [0, 1]-ish."""

    def __init__(self) -> None:
        self._f = SequenceFeaturizer()
        self._model = Ridge(alpha=1.0)
        self._lo = 0.0
        self._hi = 1.0

    def fit(self, sequences: list[str], scores: np.ndarray) -> _Surrogate:
        x = self._f.featurize_many(sequences)
        self._model.fit(x, scores)
        self._lo, self._hi = float(scores.min()), float(scores.max())
        return self

    def log_reward(self, sequences: list[str], beta: float) -> np.ndarray:
        x = self._f.featurize_many(sequences)
        pred = self._model.predict(x)
        span = (self._hi - self._lo) or 1.0
        norm = np.clip((pred - self._lo) / span, 0.0, 1.5)   # ~[0, 1.5]
        return beta * norm                                    # log-reward (log R)


class GFlowNetProposer:
    name = "gflownet"

    def __init__(self, hidden: int = 64, train_steps: int = 80, batch: int = 16,
                 beta: float = 4.0, lr: float = 1e-3) -> None:
        self.hidden = hidden
        self.train_steps = train_steps
        self.batch = batch
        self.beta = beta
        self.lr = lr
        self.losses: list[float] = []

    # -- policy ------------------------------------------------------------
    def _build_policy(self, torch):
        from torch import nn

        vocab = len(AA_ALPHABET)

        class Policy(nn.Module):
            def __init__(self, hidden: int) -> None:
                super().__init__()
                self.embed = nn.Embedding(vocab + 1, hidden)  # +1 for BOS
                self.cell = nn.GRUCell(hidden, hidden)
                self.head = nn.Linear(hidden, vocab)
                self.logZ = nn.Parameter(torch.zeros(1))
                self.hidden = hidden
                self.bos = vocab

            def forward(self, batch: int, length: int, device):
                h = torch.zeros(batch, self.hidden, device=device)
                tok = torch.full((batch,), self.bos, dtype=torch.long, device=device)
                logpf = torch.zeros(batch, device=device)
                seqs = []
                for _ in range(length):
                    h = self.cell(self.embed(tok), h)
                    logits = torch.log_softmax(self.head(h), dim=-1)
                    dist = torch.distributions.Categorical(logits=logits)
                    tok = dist.sample()
                    logpf = logpf + dist.log_prob(tok)
                    seqs.append(tok)
                return torch.stack(seqs, dim=1), logpf  # (B, L), (B,)

        return Policy(self.hidden)

    def _decode(self, tokens) -> list[str]:
        return ["".join(AA_ALPHABET[int(t)] for t in row) for row in tokens]

    # -- proposer interface ------------------------------------------------
    def propose(self, evaluated: list[Evaluated], k: int, cycle: int,
                rng: random.Random) -> list[Candidate]:
        import torch

        if len(evaluated) < 4:
            # not enough to fit a surrogate; return random-length mutants of parents
            from sonoforge.optimize.variation import make_child

            return [make_child([rng.choice(evaluated).candidate], cycle, 0.1, rng)
                    for _ in range(k)]

        seqs = [e.candidate.sequence for e in evaluated]
        scores = np.array([float(np.mean(e.objectives)) for e in evaluated])
        surrogate = _Surrogate().fit(seqs, scores)
        length = len(rng.choice(evaluated).candidate.sequence)

        torch.manual_seed(cycle)
        policy = self._build_policy(torch)
        opt = torch.optim.Adam(policy.parameters(), lr=self.lr)
        device = torch.device("cpu")

        self.losses = []
        for _ in range(self.train_steps):
            tokens, logpf = policy(self.batch, length, device)
            batch_seqs = self._decode(tokens)
            log_r = torch.tensor(surrogate.log_reward(batch_seqs, self.beta), dtype=torch.float32)
            # Trajectory Balance: (logZ + sum logP_F - logR)^2
            loss = ((policy.logZ + logpf - log_r) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.losses.append(float(loss.detach()))

        with torch.no_grad():
            tokens, _ = policy(k, length, device)
        parents = [e.candidate.sequence for e in evaluated[:2]]
        return [Candidate(sequence=s, cycle=cycle, parents=parents) for s in self._decode(tokens)]
