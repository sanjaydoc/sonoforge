"""Sequence variation operators for the discrete optimizers."""

from __future__ import annotations

import random

from sonoforge.data.types import AA_ALPHABET, Candidate


def mutate(sequence: str, rate: float, rng: random.Random) -> str:
    """Independently substitute each residue with probability ``rate``."""
    chars = list(sequence)
    for i in range(len(chars)):
        if rng.random() < rate:
            chars[i] = rng.choice(AA_ALPHABET)
    return "".join(chars)


def crossover(a: str, b: str, rng: random.Random) -> str:
    """Single-point crossover within the shorter length."""
    n = min(len(a), len(b))
    if n < 2:
        return a
    pt = rng.randint(1, n - 1)
    return a[:pt] + b[pt:n]


def make_child(parents: list[Candidate], cycle: int, rate: float, rng: random.Random) -> Candidate:
    """Produce one child candidate from 1–2 parents via crossover + mutation."""
    if len(parents) >= 2 and rng.random() < 0.5:
        seq = crossover(parents[0].sequence, parents[1].sequence, rng)
        parent_seqs = [parents[0].sequence, parents[1].sequence]
    else:
        seq = parents[0].sequence
        parent_seqs = [parents[0].sequence]
    seq = mutate(seq, rate, rng)
    return Candidate(sequence=seq, cycle=cycle, parents=parent_seqs)
