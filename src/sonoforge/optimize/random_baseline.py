"""Random-search baseline: mutate random archive members (no selection pressure)."""

from __future__ import annotations

import random

from sonoforge.data.types import Candidate
from sonoforge.optimize.base import Evaluated
from sonoforge.optimize.variation import make_child


class RandomProposer:
    name = "random"

    def __init__(self, mutation_rate: float = 0.1) -> None:
        self.mutation_rate = mutation_rate

    def propose(self, evaluated: list[Evaluated], k: int, cycle: int,
                rng: random.Random) -> list[Candidate]:
        if not evaluated:
            return []
        children: list[Candidate] = []
        for _ in range(k):
            parent = rng.choice(evaluated).candidate
            children.append(make_child([parent], cycle, self.mutation_rate, rng))
        return children
