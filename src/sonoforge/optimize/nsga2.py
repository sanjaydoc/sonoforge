"""NSGA-II multi-objective optimizer over discrete sequences.

Fast non-dominated sort + crowding distance with **constrained domination**
(feasible solutions dominate infeasible ones; among infeasible, smaller
constraint violation wins). Parents are chosen by binary tournament on
(rank, crowding) and recombined with the sequence variation operators.
"""

from __future__ import annotations

import random

import numpy as np

from sonoforge.data.types import Candidate
from sonoforge.optimize.base import Evaluated
from sonoforge.optimize.variation import make_child


def _constrained_dominates(a: Evaluated, b: Evaluated) -> bool:
    if a.feasible and not b.feasible:
        return True
    if b.feasible and not a.feasible:
        return False
    if not a.feasible and not b.feasible:
        return a.violation < b.violation
    x, y = a.objectives, b.objectives
    return bool(np.all(x >= y) and np.any(x > y))


def fast_non_dominated_sort(pop: list[Evaluated]) -> list[list[int]]:
    """Return a list of fronts; each front is a list of indices into ``pop``."""
    n = len(pop)
    dominated: list[list[int]] = [[] for _ in range(n)]
    count = [0] * n
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _constrained_dominates(pop[p], pop[q]):
                dominated[p].append(q)
            elif _constrained_dominates(pop[q], pop[p]):
                count[p] += 1
        if count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in dominated[p]:
                count[q] -= 1
                if count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(pop: list[Evaluated], front: list[int]) -> dict[int, float]:
    """Crowding distance for indices in a single front."""
    dist = dict.fromkeys(front, 0.0)
    if len(front) <= 2:
        return dict.fromkeys(front, float("inf"))
    objs = np.array([pop[i].objectives for i in front])
    m = objs.shape[1]
    for k in range(m):
        order = np.argsort(objs[:, k])
        lo, hi = objs[order[0], k], objs[order[-1], k]
        dist[front[order[0]]] = float("inf")
        dist[front[order[-1]]] = float("inf")
        span = hi - lo or 1.0
        for r in range(1, len(front) - 1):
            i = front[order[r]]
            dist[i] += float((objs[order[r + 1], k] - objs[order[r - 1], k]) / span)
    return dist


def rank_population(pop: list[Evaluated]) -> tuple[dict[int, int], dict[int, float]]:
    """Return (rank per index, crowding per index)."""
    rank: dict[int, int] = {}
    crowd: dict[int, float] = {}
    for r, front in enumerate(fast_non_dominated_sort(pop)):
        cd = crowding_distance(pop, front)
        for i in front:
            rank[i] = r
            crowd[i] = cd[i]
    return rank, crowd


class NSGA2Proposer:
    name = "nsga2"

    def __init__(self, mutation_rate: float = 0.05) -> None:
        self.mutation_rate = mutation_rate

    def propose(self, evaluated: list[Evaluated], k: int, cycle: int,
                rng: random.Random) -> list[Candidate]:
        if not evaluated:
            return []
        rank, crowd = rank_population(evaluated)

        def tournament() -> Evaluated:
            i, j = rng.randrange(len(evaluated)), rng.randrange(len(evaluated))
            if (rank[i], -crowd[i]) <= (rank[j], -crowd[j]):
                return evaluated[i]
            return evaluated[j]

        children: list[Candidate] = []
        for _ in range(k):
            parents = [tournament().candidate, tournament().candidate]
            children.append(make_child(parents, cycle, self.mutation_rate, rng))
        return children
