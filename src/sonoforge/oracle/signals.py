"""Individual oracle signals for acoustic-reporter design (all in-silico proxies).

- **contrast**       ultrasound scattering proxy (maximize)
- **collapse_pressure**  MPa-scale mechanical proxy from physics (target a set-point)
- **expressibility** / **solubility**  mammalian-expression proxies (maximize)
- **immunogenicity** MHC-II epitope-load proxy (CONSTRAINT: keep below a ceiling)

Each is a deterministic function of sequence (and backbone, where available), so
results are reproducible. They are heuristics standing in for wet-lab assays and
are labelled as such throughout.
"""

from __future__ import annotations

import numpy as np

from sonoforge.data.types import AA_ALPHABET, Candidate
from sonoforge.physics import collapse_pressure as _collapse_pressure
from sonoforge.physics.elastic_network import _BETA_PROPENSITY

_KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
_CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}
# MHC-II hydrophobic anchor residues (P1 pocket favors large hydrophobics).
_MHCII_ANCHORS = frozenset("FILMVWY")


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _valid(seq: str) -> list[str]:
    return [a for a in seq.upper() if a in AA_ALPHABET]


def contrast(candidate: Candidate) -> float:
    """Scattering proxy: a rigid β-sheet shell with a hydrophobic gas-facing core
    scatters ultrasound more strongly. Combine β-propensity and hydrophobicity."""
    seq = _valid(candidate.sequence)
    if not seq:
        return 0.0
    beta = np.mean([_BETA_PROPENSITY[a] for a in seq]) / 1.7   # ~[0, 1]
    hydro = np.mean([_KD[a] for a in seq])                      # ~[-4.5, 4.5]
    hydro01 = (hydro + 4.5) / 9.0
    return _clip01(0.7 * beta + 0.3 * hydro01)


def collapse_pressure(candidate: Candidate) -> float:
    """MPa-scale collapse-pressure proxy (GNM if a backbone is present)."""
    return _collapse_pressure(candidate.sequence, candidate.backbone)


def expressibility(candidate: Candidate) -> float:
    """Expression proxy: penalize extreme hydrophobicity and very long sequences."""
    seq = _valid(candidate.sequence)
    if not seq:
        return 0.0
    hydro = np.mean([_KD[a] for a in seq])
    length_pen = max(0.0, (len(seq) - 250) / 250.0)            # penalize > 250 aa
    aromatic = np.mean([a in "FWY" for a in seq])
    return _clip01(0.9 - 0.08 * abs(hydro) - length_pen - 0.5 * max(0.0, aromatic - 0.15))


def solubility(candidate: Candidate) -> float:
    """Solubility proxy: favor net charge away from zero and low mean hydrophobicity."""
    seq = _valid(candidate.sequence)
    if not seq:
        return 0.0
    net = sum(_CHARGE.get(a, 0.0) for a in seq) / len(seq)
    hydro = np.mean([_KD[a] for a in seq])
    return _clip01(0.6 + 0.8 * abs(net) - 0.1 * max(0.0, hydro))


def immunogenicity(candidate: Candidate) -> float:
    """MHC-II epitope-load proxy (>=0, lower is better).

    Slide a 15-mer window; a window scores if it has a hydrophobic anchor and high
    mean hydrophobicity (crude stand-in for MHC-II binding). The epitope density
    (hits per residue) is the returned load. Constraint, not an objective.
    """
    seq = _valid(candidate.sequence)
    if len(seq) < 15:
        return 0.0
    hits = 0
    windows = 0
    for i in range(len(seq) - 14):
        w = seq[i : i + 15]
        windows += 1
        anchor = any(a in _MHCII_ANCHORS for a in w[2:9])       # core anchor region
        hydro = np.mean([_KD[a] for a in w])
        if anchor and hydro > 1.0:
            hits += 1
    return float(hits / windows) if windows else 0.0
