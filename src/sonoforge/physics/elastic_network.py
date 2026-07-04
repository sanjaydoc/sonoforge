"""Physics-based mechanical model of the gas-vesicle shell.

The collapse pressure of a gas vesicle is a *mechanical* property — the pressure
at which the protein shell buckles — so it is a natural target for a
first-principles model. Here we use a **Gaussian Network Model (GNM)**: residues
are nodes, springs connect residues in contact, and the eigenvalues of the
Kirchhoff (graph-Laplacian) matrix give the vibrational stiffness spectrum. The
mean non-trivial stiffness is a physically-grounded proxy for shell rigidity and
hence collapse pressure — computable directly from a backbone, no forcefield
required.

For production this is the drop-in point for a full **OpenMM** molecular-dynamics
pipeline (explicit forcefield, applied external pressure, buckling detection);
``openmm_collapse_pressure`` marks that hook. When neither a backbone nor OpenMM
is available, a sequence-based rigidity proxy is used (β-sheet propensity: the
GvpA shell is a rigid β-sheet).

Pure numpy for the GNM path.
"""

from __future__ import annotations

import numpy as np

# Chou–Fasman β-sheet propensities (higher => more β-sheet, more rigid shell).
_BETA_PROPENSITY = {
    "V": 1.70, "I": 1.60, "Y": 1.47, "C": 1.19, "W": 1.37, "F": 1.38, "L": 1.30,
    "T": 1.19, "Q": 1.10, "M": 1.05, "R": 0.93, "N": 0.89, "H": 0.87, "A": 0.83,
    "S": 0.75, "G": 0.75, "K": 0.74, "P": 0.55, "D": 0.54, "E": 0.37,
}
# scale constant mapping dimensionless stiffness -> an "MPa-scale" proxy value
_MPA_SCALE = 0.6


def gnm_rigidity(coords: np.ndarray) -> float:
    """Mean non-trivial Kirchhoff eigenvalue of a GNM built from Cα coords (N, 3)."""
    n = coords.shape[0]
    if n < 3:
        return 0.0
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    # scale-adaptive cutoff: 1.6x the median sequential Cα-Cα spacing
    seq_d = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    cutoff = 1.6 * float(np.median(seq_d)) if seq_d.size else 1.0
    contact = (dist <= cutoff) & (dist > 0)
    kirchhoff = np.diag(contact.sum(1).astype(float)) - contact.astype(float)
    eig = np.linalg.eigvalsh(kirchhoff)
    nontrivial = eig[eig > 1e-6]              # drop the single zero mode
    return float(nontrivial.mean()) if nontrivial.size else 0.0


def sequence_rigidity(sequence: str) -> float:
    """β-sheet-propensity rigidity proxy in [0, ~1.7] for a bare sequence."""
    seq = [aa for aa in sequence.upper() if aa in _BETA_PROPENSITY]
    if not seq:
        return 0.0
    return float(np.mean([_BETA_PROPENSITY[aa] for aa in seq]))


def collapse_pressure(sequence: str, coords: np.ndarray | None = None) -> float:
    """Collapse-pressure proxy (MPa-scale). Uses the GNM if a backbone is given."""
    if coords is not None and len(coords) >= 3:
        return _MPA_SCALE * gnm_rigidity(np.asarray(coords, dtype=float))
    return _MPA_SCALE * sequence_rigidity(sequence)


def openmm_collapse_pressure(sequence: str, coords: np.ndarray) -> float:  # pragma: no cover
    """Production hook: full OpenMM MD collapse-pressure estimate.

    Intentionally not implemented in the reference build (requires openmm + a
    forcefield + a solvated shell model). Kept as the documented drop-in point so
    the oracle interface is stable when the MD pipeline lands.
    """
    raise NotImplementedError(
        "OpenMM MD collapse-pressure is a production hook; install '.[physics]' "
        "and implement the solvated-shell buckling simulation here."
    )
