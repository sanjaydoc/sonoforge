"""Rigid-body frame utilities for backbone geometry.

A residue frame is a rigid transform ``(R, t)`` with ``R in SO(3)`` and ``t in
R^3`` built from the backbone N–Cα–C atoms (Gram–Schmidt), following the
convention used by AlphaFold's frame representation. These are the objects the
generator ultimately places; the Phase-3 flow model operates on the Cα
translations, and this module provides the orientation machinery the frame-level
extension builds on.

Pure numpy — no torch required.
"""

from __future__ import annotations

import numpy as np


def gram_schmidt(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Build a rotation matrix (3, 3) from backbone N, Cα, C coordinates.

    Column convention: e1 along (C - Cα), e2 orthogonalized (N - Cα), e3 = e1×e2.
    """
    v1 = c - ca
    v2 = n - ca
    e1 = v1 / (np.linalg.norm(v1) + 1e-8)
    u2 = v2 - np.dot(v2, e1) * e1
    e2 = u2 / (np.linalg.norm(u2) + 1e-8)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=1)  # (3, 3), columns are the frame axes


def apply_frame(rot: np.ndarray, trans: np.ndarray, local: np.ndarray) -> np.ndarray:
    """Map local coordinates (..., 3) into the global frame: R @ x + t."""
    return local @ rot.T + trans


def invert_frame(rot: np.ndarray, trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse rigid transform."""
    rt = rot.T
    return rt, -rt @ trans


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly-random rotation matrix via QR of a Gaussian matrix."""
    a = rng.standard_normal((3, 3))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))          # fix signs
    if np.linalg.det(q) < 0:             # ensure proper rotation (det +1)
        q[:, 0] = -q[:, 0]
    return q


def is_rotation(rot: np.ndarray, tol: float = 1e-5) -> bool:
    """True if ``rot`` is a proper rotation (orthonormal, det +1)."""
    ortho = np.allclose(rot @ rot.T, np.eye(3), atol=tol)
    return ortho and abs(np.linalg.det(rot) - 1.0) < tol
