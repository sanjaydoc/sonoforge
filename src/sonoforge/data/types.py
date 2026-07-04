"""Core data types for SonoForge candidates.

A ``Candidate`` is the unit that flows through the DBTL loop. It carries the
designed sequence (and optionally a backbone), a ``PropertyRecord`` of oracle
outputs, and provenance so any design is traceable back through cycles.

The property fields mirror the multi-objective problem defined in the
whitepaper: maximize acoustic contrast, hit a target collapse pressure, remain
expressible/soluble, and stay *below* an immunogenicity ceiling (a hard
constraint, not an objective to maximize).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# 20 canonical amino acids.
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_AA_SET = frozenset(AA_ALPHABET)


@dataclass
class PropertyRecord:
    """Oracle outputs for one candidate. ``None`` means "not yet measured".

    All fields are in-silico proxies in the reference platform and are labelled
    as such; real wet-lab measurements plug into the same schema.
    """

    contrast: float | None = None          # ultrasound scattering proxy (higher is better)
    collapse_pressure: float | None = None  # MPa-scale proxy; optimized toward a target set-point
    expressibility: float | None = None     # [0, 1]
    solubility: float | None = None         # [0, 1]
    immunogenicity: float | None = None     # MHC-II epitope load; CONSTRAINT (lower is better)

    def is_complete(self) -> bool:
        return all(getattr(self, f) is not None for f in self.__dataclass_fields__)


@dataclass
class Candidate:
    """A designed biomolecule flowing through the DBTL loop."""

    sequence: str
    backbone: Any | None = None            # (L, 4, 3) frames or (L, 3) Cα; None until designed
    properties: PropertyRecord = field(default_factory=PropertyRecord)
    cycle: int = 0                            # DBTL cycle in which it was proposed
    parents: list[str] = field(default_factory=list)  # sequences it was derived from
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sequence = self.sequence.upper().strip()
        bad = set(self.sequence) - _AA_SET
        if bad:
            raise ValueError(f"sequence contains non-standard residues: {sorted(bad)}")
        if not self.sequence:
            raise ValueError("sequence must be non-empty")

    def __len__(self) -> int:
        return len(self.sequence)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["backbone"] = None if self.backbone is None else "<array>"
        return d
