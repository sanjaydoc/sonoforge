"""Phase 4 tests: physics collapse pressure, oracle signals, and the OracleStack."""

import numpy as np

from sonoforge.data.types import Candidate
from sonoforge.oracle import OBJECTIVE_NAMES, OracleStack, signals
from sonoforge.physics import collapse_pressure, gnm_rigidity, sequence_rigidity

# --- physics ---------------------------------------------------------------

def test_gnm_rigidity_compact_more_rigid_than_extended():
    rng = np.random.default_rng(0)
    compact = rng.standard_normal((20, 3)) * 0.5           # tight blob (many contacts)
    extended = np.stack([np.arange(20), np.zeros(20), np.zeros(20)], axis=1) * 1.0
    assert gnm_rigidity(compact) > gnm_rigidity(extended)


def test_collapse_pressure_uses_backbone_and_sequence_paths():
    seq = "VIVIVIVIVIVIVIVI"
    cp_seq = collapse_pressure(seq, None)
    coords = np.random.default_rng(1).standard_normal((16, 3))
    cp_bb = collapse_pressure(seq, coords)
    assert cp_seq > 0 and np.isfinite(cp_bb)


def test_sequence_rigidity_beta_vs_helix():
    # β-favoring residues (V,I) should score more rigid than helix/coil-favoring (E,P)
    assert sequence_rigidity("VIVIVIVI") > sequence_rigidity("EPEPEPEP")


# --- signals ---------------------------------------------------------------

def test_signals_in_range_and_deterministic():
    c = Candidate(sequence="MKVIVTLLAGSSEEFWY" * 3)
    for fn in (signals.contrast, signals.expressibility, signals.solubility):
        v = fn(c)
        assert 0.0 <= v <= 1.0
        assert fn(c) == v  # deterministic


def test_immunogenicity_higher_for_hydrophobic_anchors():
    hydrophobic = Candidate(sequence="FILMVWYFILMVWYFILMVWYFILMVWY")
    hydrophilic = Candidate(sequence="EKEKEKDKDKDKSNSNSNSTSTSTQGQG")
    assert signals.immunogenicity(hydrophobic) > signals.immunogenicity(hydrophilic)


# --- OracleStack -----------------------------------------------------------

def test_oracle_stack_fills_record_and_objectives():
    stack = OracleStack()
    c = Candidate(sequence="MKVIVTLLAGSSEE" * 4)
    rec = stack.evaluate(c)
    assert rec.is_complete()
    obj = stack.objectives(rec)
    assert obj.shape == (len(OBJECTIVE_NAMES),)
    assert np.isfinite(obj).all()
    assert isinstance(stack.feasible(rec), bool)


def test_collapse_closeness_peaks_at_target():
    stack = OracleStack(collapse_target=0.5)
    assert stack.collapse_closeness(0.5) == 1.0
    assert stack.collapse_closeness(0.5) > stack.collapse_closeness(0.9)


def test_feasibility_constraint():
    strict = OracleStack(immunogenicity_ceiling=-1.0)  # impossible ceiling
    loose = OracleStack(immunogenicity_ceiling=1.0)
    c = Candidate(sequence="FILMVWYFILMVWYFILMVWY")
    rec = loose.evaluate(c)
    assert loose.feasible(rec) is True
    assert strict.feasible(rec) is False


def test_oracle_cache_roundtrip(tmp_path):
    stack = OracleStack(cache_dir=tmp_path)
    c = Candidate(sequence="MKVIVTLLAGSSEE" * 3)
    r1 = stack.evaluate(c)
    # cache files written per signal
    assert any(tmp_path.glob("*.json"))
    # a fresh stack reading the same cache reproduces the values
    r2 = OracleStack(cache_dir=tmp_path).evaluate(Candidate(sequence=c.sequence))
    assert r1.contrast == r2.contrast
    assert r1.immunogenicity == r2.immunogenicity
