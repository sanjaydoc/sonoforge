"""Phase 0 scaffold tests: the package imports and the Candidate schema behaves."""

import pytest

import sonoforge
from sonoforge import Candidate, PropertyRecord
from sonoforge.utils import set_seed


def test_version():
    assert sonoforge.__version__ == "0.1.0"


def test_candidate_valid():
    c = Candidate(sequence="mkleiv", cycle=1)
    assert len(c) == 6
    assert c.sequence == "MKLEIV"  # normalized to upper-case
    assert isinstance(c.properties, PropertyRecord)
    assert not c.properties.is_complete()


def test_candidate_rejects_nonstandard_residues():
    with pytest.raises(ValueError):
        Candidate(sequence="MKZX1")


def test_candidate_rejects_empty():
    with pytest.raises(ValueError):
        Candidate(sequence="   ")


def test_property_record_completeness():
    p = PropertyRecord(contrast=1.0, collapse_pressure=0.5, expressibility=0.8,
                       solubility=0.7, immunogenicity=0.1)
    assert p.is_complete()


def test_set_seed_runs():
    set_seed(0)  # must not raise even without numpy/torch installed
