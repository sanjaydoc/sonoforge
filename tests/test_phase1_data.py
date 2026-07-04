"""Phase 1 data-layer tests: featurization, filtering, dedup, and round-trip I/O."""

import numpy as np

from sonoforge.data import (
    Candidate,
    SequenceFeaturizer,
    build_dataset,
    dedup_by_sequence,
    filter_by_length,
    load_candidates,
    one_hot,
    records_to_candidates,
    save_candidates,
)
from sonoforge.data.fetch import SequenceRecord, synthetic_seed_records


def _rec(seq, acc="A", gene="gvpA", source="synthetic-seed"):
    return SequenceRecord(accession=acc, name="n", sequence=seq, organism="o",
                          gene=gene, source=source)


# --- featurization ---------------------------------------------------------

def test_featurizer_shape_and_composition_sums_to_one():
    f = SequenceFeaturizer()
    v = f.featurize("ACDEFGHIKLMNPQRSTVWY")
    assert v.shape == (f.n_features,)
    assert len(f.feature_names) == f.n_features
    # first 20 entries are composition; equal residues -> uniform 0.05
    assert np.isclose(v[:20].sum(), 1.0)
    assert np.allclose(v[:20], 0.05)


def test_featurize_many_stacks():
    f = SequenceFeaturizer()
    m = f.featurize_many(["ACDE", "KLMN", "WYWY"])
    assert m.shape == (3, f.n_features)


def test_one_hot_pads_and_truncates():
    oh = one_hot("ACD", max_len=5)
    assert oh.shape == (5, 20)
    assert oh.sum() == 3  # three residues set, rest zero-padded
    assert one_hot("ACDEFGHIKL", max_len=4).shape == (4, 20)


# --- filtering / dedup -----------------------------------------------------

def test_filter_by_length():
    recs = [_rec("A" * 10), _rec("A" * 70), _rec("A" * 500)]
    kept = filter_by_length(recs, lo=40, hi=400)
    assert [len(r.sequence) for r in kept] == [70]


def test_dedup_by_sequence_and_reject_nonstandard():
    recs = [_rec("ACDEFG"), _rec("acdefg"), _rec("ACDEFG"), _rec("ACDXZ1")]
    kept = dedup_by_sequence(recs)
    seqs = [r.sequence for r in kept]
    # case-insensitive dedup keeps one; the non-standard residue record is dropped
    assert len(kept) == 1
    assert seqs[0] in ("ACDEFG", "acdefg")


def test_records_to_candidates_carries_provenance():
    cands = records_to_candidates([_rec("ACDEFGHIKL", acc="X9", gene="gvpC")])
    assert len(cands) == 1
    assert isinstance(cands[0], Candidate)
    assert cands[0].meta["accession"] == "X9"
    assert cands[0].meta["gene"] == "gvpC"


# --- offline build + round-trip -------------------------------------------

def test_build_dataset_offline_fallback():
    # allow_fallback=True with no network -> synthetic seeds, all valid Candidates
    cands = build_dataset(cache_dir=None, allow_fallback=True)
    assert len(cands) > 0
    assert all(isinstance(c, Candidate) for c in cands)
    assert all(c.meta["source"] in ("uniprot", "synthetic-seed") for c in cands)


def test_save_load_round_trip(tmp_path):
    cands = records_to_candidates(synthetic_seed_records("gvpA", n=5))
    cands[0].properties.contrast = 1.23
    path = tmp_path / "lib.jsonl"
    save_candidates(cands, path)
    loaded = load_candidates(path)
    assert len(loaded) == len(cands)
    assert loaded[0].sequence == cands[0].sequence
    assert loaded[0].properties.contrast == 1.23
    assert loaded[0].meta["gene"] == "gvpA"
