"""Phase 2 PLM tests: tokenizer, ProfileScorer fallback, and (if torch) ProteinSSM."""

import numpy as np
import pytest

from sonoforge.plm import ProfileScorer, ProteinTokenizer, make_scorer, torch_available

# --- tokenizer -------------------------------------------------------------

def test_tokenizer_round_trip():
    tok = ProteinTokenizer()
    ids = tok.encode("ACDEFG")
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids) == "ACDEFG"


def test_tokenizer_batch_padding():
    tok = ProteinTokenizer()
    batch = tok.encode_batch(["ACD", "ACDEFGH"])
    assert batch.shape[0] == 2
    assert batch.shape[1] == 9  # longest (7) + BOS + EOS
    assert (batch[0] == tok.pad_id).any()  # shorter seq is padded


def test_tokenizer_unknown_residue_maps_to_unk():
    tok = ProteinTokenizer()
    ids = tok.encode("AXB", add_special=False)  # X, B are non-standard here
    assert tok.unk_id in ids


# --- ProfileScorer fallback ------------------------------------------------

def test_profile_scorer_pll_prefers_in_distribution():
    # Train on alanine-rich sequences; an alanine-rich query should score higher
    # (less negative) than a query built from rare residues.
    train = ["AAAAAAGGGG", "AAAAGGAAAA", "GAAAAAAAAG"]
    ps = ProfileScorer().fit(train)
    assert ps.pseudo_log_likelihood("AAAAAAAAAA") > ps.pseudo_log_likelihood("WWWWCCWWWW")


def test_profile_scorer_embed_shape():
    ps = ProfileScorer().fit(["ACDEFG", "KLMNPQ"])
    emb = ps.embed(["ACDEFG", "KLMNPQ"])
    assert emb.shape == (2, ps.embed_dim)
    assert np.isfinite(emb).all()


def test_make_scorer_defaults_to_profile_without_checkpoint():
    scorer = make_scorer(sequences=["ACDEFG", "KLMNPQ"])
    assert isinstance(scorer, ProfileScorer)
    assert scorer.embed(["ACDEFG"]).shape[0] == 1


# --- torch backend (skipped when torch is absent) --------------------------

def test_protein_ssm_forward_and_train_step():
    torch = pytest.importorskip("torch")
    from sonoforge.plm.model import ProteinSSM
    from sonoforge.plm.train import train_step

    tok = ProteinTokenizer()
    seqs = ["ACDEFGHIKL", "MNPQRSTVWY", "AAAAGGGGSS"]
    tokens = torch.as_tensor(tok.encode_batch(seqs))

    model = ProteinSSM(tok.vocab_size, d_model=32, n_layers=2, pad_id=tok.pad_id)
    out = model(tokens)
    assert out["logits"].shape == (3, tokens.shape[1], tok.vocab_size)
    assert out["embedding"].shape == (3, 32)
    assert out["properties"].shape == (3, 3)

    # one masked-LM step reduces (or at least yields finite) loss
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = [train_step(model, tokens, tok, opt) for _ in range(5)]
    assert all(np.isfinite(losses))
    assert losses[-1] <= losses[0] + 1e-3  # not diverging


def test_ssm_scorer_embed_and_pll():
    pytest.importorskip("torch")
    if not torch_available():  # pragma: no cover
        pytest.skip("torch unavailable")
    from sonoforge.plm.model import ProteinSSM
    from sonoforge.plm.scorer import SSMScorer

    tok = ProteinTokenizer()
    model = ProteinSSM(tok.vocab_size, d_model=32, n_layers=2, pad_id=tok.pad_id)
    scorer = SSMScorer(model, tok)
    emb = scorer.embed(["ACDEFGHIKL", "MNPQRSTVWY"])
    assert emb.shape == (2, 32)
    assert isinstance(scorer.pseudo_log_likelihood("ACDEFGHIKL"), float)
