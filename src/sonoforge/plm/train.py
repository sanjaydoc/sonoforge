"""Masked-language-model training for ProteinSSM.

Pretrain on a broad corpus, then fine-tune on the (small, sparse) gas-vesicle
family — the transfer-learning strategy the roles call for. This module holds the
training step and a ``--synthetic`` smoke path that needs no data or GPU.

Requires PyTorch; the CLI exits gracefully with instructions if it is absent.
"""

from __future__ import annotations

import argparse
import random

from sonoforge.plm.scorer import torch_available
from sonoforge.plm.tokenizer import ProteinTokenizer


def _mask_tokens(tokens, tokenizer: ProteinTokenizer, p: float = 0.15):
    """BERT-style masking. Returns (masked_input, labels) with -100 on kept positions."""
    import torch

    from sonoforge.plm.tokenizer import SPECIAL_TOKENS

    labels = tokens.clone()
    probs = torch.rand(tokens.shape)
    is_aa = tokens.ge(len(SPECIAL_TOKENS))  # amino-acid ids come after the special tokens
    maskable = is_aa & (probs < p)
    labels[~maskable] = -100
    masked = tokens.clone()
    masked[maskable] = tokenizer.mask_id
    return masked, labels


def train_step(model, tokens, tokenizer, optimizer):
    import torch.nn.functional as F

    model.train()
    masked, labels = _mask_tokens(tokens, tokenizer)
    logits = model(masked)["logits"]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _synthetic_sequences(n: int = 64, length: int = 60, seed: int = 0) -> list[str]:
    from sonoforge.data.types import AA_ALPHABET

    rng = random.Random(seed)
    # a biased alphabet so there is signal for the LM to learn
    weights = [3 if aa in "AGLVSE" else 1 for aa in AA_ALPHABET]
    return ["".join(rng.choices(AA_ALPHABET, weights=weights, k=length)) for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ProteinSSM (masked LM).")
    ap.add_argument("--synthetic", action="store_true", help="smoke-train on synthetic data")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    args = ap.parse_args()

    if not torch_available():
        raise SystemExit(
            "PyTorch is required to train the PLM. Install it with:\n"
            "  pip install -e '.[ml]'\n"
            "(The DBTL loop still runs without it via the ProfileScorer fallback.)"
        )

    import torch

    from sonoforge.plm.model import ProteinSSM

    tok = ProteinTokenizer()
    if args.synthetic:
        seqs = _synthetic_sequences()
    else:
        from pathlib import Path

        from sonoforge.data.dataset import load_candidates

        lib = Path("data/seed_library.jsonl")
        if not lib.exists():
            raise SystemExit("No data/seed_library.jsonl — run scripts/download_data.py first, "
                             "or use --synthetic.")
        seqs = [c.sequence for c in load_candidates(lib)]
    tokens = torch.as_tensor(tok.encode_batch(seqs))
    model = ProteinSSM(tok.vocab_size, d_model=args.d_model, n_layers=args.n_layers, pad_id=tok.pad_id)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"ProteinSSM params: {model.num_parameters():,}")
    for epoch in range(args.epochs):
        loss = train_step(model, tokens, tok, opt)
        print(f"epoch {epoch + 1}/{args.epochs}  masked-LM loss = {loss:.4f}")


if __name__ == "__main__":
    main()
