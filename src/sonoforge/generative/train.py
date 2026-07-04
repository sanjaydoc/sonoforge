"""Smoke-train the SE(3) flow-matching backbone generator.

Trains on a tiny synthetic backbone distribution (a noisy helix) so the training
loop is exercised end-to-end with no data or GPU. Real training consumes backbone
coordinates assembled in the data layer.

Requires PyTorch; exits with instructions if it is absent.
"""

from __future__ import annotations

import argparse
import math


def _synthetic_backbones(batch, n_res, jitter, torch):
    base = torch.tensor(
        [[math.cos(i), math.sin(i), 0.3 * i] for i in range(n_res)], dtype=torch.float32
    )
    return base.unsqueeze(0).repeat(batch, 1, 1) + jitter * torch.randn(batch, n_res, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-train the flow-matching generator.")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--n-res", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=3)
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch required: pip install -e '.[ml]'") from None

    from sonoforge.generative.egnn import EGNNVelocity
    from sonoforge.generative.flow import FlowMatcher

    torch.manual_seed(0)
    fm = FlowMatcher(EGNNVelocity(node_dim=16, n_layers=args.n_layers, hidden_dim=32))
    opt = torch.optim.Adam(fm.model.parameters(), lr=5e-3)
    print(f"EGNN velocity field params: {sum(p.numel() for p in fm.model.parameters()):,}")
    for step in range(args.steps):
        loss = fm.train_step(_synthetic_backbones(args.batch, args.n_res, 0.02, torch), opt)
        if (step + 1) % 10 == 0:
            print(f"step {step + 1}/{args.steps}  flow-matching loss = {loss:.4f}")
    sample = fm.sample(n_residues=args.n_res, n_samples=2, steps=20)
    print(f"sampled backbones: {tuple(sample.shape)}")


if __name__ == "__main__":
    main()
