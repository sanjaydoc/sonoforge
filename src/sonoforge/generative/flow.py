"""Conditional flow matching over backbone Cα coordinates.

Uses the linear (optimal-transport) probability path: for data ``x1`` and noise
``x0 ~ N(0, I)``,

    x_t = (1 - t) * x0 + t * x1,      target velocity  u = x1 - x0.

The model :class:`~sonoforge.generative.egnn.EGNNVelocity` regresses ``u`` from
``(x_t, t)``. Because the velocity field is SE(3)-equivariant, the learned
generative flow respects rotational/translational symmetry of protein backbones.
Sampling integrates the ODE ``dx/dt = v_theta(x_t, t)`` with explicit Euler steps.

Requires PyTorch.
"""

from __future__ import annotations

import torch

from sonoforge.generative.egnn import EGNNVelocity


def _center(x: torch.Tensor) -> torch.Tensor:
    """Remove per-sample centre of mass (work in a translation-free frame)."""
    return x - x.mean(dim=1, keepdim=True)


class FlowMatcher:
    """Conditional flow matching with an EGNN velocity field."""

    def __init__(self, model: EGNNVelocity | None = None, **model_kwargs) -> None:
        self.model = model or EGNNVelocity(**model_kwargs)

    def loss(self, x1: torch.Tensor) -> torch.Tensor:
        """Flow-matching MSE loss for a batch of backbones x1 (B, N, 3)."""
        b = x1.shape[0]
        x1 = _center(x1)
        x0 = _center(torch.randn_like(x1))
        t = torch.rand(b, 1, device=x1.device)
        xt = (1.0 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * x1
        target = x1 - x0
        pred = self.model(xt, t.squeeze(-1))
        return ((pred - target) ** 2).mean()

    def train_step(self, x1: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        loss = self.loss(x1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    @torch.no_grad()
    def sample(self, n_residues: int, n_samples: int = 1, steps: int = 50, device: str = "cpu") -> torch.Tensor:
        """Integrate the flow from noise to generate backbones (n_samples, N, 3)."""
        self.model.eval()
        x = _center(torch.randn(n_samples, n_residues, 3, device=device))
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((n_samples,), i * dt, device=device)
            x = x + self.model(x, t) * dt
        return _center(x)
