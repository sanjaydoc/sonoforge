"""E(3)-equivariant graph network (EGNN) — the velocity field for flow matching.

Implements the EGNN update of Satorras, Hoogeboom & Welling (2021). The key
property, verified in the test suite: node **features are invariant** and
**coordinate updates are equivariant** under any rotation + translation, because
every coordinate update is a sum of relative differences ``(x_i - x_j)`` weighted
by functions of *distances* only. That makes the whole network SE(3)-equivariant
by construction — no data augmentation required.

Used as the time-conditioned velocity field ``v_theta(x_t, t)`` for the
conditional flow-matching generator over backbone Cα coordinates.

Requires PyTorch (imported only when the torch generator is used).
"""

from __future__ import annotations

import torch
from torch import nn


class EGNNLayer(nn.Module):
    """One equivariant message-passing layer over a fully-connected point set."""

    def __init__(self, node_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        # scalar coordinate weight per edge (keeps updates equivariant)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: (B, N, node_dim)   x: (B, N, 3)
        b, n, _ = x.shape
        xi = x.unsqueeze(2).expand(b, n, n, 3)
        xj = x.unsqueeze(1).expand(b, n, n, 3)
        diff = xi - xj                                   # (B, N, N, 3)
        dist2 = (diff ** 2).sum(-1, keepdim=True)        # (B, N, N, 1) — invariant

        hi = h.unsqueeze(2).expand(b, n, n, h.shape[-1])
        hj = h.unsqueeze(1).expand(b, n, n, h.shape[-1])
        edge_in = torch.cat([hi, hj, dist2], dim=-1)
        m = self.edge_mlp(edge_in)                       # (B, N, N, hidden)

        # equivariant coordinate update: weighted sum of relative differences.
        # normalize direction by (dist + 1) and bound the scalar weight with tanh
        # so updates stay O(1) and training is stable (still equivariant: scalar * diff).
        weight = torch.tanh(self.coord_mlp(m))           # (B, N, N, 1) in [-1, 1]
        # +eps inside sqrt: the diagonal has dist2=0 where sqrt has an infinite
        # gradient, which would produce NaNs in backward.
        dist = torch.sqrt(dist2 + 1e-8)
        coord_upd = (diff / (dist + 1.0) * weight)
        # zero self-edges
        eye = torch.eye(n, device=x.device).view(1, n, n, 1)
        coord_upd = coord_upd * (1.0 - eye)
        delta = coord_upd.sum(dim=2) / max(n - 1, 1)     # (B, N, 3)

        # invariant feature update
        m_agg = (m * (1.0 - eye)).sum(dim=2)             # (B, N, hidden)
        h_new = h + self.node_mlp(torch.cat([h, m_agg], dim=-1))
        return h_new, x + delta


class EGNNVelocity(nn.Module):
    """Time-conditioned SE(3)-equivariant velocity field over Cα coordinates."""

    def __init__(self, node_dim: int = 32, n_layers: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.time_mlp = nn.Sequential(nn.Linear(1, node_dim), nn.SiLU(), nn.Linear(node_dim, node_dim))
        self.layers = nn.ModuleList([EGNNLayer(node_dim, hidden_dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return per-node velocity (B, N, 3): equivariant to rotation, invariant to translation."""
        b, n, _ = x.shape
        if t.dim() == 1:
            t = t.view(b, 1)
        h = self.time_mlp(t).unsqueeze(1).expand(b, n, self.node_dim).contiguous()
        x0 = x
        xt = x
        for layer in self.layers:
            h, xt = layer(h, xt)
        return xt - x0  # velocity = net displacement (translation-invariant, rotation-equivariant)
