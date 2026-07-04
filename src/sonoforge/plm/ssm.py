"""A selective state-space (SSM) block — a readable, dependency-free reference.

This implements the core recurrence behind Mamba/S6: a diagonal state-space
model whose input, output, and timestep (``delta``) projections are
**input-dependent** ("selective"), which is what lets an SSM gate information
content-dependently the way attention does — at linear cost in sequence length.

The scan here is an explicit Python loop over time: correct and easy to read,
but O(L) sequential steps. Production deployments swap in the fused parallel-scan
CUDA kernels from ``mamba-ssm``; the module interface is identical.

Requires PyTorch (only imported when the PLM's torch backend is used).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class SelectiveSSM(nn.Module):
    """Single selective diagonal SSM layer operating on (B, L, d_model)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1,
        )
        # input-dependent delta, B, C
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + 1)
        self.dt_proj = nn.Linear(1, self.d_inner)

        # log of (negative) diagonal state matrix A, parameterized per (inner, state)
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        xz = self.in_proj(x)                       # (B, L, 2*d_inner)
        xin, z = xz.chunk(2, dim=-1)               # gate branch z

        # depthwise causal conv over time
        xin = xin.transpose(1, 2)                  # (B, d_inner, L)
        xin = self.conv1d(xin)[..., :length]
        xin = F.silu(xin).transpose(1, 2)          # (B, L, d_inner)

        # input-dependent params
        proj = self.x_proj(xin)                    # (B, L, 2*d_state + 1)
        dt, bmat, cmat = torch.split(proj, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))       # (B, L, d_inner)
        a = -torch.exp(self.A_log)                 # (d_inner, d_state)

        # sequential selective scan
        h = x.new_zeros(b, self.d_inner, self.d_state)
        ys = []
        for t in range(length):
            dt_t = delta[:, t]                     # (B, d_inner)
            da = torch.exp(dt_t.unsqueeze(-1) * a)             # (B, d_inner, d_state)
            db = dt_t.unsqueeze(-1) * bmat[:, t].unsqueeze(1)  # (B, d_inner, d_state)
            h = da * h + db * xin[:, t].unsqueeze(-1)
            y_t = torch.einsum("bds,bs->bd", h, cmat[:, t])    # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                 # (B, L, d_inner)
        y = y + xin * self.D
        y = y * F.silu(z)                          # gated
        return self.out_proj(y)


class SSMResidualBlock(nn.Module):
    """Pre-norm residual wrapper around a :class:`SelectiveSSM`."""

    def __init__(self, d_model: int, **kwargs) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))
