"""ProteinSSM — the SonoForge protein language model.

A stack of selective SSM blocks over amino-acid tokens with three heads:

- **LM head** — per-position logits over the vocabulary (masked-LM training and
  inverse-folding-style sequence design).
- **pooled embedding** — a mean-pooled representation used as the Bayesian-
  optimization search space (representation learning).
- **property head** — regresses candidate properties (expressibility, solubility,
  contrast surrogate) directly from the pooled embedding.

Requires PyTorch (only imported when the torch backend is used).
"""

from __future__ import annotations

import torch
from torch import nn

from sonoforge.plm.ssm import SSMResidualBlock


class ProteinSSM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_properties: int = 3,
        pad_id: int = 0,
        d_state: int = 16,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.blocks = nn.ModuleList(
            [SSMResidualBlock(d_model, d_state=d_state) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.property_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, n_properties)
        )

    def backbone(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.embed(tokens)
        for blk in self.blocks:
            h = blk(h)
        return self.norm_f(h)

    def pooled(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mask-aware mean pooling over non-pad positions -> (B, d_model)."""
        h = self.backbone(tokens)
        mask = (tokens != self.pad_id).unsqueeze(-1).to(h.dtype)
        summed = (h * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(tokens)
        mask = (tokens != self.pad_id).unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return {
            "logits": self.lm_head(h),          # (B, L, vocab)
            "embedding": pooled,                # (B, d_model)
            "properties": self.property_head(pooled),  # (B, n_properties)
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
