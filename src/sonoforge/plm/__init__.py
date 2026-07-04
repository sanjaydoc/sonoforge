"""Protein language model (state-space / Mamba-style) for sequence design & scoring.

Only torch-free symbols are exported at package import time so ``import
sonoforge.plm`` works without PyTorch. The torch backend (``model``, ``ssm``,
``train``) is imported lazily by :func:`make_scorer` / training entrypoints.
"""

from sonoforge.plm.scorer import ProfileScorer, make_scorer, torch_available
from sonoforge.plm.tokenizer import ProteinTokenizer

__all__ = ["ProfileScorer", "ProteinTokenizer", "make_scorer", "torch_available"]
