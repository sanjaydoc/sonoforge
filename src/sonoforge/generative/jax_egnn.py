"""JAX/Flax reference for the E(3)-equivariant coordinate update.

A compact, self-contained JAX implementation of the same equivariant update used
by the PyTorch :mod:`~sonoforge.generative.egnn`, provided because the De Novo
Design role asks for **Jax** alongside PyTorch. It demonstrates that the
equivariance is a property of the *computation* (relative differences weighted by
invariant scalars), not of any framework.

``equivariant_update`` is parameter-free (weights are a fixed function of
distance), so its equivariance can be checked directly. ``EquivariantLayer`` is a
learnable Flax module with the same structure.

Requires ``jax`` and ``flax`` (install with ``pip install 'sonoforge[ml]'`` or
``pip install jax flax``).
"""

from __future__ import annotations


def equivariant_update(x, h=None):
    """Parameter-free E(3)-equivariant coordinate delta for points x (N, 3).

    delta_i = mean_j (x_i - x_j) * exp(-||x_i - x_j||^2).  Invariant scalar weight
    => equivariant to rotation, invariant to translation.
    """
    import jax.numpy as jnp

    diff = x[:, None, :] - x[None, :, :]           # (N, N, 3)
    dist2 = jnp.sum(diff ** 2, axis=-1, keepdims=True)
    weight = jnp.exp(-dist2)
    n = x.shape[0]
    eye = jnp.eye(n)[:, :, None]
    delta = jnp.sum(diff * weight * (1.0 - eye), axis=1) / max(n - 1, 1)
    return delta


def _build_layer():
    """Return a Flax EquivariantLayer class (imported lazily so torch-free import works)."""
    import flax.linen as nn
    import jax.numpy as jnp

    class EquivariantLayer(nn.Module):
        hidden: int = 32

        @nn.compact
        def __call__(self, x):
            diff = x[:, None, :] - x[None, :, :]
            dist2 = jnp.sum(diff ** 2, axis=-1, keepdims=True)
            w = nn.Dense(self.hidden)(dist2)
            w = nn.silu(w)
            w = nn.Dense(1)(w)                       # invariant per-edge scalar
            n = x.shape[0]
            eye = jnp.eye(n)[:, :, None]
            delta = jnp.sum(diff * w * (1.0 - eye), axis=1) / max(n - 1, 1)
            return x + delta

    return EquivariantLayer


def make_layer(hidden: int = 32):
    return _build_layer()(hidden=hidden)
