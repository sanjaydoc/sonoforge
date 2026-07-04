"""Phase 3 tests: SE(3) equivariance, flow-matching, frames, and the JAX reference."""

import numpy as np
import pytest

from sonoforge.generative import gram_schmidt, is_rotation, random_rotation
from sonoforge.generative.frames import apply_frame, invert_frame

# --- frames (numpy, always run) -------------------------------------------

def test_random_rotation_is_proper():
    rng = np.random.default_rng(0)
    for _ in range(5):
        assert is_rotation(random_rotation(rng))


def test_gram_schmidt_builds_rotation():
    n = np.array([0.0, 1.0, 0.0])
    ca = np.array([0.0, 0.0, 0.0])
    c = np.array([1.0, 0.0, 0.0])
    rot = gram_schmidt(n, ca, c)
    assert is_rotation(rot)


def test_frame_apply_invert_round_trip():
    rng = np.random.default_rng(1)
    rot = random_rotation(rng)
    trans = rng.standard_normal(3)
    x = rng.standard_normal((6, 3))
    mapped = apply_frame(rot, trans, x)
    rinv, tinv = invert_frame(rot, trans)
    recovered = apply_frame(rinv, tinv, mapped)
    assert np.allclose(recovered, x, atol=1e-6)


# --- torch EGNN equivariance + flow matching -------------------------------

def test_egnn_velocity_is_se3_equivariant():
    torch = pytest.importorskip("torch")
    from sonoforge.generative.egnn import EGNNVelocity

    torch.manual_seed(0)
    model = EGNNVelocity(node_dim=16, n_layers=3, hidden_dim=32).eval()
    x = torch.randn(2, 8, 3)
    t = torch.rand(2)

    # random rotation + translation
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    trans = torch.randn(3)

    with torch.no_grad():
        v = model(x, t)
        v_transformed = model(x @ q.T + trans, t)
    # velocity should rotate with the input and be invariant to translation
    assert torch.allclose(v_transformed, v @ q.T, atol=1e-4)


def test_flow_matching_trains_and_samples():
    torch = pytest.importorskip("torch")
    from sonoforge.generative.egnn import EGNNVelocity
    from sonoforge.generative.flow import FlowMatcher

    torch.manual_seed(0)
    n_res = 6
    # tiny synthetic "structure" distribution: a fixed helix + small noise
    base = torch.tensor(
        [[np.cos(i), np.sin(i), 0.3 * i] for i in range(n_res)], dtype=torch.float32
    )
    fm = FlowMatcher(EGNNVelocity(node_dim=16, n_layers=3, hidden_dim=32))
    opt = torch.optim.Adam(fm.model.parameters(), lr=5e-3)

    def batch():
        return base.unsqueeze(0).repeat(8, 1, 1) + 0.02 * torch.randn(8, n_res, 3)

    losses = [fm.train_step(batch(), opt) for _ in range(30)]
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]  # learning

    samples = fm.sample(n_residues=n_res, n_samples=3, steps=20)
    assert samples.shape == (3, n_res, 3)
    assert torch.isfinite(samples).all()


# --- JAX / Flax reference --------------------------------------------------

def test_jax_equivariant_update_is_equivariant():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    import numpy as _np

    from sonoforge.generative.jax_egnn import equivariant_update

    rng = _np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((7, 3)))
    q = jnp.asarray(random_rotation(rng))
    trans = jnp.asarray(rng.standard_normal(3))

    d = equivariant_update(x)
    d_t = equivariant_update(x @ q.T + trans)
    assert bool(jnp.allclose(d_t, d @ q.T, atol=1e-5))


def test_jax_flax_layer_runs_and_is_equivariant():
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax
    import jax.numpy as jnp
    import numpy as _np

    from sonoforge.generative.jax_egnn import make_layer

    rng = _np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((5, 3)))
    layer = make_layer(hidden=16)
    params = layer.init(jax.random.PRNGKey(0), x)

    q = jnp.asarray(random_rotation(rng))
    trans = jnp.asarray(rng.standard_normal(3))
    y = layer.apply(params, x)
    y_t = layer.apply(params, x @ q.T + trans)
    # output is x + equivariant delta, so it maps as R y + t
    assert bool(jnp.allclose(y_t, y @ q.T + trans, atol=1e-4))
