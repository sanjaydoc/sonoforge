"""SE(3)-equivariant flow-matching generator for backbone design.

Frame utilities (:mod:`frames`) are pure-numpy and exported here. The torch
generator (:mod:`egnn`, :mod:`flow`) and the JAX reference (:mod:`jax_egnn`) are
imported lazily so ``import sonoforge.generative`` works without torch/jax.
"""

from sonoforge.generative.frames import (
    apply_frame,
    gram_schmidt,
    invert_frame,
    is_rotation,
    random_rotation,
)

__all__ = [
    "apply_frame",
    "gram_schmidt",
    "invert_frame",
    "is_rotation",
    "random_rotation",
]
