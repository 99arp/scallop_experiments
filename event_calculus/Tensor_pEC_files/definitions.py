"""Tensor-pEC-style definitions for STL predicate fluents."""

import jax.numpy as jnp


def initiatedAtStlPredicate(input_events: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return input_events["true"]


def terminatedAtStlPredicate(input_events: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return input_events["false"]


def holdsAtStlPredicate(open_tensor, termAtQt, input_events):
    initiations = initiatedAtStlPredicate(input_events)
    terminations = terminatedAtStlPredicate(input_events)
    matrix_B = initiations + open_tensor
    matrix_C = terminations + termAtQt
    return matrix_B, matrix_C


definitions = {
    ("stl_predicate", "holds", "true"): {
        "initiatedAt": initiatedAtStlPredicate,
        "terminatedAt": terminatedAtStlPredicate,
        "holdsAt": holdsAtStlPredicate,
    },
}


def readDefinitions():
    tensors_dim = [1]
    return definitions.keys(), definitions, tensors_dim
