"""Tensor-pEC-style declarations for STL predicate Event Calculus."""

InputEvent = "InputEvent"
SimpleFluent = "SimpleFluent"

declarations = {
    ("stl_predicate", "true"): {
        "type": InputEvent,
        "index": ("source_kind", "name"),
        "args": ("value",),
        "Ndim": 1,
        "library": "jax",
    },
    ("stl_predicate", "false"): {
        "type": InputEvent,
        "index": ("source_kind", "name"),
        "args": ("value",),
        "Ndim": 1,
        "library": "jax",
    },
    ("stl_predicate", "holds", "true"): {
        "type": SimpleFluent,
        "index": ("source_kind", "name"),
        "args": ("value",),
        "Ndim": 1,
        "library": "jax",
    },
}
