import functools

import jax
import jax.numpy as jnp


def cond(pred, true_fun, false_fun, *operands):
    if pred:
        return true_fun(*operands)
    return false_fun(*operands)


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)


def smooth_mask(T, t_start, t_end, scale):
    xs = jnp.arange(T) * 1.0
    return jax.nn.sigmoid(scale * (xs - t_start * T)) - jax.nn.sigmoid(
        scale * (xs - t_end * T)
    )


def anneal(i):
    return jax.nn.sigmoid(15 * (i - 0.5))


@jax.jit
def bar_plus(signal, p=2):
    return jax.nn.relu(signal) ** p


@jax.jit
def bar_minus(signal, p=2):
    return (-jax.nn.relu(-signal)) ** p


@functools.partial(jax.jit, static_argnames=("axis", "keepdims"))
def M0(signal, eps, weights=None, axis=1, keepdims=True):
    if weights is None:
        weights = jnp.ones_like(signal)
    sum_w = weights.sum(axis, keepdims=keepdims)
    return (eps**sum_w + jnp.prod(signal**weights, axis=axis, keepdims=keepdims)) ** (
        1 / sum_w
    )


@functools.partial(jax.jit, static_argnames=("axis", "keepdims"))
def Mp(signal, eps, p, weights=None, axis=1, keepdims=True):
    if weights is None:
        weights = jnp.ones_like(signal)
    sum_w = weights.sum(axis, keepdims=keepdims)
    return (
        eps**p
        + 1 / sum_w * jnp.sum(weights * signal**p, axis=axis, keepdims=keepdims)
    ) ** (1 / p)


@functools.partial(jax.jit, static_argnames=("axis", "keepdims"))
def gmsr_min(signal, eps, p, weights=None, axis=1, keepdims=True):
    return (
        M0(bar_plus(signal, 2), eps, weights=weights, axis=axis, keepdims=keepdims)
        ** 0.5
        - Mp(
            bar_minus(signal, 2), eps, p, weights=weights, axis=axis, keepdims=keepdims
        )
        ** 0.5
    )


@functools.partial(jax.jit, static_argnames=("axis", "keepdims"))
def gmsr_max(signal, eps, p, weights=None, axis=1, keepdims=True):
    return -gmsr_min(-signal, eps, p, weights=weights, axis=axis, keepdims=keepdims)


@functools.partial(jax.jit, static_argnames=("axis", "keepdims"))
def gmsr_min_turbo(signal, eps, p, weights=None, axis=1, keepdims=True):
    pos_idx = signal > 0.0
    neg_idx = ~pos_idx

    return jnp.where(
        neg_idx.sum(axis, keepdims=keepdims) > 0,
        eps**0.5
        - Mp(
            bar_minus(signal, 2),
            eps,
            p,
            weights=weights,
            axis=axis,
            keepdims=keepdims,
        )
        ** 0.5,
        M0(bar_plus(signal, 2), eps, weights=weights, axis=axis, keepdims=keepdims)
        ** 0.5
        - eps**0.5,
    )


@functools.partial(jax.jit, static_argnames=("axis", "keepdims"))
def gmsr_max_turbo(signal, eps, p, weights=None, axis=1, keepdims=True):
    return -gmsr_min_turbo(-signal, eps, p, weights=weights, axis=axis, keepdims=keepdims)


def gmsr_min_fast(signal, eps, p):
    pos_idx = signal > 0.0
    neg_idx = ~pos_idx
    weights = jnp.ones_like(signal)
    sum_w = weights.sum()

    if signal[neg_idx].size > 0:
        sums = jnp.sum(weights[neg_idx] * (signal[neg_idx] ** (2 * p)))
        mp_value = (eps**p + (sums / sum_w)) ** (1 / p)
        h_min = eps**0.5 - mp_value**0.5
    else:
        mult = jnp.prod(signal[pos_idx] ** (2 * weights[pos_idx]))
        m0_value = (eps**sum_w + mult) ** (1 / sum_w)
        h_min = m0_value**0.5 - eps**0.5

    return jnp.reshape(h_min, (1, 1, 1))


def gmsr_max_fast(signal, eps, p):
    return -gmsr_min_fast(-signal, eps, p)


@functools.partial(
    jax.jit,
    static_argnames=("axis", "keepdims", "approx_method", "padding"),
)
def maxish(signal, axis, keepdims=True, approx_method="true", temperature=None, **kwargs):
    del kwargs
    match approx_method:
        case "true":
            return jnp.max(signal, axis, keepdims=keepdims)
        case "logsumexp":
            assert temperature is not None, "need a temperature value"
            return (
                jax.scipy.special.logsumexp(
                    temperature * signal, axis=axis, keepdims=keepdims
                )
                / temperature
            )
        case "softmax":
            assert temperature is not None, "need a temperature value"
            return (jax.nn.softmax(temperature * signal, axis) * signal).sum(
                axis, keepdims=keepdims
            )
        case "gmsr":
            assert temperature is not None, "temperature tuple containing (eps, p) is required"
            (eps, p) = temperature
            return gmsr_max(signal, eps, p, axis=axis, keepdims=keepdims)
        case _:
            raise ValueError("Invalid approx_method")


@functools.partial(
    jax.jit,
    static_argnames=("axis", "keepdims", "approx_method", "padding"),
)
def minish(signal, axis, keepdims=True, approx_method="true", temperature=None, **kwargs):
    return -maxish(-signal, axis, keepdims, approx_method, temperature, **kwargs)
