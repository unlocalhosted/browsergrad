/**
 * Functional ops — pure functions that take Tensor(s), return Tensor.
 *
 * v0.2: softmax, log_softmax, gelu, leaky_relu, cross_entropy_loss.
 * v0.1 ops (relu, sigmoid, tanh, mse_loss) retained unchanged.
 *
 * cross_entropy_loss is a fused op (softmax → -log → gather → mean). Fused
 * because:
 *   1. Numerical stability — naive softmax + log overflows for large logits.
 *   2. Bypasses the need for differentiable integer indexing — the gradient
 *      simplifies to `(softmax(x) - one_hot(y)) / N` directly.
 */

export const FUNCTIONAL_PY = `
"""browsergrad_grad.functional — stateless ops with autograd."""

import numpy as np
from .tensor import Tensor, _build_ctx


# ─── Activations ───────────────────────────────────────────

def relu(x: Tensor) -> Tensor:
    out_data = np.maximum(x.data, 0.0)
    out = Tensor(out_data)
    mask = (x.data > 0).astype(np.float32)
    return _build_ctx(out, (x,), lambda g: (g.data * mask,))


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    out_data = np.where(x.data > 0, x.data, x.data * negative_slope)
    out = Tensor(out_data)
    mask = np.where(x.data > 0, 1.0, negative_slope).astype(np.float32)
    return _build_ctx(out, (x,), lambda g: (g.data * mask,))


def sigmoid(x: Tensor) -> Tensor:
    s = 1.0 / (1.0 + np.exp(-x.data))
    out = Tensor(s)
    return _build_ctx(out, (x,), lambda g: (g.data * s * (1.0 - s),))


def tanh(x: Tensor) -> Tensor:
    t = np.tanh(x.data)
    out = Tensor(t)
    return _build_ctx(out, (x,), lambda g: (g.data * (1.0 - t * t),))


# Tanh-approximation GELU (GPT-2 / BERT variant)
_GELU_C = float(np.sqrt(2.0 / np.pi))

def gelu(x: Tensor) -> Tensor:
    xd = x.data
    inner = _GELU_C * (xd + 0.044715 * (xd ** 3))
    tanh_inner = np.tanh(inner)
    out_data = 0.5 * xd * (1.0 + tanh_inner)
    out = Tensor(out_data)
    # Derivative: 0.5*(1+tanh(inner)) + 0.5*x*sech²(inner) * d(inner)/dx
    # sech²(t) = 1 - tanh²(t); d(inner)/dx = C*(1 + 3*0.044715*x²)
    sech2 = 1.0 - tanh_inner * tanh_inner
    d_inner = _GELU_C * (1.0 + 3.0 * 0.044715 * (xd ** 2))
    deriv = 0.5 * (1.0 + tanh_inner) + 0.5 * xd * sech2 * d_inner
    return _build_ctx(out, (x,), lambda g: (g.data * deriv,))


# ─── Softmax family ────────────────────────────────────────

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Stable softmax along \`dim\`."""
    xd = x.data
    shifted = xd - xd.max(axis=dim, keepdims=True)
    exp_data = np.exp(shifted)
    sum_data = exp_data.sum(axis=dim, keepdims=True)
    s = exp_data / sum_data
    out = Tensor(s)
    # d(softmax)/dx_i for row r:  s_i * (g_i - sum_j(s_j * g_j))
    def backward(g):
        # weighted sum along dim
        ws = (g.data * s).sum(axis=dim, keepdims=True)
        return (s * (g.data - ws),)
    return _build_ctx(out, (x,), backward)


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable log_softmax along \`dim\`."""
    xd = x.data
    shifted = xd - xd.max(axis=dim, keepdims=True)
    log_sum = np.log(np.exp(shifted).sum(axis=dim, keepdims=True))
    out_data = shifted - log_sum
    out = Tensor(out_data)
    # d(log_softmax)/dx_i for row r: g_i - softmax_i * sum_j(g_j)
    s = np.exp(out_data)  # softmax recovered from log_softmax
    def backward(g):
        g_sum = g.data.sum(axis=dim, keepdims=True)
        return (g.data - s * g_sum,)
    return _build_ctx(out, (x,), backward)


# ─── Losses ────────────────────────────────────────────────

def mse_loss(y_hat: Tensor, y: Tensor) -> Tensor:
    """Mean squared error. Returns a scalar Tensor."""
    if y_hat.data.shape != y.data.shape:
        raise ValueError(
            f"mse_loss: shape mismatch {y_hat.data.shape} vs {y.data.shape}"
        )
    diff = y_hat.data - y.data
    loss_data = float(np.mean(diff * diff))
    out = Tensor(np.float32(loss_data))
    n = float(y.data.size)
    return _build_ctx(out, (y_hat,), lambda g: (g.data * (2.0 / n) * diff,))


def cross_entropy_loss(logits: Tensor, targets) -> Tensor:
    """Fused softmax + NLL: -mean(log_softmax(logits)[range(N), targets]).

    \`logits\`: shape (N, C). \`targets\`: int array shape (N,) — class indices.
    Returns a scalar Tensor. The fused form avoids both the numerical-stability
    pitfalls of naive log(softmax) and the need for differentiable indexing.
    """
    if isinstance(targets, Tensor):
        targets_np = targets.data.astype(np.int64)
    else:
        targets_np = np.asarray(targets, dtype=np.int64)
    if logits.data.ndim != 2:
        raise ValueError(f"cross_entropy_loss: logits must be 2D (N, C), got {logits.data.ndim}D")
    if targets_np.ndim != 1 or targets_np.shape[0] != logits.data.shape[0]:
        raise ValueError(
            f"cross_entropy_loss: targets shape {targets_np.shape} doesn't match logits batch {logits.data.shape[0]}"
        )
    N, C = logits.data.shape
    if (targets_np < 0).any() or (targets_np >= C).any():
        raise ValueError(f"cross_entropy_loss: targets out of range [0, {C})")

    # Numerically stable log_softmax.
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    log_sum = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probs = shifted - log_sum
    # Pick targets and average.
    loss_data = float(-log_probs[np.arange(N), targets_np].mean())
    out = Tensor(np.float32(loss_data))
    # Gradient w.r.t. logits: (softmax - one_hot(targets)) / N
    softmax_probs = np.exp(log_probs)
    one_hot = np.zeros_like(softmax_probs)
    one_hot[np.arange(N), targets_np] = 1.0
    grad_logits = (softmax_probs - one_hot) / N
    return _build_ctx(out, (logits,), lambda g: (g.data * grad_logits,))


def nll_loss(log_probs: Tensor, targets) -> Tensor:
    """Negative log-likelihood: -mean(log_probs[range(N), targets]).

    Expects \`log_probs\` to be the output of log_softmax already.
    Prefer \`cross_entropy_loss\` directly on logits — it's more stable.
    """
    if isinstance(targets, Tensor):
        targets_np = targets.data.astype(np.int64)
    else:
        targets_np = np.asarray(targets, dtype=np.int64)
    if log_probs.data.ndim != 2:
        raise ValueError(f"nll_loss: log_probs must be 2D (N, C), got {log_probs.data.ndim}D")
    N, C = log_probs.data.shape
    if (targets_np < 0).any() or (targets_np >= C).any():
        raise ValueError(f"nll_loss: targets out of range [0, {C})")
    loss_data = float(-log_probs.data[np.arange(N), targets_np].mean())
    out = Tensor(np.float32(loss_data))
    one_hot = np.zeros_like(log_probs.data)
    one_hot[np.arange(N), targets_np] = 1.0
    grad_log_probs = -one_hot / N
    return _build_ctx(out, (log_probs,), lambda g: (g.data * grad_log_probs,))
`;
