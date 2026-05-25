/**
 * Functional ops — pure functions that take Tensor(s), return Tensor.
 *
 * v0 keeps this small: relu, tanh, sigmoid, mse_loss. Each function is
 * structurally identical to a tensor.py op (forward + backward closure
 * captured in _ctx). Anything that has trainable parameters lives in nn.ts
 * instead.
 */

export const FUNCTIONAL_PY = `
"""browsergrad_grad.functional — stateless ops with autograd."""

import numpy as np
from .tensor import Tensor, _build_ctx


def relu(x: Tensor) -> Tensor:
    out_data = np.maximum(x.data, 0.0)
    out = Tensor(out_data)
    mask = (x.data > 0).astype(np.float32)
    return _build_ctx(out, (x,), lambda g: (g.data * mask,))


def sigmoid(x: Tensor) -> Tensor:
    s = 1.0 / (1.0 + np.exp(-x.data))
    out = Tensor(s)
    # ds/dx = s * (1 - s)
    return _build_ctx(out, (x,), lambda g: (g.data * s * (1.0 - s),))


def tanh(x: Tensor) -> Tensor:
    t = np.tanh(x.data)
    out = Tensor(t)
    # dtanh/dx = 1 - t^2
    return _build_ctx(out, (x,), lambda g: (g.data * (1.0 - t * t),))


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
    # d(loss)/d(y_hat)_i = 2 * (y_hat_i - y_i) / n
    return _build_ctx(
        out,
        (y_hat,),
        lambda g: (g.data * (2.0 / n) * diff,),
    )
`;
