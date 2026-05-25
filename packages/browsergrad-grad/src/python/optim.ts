/**
 * Optimizers.
 *
 * v0.2 adds Adam and AdamW alongside v0.1's SGD. Same update formulas as
 * the corresponding torch.optim classes — using these is the v0.2 way to
 * train any non-toy model.
 */

export const OPTIM_PY = `
"""browsergrad_grad.optim — gradient-descent optimizers."""

import numpy as np
from typing import Iterable, List
from .tensor import Tensor


class Optimizer:
    """Base class. Subclasses implement step()."""

    def __init__(self, params: Iterable[Tensor], lr: float):
        self.params: List[Tensor] = [p for p in params]
        if any(not isinstance(p, Tensor) or not p.requires_grad for p in self.params):
            raise ValueError("All optimizer parameters must be Tensors with requires_grad=True")
        self.lr = float(lr)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and weight decay.

    Update rule (matches PyTorch):
      g_t = grad + weight_decay * param          (if weight_decay)
      b_t = momentum * b_{t-1} + g_t              (if momentum)
      param = param - lr * (b_t if momentum else g_t)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self._velocity: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data
            if self.momentum != 0.0:
                self._velocity[i] = self.momentum * self._velocity[i] + g
                update = self._velocity[i]
            else:
                update = g
            p.data = p.data - self.lr * update

    def __repr__(self):
        return (
            f"SGD(lr={self.lr}, momentum={self.momentum}, "
            f"weight_decay={self.weight_decay}, params={len(self.params)})"
        )


class Adam(Optimizer):
    """Adam optimizer (Kingma & Ba, 2014).

    Update rule (matches torch.optim.Adam with default amsgrad=False):
      g_t        = grad + weight_decay * param                (L2 regularization)
      m_t        = beta1 * m_{t-1} + (1 - beta1) * g_t
      v_t        = beta2 * v_{t-1} + (1 - beta2) * g_t²
      m_hat      = m_t / (1 - beta1^t)
      v_hat      = v_t / (1 - beta2^t)
      param      = param - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._step_count = 0
        self._m: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]
        self._v: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self._step_count += 1
        bc1 = 1.0 - self.beta1 ** self._step_count
        bc2 = 1.0 - self.beta2 ** self._step_count
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def __repr__(self):
        return (
            f"Adam(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
            f"eps={self.eps}, weight_decay={self.weight_decay}, "
            f"params={len(self.params)})"
        )


class AdamW(Optimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2017).

    Difference from Adam: weight decay is applied directly to the parameter
    after the gradient update, NOT folded into the gradient. This is what
    transformer pre-training typically uses.

      m_t        = beta1 * m_{t-1} + (1 - beta1) * g_t
      v_t        = beta2 * v_{t-1} + (1 - beta2) * g_t²
      m_hat      = m_t / (1 - beta1^t)
      v_hat      = v_t / (1 - beta2^t)
      param      = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._step_count = 0
        self._m: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]
        self._v: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self._step_count += 1
        bc1 = 1.0 - self.beta1 ** self._step_count
        bc2 = 1.0 - self.beta2 ** self._step_count
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay != 0.0:
                p.data = p.data - self.lr * (update + self.weight_decay * p.data)
            else:
                p.data = p.data - self.lr * update

    def __repr__(self):
        return (
            f"AdamW(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
            f"eps={self.eps}, weight_decay={self.weight_decay}, "
            f"params={len(self.params)})"
        )
`;
