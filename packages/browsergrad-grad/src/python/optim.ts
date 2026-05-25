/**
 * Optimizers.
 *
 * v0: SGD with optional momentum and weight decay. Adam in v0.2.
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
        self._velocity: List[np.ndarray] = [
            np.zeros_like(p.data) for p in self.params
        ]

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
`;
