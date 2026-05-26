"""browsergrad_jit._optim — SGD and Adam optimizers.

INTERNAL. Users import as `browsergrad_jit.optim`.

Both optimizers follow PyTorch's `torch.optim` semantics:
  - Take an iterable of Parameters at construction.
  - `zero_grad()` resets every parameter's .grad to None.
  - `step()` reads each parameter's .grad and updates the parameter's
    underlying buffer in place via the BufferTable.

We don't fully decompose the optimizer math into IR ops in v0 — the
update is a single CUSTOM-shape NumPy walk per parameter. PRD-005's
critique calls out per-shape optimizer-trace caching as a P1 follow-up
once the IR layer supports the in-place STORE pattern cleanly.
"""

from __future__ import annotations
from typing import Iterable, List, Optional

import numpy as np

from ._tensor_proxy import TensorProxy
from ._errors import RealizationError


def _param_buffer_id(p: TensorProxy) -> str:
    """Extract the underlying BUFFER's id from a Parameter (which is a
    TensorProxy wrapping LOAD(BUFFER))."""
    uop = p._uop
    if uop.op == "LOAD" and len(uop.inputs) == 1 and uop.inputs[0].op == "BUFFER":
        return uop.inputs[0].arg
    raise RealizationError(
        f"optimizer: parameter is not a LOAD-of-BUFFER (op={uop.op}); "
        f"optimizers operate only on leaf parameters."
    )


class Optimizer:
    """Base — minimal protocol matching torch.optim.Optimizer."""

    def __init__(self, params: Iterable[TensorProxy]) -> None:
        self._params: List[TensorProxy] = list(params)
        if not self._params:
            raise ValueError("optimizer: parameter list is empty")
        for p in self._params:
            if not p.requires_grad:
                # Allow non-grad params to coexist (e.g. frozen layers), but
                # never step them. Matching PyTorch's behavior — frozen params
                # have requires_grad=False and step skips them.
                pass

    def zero_grad(self) -> None:
        for p in self._params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Standard SGD with optional momentum.

    Mirrors `torch.optim.SGD(params, lr, momentum=0)`.
    """

    def __init__(
        self,
        params: Iterable[TensorProxy],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        if lr < 0:
            raise ValueError(f"SGD: lr must be >= 0, got {lr}")
        if momentum < 0:
            raise ValueError(f"SGD: momentum must be >= 0, got {momentum}")
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Momentum buffers per parameter — indexed by Parameter identity.
        self._velocity: dict[int, np.ndarray] = {}

    def step(self) -> None:
        for p in self._params:
            if not p.requires_grad or p.grad is None:
                continue
            grad = p.grad.numpy()  # realize the gradient
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.numpy()
            if self.momentum != 0.0:
                if id(p) not in self._velocity:
                    self._velocity[id(p)] = np.zeros_like(grad)
                self._velocity[id(p)] = self.momentum * self._velocity[id(p)] + grad
                update = self._velocity[id(p)]
            else:
                update = grad
            # In-place buffer update via the session's BufferTable.
            bid = _param_buffer_id(p)
            sess = p._get_session()
            current = sess.buffer_table.get(bid)
            new_value = current - self.lr * update
            sess.buffer_table.update(bid, new_value.astype(current.dtype, copy=False))


class Adam(Optimizer):
    """Adam optimizer matching torch.optim.Adam defaults."""

    def __init__(
        self,
        params: Iterable[TensorProxy],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params)
        if lr < 0:
            raise ValueError(f"Adam: lr must be >= 0, got {lr}")
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._step = 0
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}

    def step(self) -> None:
        self._step += 1
        for p in self._params:
            if not p.requires_grad or p.grad is None:
                continue
            grad = p.grad.numpy()
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.numpy()
            if id(p) not in self._m:
                self._m[id(p)] = np.zeros_like(grad)
                self._v[id(p)] = np.zeros_like(grad)
            self._m[id(p)] = self.beta1 * self._m[id(p)] + (1 - self.beta1) * grad
            self._v[id(p)] = self.beta2 * self._v[id(p)] + (1 - self.beta2) * (grad * grad)
            m_hat = self._m[id(p)] / (1 - self.beta1 ** self._step)
            v_hat = self._v[id(p)] / (1 - self.beta2 ** self._step)
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            bid = _param_buffer_id(p)
            sess = p._get_session()
            current = sess.buffer_table.get(bid)
            new_value = current - self.lr * update
            sess.buffer_table.update(bid, new_value.astype(current.dtype, copy=False))


class AdamW(Adam):
    """Adam with decoupled weight decay (the right Adam most papers actually
    use). Matches torch.optim.AdamW."""

    def step(self) -> None:
        self._step += 1
        for p in self._params:
            if not p.requires_grad or p.grad is None:
                continue
            grad = p.grad.numpy()
            if id(p) not in self._m:
                self._m[id(p)] = np.zeros_like(grad)
                self._v[id(p)] = np.zeros_like(grad)
            self._m[id(p)] = self.beta1 * self._m[id(p)] + (1 - self.beta1) * grad
            self._v[id(p)] = self.beta2 * self._v[id(p)] + (1 - self.beta2) * (grad * grad)
            m_hat = self._m[id(p)] / (1 - self.beta1 ** self._step)
            v_hat = self._v[id(p)] / (1 - self.beta2 ** self._step)
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            bid = _param_buffer_id(p)
            sess = p._get_session()
            current = sess.buffer_table.get(bid)
            # AdamW: weight decay decoupled, applied directly to parameters.
            new_value = current - self.lr * update - self.lr * self.weight_decay * current
            sess.buffer_table.update(bid, new_value.astype(current.dtype, copy=False))


__all__ = ["Optimizer", "SGD", "Adam", "AdamW"]
