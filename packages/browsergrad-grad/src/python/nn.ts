/**
 * nn — neural network modules with parameter management.
 *
 * v0: Module (base class), Linear, Sequential. Enough to build a multi-layer
 * fully-connected net. Conv, BatchNorm, etc. land in v0.2.
 */

export const NN_PY = `
"""browsergrad_grad.nn — neural network building blocks."""

import math
import numpy as np
from typing import Iterator, List
from .tensor import Tensor


class Module:
    """Base class for everything with learnable parameters.

    Subclasses should:
      1. Assign Tensors with requires_grad=True as attributes (or wrap them
         in Module attributes, which Module discovers transitively).
      2. Override .forward(*args, **kwargs).

    The default __call__ delegates to forward.
    """

    def __init__(self):
        self._modules: dict[str, "Module"] = {}
        self._parameters: dict[str, Tensor] = {}

    def __setattr__(self, name, value):
        if isinstance(value, Tensor) and value.requires_grad:
            # Track as a parameter automatically when it's a Tensor with requires_grad.
            self.__dict__.setdefault("_parameters", {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        object.__setattr__(self, name, value)

    def parameters(self) -> Iterator[Tensor]:
        """Yield every parameter Tensor in this module and its submodules."""
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()"
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    """y = x @ W^T + b (PyTorch convention).

    W has shape (out_features, in_features); b has shape (out_features,).
    Initialized via Kaiming uniform on W and zeros on b, matching torch.nn.Linear.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Kaiming uniform: U(-bound, bound), bound = 1/sqrt(in_features)
        bound = 1.0 / math.sqrt(in_features)
        rng = np.random.default_rng()
        W_data = rng.uniform(-bound, bound, size=(out_features, in_features)).astype(np.float32)
        self.weight = Tensor(W_data, requires_grad=True)
        if bias:
            b_data = np.zeros(out_features, dtype=np.float32)
            self.bias = Tensor(b_data, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., in_features) — v0 supports 2D inputs only.
        if x.data.ndim != 2:
            raise ValueError(
                f"Linear: v0 expects 2D input (batch, features); got shape {x.data.shape}"
            )
        if x.data.shape[1] != self.in_features:
            raise ValueError(
                f"Linear: input feature dim {x.data.shape[1]} ≠ in_features {self.in_features}"
            )
        out = x @ self.weight.T_tensor()
        if self.bias is not None:
            out = out + self.bias.broadcast_to_rows(out.data.shape[0])
        return out

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


class Sequential(Module):
    """Compose modules in sequence: out = mₙ(...m₂(m₁(x))...)."""

    def __init__(self, *modules: Module):
        super().__init__()
        for i, m in enumerate(modules):
            setattr(self, f"_seq_{i}", m)
        self._n = len(modules)

    def forward(self, x):
        for i in range(self._n):
            x = getattr(self, f"_seq_{i}")(x)
        return x

    def __repr__(self):
        parts = [repr(getattr(self, f"_seq_{i}")) for i in range(self._n)]
        return "Sequential(\\n  " + ",\\n  ".join(parts) + "\\n)"


# ─── Tensor extensions used by Linear ──────────────────────
#
# Defined here rather than tensor.py so the core tensor file stays focused
# on autograd primitives. These are convenience methods, not new ops in the
# graph sense — they wrap existing ops.

def _T_tensor(self: Tensor) -> Tensor:
    """Return a transposed view as a new Tensor in the graph (for matmul backward)."""
    from .tensor import _build_ctx as _bc
    out = Tensor(self.data.T)
    return _bc(out, (self,), lambda g: (g.data.T,))


def _broadcast_to_rows(self: Tensor, n_rows: int) -> Tensor:
    """Tile a 1D bias vector into a 2D shape (n_rows, len(bias))."""
    from .tensor import _build_ctx as _bc
    if self.data.ndim != 1:
        raise ValueError(f"broadcast_to_rows: expected 1D, got {self.data.ndim}D")
    out_data = np.broadcast_to(self.data[None, :], (n_rows, self.data.shape[0])).copy()
    out = Tensor(out_data)
    # backward: sum gradients over rows
    return _bc(out, (self,), lambda g: (g.data.sum(axis=0),))


Tensor.T_tensor = _T_tensor
Tensor.broadcast_to_rows = _broadcast_to_rows
`;
