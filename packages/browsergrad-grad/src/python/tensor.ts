/**
 * Tensor + autograd Python source.
 *
 * Embedded as a TypeScript string so the build is a plain `tsc` — no asset
 * copy step, no bundler magic. Editors will lose Python syntax highlighting
 * inside the string; in exchange the package is one bundle and trivially
 * portable. The Python is short enough (~250 lines) that this trade is fine
 * for v0; if it grows past ~500 lines we'll switch to .py + a build step.
 *
 * NOT exported from index.ts — consumers install the whole library via
 * `installGrad(target)`; they don't import individual files.
 */

export const TENSOR_PY = `
"""browsergrad_grad.tensor — Tensor + reverse-mode autograd."""

from __future__ import annotations
import numpy as np
from typing import Callable, List, Optional, Tuple, Union

Number = Union[int, float]
ArrayLike = Union[Number, List, Tuple, "np.ndarray", "Tensor"]


class Tensor:
    """A tensor of f32 values with optional gradient tracking.

    Backed by numpy. The compute graph is built lazily through op-overloads
    (__add__, __matmul__, etc.). Calling .backward() on a scalar output
    walks the graph in reverse topological order and accumulates gradients
    into the .grad attribute of every leaf with requires_grad=True.
    """

    __slots__ = ("data", "requires_grad", "grad", "_ctx", "_is_leaf")

    def __init__(self, data: ArrayLike, requires_grad: bool = False, _ctx=None):
        if isinstance(data, Tensor):
            self.data = data.data
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float32, copy=False)
        else:
            self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None
        self._ctx = _ctx  # ((parent_tensors,), backward_fn) or None
        self._is_leaf = _ctx is None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    def detach(self) -> "Tensor":
        """Return a copy that doesn't participate in autograd."""
        return Tensor(self.data.copy(), requires_grad=False)

    def numpy(self) -> np.ndarray:
        """Return a copy of the underlying numpy array."""
        return self.data.copy()

    def tolist(self):
        return self.data.tolist()

    def item(self) -> float:
        """Return a Python float — only valid for scalar tensors."""
        if self.data.size != 1:
            raise ValueError(f"item() only valid on scalar tensors, got shape {self.data.shape}")
        return float(self.data.flat[0])

    def zero_grad(self):
        """Reset .grad to None. Called by Optimizer.zero_grad on every parameter."""
        self.grad = None

    # ─── Arithmetic ops ─────────────────────────────────────

    def __add__(self, other):
        return _add(self, other)
    __radd__ = __add__

    def __sub__(self, other):
        return _sub(self, other)
    def __rsub__(self, other):
        return _sub(other, self)

    def __mul__(self, other):
        return _mul(self, other)
    __rmul__ = __mul__

    def __truediv__(self, other):
        return _div(self, other)
    def __rtruediv__(self, other):
        return _div(other, self)

    def __neg__(self):
        return _mul(self, -1.0)

    def __matmul__(self, other):
        return _matmul(self, other)

    def __pow__(self, p):
        if not isinstance(p, (int, float)):
            raise TypeError("Tensor ** Tensor not supported in v0; use ** with a number")
        return _pow(self, float(p))

    # ─── Reductions ────────────────────────────────────────

    def sum(self):
        return _sum(self)

    def mean(self):
        return _mean(self)

    # ─── Backward ──────────────────────────────────────────

    def backward(self, grad: Optional["Tensor"] = None):
        """Compute gradients of this tensor w.r.t. all leaves with requires_grad."""
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "backward() can only be called on scalar tensors without an explicit grad"
                )
            grad = Tensor(np.ones_like(self.data))

        # Initialize this output's grad.
        if self.grad is None:
            self.grad = Tensor(grad.data.copy())
        else:
            self.grad.data = self.grad.data + grad.data

        # Topological sort.
        topo: List[Tensor] = []
        visited = set()
        def build(t: Tensor):
            if id(t) in visited:
                return
            visited.add(id(t))
            if t._ctx is not None:
                parents, _ = t._ctx
                for p in parents:
                    if p.requires_grad:
                        build(p)
            topo.append(t)
        build(self)

        # Reverse-mode propagation.
        for t in reversed(topo):
            if t._ctx is None or t.grad is None:
                continue
            parents, backward_fn = t._ctx
            grads = backward_fn(t.grad)
            for parent, g in zip(parents, grads):
                if not parent.requires_grad or g is None:
                    continue
                if parent.grad is None:
                    parent.grad = Tensor(g)
                else:
                    parent.grad.data = parent.grad.data + g


# ─── Op implementations ────────────────────────────────────
#
# Every op produces a new Tensor whose _ctx records:
#   (parents: tuple[Tensor, ...], backward_fn: Callable[[Tensor], tuple[ndarray|None, ...]])
# The backward_fn returns one gradient per parent, in the same order.
# None means "this parent didn't participate in the computation."

def _wrap(x: ArrayLike) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _build_ctx(out: Tensor, parents: Tuple[Tensor, ...], backward_fn: Callable):
    if any(p.requires_grad for p in parents):
        out.requires_grad = True
        out._ctx = (parents, backward_fn)
        out._is_leaf = False
    return out


def _check_shapes(name: str, a: Tensor, b: Tensor):
    if a.data.shape != b.data.shape:
        raise ValueError(
            f"{name}: v0 does not support tensor-tensor broadcasting "
            f"(got shapes {a.data.shape} and {b.data.shape}). "
            "Use scalar broadcasting or reshape manually."
        )


def _add(a: ArrayLike, b: ArrayLike) -> Tensor:
    if not isinstance(a, Tensor) and isinstance(b, Tensor):
        a, b = b, a  # commute so 'a' is the Tensor in the scalar path
    a_t = _wrap(a)
    if isinstance(b, (int, float)):
        out = Tensor(a_t.data + b)
        return _build_ctx(out, (a_t,), lambda g: (g.data,))
    b_t = _wrap(b)
    _check_shapes("add", a_t, b_t)
    out = Tensor(a_t.data + b_t.data)
    return _build_ctx(out, (a_t, b_t), lambda g: (g.data, g.data))


def _sub(a: ArrayLike, b: ArrayLike) -> Tensor:
    if isinstance(a, (int, float)) and isinstance(b, Tensor):
        out = Tensor(a - b.data)
        return _build_ctx(out, (b,), lambda g: (-g.data,))
    a_t = _wrap(a)
    if isinstance(b, (int, float)):
        out = Tensor(a_t.data - b)
        return _build_ctx(out, (a_t,), lambda g: (g.data,))
    b_t = _wrap(b)
    _check_shapes("sub", a_t, b_t)
    out = Tensor(a_t.data - b_t.data)
    return _build_ctx(out, (a_t, b_t), lambda g: (g.data, -g.data))


def _mul(a: ArrayLike, b: ArrayLike) -> Tensor:
    if not isinstance(a, Tensor) and isinstance(b, Tensor):
        a, b = b, a
    a_t = _wrap(a)
    if isinstance(b, (int, float)):
        c = float(b)
        out = Tensor(a_t.data * c)
        return _build_ctx(out, (a_t,), lambda g: (g.data * c,))
    b_t = _wrap(b)
    _check_shapes("mul", a_t, b_t)
    out = Tensor(a_t.data * b_t.data)
    # Capture a_data, b_data so the backward function doesn't depend on Tensor mutation.
    a_data, b_data = a_t.data, b_t.data
    return _build_ctx(out, (a_t, b_t), lambda g: (g.data * b_data, g.data * a_data))


def _div(a: ArrayLike, b: ArrayLike) -> Tensor:
    if isinstance(a, (int, float)) and isinstance(b, Tensor):
        out = Tensor(a / b.data)
        b_data = b.data
        return _build_ctx(out, (b,), lambda g: (-g.data * a / (b_data * b_data),))
    a_t = _wrap(a)
    if isinstance(b, (int, float)):
        c = float(b)
        out = Tensor(a_t.data / c)
        return _build_ctx(out, (a_t,), lambda g: (g.data / c,))
    b_t = _wrap(b)
    _check_shapes("div", a_t, b_t)
    out = Tensor(a_t.data / b_t.data)
    a_data, b_data = a_t.data, b_t.data
    return _build_ctx(
        out,
        (a_t, b_t),
        lambda g: (g.data / b_data, -g.data * a_data / (b_data * b_data)),
    )


def _matmul(a: ArrayLike, b: ArrayLike) -> Tensor:
    a_t = _wrap(a)
    b_t = _wrap(b)
    if a_t.data.ndim != 2 or b_t.data.ndim != 2:
        raise ValueError(
            f"matmul: v0 supports 2D × 2D only (got {a_t.data.ndim}D and {b_t.data.ndim}D)"
        )
    if a_t.data.shape[1] != b_t.data.shape[0]:
        raise ValueError(
            f"matmul: inner dimensions don't match ({a_t.data.shape[1]} vs {b_t.data.shape[0]})"
        )
    out = Tensor(a_t.data @ b_t.data)
    a_data, b_data = a_t.data, b_t.data
    return _build_ctx(
        out,
        (a_t, b_t),
        lambda g: (g.data @ b_data.T, a_data.T @ g.data),
    )


def _pow(a: Tensor, p: float) -> Tensor:
    out = Tensor(np.power(a.data, p))
    a_data = a.data
    return _build_ctx(out, (a,), lambda g: (g.data * p * np.power(a_data, p - 1),))


def _sum(a: Tensor) -> Tensor:
    out = Tensor(a.data.sum())
    shape = a.data.shape
    return _build_ctx(out, (a,), lambda g: (np.broadcast_to(g.data, shape).copy(),))


def _mean(a: Tensor) -> Tensor:
    out = Tensor(a.data.mean())
    shape = a.data.shape
    n = float(a.data.size)
    return _build_ctx(out, (a,), lambda g: (np.broadcast_to(g.data / n, shape).copy(),))


# ─── Convenience constructors ──────────────────────────────

def zeros(*shape, requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)


def ones(*shape, requires_grad: bool = False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)


def randn(*shape, requires_grad: bool = False, seed: Optional[int] = None) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    rng = np.random.default_rng(seed)
    return Tensor(rng.standard_normal(shape).astype(np.float32), requires_grad=requires_grad)
`;
