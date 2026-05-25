/**
 * Tensor + autograd Python source.
 *
 * v0.2 upgrades v0.1 in-place — same public API, broadcasting now works,
 * matmul accepts batch dimensions, plus a handful of new Tensor methods
 * (exp, log, transpose, reshape, view) needed by the functional / nn layers.
 *
 * Embedded as a TypeScript string so the build is a plain `tsc` — no asset
 * copy step, no bundler magic. Editors lose Python syntax highlighting
 * inside the string; in exchange the package is one bundle and trivially
 * portable.
 *
 * NOT exported from index.ts — consumers install the whole library via
 * `installGrad(target)`; they don't import individual files.
 */

export const TENSOR_PY = `
"""browsergrad_grad.tensor — Tensor + reverse-mode autograd.

v0.2: NumPy-style broadcasting in every binary op. Higher-rank matmul.
New methods: exp, log, transpose (axis-aware), reshape, view.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, List, Optional, Tuple, Union

Number = Union[int, float]
ArrayLike = Union[Number, List, Tuple, "np.ndarray", "Tensor"]


# ─── Autograd-enabled flag + no_grad context ──────────────────
#
# When _GRAD_ENABLED is False, no Tensor created inside the block will
# build a backward graph, regardless of its parents' requires_grad. The
# context manager saves and restores the previous value so nesting works.

_GRAD_ENABLED: bool = True


class no_grad:
    """Context manager that disables autograd graph building.

    Use during inference / accuracy probes to avoid building a graph that
    will never be traversed:

        with grad.no_grad():
            logits = model(x)
            preds = logits.argmax(dim=-1)
    """
    __slots__ = ("_prev",)

    def __init__(self):
        self._prev = None

    def __enter__(self):
        global _GRAD_ENABLED
        self._prev = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, exc_type, exc, tb):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev
        return False


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

    def __int__(self) -> int:
        if self.data.size != 1:
            raise TypeError(f"int() only valid on scalar tensors, got shape {self.data.shape}")
        return int(self.data.flat[0])

    def __float__(self) -> float:
        if self.data.size != 1:
            raise TypeError(f"float() only valid on scalar tensors, got shape {self.data.shape}")
        return float(self.data.flat[0])

    def __bool__(self) -> bool:
        if self.data.size != 1:
            raise TypeError(f"bool() only valid on scalar tensors, got shape {self.data.shape}")
        return bool(self.data.flat[0])

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
            raise TypeError("Tensor ** Tensor not supported; use ** with a number")
        return _pow(self, float(p))

    # ─── Reductions ────────────────────────────────────────

    def sum(self, axis=None, keepdims: bool = False):
        return _sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims: bool = False):
        return _mean(self, axis=axis, keepdims=keepdims)

    def argmax(self, dim=None) -> "Tensor":
        """Return indices of the maximum values along \`dim\`.

        \`dim=None\` (default) returns a scalar — the flat index into the
        original tensor's storage. Any other \`dim\` returns a tensor with
        that axis reduced. Result dtype is int64 (PyTorch convention).

        Not differentiable — argmax is non-smooth in input. The output
        is a regular Tensor for ergonomic .tolist() / indexing but never
        participates in autograd.
        """
        idx = np.argmax(self.data, axis=dim)
        # We force an f32 backing so the rest of the library doesn't have
        # to special-case int64 tensors. argmax results are typically used
        # as plain Python ints (item(), tolist()), not for further math.
        return Tensor(idx.astype(np.float32))

    # ─── Elementwise unary ─────────────────────────────────

    def exp(self) -> "Tensor":
        return _exp(self)

    def log(self) -> "Tensor":
        return _log(self)

    # ─── Shape ops ─────────────────────────────────────────

    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _reshape(self, tuple(shape))

    def view(self, *shape) -> "Tensor":
        """Alias for reshape — provided for PyTorch familiarity."""
        return self.reshape(*shape)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        return _transpose(self, dim0, dim1)

    @property
    def T(self) -> "Tensor":
        """For 2D tensors only — transposes the two dims. Use .transpose for higher rank."""
        if self.ndim != 2:
            raise ValueError(f"Tensor.T only defined for 2D tensors, got {self.ndim}D")
        return self.transpose(0, 1)

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


# ─── Autograd-internal helpers ─────────────────────────────

def _wrap(x: ArrayLike) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _build_ctx(out: Tensor, parents: Tuple[Tensor, ...], backward_fn: Callable):
    # Skip graph-building when inside no_grad — even if parents.requires_grad.
    if _GRAD_ENABLED and any(p.requires_grad for p in parents):
        out.requires_grad = True
        out._ctx = (parents, backward_fn)
        out._is_leaf = False
    return out


def _unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Reduce \`grad\` back to \`target_shape\` by summing over broadcasted dims.

    NumPy broadcasting matches dims right-to-left:
      target [3, 1] broadcast with [3, 4]  → output [3, 4]
      In backward: gradient w.r.t. target must sum the broadcasted dim (axis=1).
    """
    # 1. Collapse extra leading axes the broadcast added.
    extra = grad.ndim - len(target_shape)
    for _ in range(extra):
        grad = grad.sum(axis=0)
    # 2. For each remaining axis, if target dim is 1 but grad dim > 1, sum it.
    for axis, (target_dim, grad_dim) in enumerate(zip(target_shape, grad.shape)):
        if target_dim == 1 and grad_dim != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


# ─── Binary ops (broadcasting + autograd) ──────────────────

def _binop(a: ArrayLike, b: ArrayLike, fwd, grad_a, grad_b) -> Tensor:
    """Generic broadcasting binary op.

    fwd(a_data, b_data) → out_data
    grad_a(g, a_data, b_data) → gradient w.r.t. a, pre-unbroadcast
    grad_b(g, a_data, b_data) → gradient w.r.t. b, pre-unbroadcast
    """
    a_t = _wrap(a)
    b_t = _wrap(b)
    out = Tensor(fwd(a_t.data, b_t.data))
    a_shape, b_shape = a_t.data.shape, b_t.data.shape
    a_data, b_data = a_t.data, b_t.data
    def backward(g):
        return (
            _unbroadcast(grad_a(g.data, a_data, b_data), a_shape),
            _unbroadcast(grad_b(g.data, a_data, b_data), b_shape),
        )
    return _build_ctx(out, (a_t, b_t), backward)


def _add(a, b):
    return _binop(a, b, np.add,
                  lambda g, _a, _b: g,
                  lambda g, _a, _b: g)

def _sub(a, b):
    return _binop(a, b, np.subtract,
                  lambda g, _a, _b: g,
                  lambda g, _a, _b: -g)

def _mul(a, b):
    return _binop(a, b, np.multiply,
                  lambda g, _a, b_: g * b_,
                  lambda g, a_, _b: g * a_)

def _div(a, b):
    return _binop(a, b, np.divide,
                  lambda g, _a, b_: g / b_,
                  lambda g, a_, b_: -g * a_ / (b_ * b_))


def _matmul(a, b) -> Tensor:
    a_t = _wrap(a)
    b_t = _wrap(b)
    if a_t.data.ndim < 2 or b_t.data.ndim < 2:
        raise ValueError(
            f"matmul: both inputs must be at least 2D (got {a_t.data.ndim}D and {b_t.data.ndim}D)"
        )
    if a_t.data.shape[-1] != b_t.data.shape[-2]:
        raise ValueError(
            f"matmul: inner dimensions don't match ({a_t.data.shape[-1]} vs {b_t.data.shape[-2]})"
        )
    out = Tensor(a_t.data @ b_t.data)
    a_data, b_data = a_t.data, b_t.data
    a_shape, b_shape = a_t.data.shape, b_t.data.shape
    def backward(g):
        # @ broadcasts batch dims; we _unbroadcast back to each parent's shape.
        ga = g.data @ np.swapaxes(b_data, -1, -2)
        gb = np.swapaxes(a_data, -1, -2) @ g.data
        return (_unbroadcast(ga, a_shape), _unbroadcast(gb, b_shape))
    return _build_ctx(out, (a_t, b_t), backward)


# ─── Unary ops ─────────────────────────────────────────────

def _pow(a: Tensor, p: float) -> Tensor:
    out = Tensor(np.power(a.data, p))
    a_data = a.data
    return _build_ctx(out, (a,), lambda g: (g.data * p * np.power(a_data, p - 1),))


def _exp(a: Tensor) -> Tensor:
    exp_data = np.exp(a.data)
    out = Tensor(exp_data)
    return _build_ctx(out, (a,), lambda g: (g.data * exp_data,))


def _log(a: Tensor) -> Tensor:
    out = Tensor(np.log(a.data))
    a_data = a.data
    return _build_ctx(out, (a,), lambda g: (g.data / a_data,))


# ─── Reductions ────────────────────────────────────────────

def _sum(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    out_data = a.data.sum(axis=axis, keepdims=keepdims)
    out = Tensor(out_data)
    a_shape = a.data.shape
    # Pre-compute the shape needed to broadcast grad back to a's shape.
    # If keepdims=False, we need to insert size-1 dims at the reduced axes.
    if axis is None:
        def backward(g):
            return (np.broadcast_to(g.data, a_shape).copy(),)
    else:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        def backward(g):
            gd = g.data
            if not keepdims:
                for ax in sorted(ax % len(a_shape) for ax in axes):
                    gd = np.expand_dims(gd, ax)
            return (np.broadcast_to(gd, a_shape).copy(),)
    return _build_ctx(out, (a,), backward)


def _mean(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    out_data = a.data.mean(axis=axis, keepdims=keepdims)
    out = Tensor(out_data)
    a_shape = a.data.shape
    n = float(np.prod(a_shape) if axis is None else
              np.prod([a_shape[ax] for ax in ((axis,) if isinstance(axis, int) else axis)]))
    if axis is None:
        def backward(g):
            return (np.broadcast_to(g.data / n, a_shape).copy(),)
    else:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        def backward(g):
            gd = g.data / n
            if not keepdims:
                for ax in sorted(ax % len(a_shape) for ax in axes):
                    gd = np.expand_dims(gd, ax)
            return (np.broadcast_to(gd, a_shape).copy(),)
    return _build_ctx(out, (a,), backward)


# ─── Shape ops ─────────────────────────────────────────────

def _reshape(a: Tensor, shape: tuple) -> Tensor:
    out = Tensor(a.data.reshape(shape))
    a_shape = a.data.shape
    return _build_ctx(out, (a,), lambda g: (g.data.reshape(a_shape).copy(),))


def _transpose(a: Tensor, dim0: int, dim1: int) -> Tensor:
    out_data = np.swapaxes(a.data, dim0, dim1)
    out = Tensor(out_data)
    return _build_ctx(out, (a,), lambda g: (np.swapaxes(g.data, dim0, dim1).copy(),))


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
