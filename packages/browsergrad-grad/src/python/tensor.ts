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

    def __array__(self, dtype=None):
        """Numpy array protocol. Lets np.asarray(tensor) and any code that
        feeds Tensors into NumPy reductions just work, without callers needing
        to reach for .data or .numpy().
        """
        if dtype is None:
            return self.data
        return self.data.astype(dtype, copy=False)

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

    # ─── Indexing ──────────────────────────────────────────

    def __getitem__(self, key):
        """Index/slice/mask/fancy-index, all routed through numpy's getitem.

        Boolean masks (np.bool_ array) and integer-array indexing produce
        a gather-style op whose backward scatters gradient back to the
        original positions via np.add.at — handles duplicate indices
        correctly. Slice indexing backward is just a slot-assign on a zero
        gradient.
        """
        return _getitem(self, key)

    # ─── Comparison ops ───────────────────────────────────
    # Return float-encoded 0/1 tensors (PyTorch returns bool but our
    # downstream ops are all f32 — float-encoded is more useful for the
    # common (pred == target).float().mean() pattern).

    def __eq__(self, other):
        return _compare(self, other, np.equal)
    def __ne__(self, other):
        return _compare(self, other, np.not_equal)
    def __lt__(self, other):
        return _compare(self, other, np.less)
    def __gt__(self, other):
        return _compare(self, other, np.greater)
    def __le__(self, other):
        return _compare(self, other, np.less_equal)
    def __ge__(self, other):
        return _compare(self, other, np.greater_equal)
    # __hash__ is removed by Python when __eq__ is overridden in a class
    # with __slots__. Re-add identity-hash so Tensors can still be put in
    # sets / dict keys (needed by some user code that caches by tensor).
    def __hash__(self):
        return id(self)

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

    def abs(self) -> "Tensor":
        return _abs(self)

    def sign(self) -> "Tensor":
        # Non-differentiable; matches PyTorch behavior.
        return Tensor(np.sign(self.data).astype(np.float32))

    def sqrt(self) -> "Tensor":
        return _pow(self, 0.5)

    def pow(self, exponent) -> "Tensor":
        return _pow(self, float(exponent))

    def clamp(self, min=None, max=None) -> "Tensor":
        """Clamp values to [min, max]. min or max may be None to leave that side open."""
        return _clamp(self, min, max)

    # alias to match PyTorch's torch.clip / Tensor.clip
    def clip(self, min=None, max=None) -> "Tensor":
        return self.clamp(min, max)

    def topk(self, k: int):
        """Return (values, indices) of the k largest elements (1-D only in v0).
        Returns a tuple of Tensors. Non-differentiable.
        """
        if self.data.ndim != 1:
            raise ValueError(f"topk: v0 supports 1-D tensors only, got {self.data.ndim}D")
        idx = np.argsort(-self.data)[:k]
        return Tensor(self.data[idx].astype(np.float32)), Tensor(idx.astype(np.float32))

    def expand(self, *shape) -> "Tensor":
        """Broadcast size-1 dims to a larger size. Returns a tensor that
        appears to have the new shape; we use np.broadcast_to + copy to
        materialize (PyTorch returns a view but our backward handles it
        either way)."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _expand(self, tuple(shape))

    def repeat(self, *sizes) -> "Tensor":
        """Tile the tensor along each dimension."""
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])
        return _repeat(self, tuple(sizes))

    def flip(self, dim: int) -> "Tensor":
        return _flip(self, int(dim))

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

    def permute(self, *dims) -> "Tensor":
        """Reorder axes via a permutation. Matches torch.Tensor.permute."""
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        return _permute(self, tuple(dims))

    def unsqueeze(self, dim: int) -> "Tensor":
        """Insert a size-1 dim at \`dim\`. Matches torch.Tensor.unsqueeze."""
        nd = self.data.ndim
        if dim < 0:
            dim = nd + 1 + dim
        new_shape = list(self.data.shape)
        new_shape.insert(dim, 1)
        return self.reshape(tuple(new_shape))

    def squeeze(self, dim=None) -> "Tensor":
        """Remove size-1 dims. dim=None removes all; dim=k removes only that one
        (and returns self unchanged if that dim isn't 1). Matches torch.Tensor.squeeze.
        """
        if dim is None:
            new_shape = tuple(d for d in self.data.shape if d != 1)
        else:
            if dim < 0:
                dim = self.data.ndim + dim
            if self.data.shape[dim] != 1:
                return self
            new_shape = tuple(d for i, d in enumerate(self.data.shape) if i != dim)
        return self.reshape(new_shape)

    def size(self, dim=None):
        """torch.Tensor.size() returns the shape; size(dim) returns the int."""
        if dim is None:
            return self.data.shape
        return int(self.data.shape[dim])

    # Device manipulation — no-op stubs (browsergrad has one notional device).
    # Return self for chaining: tensor(...).to(device).requires_grad_()-style code.

    def to(self, *args, **kwargs) -> "Tensor":
        return self

    def cpu(self) -> "Tensor":
        return self

    def cuda(self) -> "Tensor":
        return self

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


def _abs(a: Tensor) -> Tensor:
    out = Tensor(np.abs(a.data))
    a_data = a.data
    sign = np.sign(a_data).astype(np.float32)
    return _build_ctx(out, (a,), lambda g: (g.data * sign,))


def _clamp(a: Tensor, lo, hi) -> Tensor:
    out_data = a.data
    if lo is not None:
        out_data = np.maximum(out_data, float(lo))
    if hi is not None:
        out_data = np.minimum(out_data, float(hi))
    out = Tensor(out_data.astype(np.float32))
    a_data = a.data
    lo_v = float(lo) if lo is not None else None
    hi_v = float(hi) if hi is not None else None
    def backward(g):
        mask = np.ones_like(a_data, dtype=np.float32)
        if lo_v is not None:
            mask = mask * (a_data >= lo_v).astype(np.float32)
        if hi_v is not None:
            mask = mask * (a_data <= hi_v).astype(np.float32)
        return (g.data * mask,)
    return _build_ctx(out, (a,), backward)


def _expand(a: Tensor, shape: tuple) -> Tensor:
    # PyTorch lets -1 mean "keep this dim". Resolve.
    in_shape = a.data.shape
    if len(shape) < len(in_shape):
        raise ValueError(f"expand: target {shape} has fewer dims than input {in_shape}")
    # Pad input shape with 1s on the left to match output rank.
    padded = (1,) * (len(shape) - len(in_shape)) + in_shape
    resolved = []
    for i, t in enumerate(shape):
        if t == -1:
            resolved.append(padded[i])
        else:
            resolved.append(t)
    resolved = tuple(resolved)
    out_data = np.broadcast_to(a.data.reshape(padded), resolved).copy()
    out = Tensor(out_data)
    def backward(g):
        # Sum the gradient back over the broadcasted dimensions.
        grad = g.data
        for i, (orig, new) in enumerate(zip(padded, resolved)):
            if orig == 1 and new != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return (grad.reshape(in_shape),)
    return _build_ctx(out, (a,), backward)


def _repeat(a: Tensor, sizes: tuple) -> Tensor:
    # np.tile semantics: sizes is the repeat count along each dim.
    out_data = np.tile(a.data, sizes)
    out = Tensor(out_data.astype(np.float32))
    in_shape = a.data.shape
    def backward(g):
        # Inverse of tile: sum the gradient over each tile-block back to
        # the original shape. We do this by reshaping into a higher-dim
        # array where each tile is its own axis, then summing those.
        gd = g.data
        # Pad input shape with 1s on the left so it matches len(sizes).
        if len(in_shape) < len(sizes):
            padded_in = (1,) * (len(sizes) - len(in_shape)) + in_shape
        else:
            padded_in = in_shape
        # Reshape g so each axis splits into (repeat_count, orig_dim).
        new_shape = []
        for rep, orig in zip(sizes, padded_in):
            new_shape.extend([rep, orig])
        gd = gd.reshape(new_shape)
        # Sum over every "repeat" axis (the even-indexed ones).
        for i in range(len(sizes) - 1, -1, -1):
            gd = gd.sum(axis=2 * i)
        return (gd.reshape(in_shape),)
    return _build_ctx(out, (a,), backward)


def _flip(a: Tensor, dim: int) -> Tensor:
    out_data = np.flip(a.data, axis=dim).copy()
    out = Tensor(out_data)
    return _build_ctx(out, (a,), lambda g: (np.flip(g.data, axis=dim).copy(),))


def where(condition, a, b) -> Tensor:
    """Element-wise select: condition[i] non-zero → a[i] else b[i].

    Matches torch.where(cond, a, b). Backward routes gradient to whichever
    branch was selected.
    """
    cond_data = condition.data if isinstance(condition, Tensor) else np.asarray(condition)
    a_t = _wrap(a)
    b_t = _wrap(b)
    out_data = np.where(cond_data != 0, a_t.data, b_t.data).astype(np.float32)
    out = Tensor(out_data)
    mask_a = (cond_data != 0).astype(np.float32)
    mask_b = (cond_data == 0).astype(np.float32)
    def backward(g):
        return (None, g.data * mask_a, g.data * mask_b)
    return _build_ctx(out, (_wrap(condition), a_t, b_t), backward)


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


def _getitem(a: Tensor, key) -> Tensor:
    """Generic indexing. numpy handles every flavor (slice / int / int-array /
    bool-mask / tuple of any of the above). For backward we route through
    np.add.at so duplicate indices accumulate correctly.
    """
    # Normalize Tensor keys to numpy arrays
    def _to_np(k):
        if isinstance(k, Tensor):
            # Heuristic: a Tensor used as a mask/index is converted to an
            # int64 array if it looks integral, else used as a bool mask.
            arr = k.data
            if arr.dtype == np.bool_:
                return arr
            # All-integral float → int64
            if np.all(np.equal(np.mod(arr, 1), 0)):
                return arr.astype(np.int64)
            return arr
        return k

    if isinstance(key, tuple):
        norm_key = tuple(_to_np(k) for k in key)
    else:
        norm_key = _to_np(key)

    out_data = a.data[norm_key]
    # Ensure we have a Tensor (numpy may have returned a scalar)
    out_arr = np.asarray(out_data, dtype=np.float32)
    out = Tensor(out_arr)
    in_shape = a.data.shape

    def backward(g):
        grad_a = np.zeros(in_shape, dtype=np.float32)
        # np.add.at handles duplicate indices for fancy indexing AND works
        # for slice / bool-mask / int indices.
        np.add.at(grad_a, norm_key, g.data)
        return (grad_a,)

    return _build_ctx(out, (a,), backward)


def _compare(a, b, np_op) -> Tensor:
    """Element-wise comparison; returns a float-encoded 0/1 tensor.
    Non-differentiable — no _ctx attached.
    """
    a_data = a.data if isinstance(a, Tensor) else np.asarray(a, dtype=np.float32)
    b_data = b.data if isinstance(b, Tensor) else np.asarray(b, dtype=np.float32)
    out = np_op(a_data, b_data).astype(np.float32)
    return Tensor(out)


def _permute(a: Tensor, dims: tuple) -> Tensor:
    # Normalize negative dims
    nd = a.data.ndim
    norm = tuple(d if d >= 0 else nd + d for d in dims)
    out_data = np.transpose(a.data, norm)
    out = Tensor(out_data)
    # Inverse permutation for backward
    inv = [0] * len(norm)
    for i, d in enumerate(norm):
        inv[d] = i
    inv_t = tuple(inv)
    return _build_ctx(out, (a,), lambda g: (np.transpose(g.data, inv_t).copy(),))


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
    # If no explicit seed, draw from numpy's global state (which manual_seed sets).
    if seed is None:
        return Tensor(np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad)
    rng = np.random.default_rng(seed)
    return Tensor(rng.standard_normal(shape).astype(np.float32), requires_grad=requires_grad)


def from_numpy(arr) -> Tensor:
    """Wrap a numpy array as a Tensor. Matches torch.from_numpy."""
    return Tensor(arr)


def manual_seed(seed: int) -> None:
    """Seed numpy's global RNG. Matches torch.manual_seed for reproducibility."""
    np.random.seed(int(seed))


# Functional aliases for ops that are also methods/operators.
# These exist so user code matching torch.* function-form works.

def matmul(a, b) -> Tensor:
    return _matmul(a, b)


def mm(a, b) -> Tensor:
    """2-D matmul. PyTorch errors on non-2D; we don't enforce since matmul handles all ranks."""
    return _matmul(a, b)


def bmm(a, b) -> Tensor:
    """Batched matmul (batch dim broadcasts via numpy.matmul semantics)."""
    return _matmul(a, b)


def exp(x) -> Tensor:
    return _exp(_wrap(x))


def log(x) -> Tensor:
    return _log(_wrap(x))


def sum(x, axis=None, keepdims: bool = False) -> Tensor:
    return _sum(_wrap(x), axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims: bool = False) -> Tensor:
    return _mean(_wrap(x), axis=axis, keepdims=keepdims)


def argmax(x, dim=None) -> Tensor:
    return _wrap(x).argmax(dim=dim)


# ─── Multi-tensor ops ──────────────────────────────────────

def cat(tensors, dim: int = 0) -> Tensor:
    """Concatenate \`tensors\` along an existing axis.

    PyTorch behavior: all inputs must agree on every dim except \`dim\`.
    Backward splits the gradient along \`dim\` and distributes to each parent.
    """
    if len(tensors) == 0:
        raise ValueError("cat: empty tensor list")
    tensors = tuple(_wrap(t) for t in tensors)
    out_data = np.concatenate([t.data for t in tensors], axis=dim)
    out = Tensor(out_data)
    # Capture the split-point sizes so backward knows how to slice.
    sizes = [t.data.shape[dim] for t in tensors]

    def backward(g):
        # numpy.split takes split-points (cumulative), not section sizes.
        cuts = np.cumsum(sizes)[:-1].tolist()
        parts = np.split(g.data, cuts, axis=dim)
        return tuple(p.copy() for p in parts)

    return _build_ctx(out, tensors, backward)


def stack(tensors, dim: int = 0) -> Tensor:
    """Concatenate \`tensors\` along a new axis inserted at \`dim\`.

    PyTorch behavior: all inputs must have identical shape.
    """
    if len(tensors) == 0:
        raise ValueError("stack: empty tensor list")
    tensors = tuple(_wrap(t) for t in tensors)
    out_data = np.stack([t.data for t in tensors], axis=dim)
    out = Tensor(out_data)
    n = len(tensors)

    def backward(g):
        # The new axis at \`dim\` has size n; iterate it to recover per-parent grads.
        parts = []
        for i in range(n):
            # Index along the new axis to get a slice with that dim removed.
            idx = [slice(None)] * g.data.ndim
            idx[dim] = i
            parts.append(g.data[tuple(idx)].copy())
        return tuple(parts)

    return _build_ctx(out, tensors, backward)


# ─── einsum ────────────────────────────────────────────────

def einsum(equation: str, *operands: "Tensor") -> "Tensor":
    """Wrap np.einsum with autograd.

    Backward derivation: given out = einsum(eq, A, B), the gradient wrt
    operand A is einsum(eq_with_A_and_out_swapped, grad_out, B). We do this
    by rebuilding the equation string per operand.

    Supports the typical lab cases: 1 or 2 operands. Three-plus operands raise.
    """
    if not operands:
        raise ValueError("einsum: need at least one operand")
    if "->" in equation:
        in_part, out_subs = equation.split("->", 1)
        out_subs = out_subs.strip()
    else:
        in_part = equation
        # Implicit: einsum sums duplicates, output is alphabetic single-appearance subscripts.
        # For our supported cases we require explicit '->' to keep backward simple.
        raise ValueError("einsum: explicit '->' is required for autograd-able einsum")
    in_subs = [s.strip() for s in in_part.split(",")]
    if len(in_subs) != len(operands):
        raise ValueError(f"einsum: {len(in_subs)} subscript groups but {len(operands)} operands")

    arrays = [op.data for op in operands]
    out_data = np.einsum(equation, *arrays).astype(np.float32)
    out = Tensor(out_data)

    if len(operands) == 1:
        in_a = in_subs[0]
        def backward(g):
            # da = einsum(out_subs -> in_a, g)
            da = np.einsum(f"{out_subs}->{in_a}", g.data)
            return (da.astype(np.float32),)
        return _build_ctx(out, operands, backward)

    if len(operands) == 2:
        in_a, in_b = in_subs
        a_data = arrays[0]; b_data = arrays[1]
        def backward(g):
            # da = einsum("out_subs,in_b -> in_a", g, b)
            da = np.einsum(f"{out_subs},{in_b}->{in_a}", g.data, b_data)
            # db = einsum("in_a,out_subs -> in_b", a, g)
            db = np.einsum(f"{in_a},{out_subs}->{in_b}", a_data, g.data)
            return (da.astype(np.float32), db.astype(np.float32))
        return _build_ctx(out, operands, backward)

    raise NotImplementedError("einsum: more than 2 operands not supported yet")
`;
