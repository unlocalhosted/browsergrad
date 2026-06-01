"""browsergrad_jit._tensor_proxy — the user-facing lazy Tensor.

INTERNAL module. Users import `TensorProxy` (and the factory functions
exposed by the public `browsergrad_jit` namespace) — not from this file.

A TensorProxy wraps:
  - a UOp (the IR node that produces this tensor)
  - a reference to the Session whose BufferTable owns the leaf buffers
  - the autograd slots (`requires_grad`, `grad`, `_ctx` for backward closures)

Construction is via factory functions (`tensor`, `zeros`, `ones`, `randn`,
`from_numpy`, `arange`) registered on the top-level `browsergrad_jit`
namespace. Direct `TensorProxy(uop)` construction is internal-only.

Realization contract — what triggers a graph walk:
  EXPLICIT: .numpy(), .tolist(), .item(), .backward(), optimizer.step()
  IMPLICIT: __bool__, __float__, __int__, __iter__
  NEVER:    .shape, .dtype, .ndim, .requires_grad, .grad (as a property),
            __repr__, len(), .data (raises), __array__ (raises),
            __setattr__ on autograd slots, arithmetic that builds new
            TensorProxies (those build IR; they don't compute values).

All arithmetic dunders are wired to small builder helpers below that
construct new UOps + new TensorProxies. Broadcasting is delegated to
NumPy's broadcast_shapes — the shape we record at IR-construction time
is the broadcast shape; if NumPy decides at realization that the inputs
aren't compatible, the realizer will surface a clear error.

Closure-based backward is the v0 plan from PRD-005. Every op that produces
a TensorProxy with requires_grad=True also registers a `_BackwardClosure`
on the result: the closure receives the upstream gradient (as a TensorProxy)
and returns the per-input gradients. The closure may reference forward
inputs by their UOp, NOT by holding ndarray references — PRD-005 critique
P0-4. Concrete arrays are looked up via the BufferTable at backward time.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ._ir import (
    UOp,
    OP_ADD, OP_MUL, OP_DIV, OP_NEG, OP_EXP, OP_LOG, OP_CMP,
    OP_MATMUL, OP_REDUCE, OP_RESHAPE, OP_PERMUTE, OP_SLICE, OP_PAD,
    OP_WHERE, OP_CAST, OP_CONST,
)
from ._errors import JitNotImplementedError, ShapeError, RealizationError

if TYPE_CHECKING:
    from ._buffer_table import BufferTable


# ---------------------------------------------------------------------------
# Backward closures.
#
# Each closure is invoked at backward time with the upstream gradient (a
# concrete np.ndarray) and the input proxies' realized values (also ndarrays).
# The closure returns one ndarray per input UOp whose corresponding input
# proxy has requires_grad=True. Inputs whose proxy doesn't require grad are
# represented by `None` to keep the index layout consistent.
#
# Keeping these as functions (not closures-with-captured-arrays) sidesteps
# the memory blow-up PRD-005 critique called out: the only state a closure
# captures is the UOps + scalar args, which are cheap.
# ---------------------------------------------------------------------------


BackwardFn = Callable[
    [np.ndarray, Tuple[np.ndarray, ...]],
    Tuple[Optional[np.ndarray], ...],
]


@dataclass(slots=True)
class _BackwardCtx:
    """Carries the inputs needed to invoke a backward closure.

    Held on a TensorProxy's `_ctx` slot iff the proxy participates in
    autograd. None on leaves and on `requires_grad=False` results.
    """
    fn: BackwardFn
    input_proxies: Tuple["TensorProxy", ...]  # original input proxies; need
                                              # their ._uop for graph walk and
                                              # requires_grad flags
    # Optional opaque payload for op-specific cache (e.g. saved indices for
    # argmax backward). Kept ndarray-light: payloads bigger than a few KB
    # should go through the BufferTable, not here.
    saved: tuple = ()


# ---------------------------------------------------------------------------
# Broadcasting helpers
# ---------------------------------------------------------------------------


def _amp_arg(arg: Any = None) -> Any:
    """Stamp the active autocast dtype onto an arg dict, if any.

    The cast-insertion pass (`_amp.insert_cast_pass`) only acts on UOps
    whose `arg["autocast_hint"]` matches the active autocast dtype, so
    we tag at construction time. When autocast is inactive this returns
    `arg` unchanged — zero overhead on the no-autocast path.

    Why centralize: every UOp builder in this file that produces an
    op the AMP policy cares about (MATMUL, ADD/MUL/DIV/NEG, EXP/LOG,
    REDUCE) calls this helper. Adding new ops to the policy means
    updating their `arg=` site here too — easy to grep for.
    """
    from . import _amp as _amp_mod
    active = _amp_mod._active_dtype()
    if active is None:
        return arg
    if arg is None:
        return {"autocast_hint": active}
    if isinstance(arg, dict):
        return {**arg, "autocast_hint": active}
    return arg


def _broadcast_shape(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    """np.broadcast_shapes returns a tuple; wrap with a friendlier error."""
    try:
        return tuple(np.broadcast_shapes(*shapes))
    except ValueError as e:
        raise ShapeError(
            f"shapes {shapes} are not broadcastable: {e}"
        ) from e


def _promote_dtype(a_dtype: str, b_dtype: str) -> str:
    """NumPy's type promotion. Documented contract: result dtype is the
    NumPy promotion of the inputs. Returns the dtype name as a string —
    that's what the IR carries."""
    out = np.promote_types(np.dtype(a_dtype), np.dtype(b_dtype))
    return out.name


def _unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Sum `grad` along broadcast-extended dimensions so it matches
    `target_shape`. Symmetric to the forward broadcast.
    """
    if grad.shape == target_shape:
        return grad
    # Strip leading extra dims by summing them out.
    extra = grad.ndim - len(target_shape)
    for _ in range(extra):
        grad = grad.sum(axis=0)
    # Strip broadcast-introduced singleton dims.
    for i, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


# ---------------------------------------------------------------------------
# TensorProxy
# ---------------------------------------------------------------------------


class TensorProxy:
    """A lazy tensor backed by a UOp graph.

    Construct via factory functions on the top-level namespace
    (`browsergrad_jit.tensor`, `.zeros`, `.ones`, `.randn`, `.from_numpy`).
    Direct construction from a UOp is supported but intended for internal
    use only.

    Shape and dtype are exposed as attributes for two reasons: (1) PyTorch's
    Tensor accepts both `t.shape` as tuple-like; (2) matching tinygrad's
    "shape is just a tuple" reduces the API drift students have to learn.
    """

    __slots__ = ("_uop", "_session", "requires_grad", "grad", "_ctx")

    def __init__(
        self,
        uop: UOp,
        session: "Optional[Any]" = None,
        requires_grad: bool = False,
        ctx: Optional[_BackwardCtx] = None,
    ) -> None:
        self._uop = uop
        # Late import to avoid a circular dependency between _tensor_proxy
        # and __init__.py (which defines Session). The session is resolved
        # lazily — None means "use default at realization time."
        self._session = session
        self.requires_grad: bool = bool(requires_grad)
        self.grad: Optional["TensorProxy"] = None
        self._ctx: Optional[_BackwardCtx] = ctx

    # ------------------------------------------------------------------
    # Metadata — never realizes
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._uop.shape

    @property
    def dtype(self) -> str:
        return self._uop.dtype

    @property
    def ndim(self) -> int:
        return len(self._uop.shape)

    def size(self, dim: Optional[int] = None) -> Any:
        if dim is None:
            return self._uop.shape
        if dim < -self.ndim or dim >= self.ndim:
            raise IndexError(
                f"size(dim={dim}) out of range for tensor of ndim={self.ndim}"
            )
        return self._uop.shape[dim]

    def numel(self) -> int:
        n = 1
        for d in self._uop.shape:
            n *= d
        return n

    @property
    def data(self) -> Any:
        raise AttributeError(
            ".data is not available on TensorProxy (lazy). "
            "Use .numpy() to realize the array, or .shape/.dtype for metadata. "
            "If you're porting code from browsergrad_grad: replace `x.data` "
            "with `x.numpy()` (one-shot realization) or with `x` itself "
            "(if downstream code can stay lazy)."
        )

    def __array__(self, dtype: Any = None) -> Any:
        raise RuntimeError(
            "Cannot convert TensorProxy to np.ndarray implicitly. "
            "Use np.asarray(tensor.numpy()) to force realization, or "
            "tensor.shape / tensor.dtype if metadata is all you need."
        )

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of a 0-dim tensor is undefined")
        return self._uop.shape[0]

    def __repr__(self) -> str:
        return (
            f"TensorProxy(shape={self._uop.shape}, dtype={self._uop.dtype!r}, "
            f"requires_grad={self.requires_grad}, op={self._uop.op})"
        )

    # ------------------------------------------------------------------
    # Realization
    # ------------------------------------------------------------------

    def _get_session(self) -> Any:
        """Resolve the session: prefer the one set at construction;
        otherwise the implicit default. Imported lazily to avoid the
        circular import between this file and __init__.py."""
        if self._session is not None:
            return self._session
        import browsergrad_jit
        return browsergrad_jit.get_default_session()

    def _realize_array(self) -> np.ndarray:
        from ._realize import realize
        sess = self._get_session()
        return realize(self._uop, sess.buffer_table)

    def numpy(self) -> np.ndarray:
        """Realize the graph and return a NumPy ndarray.

        The returned array is a copy — safe for the caller to mutate without
        affecting any in-flight computation."""
        arr = self._realize_array()
        return np.array(arr, copy=True)

    def tolist(self) -> Any:
        return self._realize_array().tolist()

    def item(self) -> Any:
        if self.ndim != 0 and self.numel() != 1:
            raise ValueError(
                f"item() only works on 0-d or 1-element tensors; this tensor "
                f"has shape {self._uop.shape}. Use .numpy() / .tolist() for "
                f"higher rank."
            )
        arr = self._realize_array()
        return arr.item()

    def peek(self, n: int = 5) -> str:
        """Realize and render the first `n` flat elements inline."""
        arr = self._realize_array()
        flat = arr.reshape(-1)[:n].tolist()
        return (
            f"TensorProxy[{self._uop.op}, shape={self.shape}, "
            f"dtype={self._uop.dtype}] first {n}: {flat}"
        )

    # ------------------------------------------------------------------
    # Python protocol methods that DO realize
    # ------------------------------------------------------------------

    def __bool__(self) -> bool:
        if self.numel() != 1:
            raise RuntimeError(
                f"Boolean value of TensorProxy with {self.numel()} elements "
                f"is ambiguous. Use .any() / .all() to reduce to a scalar "
                f"first, or .item() for 1-element tensors."
            )
        return bool(self._realize_array().item())

    def __float__(self) -> float:
        if self.numel() != 1:
            raise TypeError(
                f"only 1-element TensorProxy can be converted to float "
                f"(got {self.numel()} elements)"
            )
        return float(self._realize_array().item())

    def __int__(self) -> int:
        if self.numel() != 1:
            raise TypeError(
                f"only 1-element TensorProxy can be converted to int "
                f"(got {self.numel()} elements)"
            )
        return int(self._realize_array().item())

    def __iter__(self) -> Any:
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d tensor is undefined")
        arr = self._realize_array()
        for slice_arr in arr:
            yield from_numpy(np.asarray(slice_arr).copy(),
                             session=self._get_session())

    # ------------------------------------------------------------------
    # Arithmetic dunders — build IR, never realize
    # ------------------------------------------------------------------

    def _binop(
        self,
        other: Any,
        op_code: str,
        np_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        backward: Optional[BackwardFn] = None,
    ) -> "TensorProxy":
        rhs = _to_proxy(other, session=self._get_session())
        out_shape = _broadcast_shape(self._uop.shape, rhs._uop.shape)
        out_dtype = _promote_dtype(self._uop.dtype, rhs._uop.dtype)
        uop = UOp(
            op=op_code,
            inputs=(self._uop, rhs._uop),
            shape=out_shape,
            dtype=out_dtype,
            arg=_amp_arg(None),
        )
        requires = self.requires_grad or rhs.requires_grad
        ctx = (
            _BackwardCtx(fn=backward, input_proxies=(self, rhs))
            if requires and backward is not None
            else None
        )
        return TensorProxy(uop, session=self._get_session(),
                           requires_grad=requires, ctx=ctx)

    def __add__(self, other: Any) -> "TensorProxy":
        def _bw(dy: np.ndarray, _ins) -> Tuple[Optional[np.ndarray], ...]:
            return _unbroadcast(dy, self._uop.shape), _unbroadcast(dy, _to_proxy(other, self._get_session())._uop.shape)
        return self._binop(other, OP_ADD, np.add, backward=_bw)

    def __radd__(self, other: Any) -> "TensorProxy":
        return _to_proxy(other, self._get_session()).__add__(self)

    def __sub__(self, other: Any) -> "TensorProxy":
        return self.__add__(-_to_proxy(other, self._get_session()))

    def __rsub__(self, other: Any) -> "TensorProxy":
        return _to_proxy(other, self._get_session()).__sub__(self)

    def __mul__(self, other: Any) -> "TensorProxy":
        rhs = _to_proxy(other, self._get_session())
        def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
            a_arr, b_arr = ins
            return (
                _unbroadcast(dy * b_arr, self._uop.shape),
                _unbroadcast(dy * a_arr, rhs._uop.shape),
            )
        return self._binop(other, OP_MUL, np.multiply, backward=_bw)

    def __rmul__(self, other: Any) -> "TensorProxy":
        return _to_proxy(other, self._get_session()).__mul__(self)

    def __truediv__(self, other: Any) -> "TensorProxy":
        rhs = _to_proxy(other, self._get_session())
        def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
            a_arr, b_arr = ins
            return (
                _unbroadcast(dy / b_arr, self._uop.shape),
                _unbroadcast(-dy * a_arr / (b_arr * b_arr), rhs._uop.shape),
            )
        return self._binop(other, OP_DIV, np.divide, backward=_bw)

    def __rtruediv__(self, other: Any) -> "TensorProxy":
        return _to_proxy(other, self._get_session()).__truediv__(self)

    def __neg__(self) -> "TensorProxy":
        def _bw(dy: np.ndarray, _ins) -> Tuple[Optional[np.ndarray], ...]:
            return (-dy,)
        uop = UOp(op=OP_NEG, inputs=(self._uop,), shape=self._uop.shape,
                  dtype=self._uop.dtype, arg=_amp_arg(None))
        ctx = _BackwardCtx(fn=_bw, input_proxies=(self,)) if self.requires_grad else None
        return TensorProxy(uop, session=self._get_session(),
                           requires_grad=self.requires_grad, ctx=ctx)

    def __pow__(self, exponent: Any) -> "TensorProxy":
        """Universal MSE idiom support: `(pred - target) ** 2`.

        Integer exponents up to 8 unroll into a chain of MUL UOps —
        cheap, no transcendentals, fusion-friendly. Larger or non-int
        exponents lower to exp(log(x) * exponent), which preserves
        autograd through the existing EXP/LOG/MUL VJP rules but
        requires positive x. Negative-base non-int exponent raises.
        """
        if isinstance(exponent, int) and 0 <= exponent <= 8:
            if exponent == 0:
                # x ** 0 = 1 — return a CONST proxy of ones.
                arr = np.ones(self._uop.shape, dtype=np.dtype(self._uop.dtype))
                return from_numpy(arr, session=self._get_session())
            result = self
            for _ in range(exponent - 1):
                result = result * self
            return result
        # General case: exp(log(x) * exponent).
        if isinstance(exponent, (int, float)):
            return (self.log() * float(exponent)).exp()
        return (self.log() * exponent).exp()

    def __rpow__(self, base: Any) -> "TensorProxy":
        """`scalar ** TensorProxy` — `base ** self = exp(log(base) * self)`."""
        if not isinstance(base, (int, float)):
            return NotImplemented
        if base <= 0:
            raise ValueError(
                f"TensorProxy.__rpow__: base must be positive (got {base})"
            )
        return (self * float(np.log(base))).exp()

    def __matmul__(self, other: Any) -> "TensorProxy":
        rhs = _to_proxy(other, self._get_session())
        # Inline matmul shape inference for clarity; np.matmul agrees.
        a_shape, b_shape = self._uop.shape, rhs._uop.shape
        if len(a_shape) == 0 or len(b_shape) == 0:
            raise ShapeError(f"matmul: 0-d tensors not supported ({a_shape}, {b_shape})")
        # Standard contract: last dim of A must match second-to-last dim of B.
        if a_shape[-1] != b_shape[-2 if len(b_shape) > 1 else -1]:
            raise ShapeError(
                f"matmul: inner dimensions don't match: {a_shape} @ {b_shape}"
            )
        if len(a_shape) == 1 and len(b_shape) == 1:
            out_shape: Tuple[int, ...] = ()
        elif len(a_shape) == 1:
            out_shape = b_shape[:-2] + (b_shape[-1],)
        elif len(b_shape) == 1:
            out_shape = a_shape[:-1]
        else:
            # Batched matmul: broadcast leading dims, then (M, K) @ (K, N) → (M, N)
            lead = _broadcast_shape(a_shape[:-2], b_shape[:-2])
            out_shape = lead + (a_shape[-2], b_shape[-1])
        dtype = _promote_dtype(self._uop.dtype, rhs._uop.dtype)
        def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
            a_arr, b_arr = ins
            # Standard rules: dL/dA = dL/dy @ B.T;  dL/dB = A.T @ dL/dy
            if a_arr.ndim == 1 and b_arr.ndim == 1:
                # vector dot product
                return (dy * b_arr, dy * a_arr)
            if a_arr.ndim == 1:
                # (K,) @ (..., K, N): dy has shape (..., N); dA = dy @ B.T
                return (dy @ np.swapaxes(b_arr, -1, -2), np.outer(a_arr, dy))
            if b_arr.ndim == 1:
                # (..., M, K) @ (K,): dy has shape (..., M); dB = A.T @ dy
                return (np.expand_dims(dy, -1) * b_arr, dy @ a_arr)
            return (
                dy @ np.swapaxes(b_arr, -1, -2),
                np.swapaxes(a_arr, -1, -2) @ dy,
            )
        requires = self.requires_grad or rhs.requires_grad
        ctx = _BackwardCtx(fn=_bw, input_proxies=(self, rhs)) if requires else None
        uop = UOp(op=OP_MATMUL, inputs=(self._uop, rhs._uop),
                  shape=out_shape, dtype=dtype, arg=_amp_arg(None))
        return TensorProxy(uop, session=self._get_session(),
                           requires_grad=requires, ctx=ctx)

    def __rmatmul__(self, other: Any) -> "TensorProxy":
        return _to_proxy(other, self._get_session()).__matmul__(self)

    # Comparison dunders → CMP UOps with bool dtype, no autograd contribution.
    def _cmpop(self, other: Any, op_str: str) -> "TensorProxy":
        rhs = _to_proxy(other, self._get_session())
        out_shape = _broadcast_shape(self._uop.shape, rhs._uop.shape)
        uop = UOp(op=OP_CMP, inputs=(self._uop, rhs._uop),
                  shape=out_shape, dtype="bool", arg={"op": op_str})
        return TensorProxy(uop, session=self._get_session(), requires_grad=False)

    def __eq__(self, other: Any) -> Any:  # type: ignore[override]
        return self._cmpop(other, "eq")

    def __ne__(self, other: Any) -> Any:  # type: ignore[override]
        return self._cmpop(other, "ne")

    def __lt__(self, other: Any) -> "TensorProxy":
        return self._cmpop(other, "lt")

    def __le__(self, other: Any) -> "TensorProxy":
        return self._cmpop(other, "le")

    def __gt__(self, other: Any) -> "TensorProxy":
        return self._cmpop(other, "gt")

    def __ge__(self, other: Any) -> "TensorProxy":
        return self._cmpop(other, "ge")

    # __hash__ is required when __eq__ is overridden. Use the underlying UOp's
    # identity hash so TensorProxies are usable as dict keys for autograd-time
    # bookkeeping (gradient accumulator maps).
    def __hash__(self) -> int:  # type: ignore[override]
        return id(self)

    # ------------------------------------------------------------------
    # Unary ops, reductions, shape ops
    # ------------------------------------------------------------------

    def exp(self) -> "TensorProxy":
        uop = UOp(op=OP_EXP, inputs=(self._uop,), shape=self._uop.shape,
                  dtype=self._uop.dtype, arg=_amp_arg(None))
        def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
            (x_arr,) = ins
            return (dy * np.exp(x_arr),)
        ctx = _BackwardCtx(fn=_bw, input_proxies=(self,)) if self.requires_grad else None
        return TensorProxy(uop, session=self._get_session(),
                           requires_grad=self.requires_grad, ctx=ctx)

    def log(self) -> "TensorProxy":
        uop = UOp(op=OP_LOG, inputs=(self._uop,), shape=self._uop.shape,
                  dtype=self._uop.dtype, arg=_amp_arg(None))
        def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
            (x_arr,) = ins
            return (dy / x_arr,)
        ctx = _BackwardCtx(fn=_bw, input_proxies=(self,)) if self.requires_grad else None
        return TensorProxy(uop, session=self._get_session(),
                           requires_grad=self.requires_grad, ctx=ctx)

    def _reduce(self, op: str, axis: Any = None, keepdims: bool = False) -> "TensorProxy":
        # Normalize axis to a tuple or None.
        if axis is None:
            reduced_axes: Tuple[int, ...] = tuple(range(self.ndim))
            out_shape: Tuple[int, ...] = (1,) * self.ndim if keepdims else ()
        else:
            if isinstance(axis, int):
                axes_tuple = (axis,)
            else:
                axes_tuple = tuple(axis)
            reduced_axes = tuple(a % self.ndim for a in axes_tuple)
            out_dims = []
            for i, d in enumerate(self._uop.shape):
                if i in reduced_axes:
                    if keepdims:
                        out_dims.append(1)
                else:
                    out_dims.append(d)
            out_shape = tuple(out_dims)
        # Result dtype: same as input for sum/mean/max/min; int64 for argmax.
        if op in ("argmax", "argmin"):
            out_dtype = "int64"
        else:
            out_dtype = self._uop.dtype
        uop = UOp(
            op=OP_REDUCE,
            inputs=(self._uop,),
            shape=out_shape,
            dtype=out_dtype,
            arg=_amp_arg({"op": op, "axis": axis, "keepdims": keepdims}),
        )
        # Backward for sum/mean is broadcast-of-dy back to input shape.
        # max/min/argmax are non-differentiable on the indices; for max/min
        # we use a "distribute equally among tied positions" rule.
        if op in ("sum", "mean") and self.requires_grad:
            input_shape = self._uop.shape
            n_reduced = 1
            for a in reduced_axes:
                n_reduced *= input_shape[a] if input_shape else 1
            scale = (1.0 / n_reduced) if op == "mean" else 1.0
            def _bw(dy: np.ndarray, _ins) -> Tuple[Optional[np.ndarray], ...]:
                # Expand dy to broadcast back over the reduced axes.
                expanded = dy * scale
                if not keepdims:
                    for a in sorted(reduced_axes):
                        expanded = np.expand_dims(expanded, axis=a)
                # Broadcast to input shape via np.broadcast_to + copy
                return (np.broadcast_to(expanded, input_shape).copy(),)
            ctx = _BackwardCtx(fn=_bw, input_proxies=(self,))
        else:
            ctx = None
        return TensorProxy(uop, session=self._get_session(),
                           requires_grad=(op in ("sum", "mean")) and self.requires_grad,
                           ctx=ctx)

    def sum(self, axis: Any = None, keepdims: bool = False) -> "TensorProxy":
        return self._reduce("sum", axis=axis, keepdims=keepdims)

    def mean(self, axis: Any = None, keepdims: bool = False) -> "TensorProxy":
        return self._reduce("mean", axis=axis, keepdims=keepdims)

    def max(self, axis: Any = None, keepdims: bool = False) -> "TensorProxy":
        return self._reduce("max", axis=axis, keepdims=keepdims)

    def min(self, axis: Any = None, keepdims: bool = False) -> "TensorProxy":
        return self._reduce("min", axis=axis, keepdims=keepdims)

    def argmax(self, axis: Any = None, keepdims: bool = False) -> "TensorProxy":
        return self._reduce("argmax", axis=axis, keepdims=keepdims)

    def reshape(self, *shape: Any) -> "TensorProxy":
        # Accept reshape(3, 4) or reshape((3, 4)).
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = tuple(shape)
        # Resolve a single -1 dim by deducing it from numel.
        if new_shape.count(-1) > 1:
            raise ShapeError(f"reshape: can have at most one -1 dim, got {new_shape}")
        if -1 in new_shape:
            known = 1
            for d in new_shape:
                if d != -1:
                    known *= d
            if known == 0 or self.numel() % known != 0:
                raise ShapeError(
                    f"reshape: shape {new_shape} not compatible with numel {self.numel()}"
                )
            inferred = self.numel() // known
            new_shape = tuple(inferred if d == -1 else d for d in new_shape)
        if int(np.prod(new_shape)) != self.numel():
            raise ShapeError(
                f"reshape: shape {new_shape} has {int(np.prod(new_shape))} "
                f"elements but input has {self.numel()}"
            )
        uop = UOp(op=OP_RESHAPE, inputs=(self._uop,), shape=new_shape,
                  dtype=self._uop.dtype, arg={"new_shape": new_shape})
        old_shape = self._uop.shape
        def _bw(dy: np.ndarray, _ins) -> Tuple[Optional[np.ndarray], ...]:
            return (dy.reshape(old_shape),)
        ctx = _BackwardCtx(fn=_bw, input_proxies=(self,)) if self.requires_grad else None
        return TensorProxy(uop, session=self._get_session(),
                           requires_grad=self.requires_grad, ctx=ctx)

    def view(self, *shape: Any) -> "TensorProxy":
        return self.reshape(*shape)

    def transpose(self, dim0: int, dim1: int) -> "TensorProxy":
        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return self.permute(*axes)

    def permute(self, *axes: int) -> "TensorProxy":
        axes_t = tuple(axes)
        new_shape = tuple(self._uop.shape[a] for a in axes_t)
        uop = UOp(op=OP_PERMUTE, inputs=(self._uop,), shape=new_shape,
                  dtype=self._uop.dtype, arg={"axes": axes_t})
        # Inverse permutation for backward.
        inv = [0] * len(axes_t)
        for i, a in enumerate(axes_t):
            inv[a] = i
        inv_t = tuple(inv)
        def _bw(dy: np.ndarray, _ins) -> Tuple[Optional[np.ndarray], ...]:
            return (np.transpose(dy, axes=inv_t),)
        ctx = _BackwardCtx(fn=_bw, input_proxies=(self,)) if self.requires_grad else None
        return TensorProxy(uop, session=self._get_session(),
                           requires_grad=self.requires_grad, ctx=ctx)

    @property
    def T(self) -> "TensorProxy":
        """Reverse all dims (NumPy semantics; for 2-D this is the matrix transpose)."""
        return self.permute(*reversed(range(self.ndim)))

    def cast(self, dtype: str) -> "TensorProxy":
        if dtype == self._uop.dtype:
            return self
        uop = UOp(op=OP_CAST, inputs=(self._uop,), shape=self._uop.shape,
                  dtype=dtype, arg={"dtype": dtype})
        src_dtype = self._uop.dtype
        def _bw(dy: np.ndarray, _ins) -> Tuple[Optional[np.ndarray], ...]:
            return (dy.astype(np.dtype(src_dtype), copy=False),)
        ctx = _BackwardCtx(fn=_bw, input_proxies=(self,)) if self.requires_grad else None
        return TensorProxy(uop, session=self._get_session(),
                           requires_grad=self.requires_grad, ctx=ctx)

    def float(self) -> "TensorProxy":
        return self.cast("float32")

    def long(self) -> "TensorProxy":
        return self.cast("int64")

    def bool(self) -> "TensorProxy":
        return self.cast("bool")

    # ------------------------------------------------------------------
    # Autograd
    # ------------------------------------------------------------------

    def backward(
        self,
        grad: Optional["TensorProxy"] = None,
        loss_scale: float = 1.0,
    ) -> None:
        """Compute gradients via reverse-topological walk of the graph.

        Populates `.grad` on every leaf TensorProxy that has
        `requires_grad=True`. Per PyTorch semantics, gradients accumulate
        across calls; reset via `param.grad = None` (or `optimizer.zero_grad()`).

        For scalar outputs, `grad` defaults to ones-like. For non-scalar
        outputs the caller must pass an explicit `grad` of matching shape.

        `loss_scale` multiplies the seed gradient. PRD-010's GradScaler
        passes a large value (e.g. 2**16) to keep fp16 backward in the
        representable range; default 1.0 is a no-op.
        """
        if grad is None:
            if self.ndim != 0 and self.numel() != 1:
                raise RuntimeError(
                    f"backward() called on non-scalar output of shape "
                    f"{self.shape}; pass an explicit `grad` argument of "
                    f"matching shape."
                )
            grad_arr = np.full(
                self._uop.shape,
                float(loss_scale),
                dtype=np.dtype(self._uop.dtype),
            )
        else:
            grad_arr = grad._realize_array() * float(loss_scale)

        # 1. Collect every proxy on the autograd chain rooted at self.
        #    `proxy_by_uop_id` lets the reverse walk find a proxy by UOp.
        #    `proxy_by_id` lets the final accumulation step resolve a proxy
        #    Python id back to the proxy object — without it we'd have to
        #    linear-scan the values dict for every grad entry, which is
        #    quadratic on large graphs.
        proxy_by_uop_id: dict[int, "TensorProxy"] = {}
        _collect_proxies(self, proxy_by_uop_id)
        proxy_by_id: dict[int, "TensorProxy"] = {
            id(p): p for p in proxy_by_uop_id.values()
        }

        # 2. Decide: symbolic or closure path?
        #    Symbolic path requires every non-leaf proxy on the chain to
        #    have a registered VJP rule. If even one is missing, fall back
        #    to the legacy closure path (which has been validated since
        #    PRD-005 and stays the safety net while PRD-007 lands).
        from ._ir import toposort
        from ._realize import realize
        from . import _fusion_config, _vjp
        order = toposort(self._uop)
        sess = self._get_session()

        symbolic_viable = True
        for proxy in proxy_by_uop_id.values():
            if proxy._ctx is None:
                continue
            if _vjp.get_rule(proxy._uop.op) is None:
                symbolic_viable = False
                break

        # 3. Build gradients via the chosen path.
        if symbolic_viable:
            # SYMBOLIC PATH (PRD-007 W2).
            # Build a parallel "gradient UOp" graph: for each forward UOp,
            # record the UOp that represents its accumulated gradient.
            # Then realize the gradients in one shot — fusion stays ON
            # because the closures aren't reading by-id from the value
            # cache; only the leaf gradient buffers matter.
            from ._ir import buffer as _buffer_uop, load as _load_uop

            # Seed: the loss's gradient UOp is grad_arr loaded via a BUFFER.
            seed_bid = sess.buffer_table.new_buffer(grad_arr.astype(
                np.dtype(self._uop.dtype), copy=False))
            seed_buf = _buffer_uop(seed_bid, self._uop.shape, self._uop.dtype)
            seed_uop = _load_uop(seed_buf)

            grad_uops: dict[int, UOp] = {id(self): seed_uop}

            for node in reversed(order):
                proxy = proxy_by_uop_id.get(id(node))
                if proxy is None or proxy._ctx is None:
                    continue
                if id(proxy) not in grad_uops:
                    continue
                dy_uop = grad_uops[id(proxy)]
                rule = _vjp.get_rule(proxy._uop.op)
                # rule is non-None by the viability check above.
                input_uops = tuple(inp._uop for inp in proxy._ctx.input_proxies)
                emitted = rule(proxy._uop, input_uops, dy_uop)
                # Per-input UOp may be None (non-differentiable arm of
                # WHERE/CMP, etc.) — but no W1 rule emits None in active
                # use; the dispatcher is robust to it for future rules.
                for inp_proxy, gnode in zip(proxy._ctx.input_proxies, emitted):
                    if gnode is None:
                        continue
                    if not inp_proxy.requires_grad and inp_proxy._ctx is None:
                        continue
                    pid = id(inp_proxy)
                    if pid in grad_uops:
                        # Accumulate via ADD UOp.
                        existing = grad_uops[pid]
                        grad_uops[pid] = UOp(
                            op="ADD",
                            inputs=(existing, gnode),
                            shape=existing.shape,
                            dtype=existing.dtype,
                            arg={"vjp_of": proxy._uop, "accumulator": True},
                        )
                    else:
                        grad_uops[pid] = gnode

            # Realize each leaf gradient UOp once. Fusion stays ON — the
            # value cache inside realize() is a fresh dict each call, so
            # the only outputs anyone reads by-id are the realized leaf
            # gradient ndarrays we return here.
            #
            # Before realizing, run the gradient checkpoint rewrite: any
            # gradient-graph reference to a UOp marked as "interior to a
            # checkpoint region" gets replaced with a freshly-cloned
            # forward subgraph rooted at the region's anchors. The
            # cloned UOps re-realize from anchors during this backward
            # realize() call; the original forward intermediates are
            # never re-touched (they've been GC'd by the time we get
            # here in any non-toy graph).
            from . import _checkpoint
            grads: dict[int, np.ndarray] = {}
            for pid, gnode in grad_uops.items():
                proxy = proxy_by_id.get(pid)
                if proxy is None or not proxy.requires_grad:
                    continue
                if proxy._ctx is not None:
                    continue  # only leaves write into .grad
                if _checkpoint.has_any_region():
                    gnode = _checkpoint.apply_checkpoint_rewrite(gnode)
                grads[pid] = realize(gnode, sess.buffer_table)
        else:
            # CLOSURE PATH (PRD-005 legacy, kept as safety net).
            # Fusion is disabled here because backward closures read
            # intermediate values by id from the local value_cache; fusion
            # would absorb those intermediates into FUSED_* nodes whose
            # outputs don't carry the same ids.
            value_cache: dict[int, np.ndarray] = {}
            was_enabled = _fusion_config.is_enabled()
            _fusion_config.use_fusion(False)
            try:
                for node in order:
                    value_cache[id(node)] = realize(node, sess.buffer_table)
            finally:
                _fusion_config.use_fusion(was_enabled)

            grads = {id(self): grad_arr}
            for node in reversed(order):
                proxy = proxy_by_uop_id.get(id(node))
                if proxy is None or proxy._ctx is None:
                    continue
                if id(proxy) not in grads:
                    continue
                dy = grads[id(proxy)]
                ctx = proxy._ctx
                input_arrays = tuple(
                    value_cache[id(inp_proxy._uop)] for inp_proxy in ctx.input_proxies
                )
                partials = ctx.fn(dy, input_arrays)
                for inp_proxy, partial in zip(ctx.input_proxies, partials):
                    if partial is None:
                        continue
                    if not inp_proxy.requires_grad and inp_proxy._ctx is None:
                        continue
                    pid = id(inp_proxy)
                    grads[pid] = (grads[pid] + partial) if pid in grads else partial

        # 4. Stage the accumulated gradients onto leaf parameters.
        for pid, g in grads.items():
            proxy = proxy_by_id.get(pid)
            if proxy is None or not proxy.requires_grad or proxy._ctx is not None:
                continue
            if proxy.grad is None:
                proxy.grad = from_numpy(g.copy(), session=proxy._get_session())
            else:
                proxy.grad = from_numpy(
                    proxy.grad._realize_array() + g,
                    session=proxy._get_session(),
                )


def _collect_proxies(root: "TensorProxy", out: dict) -> None:
    """Walk the proxy tree, recording one proxy per UOp.

    Uses a separate `visited` set keyed on `id(proxy)` so we don't conflate
    "this UOp already has a proxy entry" with "we've already walked this
    proxy." The earlier version reused the `out` map for both jobs, which
    silently skipped the root proxy when callers pre-seeded the map.
    """
    stack: list["TensorProxy"] = [root]
    visited: set[int] = set()
    while stack:
        p = stack.pop()
        if id(p) in visited:
            continue
        visited.add(id(p))
        out.setdefault(id(p._uop), p)
        if p._ctx is not None:
            for child in p._ctx.input_proxies:
                if id(child) not in visited:
                    stack.append(child)


# ---------------------------------------------------------------------------
# Factory functions — public via the top-level namespace
# ---------------------------------------------------------------------------


def _to_proxy(value: Any, session: Any) -> "TensorProxy":
    """Coerce a Python scalar / NumPy array / TensorProxy into a TensorProxy
    bound to `session`. Used by the arithmetic dunders for the rhs operand."""
    if isinstance(value, TensorProxy):
        return value
    if isinstance(value, (int, float, bool)):
        # Scalar — wrap as a CONST UOp. Dtype follows NumPy's rules.
        np_val = np.asarray(value)
        uop = UOp(op=OP_CONST, inputs=(), shape=(), dtype=np_val.dtype.name,
                  arg={"value": value})
        return TensorProxy(uop, session=session)
    if isinstance(value, np.ndarray):
        return from_numpy(value, session=session)
    if isinstance(value, (list, tuple)):
        return from_numpy(np.asarray(value), session=session)
    raise TypeError(
        f"cannot coerce {type(value).__name__} to TensorProxy"
    )


def from_numpy(
    array: Any,
    requires_grad: bool = False,
    session: Any = None,
) -> "TensorProxy":
    """Wrap an existing NumPy array as a TensorProxy backed by a fresh
    BUFFER+LOAD pair in the session's BufferTable.

    The array is stored by reference (no copy) — callers must not mutate
    it after registration. Use `array.copy()` if you need independence.

    UX convenience (from end-to-end harness feedback): accepts an
    existing TensorProxy as identity. This lets natural call sites
    like `model.weight = bg.from_numpy(restored['weight'])` work even
    when the restored value is already a proxy (e.g. from
    load_safetensors). The proxy's session is preserved; requires_grad
    can be flipped True via this path (mirror torch.nn.Parameter idiom).
    """
    if isinstance(array, TensorProxy):
        if requires_grad and not array.requires_grad:
            # Re-wrap with the requested flag — the underlying UOp stays.
            return TensorProxy(
                array._uop,
                session=array._session if session is None else session,
                requires_grad=True,
            )
        return array
    if session is None:
        import browsergrad_jit
        session = browsergrad_jit.get_default_session()
    bid = session.buffer_table.new_buffer(array)
    from ._ir import buffer, load
    buf_uop = buffer(bid, tuple(array.shape), array.dtype.name)
    load_uop = load(buf_uop)
    return TensorProxy(load_uop, session=session, requires_grad=requires_grad)


def tensor(
    value: Any,
    requires_grad: bool = False,
    dtype: Optional[str] = None,
    session: Any = None,
) -> "TensorProxy":
    """Factory: turn a Python value / NumPy array into a TensorProxy.

    Mirrors `torch.tensor(...)`. The default dtype is float32 for floats,
    int64 for ints, bool for booleans — matching PyTorch."""
    if isinstance(value, TensorProxy):
        # Identity / dtype-coerce path.
        out = value
        if dtype is not None and out.dtype != dtype:
            out = out.cast(dtype)
        if requires_grad and not out.requires_grad:
            # We can't just flip the flag — autograd attribution lives on the
            # original tensor. Surface this as an explicit error.
            raise ValueError(
                "tensor(): cannot set requires_grad=True on an already-built "
                "TensorProxy; build a new tensor with from_numpy(arr, "
                "requires_grad=True) instead."
            )
        return out
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(np.dtype(dtype), copy=False)
    elif arr.dtype == np.float64:
        # PyTorch default: float32, not float64.
        arr = arr.astype(np.float32, copy=False)
    return from_numpy(arr, requires_grad=requires_grad, session=session)


def zeros(*shape: int, dtype: str = "float32",
          requires_grad: bool = False, session: Any = None) -> "TensorProxy":
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape_t = tuple(shape[0])
    else:
        shape_t = shape
    return from_numpy(np.zeros(shape_t, dtype=np.dtype(dtype)),
                      requires_grad=requires_grad, session=session)


def ones(*shape: int, dtype: str = "float32",
         requires_grad: bool = False, session: Any = None) -> "TensorProxy":
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape_t = tuple(shape[0])
    else:
        shape_t = shape
    return from_numpy(np.ones(shape_t, dtype=np.dtype(dtype)),
                      requires_grad=requires_grad, session=session)


def randn(*shape: int, dtype: str = "float32",
          requires_grad: bool = False, session: Any = None,
          seed: Optional[int] = None) -> "TensorProxy":
    """Normal random tensor. By default uses a fresh PRNG each call —
    pass `seed` for determinism in tests."""
    rng = np.random.default_rng(seed)
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape_t = tuple(shape[0])
    else:
        shape_t = shape
    arr = rng.standard_normal(shape_t).astype(np.dtype(dtype), copy=False)
    return from_numpy(arr, requires_grad=requires_grad, session=session)


def arange(*args: Any, dtype: str = "int64",
           requires_grad: bool = False, session: Any = None) -> "TensorProxy":
    arr = np.arange(*args, dtype=np.dtype(dtype))
    return from_numpy(arr, requires_grad=requires_grad, session=session)


__all__ = [
    "TensorProxy",
    "from_numpy",
    "tensor",
    "zeros",
    "ones",
    "randn",
    "arange",
]
