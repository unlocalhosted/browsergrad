"""browsergrad_jit._tensor_proxy — the user-facing lazy Tensor.

INTERNAL. Users import `TensorProxy` (and its convenience factories) from
the top-level `browsergrad_jit` namespace, not from `_tensor_proxy`.

A `TensorProxy` is a thin wrapper around a `UOp` plus two autograd slots
(`requires_grad`, `grad`). It carries shape and dtype without holding any
realized data. Realization is explicit — callers invoke `.numpy()`,
`.tolist()`, `.item()`, `.backward()`, or trip one of the Python protocol
methods documented below.

Design contract:

  * Metadata access never realizes. `.shape`, `.dtype`, `.requires_grad`,
    `.grad` (if set), `__repr__`, `len(t)` are all O(1) graph lookups.
  * Explicit realization triggers: `.numpy()`, `.tolist()`, `.item()`,
    `.backward()`, optimizer reads of `.grad`. Each goes through
    `_realize()` (added in Week 3) which builds the value table from the
    BufferTable in a single topological walk.
  * Implicit realization triggers (Python protocols): `__bool__`,
    `__float__`, `__int__`, `__iter__`. These exist so `if loss > 0.1:`
    and `for x in tensor:` do the right thing in lab code rather than
    silently hanging or returning a proxy.
  * Two protocol methods explicitly REFUSE to realize: `__array__` and
    `.data`. Both would silently work but they mask the "this is a lazy
    proxy" mental model and lead to surprising behavior. Force the user
    to be explicit with `.numpy()`.

This file ships in PRD-005 Week 1 as a stub: the metadata properties
work; realization triggers raise `JitNotImplementedError` until the
realizer lands in Week 3. The arithmetic dunders land in Weeks 4-5.
"""

from __future__ import annotations
from typing import Any, Optional, Tuple

from ._ir import UOp
from ._errors import JitNotImplementedError


class TensorProxy:
    """A lazy tensor backed by a UOp graph.

    Construct via the factory functions (`browsergrad_jit.tensor`,
    `.zeros`, `.ones`, `.randn`, `.from_numpy`) — direct construction
    from a UOp is supported but intended for internal use only.

    Shape and dtype are exposed as attributes (not properties returning
    PyTorch's `torch.Size`) for two reasons: (1) PyTorch's own surface
    accepts both `t.shape` as tuple-like; (2) matching tinygrad's
    "shape is just a tuple" convention reduces the API drift students
    have to learn.
    """

    __slots__ = ("_uop", "requires_grad", "grad")

    def __init__(self, uop: UOp, requires_grad: bool = False) -> None:
        self._uop = uop
        self.requires_grad: bool = bool(requires_grad)
        self.grad: Optional["TensorProxy"] = None

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
        """PyTorch's `Tensor.size()` — returns shape tuple if dim is None,
        otherwise the integer length along `dim`. Never realizes."""
        if dim is None:
            return self._uop.shape
        if dim < -self.ndim or dim >= self.ndim:
            raise IndexError(
                f"size(dim={dim}) out of range for tensor of ndim={self.ndim}"
            )
        return self._uop.shape[dim]

    def numel(self) -> int:
        """Total element count. Never realizes."""
        n = 1
        for d in self._uop.shape:
            n *= d
        return n

    @property
    def data(self) -> Any:
        """`.data` is not available on TensorProxy.

        On the eager `browsergrad-grad.Tensor`, `.data` was the raw NumPy
        array. Here, there is no raw array until realization. Surfacing
        a fake `.data` would either silently realize (slow + a footgun
        inside `nn` internals) or return a proxy (confusing because user
        code expects an ndarray). Refusing forces a deliberate choice:
        use `.numpy()` to realize, or `.shape`/`.dtype` for metadata.
        """
        raise AttributeError(
            ".data is not available on TensorProxy (lazy). "
            "Use .numpy() to realize the array, or .shape/.dtype for metadata. "
            "If you're porting code from browsergrad_grad: replace `x.data` "
            "with `x.numpy()` (one-shot realization) or with `x` itself "
            "(if downstream code can stay lazy)."
        )

    def __array__(self, dtype: Any = None) -> Any:
        """Block numpy's `np.asarray(t)` from silently realizing.

        NumPy calls `__array__` on any object passed to `np.asarray`. If we
        allowed it, every `np.asarray(t)` in user code would trigger a full
        graph walk — usually unintentionally (the user expected metadata,
        not values). Raise instead. The escape hatch is the explicit
        `np.asarray(t.numpy())`.
        """
        raise RuntimeError(
            "Cannot convert TensorProxy to np.ndarray implicitly. "
            "Use np.asarray(tensor.numpy()) to force realization, or "
            "tensor.shape / tensor.dtype if metadata is all you need."
        )

    def __len__(self) -> int:
        """First-dim size. Never realizes — `len(t)` is metadata."""
        if self.ndim == 0:
            raise TypeError("len() of a 0-dim tensor is undefined")
        return self._uop.shape[0]

    def __repr__(self) -> str:
        """Show shape, dtype, requires_grad, and the root opcode.

        Never realizes. Showing the opcode is the pedagogical hook — students
        see `op=MATMUL` and `op=LOAD` and learn the IR exists. Use `.peek()`
        when you want to see values.
        """
        return (
            f"TensorProxy(shape={self._uop.shape}, dtype={self._uop.dtype!r}, "
            f"requires_grad={self.requires_grad}, op={self._uop.op})"
        )

    def peek(self, n: int = 5) -> str:
        """Realize the first `n` elements and render them inline.

        This is the debug surface for "what are the actual values in this
        tensor right now." Costs a full realization (the underlying IR
        currently has no support for partial realization — that's a
        PRD-006 fusion optimization) but is intended for interactive
        notebook use, not production. Returns a string so it composes
        naturally with `print(t.peek())`.
        """
        # We can't actually realize until the realizer ships in Week 3.
        # Surface the limitation explicitly so test fixtures can xfail it.
        raise JitNotImplementedError(
            "TensorProxy.peek() requires the realizer (PRD-005 Week 3). "
            "Track progress at docs/prd/PRD-005-jit-foundation.md §Implementation."
        )

    # ------------------------------------------------------------------
    # Realization triggers — wired up incrementally across PRD-005 weeks
    # ------------------------------------------------------------------

    def numpy(self) -> Any:
        """Realize the graph and return a NumPy ndarray.

        Triggers a full topological walk + dispatch table execution. The
        result is a fresh array, not a view into any BufferTable entry —
        callers may safely mutate it.
        """
        raise JitNotImplementedError(
            "TensorProxy.numpy() requires the realizer (PRD-005 Week 3)."
        )

    def tolist(self) -> Any:
        """Realize and return a (possibly nested) Python list."""
        raise JitNotImplementedError(
            "TensorProxy.tolist() requires the realizer (PRD-005 Week 3)."
        )

    def item(self) -> Any:
        """Realize a 0-d tensor and return the Python scalar.

        Raises `ValueError` (matching PyTorch) if the tensor is not 0-d.
        """
        if self.ndim != 0:
            raise ValueError(
                f"item() only works on 0-d tensors; this tensor has shape "
                f"{self._uop.shape}. Use .numpy()/.tolist() for higher rank."
            )
        raise JitNotImplementedError(
            "TensorProxy.item() requires the realizer (PRD-005 Week 3)."
        )

    def backward(self) -> None:
        """Build and realize the backward graph; populate `.grad` on every
        leaf TensorProxy that has `requires_grad=True`.

        See PRD-007 for the symbolic-backward design; in v0 the backward
        is closure-driven via PRD-005's `_build_ctx` pattern.
        """
        raise JitNotImplementedError(
            "TensorProxy.backward() requires the autograd path (PRD-005 Week 5)."
        )

    # ------------------------------------------------------------------
    # Python protocol methods that DO realize.
    # ------------------------------------------------------------------

    def __bool__(self) -> bool:
        """Truth-value test. Realizes the tensor; mirrors `torch.Tensor.__bool__`.

        For 0-d tensors returns the scalar's truthiness. For multi-element
        tensors raises (PyTorch does too) because `if tensor:` on a vector
        is ambiguous.
        """
        if self.numel() != 1:
            raise RuntimeError(
                f"Boolean value of TensorProxy with {self.numel()} elements is ambiguous. "
                f"Use .any() / .all() to reduce to a scalar first, or .item() for 1-element tensors."
            )
        raise JitNotImplementedError(
            "TensorProxy.__bool__() requires the realizer (PRD-005 Week 3)."
        )

    def __float__(self) -> float:
        if self.numel() != 1:
            raise TypeError(
                f"only 1-element TensorProxy can be converted to float "
                f"(got {self.numel()} elements)"
            )
        raise JitNotImplementedError(
            "TensorProxy.__float__() requires the realizer (PRD-005 Week 3)."
        )

    def __int__(self) -> int:
        if self.numel() != 1:
            raise TypeError(
                f"only 1-element TensorProxy can be converted to int "
                f"(got {self.numel()} elements)"
            )
        raise JitNotImplementedError(
            "TensorProxy.__int__() requires the realizer (PRD-005 Week 3)."
        )

    def __iter__(self) -> Any:
        """Iterate over the first axis. Realizes once and yields slices.

        Matches `torch.Tensor.__iter__`. Real implementation lives behind
        the realizer; the stub raises so misuse is caught early.
        """
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d tensor is undefined")
        raise JitNotImplementedError(
            "TensorProxy.__iter__() requires the realizer (PRD-005 Week 3)."
        )


__all__ = ["TensorProxy"]
