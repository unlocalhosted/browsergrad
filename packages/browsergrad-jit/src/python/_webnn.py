"""browsergrad_jit._webnn — experimental WebNN spike (PRD-011).

INTERNAL. Public surface: `bg.experimental.webnn.{is_available, matmul}`.

Per the DL/GPU review's verdict: full WebNN backend tier defers until
Chrome WebNN GA (~2027) and a meaningful user fraction. But the *spike*
— prove the Pyodide → JS → WebNN bridge works for one op — fits in a
day and unblocks PRD-019's future "NPU dispatch" story.

This module:
  * Detects `navigator.ml` via Pyodide's `js` import.
  * Constructs OP_CUSTOM("webnn_matmul") UOps that the WebGPU realizer
    bridge can dispatch via WebNN if available, or refuse otherwise.
  * Lives behind `bg.experimental` to signal instability.

What this is NOT:
  * A full WebNN backend with op partitioner, tier selector, fallback
    machinery (the deferred PRD-011 scope).
  * A WGSL replacement — WGSL stays the default GPU path.

When Chrome ships WebNN to stable and NPU access fraction passes ~30%,
this spike becomes the seed of the real PRD-011 — same surface, more
ops, real tier selection.
"""

from __future__ import annotations
from typing import Any, Optional

from ._ir import UOp, OP_CUSTOM
from ._errors import JitNotImplementedError


def is_available() -> bool:
    """Report whether navigator.ml exists in the runtime. Returns False
    outside Pyodide (e.g. in Node-based tests) — the import will succeed
    but the attribute won't be present."""
    try:
        import js  # type: ignore[import-not-found]
    except ImportError:
        return False
    nav = getattr(js, "navigator", None)
    if nav is None:
        return False
    return hasattr(nav, "ml") and nav.ml is not None


def matmul(a: Any, b: Any) -> Any:
    """Build a CUSTOM(webnn_matmul) UOp. Realize via bg.realize_webgpu;
    the bridge dispatches through WebNN if the registered bridge has
    a webnn_matmul method, otherwise falls back to standard matmul.

    Shapes: 2-D only in v0. f32 only.

    This is the spike entry point — when the bridge's webnn_matmul
    method materialises, we get NPU dispatch on supported devices.
    """
    from ._tensor_proxy import TensorProxy
    if not isinstance(a, TensorProxy) or not isinstance(b, TensorProxy):
        raise TypeError(
            "bg.experimental.webnn.matmul: both inputs must be TensorProxy"
        )
    if a.ndim != 2 or b.ndim != 2:
        raise JitNotImplementedError(
            f"bg.experimental.webnn.matmul: 2-D only in v0 (got {a.shape} @ "
            f"{b.shape}). Batched matmul lands when the full PRD-011 tier "
            f"lands — after Chrome WebNN GA."
        )
    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        from ._errors import ShapeError
        raise ShapeError(
            f"webnn_matmul: inner dims don't match: {a.shape} @ {b.shape}"
        )
    arg = {
        "op": "webnn_matmul",
        "m": int(M),
        "k": int(K),
        "n": int(N),
    }
    uop = UOp(op=OP_CUSTOM, inputs=(a._uop, b._uop),
              shape=(M, N), dtype=a.dtype, arg=arg)
    return TensorProxy(uop, session=a._get_session(), requires_grad=False)


__all__ = ["is_available", "matmul"]
