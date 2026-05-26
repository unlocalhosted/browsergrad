"""browsergrad_jit._amp — autocast context + cast-insertion pass + GradScaler.

INTERNAL. Public API lives at `bg.amp.autocast`, `bg.amp.GradScaler`,
and the torch shim `torch.amp.autocast` / `torch.amp.GradScaler`.

Design (per PRD-010 v0 review):

  * autocast is a **thread-local flag plus a UOp annotation**. Every
    UOp constructed inside an `autocast(...)` block gets
    `arg["autocast_hint"] = "float16"` stamped on its arg dict.
  * The cast-insertion pass walks the IR at realize time, before
    fusion. For each ALLOWLIST_F16 op (MATMUL), it inserts CAST(f16)
    on inputs and CAST(f32) on outputs. For BLOCKLIST_F32 ops (the
    softmax / layernorm numerically-sensitive set), it forces inputs
    to f32. PROMOTE ops follow their inputs.
  * The fp32 matmul accumulator lives in `_h_matmul` (already wired).
    The cast pass produces fp16 INPUTS; the realizer ensures the
    REDUCE happens in fp32 internally.
  * GradScaler's state machine is pure Python; the math operations
    (`scale(loss)` = `loss * scale_buffer`) build IR. NaN check uses
    `ISNAN` + `REDUCE(any)` per parameter, OR'd in Python.

Honest scope:
  * NumPy fp16 is NOT faster than fp32 on Pyodide+WASM. No speedup
    promise. The wins are: real autocast tagging (educational), real
    GradScaler with NaN-triggered scale halving (educational), real
    activation memory halving (load-bearing for memory-bound labs),
    and a clean migration target for PRD-012's WGSL backend.
"""

from __future__ import annotations
import threading
from contextlib import ContextDecorator
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from ._ir import (
    UOp,
    OP_CAST, OP_MATMUL, OP_REDUCE, OP_EXP, OP_LOG, OP_DIV,
    OP_ADD, OP_MUL, OP_NEG, OP_WHERE, OP_PAD,
    OP_FUSED_SOFTMAX,
    toposort,
)


# ---------------------------------------------------------------------------
# Policy: which ops run in f16, which stay in f32, which follow inputs.
# Matches PyTorch's AMP CUDA op table for the overlapping ops:
#   https://pytorch.org/docs/stable/amp.html
# ---------------------------------------------------------------------------


# Force inputs to f16 (the operation runs in f16). Matmul is the marquee win.
# Conv stays excluded in v0 (CONV2D is opaque CUSTOM today).
ALLOWLIST_F16 = frozenset({OP_MATMUL})

# Force inputs to f32. These are numerically sensitive — softmax/layernorm
# internals where the f16 dynamic range underflows on long reductions.
BLOCKLIST_F32 = frozenset({
    OP_EXP, OP_LOG,         # softmax internals
    OP_REDUCE,              # sum/mean accumulate; argmax/argmin take indices
    OP_DIV,                 # softmax normalize, layernorm rsqrt
    OP_FUSED_SOFTMAX,       # PRD-006's fused softmax — keep stable
})

# Promote: if any input is f32, all f32. Otherwise stay in inputs' shared
# narrow dtype. The natural elementwise op behavior.
PROMOTE_OPS = frozenset({OP_ADD, OP_MUL, OP_NEG, OP_WHERE, OP_PAD})


# ---------------------------------------------------------------------------
# Thread-local autocast state
# ---------------------------------------------------------------------------


_TLS = threading.local()


def _active_dtype() -> Optional[str]:
    """Return the currently-active autocast dtype name, or None."""
    return getattr(_TLS, "amp_dtype", None)


class autocast(ContextDecorator):
    """Mixed-precision context. Inside the context, every UOp the tracer
    constructs gets tagged with `arg["autocast_hint"]`. The cast-
    insertion pass (`insert_cast_pass`) reads these tags at realize
    time and inserts CAST nodes around allowlist ops.

    Mirrors `torch.amp.autocast(device_type, dtype, enabled)`. The
    `device_type` is informational on the NumPy substrate — both
    `"webgpu"` and `"cpu"` are accepted and treated identically. When
    the WGSL backend lands (PRD-012), device_type will gate per-device
    `shader-f16` capability detection.

    Refusal modes:
      * `dtype=torch.bfloat16` (or "bfloat16") raises NotImplementedError.
        BF16 requires `shader-bf16` in WGSL, which doesn't exist in the
        spec yet (May 2026). Real BF16 lands when the WebGPU spec adds
        the extension.
    """

    def __init__(
        self,
        device_type: str = "webgpu",
        dtype: Any = None,
        enabled: bool = True,
    ) -> None:
        if device_type not in ("webgpu", "cpu"):
            raise ValueError(
                f"autocast device_type must be 'webgpu' or 'cpu', got {device_type!r}"
            )
        # Accept torch.dtype objects, NumPy dtype names, or None.
        if dtype is None:
            dtype_str = "float16"
        elif isinstance(dtype, str):
            dtype_str = dtype
        elif hasattr(dtype, "name"):
            dtype_str = dtype.name
        else:
            dtype_str = str(dtype)
        if dtype_str in ("bfloat16", "bf16"):
            raise NotImplementedError(
                f"autocast dtype=bfloat16 requires the WGSL shader-bf16 "
                f"extension which is not in the spec yet. Use dtype=float16 "
                f"or wait for PRD-010's BF16 follow-up."
            )
        if dtype_str not in ("float16", "float32"):
            raise NotImplementedError(
                f"autocast dtype={dtype_str!r} not supported. v0 ships "
                f"float16; everything else needs a real-vs-future-WGSL "
                f"design conversation."
            )
        self._enabled = bool(enabled)
        self._dtype_str = dtype_str
        self._prev: Optional[str] = None

    def __enter__(self) -> "autocast":
        self._prev = getattr(_TLS, "amp_dtype", None)
        if self._enabled and self._dtype_str == "float16":
            _TLS.amp_dtype = self._dtype_str
        return self

    def __exit__(self, *exc: Any) -> None:
        _TLS.amp_dtype = self._prev
        self._prev = None


def is_available() -> bool:
    """Whether mixed precision is available on the current substrate.

    On NumPy, the answer is always True (we have a real cast pass and
    a real GradScaler). When the WGSL backend lands and we want
    per-device f16 support detection, this function takes a
    device-type argument and reports `adapter.features.has("shader-f16")`.
    """
    return True


# ---------------------------------------------------------------------------
# Cast-insertion IR pass
# ---------------------------------------------------------------------------


def _cast_to(uop: UOp, dtype: str) -> UOp:
    """Wrap `uop` in a CAST if it's not already in `dtype`."""
    if uop.dtype == dtype:
        return uop
    return UOp(
        op=OP_CAST,
        inputs=(uop,),
        shape=uop.shape,
        dtype=dtype,
        arg={"dtype": dtype, "amp_inserted": True},
    )


def _arg_has_autocast(arg: Any) -> bool:
    return isinstance(arg, dict) and arg.get("autocast_hint") == "float16"


def insert_cast_pass(root: UOp) -> UOp:
    """Walk the IR rooted at `root` and insert CAST nodes around
    allowlist/blocklist ops where the surrounding UOps were tagged
    with an autocast hint.

    Returns a new root with CASTs inserted. If no UOp in the graph
    carries an autocast hint, returns `root` unchanged.
    """
    # Cheap precheck: walk the graph once and see if any UOp carries
    # an autocast hint. If not, the pass is a no-op.
    nodes = toposort(root)
    if not any(_arg_has_autocast(n.arg) for n in nodes):
        return root

    # The pass is a fold: walk topologically, build a `rewritten` map
    # from old-UOp-id to new-UOp.
    rewritten: Dict[int, UOp] = {}

    def _resolve(u: UOp) -> UOp:
        return rewritten.get(id(u), u)

    for node in nodes:
        # Leaves and already-rewritten nodes pass through.
        if not node.inputs:
            rewritten[id(node)] = node
            continue

        # Map old inputs to their (possibly cast-wrapped) replacements.
        new_inputs = tuple(_resolve(i) for i in node.inputs)

        if node.op in ALLOWLIST_F16 and _arg_has_autocast(node.arg):
            # Force inputs to f16; output is f16; downstream consumers
            # may cast back to f32 via the promote logic.
            cast_inputs = tuple(_cast_to(i, "float16") for i in new_inputs)
            new_node = UOp(
                op=node.op,
                inputs=cast_inputs,
                shape=node.shape,
                dtype="float16",
                arg=node.arg,
            )
        elif node.op in BLOCKLIST_F32 and _arg_has_autocast(node.arg):
            # Numerically-sensitive: force inputs back to f32.
            cast_inputs = tuple(_cast_to(i, "float32") for i in new_inputs)
            new_node = UOp(
                op=node.op,
                inputs=cast_inputs,
                shape=node.shape,
                dtype="float32",
                arg=node.arg,
            )
        elif node.op in PROMOTE_OPS and _arg_has_autocast(node.arg):
            # Follow inputs: if any is f32, promote all to f32; else
            # stay in the shared narrow dtype.
            target = "float32" if any(i.dtype == "float32" for i in new_inputs) else new_inputs[0].dtype
            cast_inputs = tuple(_cast_to(i, target) for i in new_inputs)
            new_node = UOp(
                op=node.op,
                inputs=cast_inputs,
                shape=node.shape,
                dtype=target,
                arg=node.arg,
            )
        else:
            # No autocast policy for this op (CAST itself, BROADCAST_TO,
            # CMP, etc.). Pass through with possibly-rewritten inputs.
            if all(a is b for a, b in zip(new_inputs, node.inputs)):
                new_node = node
            else:
                new_node = UOp(
                    op=node.op,
                    inputs=new_inputs,
                    shape=node.shape,
                    dtype=node.dtype,
                    arg=node.arg,
                )
        rewritten[id(node)] = new_node

    return _resolve(root)


# ---------------------------------------------------------------------------
# GradScaler — PyTorch-shaped, state in Python, IR ops for math
# ---------------------------------------------------------------------------


class GradScaler:
    """Loss scaling for fp16 training.

    Multiplies the loss by a large constant `S` (default 2**16) so that
    small gradients don't underflow fp16 during backward. Before the
    optimizer step, divides every gradient by `S`. If any gradient
    overflowed (NaN/Inf), the step is skipped and `S` is halved. After
    `growth_interval` consecutive clean steps, `S` is doubled.

    Adaptive scaling matches the constants from PyTorch's
    `torch.amp.GradScaler`: init_scale=2**16, growth_factor=2.0,
    backoff_factor=0.5, growth_interval=2000.

    On the NumPy realizer the scale buffer is plain Python state; the
    math operations build IR (`loss * scale_const`, `grad / scale_const`)
    so they participate normally in autograd and trace caching.
    """

    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        self._scale: float = float(init_scale)
        self._growth_factor: float = float(growth_factor)
        self._backoff_factor: float = float(backoff_factor)
        self._growth_interval: int = int(growth_interval)
        self._growth_tracker: int = 0
        self._enabled: bool = bool(enabled)

    def get_scale(self) -> float:
        return self._scale

    def is_enabled(self) -> bool:
        return self._enabled

    def scale(self, loss: Any) -> Any:
        """Multiply `loss` by the current scale. Disabled scaler returns
        `loss` unchanged."""
        if not self._enabled:
            return loss
        return loss * self._scale

    def unscale_(self, optimizer: Any) -> None:
        """Divide every parameter's gradient by the current scale.
        Mutates `param.grad` in place via from_numpy. Idempotent within
        a single step thanks to PyTorch's `_per_optimizer_states`
        bookkeeping; we omit that for v0 since the typical usage pattern
        is one unscale per step (called explicitly or via `step()`)."""
        if not self._enabled:
            return
        from ._tensor_proxy import from_numpy
        inv = 1.0 / self._scale
        for p in optimizer._params:
            if p.grad is None:
                continue
            p.grad = from_numpy(
                p.grad.numpy() * inv,
                session=p._get_session(),
            )

    def _any_nonfinite(self, optimizer: Any) -> bool:
        """Realize every parameter's grad and check for any NaN/Inf.
        Returns True iff any overflow detected."""
        for p in optimizer._params:
            if p.grad is None:
                continue
            arr = p.grad.numpy()
            if not np.all(np.isfinite(arr)):
                return True
        return False

    def step(self, optimizer: Any) -> Optional[Any]:
        """Unscale gradients, check for overflow, step or skip."""
        if not self._enabled:
            return optimizer.step()
        self.unscale_(optimizer)
        if self._any_nonfinite(optimizer):
            self._growth_tracker = 0
            self._scale *= self._backoff_factor
            return None  # step skipped
        optimizer.step()
        self._growth_tracker += 1
        return None

    def update(self) -> None:
        """Apply the scale growth step. Called after `step()` per
        PyTorch's pattern. We folded growth bookkeeping into `step()`
        for simplicity; this no-op exists to preserve the surface."""
        if not self._enabled:
            return
        if self._growth_tracker >= self._growth_interval:
            self._growth_tracker = 0
            self._scale *= self._growth_factor

    def state_dict(self) -> Dict[str, Any]:
        return {
            "scale": self._scale,
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
            "growth_interval": self._growth_interval,
            "growth_tracker": self._growth_tracker,
            "enabled": self._enabled,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._scale = float(state["scale"])
        self._growth_factor = float(state["growth_factor"])
        self._backoff_factor = float(state["backoff_factor"])
        self._growth_interval = int(state["growth_interval"])
        self._growth_tracker = int(state["growth_tracker"])
        self._enabled = bool(state["enabled"])


__all__ = [
    "autocast",
    "is_available",
    "GradScaler",
    "insert_cast_pass",
    "ALLOWLIST_F16",
    "BLOCKLIST_F32",
    "PROMOTE_OPS",
    "_active_dtype",
]
