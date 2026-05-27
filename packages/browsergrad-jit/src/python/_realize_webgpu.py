"""browsergrad_jit._realize_webgpu — WebGPU realizer (forward-only spike).

INTERNAL. The spike scope per the PRD-012 DL/GPU review:

  * Forward-only inference. Backward stays on the NumPy realizer.
  * Operates on the IR after `_amp.insert_cast_pass` but BEFORE
    `_fusion.fuse` — we want fusion's `OP_FUSED_*` opcodes to land here
    in their fused form (one bridge call per kernel). The realizer
    invokes both passes itself, mirroring `_realize.realize()`.
  * Whitelist of supported opcodes; everything else raises
    `JitNotImplementedError` with a pointer to the NumPy fallback.

Lifecycle:
  * Seed buffers (LOAD wrapping BUFFER) get uploaded once per session
    via the `GpuBufferTable`. Subsequent realize-webgpu calls reuse
    those handles.
  * Intermediate handles live in a per-call `value_table` keyed by
    `id(uop)` (same pattern as the NumPy realizer). They are released
    after the realize call finishes — except for the root, whose
    bytes the caller materialises before we release the handle.

The realizer is sync. Production deployment relies on JSPI (Pyodide
0.27+ on Chrome 144+) to let the JS bridge's async WebGPU dispatch
appear synchronous to Python. Tests inject a NumPy-backed bridge that
is genuinely sync.

What this does NOT do:
  * No pattern matcher for transformer blocks (deferred to PRD-012a).
  * No autotune sweeps (deferred to PRD-012b).
  * No backward joint fusion (deferred indefinitely per review).
  * No CONV2D / RANDOM / SCATTER_ADD / INDEX / MASK / CMP / ISNAN.
    The realizer's whitelist is small and honest.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._ir import (
    UOp, toposort,
    OP_BUFFER, OP_LOAD, OP_CONST, OP_CAST,
    OP_ADD, OP_MUL, OP_DIV, OP_NEG, OP_EXP, OP_LOG,
    OP_MATMUL, OP_FUSED_ELEMENTWISE, OP_CUSTOM,
)
from ._errors import JitNotImplementedError, RealizationError
from ._gpu_buffer_table import GpuBufferTable


# --------------------------------------------------------------------------
# Handler signature
#
# Receives (node, value_table, gpu_buffer_table, bridge) and returns an opaque
# handle. Same shape as the NumPy realizer's Handler, but the return type is
# `Any` (a JS-proxied GPUBuffer or a mock handle) instead of np.ndarray.
# --------------------------------------------------------------------------


def _h_buffer(node: UOp, vt: dict, gbt: GpuBufferTable, br: Any,
              numpy_bt: Any) -> Any:
    """Wrap a BufferTable id as a GPU handle. Lazy-uploads from the
    NumPy BufferTable on first sight."""
    buffer_id = node.arg
    handle = gbt.get(buffer_id)
    if handle is not None:
        return handle
    # Need to upload from the NumPy-side BufferTable. BufferTable.get
    # raises BufferTableError on unknown / cross-session ids — we let
    # that propagate so cross-session contamination still surfaces here.
    arr = numpy_bt.get(buffer_id)
    handle = gbt.upload(
        buffer_id,
        arr.tobytes(),
        tuple(arr.shape),
        arr.dtype.name,
    )
    return handle


def _h_load(node: UOp, vt: dict, gbt: GpuBufferTable, br: Any,
            numpy_bt: Any) -> Any:
    """LOAD unwraps a BUFFER — the BUFFER handler already ran, so the
    handle is in the value table."""
    return vt[id(node.inputs[0])]


def _h_const(node: UOp, vt: dict, gbt: GpuBufferTable, br: Any,
             numpy_bt: Any) -> Any:
    """Upload a CONST scalar. CONSTs are small; we don't bother caching
    across realize-webgpu calls — every call gets a fresh handle. The
    real cost is the bridge cross, not the bytes."""
    value = node.arg["value"]
    arr = np.asarray(value, dtype=np.dtype(node.dtype))
    return br.upload(arr.tobytes(), tuple(arr.shape) or (1,), node.dtype)


def _h_cast(node: UOp, vt: dict, gbt: GpuBufferTable, br: Any,
            numpy_bt: Any) -> Any:
    src = vt[id(node.inputs[0])]
    return br.cast(
        src,
        node.inputs[0].dtype,
        node.dtype,
        tuple(node.shape),
    )


def _h_matmul(node: UOp, vt: dict, gbt: GpuBufferTable, br: Any,
              numpy_bt: Any) -> Any:
    a_uop, b_uop = node.inputs
    a = vt[id(a_uop)]
    b = vt[id(b_uop)]
    # 2-D only in v0; batched matmul is a follow-on (decompose to per-batch
    # bridge calls or land a batched WGSL kernel — PRD-012a's call).
    if len(a_uop.shape) != 2 or len(b_uop.shape) != 2:
        raise JitNotImplementedError(
            f"WebGPU realizer: only 2-D matmul in v0 (got A={a_uop.shape}, "
            f"B={b_uop.shape}). Fall back to bg.realize() for batched matmul "
            f"until PRD-012a lands."
        )
    m, k = a_uop.shape
    k2, n = b_uop.shape
    if k != k2:
        raise RealizationError(
            f"WebGPU matmul: inner dims mismatch: {a_uop.shape} @ {b_uop.shape}"
        )
    return br.matmul(a, b, m, k, n, node.dtype)


def _h_fused_elementwise(node: UOp, vt: dict, gbt: GpuBufferTable, br: Any,
                         numpy_bt: Any) -> Any:
    """One bridge call for the whole fused chain. The bridge implementation
    decides whether to emit one WGSL kernel per chain (production) or to
    walk the ops list via NumPy (mock)."""
    inputs = [vt[id(inp)] for inp in node.inputs]
    ops = list(node.arg["ops"])
    return br.fused_elementwise(inputs, ops, tuple(node.shape), node.dtype)


def _h_custom(node: UOp, vt: dict, gbt: GpuBufferTable, br: Any,
              numpy_bt: Any) -> Any:
    """Routes by `arg["op"]` to the corresponding bridge method.

    The opt-in surface for Flash Attention forward in v0:
        UOp(op=OP_CUSTOM, inputs=(q_uop, k_uop, v_uop) [+ mask_uop],
            shape=out_shape, dtype="float32",
            arg={"op": "flash_attention", "b": B, "h": H, "sq": Sq,
                 "sk": Sk, "d": D, "scale": 1.0/sqrt(D),
                 "has_mask": True/False})
    """
    op_name = node.arg.get("op") if isinstance(node.arg, dict) else None
    inputs = [vt[id(inp)] for inp in node.inputs]
    arg = node.arg
    if op_name == "flash_attention":
        has_mask = bool(arg.get("has_mask", False))
        if has_mask:
            q, k, v, mask = inputs
        else:
            q, k, v = inputs
            mask = None
        return br.flash_attention(
            q, k, v, mask,
            int(arg["b"]),
            int(arg["h"]),
            int(arg["sq"]),
            int(arg["sk"]),
            int(arg["d"]),
            float(arg["scale"]),
            node.dtype,
        )
    if op_name == "user":
        # User WGSL kernel (PRD-015). The Python-side registry holds the
        # WGSL source by hash; the bridge looks it up to dispatch.
        from ._custom_kernel import get_registry
        spec = get_registry().get(arg["kernel_hash"])
        if spec is None:
            raise RealizationError(
                f"WebGPU realizer: user kernel {arg['kernel_name']!r} hash "
                f"{arg['kernel_hash'][:8]!r} not found in registry. Ensure "
                f"the @custom_kernel decorator ran in this process before "
                f"realize_webgpu."
            )
        out_len = 1
        for d in arg["output_shape"]:
            out_len *= d
        if out_len == 0:
            out_len = 1
        return br.run_user_kernel(
            inputs,
            spec.wgsl,
            arg["kernel_name"],
            arg["kernel_hash"],
            tuple(arg["workgroup_size"]),
            tuple(arg["dispatch_shape"]),
            int(out_len),
            tuple(arg["output_shape"]),
            node.dtype,
        )
    raise JitNotImplementedError(
        f"WebGPU realizer: CUSTOM op {op_name!r} is not supported in "
        f"v0. Supported: 'flash_attention', 'user'. Fall back to "
        f"bg.realize() for anything else."
    )


# Dispatch table — strictly bounded. The DL/GPU review's "do not do" list
# is enforced by the keys we don't include.
_DISPATCH = {
    OP_BUFFER: _h_buffer,
    OP_LOAD: _h_load,
    OP_CONST: _h_const,
    OP_CAST: _h_cast,
    OP_MATMUL: _h_matmul,
    OP_FUSED_ELEMENTWISE: _h_fused_elementwise,
    OP_CUSTOM: _h_custom,
}


def supported_opcodes() -> frozenset:
    """Public introspection: what UOps does the WebGPU realizer handle?
    `bg.realize_webgpu` raises with a pointer to this set when it sees
    anything outside it."""
    return frozenset(_DISPATCH)


def realize_webgpu(
    root: UOp,
    *,
    numpy_buffer_table: Any,
    gpu_buffer_table: GpuBufferTable,
) -> np.ndarray:
    """Walk the IR via the WebGPU bridge and return the realized ndarray.

    Algorithm mirrors `_realize.realize()` but every handler returns a
    GPU handle. The final handle for `root` is materialised back to
    bytes → ndarray exactly once at the end. Intermediate handles are
    released as soon as their last consumer fires (a simple
    reference-count walk over the topo order).
    """
    bridge = gpu_buffer_table.bridge

    # 1. Build the toposort. The IR is already post-cast-pass + post-fusion
    #    when callers go through `bg.realize_webgpu`, which mirrors what
    #    `_realize.realize()` does at the same point.
    order = toposort(root)

    # 2. Reference-count last-use of each UOp so we can release
    #    intermediate handles as soon as they're no longer needed. The
    #    root is excluded — we materialise it before release.
    last_use: Dict[int, int] = {}
    for idx, node in enumerate(order):
        for inp in node.inputs:
            last_use[id(inp)] = idx

    # 3. Dispatch loop.
    value_table: Dict[int, Any] = {}
    intermediate_handles: List[Any] = []  # tracks what we own + must release
    for idx, node in enumerate(order):
        handler = _DISPATCH.get(node.op)
        if handler is None:
            raise JitNotImplementedError(
                f"WebGPU realizer: opcode {node.op!r} is not supported. "
                f"Supported set: {sorted(supported_opcodes())}. Fall back to "
                f"bg.realize() (NumPy) or lower this op through fusion "
                f"first."
            )
        try:
            handle = handler(node, value_table, gpu_buffer_table, bridge,
                             numpy_buffer_table)
        except (JitNotImplementedError, RealizationError):
            raise
        except Exception as e:
            raise RealizationError(
                f"WebGPU realizer: {node.op} (shape={node.shape}, "
                f"dtype={node.dtype}) failed: {e}"
            ) from e
        value_table[id(node)] = handle

        # We own handles minted by this call (not BUFFER lookups which the
        # GpuBufferTable owns across sessions). Release after last use.
        owned = node.op not in (OP_BUFFER, OP_LOAD)
        if owned and node is not root:
            intermediate_handles.append((id(node), handle))

        # Free any input whose last consumer is this node.
        # Buffers in the GpuBufferTable persist across calls; only
        # intermediates here count.
        for inp in node.inputs:
            if last_use.get(id(inp)) == idx:
                if inp.op in (OP_BUFFER, OP_LOAD):
                    continue  # session-scoped lifetime
                input_handle = value_table.get(id(inp))
                if input_handle is None:
                    continue
                bridge.release(input_handle)
                # Mark as released by removing from value_table so a
                # buggy double-free attempt later would KeyError clearly.
                value_table.pop(id(inp), None)
                # Also strip from intermediate_handles so the final cleanup
                # pass doesn't double-release.
                intermediate_handles = [
                    h for h in intermediate_handles if h[0] != id(inp)
                ]

    # 4. Materialise the root, then release remaining handles.
    root_handle = value_table[id(root)]
    data = bridge.materialize(root_handle, tuple(root.shape), root.dtype)
    # The root may have been put on the cleanup list above if it has a
    # consumer — but it has no consumer (it's the root), so it wasn't.
    # Release it after materialise.
    if root.op not in (OP_BUFFER, OP_LOAD):
        bridge.release(root_handle)
    # Defensive: release any handle that escaped the last-use sweep.
    for _, handle in intermediate_handles:
        # The release path above already pruned freed entries; anything
        # surviving here lacks a downstream consumer (e.g. dead code in
        # the IR — shouldn't happen post-fusion but cheap to guard).
        bridge.release(handle)

    # 5. Reconstruct ndarray. `data` is bytes from the bridge.
    arr = np.frombuffer(data, dtype=np.dtype(root.dtype))
    # Reshape if non-scalar.
    if root.shape:
        arr = arr.reshape(root.shape)
    return np.array(arr, copy=True)  # owning copy — `data` may be a view


# --------------------------------------------------------------------------
# Bridge registry — module-global mutable state, intentionally simple.
# --------------------------------------------------------------------------


_REGISTERED_BRIDGE: Optional[Any] = None
_REGISTERED_GBT: Optional[GpuBufferTable] = None


def register_webgpu_bridge(bridge: Any) -> None:
    """Install a bridge for `bg.realize_webgpu`. Calling again with a
    different bridge replaces the prior one and releases any cached
    handles."""
    global _REGISTERED_BRIDGE, _REGISTERED_GBT
    if _REGISTERED_GBT is not None and bridge is not _REGISTERED_BRIDGE:
        _REGISTERED_GBT.release_all()
    _REGISTERED_BRIDGE = bridge
    _REGISTERED_GBT = GpuBufferTable(bridge) if bridge is not None else None


def get_registered_bridge() -> Optional[Any]:
    return _REGISTERED_BRIDGE


def get_registered_gpu_buffer_table() -> Optional[GpuBufferTable]:
    return _REGISTERED_GBT


def is_available() -> bool:
    return _REGISTERED_BRIDGE is not None


def unregister_webgpu_bridge() -> None:
    """Symmetric to register; releases all handles and drops the bridge."""
    global _REGISTERED_BRIDGE, _REGISTERED_GBT
    if _REGISTERED_GBT is not None:
        _REGISTERED_GBT.release_all()
    _REGISTERED_BRIDGE = None
    _REGISTERED_GBT = None


__all__ = [
    "realize_webgpu",
    "register_webgpu_bridge",
    "unregister_webgpu_bridge",
    "get_registered_bridge",
    "get_registered_gpu_buffer_table",
    "is_available",
    "supported_opcodes",
]
