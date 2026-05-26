"""browsergrad_jit._realize — NumPy realization of a UOp graph.

INTERNAL. Realization is the moment a lazy UOp graph turns into concrete
np.ndarray values. The realize() function below is the single entry point;
every realization trigger on TensorProxy (.numpy, .tolist, .item, __bool__,
__float__, __int__, __iter__, .backward, optimizer.step) goes through it.

Algorithm — one topological walk + one dispatch-table call per node:

  1. topo = toposort(root)
  2. for node in topo:
       value_table[node] = _DISPATCH[node.op](node, value_table, buffer_table)
  3. return value_table[root]

The value_table is a per-realization dict keyed by id(uop) (UOps are hashable
but we use id() because two structurally-equal UOps that happen to be
different objects should be cached separately — the cost of structural
re-hashing on a 10K-node graph isn't worth saving the slot).

Dispatch handlers are pure functions: they receive the node, the in-flight
value_table, and the BufferTable, and return an ndarray. They never mutate
the input arrays — gradient accumulation is the ONE legitimate path that
mutates a BufferTable entry, and it goes through STORE, not through a
handler's return value.
"""

from __future__ import annotations
from typing import Any, Callable

import numpy as np

from ._ir import (
    UOp, ALL_OPS, toposort,
    OP_BUFFER, OP_LOAD, OP_STORE, OP_CONST, OP_RANDOM,
    OP_CAST, OP_ADD, OP_MUL, OP_DIV, OP_NEG,
    OP_EXP, OP_LOG, OP_CMP, OP_MATMUL, OP_REDUCE,
    OP_RESHAPE, OP_PERMUTE, OP_SLICE, OP_PAD,
    OP_WHERE, OP_INDEX, OP_MASK, OP_CUSTOM,
)
from ._buffer_table import BufferTable
from ._errors import RealizationError


# A dispatch handler. Receives the node, the in-flight value table, and the
# BufferTable. Returns a concrete np.ndarray. Must not mutate inputs.
Handler = Callable[[UOp, dict, BufferTable], np.ndarray]


# ---------------------------------------------------------------------------
# Per-opcode handlers
# ---------------------------------------------------------------------------


def _h_buffer(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    arr = bt.get(node.arg)
    # Defense in depth: confirm shape/dtype agree with what the IR promised.
    if tuple(arr.shape) != tuple(node.shape):
        raise RealizationError(
            f"BUFFER {node.arg!r} on the table has shape {arr.shape} but the "
            f"IR declared {node.shape}"
        )
    return arr


def _h_load(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    # LOAD just unwraps the BUFFER it wraps. The BUFFER handler already
    # ran (topological order), so we have the value cached.
    return vt[id(node.inputs[0])]


def _h_const(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    value = node.arg["value"]
    return np.asarray(value, dtype=np.dtype(node.dtype))


def _h_random(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    arg = node.arg
    rng = np.random.default_rng(arg["seed_key"])
    dist = arg["dist"]
    if dist == "uniform":
        # Optional 'low'/'high' overrides default [0, 1).
        low = arg.get("low", 0.0)
        high = arg.get("high", 1.0)
        out = rng.uniform(low=low, high=high, size=node.shape)
    elif dist == "normal":
        mean = arg.get("mean", 0.0)
        std = arg.get("std", 1.0)
        out = rng.normal(loc=mean, scale=std, size=node.shape)
    else:
        raise RealizationError(f"RANDOM: unknown dist {dist!r}")
    return out.astype(np.dtype(node.dtype), copy=False)


def _h_cast(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    return x.astype(np.dtype(node.arg["dtype"]), copy=False)


def _h_add(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    a = vt[id(node.inputs[0])]
    b = vt[id(node.inputs[1])]
    return a + b


def _h_mul(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    a = vt[id(node.inputs[0])]
    b = vt[id(node.inputs[1])]
    return a * b


def _h_div(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    a = vt[id(node.inputs[0])]
    b = vt[id(node.inputs[1])]
    # True division. For integer dtypes NumPy would normally float-promote;
    # the IR promises the output dtype matches the broadcast of the inputs,
    # so we honor that here.
    out = a / b
    return out.astype(np.dtype(node.dtype), copy=False)


def _h_neg(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    return -vt[id(node.inputs[0])]


def _h_exp(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    return np.exp(vt[id(node.inputs[0])])


def _h_log(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    return np.log(vt[id(node.inputs[0])])


_CMP_OPS = {
    "eq": np.equal,
    "ne": np.not_equal,
    "lt": np.less,
    "le": np.less_equal,
    "gt": np.greater,
    "ge": np.greater_equal,
}


def _h_cmp(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    a = vt[id(node.inputs[0])]
    b = vt[id(node.inputs[1])]
    op = node.arg["op"]
    if op not in _CMP_OPS:
        raise RealizationError(f"CMP: unknown comparison op {op!r}")
    return _CMP_OPS[op](a, b)


def _h_matmul(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    a = vt[id(node.inputs[0])]
    b = vt[id(node.inputs[1])]
    return a @ b


_REDUCE_OPS = {
    "sum": np.sum,
    "max": np.max,
    "min": np.min,
    "mean": np.mean,
    "argmax": np.argmax,
    "argmin": np.argmin,
    "any": np.any,
    "all": np.all,
}


def _h_reduce(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    op = node.arg["op"]
    axis = node.arg.get("axis", None)
    keepdims = node.arg.get("keepdims", False)
    fn = _REDUCE_OPS.get(op)
    if fn is None:
        raise RealizationError(f"REDUCE: unknown op {op!r}")
    if op in ("argmax", "argmin"):
        # NumPy's argmax/argmin take axis but not keepdims.
        out = fn(x, axis=axis)
        if keepdims and axis is not None:
            out = np.expand_dims(out, axis=axis)
    elif op in ("any", "all"):
        out = fn(x, axis=axis, keepdims=keepdims)
    else:
        out = fn(x, axis=axis, keepdims=keepdims)
    return np.asarray(out, dtype=np.dtype(node.dtype))


def _h_reshape(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    return x.reshape(node.arg["new_shape"])


def _h_permute(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    return np.transpose(x, axes=node.arg["axes"])


def _h_slice(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    slices = node.arg["slices"]
    # Cast to numpy slicing tuple.
    return x[tuple(slices)]


def _h_pad(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    pad_width = node.arg["pad_width"]
    mode = node.arg.get("mode", "constant")
    value = node.arg.get("value", 0.0)
    if mode == "constant":
        return np.pad(x, pad_width=pad_width, mode="constant", constant_values=value)
    return np.pad(x, pad_width=pad_width, mode=mode)


def _h_where(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    cond = vt[id(node.inputs[0])]
    a = vt[id(node.inputs[1])]
    b = vt[id(node.inputs[2])]
    return np.where(cond, a, b)


def _h_index(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    idx = vt[id(node.inputs[1])]
    dim = node.arg.get("dim", 0)
    return np.take(x, idx, axis=dim)


def _h_mask(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    mask = vt[id(node.inputs[1])]
    return x[mask]


def _h_custom(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    """Opaque escape hatch. The `arg` carries a callable that receives the
    input ndarrays and returns the output ndarray.

    Used in PRD-005 v0 to express Conv2d / Pool / MultiHeadAttention without
    decomposing them into the primitive op set — PRD-006 lifts those into
    real opcodes."""
    fn = node.arg.get("fn")
    if fn is None:
        raise RealizationError(
            f"CUSTOM UOp: arg must carry a 'fn' callable; got {node.arg!r}"
        )
    inputs = tuple(vt[id(inp)] for inp in node.inputs)
    captures = node.arg.get("captures", ())
    out = fn(*inputs, *captures)
    if not isinstance(out, np.ndarray):
        raise RealizationError(
            f"CUSTOM UOp: fn returned {type(out).__name__}, expected np.ndarray"
        )
    return out


def _h_store(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    """STORE writes its source into the BUFFER referenced by inputs[0].

    arg: {accumulate: bool}. With accumulate=True, the existing buffer
    content is added; otherwise overwritten. Returns the stored value so
    callers can chain — though typically STORE is a sink.
    """
    buf_uop = node.inputs[0]
    src = vt[id(node.inputs[1])]
    if buf_uop.op != OP_BUFFER:
        raise RealizationError(
            f"STORE: first input must be a BUFFER, got {buf_uop.op}"
        )
    target = bt.get(buf_uop.arg)
    accumulate = node.arg.get("accumulate", False) if node.arg else False
    new_value = (target + src) if accumulate else src
    bt.update(buf_uop.arg, new_value.astype(target.dtype, copy=False))
    return new_value


# Dispatch table. Adding a new opcode here is also an update to ALL_OPS in
# _ir.py — the sanity check at module-import time below would fire if we
# forget.
_DISPATCH: dict[str, Handler] = {
    OP_BUFFER:  _h_buffer,
    OP_LOAD:    _h_load,
    OP_STORE:   _h_store,
    OP_CONST:   _h_const,
    OP_RANDOM:  _h_random,
    OP_CAST:    _h_cast,
    OP_ADD:     _h_add,
    OP_MUL:     _h_mul,
    OP_DIV:     _h_div,
    OP_NEG:     _h_neg,
    OP_EXP:     _h_exp,
    OP_LOG:     _h_log,
    OP_CMP:     _h_cmp,
    OP_MATMUL:  _h_matmul,
    OP_REDUCE:  _h_reduce,
    OP_RESHAPE: _h_reshape,
    OP_PERMUTE: _h_permute,
    OP_SLICE:   _h_slice,
    OP_PAD:     _h_pad,
    OP_WHERE:   _h_where,
    OP_INDEX:   _h_index,
    OP_MASK:    _h_mask,
    OP_CUSTOM:  _h_custom,
}


# Module-load assertion: every opcode in ALL_OPS has a dispatch handler.
# Forgetting one would otherwise surface as a confusing KeyError mid-realization.
_missing = ALL_OPS - set(_DISPATCH)
if _missing:
    raise RuntimeError(
        f"_realize: opcodes lack dispatch handlers: {sorted(_missing)}"
    )
_extra = set(_DISPATCH) - ALL_OPS
if _extra:
    raise RuntimeError(
        f"_realize: dispatch table has handlers for unknown opcodes: {sorted(_extra)}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def realize(root: UOp, buffer_table: BufferTable) -> np.ndarray:
    """Walk the IR rooted at `root` in topological order, dispatching each
    UOp to its NumPy handler. Returns a fresh np.ndarray; safe to mutate.

    The dispatched values are held in a per-call dict (`value_table`).
    Intermediates that are no longer needed are not evicted in this v0
    implementation — gradient checkpointing (PRD-009) is the right tool
    for memory-limited workloads.

    Raises `RealizationError` if any handler fails. The error carries the
    offending UOp's opcode + shape in the message so debugging is local.
    """
    value_table: dict[int, np.ndarray] = {}
    order = toposort(root)
    for node in order:
        handler = _DISPATCH.get(node.op)
        if handler is None:
            # Should be unreachable thanks to the module-load assertion; if
            # it ever fires, the dispatch table drifted from ALL_OPS without
            # raising at import. Surface the bug clearly.
            raise RealizationError(
                f"no dispatch handler for opcode {node.op!r}"
            )
        try:
            value_table[id(node)] = handler(node, value_table, buffer_table)
        except Exception as e:
            # Wrap with context. Keep the cause chain so debuggers can
            # walk the original traceback.
            raise RealizationError(
                f"{node.op} (shape={node.shape}, dtype={node.dtype}) failed: {e}"
            ) from e
    return value_table[id(root)]


__all__ = ["realize"]
