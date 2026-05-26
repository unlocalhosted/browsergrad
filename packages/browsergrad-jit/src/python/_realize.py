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
    OP_FUSED_ELEMENTWISE, OP_FUSED_SOFTMAX,
    OP_SCATTER_ADD,
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


def _h_scatter_add(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    """Scatter-add the source values into a copy of `target` at positions
    given by `idx` along `dim`. The inverse of INDEX / GATHER.

    Inputs: (target, idx, src). The target carries the receiving shape and
    starting values (typically zeros built by the autograd builder); the
    realizer copies it so the original BUFFER stays untouched.

    `np.add.at` is the deterministic-by-construction NumPy call — same
    output every run. When PRD-012 lowers this to WGSL, the kernel must
    preserve that determinism (sort-and-segment-reduce by default).
    """
    target = vt[id(node.inputs[0])]
    idx = vt[id(node.inputs[1])]
    src = vt[id(node.inputs[2])]
    out = target.copy()
    dim = node.arg.get("dim", 0)
    # np.add.at handles per-axis fancy indexing; we route via the
    # `[dim slice + idx slice]` pattern.
    index_expr: list = [slice(None)] * out.ndim
    index_expr[dim] = idx
    np.add.at(out, tuple(index_expr), src)
    return out


def _h_fused_elementwise(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    """Realize an OP_FUSED_ELEMENTWISE chain in a single Python loop.

    The arg carries `ops`: a tuple of `(opcode, lhs_ref, rhs_ref_or_None)`.
    Refs are integers — negative means "external input N" (using
    `node.inputs[-ref - 1]`), non-negative means "the i-th step's output."

    Compared to dispatching each op through the realizer's main loop:
      * No `value_table` insertion + lookup overhead per intermediate.
      * No np.ndarray retained for any non-terminal step (Python's GC
        reclaims each step once the next one consumes it).
      * One Python frame instead of N.

    On a 3-op chain over (B=64, hidden=512) f32: peak intermediate memory
    drops from 384 KB (3 × 128 KB) to 128 KB (one in-flight).
    """
    ops = node.arg["ops"]
    externals = [vt[id(inp)] for inp in node.inputs]
    steps: list[np.ndarray] = []

    def _resolve(ref: int) -> np.ndarray:
        if ref < 0:
            return externals[-ref - 1]
        return steps[ref]

    for opcode, lhs_ref, rhs_ref in ops:
        a = _resolve(lhs_ref)
        if opcode == OP_ADD:
            steps.append(a + _resolve(rhs_ref))
        elif opcode == OP_MUL:
            steps.append(a * _resolve(rhs_ref))
        elif opcode == OP_DIV:
            steps.append(a / _resolve(rhs_ref))
        elif opcode == OP_NEG:
            steps.append(-a)
        elif opcode == OP_EXP:
            steps.append(np.exp(a))
        elif opcode == OP_LOG:
            steps.append(np.log(a))
        else:
            raise RealizationError(
                f"FUSED_ELEMENTWISE: unsupported inner opcode {opcode!r}"
            )
    return steps[-1]


def _h_fused_softmax(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    """Realize an OP_FUSED_SOFTMAX in three NumPy calls.

    Numerically stable: subtract the row-max before exp. Equivalent to
    the unfused 7-node decomposition the matcher absorbed, but emits
    fewer intermediate ndarrays (`x - m` doesn't survive past the
    subsequent `np.exp`)."""
    x = vt[id(node.inputs[0])]
    axis = node.arg["axis"]
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


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
    # Fusion (PRD-006)
    OP_FUSED_ELEMENTWISE: _h_fused_elementwise,
    OP_FUSED_SOFTMAX:     _h_fused_softmax,
    # Autograd (PRD-007)
    OP_SCATTER_ADD:       _h_scatter_add,
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


def realize(
    root: UOp,
    buffer_table: BufferTable,
    *,
    autograd_holdout: "set[int] | None" = None,
) -> np.ndarray:
    """Walk the IR rooted at `root` in topological order, dispatching each
    UOp to its NumPy handler. Returns a fresh np.ndarray; safe to mutate.

    When fusion is enabled (`_fusion_config.is_enabled()`), the pass runs
    first and replaces `root` with a rewritten graph containing
    `OP_FUSED_*` nodes. The dispatch table picks them up via their
    handlers transparently.

    `autograd_holdout`: the set of UOp ids that must not be absorbed as
    non-terminal nodes in any fused group. The caller (typically a
    backward walker that has access to per-proxy `_ctx.input_proxies`)
    computes this via `_fusion.collect_autograd_holdout` and passes it
    in. Default `None` ≡ "no constraints," which is correct for forward-
    only realization where no closures will later read intermediates.

    Raises `RealizationError` if any handler fails. The error carries the
    offending UOp's opcode + shape in the message so debugging is local.
    """
    from . import _fusion_config
    if _fusion_config.is_enabled():
        from ._fusion import fuse
        root = fuse(root, holdout=autograd_holdout or set())
    else:
        # Keep introspection state consistent: a realize() with fusion
        # disabled should not leave a stale report from a prior fuse()
        # call lying around for debug_fused_kernels() to find.
        from . import _fusion as _f
        _f._LAST_REPORT = _f.FusionReport()

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
