"""browsergrad_jit._ir — the UOp IR.

INTERNAL. Not part of the published API. The leading underscore is the
contract: opcode strings, the UOp dataclass shape, and helpers in this
module may change in any minor or patch release. User code must never
import from `browsergrad_jit._ir`.

The IR is a directed acyclic graph of UOp nodes. Each UOp carries:
  - op: an opcode string (one of OP_*)
  - inputs: a tuple of upstream UOps
  - shape: a tuple of ints (or () for scalars)
  - dtype: a NumPy dtype name (e.g. "float32", "int64", "bool")
  - arg: opcode-specific metadata (None, a dict, a tuple, or a primitive)

UOps are frozen dataclasses — immutable, hashable, structurally compared.
Two UOps built with the same inputs / shape / dtype / arg are equal and
hash the same. This is what lets the trace cache and pipeline cache
(PRD-008) key on UOp identity.

Design notes:

  * 78 opcodes total — corrects PRD-005's nominal 19. The extras are
    documented per-opcode below. Headline additions: RANDOM (dropout +
    nn.init at trace time), CMP (boolean results), CONV1D/CONV2D/CONV3D
    primitive CNN ops and backward refs, CUSTOM
    (opaque-NumPy escape hatch for Pool/Attention/etc. until
    PRD-006 decomposes them), INDEX/MASK (replacing PRD-005's overloaded
    GATHER).
  * Shape inference happens at construction. UOp(...) raises ShapeError
    with the user's traceback intact. Never pass shape inference downstream.
  * The hash is cached lazily on first access. Without this, hashing the
    root of a 10K-node graph would recurse structurally and blow stack
    + cost ~O(n²) on repeated hashing.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, FrozenSet

from ._errors import ShapeError
from ._framework_contracts import (
    has_framework_operation_contract,
    has_internal_operation_contract,
    validate_framework_operation_contract,
    validate_internal_operation_contract,
)


# ---------------------------------------------------------------------------
# Opcode catalog. Strings (not enums) so grep finds usages cleanly across
# Python + TypeScript test fixtures + future serialization (if any).
#
# Categories:
#   Leaves:        BUFFER, LOAD, CONST, RANDOM
#   Stores:        STORE
#   Elementwise:   ADD, MUL, DIV, NEG, EXP, LOG, CAST
#   Compare:       CMP
#   Matmul:        MATMUL
#   Reductions:    REDUCE
#   Shape ops:     RESHAPE, PERMUTE, SLICE, PAD
#   Conditional:   WHERE
#   Indexing:      INDEX, MASK
#   CNN:           CONV1D, CONV1D_BACKWARD_INPUT, CONV1D_BACKWARD_WEIGHT,
#                  CONV1D_BACKWARD_BIAS, CONV2D,
#                  CONV2D_BACKWARD_INPUT, CONV2D_BACKWARD_WEIGHT,
#                  CONV2D_BACKWARD_BIAS, CONV_TRANSPOSE2D,
#                  CONV_TRANSPOSE2D_BACKWARD_INPUT,
#                  CONV_TRANSPOSE2D_BACKWARD_WEIGHT,
#                  CONV_TRANSPOSE2D_BACKWARD_BIAS, CONV3D,
#                  CONV3D_BACKWARD_INPUT, CONV3D_BACKWARD_WEIGHT,
#                  CONV3D_BACKWARD_BIAS
#   Norm:          LAYER_NORM, LAYER_NORM_BACKWARD_INPUT,
#                  LAYER_NORM_BACKWARD_WEIGHT, LAYER_NORM_BACKWARD_BIAS
#   Escape hatch:  CUSTOM
#   Optimizer:     SGD_UPDATE, ADAMW_UPDATE_*, ADAM_UPDATE_*
# ---------------------------------------------------------------------------

OP_BUFFER  = "BUFFER"   # arg: buffer_id (str)        ← leaf, no inputs
OP_LOAD    = "LOAD"     # arg: None                   ← reads BUFFER
OP_STORE   = "STORE"    # arg: {accumulate: bool}     ← writes into BUFFER
OP_CONST   = "CONST"    # arg: {value: int|float}     ← scalar leaf
OP_RANDOM  = "RANDOM"   # arg: {dist, seed_key}       ← per-call PRNG
OP_CAST    = "CAST"     # arg: {dtype: str}
OP_ADD     = "ADD"
OP_MUL     = "MUL"
OP_DIV     = "DIV"
OP_NEG     = "NEG"
OP_EXP     = "EXP"
OP_LOG     = "LOG"
OP_ABS     = "ABS"
OP_CLAMP   = "CLAMP"
OP_COS     = "COS"
OP_FLIP    = "FLIP"
OP_CUMSUM  = "CUMSUM"
OP_CONCAT  = "CONCAT"
OP_STACK   = "STACK"
OP_NARROW  = "NARROW"
OP_TRIL    = "TRIL"
OP_TRIU    = "TRIU"
OP_PROD    = "PROD"
OP_VAR     = "VAR"
OP_REPEAT  = "REPEAT"
OP_REPEAT_INTERLEAVE = "REPEAT_INTERLEAVE"
OP_SIGN    = "SIGN"
OP_SIN     = "SIN"
OP_CMP     = "CMP"      # arg: {op: 'eq'|'lt'|'le'|'gt'|'ge'|'ne'}
OP_MATMUL  = "MATMUL"
OP_CONV1D  = "CONV1D"   # arg: {n,c_in,l_in,c_out,k,stride,padding,dilation,groups,l_out,has_bias}
OP_CONV1D_BACKWARD_INPUT  = "CONV1D_BACKWARD_INPUT"   # inputs: (dy, weight), arg: conv1d metadata
OP_CONV1D_BACKWARD_WEIGHT = "CONV1D_BACKWARD_WEIGHT"  # inputs: (dy, input), arg: conv1d metadata
OP_CONV1D_BACKWARD_BIAS   = "CONV1D_BACKWARD_BIAS"    # inputs: (dy,), arg: conv1d metadata
OP_CONV2D  = "CONV2D"   # arg: {n,c_in,h,w,c_out,kh,kw,stride_h,stride_w,pad_h,pad_w,dilation_h,dilation_w,groups,out_h,out_w,has_bias}
OP_CONV2D_BACKWARD_INPUT  = "CONV2D_BACKWARD_INPUT"   # inputs: (dy, weight), arg: conv2d metadata
OP_CONV2D_BACKWARD_WEIGHT = "CONV2D_BACKWARD_WEIGHT"  # inputs: (dy, input), arg: conv2d metadata
OP_CONV2D_BACKWARD_BIAS   = "CONV2D_BACKWARD_BIAS"    # inputs: (dy,), arg: conv2d metadata
OP_CONV_TRANSPOSE2D  = "CONV_TRANSPOSE2D"   # arg: transposed conv2d metadata
OP_CONV_TRANSPOSE2D_BACKWARD_INPUT  = "CONV_TRANSPOSE2D_BACKWARD_INPUT"   # inputs: (dy, weight)
OP_CONV_TRANSPOSE2D_BACKWARD_WEIGHT = "CONV_TRANSPOSE2D_BACKWARD_WEIGHT"  # inputs: (dy, input)
OP_CONV_TRANSPOSE2D_BACKWARD_BIAS   = "CONV_TRANSPOSE2D_BACKWARD_BIAS"    # inputs: (dy,)
OP_CONV3D  = "CONV3D"   # arg: {n,c_in,d,h,w,c_out,kd,kh,kw,stride_d,stride_h,stride_w,pad_d,pad_h,pad_w,dilation_d,dilation_h,dilation_w,groups,out_d,out_h,out_w,has_bias}
OP_CONV3D_BACKWARD_INPUT  = "CONV3D_BACKWARD_INPUT"   # inputs: (dy, weight), arg: conv3d metadata
OP_CONV3D_BACKWARD_WEIGHT = "CONV3D_BACKWARD_WEIGHT"  # inputs: (dy, input), arg: conv3d metadata
OP_CONV3D_BACKWARD_BIAS   = "CONV3D_BACKWARD_BIAS"    # inputs: (dy,), arg: conv3d metadata
OP_LAYER_NORM = "LAYER_NORM"  # inputs: (x, weight, bias), arg: {normalized_shape, rows, cols, eps}
OP_LAYER_NORM_BACKWARD_INPUT = "LAYER_NORM_BACKWARD_INPUT"  # inputs: (dy, x, weight)
OP_LAYER_NORM_BACKWARD_WEIGHT = "LAYER_NORM_BACKWARD_WEIGHT"  # inputs: (dy, x)
OP_LAYER_NORM_BACKWARD_BIAS = "LAYER_NORM_BACKWARD_BIAS"  # inputs: (dy,)
OP_REDUCE  = "REDUCE"   # arg: {op, axis, keepdims}
OP_RESHAPE = "RESHAPE"  # arg: {new_shape: tuple[int,...]}
OP_PERMUTE = "PERMUTE"  # arg: {axes: tuple[int,...]}
OP_SLICE   = "SLICE"    # arg: {slices: tuple[slice,...]}
OP_PAD     = "PAD"      # arg: {pad_width, mode, value}
OP_SORT_INDICES = "SORT_INDICES"  # arg: {axis, descending, stable}, inputs: (source,)
OP_SORT_VALUES = "SORT_VALUES"    # same arg, inputs: (source, SORT_INDICES)
OP_TOPK_INDICES = "TOPK_INDICES"  # arg: {axis, k, largest, sorted}, inputs: (source,)
OP_TOPK_VALUES = "TOPK_VALUES"    # same arg, inputs: (source, TOPK_INDICES)
OP_SCATTER = "SCATTER"  # arg: {dim}, inputs: (target, int64 index, source)
OP_EINSUM = "EINSUM"    # arg: {equation, batch_rank}, inputs: variadic operands
OP_WHERE   = "WHERE"
OP_INDEX   = "INDEX"    # arg: {dim: int}, inputs: (source, int64 index) ← gather elements
OP_MASK    = "MASK"     # arg: None                    ← bool-mask indexing
OP_CUSTOM  = "CUSTOM"   # arg: {fn_id, captures, ...}  ← opaque NumPy callable

# Fusion-emitted opcodes (PRD-006). These never appear in user-built IR;
# they're created by `_fusion.fuse()` as graph rewrites of the elementwise
# and softmax subgraphs the matcher recognizes. The realizer dispatches
# each as a single NumPy handler, eliminating per-op intermediate
# allocations. Equally important: they create the seam that PRD-012's
# WGSL megakernel codegen plugs into without touching pattern matching.
OP_FUSED_ELEMENTWISE = "FUSED_ELEMENTWISE"  # arg: {ops, external_inputs}
OP_FUSED_SOFTMAX     = "FUSED_SOFTMAX"      # arg: {axis: int}

# Autograd primitives (PRD-007). SCATTER_ADD is the inverse of INDEX/MASK;
# emitted by the VJP rules for those opcodes and by NLL/cross-entropy
# backwards that distribute gradient back to per-class positions. The
# NumPy realizer uses `np.add.at` — deterministic by construction.
# When PRD-012 lowers SCATTER_ADD to WGSL, the kernel must default to a
# deterministic sort-and-segment-reduce, with an opt-in atomic-add fast
# path. See PRD-007's deliverable 7 (OQ6).
OP_SCATTER_ADD  = "SCATTER_ADD"   # arg: {dim: int}, inputs: (target, idx, src)
OP_BROADCAST_TO = "BROADCAST_TO"  # arg: {shape: tuple[int,...]}, inputs: (x,)
OP_EINSUM_VJP = "EINSUM_VJP"      # inputs: (dy, *operands), arg: equation/operand

# Mixed-precision support (PRD-010). ISNAN is the per-element NaN
# check used by GradScaler's overflow detection. The "OR across all
# parameters" reduction reuses REDUCE(any) from PRD-005's reductions
# — no separate OR opcode needed.
OP_ISNAN = "ISNAN"  # inputs: (x,) → bool-typed mask of same shape

# Optimizer/update IR. This is the first functional update node for the
# GPU-native path: param + grad in, updated param out. In-place STORE/resident
# optimizer state can build on this primitive without doing math in Python.
OP_SGD_UPDATE = "SGD_UPDATE"  # inputs: (param, grad), arg: {lr, weight_decay}
OP_ADAMW_UPDATE_M = "ADAMW_UPDATE_M"  # inputs: (m, grad), arg: {beta1}
OP_ADAMW_UPDATE_V = "ADAMW_UPDATE_V"  # inputs: (v, grad), arg: {beta2}
OP_ADAMW_UPDATE_PARAM = "ADAMW_UPDATE_PARAM"  # inputs: (param, grad, m_new, v_new)
OP_ADAM_UPDATE_M = "ADAM_UPDATE_M"  # inputs: (param, grad, m), arg: {beta1, weight_decay}
OP_ADAM_UPDATE_V = "ADAM_UPDATE_V"  # inputs: (param, grad, v), arg: {beta2, weight_decay}
OP_ADAM_UPDATE_PARAM = "ADAM_UPDATE_PARAM"  # inputs: (param, m_new, v_new), arg: {lr,beta1,beta2,eps,step}

ALL_OPS: FrozenSet[str] = frozenset({
    OP_BUFFER, OP_LOAD, OP_STORE, OP_CONST, OP_RANDOM, OP_CAST,
    OP_ADD, OP_MUL, OP_DIV, OP_NEG, OP_EXP, OP_LOG,
    OP_ABS, OP_CLAMP, OP_COS, OP_FLIP, OP_CUMSUM, OP_CONCAT, OP_STACK, OP_NARROW, OP_TRIL, OP_TRIU, OP_PROD, OP_VAR, OP_REPEAT, OP_REPEAT_INTERLEAVE,
    OP_SIGN, OP_SIN, OP_CMP,
    OP_MATMUL, OP_CONV1D, OP_CONV1D_BACKWARD_INPUT,
    OP_CONV1D_BACKWARD_WEIGHT, OP_CONV1D_BACKWARD_BIAS,
    OP_CONV2D, OP_CONV2D_BACKWARD_INPUT,
    OP_CONV2D_BACKWARD_WEIGHT, OP_CONV2D_BACKWARD_BIAS,
    OP_CONV_TRANSPOSE2D, OP_CONV_TRANSPOSE2D_BACKWARD_INPUT,
    OP_CONV_TRANSPOSE2D_BACKWARD_WEIGHT, OP_CONV_TRANSPOSE2D_BACKWARD_BIAS,
    OP_CONV3D, OP_CONV3D_BACKWARD_INPUT, OP_CONV3D_BACKWARD_WEIGHT,
    OP_CONV3D_BACKWARD_BIAS, OP_LAYER_NORM, OP_LAYER_NORM_BACKWARD_INPUT,
    OP_LAYER_NORM_BACKWARD_WEIGHT, OP_LAYER_NORM_BACKWARD_BIAS, OP_REDUCE,
    OP_RESHAPE, OP_PERMUTE, OP_SLICE, OP_PAD, OP_SORT_INDICES, OP_SORT_VALUES,
    OP_TOPK_INDICES, OP_TOPK_VALUES, OP_SCATTER, OP_EINSUM,
    OP_WHERE, OP_INDEX, OP_MASK, OP_CUSTOM,
    # Fusion-emitted (PRD-006)
    OP_FUSED_ELEMENTWISE, OP_FUSED_SOFTMAX,
    # Autograd-emitted (PRD-007)
    OP_SCATTER_ADD, OP_BROADCAST_TO, OP_EINSUM_VJP,
    # Mixed precision (PRD-010)
    OP_ISNAN, OP_SGD_UPDATE, OP_ADAMW_UPDATE_M, OP_ADAMW_UPDATE_V,
    OP_ADAMW_UPDATE_PARAM, OP_ADAM_UPDATE_M, OP_ADAM_UPDATE_V,
    OP_ADAM_UPDATE_PARAM,
})
assert len(ALL_OPS) == 78, "opcode count drifted from PRD-005+006+007+010+CNN+norm+optimizer"


# Opcodes that take zero IR inputs. Their data lives entirely in `arg`.
_LEAF_OPS: FrozenSet[str] = frozenset({OP_BUFFER, OP_CONST, OP_RANDOM})


@dataclass(frozen=True, slots=True)
class UOp:
    """A single node in the IR graph.

    Construction validates the opcode and runs shape-inference invariants
    where they can be checked locally (correct input arity, leaf-vs-non-leaf,
    dtype well-formed). Per-opcode shape inference lives in `_infer_shape`
    and is called by the high-level builder methods on TensorProxy — UOps
    constructed directly from user code skip the full inference path and
    must supply a correct `shape` themselves.

    Equality and hashing are structural: two UOps with the same
    (op, inputs, shape, dtype, arg) are equal. The dataclass `frozen=True`
    auto-generates these but recurses through `inputs`. For very large
    graphs we cache the hash via a non-field slot to avoid O(n²) repeated
    structural hashing.
    """

    op: str
    inputs: Tuple["UOp", ...]
    shape: Tuple[int, ...]
    dtype: str
    arg: Any = None

    # Lazy cache for structural hash. Frozen-dataclass + slots means we
    # can't store this normally; we tunnel it through a dict-keyed-by-id
    # at module scope. Trade-off: small memory growth proportional to
    # number of distinct UOps hashed. Cleared on Python interpreter exit.
    def __hash__(self) -> int:  # type: ignore[override]
        cached = _HASH_CACHE.get(id(self))
        if cached is not None:
            return cached
        # Hash field-by-field; tuple of UOps recurses via their __hash__.
        h = hash((self.op, self.inputs, self.shape, self.dtype, _hashable_arg(self.arg)))
        _HASH_CACHE[id(self)] = h
        return h

    def __post_init__(self) -> None:
        # Cheap structural validation. Anything more expensive (shape
        # inference, broadcast rules) lives in the high-level builder.
        if self.op not in ALL_OPS:
            raise ValueError(
                f"unknown opcode {self.op!r}; expected one of {sorted(ALL_OPS)}"
            )
        if self.op in _LEAF_OPS:
            if self.inputs:
                raise ShapeError(
                    f"{self.op} is a leaf opcode and must have zero inputs, "
                    f"got {len(self.inputs)}"
                )
        else:
            if not self.inputs:
                raise ShapeError(
                    f"{self.op} is not a leaf and must have at least one input"
                )
        # Shape must be a tuple of non-negative ints (0-d is ()).
        if not isinstance(self.shape, tuple):
            raise ShapeError(
                f"{self.op}: shape must be a tuple, got {type(self.shape).__name__}"
            )
        for i, d in enumerate(self.shape):
            if not isinstance(d, int) or d < 0:
                raise ShapeError(
                    f"{self.op}: shape[{i}] must be a non-negative int, got {d!r}"
                )
        # Refuse a too-large allocation up front. 2**30 elements is ~4 GB
        # in fp32 — well above the WebGPU buffer cap and certainly above
        # what any sane in-browser lab should request. Catching it here
        # turns "tab hangs forever" into "Python raises with a stack".
        n_elements = 1
        for d in self.shape:
            n_elements *= max(d, 1)
        if n_elements > _MAX_ELEMENTS:
            raise ShapeError(
                f"{self.op}: shape {self.shape} = {n_elements} elements exceeds "
                f"the {_MAX_ELEMENTS}-element ceiling. If this is intentional, "
                f"raise the bar via browsergrad_jit._ir.set_max_elements()."
            )
        # dtype must be a NumPy-recognized name. We don't import numpy here
        # to keep the IR module hot-importable; checking a whitelist of
        # known names instead.
        if self.dtype not in _KNOWN_DTYPES:
            raise ValueError(
                f"{self.op}: unknown dtype {self.dtype!r}; expected one of "
                f"{sorted(_KNOWN_DTYPES)}"
            )
        if has_framework_operation_contract(self.op):
            validate_framework_operation_contract(self)
        if has_internal_operation_contract(self.op):
            validate_internal_operation_contract(self)


# Dtypes the IR knows about. Add new dtypes here AND update the dispatch
# tables in _realize.py (added in Week 3) and any opcode-specific casts.
_KNOWN_DTYPES: FrozenSet[str] = frozenset({
    "float16", "float32", "float64",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "bool",
})


# Maximum element count per tensor. Below the WebGPU 2 GB buffer cap with
# a 4 GB headroom; high enough that real models pass, low enough to refuse
# obviously-malicious requests like shape=(10**12,). Set via
# `set_max_elements()` if you're doing something legitimate at that scale.
_MAX_ELEMENTS: int = 2 ** 30


def set_max_elements(n: int) -> None:
    """Raise (or lower) the max-elements ceiling. Intended for tests."""
    global _MAX_ELEMENTS
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"max_elements must be a positive int, got {n!r}")
    _MAX_ELEMENTS = n


# Hash cache. Module-scope dict keyed by `id(uop)`. Cleared on interpreter
# shutdown; bounded in size by the number of distinct UOps the user creates.
# A non-frozen field would be simpler but breaks `frozen=True` immutability.
_HASH_CACHE: dict[int, int] = {}


def _hashable_arg(arg: Any) -> Any:
    """Convert an opcode's `arg` payload into something hashable.

    Most args are None, tuples, frozensets, ints, floats, or strings — already
    hashable. The exception is `dict` (used for REDUCE/CAST/RANDOM/etc); we
    canonicalize via sorted-items so hash stability holds across dict
    construction orders.
    """
    if arg is None or isinstance(arg, (int, float, str, bool, frozenset)):
        return arg
    if isinstance(arg, tuple):
        return tuple(_hashable_arg(a) for a in arg)
    if isinstance(arg, dict):
        return tuple(sorted((k, _hashable_arg(v)) for k, v in arg.items()))
    if isinstance(arg, slice):
        return ("slice", arg.start, arg.stop, arg.step)
    # Fallback: rely on the object's own __hash__. If it's unhashable, this
    # will raise — fine, the user passed something the IR doesn't understand.
    return arg


# ---------------------------------------------------------------------------
# Graph walkers
# ---------------------------------------------------------------------------


def toposort(root: UOp) -> list[UOp]:
    """Topological sort, leaves first.

    Uses iterative depth-first traversal keyed on id() to avoid Python's
    recursion limit on deep graphs (transformer training graphs hit ~10K
    nodes for a few-layer model). The id() key handles structurally-equal
    UOps that are distinct objects — we treat them as distinct nodes.
    """
    order: list[UOp] = []
    visited: set[int] = set()
    # Stack frames are (node, child_iter, visited_yet). We push child_iter
    # so we can resume after a yield rather than re-iterating from start.
    stack: list[tuple[UOp, int]] = [(root, 0)]
    while stack:
        node, child_idx = stack[-1]
        if id(node) in visited:
            stack.pop()
            continue
        if child_idx < len(node.inputs):
            stack[-1] = (node, child_idx + 1)
            child = node.inputs[child_idx]
            if id(child) not in visited:
                stack.append((child, 0))
            continue
        visited.add(id(node))
        order.append(node)
        stack.pop()
    return order


def all_buffers(root: UOp) -> list[UOp]:
    """All BUFFER leaves reachable from root, in topological (leaves-first) order.

    Used by realization to know which buffers must be resident in the
    BufferTable before the realizer can execute the graph.
    """
    return [u for u in toposort(root) if u.op == OP_BUFFER]


# ---------------------------------------------------------------------------
# Convenience constructors. These do NOT do shape inference — that's the
# tensor_proxy layer's job. They exist for tests and for the realizer to
# build synthetic UOps in unit tests.
# ---------------------------------------------------------------------------


def buffer(buffer_id: str, shape: Tuple[int, ...], dtype: str) -> UOp:
    """Construct a BUFFER leaf for `buffer_id`. The buffer must already
    be registered in a BufferTable; the IR carries only the id, not the
    array data."""
    return UOp(op=OP_BUFFER, inputs=(), shape=shape, dtype=dtype, arg=buffer_id)


def load(buf: UOp) -> UOp:
    """Wrap a BUFFER in a LOAD so it can be used as an op input."""
    if buf.op != OP_BUFFER:
        raise ShapeError(f"load() expects a BUFFER UOp, got {buf.op!r}")
    return UOp(op=OP_LOAD, inputs=(buf,), shape=buf.shape, dtype=buf.dtype, arg=None)


def const(value: float | int, dtype: str = "float32") -> UOp:
    """0-d CONST scalar."""
    return UOp(op=OP_CONST, inputs=(), shape=(), dtype=dtype, arg={"value": value})


__all__ = [
    # Opcode strings
    "OP_BUFFER", "OP_LOAD", "OP_STORE", "OP_CONST", "OP_RANDOM",
    "OP_CAST", "OP_ADD", "OP_MUL", "OP_DIV", "OP_NEG",
    "OP_EXP", "OP_LOG", "OP_ABS", "OP_CLAMP", "OP_COS", "OP_FLIP", "OP_CUMSUM", "OP_CONCAT", "OP_STACK", "OP_NARROW", "OP_TRIL", "OP_TRIU",
    "OP_PROD", "OP_VAR", "OP_REPEAT", "OP_REPEAT_INTERLEAVE", "OP_SIGN", "OP_SIN",
    "OP_CMP", "OP_MATMUL",
    "OP_CONV1D", "OP_CONV1D_BACKWARD_INPUT",
    "OP_CONV1D_BACKWARD_WEIGHT", "OP_CONV1D_BACKWARD_BIAS",
    "OP_CONV2D",
    "OP_CONV2D_BACKWARD_INPUT", "OP_CONV2D_BACKWARD_WEIGHT",
    "OP_CONV2D_BACKWARD_BIAS",
    "OP_CONV_TRANSPOSE2D", "OP_CONV_TRANSPOSE2D_BACKWARD_INPUT",
    "OP_CONV_TRANSPOSE2D_BACKWARD_WEIGHT", "OP_CONV_TRANSPOSE2D_BACKWARD_BIAS",
    "OP_CONV3D", "OP_CONV3D_BACKWARD_INPUT",
    "OP_CONV3D_BACKWARD_WEIGHT", "OP_CONV3D_BACKWARD_BIAS",
    "OP_LAYER_NORM", "OP_LAYER_NORM_BACKWARD_INPUT",
    "OP_LAYER_NORM_BACKWARD_WEIGHT", "OP_LAYER_NORM_BACKWARD_BIAS",
    "OP_REDUCE",
    "OP_RESHAPE", "OP_PERMUTE", "OP_SLICE", "OP_PAD",
    "OP_SORT_INDICES", "OP_SORT_VALUES", "OP_TOPK_INDICES", "OP_TOPK_VALUES",
    "OP_SCATTER", "OP_EINSUM",
    "OP_WHERE", "OP_INDEX", "OP_MASK", "OP_CUSTOM",
    "OP_FUSED_ELEMENTWISE", "OP_FUSED_SOFTMAX",
    "OP_SCATTER_ADD", "OP_BROADCAST_TO", "OP_EINSUM_VJP",
    "OP_ISNAN", "OP_SGD_UPDATE", "OP_ADAMW_UPDATE_M",
    "OP_ADAMW_UPDATE_V", "OP_ADAMW_UPDATE_PARAM",
    "OP_ADAM_UPDATE_M", "OP_ADAM_UPDATE_V", "OP_ADAM_UPDATE_PARAM",
    "ALL_OPS",
    # Core class + helpers
    "UOp", "toposort", "all_buffers",
    # Convenience constructors
    "buffer", "load", "const",
    # Test hooks
    "set_max_elements",
]
