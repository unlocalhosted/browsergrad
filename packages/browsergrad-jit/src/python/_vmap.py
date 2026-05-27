"""browsergrad_jit._vmap — JAX-style batching transform.

INTERNAL. Lifts `bg.func.vmap` from refuses-with-pointer to a real
batching-transform that handles the 15 most common opcodes. Per the
PRD-014 review, this is the trace-once approach: walk the IR, apply
per-opcode batching rules, return a transformed IR whose every leaf
input has a leading batch dim.

Algorithm (mirrors JAX's `vmap`):
  1. Caller calls `fn(*args)` once with regular (unbatched) inputs to
     get the unbatched IR rooted at `out`.
  2. We walk the IR and apply a batching rule per opcode. Each rule
     returns a new UOp with a batch dim added to its output shape.
  3. The user supplies `in_dims` to say which input axis is the batch
     dim. v0 supports `in_dims=0` (batch is the leading axis).
  4. `out_dims=0` puts the batch dim back at axis 0 on the output.

Rule signature:
  (node: UOp, batched_inputs: dict[int, UOp], B: int) -> UOp

Each rule receives the original UOp, a map from input UOp id to its
already-batched replacement, and the batch size. Returns the batched
output UOp.

What we cover (v0):
  Lifecycle: BUFFER, LOAD, CONST (CONST broadcasts; BUFFER is a leaf
    the caller batches — see step 3 above).
  Elementwise: ADD, MUL, DIV, NEG, EXP, LOG, CAST, CMP, WHERE.
  Shape: RESHAPE (prepend B), PERMUTE (shift axes), BROADCAST_TO
    (prepend B to target).
  Compute: MATMUL (becomes batched matmul; NumPy realizer's `@`
    already broadcasts over leading batch dims correctly).
  Reduce: REDUCE (shift axis numeric +1, or skip the new batch dim
    when axis=None).

What we don't (raises): SCATTER_ADD, INDEX, MASK, RANDOM, CUSTOM,
FUSED_*, PAD, SLICE. The rules to add are mechanical — follow the
same template — but PRD-014b owns them per the review's scope.

Why this isn't `jit.trace` re-bound:
  The trace-cache `_rebind` pattern substitutes BUFFER ids but
  preserves UOp shapes. vmap *changes* every UOp's shape (prepends B).
  So we walk and rebuild the graph.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np

from ._ir import (
    UOp, toposort,
    OP_BUFFER, OP_LOAD, OP_CONST, OP_CAST,
    OP_ADD, OP_MUL, OP_DIV, OP_NEG, OP_EXP, OP_LOG, OP_CMP,
    OP_MATMUL, OP_REDUCE, OP_RESHAPE, OP_PERMUTE,
    OP_WHERE, OP_BROADCAST_TO,
)
from ._errors import JitNotImplementedError


_VMAP_RULES: Dict[str, Callable[..., UOp]] = {}


def register_vmap(op: str) -> Callable[[Callable[..., UOp]], Callable[..., UOp]]:
    def deco(fn: Callable[..., UOp]) -> Callable[..., UOp]:
        _VMAP_RULES[op] = fn
        return fn
    return deco


def get_vmap_rule(op: str) -> Any:
    return _VMAP_RULES.get(op)


# Lifecycle ops --------------------------------------------------------


@register_vmap(OP_BUFFER)
def _vmap_buffer(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    """BUFFER leaves that came from a vmapped input are pre-populated
    in `batched` by the entry point. Internal BUFFERs (e.g. grad's seed,
    weight initializers, CONST leaves promoted to BUFFER) pass through
    unchanged — NumPy broadcasting handles their interaction with
    batched neighbours downstream."""
    return batched.get(id(node), node)


@register_vmap(OP_LOAD)
def _vmap_load(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    inner = batched[id(node.inputs[0])]
    # If the BUFFER passed through unchanged (internal constant), the
    # LOAD does too. Saves an alloc and keeps caches stable.
    if inner is node.inputs[0]:
        return node
    return UOp(op=OP_LOAD, inputs=(inner,),
               shape=inner.shape, dtype=inner.dtype, arg=node.arg)


@register_vmap(OP_CONST)
def _vmap_const(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    """CONSTs don't carry the batch dim — they broadcast naturally. Keep
    them un-batched; downstream binary ops will broadcast against the
    batched operand."""
    return node


# Elementwise ----------------------------------------------------------


def _batched_shape(shape: Tuple[int, ...], B: int) -> Tuple[int, ...]:
    return (B,) + tuple(shape)


def _broadcast(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    """Broadcast like np.broadcast_shapes but return a tuple."""
    return tuple(np.broadcast_shapes(*shapes))


def _elementwise_binop(op_code: str):
    def _rule(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
        a = batched[id(node.inputs[0])]
        b = batched[id(node.inputs[1])]
        # Compute shape from the actual batched-input shapes — the
        # original IR's recorded shape may be smaller than the actual
        # broadcast result (a quirk of VJP helpers using dy.shape).
        new_shape = _broadcast(a.shape, b.shape)
        return UOp(op=op_code, inputs=(a, b), shape=new_shape,
                   dtype=node.dtype, arg=node.arg)
    return _rule


_VMAP_RULES[OP_ADD] = _elementwise_binop(OP_ADD)
_VMAP_RULES[OP_MUL] = _elementwise_binop(OP_MUL)
_VMAP_RULES[OP_DIV] = _elementwise_binop(OP_DIV)


@register_vmap(OP_NEG)
def _vmap_neg(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    inner = batched[id(node.inputs[0])]
    # Derive shape from the batched input, NOT from node.shape — VJP
    # rules sometimes record dy.shape (smaller than the broadcast result).
    return UOp(op=OP_NEG, inputs=(inner,),
               shape=inner.shape, dtype=node.dtype, arg=node.arg)


@register_vmap(OP_EXP)
def _vmap_exp(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    inner = batched[id(node.inputs[0])]
    return UOp(op=OP_EXP, inputs=(inner,),
               shape=inner.shape, dtype=node.dtype, arg=node.arg)


@register_vmap(OP_LOG)
def _vmap_log(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    inner = batched[id(node.inputs[0])]
    return UOp(op=OP_LOG, inputs=(inner,),
               shape=inner.shape, dtype=node.dtype, arg=node.arg)


@register_vmap(OP_CAST)
def _vmap_cast(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    inner = batched[id(node.inputs[0])]
    return UOp(op=OP_CAST, inputs=(inner,),
               shape=inner.shape, dtype=node.dtype, arg=node.arg)


@register_vmap(OP_CMP)
def _vmap_cmp(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    a = batched[id(node.inputs[0])]
    b = batched[id(node.inputs[1])]
    new_shape = _broadcast(a.shape, b.shape)
    return UOp(op=OP_CMP, inputs=(a, b),
               shape=new_shape, dtype=node.dtype, arg=node.arg)


@register_vmap(OP_WHERE)
def _vmap_where(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    cond = batched[id(node.inputs[0])]
    a = batched[id(node.inputs[1])]
    b = batched[id(node.inputs[2])]
    new_shape = _broadcast(cond.shape, a.shape, b.shape)
    return UOp(op=OP_WHERE, inputs=(cond, a, b),
               shape=new_shape, dtype=node.dtype, arg=node.arg)


# Compute --------------------------------------------------------------


def _matmul_out_shape(a_shape: Tuple[int, ...], b_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Resolve the output shape of a @ b given the batched inputs."""
    if len(a_shape) < 2 and len(b_shape) < 2:
        raise ValueError(f"vmap matmul: both inputs are vectors: {a_shape}, {b_shape}")
    if len(a_shape) == 1 and len(b_shape) >= 2:
        return b_shape[:-2] + (b_shape[-1],)
    if len(a_shape) >= 2 and len(b_shape) == 1:
        return a_shape[:-1]
    # Both have ≥2 dims: leading dims broadcast, last two contract.
    lead = _broadcast(a_shape[:-2], b_shape[:-2])
    return lead + (a_shape[-2], b_shape[-1])


@register_vmap(OP_MATMUL)
def _vmap_matmul(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    """NumPy's `@` broadcasts over leading batch dims correctly. We emit
    a MATMUL with batched shapes; the realizer's `_h_matmul` Just Works."""
    a = batched[id(node.inputs[0])]
    b = batched[id(node.inputs[1])]
    new_shape = _matmul_out_shape(a.shape, b.shape)
    return UOp(op=OP_MATMUL, inputs=(a, b), shape=new_shape,
               dtype=node.dtype, arg=node.arg)


# Reduce ---------------------------------------------------------------


@register_vmap(OP_REDUCE)
def _vmap_reduce(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    """REDUCE axis shifts +1 (batch is the new leading dim, which we
    must NOT reduce over). axis=None becomes "reduce all but batch":
    explicit tuple of (1, 2, ..., ndim_in)."""
    inner = batched[id(node.inputs[0])]
    arg = dict(node.arg)
    axis = arg.get("axis")
    keepdims = arg.get("keepdims", False)
    in_ndim = len(node.inputs[0].shape)  # un-batched input ndim
    if axis is None:
        # Reduce over every non-batch axis.
        new_axis = tuple(range(1, in_ndim + 1))
    elif isinstance(axis, int):
        new_axis = (axis + 1,) if axis >= 0 else (axis,)
    else:
        new_axis = tuple((a + 1) if a >= 0 else a for a in axis)
    arg["axis"] = new_axis
    # Output shape: prepend B; for the rest, drop the reduced axes
    # unless keepdims=True.
    out_dims = [B]
    for i, d in enumerate(node.inputs[0].shape):
        batched_axis = i + 1
        if batched_axis in new_axis or (batched_axis - in_ndim - 1) in new_axis:
            if keepdims:
                out_dims.append(1)
        else:
            out_dims.append(d)
    new_shape = tuple(out_dims)
    return UOp(op=OP_REDUCE, inputs=(inner,), shape=new_shape,
               dtype=node.dtype, arg=arg)


# Shape ----------------------------------------------------------------


@register_vmap(OP_RESHAPE)
def _vmap_reshape(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    inner = batched[id(node.inputs[0])]
    new_shape = _batched_shape(node.arg["new_shape"], B)
    return UOp(op=OP_RESHAPE, inputs=(inner,), shape=new_shape,
               dtype=node.dtype,
               arg={**node.arg, "new_shape": new_shape})


@register_vmap(OP_PERMUTE)
def _vmap_permute(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    """Shift every axis index by +1; the new axis-0 (batch) stays at 0."""
    inner = batched[id(node.inputs[0])]
    shifted = (0,) + tuple(a + 1 for a in node.arg["axes"])
    new_shape = tuple(inner.shape[a] for a in shifted)
    return UOp(op=OP_PERMUTE, inputs=(inner,), shape=new_shape,
               dtype=node.dtype, arg={**node.arg, "axes": shifted})


@register_vmap(OP_BROADCAST_TO)
def _vmap_broadcast_to(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    inner = batched[id(node.inputs[0])]
    new_shape = _batched_shape(node.arg["shape"], B)
    return UOp(op=OP_BROADCAST_TO, inputs=(inner,), shape=new_shape,
               dtype=node.dtype, arg={**node.arg, "shape": new_shape})


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def vmap(
    fn: Callable[..., Any],
    in_dims: Union[int, Tuple[int, ...]] = 0,
    out_dims: int = 0,
) -> Callable[..., Any]:
    """JAX-style batching transform.

    Limitations (v0):
      * `in_dims=0` only — batch must be the leading axis on every
        TensorProxy input. Other axes raise.
      * `out_dims=0` only — output's batch ends up at axis 0.
      * Supported opcodes: BUFFER, LOAD, CONST, ADD, MUL, DIV, NEG,
        EXP, LOG, CMP, WHERE, CAST, MATMUL, REDUCE, RESHAPE, PERMUTE,
        BROADCAST_TO. Anything else raises with a pointer to PRD-014b.
    """
    if isinstance(in_dims, tuple):
        if not all(d == 0 for d in in_dims):
            raise JitNotImplementedError(
                f"bg.func.vmap: in_dims must be 0 (or tuple of 0s) in v0. "
                f"Got {in_dims}. Arbitrary axis support lands in PRD-014b."
            )
    elif in_dims != 0:
        raise JitNotImplementedError(
            f"bg.func.vmap: in_dims=0 only in v0 (got {in_dims}). "
            f"PRD-014b will add arbitrary-axis support."
        )
    if out_dims != 0:
        raise JitNotImplementedError(
            f"bg.func.vmap: out_dims=0 only in v0 (got {out_dims})."
        )

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        from ._tensor_proxy import TensorProxy, from_numpy
        from ._ir import buffer as _buffer_uop, load as _load_uop

        # 1. Resolve the batch size from the first TensorProxy input's
        #    leading dim.
        B = None
        for a in args:
            if isinstance(a, TensorProxy):
                if a.ndim == 0:
                    raise ValueError(
                        "bg.func.vmap: input has ndim=0, no batch axis to map"
                    )
                if B is None:
                    B = a.shape[0]
                elif a.shape[0] != B:
                    raise ValueError(
                        f"bg.func.vmap: batch sizes disagree across inputs "
                        f"({B} vs {a.shape[0]})"
                    )
        if B is None:
            raise TypeError(
                "bg.func.vmap: no TensorProxy inputs; nothing to map over"
            )

        # 2. Run `fn` with un-batched inputs (slice index 0) to get a
        #    reference IR. This re-tracing is acceptable: TensorProxy
        #    construction is cheap and the IR captures the shape-
        #    polymorphic semantics we'll rebatch.
        unbatched_args = []
        sess = None
        for a in args:
            if isinstance(a, TensorProxy):
                # Strip the batch dim: take any slice (e.g., zero index)
                # to construct the un-batched shape. We don't realize —
                # we want a TensorProxy whose shape matches the un-batched
                # function signature.
                # Cheapest: use the existing tensor's first row as a fresh
                # leaf with the un-batched shape, materialised once.
                arr = a.numpy()
                unbatched_args.append(from_numpy(arr[0], session=a._get_session()))
                if sess is None:
                    sess = a._get_session()
            else:
                unbatched_args.append(a)

        out = fn(*unbatched_args, **kwargs)
        if not isinstance(out, TensorProxy):
            raise TypeError(
                f"bg.func.vmap: fn must return a TensorProxy, got "
                f"{type(out).__name__}"
            )

        # 3. Walk the IR. For every leaf BUFFER that came from one of
        #    the original batched inputs, swap it for the full batched
        #    LOAD(BUFFER). For everything else, apply the per-op rule.
        order = toposort(out._uop)
        batched_map: Dict[int, UOp] = {}

        # Map un-batched leaf BUFFER id → batched BUFFER UOp from the
        # original input. We do this by indexing into args/unbatched_args
        # in parallel and walking the un-batched proxy's LOAD chain.
        leaf_swap: Dict[int, UOp] = {}
        for orig, unbat in zip(args, unbatched_args):
            if not isinstance(orig, TensorProxy):
                continue
            u = unbat._uop
            if u.op == OP_LOAD:
                u = u.inputs[0]
            if u.op == OP_BUFFER:
                # Map this un-batched BUFFER to the original (batched) BUFFER.
                orig_u = orig._uop
                if orig_u.op == OP_LOAD:
                    orig_u = orig_u.inputs[0]
                leaf_swap[id(u)] = orig_u

        for node in order:
            if id(node) in batched_map:
                continue
            if node.op == OP_BUFFER and id(node) in leaf_swap:
                # Replace with the original batched BUFFER.
                batched_map[id(node)] = leaf_swap[id(node)]
                continue
            rule = get_vmap_rule(node.op)
            if rule is None:
                raise JitNotImplementedError(
                    f"bg.func.vmap: opcode {node.op!r} has no batching rule "
                    f"in v0. Supported set: {sorted(_VMAP_RULES)}. "
                    f"Additional rules land in PRD-014b."
                )
            batched_map[id(node)] = rule(node, batched_map, B)

        batched_out_uop = batched_map[id(out._uop)]
        return TensorProxy(batched_out_uop, session=sess,
                           requires_grad=out.requires_grad)

    return wrapped


__all__ = ["vmap", "register_vmap", "get_vmap_rule"]
