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

What we don't (raises): MASK, RANDOM, CUSTOM,
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
    OP_ADD, OP_MUL, OP_DIV, OP_NEG, OP_EXP, OP_LOG,
    OP_ABS, OP_CLAMP, OP_COS, OP_FLIP, OP_CUMSUM, OP_CONCAT, OP_STACK, OP_NARROW, OP_TRIL, OP_TRIU, OP_PROD, OP_VAR, OP_REPEAT, OP_REPEAT_INTERLEAVE,
    OP_SIGN, OP_SIN, OP_CMP,
    OP_MATMUL, OP_CONV1D, OP_CONV1D_BACKWARD_INPUT,
    OP_CONV1D_BACKWARD_WEIGHT, OP_CONV1D_BACKWARD_BIAS,
    OP_CONV2D, OP_CONV2D_BACKWARD_INPUT,
    OP_CONV2D_BACKWARD_WEIGHT, OP_CONV2D_BACKWARD_BIAS,
    OP_CONV_TRANSPOSE2D, OP_CONV_TRANSPOSE2D_BACKWARD_INPUT,
    OP_CONV_TRANSPOSE2D_BACKWARD_WEIGHT, OP_CONV_TRANSPOSE2D_BACKWARD_BIAS,
    OP_CONV3D, OP_CONV3D_BACKWARD_INPUT,
    OP_CONV3D_BACKWARD_WEIGHT, OP_CONV3D_BACKWARD_BIAS,
    OP_LAYER_NORM, OP_LAYER_NORM_BACKWARD_INPUT,
    OP_LAYER_NORM_BACKWARD_WEIGHT, OP_LAYER_NORM_BACKWARD_BIAS,
    OP_REDUCE, OP_RESHAPE, OP_PERMUTE,
    OP_WHERE, OP_BROADCAST_TO, OP_ISNAN, OP_SGD_UPDATE,
    OP_ADAMW_UPDATE_M, OP_ADAMW_UPDATE_V, OP_ADAMW_UPDATE_PARAM,
    OP_ADAM_UPDATE_M, OP_ADAM_UPDATE_V, OP_ADAM_UPDATE_PARAM,
    OP_PAD, OP_SLICE, OP_SORT_INDICES, OP_SORT_VALUES,
    OP_TOPK_INDICES, OP_TOPK_VALUES, OP_SCATTER, OP_EINSUM, OP_EINSUM_VJP,
    OP_FUSED_ELEMENTWISE, OP_FUSED_SOFTMAX,
    OP_SCATTER_ADD, OP_INDEX, OP_MASK, OP_RANDOM, OP_CUSTOM,
    OP_STORE,
)
from ._errors import JitNotImplementedError
from ._framework_contracts import (
    MASKED_FILL_CONTRACT_ID,
    validate_broadcast_to_contract,
    validate_cat_contract,
    validate_stack_contract,
    validate_pad_contract,
    validate_sort_indices_contract,
    validate_sort_values_contract,
    validate_topk_indices_contract,
    validate_topk_values_contract,
    validate_scatter_contract,
    validate_einsum_contract,
    validate_einsum_vjp_contract,
    validate_clamp_contract,
    validate_cumsum_contract,
    validate_flip_contract,
    validate_tril_contract,
    validate_triu_contract,
    validate_gather_contract,
    validate_gather_scatter_add_contract,
    validate_narrow_contract,
    validate_prod_contract,
    validate_var_contract,
    validate_where_contract,
    validate_repeat_contract,
    validate_repeat_interleave_contract,
    validate_typed_unary_contract,
    infer_einsum_contract,
)


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


def _vmap_typed_unary(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    validate_typed_unary_contract(node)
    inner = batched[id(node.inputs[0])]
    return UOp(
        op=node.op,
        inputs=(inner,),
        shape=inner.shape,
        dtype=node.dtype,
        arg=node.arg,
    )


_VMAP_RULES[OP_ABS] = _vmap_typed_unary
_VMAP_RULES[OP_COS] = _vmap_typed_unary
_VMAP_RULES[OP_SIGN] = _vmap_typed_unary
_VMAP_RULES[OP_SIN] = _vmap_typed_unary


@register_vmap(OP_CLAMP)
def _vmap_clamp(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    validate_clamp_contract(node)
    inner = batched[id(node.inputs[0])]
    return UOp(
        op=OP_CLAMP,
        inputs=(inner,),
        shape=inner.shape,
        dtype=node.dtype,
        arg=node.arg,
    )


@register_vmap(OP_FLIP)
def _vmap_flip(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    axis = validate_flip_contract(node)
    inner = batched[id(node.inputs[0])]
    return UOp(
        op=OP_FLIP,
        inputs=(inner,),
        shape=inner.shape,
        dtype=node.dtype,
        arg={"axis": axis + 1},
    )


@register_vmap(OP_CUMSUM)
def _vmap_cumsum(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    axis, reverse = validate_cumsum_contract(node)
    inner = batched[id(node.inputs[0])]
    if len(inner.shape) != len(node.inputs[0].shape) + 1 or inner.shape[0] != B:
        raise JitNotImplementedError(
            "vmap cumsum requires the source tensor on the leading mapped axis"
        )
    return UOp(
        op=OP_CUMSUM,
        inputs=(inner,),
        shape=inner.shape,
        dtype=node.dtype,
        arg={"axis": axis + 1, "reverse": reverse},
    )


def _vmap_variadic_inputs(
    inputs: Tuple[UOp, ...],
    batched: Dict[int, UOp],
    B: int,
    operation: str,
    included: Tuple[bool, ...],
) -> Tuple[UOp, ...]:
    mapped_inputs = []
    for source, keep in zip(inputs, included):
        if not keep:
            continue
        inner = batched[id(source)]
        if inner.shape == (B,) + source.shape:
            mapped_inputs.append(inner)
            continue
        if inner.shape != source.shape:
            raise JitNotImplementedError(
                f"vmap {operation} requires each source to be captured or on "
                "the leading mapped axis"
            )
        singleton_shape = (1,) + source.shape
        reshaped = UOp(
            op=OP_RESHAPE,
            inputs=(inner,),
            shape=singleton_shape,
            dtype=source.dtype,
            arg={"new_shape": singleton_shape},
        )
        target_shape = (B,) + source.shape
        mapped_inputs.append(UOp(
            op=OP_BROADCAST_TO,
            inputs=(reshaped,),
            shape=target_shape,
            dtype=source.dtype,
            arg={"shape": target_shape},
        ))
    return tuple(mapped_inputs)


@register_vmap(OP_EINSUM)
def _vmap_einsum(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    contract = validate_einsum_contract(node)
    mapped_inputs = _vmap_variadic_inputs(
        node.inputs,
        batched,
        B,
        "einsum",
        tuple(True for _ in node.inputs),
    )
    mapped_contract = infer_einsum_contract(
        mapped_inputs,
        contract.equation,
        contract.batch_rank + 1,
    )
    return UOp(
        op=OP_EINSUM,
        inputs=mapped_inputs,
        shape=mapped_contract.output_shape,
        dtype=mapped_contract.output_dtype,
        arg={
            "equation": mapped_contract.equation,
            "batch_rank": mapped_contract.batch_rank,
        },
    )


@register_vmap(OP_EINSUM_VJP)
def _vmap_einsum_vjp(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    contract, operand = validate_einsum_vjp_contract(node)
    mapped_inputs = _vmap_variadic_inputs(
        node.inputs,
        batched,
        B,
        "einsum_vjp",
        tuple(True for _ in node.inputs),
    )
    mapped_contract = infer_einsum_contract(
        mapped_inputs[1:],
        contract.equation,
        contract.batch_rank + 1,
    )
    arg = {
        "equation": mapped_contract.equation,
        "batch_rank": mapped_contract.batch_rank,
        "operand": operand,
    }
    if "vjp_of" in node.arg:
        arg["vjp_of"] = node.arg["vjp_of"]
    return UOp(
        op=OP_EINSUM_VJP,
        inputs=mapped_inputs,
        shape=mapped_inputs[operand + 1].shape,
        dtype=mapped_inputs[operand + 1].dtype,
        arg=arg,
    )


@register_vmap(OP_CONCAT)
def _vmap_concat(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    axis, _, legacy_empty = validate_cat_contract(node)
    included = tuple(not (empty and len(node.shape) != 1) for empty in legacy_empty)
    mapped_inputs = _vmap_variadic_inputs(
        node.inputs,
        batched,
        B,
        "concat",
        included,
    )
    return UOp(
        op=OP_CONCAT,
        inputs=mapped_inputs,
        shape=(B,) + node.shape,
        dtype=node.dtype,
        arg={"axis": axis + 1},
    )


@register_vmap(OP_STACK)
def _vmap_stack(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    axis = validate_stack_contract(node)
    mapped_inputs = _vmap_variadic_inputs(
        node.inputs,
        batched,
        B,
        "stack",
        (True,) * len(node.inputs),
    )
    return UOp(
        op=OP_STACK,
        inputs=mapped_inputs,
        shape=(B,) + node.shape,
        dtype=node.dtype,
        arg={"axis": axis + 1},
    )


@register_vmap(OP_NARROW)
def _vmap_narrow(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    axis, start, length = validate_narrow_contract(node)
    source = node.inputs[0]
    inner = batched[id(source)]
    if inner.shape == (B,) + source.shape:
        return UOp(
            op=OP_NARROW,
            inputs=(inner,),
            shape=(B,) + node.shape,
            dtype=node.dtype,
            arg={"axis": axis + 1, "start": start, "length": length},
        )
    if inner.shape != source.shape:
        raise JitNotImplementedError(
            "vmap narrow requires its source to be captured or on the leading mapped axis"
        )
    singleton_shape = (1,) + node.shape
    reshaped = UOp(
        op=OP_RESHAPE,
        inputs=(node,),
        shape=singleton_shape,
        dtype=node.dtype,
        arg={"new_shape": singleton_shape},
    )
    target_shape = (B,) + node.shape
    return UOp(
        op=OP_BROADCAST_TO,
        inputs=(reshaped,),
        shape=target_shape,
        dtype=node.dtype,
        arg={"shape": target_shape},
    )


@register_vmap(OP_TRIL)
def _vmap_tril(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    diagonal = validate_tril_contract(node)
    inner = batched[id(node.inputs[0])]
    if len(inner.shape) != len(node.inputs[0].shape) + 1 or inner.shape[0] != B:
        raise JitNotImplementedError(
            "vmap tril requires the source tensor on the leading mapped axis"
        )
    return UOp(
        op=OP_TRIL,
        inputs=(inner,),
        shape=inner.shape,
        dtype=node.dtype,
        arg={"diagonal": diagonal},
    )


@register_vmap(OP_TRIU)
def _vmap_triu(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    diagonal = validate_triu_contract(node)
    inner = batched[id(node.inputs[0])]
    if len(inner.shape) != len(node.inputs[0].shape) + 1 or inner.shape[0] != B:
        raise JitNotImplementedError(
            "vmap triu requires the source tensor on the leading mapped axis"
        )
    return UOp(
        op=OP_TRIU,
        inputs=(inner,),
        shape=inner.shape,
        dtype=node.dtype,
        arg={"diagonal": diagonal},
    )


@register_vmap(OP_REPEAT)
def _vmap_repeat(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    repeats, padded_shape = validate_repeat_contract(node)
    inner = batched[id(node.inputs[0])]
    is_batched = len(inner.shape) > len(node.inputs[0].shape)
    if is_batched:
        if padded_shape != node.inputs[0].shape:
            reshaped = (B,) + padded_shape
            inner = UOp(
                op=OP_RESHAPE,
                inputs=(inner,),
                shape=reshaped,
                dtype=node.dtype,
                arg={"new_shape": reshaped},
            )
        mapped_repeats = (1,) + repeats
        mapped_shape = (B,) + node.shape
    else:
        mapped_repeats = repeats
        mapped_shape = node.shape
    return UOp(
        op=OP_REPEAT,
        inputs=(inner,),
        shape=mapped_shape,
        dtype=node.dtype,
        arg={"repeats": mapped_repeats},
    )


@register_vmap(OP_PROD)
def _vmap_prod(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    axes, keepdims, _ = validate_prod_contract(node)
    inner = batched[id(node.inputs[0])]
    is_batched = len(inner.shape) > len(node.inputs[0].shape)
    mapped_axes = tuple(axis + 1 for axis in axes) if is_batched else axes
    mapped_shape = (B,) + node.shape if is_batched else node.shape
    return UOp(
        op=OP_PROD,
        inputs=(inner,),
        shape=mapped_shape,
        dtype=node.dtype,
        arg={"axes": mapped_axes, "keepdims": keepdims},
    )


@register_vmap(OP_VAR)
def _vmap_var(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    axes, correction, keepdims, _, _ = validate_var_contract(node)
    inner = batched[id(node.inputs[0])]
    is_batched = len(inner.shape) > len(node.inputs[0].shape)
    mapped_axes = tuple(axis + 1 for axis in axes) if is_batched else axes
    mapped_shape = (B,) + node.shape if is_batched else node.shape
    return UOp(
        op=OP_VAR,
        inputs=(inner,),
        shape=mapped_shape,
        dtype=node.dtype,
        arg={
            "axes": mapped_axes,
            "correction": correction,
            "keepdims": keepdims,
        },
    )


@register_vmap(OP_REPEAT_INTERLEAVE)
def _vmap_repeat_interleave(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    repeats, axis = validate_repeat_interleave_contract(node)
    inner = batched[id(node.inputs[0])]
    is_batched = len(inner.shape) > len(node.inputs[0].shape)
    mapped_axis = axis + 1 if is_batched else axis
    mapped_shape = (B,) + node.shape if is_batched else node.shape
    return UOp(
        op=OP_REPEAT_INTERLEAVE,
        inputs=(inner,),
        shape=mapped_shape,
        dtype=node.dtype,
        arg={"repeats": repeats, "axis": mapped_axis},
    )


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
    validate_where_contract(node)
    cond = batched[id(node.inputs[0])]
    a = batched[id(node.inputs[1])]
    b = batched[id(node.inputs[2])]
    if node.arg == MASKED_FILL_CONTRACT_ID:
        source_was_mapped = len(b.shape) == len(node.inputs[2].shape) + 1
        if not source_was_mapped or not b.shape or b.shape[0] != B:
            raise JitNotImplementedError(
                "vmap masked_fill requires the source tensor on the leading mapped axis"
            )
    new_shape = _broadcast(cond.shape, a.shape, b.shape)
    mapped = UOp(op=OP_WHERE, inputs=(cond, a, b),
                 shape=new_shape, dtype=node.dtype, arg=node.arg)
    validate_where_contract(mapped)
    return mapped


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
    """If the input was un-batched (pass-through scalar / const), the
    reshape keeps its original target shape. Otherwise we prepend B.

    The original IR's `node.inputs[0].shape` tells us what was expected;
    `inner.shape` tells us what arrived. Same ndim → unbatched; +1 ndim
    → batched (and we prepend B).
    """
    inner = batched[id(node.inputs[0])]
    orig_input_ndim = len(node.inputs[0].shape)
    is_batched = len(inner.shape) > orig_input_ndim
    if is_batched:
        new_shape = _batched_shape(node.arg["new_shape"], B)
    else:
        new_shape = tuple(node.arg["new_shape"])
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


# Remaining v0 rules: ISNAN (passthrough), PAD (prepend (0,0)),
# SLICE (prepend slice(None)), FUSED_SOFTMAX (shift axis),
# FUSED_ELEMENTWISE (re-broadcast over batched inputs), INDEX (shift dim),
# SCATTER_ADD (shift dim).
# Refused (need richer semantics): RANDOM, MASK, CUSTOM, STORE.


@register_vmap(OP_ISNAN)
def _vmap_isnan(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    inner = batched[id(node.inputs[0])]
    return UOp(op=OP_ISNAN, inputs=(inner,),
               shape=inner.shape, dtype=node.dtype, arg=node.arg)


@register_vmap(OP_PAD)
def _vmap_pad(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    """Preserve the leading mapped axis outside the declared pad."""
    pad_width, value = validate_pad_contract(node)
    source = node.inputs[0]
    inner = batched[id(node.inputs[0])]
    if inner.shape == source.shape:
        return node
    if inner.shape != (B,) + source.shape:
        raise JitNotImplementedError(
            "vmap pad requires its source to be captured or on the leading mapped axis"
        )
    mapped_pad_width = ((0, 0),) + pad_width
    return UOp(
        op=OP_PAD,
        inputs=(inner,),
        shape=(B,) + node.shape,
        dtype=node.dtype,
        arg={"pad_width": mapped_pad_width, "mode": "constant", "value": value},
    )


@register_vmap(OP_SORT_INDICES)
def _vmap_sort_indices(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    """Shift the logical sort axis past one leading mapped axis."""
    axis, descending, stable = validate_sort_indices_contract(node)
    source = node.inputs[0]
    inner = batched[id(source)]
    if inner.shape == source.shape:
        return node
    if inner.shape != (B,) + source.shape:
        raise JitNotImplementedError(
            "vmap sort requires its source to be captured or on the leading mapped axis"
        )
    if not source.shape:
        zero = UOp(OP_CONST, (), (), "int64", arg={"value": 0})
        return UOp(
            OP_BROADCAST_TO,
            (zero,),
            (B,),
            "int64",
            arg={"shape": (B,)},
        )
    return UOp(
        op=OP_SORT_INDICES,
        inputs=(inner,),
        shape=(B,) + node.shape,
        dtype="int64",
        arg={"axis": axis + 1, "descending": descending, "stable": stable},
    )


@register_vmap(OP_SORT_VALUES)
def _vmap_sort_values(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    """Map the source and its paired ordering together."""
    axis, descending, stable = validate_sort_values_contract(node)
    source, indices = node.inputs
    mapped_source = batched[id(source)]
    mapped_indices = batched[id(indices)]
    source_batched = mapped_source.shape != source.shape
    indices_batched = mapped_indices.shape != indices.shape
    if source_batched != indices_batched:
        raise JitNotImplementedError(
            "vmap sort values require source and indices to share the leading batch axis"
        )
    if not source_batched:
        return node
    if mapped_source.shape != (B,) + source.shape:
        raise JitNotImplementedError(
            "vmap sort requires its source on the leading mapped axis"
        )
    if not source.shape:
        return mapped_source
    return UOp(
        op=OP_SORT_VALUES,
        inputs=(mapped_source, mapped_indices),
        shape=(B,) + node.shape,
        dtype=node.dtype,
        arg={"axis": axis + 1, "descending": descending, "stable": stable},
    )


@register_vmap(OP_TOPK_INDICES)
def _vmap_topk_indices(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    """Shift a top-k selection axis past one leading mapped axis."""
    axis, k, largest, sorted_output = validate_topk_indices_contract(node)
    source = node.inputs[0]
    inner = batched[id(source)]
    if inner.shape == source.shape:
        return node
    if inner.shape != (B,) + source.shape:
        raise JitNotImplementedError(
            "vmap topk requires its source to be captured or on the leading mapped axis"
        )
    return UOp(
        op=OP_TOPK_INDICES,
        inputs=(inner,),
        shape=(B,) + node.shape,
        dtype="int64",
        arg={
            "axis": axis + 1,
            "k": k,
            "largest": largest,
            "sorted": sorted_output,
        },
    )


@register_vmap(OP_TOPK_VALUES)
def _vmap_topk_values(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    """Map the source and paired top-k indices together."""
    axis, k, largest, sorted_output = validate_topk_values_contract(node)
    source, indices = node.inputs
    mapped_source = batched[id(source)]
    mapped_indices = batched[id(indices)]
    source_batched = mapped_source.shape != source.shape
    indices_batched = mapped_indices.shape != indices.shape
    if source_batched != indices_batched:
        raise JitNotImplementedError(
            "vmap topk values require source and indices to share the leading batch axis"
        )
    if not source_batched:
        return node
    if mapped_source.shape != (B,) + source.shape:
        raise JitNotImplementedError(
            "vmap topk requires its source on the leading mapped axis"
        )
    return UOp(
        op=OP_TOPK_VALUES,
        inputs=(mapped_source, mapped_indices),
        shape=(B,) + node.shape,
        dtype=node.dtype,
        arg={
            "axis": axis + 1,
            "k": k,
            "largest": largest,
            "sorted": sorted_output,
        },
    )


@register_vmap(OP_SLICE)
def _vmap_slice(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    """Prepend slice(None) for the batch dim. Original slices stay."""
    inner = batched[id(node.inputs[0])]
    orig_input_ndim = len(node.inputs[0].shape)
    is_batched = len(inner.shape) > orig_input_ndim
    arg = dict(node.arg)
    slices = list(arg.get("slices", ()))
    if is_batched:
        slices = [slice(None)] + slices
    arg["slices"] = slices
    # Output shape: batch dim from inner + sliced unbatched dims.
    new_shape = list(node.shape)
    if is_batched:
        new_shape = [B] + new_shape
    return UOp(op=OP_SLICE, inputs=(inner,), shape=tuple(new_shape),
               dtype=node.dtype, arg=arg)


@register_vmap(OP_FUSED_SOFTMAX)
def _vmap_fused_softmax(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    inner = batched[id(node.inputs[0])]
    arg = dict(node.arg)
    axis = arg.get("axis", -1)
    # Shift positive axis by +1 (only when input was batched).
    orig_input_ndim = len(node.inputs[0].shape)
    is_batched = len(inner.shape) > orig_input_ndim
    if is_batched and isinstance(axis, int) and axis >= 0:
        arg["axis"] = axis + 1
    return UOp(op=OP_FUSED_SOFTMAX, inputs=(inner,),
               shape=inner.shape, dtype=node.dtype, arg=arg)


@register_vmap(OP_FUSED_ELEMENTWISE)
def _vmap_fused_elementwise(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    """All inputs gain (or keep) their shapes; the ops list stays the
    same. The realizer's _h_fused_elementwise loops over flat indices
    so it Just Works with the new batched shapes."""
    new_inputs = tuple(batched[id(inp)] for inp in node.inputs)
    # Output shape: broadcast all batched input shapes.
    new_shape = _broadcast(*[i.shape for i in new_inputs])
    return UOp(op=OP_FUSED_ELEMENTWISE, inputs=new_inputs,
               shape=new_shape, dtype=node.dtype, arg=node.arg)


@register_vmap(OP_SCATTER)
def _vmap_scatter(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    """Batch overwrite scatter and broadcast captured target/index/source."""
    axis = validate_scatter_contract(node)
    target, index, source = node.inputs
    mapped_target, mapped_index = _vmap_variadic_inputs(
        (target, index),
        batched,
        B,
        "scatter",
        (True, True),
    )
    mapped_source = batched[id(source)]
    if source.shape == ():
        if mapped_source.shape != ():
            raise JitNotImplementedError(
                "vmap scatter does not accept a mapped tensor scalar source; "
                "use the scalar-value overload or an index-shaped source"
            )
    else:
        (mapped_source,) = _vmap_variadic_inputs(
            (source,),
            batched,
            B,
            "scatter",
            (True,),
        )
    mapped = UOp(
        op=OP_SCATTER,
        inputs=(mapped_target, mapped_index, mapped_source),
        shape=(B,) + node.shape,
        dtype=node.dtype,
        arg={**node.arg, "dim": axis + 1},
    )
    validate_scatter_contract(mapped)
    return mapped


@register_vmap(OP_INDEX)
def _vmap_index(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    """Map source and index together and shift gather dim past batch."""
    axis = validate_gather_contract(node)
    data = batched[id(node.inputs[0])]
    idx = batched[id(node.inputs[1])]
    data_batched = len(data.shape) > len(node.inputs[0].shape)
    index_batched = len(idx.shape) > len(node.inputs[1].shape)
    if data_batched != index_batched:
        raise JitNotImplementedError(
            "bg.func.vmap: INDEX requires source and index to share the leading batch axis"
        )
    mapped_axis = axis + 1 if data_batched else axis
    new_shape = (B,) + node.shape if data_batched else node.shape
    return UOp(op=OP_INDEX, inputs=(data, idx),
               shape=new_shape, dtype=node.dtype,
               arg={**node.arg, "dim": mapped_axis})


@register_vmap(OP_SCATTER_ADD)
def _vmap_scatter_add(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    """Map target, index, and source together and shift dim past batch."""
    axis = validate_gather_scatter_add_contract(node)
    target = batched[id(node.inputs[0])]
    idx = batched[id(node.inputs[1])]
    src = batched[id(node.inputs[2])]
    mapped = (
        len(target.shape) > len(node.inputs[0].shape),
        len(idx.shape) > len(node.inputs[1].shape),
        len(src.shape) > len(node.inputs[2].shape),
    )
    if len(set(mapped)) != 1:
        raise JitNotImplementedError(
            "bg.func.vmap: SCATTER_ADD requires target, index, and source to share "
            "the leading batch axis"
        )
    mapped_axis = axis + 1 if mapped[0] else axis
    return UOp(op=OP_SCATTER_ADD, inputs=(target, idx, src),
               shape=target.shape, dtype=node.dtype,
               arg={**node.arg, "dim": mapped_axis})


# Refusal stubs — these ops have semantics that don't translate
# trivially under vmap. We document the gap rather than silently
# producing wrong results.


def _refuse(op_name: str, reason: str):
    def _rule(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
        raise JitNotImplementedError(
            f"bg.func.vmap: {op_name} is not vmappable in v0 — {reason}. "
            f"Use a Python for-loop or refactor your function to avoid "
            f"this op."
        )
    return _rule


_VMAP_RULES[OP_RANDOM] = _refuse(
    "OP_RANDOM",
    "random sampling requires a per-invocation key split (JAX uses "
    "PRNGKey for this). v0 has no key split — every randn() call returns "
    "the same sequence across batches",
)

_VMAP_RULES[OP_MASK] = _refuse(
    "OP_MASK",
    "boolean indexing produces a data-dependent output shape; vmap "
    "needs a static shape across batches",
)

_VMAP_RULES[OP_CUSTOM] = _refuse(
    "OP_CUSTOM",
    "user-defined ops can have arbitrary semantics; provide a hand-"
    "written vmap rule via _vmap.register_vmap if you need it",
)

_VMAP_RULES[OP_CONV1D] = _refuse(
    "OP_CONV1D",
    "CNN batching needs an explicit rule for whether vmap maps over N, "
    "channels, groups, or an outer batch dim",
)

_VMAP_RULES[OP_CONV1D_BACKWARD_INPUT] = _refuse(
    "OP_CONV1D_BACKWARD_INPUT",
    "conv backward batching needs explicit semantics for mapped batch, "
    "channels, and groups",
)

_VMAP_RULES[OP_CONV1D_BACKWARD_WEIGHT] = _refuse(
    "OP_CONV1D_BACKWARD_WEIGHT",
    "conv backward batching needs explicit semantics for mapped batch, "
    "channels, and groups",
)

_VMAP_RULES[OP_CONV1D_BACKWARD_BIAS] = _refuse(
    "OP_CONV1D_BACKWARD_BIAS",
    "conv backward batching needs explicit semantics for mapped batch "
    "and output channels",
)

_VMAP_RULES[OP_CONV2D] = _refuse(
    "OP_CONV2D",
    "CNN batching needs an explicit rule for whether vmap maps over N, "
    "channels, groups, or an outer batch dim. Add that rule before using "
    "Conv2d under vmap",
)

_VMAP_RULES[OP_CONV2D_BACKWARD_INPUT] = _refuse(
    "OP_CONV2D_BACKWARD_INPUT",
    "conv backward batching needs explicit semantics for mapped batch, "
    "channels, and groups",
)

_VMAP_RULES[OP_CONV2D_BACKWARD_WEIGHT] = _refuse(
    "OP_CONV2D_BACKWARD_WEIGHT",
    "conv backward batching needs explicit semantics for mapped batch, "
    "channels, and groups",
)

_VMAP_RULES[OP_CONV2D_BACKWARD_BIAS] = _refuse(
    "OP_CONV2D_BACKWARD_BIAS",
    "conv backward batching needs explicit semantics for mapped batch "
    "and output channels",
)

_VMAP_RULES[OP_CONV_TRANSPOSE2D] = _refuse(
    "OP_CONV_TRANSPOSE2D",
    "transposed CNN batching needs an explicit rule for whether vmap maps over N, "
    "channels, groups, or an outer batch dim. Add that rule before using "
    "ConvTranspose2d under vmap",
)

_VMAP_RULES[OP_CONV_TRANSPOSE2D_BACKWARD_INPUT] = _refuse(
    "OP_CONV_TRANSPOSE2D_BACKWARD_INPUT",
    "transposed conv backward batching needs explicit semantics for mapped batch, "
    "channels, and groups",
)

_VMAP_RULES[OP_CONV_TRANSPOSE2D_BACKWARD_WEIGHT] = _refuse(
    "OP_CONV_TRANSPOSE2D_BACKWARD_WEIGHT",
    "transposed conv backward batching needs explicit semantics for mapped batch, "
    "channels, and groups",
)

_VMAP_RULES[OP_CONV_TRANSPOSE2D_BACKWARD_BIAS] = _refuse(
    "OP_CONV_TRANSPOSE2D_BACKWARD_BIAS",
    "transposed conv backward batching needs explicit semantics for mapped batch "
    "and output channels",
)

_VMAP_RULES[OP_CONV3D] = _refuse(
    "OP_CONV3D",
    "CNN batching needs an explicit rule for whether vmap maps over N, "
    "channels, groups, or an outer batch dim. Add that rule before using "
    "Conv3d under vmap",
)

_VMAP_RULES[OP_CONV3D_BACKWARD_INPUT] = _refuse(
    "OP_CONV3D_BACKWARD_INPUT",
    "conv backward batching needs explicit semantics for mapped batch, "
    "channels, groups, and volume axes",
)

_VMAP_RULES[OP_CONV3D_BACKWARD_WEIGHT] = _refuse(
    "OP_CONV3D_BACKWARD_WEIGHT",
    "conv backward batching needs explicit semantics for mapped batch, "
    "channels, groups, and volume axes",
)

_VMAP_RULES[OP_CONV3D_BACKWARD_BIAS] = _refuse(
    "OP_CONV3D_BACKWARD_BIAS",
    "conv backward batching needs explicit semantics for mapped batch "
    "and output channels",
)

_VMAP_RULES[OP_STORE] = _refuse(
    "OP_STORE",
    "STORE mutates a BUFFER; that's an autograd-time concern, not a "
    "vmap-time one. If you're seeing this, you constructed a graph "
    "with explicit STORE — restructure to use functional ops",
)

_VMAP_RULES[OP_SGD_UPDATE] = _refuse(
    "OP_SGD_UPDATE",
    "optimizer update batching needs explicit state semantics; vmap the "
    "loss/grad function, then apply optimizer updates outside vmap",
)

for _op_name, _op in (
    ("OP_ADAMW_UPDATE_M", OP_ADAMW_UPDATE_M),
    ("OP_ADAMW_UPDATE_V", OP_ADAMW_UPDATE_V),
    ("OP_ADAMW_UPDATE_PARAM", OP_ADAMW_UPDATE_PARAM),
    ("OP_ADAM_UPDATE_M", OP_ADAM_UPDATE_M),
    ("OP_ADAM_UPDATE_V", OP_ADAM_UPDATE_V),
    ("OP_ADAM_UPDATE_PARAM", OP_ADAM_UPDATE_PARAM),
):
    _VMAP_RULES[_op] = _refuse(
        _op_name,
        "Adam update batching needs explicit optimizer-state semantics; "
        "vmap the loss/grad function, then apply optimizer updates outside vmap",
    )

for _op_name, _op in (
    ("OP_LAYER_NORM", OP_LAYER_NORM),
    ("OP_LAYER_NORM_BACKWARD_INPUT", OP_LAYER_NORM_BACKWARD_INPUT),
    ("OP_LAYER_NORM_BACKWARD_WEIGHT", OP_LAYER_NORM_BACKWARD_WEIGHT),
    ("OP_LAYER_NORM_BACKWARD_BIAS", OP_LAYER_NORM_BACKWARD_BIAS),
):
    _VMAP_RULES[_op] = _refuse(
        _op_name,
        "LayerNorm batching needs normalized_shape-aware axis mapping; "
        "vmap an outer module over examples instead of batching this primitive directly",
    )


@register_vmap(OP_BROADCAST_TO)
def _vmap_broadcast_to(node: UOp, batched: Dict[int, UOp], B: int) -> UOp:
    """Same input-batched-or-not check as RESHAPE: if input is the
    un-batched pass-through, broadcast target stays as recorded."""
    validate_broadcast_to_contract(node)
    inner = batched[id(node.inputs[0])]
    orig_input_ndim = len(node.inputs[0].shape)
    is_batched = len(inner.shape) > orig_input_ndim
    if is_batched:
        new_shape = _batched_shape(node.arg["shape"], B)
    else:
        new_shape = tuple(node.arg["shape"])
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
      * Supported opcodes include BUFFER, LOAD, CONST, ADD, MUL, DIV, NEG,
        EXP, LOG, ABS, CLAMP, COS, FLIP, TRIL, TRIU, INDEX, REPEAT, VAR, SIGN, SIN, CMP, WHERE,
        CAST, MATMUL, REDUCE, RESHAPE, PERMUTE, SCATTER_ADD, and BROADCAST_TO. Anything else
        raises with a pointer to PRD-014b.
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
