"""browsergrad_jit._vjp — symbolic vector-Jacobian-product rules.

INTERNAL. PRD-007 W1 deliverable.

Each rule is a pure function that, given the forward UOp's output, its
input UOps, and the upstream gradient UOp `dy`, emits one new UOp per
input representing that input's gradient.

Design contract (per PRD-007 DL/GPU review):

  * **Reference forward outputs explicitly when the math allows.** EXP's
    VJP is `dy * output`, not `dy * exp(input)`. This lets PRD-006/012
    see the forward EXP and the backward MUL as adjacent in the joint
    graph and fuse them — the Flash Attention v2 reuse pattern that
    avoids a DRAM round-trip for `y = exp(x)` in attention softmax.

  * **Annotate every VJP-emitted UOp with `arg["vjp_of"] = forward_uop`.**
    PRD-009 (gradient checkpointing) walks the backward graph and uses
    this tag to identify recompute candidates. Without it, the backward
    IR is opaque to PRD-009.

  * **Never compute on realized values.** A VJP rule reads forward UOp
    shape/dtype (metadata) and emits IR. It does not call NumPy. This
    invariant lets PRD-009 re-run the forward at backward time without
    breaking VJP rule semantics.

  * **Un-broadcast** at op boundaries: gradient may arrive at a shape
    larger than the input it's about to flow into (broadcast expanded
    the input). The rule sums dy over expanded axes before returning.
    `_unbroadcast_uop(dy, target_shape)` handles this.

  * **Conformance**: per-op VJPs match PyTorch's `derivatives.yaml`
    semantics for the shipped ops. Tie-breaking on REDUCE(max) follows
    PyTorch (split gradient equally among tied positions).

  * **Fallback**: if an opcode has no registered VJP, the closure path
    in `_tensor_proxy.backward()` runs instead. This allows incremental
    migration — Week 1 lands 12 rules; later weeks add the rest.

  * **Higher-order**: v0 is first-order. Many emitted UOps have registered
    VJPs, but specialized backward primitives such as CONV*_BACKWARD_* do
    not. Higher-order conv grads must fail explicitly until those rules land.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple

from ._ir import (
    UOp,
    OP_ADD, OP_MUL, OP_DIV, OP_NEG, OP_EXP, OP_LOG,
    OP_ABS, OP_CLAMP, OP_COS, OP_FLIP, OP_CUMSUM, OP_CONCAT, OP_STACK, OP_NARROW, OP_TRIL, OP_TRIU, OP_PROD, OP_VAR, OP_REPEAT, OP_REPEAT_INTERLEAVE,
    OP_SIGN, OP_SIN, OP_CAST, OP_CMP,
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
    OP_REDUCE, OP_RESHAPE, OP_PERMUTE, OP_SLICE, OP_PAD,
    OP_SORT_INDICES, OP_SORT_VALUES,
    OP_TOPK_INDICES, OP_TOPK_VALUES, OP_SCATTER,
    OP_CONST, OP_BROADCAST_TO, OP_WHERE, OP_INDEX, OP_SCATTER_ADD,
    OP_ISNAN,
)
from ._framework_contracts import (
    validate_broadcast_to_contract,
    validate_cat_contract,
    validate_stack_contract,
    validate_pad_contract,
    validate_sort_indices_contract,
    validate_sort_values_contract,
    validate_topk_indices_contract,
    validate_topk_values_contract,
    validate_scatter_contract,
    validate_clamp_contract,
    validate_cumsum_contract,
    validate_flip_contract,
    validate_tril_contract,
    validate_triu_contract,
    validate_gather_contract,
    validate_prod_contract,
    validate_var_contract,
    validate_where_contract,
    validate_repeat_contract,
    validate_repeat_interleave_contract,
    validate_real_numeric_unary_contract,
    validate_typed_unary_contract,
)


# ---------------------------------------------------------------------------
# Rule signature
#
#   (output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]
#
# Output ordering matches `inputs` ordering. Return `None` at position i
# to indicate "input i has no gradient" (non-differentiable, e.g. the
# condition of WHERE).
# ---------------------------------------------------------------------------


VJPRule = Callable[[UOp, Tuple[UOp, ...], UOp], Tuple[Optional[UOp], ...]]


_VJP_RULES: Dict[str, VJPRule] = {}


def register_vjp(op: str) -> Callable[[VJPRule], VJPRule]:
    """Decorator: register `fn` as the VJP rule for opcode `op`.

    Multiple registrations for the same op are an error — fail fast at
    import time so a rename or duplicate file doesn't silently shadow
    the load-bearing rule.
    """
    def deco(fn: VJPRule) -> VJPRule:
        if op in _VJP_RULES:
            raise RuntimeError(
                f"VJP rule for {op!r} is already registered "
                f"(existing: {_VJP_RULES[op].__name__}, new: {fn.__name__}). "
                f"Pick one."
            )
        _VJP_RULES[op] = fn
        return fn
    return deco


def get_rule(op: str) -> Optional[VJPRule]:
    """Return the rule for `op`, or None if no rule is registered."""
    return _VJP_RULES.get(op)


def list_registered() -> Tuple[str, ...]:
    """Return the opcodes that have a VJP rule, sorted for stability."""
    return tuple(sorted(_VJP_RULES))


# ---------------------------------------------------------------------------
# IR-construction helpers
#
# Each helper builds one UOp and tags it with the forward-op `vjp_of`
# annotation that PRD-009 needs to walk the backward graph and identify
# recompute candidates.
# ---------------------------------------------------------------------------


def _vjp_uop(
    op: str,
    inputs: Tuple[UOp, ...],
    shape: Tuple[int, ...],
    dtype: str,
    forward_node: UOp,
    arg: Any = None,
) -> UOp:
    """Construct a backward-graph UOp tagged with its forward source.

    `arg` may be None or a dict; the function merges in `vjp_of` so callers
    don't have to remember it. The forward node is identified by id() — a
    runtime side-table (`_FORWARD_REGISTRY`) would let us avoid embedding
    UOp objects in `arg`, but for v0 the embedded reference is the simplest
    correct option. PRD-009 reads it via `node.arg["vjp_of"]`.
    """
    if arg is None:
        new_arg: Dict[str, Any] = {"vjp_of": forward_node}
    elif isinstance(arg, dict):
        new_arg = {**arg, "vjp_of": forward_node}
    else:
        # Non-dict args (e.g. CMP's op-string) don't currently arise in
        # VJP-emitted UOps; if a future rule needs one, lift to a dict.
        new_arg = {"raw": arg, "vjp_of": forward_node}
    return UOp(op=op, inputs=inputs, shape=shape, dtype=dtype, arg=new_arg)


def _unbroadcast_uop(dy: UOp, target_shape: Tuple[int, ...],
                     forward_node: UOp) -> UOp:
    """Reduce `dy` back to `target_shape` by summing over broadcast-extended
    dims. The inverse of NumPy's broadcasting.

    Algorithm (matches `_tensor_proxy._unbroadcast` for ndarrays):

      1. Strip extra leading dims (sum-reduce them away).
      2. For each remaining axis where target_shape[i] == 1 and
         dy.shape[i] != 1, sum-reduce that axis with keepdims=True.

    If `dy.shape == target_shape`, returns dy unchanged.
    """
    if dy.shape == target_shape:
        return dy

    cur = dy
    extra_dims = len(cur.shape) - len(target_shape)
    if extra_dims > 0:
        # Sum over the leading extra dims. NumPy's `axis` argument accepts
        # int or tuple of int but NOT list — be explicit about the type.
        axes_t = tuple(range(extra_dims))
        reduced_shape = tuple(cur.shape[i] for i in range(extra_dims, len(cur.shape)))
        cur = _vjp_uop(
            OP_REDUCE,
            (cur,),
            reduced_shape,
            cur.dtype,
            forward_node,
            arg={"op": "sum", "axis": axes_t, "keepdims": False},
        )

    # Now cur.shape is at least as short as target_shape; walk the
    # remaining dims and squash any size-1 ↔ size-N mismatches.
    extra_axes: list[int] = []
    for i, target_dim in enumerate(target_shape):
        if target_dim == 1 and cur.shape[i] != 1:
            extra_axes.append(i)
    if extra_axes:
        extra_axes_t = tuple(extra_axes)
        new_shape = tuple(
            1 if i in extra_axes else cur.shape[i]
            for i in range(len(cur.shape))
        )
        cur = _vjp_uop(
            OP_REDUCE,
            (cur,),
            new_shape,
            cur.dtype,
            forward_node,
            arg={"op": "sum", "axis": extra_axes_t, "keepdims": True},
        )
    return cur


# ---------------------------------------------------------------------------
# Trivial rules (one-liners, no broadcasting)
# ---------------------------------------------------------------------------


@register_vjp(OP_NEG)
def _vjp_neg(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    (x,) = inputs
    return (_vjp_uop(OP_NEG, (dy,), x.shape, x.dtype, output),)


@register_vjp(OP_EXP)
def _vjp_exp(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """d/dx exp(x) = exp(x) — reuse the forward output, don't re-compute.

    The Flash Attention v2 reuse pattern: this VJP references the forward
    EXP node directly, so the fusion pass sees a single graph where the
    EXP value flows from forward into backward MUL in the same kernel.
    """
    (x,) = inputs
    return (
        _vjp_uop(OP_MUL, (dy, output), x.shape, x.dtype, output),
    )


@register_vjp(OP_LOG)
def _vjp_log(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """d/dx log(x) = 1/x."""
    (x,) = inputs
    return (_vjp_uop(OP_DIV, (dy, x), x.shape, x.dtype, output),)


@register_vjp(OP_ABS)
def _vjp_abs(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """d/dx abs(x) = sign(x), with the zero subgradient selected at x=0."""
    validate_real_numeric_unary_contract(output)
    (x,) = inputs
    sign = _vjp_uop(OP_SIGN, (x,), x.shape, x.dtype, output)
    return (_vjp_uop(OP_MUL, (dy, sign), x.shape, x.dtype, output),)


@register_vjp(OP_SIGN)
def _vjp_sign(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """The selected first derivative of sign is zero everywhere."""
    validate_real_numeric_unary_contract(output)
    (x,) = inputs
    zero = _vjp_uop(
        OP_CONST,
        (),
        (),
        x.dtype,
        output,
        arg={"value": 0},
    )
    return (_vjp_uop(OP_MUL, (dy, zero), x.shape, x.dtype, output),)


@register_vjp(OP_CLAMP)
def _vjp_clamp(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """Pass dy where x is inclusively inside the declared clamp bounds."""
    minimum, maximum = validate_clamp_contract(output)
    (x,) = inputs
    masks = []
    for comparison, bound in (("ge", minimum), ("le", maximum)):
        if bound is None:
            continue
        scalar = _vjp_uop(
            OP_CONST,
            (),
            (),
            x.dtype,
            output,
            arg={"value": bound},
        )
        masks.append(_vjp_uop(
            OP_CMP,
            (x, scalar),
            x.shape,
            "bool",
            output,
            arg={"op": comparison},
        ))
    mask = masks[0]
    if len(masks) == 2:
        mask = _vjp_uop(OP_MUL, (masks[0], masks[1]), x.shape, "bool", output)
    typed_mask = _vjp_uop(
        OP_CAST,
        (mask,),
        x.shape,
        x.dtype,
        output,
        arg={"dtype": x.dtype},
    )
    return (_vjp_uop(OP_MUL, (dy, typed_mask), x.shape, x.dtype, output),)


@register_vjp(OP_FLIP)
def _vjp_flip(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """Flip is its own inverse and therefore its own VJP."""
    axis = validate_flip_contract(output)
    (x,) = inputs
    return (_vjp_uop(
        OP_FLIP,
        (dy,),
        x.shape,
        x.dtype,
        output,
        arg={"axis": axis},
    ),)


@register_vjp(OP_CONCAT)
def _vjp_concat(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """Split a concatenated cotangent back into its typed input segments."""
    axis, sizes, legacy_empty = validate_cat_contract(output)
    gradients = []
    start = 0
    floating = frozenset({"float16", "float32", "float64"})
    for source, length, empty in zip(inputs, sizes, legacy_empty):
        if output.dtype not in floating or source.dtype not in floating:
            gradients.append(None)
            start += length
            continue
        narrow_shape = list(output.shape)
        narrow_shape[axis] = length
        gradient = _vjp_uop(
            OP_NARROW,
            (dy,),
            tuple(narrow_shape),
            output.dtype,
            output,
            arg={"axis": axis, "start": start, "length": length},
        )
        if empty and gradient.shape != source.shape:
            gradient = _vjp_uop(
                OP_RESHAPE,
                (gradient,),
                source.shape,
                output.dtype,
                output,
                arg={"new_shape": source.shape},
            )
        if gradient.dtype != source.dtype:
            gradient = _vjp_uop(
                OP_CAST,
                (gradient,),
                source.shape,
                source.dtype,
                output,
                arg={"dtype": source.dtype},
            )
        gradients.append(gradient)
        start += length
    return tuple(gradients)


@register_vjp(OP_STACK)
def _vjp_stack(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """Select and reshape each static slice of a stacked cotangent."""
    axis = validate_stack_contract(output)
    gradients = []
    floating = frozenset({"float16", "float32", "float64"})
    narrow_shape = list(output.shape)
    narrow_shape[axis] = 1
    for index, source in enumerate(inputs):
        if output.dtype not in floating or source.dtype not in floating:
            gradients.append(None)
            continue
        gradient = _vjp_uop(
            OP_NARROW,
            (dy,),
            tuple(narrow_shape),
            output.dtype,
            output,
            arg={"axis": axis, "start": index, "length": 1},
        )
        gradient = _vjp_uop(
            OP_RESHAPE,
            (gradient,),
            source.shape,
            output.dtype,
            output,
            arg={"new_shape": source.shape},
        )
        if gradient.dtype != source.dtype:
            gradient = _vjp_uop(
                OP_CAST,
                (gradient,),
                source.shape,
                source.dtype,
                output,
                arg={"dtype": source.dtype},
            )
        gradients.append(gradient)
    return tuple(gradients)


@register_vjp(OP_PAD)
def _vjp_pad(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """Select the unpadded static interior of the output cotangent."""
    pad_width, _ = validate_pad_contract(output)
    (source,) = inputs
    if source.dtype not in ("float16", "float32", "float64"):
        return (None,)
    slices = tuple(
        slice(lower, lower + extent)
        for extent, (lower, _) in zip(source.shape, pad_width)
    )
    return (_vjp_uop(
        OP_SLICE,
        (dy,),
        source.shape,
        source.dtype,
        output,
        arg={"slices": slices},
    ),)


@register_vjp(OP_SORT_INDICES)
def _vjp_sort_indices(
    output: UOp,
    inputs: Tuple[UOp, ...],
    dy: UOp,
) -> Tuple[Optional[UOp], ...]:
    """Sorting indices are discrete and do not carry a source cotangent."""
    validate_sort_indices_contract(output)
    return (None,)


@register_vjp(OP_SORT_VALUES)
def _vjp_sort_values(
    output: UOp,
    inputs: Tuple[UOp, ...],
    dy: UOp,
) -> Tuple[Optional[UOp], ...]:
    """Scatter each ordered-value cotangent back through its permutation."""
    axis, _, _ = validate_sort_values_contract(output)
    source, indices = inputs
    if source.dtype not in ("float16", "float32", "float64"):
        return None, None
    if not source.shape:
        return dy, None
    zero = _vjp_uop(
        OP_CONST,
        (),
        (),
        source.dtype,
        output,
        arg={"value": 0},
    )
    target = _vjp_uop(
        OP_BROADCAST_TO,
        (zero,),
        source.shape,
        source.dtype,
        output,
        arg={"shape": source.shape},
    )
    gradient = _vjp_uop(
        OP_SCATTER_ADD,
        (target, indices, dy),
        source.shape,
        source.dtype,
        output,
        arg={"dim": axis},
    )
    return gradient, None


@register_vjp(OP_TOPK_INDICES)
def _vjp_topk_indices(
    output: UOp,
    inputs: Tuple[UOp, ...],
    dy: UOp,
) -> Tuple[Optional[UOp], ...]:
    """Top-k indices are discrete and do not carry a source cotangent."""
    validate_topk_indices_contract(output)
    return (None,)


@register_vjp(OP_TOPK_VALUES)
def _vjp_topk_values(
    output: UOp,
    inputs: Tuple[UOp, ...],
    dy: UOp,
) -> Tuple[Optional[UOp], ...]:
    """Scatter selected-value cotangents back through the paired indices."""
    axis, _, _, _ = validate_topk_values_contract(output)
    source, indices = inputs
    if source.dtype not in ("float16", "float32", "float64"):
        return None, None
    zero = _vjp_uop(
        OP_CONST,
        (),
        (),
        source.dtype,
        output,
        arg={"value": 0},
    )
    target = _vjp_uop(
        OP_BROADCAST_TO,
        (zero,),
        source.shape,
        source.dtype,
        output,
        arg={"shape": source.shape},
    )
    gradient = _vjp_uop(
        OP_SCATTER_ADD,
        (target, indices, dy),
        source.shape,
        source.dtype,
        output,
        arg={"dim": axis},
    )
    return gradient, None


@register_vjp(OP_SCATTER)
def _vjp_scatter(
    output: UOp,
    inputs: Tuple[UOp, ...],
    dy: UOp,
) -> Tuple[Optional[UOp], ...]:
    """Differentiate deterministic unique overwrite into target and source."""
    axis = validate_scatter_contract(output)
    target, index, source = inputs
    if target.dtype not in ("float16", "float32", "float64"):
        return None, None, None
    zero = _vjp_uop(
        OP_CONST,
        (),
        (),
        target.dtype,
        output,
        arg={"value": 0},
    )
    target_gradient = _vjp_uop(
        OP_SCATTER,
        (dy, index, zero),
        target.shape,
        target.dtype,
        output,
        arg={"dim": axis},
    )
    if source.shape == ():
        return target_gradient, None, None
    source_gradient = _vjp_uop(
        OP_INDEX,
        (dy, index),
        source.shape,
        source.dtype,
        output,
        arg={"dim": axis},
    )
    return target_gradient, None, source_gradient


@register_vjp(OP_CUMSUM)
def _vjp_cumsum(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """An inclusive scan's VJP is the opposite-direction inclusive scan."""
    axis, reverse = validate_cumsum_contract(output)
    (source,) = inputs
    gradient = _vjp_uop(
        OP_CUMSUM,
        (dy,),
        source.shape,
        output.dtype,
        output,
        arg={"axis": axis, "reverse": not reverse},
    )
    if gradient.dtype != source.dtype:
        gradient = _vjp_uop(
            OP_CAST,
            (gradient,),
            source.shape,
            source.dtype,
            output,
            arg={"dtype": source.dtype},
        )
    return (gradient,)


@register_vjp(OP_TRIL)
def _vjp_tril(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """Lower-triangular selection is idempotent and therefore its own VJP."""
    diagonal = validate_tril_contract(output)
    (source,) = inputs
    return (_vjp_uop(
        OP_TRIL,
        (dy,),
        source.shape,
        source.dtype,
        output,
        arg={"diagonal": diagonal},
    ),)


@register_vjp(OP_TRIU)
def _vjp_triu(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """Upper-triangular selection is idempotent and therefore its own VJP."""
    diagonal = validate_triu_contract(output)
    (source,) = inputs
    return (_vjp_uop(
        OP_TRIU,
        (dy,),
        source.shape,
        source.dtype,
        output,
        arg={"diagonal": diagonal},
    ),)


@register_vjp(OP_INDEX)
def _vjp_gather(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """Gather's source VJP is deterministic scatter-add; indices are discrete."""
    axis = validate_gather_contract(output)
    source, index = inputs
    zero = _vjp_uop(
        OP_CONST,
        (),
        (),
        source.dtype,
        output,
        arg={"value": 0},
    )
    target = _vjp_uop(
        OP_BROADCAST_TO,
        (zero,),
        source.shape,
        source.dtype,
        output,
        arg={"shape": source.shape},
    )
    gradient = _vjp_uop(
        OP_SCATTER_ADD,
        (target, index, dy),
        source.shape,
        source.dtype,
        output,
        arg={"dim": axis},
    )
    return gradient, None


@register_vjp(OP_REPEAT)
def _vjp_repeat(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """Sum each tile block back into the original input shape."""
    repeats, padded_shape = validate_repeat_contract(output)
    (x,) = inputs
    interleaved_shape = tuple(
        extent
        for factor, source_extent in zip(repeats, padded_shape)
        for extent in (factor, source_extent)
    )
    cur = _vjp_uop(
        OP_RESHAPE,
        (dy,),
        interleaved_shape,
        x.dtype,
        output,
        arg={"new_shape": interleaved_shape},
    )
    repeat_axes = tuple(range(0, len(interleaved_shape), 2))
    cur = _vjp_uop(
        OP_REDUCE,
        (cur,),
        padded_shape,
        x.dtype,
        output,
        arg={"op": "sum", "axis": repeat_axes, "keepdims": False},
    )
    if padded_shape != x.shape:
        cur = _vjp_uop(
            OP_RESHAPE,
            (cur,),
            x.shape,
            x.dtype,
            output,
            arg={"new_shape": x.shape},
        )
    return (cur,)


@register_vjp(OP_PROD)
def _vjp_prod(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """Zero-aware product derivative for any static reduction-axis set."""
    axes, keepdims, expanded_shape = validate_prod_contract(output)
    (x,) = inputs

    zero = _vjp_uop(
        OP_CONST, (), (), x.dtype, output, arg={"value": 0}
    )
    one = _vjp_uop(
        OP_CONST, (), (), x.dtype, output, arg={"value": 1}
    )
    zero_mask = _vjp_uop(
        OP_CMP,
        (x, zero),
        x.shape,
        "bool",
        output,
        arg={"op": "eq"},
    )
    zero_count_source = _vjp_uop(
        OP_CAST,
        (zero_mask,),
        x.shape,
        "int32",
        output,
        arg={"dtype": "int32"},
    )
    zero_count = _vjp_uop(
        OP_REDUCE,
        (zero_count_source,),
        expanded_shape,
        "int32",
        output,
        arg={"op": "sum", "axis": axes, "keepdims": True},
    )
    count_zero = _vjp_uop(
        OP_CONST, (), (), "int32", output, arg={"value": 0}
    )
    count_one = _vjp_uop(
        OP_CONST, (), (), "int32", output, arg={"value": 1}
    )
    no_zeros = _vjp_uop(
        OP_CMP,
        (zero_count, count_zero),
        expanded_shape,
        "bool",
        output,
        arg={"op": "eq"},
    )
    one_zero = _vjp_uop(
        OP_CMP,
        (zero_count, count_one),
        expanded_shape,
        "bool",
        output,
        arg={"op": "eq"},
    )
    safe_x = _vjp_uop(
        OP_WHERE,
        (zero_mask, one, x),
        x.shape,
        x.dtype,
        output,
    )
    nonzero_product = _vjp_uop(
        OP_PROD,
        (safe_x,),
        expanded_shape,
        x.dtype,
        output,
        arg={"axes": axes, "keepdims": True},
    )
    quotient = _vjp_uop(
        OP_DIV,
        (nonzero_product, safe_x),
        x.shape,
        x.dtype,
        output,
    )
    single_zero = _vjp_uop(
        OP_WHERE,
        (zero_mask, nonzero_product, zero),
        x.shape,
        x.dtype,
        output,
    )
    one_or_many_zero = _vjp_uop(
        OP_WHERE,
        (one_zero, single_zero, zero),
        x.shape,
        x.dtype,
        output,
    )
    local_derivative = _vjp_uop(
        OP_WHERE,
        (no_zeros, quotient, one_or_many_zero),
        x.shape,
        x.dtype,
        output,
    )
    upstream = dy
    if not keepdims and dy.shape != expanded_shape:
        upstream = _vjp_uop(
            OP_RESHAPE,
            (dy,),
            expanded_shape,
            dy.dtype,
            output,
            arg={"new_shape": expanded_shape},
        )
    return (_vjp_uop(
        OP_MUL,
        (upstream, local_derivative),
        x.shape,
        x.dtype,
        output,
    ),)


@register_vjp(OP_VAR)
def _vjp_var(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """Variance derivative with one canonical static correction."""
    axes, correction, keepdims, expanded_shape, reduced_elements = validate_var_contract(output)
    (x,) = inputs
    mean = _vjp_uop(
        OP_REDUCE,
        (x,),
        expanded_shape,
        x.dtype,
        output,
        arg={"op": "mean", "axis": axes, "keepdims": True},
    )
    negative_mean = _vjp_uop(
        OP_NEG,
        (mean,),
        expanded_shape,
        x.dtype,
        output,
    )
    centered = _vjp_uop(
        OP_ADD,
        (x, negative_mean),
        x.shape,
        x.dtype,
        output,
    )
    two = _vjp_uop(
        OP_CONST, (), (), x.dtype, output, arg={"value": 2}
    )
    denominator = _vjp_uop(
        OP_CONST,
        (),
        (),
        x.dtype,
        output,
        arg={"value": max(0, reduced_elements - correction)},
    )
    local = _vjp_uop(
        OP_MUL,
        (centered, two),
        x.shape,
        x.dtype,
        output,
    )
    local = _vjp_uop(
        OP_DIV,
        (local, denominator),
        x.shape,
        x.dtype,
        output,
    )
    upstream = dy
    if not keepdims and upstream.shape != expanded_shape:
        upstream = _vjp_uop(
            OP_RESHAPE,
            (upstream,),
            expanded_shape,
            upstream.dtype,
            output,
            arg={"new_shape": expanded_shape},
        )
    return (_vjp_uop(
        OP_MUL,
        (local, upstream),
        x.shape,
        x.dtype,
        output,
    ),)


@register_vjp(OP_REPEAT_INTERLEAVE)
def _vjp_repeat_interleave(
    output: UOp,
    inputs: Tuple[UOp, ...],
    dy: UOp,
) -> Tuple[Optional[UOp], ...]:
    """Collapse each selected-axis repeat block back into its source element."""
    repeats, axis = validate_repeat_interleave_contract(output)
    (x,) = inputs
    block_shape = x.shape[:axis] + (x.shape[axis], repeats) + x.shape[axis + 1:]
    grouped = _vjp_uop(
        OP_RESHAPE,
        (dy,),
        block_shape,
        x.dtype,
        output,
        arg={"new_shape": block_shape},
    )
    return (_vjp_uop(
        OP_REDUCE,
        (grouped,),
        x.shape,
        x.dtype,
        output,
        arg={"op": "sum", "axis": (axis + 1,), "keepdims": False},
    ),)


@register_vjp(OP_SIN)
def _vjp_sin(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """d/dx sin(x) = cos(x)."""
    validate_typed_unary_contract(output)
    (x,) = inputs
    cos = _vjp_uop(OP_COS, (x,), x.shape, x.dtype, output)
    return (_vjp_uop(OP_MUL, (dy, cos), x.shape, x.dtype, output),)


@register_vjp(OP_COS)
def _vjp_cos(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """d/dx cos(x) = -sin(x)."""
    validate_typed_unary_contract(output)
    (x,) = inputs
    sin = _vjp_uop(OP_SIN, (x,), x.shape, x.dtype, output)
    product = _vjp_uop(OP_MUL, (dy, sin), x.shape, x.dtype, output)
    return (_vjp_uop(OP_NEG, (product,), x.shape, x.dtype, output),)


# ---------------------------------------------------------------------------
# Elementwise binary ops with broadcasting un-projection
# ---------------------------------------------------------------------------


@register_vjp(OP_ADD)
def _vjp_add(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """d/da (a+b) = dy; d/db (a+b) = dy — with un-broadcast to match input shapes."""
    a, b = inputs
    return (
        _unbroadcast_uop(dy, a.shape, output),
        _unbroadcast_uop(dy, b.shape, output),
    )


@register_vjp(OP_MUL)
def _vjp_mul(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """d/da (a*b) = dy*b; d/db (a*b) = dy*a."""
    a, b = inputs
    da_full = _vjp_uop(OP_MUL, (dy, b), dy.shape, dy.dtype, output)
    db_full = _vjp_uop(OP_MUL, (dy, a), dy.shape, dy.dtype, output)
    return (
        _unbroadcast_uop(da_full, a.shape, output),
        _unbroadcast_uop(db_full, b.shape, output),
    )


@register_vjp(OP_DIV)
def _vjp_div(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """d/da (a/b) = dy/b; d/db (a/b) = -dy*a/(b*b)."""
    a, b = inputs
    da_full = _vjp_uop(OP_DIV, (dy, b), dy.shape, dy.dtype, output)
    b_squared = _vjp_uop(OP_MUL, (b, b), b.shape, b.dtype, output)
    dy_times_a = _vjp_uop(OP_MUL, (dy, a), dy.shape, dy.dtype, output)
    div_part = _vjp_uop(OP_DIV, (dy_times_a, b_squared), dy.shape, dy.dtype, output)
    db_full = _vjp_uop(OP_NEG, (div_part,), dy.shape, dy.dtype, output)
    return (
        _unbroadcast_uop(da_full, a.shape, output),
        _unbroadcast_uop(db_full, b.shape, output),
    )


@register_vjp(OP_WHERE)
def _vjp_where(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """Route cotangents through the selected branch; the bool mask is discrete."""
    validate_where_contract(output)
    condition, lhs, rhs = inputs
    zero = _vjp_uop(OP_CONST, (), (), dy.dtype, output, arg={"value": 0})
    lhs_full = _vjp_uop(
        OP_WHERE,
        (condition, dy, zero),
        output.shape,
        dy.dtype,
        output,
    )
    rhs_full = _vjp_uop(
        OP_WHERE,
        (condition, zero, dy),
        output.shape,
        dy.dtype,
        output,
    )
    lhs_grad = _unbroadcast_uop(lhs_full, lhs.shape, output)
    rhs_grad = _unbroadcast_uop(rhs_full, rhs.shape, output)
    if lhs_grad.dtype != lhs.dtype:
        lhs_grad = _vjp_uop(
            OP_CAST,
            (lhs_grad,),
            lhs.shape,
            lhs.dtype,
            output,
            arg={"dtype": lhs.dtype},
        )
    if rhs_grad.dtype != rhs.dtype:
        rhs_grad = _vjp_uop(
            OP_CAST,
            (rhs_grad,),
            rhs.shape,
            rhs.dtype,
            output,
            arg={"dtype": rhs.dtype},
        )
    return None, lhs_grad, rhs_grad


# ---------------------------------------------------------------------------
# Shape ops — pure routing, no broadcasting concerns
# ---------------------------------------------------------------------------


@register_vjp(OP_RESHAPE)
def _vjp_reshape(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    (x,) = inputs
    return (
        _vjp_uop(
            OP_RESHAPE,
            (dy,),
            x.shape,
            x.dtype,
            output,
            arg={"new_shape": x.shape},
        ),
    )


@register_vjp(OP_PERMUTE)
def _vjp_permute(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """The VJP of a permutation is its inverse permutation."""
    (x,) = inputs
    forward_axes = output.arg["axes"]
    inverse = [0] * len(forward_axes)
    for i, a in enumerate(forward_axes):
        inverse[a] = i
    return (
        _vjp_uop(
            OP_PERMUTE,
            (dy,),
            x.shape,
            x.dtype,
            output,
            arg={"axes": tuple(inverse)},
        ),
    )


@register_vjp(OP_BROADCAST_TO)
def _vjp_broadcast_to(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """The VJP of an explicit broadcast sums every expanded dimension."""
    validate_broadcast_to_contract(output)
    (x,) = inputs
    return (_unbroadcast_uop(dy, x.shape, output),)


@register_vjp(OP_CAST)
def _vjp_cast(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """The gradient flows back at the source dtype.

    Critical for PRD-010 mixed precision: forward `x_fp32 → CAST(fp16)`
    has VJP `dy_fp16 → CAST(fp32) → dx_fp32`. Round-trip preserves the
    upstream dtype.
    """
    (x,) = inputs
    if x.dtype == dy.dtype:
        return (dy,)
    return (
        _vjp_uop(
            OP_CAST,
            (dy,),
            x.shape,
            x.dtype,
            output,
            arg={"dtype": x.dtype},
        ),
    )


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def _expand_reduced_shape(
    input_shape: Tuple[int, ...],
    axis: Any,
    keepdims: bool,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Compute (expanded_dy_shape, reduced_axes_sorted).

    The expanded shape has size-1 dims at every reduced axis. If the
    forward used keepdims=True, dy already has those size-1 dims and the
    expanded shape equals dy.shape; if keepdims=False, we need to insert
    size-1 dims via a RESHAPE before broadcasting.
    """
    if axis is None:
        reduced_axes: Tuple[int, ...] = tuple(range(len(input_shape)))
    elif isinstance(axis, int):
        reduced_axes = (axis % len(input_shape),) if input_shape else ()
    else:
        reduced_axes = tuple(a % len(input_shape) for a in axis)
    reduced_axes = tuple(sorted(reduced_axes))
    expanded: list[int] = list(input_shape)
    for a in reduced_axes:
        expanded[a] = 1
    return tuple(expanded), reduced_axes


@register_vjp(OP_REDUCE)
def _vjp_reduce(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """VJP for sum/mean. max/min/argmax/argmin are deliberately deferred:

      * sum: broadcast dy back to input shape.
      * mean: same, scaled by 1/N where N is the product of reduced dims.
      * max/min: requires a CMP+REDUCE construct for tie-breaking (PRD-007
        W3); fall back to closure for now.
      * argmax/argmin: non-differentiable; closure already returns None.
    """
    (x,) = inputs
    op = output.arg.get("op")
    axis = output.arg.get("axis")
    keepdims = output.arg.get("keepdims", False)

    if op not in ("sum", "mean"):
        # Defer max/min/argmax/argmin to PRD-007 W3. Returning None makes
        # the dispatcher fall back to the closure path for this UOp.
        return (None,)

    expanded_shape, reduced_axes = _expand_reduced_shape(x.shape, axis, keepdims)

    cur = dy
    if not keepdims and cur.shape != expanded_shape:
        cur = _vjp_uop(
            OP_RESHAPE,
            (cur,),
            expanded_shape,
            cur.dtype,
            output,
            arg={"new_shape": expanded_shape},
        )

    if op == "mean":
        n_reduced = 1
        for a in reduced_axes:
            n_reduced *= x.shape[a]
        scale = _vjp_uop(
            OP_CONST,
            (),
            (),
            cur.dtype,
            output,
            arg={"value": 1.0 / n_reduced},
        )
        cur = _vjp_uop(OP_MUL, (cur, scale), expanded_shape, cur.dtype, output)

    # Broadcast back to input shape. The IR's shape field is metadata —
    # it doesn't coerce a NumPy op's output. We need an explicit
    # broadcast: OP_BROADCAST_TO calls np.broadcast_to under the hood.
    if cur.shape != x.shape:
        cur = _vjp_uop(
            OP_BROADCAST_TO,
            (cur,),
            x.shape,
            cur.dtype,
            output,
            arg={"shape": x.shape},
        )
    return (cur,)


# ---------------------------------------------------------------------------
# MATMUL — the load-bearing rule
# ---------------------------------------------------------------------------


def _swap_last_two(axes_count: int) -> Tuple[int, ...]:
    """Permutation that swaps the last two axes; identity on leading dims."""
    if axes_count < 2:
        raise ValueError("MATMUL VJP requires at least 2-D inputs")
    axes = list(range(axes_count))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return tuple(axes)


def _swap_last_two_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(shape) < 2:
        return shape
    out = list(shape)
    out[-1], out[-2] = out[-2], out[-1]
    return tuple(out)


@register_vjp(OP_MATMUL)
def _vjp_matmul(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """The standard rules: dA = dy @ B.T; dB = A.T @ dy.

    For batched (..., M, K) @ (..., K, N), broadcast the leading batch dims;
    un-broadcast the resulting gradient back to original input shape via
    sum-reduce on broadcast axes.

    1-D edge cases (vector dot, vector-matrix, matrix-vector) follow the
    same shape conventions as the forward — defer to PRD-007 W3 for full
    coverage; fall back to closure for now if either input is 1-D.
    """
    a, b = inputs
    if len(a.shape) < 2 or len(b.shape) < 2:
        # 1-D edge cases; closure backward handles them today, keep that.
        return (None, None)

    # da = dy @ B.T
    bT = _vjp_uop(
        OP_PERMUTE,
        (b,),
        _swap_last_two_shape(b.shape),
        b.dtype,
        output,
        arg={"axes": _swap_last_two(len(b.shape))},
    )
    # Shape after MATMUL: leading-broadcast(dy.shape[:-2], bT.shape[:-2]) + (dy.shape[-2], bT.shape[-1])
    da_full_shape = _broadcast_batch_shape(dy.shape, bT.shape) + (dy.shape[-2], bT.shape[-1])
    da_full = _vjp_uop(OP_MATMUL, (dy, bT), da_full_shape, dy.dtype, output)

    # db = A.T @ dy
    aT = _vjp_uop(
        OP_PERMUTE,
        (a,),
        _swap_last_two_shape(a.shape),
        a.dtype,
        output,
        arg={"axes": _swap_last_two(len(a.shape))},
    )
    db_full_shape = _broadcast_batch_shape(aT.shape, dy.shape) + (aT.shape[-2], dy.shape[-1])
    db_full = _vjp_uop(OP_MATMUL, (aT, dy), db_full_shape, dy.dtype, output)

    # Un-broadcast batch dims back to original input shapes.
    return (
        _unbroadcast_uop(da_full, a.shape, output),
        _unbroadcast_uop(db_full, b.shape, output),
    )


def _broadcast_batch_shape(
    a_shape: Tuple[int, ...],
    b_shape: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Broadcast the leading "batch" dims of two shapes (all but last 2).
    Returns the broadcasted batch prefix.
    """
    a_batch = a_shape[:-2]
    b_batch = b_shape[:-2]
    if not a_batch and not b_batch:
        return ()
    # Right-align and broadcast: shorter side gets implicit leading 1s.
    n = max(len(a_batch), len(b_batch))
    pad_a = (1,) * (n - len(a_batch)) + a_batch
    pad_b = (1,) * (n - len(b_batch)) + b_batch
    out = []
    for da, db in zip(pad_a, pad_b):
        if da == 1:
            out.append(db)
        elif db == 1:
            out.append(da)
        elif da == db:
            out.append(da)
        else:
            raise ValueError(
                f"batched matmul: cannot broadcast batch dims {a_batch} vs {b_batch}"
            )
    return tuple(out)


# ---------------------------------------------------------------------------
# CONV1D / CONV2D / CONV_TRANSPOSE2D / CONV3D — primitive CNN rules
# ---------------------------------------------------------------------------


@register_vjp(OP_CONV1D)
def _vjp_conv1d(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    x = inputs[0]
    weight = inputs[1]
    arg = dict(output.arg)
    arg.pop("vjp_of", None)
    grad_x = _vjp_uop(
        OP_CONV1D_BACKWARD_INPUT,
        (dy, weight),
        x.shape,
        x.dtype,
        output,
        arg=arg,
    )
    grad_w = _vjp_uop(
        OP_CONV1D_BACKWARD_WEIGHT,
        (dy, x),
        weight.shape,
        weight.dtype,
        output,
        arg=arg,
    )
    if len(inputs) == 2:
        return (grad_x, grad_w)
    bias = inputs[2]
    grad_b = _vjp_uop(
        OP_CONV1D_BACKWARD_BIAS,
        (dy,),
        bias.shape,
        bias.dtype,
        output,
        arg=arg,
    )
    return (grad_x, grad_w, grad_b)


@register_vjp(OP_CONV2D)
def _vjp_conv2d(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """Conv2d VJP as first-class IR.

    Forward inputs are (input, weight[, bias]). Backward emits explicit
    conv2d-backward primitives for input/weight and uses REDUCE(sum) for
    bias. These have CPU reference handlers plus explicit WebGPU bridge
    handlers for `realize_webgpu(...)` on gradient roots; default
    `.backward()` still mutates CPU `.grad` buffers.
    """
    x = inputs[0]
    weight = inputs[1]
    arg = dict(output.arg)
    arg.pop("vjp_of", None)
    grad_x = _vjp_uop(
        OP_CONV2D_BACKWARD_INPUT,
        (dy, weight),
        x.shape,
        x.dtype,
        output,
        arg=arg,
    )
    grad_w = _vjp_uop(
        OP_CONV2D_BACKWARD_WEIGHT,
        (dy, x),
        weight.shape,
        weight.dtype,
        output,
        arg=arg,
    )
    if len(inputs) == 2:
        return (grad_x, grad_w)
    bias = inputs[2]
    grad_b = _vjp_uop(
        OP_CONV2D_BACKWARD_BIAS,
        (dy,),
        bias.shape,
        bias.dtype,
        output,
        arg=arg,
    )
    return (grad_x, grad_w, grad_b)


@register_vjp(OP_CONV_TRANSPOSE2D)
def _vjp_conv_transpose2d(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    x = inputs[0]
    weight = inputs[1]
    arg = dict(output.arg)
    arg.pop("vjp_of", None)
    grad_x = _vjp_uop(
        OP_CONV_TRANSPOSE2D_BACKWARD_INPUT,
        (dy, weight),
        x.shape,
        x.dtype,
        output,
        arg=arg,
    )
    grad_w = _vjp_uop(
        OP_CONV_TRANSPOSE2D_BACKWARD_WEIGHT,
        (dy, x),
        weight.shape,
        weight.dtype,
        output,
        arg=arg,
    )
    if len(inputs) == 2:
        return (grad_x, grad_w)
    bias = inputs[2]
    grad_b = _vjp_uop(
        OP_CONV_TRANSPOSE2D_BACKWARD_BIAS,
        (dy,),
        bias.shape,
        bias.dtype,
        output,
        arg=arg,
    )
    return (grad_x, grad_w, grad_b)


@register_vjp(OP_CONV3D)
def _vjp_conv3d(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """Conv3d VJP as first-class IR.

    Mirrors Conv1d/Conv2d: forward inputs are (input, weight[, bias]);
    backward emits explicit conv3d-backward primitives. CPU handlers are
    authoritative today; WebGPU lowering can land behind the same opcodes.
    """
    x = inputs[0]
    weight = inputs[1]
    arg = dict(output.arg)
    arg.pop("vjp_of", None)
    grad_x = _vjp_uop(
        OP_CONV3D_BACKWARD_INPUT,
        (dy, weight),
        x.shape,
        x.dtype,
        output,
        arg=arg,
    )
    grad_w = _vjp_uop(
        OP_CONV3D_BACKWARD_WEIGHT,
        (dy, x),
        weight.shape,
        weight.dtype,
        output,
        arg=arg,
    )
    if len(inputs) == 2:
        return (grad_x, grad_w)
    bias = inputs[2]
    grad_b = _vjp_uop(
        OP_CONV3D_BACKWARD_BIAS,
        (dy,),
        bias.shape,
        bias.dtype,
        output,
        arg=arg,
    )
    return (grad_x, grad_w, grad_b)


# ---------------------------------------------------------------------------
# LAYER_NORM — primitive norm rule
# ---------------------------------------------------------------------------


@register_vjp(OP_LAYER_NORM)
def _vjp_layer_norm(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    x, weight, bias = inputs
    arg = dict(output.arg)
    arg.pop("vjp_of", None)
    grad_x = _vjp_uop(
        OP_LAYER_NORM_BACKWARD_INPUT,
        (dy, x, weight),
        x.shape,
        x.dtype,
        output,
        arg=arg,
    )
    grad_w = _vjp_uop(
        OP_LAYER_NORM_BACKWARD_WEIGHT,
        (dy, x),
        weight.shape,
        weight.dtype,
        output,
        arg=arg,
    )
    grad_b = _vjp_uop(
        OP_LAYER_NORM_BACKWARD_BIAS,
        (dy,),
        bias.shape,
        bias.dtype,
        output,
        arg=arg,
    )
    return (grad_x, grad_w, grad_b)


# ---------------------------------------------------------------------------
# Non-differentiable ops — the VJP returns `None` per input, telling the
# symbolic dispatcher to stop propagating gradient through this branch.
# ---------------------------------------------------------------------------


@register_vjp(OP_ISNAN)
def _vjp_isnan(output: UOp, inputs: Tuple[UOp, ...], dy: UOp) -> Tuple[Optional[UOp], ...]:
    """ISNAN's output is bool — no meaningful real-valued gradient flows
    back into its input. Used by GradScaler's overflow check, which is
    inherently outside the autograd graph (Python-side `_any_nonfinite`).
    """
    return (None,)


__all__ = [
    "VJPRule",
    "register_vjp",
    "get_rule",
    "list_registered",
]
