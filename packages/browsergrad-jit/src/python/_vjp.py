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

  * **Higher-order**: every UOp a VJP rule emits must itself have a
    registered VJP rule. `create_graph=True` mode wraps emitted UOps in
    TensorProxies with `requires_grad=True` so a second `.backward()`
    finds them.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple

from ._ir import (
    UOp,
    OP_ADD, OP_MUL, OP_DIV, OP_NEG, OP_EXP, OP_LOG, OP_CAST,
    OP_MATMUL, OP_REDUCE, OP_RESHAPE, OP_PERMUTE,
    OP_CONST,
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
        # Sum over the leading extra dims. NumPy/_ir REDUCE expects a
        # tuple of axes; we collapse them in one call rather than emit
        # one REDUCE per leading dim.
        axes = list(range(extra_dims))
        reduced_shape = tuple(cur.shape[i] for i in range(extra_dims, len(cur.shape)))
        cur = _vjp_uop(
            OP_REDUCE,
            (cur,),
            reduced_shape,
            cur.dtype,
            forward_node,
            arg={"op": "sum", "axis": axes, "keepdims": False},
        )

    # Now cur.shape is at least as short as target_shape; walk the
    # remaining dims and squash any size-1 ↔ size-N mismatches.
    extra_axes_to_reduce: list[int] = []
    for i, target_dim in enumerate(target_shape):
        if target_dim == 1 and cur.shape[i] != 1:
            extra_axes_to_reduce.append(i)
    if extra_axes_to_reduce:
        new_shape = tuple(
            1 if i in extra_axes_to_reduce else cur.shape[i]
            for i in range(len(cur.shape))
        )
        cur = _vjp_uop(
            OP_REDUCE,
            (cur,),
            new_shape,
            cur.dtype,
            forward_node,
            arg={"op": "sum", "axis": extra_axes_to_reduce, "keepdims": True},
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

    # Broadcast back to input shape by re-running an unbroadcast in reverse:
    # we use a MUL by ones_like(x) so the IR remains a normal graph (no
    # implicit broadcasts). Simpler in practice: tag the gradient with the
    # target shape via RESHAPE then let downstream consumers broadcast in
    # NumPy. The cleanest way is to use ADD with a zero of the input shape —
    # but ADD broadcasts NumPy-style, which is exactly what we want.
    if cur.shape != x.shape:
        zero = _vjp_uop(
            OP_CONST,
            (),
            (),
            cur.dtype,
            output,
            arg={"value": 0.0},
        )
        # We need a tensor of x.shape full of zeros to broadcast `cur` to.
        # Wrap the scalar zero in a RESHAPE-style identity: zero_like(x) is
        # `ADD(zero, zero, broadcast)` — but that's circular. The cleanest
        # is to emit an explicit broadcast: `cur + 0.0_broadcast(x.shape)`.
        # The trick: ADD broadcasts naturally if one side has the target
        # shape. So emit a zero-CONST of `x.shape` (scalar zero is enough —
        # NumPy will broadcast it).
        cur = _vjp_uop(OP_ADD, (cur, zero), x.shape, cur.dtype, output)
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


__all__ = [
    "VJPRule",
    "register_vjp",
    "get_rule",
    "list_registered",
]
