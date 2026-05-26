"""browsergrad_jit._fusion — graph-rewrite pass for elementwise + softmax fusion.

INTERNAL. The fusion pass is invoked by the realizer's entry point when
`_fusion_config.is_enabled()` returns True. It produces a new UOp graph
where:

  * Linear chains of pointwise ops (ADD, MUL, DIV, NEG, EXP, LOG) collapse
    into a single `OP_FUSED_ELEMENTWISE` node. The realizer's handler
    interprets the chain in one Python loop with one np.ndarray per step
    — eliminating intermediate ndarray allocations between ops.

  * The 7-node softmax DAG — REDUCE(max,keepdims) → NEG → ADD → EXP →
    REDUCE(sum,keepdims) → DIV with EXP shared between the SUM and the
    DIV — collapses into a single `OP_FUSED_SOFTMAX` node. The matcher
    is a fixed template; if your IR drifts from the canonical form the
    matcher refuses and the unfused path runs.

Design constraints (per PRD-006 critique):

  * **Autograd safety**. Every UOp referenced by a downstream
    `_BackwardCtx.input_proxies[i]._uop` is a "holdout" — fusing it
    would orphan the backward closure. The fusion pass receives a
    `holdout: set[int]` of UOp ids and refuses to absorb any UOp in
    that set as a *non-terminal* node of a chain or DAG. The terminal
    node is OK because backward closures only consume the *inputs* to
    each op, not its scratch.

  * **id()-keyed everywhere**. Two structurally-equal UOps (e.g. two
    independent `CONST(1.0)` leaves) must be distinguished. The realizer
    uses `id()`; we mirror that.

  * **No backend assumptions**. The fused opcodes are realized by
    `_realize.py`'s NumPy handlers in v0. PRD-012's WGSL megakernel
    swaps in a different handler without touching pattern matching.

The pass also collects per-trace introspection — what fused, what
didn't fuse and why — exposed via `bg.jit.debug_fused_kernels()` and
`bg.jit.debug_unfused_reasons()`.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Set, Tuple

from ._ir import (
    UOp,
    OP_ADD, OP_MUL, OP_DIV, OP_NEG, OP_EXP, OP_LOG,
    OP_REDUCE, OP_FUSED_ELEMENTWISE, OP_FUSED_SOFTMAX,
    toposort,
)


# ---------------------------------------------------------------------------
# Opcodes the elementwise chain matcher will absorb. Deliberately tight:
# CAST, WHERE, CMP are NOT here — CAST changes dtype mid-chain (defer to
# PRD-010), WHERE/CMP introduce control flow that the linear-chain handler
# can't express in v0. ReLU lives at `WHERE(CMP, x, 0)` which is also
# excluded — pattern matchers for activations land as a later patch
# (PRD-006 v0.2) once we have data on which patterns dominate.
# ---------------------------------------------------------------------------
ELEMENTWISE_OPS = frozenset({OP_ADD, OP_MUL, OP_DIV, OP_NEG, OP_EXP, OP_LOG})


# ---------------------------------------------------------------------------
# Introspection records — fed back to the user via debug_* helpers.
# ---------------------------------------------------------------------------


@dataclass
class FusedKernelInfo:
    """One record per fused subgraph in the most recent fuse() call."""
    pattern: str            # "elementwise_chain" | "softmax"
    n_ops: int              # number of original UOps the fused node absorbed
    shape: Tuple[int, ...]  # output shape
    dtype: str
    ops: Tuple[str, ...]    # opcode sequence inside the fused node


@dataclass
class UnfusedReason:
    """One record per matcher attempt that did not fire."""
    pattern: str            # which matcher
    op: str                 # opcode at the candidate root
    shape: Tuple[int, ...]  # shape at the candidate root
    reason: str             # short, human-readable predicate that failed


@dataclass
class FusionReport:
    """All-of-pass introspection. Pulled by debug_fused_kernels() and
    debug_unfused_reasons() — they return the most recently emitted lists.
    """
    fused: List[FusedKernelInfo] = field(default_factory=list)
    unfused: List[UnfusedReason] = field(default_factory=list)


_LAST_REPORT: FusionReport = FusionReport()


def get_last_report() -> FusionReport:
    """Return the FusionReport from the most recent `fuse()` invocation.

    Resets to an empty report whenever `fuse()` runs again. If `fuse()`
    has never been invoked in this process, returns an empty report.
    """
    return _LAST_REPORT


# ---------------------------------------------------------------------------
# Consumer index + autograd holdout
# ---------------------------------------------------------------------------


def _consumer_index(root: UOp) -> dict[int, list[UOp]]:
    """Build `id(node) -> list[consumer_uops]` for every node reachable
    from `root`. The realizer doesn't need this — fusion does. O(n) over
    the toposort."""
    consumers: dict[int, list[UOp]] = {}
    for node in toposort(root):
        consumers.setdefault(id(node), [])
        for inp in node.inputs:
            consumers.setdefault(id(inp), []).append(node)
    return consumers


# ---------------------------------------------------------------------------
# Elementwise chain matcher
#
# A chain is a maximal-length linear sequence of ELEMENTWISE_OPS, each
# with exactly one consumer (so we can safely absorb intermediates
# without orphaning anything), none of whose non-terminal nodes are in
# the autograd holdout.
#
# The matcher walks downward from each "chain root" — a node whose
# predecessor is NOT elementwise — accumulating ops until the chain
# breaks. We then return the longest contiguous prefix that's safe to
# fuse. A chain of length < 2 isn't worth fusing (no allocations to save).
# ---------------------------------------------------------------------------


def _match_elementwise_chain(
    start: UOp,
    consumers: dict[int, list[UOp]],
    holdout: Set[int],
    already_fused: Set[int],
) -> Optional[list[UOp]]:
    """Find the longest elementwise chain rooted at `start`.

    Returns the chain in topological order (first op = start), or None
    if no fusable chain ≥ 2 exists.
    """
    if start.op not in ELEMENTWISE_OPS or id(start) in already_fused:
        return None
    chain: list[UOp] = [start]
    cur = start
    while True:
        consumer_list = consumers.get(id(cur), [])
        if len(consumer_list) != 1:
            break
        nxt = consumer_list[0]
        if nxt.op not in ELEMENTWISE_OPS:
            break
        if id(nxt) in already_fused:
            break
        # The CURRENT node will become a non-terminal in the chain if we
        # advance — and non-terminals can't be in the autograd holdout.
        if id(cur) in holdout:
            break
        # Shape/dtype agreement: a fused handler expresses the chain as
        # one ndarray per step. If a step has a different shape (broadcast
        # blowup mid-chain), the realizer would have to track a per-step
        # output shape. v0 keeps the invariant simple: all ops in the
        # chain have the same shape. Drop the simpler rule in a later
        # patch once we have a broadcast story.
        if nxt.shape != cur.shape or nxt.dtype != cur.dtype:
            break
        chain.append(nxt)
        cur = nxt
    if len(chain) < 2:
        return None
    return chain


def _build_fused_elementwise_uop(chain: list[UOp]) -> UOp:
    """Construct a single OP_FUSED_ELEMENTWISE UOp that replaces `chain`.

    `arg` carries everything the realizer needs:
      * `ops`: tuple of (opcode_str, lhs_ref, rhs_ref_or_None) tuples.
        Refs use negative integers for external inputs (so e.g. -1 means
        external_inputs[0]) and non-negative integers for "the output of
        step i". This avoids stale UOp object references in the arg.
      * `external_inputs`: tuple of UOps the fused node depends on. The
        UOp's own `.inputs` field is the same tuple — kept in `arg` too
        so the realizer can look up by ref index without re-deriving.
    """
    # Walk the chain in order, collecting external inputs and emitting
    # one op-step per chain node.
    external: list[UOp] = []
    external_id_to_ref: dict[int, int] = {}

    def _ref_for(node: UOp, in_chain: list[UOp], step_index: dict[int, int]) -> int:
        """Return the integer ref for `node`: negative for external, the
        producing step index for an in-chain intermediate."""
        if id(node) in step_index:
            return step_index[id(node)]
        if id(node) in external_id_to_ref:
            return external_id_to_ref[id(node)]
        ref = -(len(external) + 1)  # -1, -2, -3, ...
        external_id_to_ref[id(node)] = ref
        external.append(node)
        return ref

    step_index: dict[int, int] = {}
    ops: list[Tuple[str, int, Optional[int]]] = []
    for i, node in enumerate(chain):
        lhs = node.inputs[0]
        lhs_ref = _ref_for(lhs, chain, step_index)
        if len(node.inputs) == 2:
            rhs_ref = _ref_for(node.inputs[1], chain, step_index)
        else:
            rhs_ref = None
        ops.append((node.op, lhs_ref, rhs_ref))
        step_index[id(node)] = i

    terminal = chain[-1]
    return UOp(
        op=OP_FUSED_ELEMENTWISE,
        inputs=tuple(external),
        shape=terminal.shape,
        dtype=terminal.dtype,
        arg={"ops": tuple(ops)},
    )


# ---------------------------------------------------------------------------
# Softmax DAG matcher
#
# Canonical softmax IR built by browsergrad_jit._functional.softmax:
#
#   x        ──→ REDUCE(max,axis=A,keepdims=True)            (= max_node)
#     │                ↓
#     │              NEG                                       (= neg_node)
#     │                ↓
#     └──→ ADD ←─────  ─                                       (= add_node)
#               ↓
#              EXP                                             (= exp_node)
#               ↓
#         ┌───── ┴ ─────┐
#         ↓             ↓
#     REDUCE(sum,axis=A,keepdims=True)   (= sum_node)
#         └─────┬───────┘
#               ↓
#              DIV                                             (= div_node)
#               ↑
#              [exp_node]                                      (numerator)
#
# Matcher entry point: any DIV node. From DIV we walk backwards.
# ---------------------------------------------------------------------------


@dataclass
class _SoftmaxMatch:
    div_node: UOp
    exp_node: UOp
    add_node: UOp
    neg_node: UOp
    max_node: UOp
    sum_node: UOp
    x: UOp
    axis: int


def _match_softmax(
    div_node: UOp,
    consumers: dict[int, list[UOp]],
    holdout: Set[int],
    already_fused: Set[int],
) -> Optional[_SoftmaxMatch]:
    """Try to match a softmax DAG with `div_node` as the root."""
    if div_node.op != OP_DIV or id(div_node) in already_fused:
        return None
    if len(div_node.inputs) != 2:
        return None
    exp_node, sum_node = div_node.inputs
    # DIV's two inputs: numerator should be EXP, denominator a REDUCE(sum).
    if exp_node.op != OP_EXP or sum_node.op != OP_REDUCE:
        return None
    if id(exp_node) in already_fused or id(sum_node) in already_fused:
        return None
    if sum_node.arg.get("op") != "sum" or not sum_node.arg.get("keepdims"):
        return None
    if sum_node.inputs != (exp_node,):
        return None
    # EXP must have exactly two consumers (the DIV numerator + the SUM).
    exp_consumers = consumers.get(id(exp_node), [])
    if len(exp_consumers) != 2:
        return None
    consumer_ops = {c.op for c in exp_consumers}
    if consumer_ops != {OP_DIV, OP_REDUCE}:
        return None
    # ADD feeding EXP.
    if len(exp_node.inputs) != 1:
        return None
    add_node = exp_node.inputs[0]
    if add_node.op != OP_ADD or len(add_node.inputs) != 2:
        return None
    if id(add_node) in already_fused:
        return None
    # ADD's inputs: one is x, the other is NEG(max(x)).
    x_candidate, neg_candidate = add_node.inputs
    if neg_candidate.op != OP_NEG:
        x_candidate, neg_candidate = neg_candidate, x_candidate
    if neg_candidate.op != OP_NEG:
        return None
    if len(neg_candidate.inputs) != 1:
        return None
    max_node = neg_candidate.inputs[0]
    if max_node.op != OP_REDUCE:
        return None
    if max_node.arg.get("op") != "max" or not max_node.arg.get("keepdims"):
        return None
    if max_node.inputs != (x_candidate,):
        return None
    # Axis on max and sum must agree.
    axis_max = max_node.arg.get("axis")
    axis_sum = sum_node.arg.get("axis")
    if axis_max != axis_sum:
        return None
    # The intermediate nodes (max, neg, add, exp, sum) become non-terminal
    # inside the fused softmax — none of them may be in the autograd holdout.
    for n in (max_node, neg_candidate, add_node, exp_node, sum_node):
        if id(n) in holdout:
            return None
    # Intermediate node consumer-count check: max_node, neg_node, add_node,
    # sum_node must each have exactly one consumer (otherwise an external
    # reader would observe a value the fused softmax obliterates). EXP we
    # already verified has two. The terminal DIV's consumer count doesn't
    # matter — its consumers will be rewired to the fused output.
    for n in (max_node, neg_candidate, add_node, sum_node):
        if len(consumers.get(id(n), [])) != 1:
            return None
    return _SoftmaxMatch(
        div_node=div_node,
        exp_node=exp_node,
        add_node=add_node,
        neg_node=neg_candidate,
        max_node=max_node,
        sum_node=sum_node,
        x=x_candidate,
        axis=axis_max,
    )


def _build_fused_softmax_uop(match: _SoftmaxMatch) -> UOp:
    return UOp(
        op=OP_FUSED_SOFTMAX,
        inputs=(match.x,),
        shape=match.div_node.shape,
        dtype=match.div_node.dtype,
        arg={"axis": match.axis},
    )


# ---------------------------------------------------------------------------
# Whole-graph rewrite
# ---------------------------------------------------------------------------


def _substitute(root: UOp, substitutions: dict[int, UOp]) -> UOp:
    """Walk the graph rooted at `root` and rebuild it with `substitutions`
    applied. Returns the (possibly new) root.

    `substitutions` maps `id(original_uop) -> replacement_uop`. Original
    UOps in the chain interior are NOT in the map directly — only the
    chain terminal (or softmax DIV) is, mapped to the fused replacement.
    Anyone downstream that references the terminal gets the fused node
    instead.
    """
    if not substitutions:
        return root
    rebuilt: dict[int, UOp] = {}

    def _rebuild(node: UOp) -> UOp:
        if id(node) in rebuilt:
            return rebuilt[id(node)]
        if id(node) in substitutions:
            new_node = substitutions[id(node)]
        elif not node.inputs:
            new_node = node  # leaf
        else:
            new_inputs = tuple(_rebuild(inp) for inp in node.inputs)
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
        rebuilt[id(node)] = new_node
        return new_node

    return _rebuild(root)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def fuse(root: UOp, holdout: Optional[Iterable[int]] = None) -> UOp:
    """Run the fusion pass on the IR rooted at `root`.

    `holdout` is the set of UOp ids that must NOT become non-terminal nodes
    in any fused group — these are the UOps that backward closures will
    consume. The caller (the realizer) is responsible for assembling this
    set by walking every `TensorProxy._ctx.input_proxies[i]._uop` reachable
    from the realization driver.

    Returns a new UOp graph. If nothing fuses, the returned object is
    `root` itself (same identity, not a copy).
    """
    global _LAST_REPORT
    _LAST_REPORT = FusionReport()
    holdout_set: Set[int] = set(holdout) if holdout else set()

    consumers = _consumer_index(root)
    already_fused: Set[int] = set()
    substitutions: dict[int, UOp] = {}

    nodes = toposort(root)

    # Pass 1: softmax. Run first because softmax includes nodes (DIV, EXP)
    # that the elementwise matcher would otherwise greedily absorb,
    # producing a worse fusion overall.
    for node in nodes:
        match = _match_softmax(node, consumers, holdout_set, already_fused)
        if match is None:
            if node.op == OP_DIV and id(node) not in already_fused:
                _LAST_REPORT.unfused.append(UnfusedReason(
                    pattern="softmax",
                    op=node.op,
                    shape=node.shape,
                    reason="DIV did not match softmax template",
                ))
            continue
        fused = _build_fused_softmax_uop(match)
        substitutions[id(match.div_node)] = fused
        for n in (match.max_node, match.neg_node, match.add_node,
                  match.exp_node, match.sum_node, match.div_node):
            already_fused.add(id(n))
        _LAST_REPORT.fused.append(FusedKernelInfo(
            pattern="softmax",
            n_ops=6,  # max, neg, add, exp, sum, div
            shape=fused.shape,
            dtype=fused.dtype,
            ops=("REDUCE(max)", "NEG", "ADD", "EXP", "REDUCE(sum)", "DIV"),
        ))

    # Pass 2: elementwise chains. Walks each node; skips ones already
    # absorbed by softmax. The matcher returns the longest fusable chain
    # rooted at the candidate — but we only want to count "chain root"
    # candidates (nodes whose predecessor is NOT elementwise) to avoid
    # producing N overlapping chains.
    for node in nodes:
        if id(node) in already_fused:
            continue
        if node.op not in ELEMENTWISE_OPS:
            continue
        # Skip if any consumer of one of this node's inputs is in the same
        # elementwise group — that means this node is mid-chain, and a
        # parent will be the chain root instead.
        is_chain_root = True
        for inp in node.inputs:
            if inp.op in ELEMENTWISE_OPS and id(inp) not in already_fused:
                # An elementwise predecessor exists; only walk from the root.
                # The walk-from-this-node loop will still find the chain when
                # we eventually hit the actual root.
                if len(consumers.get(id(inp), [])) == 1:
                    is_chain_root = False
                    break
        if not is_chain_root:
            continue
        chain = _match_elementwise_chain(node, consumers, holdout_set, already_fused)
        if chain is None:
            _LAST_REPORT.unfused.append(UnfusedReason(
                pattern="elementwise_chain",
                op=node.op,
                shape=node.shape,
                reason="chain length < 2 or autograd holdout / shape mismatch",
            ))
            continue
        fused = _build_fused_elementwise_uop(chain)
        substitutions[id(chain[-1])] = fused
        for n in chain:
            already_fused.add(id(n))
        _LAST_REPORT.fused.append(FusedKernelInfo(
            pattern="elementwise_chain",
            n_ops=len(chain),
            shape=fused.shape,
            dtype=fused.dtype,
            ops=tuple(n.op for n in chain),
        ))

    return _substitute(root, substitutions)


# ---------------------------------------------------------------------------
# Holdout collection — public so the realizer can ask "give me the set of
# UOp ids that backward closures will consume, starting from this proxy."
# ---------------------------------------------------------------------------


def collect_autograd_holdout(leaf_proxies: Iterable[Any]) -> Set[int]:
    """Collect the set of UOp ids referenced by any `_BackwardCtx` reachable
    from `leaf_proxies`. The fusion pass must not absorb these UOps as
    non-terminal nodes.

    `leaf_proxies` is typically the set of proxies the realizer is about
    to realize (e.g. `[loss]` for a `.backward()` flow), plus any proxies
    whose `_ctx` references upstream UOps.
    """
    held: Set[int] = set()
    seen: Set[int] = set()
    stack: list[Any] = list(leaf_proxies)
    while stack:
        p = stack.pop()
        if p is None or id(p) in seen:
            continue
        seen.add(id(p))
        ctx = getattr(p, "_ctx", None)
        if ctx is None:
            continue
        for inp in ctx.input_proxies:
            # The UOp the closure will read is inp._uop.
            held.add(id(inp._uop))
            stack.append(inp)
    return held


__all__ = [
    "fuse",
    "collect_autograd_holdout",
    "get_last_report",
    "FusionReport",
    "FusedKernelInfo",
    "UnfusedReason",
    "ELEMENTWISE_OPS",
]
