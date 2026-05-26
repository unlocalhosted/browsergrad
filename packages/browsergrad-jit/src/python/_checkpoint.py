"""browsergrad_jit._checkpoint — gradient checkpointing via IR rewrite.

INTERNAL. Public API is `browsergrad_jit.utils.checkpoint.checkpoint(fn, *args)`,
PyTorch-shaped.

Mechanism (per PRD-009 v0 review):

  * The forward pass runs `fn(*args)` normally. During the call, every
    UOp that gets constructed is recorded as "interior" to the region.
    The region's "anchors" are the `args` UOps; the region's "outputs"
    are the UOps of the return value.
  * The backward pass (symbolic VJP only — closure path is refused)
    walks the gradient graph. Wherever it would read an interior UOp's
    value, we substitute a freshly-CLONED forward subgraph rooted at
    the same anchors. The realizer then re-computes those values
    independently of the original forward's `value_table`, which has
    been GC'd after the forward returned.
  * On NumPy the memory win is modest (Python heap, not GPU buffer
    pool). The mechanism is load-bearing for PRD-012 (WGSL megakernels)
    and PRD-014 (vmap), where activation memory genuinely binds.

Refusal modes:

  * `use_reentrant=True` — PyTorch's deprecated reentrant autograd;
    not implemented.
  * `preserve_rng_state=True` — requires Philox RNG capture; the
    factory ops (`randn`, etc.) don't expose per-call seeds yet. The
    Korthikanti-style dropout decomposition (PRD-007 deliverable §2
    row 3) will land it.
  * Nested `checkpoint(checkpoint(fn), ...)` — region scoping isn't
    designed for nesting in v0. Raise.
  * A region whose forward contains an op without a registered VJP
    rule — the backward dispatcher would fall back to the closure
    path which can't be cleanly checkpointed.

The rewrite pass uses the same `_substitute`-style algorithm as the
fusion pass — same tree walk, different substitution map.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Iterable, Optional, Tuple

from ._ir import UOp, toposort
from ._vjp import get_rule


# ---------------------------------------------------------------------------
# Region registry
#
# Process-global. Keyed by region_id. The region holds the data the
# backward dispatcher needs to plan the rewrite:
#   * anchor_uop_ids: forward-pass UOps that are inputs to fn — these
#     stay in the original graph and are realized once.
#   * interior_uop_ids: forward-pass UOps constructed during fn's
#     execution. The backward rewrite replaces references to these.
#   * output_uop_ids: UOps that fn returned. Acts as a sanity check;
#     not load-bearing for the rewrite.
#
# Process-globality is deliberate. The TensorProxy chain points at UOps
# by Python identity; the region records UOp ids. A per-session
# registry would force every backward call to thread a session
# reference into the rewrite — wasted complexity for v0.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointRegion:
    region_id: str
    anchor_uop_ids: FrozenSet[int]
    interior_uop_ids: FrozenSet[int]
    output_uop_ids: Tuple[int, ...]
    # The anchor UOps themselves are kept by direct reference so the
    # clone walk can terminate cleanly. Holding the references prevents
    # GC of the anchor UOps mid-backward.
    anchor_uops: Tuple[UOp, ...]


_REGIONS: Dict[str, CheckpointRegion] = {}
_REGION_COUNTER: int = 0
_IN_FLIGHT: list[str] = []  # stack of currently-open region ids for nesting detection


class CheckpointError(RuntimeError):
    """Raised when a checkpointed region's preconditions are not met."""


# ---------------------------------------------------------------------------
# Region open / close
# ---------------------------------------------------------------------------


def _next_region_id() -> str:
    global _REGION_COUNTER
    _REGION_COUNTER += 1
    return f"ckpt_{_REGION_COUNTER}"


def _open_region(anchor_proxies: Tuple[Any, ...]) -> str:
    """Open a new region scoped at the given anchor proxies.

    Returns the region id. Nested opens are refused — the v0 design
    doesn't support nesting because the interior-set computation
    is unique per region (it walks back from outputs and stops at
    anchors; with nesting, "anchor of outer" could be "interior of
    inner" and the semantics get muddled).
    """
    if _IN_FLIGHT:
        raise CheckpointError(
            f"checkpoint nesting is not supported in v0. An outer "
            f"checkpoint({_IN_FLIGHT[-1]!r}) is already open. Restructure "
            f"the inner call to not need checkpointing, or wait for "
            f"PRD-009.2 which adds region collapse."
        )
    rid = _next_region_id()
    _IN_FLIGHT.append(rid)
    return rid


def _close_region(
    region_id: str,
    anchor_proxies: Tuple[Any, ...],
    output_proxies: Tuple[Any, ...],
) -> CheckpointRegion:
    """Close the region. Computes the interior UOp set by walking
    backward from outputs and stopping at anchors.

    Validates that every interior UOp's opcode has a registered VJP
    rule. Refuses the close (and the checkpoint) if any op would force
    the backward dispatcher to fall back to the closure path.
    """
    if not _IN_FLIGHT or _IN_FLIGHT[-1] != region_id:
        raise CheckpointError(
            f"_close_region: stack mismatch (expected {region_id!r}, "
            f"got {_IN_FLIGHT!r})"
        )
    _IN_FLIGHT.pop()

    anchor_uops = tuple(p._uop for p in anchor_proxies)
    anchor_ids = frozenset(id(u) for u in anchor_uops)
    output_uops = tuple(p._uop for p in output_proxies)
    output_ids = tuple(id(u) for u in output_uops)

    # Walk back from each output, stopping at anchors. Everything visited
    # that's not an anchor is interior.
    interior_ids: set[int] = set()
    interior_uops: list[UOp] = []
    stack = list(output_uops)
    seen: set[int] = set()
    while stack:
        u = stack.pop()
        if id(u) in seen:
            continue
        seen.add(id(u))
        if id(u) in anchor_ids:
            continue
        # Leaves (BUFFER, CONST, RANDOM, LOAD-of-BUFFER) are anchors-of-
        # convenience too: they're either parameters (stay in the graph)
        # or already-realized constants (can be recomputed for free).
        # Only mark as "interior" if there's actual work to recompute.
        if not u.inputs:
            continue
        interior_ids.add(id(u))
        interior_uops.append(u)
        for inp in u.inputs:
            stack.append(inp)

    # VJP-rule coverage check.
    for u in interior_uops:
        if get_rule(u.op) is None and u.op not in ("LOAD",):
            raise CheckpointError(
                f"checkpoint(fn): interior op {u.op!r} has no registered "
                f"VJP rule. Checkpointing requires the symbolic backward "
                f"path. Either rewrite fn to avoid {u.op!r}, or register "
                f"a VJP rule for it."
            )

    region = CheckpointRegion(
        region_id=region_id,
        anchor_uop_ids=anchor_ids,
        interior_uop_ids=frozenset(interior_ids),
        output_uop_ids=output_ids,
        anchor_uops=anchor_uops,
    )
    _REGIONS[region_id] = region
    return region


# ---------------------------------------------------------------------------
# Backward-time rewrite
# ---------------------------------------------------------------------------


def has_any_region() -> bool:
    """True iff any checkpoint region has been recorded. Cheap check
    the backward dispatcher uses to skip the rewrite call when no
    checkpoints exist."""
    return bool(_REGIONS)


def _all_interior_ids() -> set[int]:
    out: set[int] = set()
    for r in _REGIONS.values():
        out |= r.interior_uop_ids
    return out


def apply_checkpoint_rewrite(grad_uop: UOp) -> UOp:
    """Walk `grad_uop` and replace any reference to a forward UOp that
    sits inside any region's `interior_uop_ids` with a freshly-cloned
    forward subgraph.

    The clone walk terminates at anchors (region inputs) and leaves
    (BUFFER/CONST/etc.) — those stay shared with the original forward
    graph. Every interior UOp produces a fresh clone with the same
    opcode/shape/dtype/arg but a fresh Python identity, so the realizer
    re-computes its value independently of any GC'd `value_table` entry
    from the forward pass.
    """
    if not _REGIONS:
        return grad_uop

    interior = _all_interior_ids()
    if not interior:
        return grad_uop

    # Build a clone-cache so a re-referenced interior UOp produces the
    # SAME clone (preserving shared-subexpression structure inside the
    # cloned subgraph).
    clone_cache: Dict[int, UOp] = {}

    def _clone_if_interior(u: UOp) -> UOp:
        """If `u` is interior, return its clone (creating one if needed,
        whose inputs are recursively clones-of-interior). Otherwise
        return `u` unchanged."""
        if id(u) not in interior:
            return u
        if id(u) in clone_cache:
            return clone_cache[id(u)]
        new_inputs = tuple(_clone_if_interior(inp) for inp in u.inputs)
        clone = UOp(
            op=u.op,
            inputs=new_inputs,
            shape=u.shape,
            dtype=u.dtype,
            arg=u.arg,
        )
        clone_cache[id(u)] = clone
        return clone

    # Now walk the grad graph and rebuild any UOp whose input set
    # touches an interior node. Same algorithm as _fusion._substitute.
    rebuilt: Dict[int, UOp] = {}

    def _rebuild(node: UOp) -> UOp:
        if id(node) in rebuilt:
            return rebuilt[id(node)]
        # If `node` is itself interior, return its clone directly.
        if id(node) in interior:
            cloned = _clone_if_interior(node)
            rebuilt[id(node)] = cloned
            return cloned
        if not node.inputs:
            rebuilt[id(node)] = node
            return node
        new_inputs = tuple(_rebuild(inp) for inp in node.inputs)
        if all(a is b for a, b in zip(new_inputs, node.inputs)):
            rebuilt[id(node)] = node
            return node
        rebuilt_node = UOp(
            op=node.op,
            inputs=new_inputs,
            shape=node.shape,
            dtype=node.dtype,
            arg=node.arg,
        )
        rebuilt[id(node)] = rebuilt_node
        return rebuilt_node

    return _rebuild(grad_uop)


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------


def clear_all_regions() -> None:
    """Drop every recorded region. Intended for tests. Production code
    should never need this — regions accumulate but the per-region
    memory cost is just a few sets of integers."""
    _REGIONS.clear()
    _IN_FLIGHT.clear()


def region_count() -> int:
    """Observability hook."""
    return len(_REGIONS)


__all__ = [
    "CheckpointRegion",
    "CheckpointError",
    "_open_region",
    "_close_region",
    "apply_checkpoint_rewrite",
    "has_any_region",
    "clear_all_regions",
    "region_count",
]
