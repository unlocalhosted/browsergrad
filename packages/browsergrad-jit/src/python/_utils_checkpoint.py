"""browsergrad_jit.utils.checkpoint — PyTorch-shaped public API for
gradient checkpointing.

Surfaces:
  * `checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)`
    — wraps fn such that its forward intermediates are dropped after
    the call and re-computed during backward via IR rewrite.

INTERNAL implementation lives in `_checkpoint.py`. This module is the
thin shim that PyTorch-shaped code (`from torch.utils.checkpoint import
checkpoint`) imports.
"""

from __future__ import annotations
from typing import Any, Callable, Tuple

from . import _checkpoint


def checkpoint(
    fn: Callable[..., Any],
    *args: Any,
    use_reentrant: bool = False,
    preserve_rng_state: bool = False,
) -> Any:
    """Run `fn(*args)` and drop the forward intermediates from the
    realizer's value table after the call returns. When `.backward()`
    fires, the gradient-graph references to dropped UOps are rewritten
    into freshly-cloned forward subgraphs rooted at the original `args`.

    Memory saving on the NumPy realizer is modest (Python heap pressure
    rather than GPU buffer-pool exhaustion); the IR-rewrite mechanism
    is the load-bearing piece that unlocks PRD-012 (WGSL megakernels)
    and PRD-014 (vmap).

    Constraints:
      * `use_reentrant=True` is refused. PyTorch's deprecated reentrant
        autograd doesn't exist in this library.
      * `preserve_rng_state=True` is refused. The forward must be
        deterministic up to the args; until factory ops (`randn`, etc.)
        expose per-call Philox seeds (PRD-007 follow-up), recompute
        with random sampling diverges from the original.
      * `fn`'s forward must only use ops with registered symbolic VJP
        rules. The closure-backward path can't be safely checkpointed
        (it captures intermediate ndarrays by reference; we can't tell
        at rewrite time which are interior).
      * Nested checkpointing isn't supported in v0.

    `args` are positional only. `fn` should return a single TensorProxy
    or a tuple of them; both are handled.
    """
    if use_reentrant:
        raise _checkpoint.CheckpointError(
            "checkpoint(use_reentrant=True): reentrant autograd is "
            "PyTorch's deprecated mode and is not implemented here. "
            "Pass use_reentrant=False (the default in modern PyTorch)."
        )
    if preserve_rng_state:
        raise _checkpoint.CheckpointError(
            "checkpoint(preserve_rng_state=True): the symbolic backward "
            "doesn't yet capture per-op Philox state. Stochastic ops "
            "(dropout, randn, etc.) inside a checkpointed region would "
            "diverge from their forward values on recompute. This lands "
            "alongside the dropout-decomposition in PRD-007's CUSTOM "
            "op migration."
        )

    # The anchors are the input TensorProxies. We pass them by tuple so
    # the close walker can compute interior = reachable-from-outputs-
    # minus-anchors cleanly.
    from ._tensor_proxy import TensorProxy
    anchor_proxies: list[TensorProxy] = []
    for a in args:
        if isinstance(a, TensorProxy):
            anchor_proxies.append(a)
        # Non-TensorProxy args (scalars, ints) are allowed but don't
        # participate in the IR — they get baked into the fn's body
        # as Python constants. No anchor needed.

    region_id = _checkpoint._open_region(tuple(anchor_proxies))
    try:
        out = fn(*args)
    except BaseException:
        # On any exception (including the user pressing Ctrl-C), the
        # in-flight region stack must unwind cleanly so subsequent
        # checkpoint() calls don't see a stale "we're already inside
        # a region" state.
        _checkpoint._IN_FLIGHT.pop()
        raise

    # Normalize output to a tuple for the region recorder, then unpack
    # back to its original shape for the caller.
    if isinstance(out, tuple):
        output_proxies = out
    else:
        output_proxies = (out,)

    _checkpoint._close_region(
        region_id,
        tuple(anchor_proxies),
        tuple(p for p in output_proxies if isinstance(p, TensorProxy)),
    )
    return out


__all__ = ["checkpoint"]
