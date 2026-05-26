"""browsergrad_jit._trace_cache — in-memory IR-trace cache.

INTERNAL. Public surface is `bg.jit.use_trace_cache(bool)`,
`bg.jit.trace_cache_stats()`, and the automatic integration with
`nn.Module.__call__`.

Why this exists (PRD-008 critique #4):
  A forward pass on a 12-layer transformer at (B=8, seq=128) builds
  ~300 UOps per step. UOp construction is ~3μs each in Pyodide → ~1 ms
  of pure Python overhead per training step. A trace cache keyed on
  `(fn_identity, shape_signature, dtype_signature, training_flag)`
  reuses the IR graph across calls. First call: full trace + cache write.
  Subsequent calls with matching signature: O(input-tensor-count) leaf
  rebinding, no UOp construction.

What we cache:
  * The output `TensorProxy._uop` produced by the forward call.
  * A map from positional-input-index → the BUFFER `UOp` that input
    leaf was bound to on the cached forward.

What we DON'T cache:
  * The output proxy's `_ctx` (backward closure) — closure-based backward
    isn't graph-rewritable. v0 disables the cache when any input has
    `requires_grad=True`. PRD-007 W3+ extends this once VJP rules
    fully cover the chain.
  * Output values. Realization happens fresh per call.

Constraints (documented as conditions in the user-facing docstring):
  * Forward must be deterministic (no `.item()`, `.numpy()`, `if tensor`,
    or other realization triggers in user code).
  * Same `training` mode across calls (different mode = different sig).
  * No data-dependent control flow (`if x.shape[0] > 32`); the cache key
    only looks at static shape, not runtime values.

How leaf rebinding works:
  The first call produces an IR rooted at some UOp R. R's leaves are
  BUFFER+LOAD pairs pointing at the input tensors' BufferTable entries
  AND the module's parameters. We capture the input-side BUFFER nodes
  by index. On a subsequent call:
    * Build a substitution map: cached_input_buffer → new_input_buffer.
      Parameters are NOT in the map — they're reused as-is (the same
      module is on the call site; its parameters live at the same
      buffer ids).
    * Walk R rebuilding every UOp whose dependency chain touches a
      cached input; substitute the new BUFFER at the leaf.
  This is the same machinery as `_fusion._substitute` — different keys,
  same algorithm.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

from ._ir import UOp


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


_ENABLED: bool = True
_HITS: int = 0
_MISSES: int = 0
# Cache state. Keyed by (module-id, training_flag, shape_dtype_signature).
# The signature is a tuple of (shape, dtype) per positional input.
_CACHE: Dict[Tuple[int, bool, tuple], "_CompiledTrace"] = {}


def is_enabled() -> bool:
    return _ENABLED


def use_trace_cache(enabled: bool) -> None:
    """Toggle trace caching. Default True. Set False to debug a model
    whose forward has Python-side side effects (printing, mid-graph
    `.numpy()`, etc.) that the cache can't observe."""
    global _ENABLED
    _ENABLED = bool(enabled)


def stats() -> Dict[str, int]:
    """Cache observability surface. Read-only counters."""
    return {
        "enabled": int(_ENABLED),
        "entries": len(_CACHE),
        "hits": _HITS,
        "misses": _MISSES,
    }


def clear() -> None:
    """Wipe the cache. Useful for test isolation and for callers that
    want to free Python memory after a long-running session."""
    global _HITS, _MISSES
    _CACHE.clear()
    _HITS = 0
    _MISSES = 0


# ---------------------------------------------------------------------------
# Compiled trace record
# ---------------------------------------------------------------------------


class _CompiledTrace:
    """The cached artifact: an IR root plus the BUFFER UOps that
    represent each input's leaf in the cached graph."""

    __slots__ = ("output_uop", "output_shape", "output_dtype",
                 "input_buffer_uops", "output_requires_grad")

    def __init__(
        self,
        output_uop: UOp,
        output_shape: Tuple[int, ...],
        output_dtype: str,
        input_buffer_uops: Tuple[UOp, ...],
        output_requires_grad: bool,
    ) -> None:
        self.output_uop = output_uop
        self.output_shape = output_shape
        self.output_dtype = output_dtype
        self.input_buffer_uops = input_buffer_uops
        self.output_requires_grad = output_requires_grad


# ---------------------------------------------------------------------------
# Signature + key computation
# ---------------------------------------------------------------------------


def _signature(args: Tuple[Any, ...]) -> Optional[tuple]:
    """Compute a (shape, dtype) signature from positional args.

    Returns None if any argument:
      * Isn't a TensorProxy (we don't cache calls with kwargs / scalars /
        Python lists; the cache only handles homogeneous-tensor inputs).
      * Has `requires_grad=True` (autograd closures aren't graph-rewritable
        for v0).
    """
    from ._tensor_proxy import TensorProxy
    sig: list = []
    for a in args:
        if not isinstance(a, TensorProxy):
            return None
        if a.requires_grad:
            return None
        sig.append((a.shape, a.dtype))
    return tuple(sig)


# ---------------------------------------------------------------------------
# Leaf-rebinding rewriter
# ---------------------------------------------------------------------------


def _input_buffer_uop(proxy) -> Optional[UOp]:
    """Return the BUFFER UOp behind a TensorProxy whose graph is a single
    LOAD(BUFFER). Returns None if the proxy isn't a leaf — anything
    further upstream and we can't cleanly substitute it.
    """
    uop = proxy._uop
    if uop.op == "LOAD" and len(uop.inputs) == 1 and uop.inputs[0].op == "BUFFER":
        return uop.inputs[0]
    return None


def _rebind(root: UOp, subs: Dict[int, UOp]) -> UOp:
    """Walk `root` rebuilding every UOp whose subgraph touches a
    substituted leaf. `subs` maps `id(old_buffer)` → `new_buffer`.
    Identical algorithm to `_fusion._substitute`."""
    if not subs:
        return root
    rebuilt: Dict[int, UOp] = {}

    def _rebuild(node: UOp) -> UOp:
        if id(node) in rebuilt:
            return rebuilt[id(node)]
        if id(node) in subs:
            new_node = subs[id(node)]
        elif not node.inputs:
            new_node = node  # leaf, not in subs → keep
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
# Public lookup + insert
# ---------------------------------------------------------------------------


def maybe_cached_forward(
    module_id: int,
    training: bool,
    args: Tuple[Any, ...],
) -> Optional[Any]:
    """Return a TensorProxy reconstructed from the cache, or None on miss.

    The reconstructed proxy carries the cached IR's output UOp, rebound
    to the *current call's* input BUFFERs. requires_grad propagates from
    the cached value (always False at v0, since we refuse to cache when
    any input requires grad).
    """
    global _HITS, _MISSES
    if not _ENABLED:
        return None
    sig = _signature(args)
    if sig is None:
        return None
    key = (module_id, training, sig)
    cached = _CACHE.get(key)
    if cached is None:
        _MISSES += 1
        return None
    # Build the substitution map: cached input BUFFER → current input BUFFER.
    subs: Dict[int, UOp] = {}
    for cached_buf, new_proxy in zip(cached.input_buffer_uops, args):
        new_buf = _input_buffer_uop(new_proxy)
        if new_buf is None:
            # New input isn't a leaf LOAD-of-BUFFER; we can't substitute
            # without rewriting more of the graph. Miss.
            _MISSES += 1
            return None
        subs[id(cached_buf)] = new_buf
    new_root = _rebind(cached.output_uop, subs)
    _HITS += 1
    # Late import — same circular-dep pattern as _safetensors.
    from ._tensor_proxy import TensorProxy
    return TensorProxy(
        new_root,
        session=args[0]._get_session(),
        requires_grad=cached.output_requires_grad,
    )


def record(
    module_id: int,
    training: bool,
    args: Tuple[Any, ...],
    output: Any,
) -> None:
    """Insert a cache entry. Called after a real forward call lands.

    Refuses to insert if:
      * Any input's signature is invalid (matches `maybe_cached_forward`'s
        rejection rule).
      * Any input is not a leaf LOAD-of-BUFFER (can't capture the input
        buffer for rebinding).
      * The output proxy carries a `_ctx` (closure backward) — caching
        the output would break gradient propagation.
    """
    if not _ENABLED:
        return
    sig = _signature(args)
    if sig is None:
        return
    if output._ctx is not None:
        # Can't safely cache an output with a closure-based backward —
        # the closure captures input_proxies by identity. Rebinding
        # gives different proxies; the closure would try to read the
        # wrong values.
        return
    input_buffers = tuple(_input_buffer_uop(a) for a in args)
    if any(b is None for b in input_buffers):
        return
    key = (module_id, training, sig)
    _CACHE[key] = _CompiledTrace(
        output_uop=output._uop,
        output_shape=output._uop.shape,
        output_dtype=output._uop.dtype,
        input_buffer_uops=tuple(b for b in input_buffers if b is not None),
        output_requires_grad=output.requires_grad,
    )


__all__ = [
    "use_trace_cache",
    "is_enabled",
    "stats",
    "clear",
    "maybe_cached_forward",
    "record",
]
