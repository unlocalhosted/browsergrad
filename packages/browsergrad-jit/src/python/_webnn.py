"""browsergrad_jit._webnn — experimental WebNN capability detection.

INTERNAL. Public surface: `bg.experimental.webnn.is_available`.

The full PRD-011 WebNN backend remains deferred until BrowserGrad owns a real
IR partitioner, lowering, capability record, execution bridge, and fallback
contract. Presence detection alone does not claim that any operation can run
through WebNN.

This module:
  * Detects `navigator.ml` via Pyodide's `js` import.
  * Lives behind `bg.experimental` to signal instability.

What this is NOT:
  * A full WebNN backend with op partitioner, tier selector, fallback
    machinery, or execution evidence.
  * An operation-construction API. The former constructor-only
    `webnn_matmul` surface was removed because every realizer rejected it.
  * A WGSL replacement — WGSL stays the default GPU path.
"""

from __future__ import annotations


def is_available() -> bool:
    """Report whether navigator.ml exists in the runtime. Returns False
    outside Pyodide (e.g. in Node-based tests) — the import will succeed
    but the attribute won't be present."""
    try:
        import js  # type: ignore[import-not-found]
    except ImportError:
        return False
    nav = getattr(js, "navigator", None)
    if nav is None:
        return False
    return hasattr(nav, "ml") and nav.ml is not None


__all__ = ["is_available"]
