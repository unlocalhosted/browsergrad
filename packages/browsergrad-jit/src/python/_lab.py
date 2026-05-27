"""browsergrad_jit._lab — lab harness primitives (PRD-013).

INTERNAL. Public surface lives at `bg.lab.{assert_pytorch_match,
assert_shape_match, assert_no_nan_inf}`. These call into the runtime's
`browsergrad` module (`browsergrad.assert_pass / assert_fail /
assert_error`) which is preloaded by the worker preamble.

Why a separate module from the runtime helpers:
  * Runtime helpers ship low-level primitives (pass/fail/error/log/json).
  * Lab harness helpers are *semantic* — they encode what a rubric
    actually wants to assert (numerical match within tolerance, shape
    match, finiteness). Authors should write `assert_pytorch_match`,
    not three lines of `np.allclose` + tolerance + assert_pass.
  * Keeps lab authors from leaking into runtime internals.

Three primitives in v0 per the DL/GPU review. Adding more is mechanical
once a rubric author asks; the cost is paid in the v0 maintenance budget.

Fail-soft mode:
  * If `browsergrad` module isn't importable (we're running outside a
    runtime-managed worker), the helpers print a structured message to
    stdout instead. This lets unit tests exercise them without booting
    the full worker stack.
"""

from __future__ import annotations
import time
from typing import Any, Optional


def _get_emitter():
    """Resolve the runtime's `browsergrad` module. Returns None if we're
    running outside the worker (e.g. plain Pyodide-in-node)."""
    try:
        import browsergrad as _bg
        return _bg
    except ImportError:
        return None


def _emit_pass(name: str, duration_ms: Optional[float]) -> None:
    em = _get_emitter()
    if em is not None and hasattr(em, "assert_pass"):
        em.assert_pass(name, duration_ms)
    else:
        # Fallback: structured stdout for plain pyodide.
        print(f"[bg.lab pass] {name} ({duration_ms:.2f}ms)" if duration_ms else f"[bg.lab pass] {name}")


def _emit_fail(name: str, message: str, expected: Any, actual: Any,
               duration_ms: Optional[float]) -> None:
    em = _get_emitter()
    if em is not None and hasattr(em, "assert_fail"):
        em.assert_fail(name, message, expected, actual, duration_ms)
    else:
        print(f"[bg.lab FAIL] {name}: {message}")
        print(f"  expected: {expected!r}")
        print(f"  actual:   {actual!r}")


def _emit_error(name: str, message: str, exc: Optional[BaseException],
                duration_ms: Optional[float]) -> None:
    em = _get_emitter()
    if em is not None and hasattr(em, "assert_error"):
        em.assert_error(name, message, exc, duration_ms)
    else:
        print(f"[bg.lab ERROR] {name}: {message}")
        if exc is not None:
            print(f"  cause: {type(exc).__name__}: {exc}")


def _shape_of(x: Any) -> Any:
    """Resolve shape across TensorProxy, numpy ndarray, Python list/tuple
    leaves. Returns a tuple."""
    if hasattr(x, "shape"):
        s = x.shape
        return tuple(s)
    if isinstance(x, (list, tuple)):
        return (len(x),)
    raise TypeError(
        f"bg.lab: can't take .shape of {type(x).__name__}; pass a TensorProxy "
        f"or np.ndarray"
    )


def _to_numpy(x: Any) -> Any:
    """Resolve a value to np.ndarray. TensorProxy → .numpy(); ndarray
    pass-through; list/tuple → np.asarray."""
    import numpy as np
    if hasattr(x, "numpy") and callable(x.numpy):
        return x.numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def assert_pytorch_match(
    name: str,
    actual: Any,
    expected: Any,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    """Assert that `actual` matches `expected` within rtol/atol tolerance.

    Returns True iff the assertion passed (and emitted assert_pass);
    returns False on fail/error (and emitted assert_fail/error).

    The name "pytorch_match" is intentional — it signals the rubric is
    checking against a PyTorch reference and the tolerances are chosen
    to absorb f32 round-trip noise.
    """
    import numpy as np
    t0 = time.perf_counter()
    try:
        a = _to_numpy(actual)
        e = _to_numpy(expected)
        if a.shape != e.shape:
            duration = (time.perf_counter() - t0) * 1000
            _emit_fail(
                name,
                f"shape mismatch: actual {a.shape} vs expected {e.shape}",
                expected=e.shape,
                actual=a.shape,
                duration_ms=duration,
            )
            return False
        if not np.allclose(a, e, rtol=rtol, atol=atol):
            duration = (time.perf_counter() - t0) * 1000
            diff = float(np.max(np.abs(a - e)))
            _emit_fail(
                name,
                f"values diverge: max |a - e| = {diff:.3e} > "
                f"rtol={rtol:.0e} * |e| + atol={atol:.0e}",
                expected=expected,
                actual=actual,
                duration_ms=duration,
            )
            return False
        duration = (time.perf_counter() - t0) * 1000
        _emit_pass(name, duration)
        return True
    except Exception as exc:
        duration = (time.perf_counter() - t0) * 1000
        _emit_error(name, f"assertion raised {type(exc).__name__}",
                    exc, duration)
        return False


def assert_shape_match(
    name: str,
    actual: Any,
    expected_shape: tuple,
) -> bool:
    """Assert the shape of `actual` matches `expected_shape`."""
    t0 = time.perf_counter()
    try:
        a_shape = _shape_of(actual)
        e_shape = tuple(expected_shape)
        if a_shape != e_shape:
            duration = (time.perf_counter() - t0) * 1000
            _emit_fail(
                name,
                f"shape mismatch: got {a_shape} vs expected {e_shape}",
                expected=e_shape,
                actual=a_shape,
                duration_ms=duration,
            )
            return False
        duration = (time.perf_counter() - t0) * 1000
        _emit_pass(name, duration)
        return True
    except Exception as exc:
        duration = (time.perf_counter() - t0) * 1000
        _emit_error(name, f"shape assertion raised {type(exc).__name__}",
                    exc, duration)
        return False


def assert_no_nan_inf(
    name: str,
    actual: Any,
) -> bool:
    """Assert that `actual` contains no NaN or infinite values."""
    import numpy as np
    t0 = time.perf_counter()
    try:
        a = _to_numpy(actual)
        finite_mask = np.isfinite(a)
        if not finite_mask.all():
            duration = (time.perf_counter() - t0) * 1000
            n_bad = int((~finite_mask).sum())
            _emit_fail(
                name,
                f"found {n_bad} non-finite values in tensor of "
                f"shape {a.shape}",
                expected="all finite",
                actual=f"{n_bad} NaN/Inf",
                duration_ms=duration,
            )
            return False
        duration = (time.perf_counter() - t0) * 1000
        _emit_pass(name, duration)
        return True
    except Exception as exc:
        duration = (time.perf_counter() - t0) * 1000
        _emit_error(name, f"finiteness assertion raised {type(exc).__name__}",
                    exc, duration)
        return False


__all__ = [
    "assert_pytorch_match",
    "assert_shape_match",
    "assert_no_nan_inf",
]
