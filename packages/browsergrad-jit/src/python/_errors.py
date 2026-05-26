"""browsergrad_jit._errors — typed exceptions raised by the IR and realizer.

INTERNAL. The exception *types* are re-exported via the public namespace
so users can `except browsergrad_jit.ShapeError:` etc., but the module
path `_errors` is private.

Every error type has a docstring that is shown in IDE tooltips and printed
at the top of the traceback. Make these worth reading — the curriculum
relies on tracebacks being legible.
"""

from __future__ import annotations


class JitError(Exception):
    """Base class for every browsergrad_jit-raised exception.

    Catch this if you want to surface "the JIT failed" without caring about
    which subtype. Most user code should catch a more specific subtype."""


class ShapeError(JitError, ValueError):
    """Raised at UOp construction when shapes/dtypes don't line up.

    Subclasses ValueError too so existing PyTorch-style except clauses keep
    working. Carry the offending shapes in the message; never elide them
    — students will quote the error in chat and we'll need to debug it."""


class TorchAliasConflict(JitError, RuntimeError):
    """`install_torch_alias()` refused because another package already
    owns `sys.modules['torch']`.

    Resolution: call the conflicting package's `uninstall_torch_alias()`
    first, or pass `force=True` if you know what you're doing (tests do
    this between scenarios)."""


class NoBackwardError(JitError, RuntimeError):
    """A `TensorProxy` requested `.backward()` but the relevant op has no
    registered VJP rule.

    Most often this fires when a user-supplied `@bg.kernel("wgsl")` custom
    op (PRD-015) has not registered a backward via `.register_backward()`.
    The PRD-005 baseline ops all have backwards; if you see this in a
    stock browsergrad_jit operation, file a bug."""


class JitNotImplementedError(JitError, NotImplementedError):
    """The requested operation is on the JIT roadmap but not yet shipped
    in this version.

    Distinguishes from raw NotImplementedError so test harnesses can
    selectively `xfail` the slow features (PRD-006 fusion, PRD-007
    symbolic backward, PRD-010 mixed precision, etc.) without masking
    real bugs."""


class RealizationError(JitError, RuntimeError):
    """Something went wrong during realization of the IR — the topological
    sort, the dispatch table, or the underlying NumPy / WGSL call.

    Always carries the offending UOp's opcode + shape in the message."""


class BufferTableError(JitError, RuntimeError):
    """Raised when the per-session BufferTable rejects a request: unknown
    buffer_id, duplicate registration, or cross-session contamination."""


__all__ = [
    "JitError",
    "ShapeError",
    "TorchAliasConflict",
    "NoBackwardError",
    "JitNotImplementedError",
    "RealizationError",
    "BufferTableError",
]
