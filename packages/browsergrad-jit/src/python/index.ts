/**
 * Source-of-truth for the browsergrad_jit Python package as installed
 * into Pyodide's virtual filesystem.
 *
 * Each Python module is held as a TypeScript string constant — the actual
 * .py text lives in `src/python/*.py` and is base64-bundled into a
 * `*.generated.ts` file by `scripts/build-python-sources.mjs`. That
 * indirection keeps the .py files diffable and editable while letting
 * `tsc` ship a self-contained bundle.
 *
 * Re-exported at the `./source` subpath so external tooling (docs sites,
 * custom Pyodide bootstrap, alternative installers) can pull the source
 * without going through `installJit`.
 */

import { IR_PY } from "./_ir.generated.js";
import { ERRORS_PY } from "./_errors.generated.js";
import { BUFFER_TABLE_PY } from "./_buffer_table.generated.js";
import { TENSOR_PROXY_PY } from "./_tensor_proxy.generated.js";
import pkg from "../../package.json";

/**
 * `browsergrad_jit/__init__.py` content, built inline so we can interpolate
 * the npm package version into `__version__`. Single-source-of-truth: the
 * version that ships on npm and the version Python reports are the same.
 *
 * The init re-exports the public API and creates the implicit default
 * Session. Internal modules (`_ir`, `_buffer_table`, `_tensor_proxy`) are
 * NOT re-exported — users who reach for them are doing something we have
 * not committed to support.
 */
const INIT_PY = `"""browsergrad_jit — a PyTorch-shaped lazy tensor library backed by a UOp IR.

Public surface (semver-stable across the 0.x line):

  TensorProxy            — the lazy tensor; same API shape as PyTorch's Tensor
  Session / new_session  — per-tab/loop isolation primitive

Public error types (catch these):

  JitError, ShapeError, TorchAliasConflict, NoBackwardError,
  JitNotImplementedError, RealizationError, BufferTableError

Anything else (underscore-prefixed modules, the IR opcode strings, the
UOp class) is internal and may change between minor releases.

See \`docs/prd/PRD-005-jit-foundation.md\` for the design rationale.
"""

from ._tensor_proxy import TensorProxy
from ._buffer_table import BufferTable
from ._errors import (
    JitError,
    ShapeError,
    TorchAliasConflict,
    NoBackwardError,
    JitNotImplementedError,
    RealizationError,
    BufferTableError,
)


class Session:
    """Per-tab/loop isolation boundary.

    A Session owns one BufferTable and is the unit of "this Pyodide context's
    computation state." Two Sessions in the same Pyodide worker can co-exist
    without buffer-id collisions because every BufferTable mints a private
    session token.
    """

    __slots__ = ("_buffer_table",)

    def __init__(self) -> None:
        self._buffer_table = BufferTable()

    @property
    def buffer_table(self) -> BufferTable:
        return self._buffer_table

    def __repr__(self) -> str:
        return f"Session(session_token={self._buffer_table.session_token!r})"


_DEFAULT_SESSION: Session = Session()


def get_default_session() -> Session:
    """Return the implicit session used by factory functions."""
    return _DEFAULT_SESSION


def set_default_session(session: Session) -> None:
    """Swap the default session. Useful in tests and for lab harnesses
    that want to isolate runs from each other without re-importing the
    module."""
    global _DEFAULT_SESSION
    _DEFAULT_SESSION = session


def new_session() -> Session:
    """Construct a fresh isolated session."""
    return Session()


__version__ = "${pkg.version}"

__all__ = [
    "TensorProxy",
    "Session",
    "JitError",
    "ShapeError",
    "TorchAliasConflict",
    "NoBackwardError",
    "JitNotImplementedError",
    "RealizationError",
    "BufferTableError",
    "get_default_session",
    "set_default_session",
    "new_session",
    "__version__",
]
`;

export interface PythonSource {
  readonly path: string;
  readonly content: string;
}

/**
 * Ordered list of (path, content) pairs to write into the Pyodide virtual
 * FS. Order doesn't matter for the writes themselves, but listing
 * `__init__.py` last mirrors how the import will resolve and makes the
 * file list easy to scan.
 *
 * Underscore-prefixed modules are dependencies of the public surface
 * declared in `__init__.py`. They MUST land on disk before the import
 * statement in `__init__.py` runs.
 */
export const SOURCE_FILES: readonly PythonSource[] = [
  { path: "browsergrad_jit/_ir.py", content: IR_PY },
  { path: "browsergrad_jit/_errors.py", content: ERRORS_PY },
  { path: "browsergrad_jit/_buffer_table.py", content: BUFFER_TABLE_PY },
  { path: "browsergrad_jit/_tensor_proxy.py", content: TENSOR_PROXY_PY },
  { path: "browsergrad_jit/__init__.py", content: INIT_PY },
];

/**
 * Mount root for the package's Python sources. Adding this to sys.path
 * makes `import browsergrad_jit` resolve to our files. Chosen distinct
 * from `browsergrad_grad`'s mount root so both packages can coexist in
 * the same Pyodide worker without colliding on sys.path entries.
 */
export const MOUNT_ROOT = "/lib/browsergrad_jit_src";
