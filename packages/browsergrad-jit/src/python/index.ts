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
import { REALIZE_PY } from "./_realize.generated.js";
import { FUSION_PY } from "./_fusion.generated.js";
import { FUSION_CONFIG_PY } from "./_fusion_config.generated.js";
import { VJP_PY } from "./_vjp.generated.js";
import { TRACE_CACHE_PY } from "./_trace_cache.generated.js";
import { SAFETENSORS_PY } from "./_safetensors.generated.js";
import { TENSOR_PROXY_PY } from "./_tensor_proxy.generated.js";
import { NN_PY } from "./_nn.generated.js";
import { FUNCTIONAL_PY } from "./_functional.generated.js";
import { OPTIM_PY } from "./_optim.generated.js";
import { TORCH_COMPAT_PY } from "./_torch_compat.generated.js";
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

  Tensor (alias for TensorProxy), tensor, zeros, ones, randn, arange,
  from_numpy           — factory functions
  Session, new_session  — per-tab/loop isolation primitive
  nn                    — Module, Linear, Sequential, ReLU, Dropout, ...
  nn.functional         — relu, softmax, cross_entropy, mse_loss, linear, ...
  optim                 — SGD, Adam, AdamW

Public error types (catch these):

  JitError, ShapeError, TorchAliasConflict, NoBackwardError,
  JitNotImplementedError, RealizationError, BufferTableError

Anything else (underscore-prefixed modules, the IR opcode strings, the
UOp class) is internal and may change between minor releases.

See \`docs/prd/PRD-005-jit-foundation.md\` for the design rationale.
"""

from ._tensor_proxy import (
    TensorProxy,
    from_numpy,
    tensor,
    zeros,
    ones,
    randn,
    arange,
)
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
from . import _functional, _nn, _optim
from . import _fusion as _fusion_mod
from . import _fusion_config as _fc
from . import _trace_cache as _tc
from ._safetensors import load_safetensors, save_safetensors
from ._torch_compat import install_torch_alias, uninstall_torch_alias
import sys as _sys
import types as _types

# PyTorch-style alias for the lazy tensor type.
Tensor = TensorProxy

# Wire up the public sub-namespaces. Registering in sys.modules lets users
# import via the dotted path even though there is no on-disk file at
# browsergrad_jit/nn/__init__.py — _nn IS the nn module.
_nn.functional = _functional
_sys.modules["browsergrad_jit.nn"] = _nn
_sys.modules["browsergrad_jit.nn.functional"] = _functional
_sys.modules["browsergrad_jit.optim"] = _optim
nn = _nn
optim = _optim


# bg.jit — the introspection + control surface for the JIT.
# Public methods (semver-stable across the 0.x line):
#   use_fusion(bool)          — toggle fusion globally (default True)
#   debug_fused_kernels()     — list of FusedKernelInfo from the last run
#   debug_unfused_reasons()   — list of UnfusedReason from the last run
jit = _types.ModuleType("browsergrad_jit.jit")
jit.use_fusion = _fc.use_fusion
jit.fusion_enabled = _fc.is_enabled


def _debug_fused_kernels():
    return list(_fusion_mod.get_last_report().fused)


def _debug_unfused_reasons():
    return list(_fusion_mod.get_last_report().unfused)


jit.debug_fused_kernels = _debug_fused_kernels
jit.debug_unfused_reasons = _debug_unfused_reasons
jit.use_trace_cache = _tc.use_trace_cache
jit.trace_cache_enabled = _tc.is_enabled
jit.trace_cache_stats = _tc.stats
jit.clear_trace_cache = _tc.clear
_sys.modules["browsergrad_jit.jit"] = jit


def cache_stats() -> dict:
    """Aggregate cache observability across the trace cache and (future)
    OPFS blob cache. Adding new cache categories should extend this
    function's return shape additively to preserve backwards compat."""
    return {
        "trace": _tc.stats(),
    }


def clear_cache(scope: str = "all") -> None:
    """Wipe one or more cache categories.

    scope:
      * 'all' — every cache category in scope.
      * 'trace' — the in-memory IR-trace cache only.
    Future scopes ('opfs', 'pipelines') ship with PRD-008.2 / PRD-012.
    """
    if scope in ("all", "trace"):
        _tc.clear()
    if scope not in ("all", "trace"):
        raise ValueError(
            f"clear_cache: unknown scope {scope!r}; expected one of 'all', 'trace'"
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


def manual_seed(seed: int) -> None:
    """Seed NumPy's global RNG. Mirrors torch.manual_seed.

    The JIT's RANDOM opcode uses its own per-call seed_key when wired,
    but factory functions like randn() and the dropout op fall back to
    NumPy's default RNG, which this seeds."""
    import numpy as _np
    _np.random.seed(seed)


__version__ = "${pkg.version}"

__all__ = [
    "TensorProxy", "Tensor",
    "tensor", "zeros", "ones", "randn", "arange", "from_numpy",
    "Session", "get_default_session", "set_default_session", "new_session",
    "manual_seed",
    "nn", "optim", "jit",
    "install_torch_alias", "uninstall_torch_alias",
    "load_safetensors", "save_safetensors",
    "cache_stats", "clear_cache",
    "JitError", "ShapeError", "TorchAliasConflict",
    "NoBackwardError", "JitNotImplementedError",
    "RealizationError", "BufferTableError",
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
  { path: "browsergrad_jit/_errors.py", content: ERRORS_PY },
  { path: "browsergrad_jit/_ir.py", content: IR_PY },
  { path: "browsergrad_jit/_buffer_table.py", content: BUFFER_TABLE_PY },
  { path: "browsergrad_jit/_fusion_config.py", content: FUSION_CONFIG_PY },
  { path: "browsergrad_jit/_realize.py", content: REALIZE_PY },
  { path: "browsergrad_jit/_fusion.py", content: FUSION_PY },
  { path: "browsergrad_jit/_vjp.py", content: VJP_PY },
  { path: "browsergrad_jit/_trace_cache.py", content: TRACE_CACHE_PY },
  { path: "browsergrad_jit/_safetensors.py", content: SAFETENSORS_PY },
  { path: "browsergrad_jit/_tensor_proxy.py", content: TENSOR_PROXY_PY },
  { path: "browsergrad_jit/_functional.py", content: FUNCTIONAL_PY },
  { path: "browsergrad_jit/_nn.py", content: NN_PY },
  { path: "browsergrad_jit/_optim.py", content: OPTIM_PY },
  { path: "browsergrad_jit/_torch_compat.py", content: TORCH_COMPAT_PY },
  { path: "browsergrad_jit/__init__.py", content: INIT_PY },
];

/**
 * Mount root for the package's Python sources. Adding this to sys.path
 * makes `import browsergrad_jit` resolve to our files. Chosen distinct
 * from `browsergrad_grad`'s mount root so both packages can coexist in
 * the same Pyodide worker without colliding on sys.path entries.
 */
export const MOUNT_ROOT = "/lib/browsergrad_jit_src";
