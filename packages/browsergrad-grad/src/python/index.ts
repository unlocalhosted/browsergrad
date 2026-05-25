/**
 * The full `browsergrad_grad` Python package source, assembled in install order.
 *
 * Each module is written into the Pyodide virtual FS, then we run
 * `import browsergrad_grad` to register the package and expose its submodules.
 *
 * Re-exported at the `./source` subpath so external tooling (e.g. a docs site
 * that renders the Python verbatim, or a custom Pyodide bootstrap that wants
 * to install grad without our installer) can access it.
 */

import { TENSOR_PY } from "./tensor";
import { FUNCTIONAL_PY } from "./functional";
import { NN_PY } from "./nn";
import { OPTIM_PY } from "./optim";

const INIT_PY = `
"""browsergrad_grad — a small, readable tensor + autograd library.

NOT pytorch. NOT a polyfill. A pedagogical artifact you can read end to end.
"""

from .tensor import (
    Tensor,
    zeros,
    ones,
    randn,
)
from . import functional
from . import nn
from . import optim

__all__ = ["Tensor", "zeros", "ones", "randn", "functional", "nn", "optim"]
__version__ = "0.3.2"
`;

export interface PythonSource {
  readonly path: string;
  readonly content: string;
}

/**
 * Ordered list of (path, content) pairs to write into the Pyodide virtual FS.
 * Order doesn't matter for file writes themselves, but listing __init__ last
 * mirrors how the import will resolve.
 */
export const SOURCE_FILES: readonly PythonSource[] = [
  { path: "browsergrad_grad/tensor.py", content: TENSOR_PY },
  { path: "browsergrad_grad/functional.py", content: FUNCTIONAL_PY },
  { path: "browsergrad_grad/nn.py", content: NN_PY },
  { path: "browsergrad_grad/optim.py", content: OPTIM_PY },
  { path: "browsergrad_grad/__init__.py", content: INIT_PY },
];

/**
 * Mount root for the package's Python sources. Adding this to sys.path makes
 * `import browsergrad_grad` resolve to our files. Chosen to avoid collisions
 * with anything else a consumer might put under /home/pyodide.
 */
export const MOUNT_ROOT = "/lib/browsergrad_grad_src";
