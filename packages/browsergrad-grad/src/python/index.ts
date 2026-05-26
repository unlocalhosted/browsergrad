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

import { TENSOR_PY } from "./tensor.js";
import { FUNCTIONAL_PY } from "./functional.js";
import { NN_PY } from "./nn.js";
import { OPTIM_PY } from "./optim.js";
import { TORCH_COMPAT_PY } from "./torch_compat.js";
import { UTILS_DATA_PY } from "./utils_data.js";

const INIT_PY = `
"""browsergrad_grad — a small, readable tensor + autograd library.

PyTorch-flavored API, NumPy-backed, deliberately not PyTorch. For code
that uses 'import torch', call \`grad.install_torch_alias()\` once to
register a sys.modules shim that maps the torch namespace onto
browsergrad_grad.
"""

from .tensor import (
    Tensor,
    zeros,
    ones,
    randn,
    no_grad,
    cat,
    stack,
    where,
    einsum,
    from_numpy,
    manual_seed,
    matmul,
    mm,
    bmm,
    exp,
    log,
    sum,
    mean,
    argmax,
)
from . import functional
from . import nn
from . import optim
from . import utils  # nested namespace: browsergrad_grad.utils.data
from .torch_compat import install_torch_alias

import pickle as _bg_pickle


def save(obj, path):
    """Pickle obj to path. Pairs with load() and torch.save() in the shim.

    The intended use is checkpointing — pass a Module.state_dict() (which is
    a dict[str, np.ndarray]) and load it back via load() + load_state_dict().
    """
    with open(path, "wb") as f:
        _bg_pickle.dump(obj, f)


def load(path, **kwargs):
    """Unpickle from path. Ignores PyTorch-specific kwargs like map_location
    and weights_only — we're in-browser, none of them apply."""
    with open(path, "rb") as f:
        return _bg_pickle.load(f)


__all__ = [
    "Tensor", "zeros", "ones", "randn", "no_grad", "cat", "stack", "where",
    "einsum", "from_numpy", "manual_seed",
    "matmul", "mm", "bmm", "exp", "log", "sum", "mean", "argmax",
    "functional", "nn", "optim", "utils",
    "save", "load",
    "install_torch_alias",
]
__version__ = "0.4.14"
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
  { path: "browsergrad_grad/torch_compat.py", content: TORCH_COMPAT_PY },
  { path: "browsergrad_grad/utils/__init__.py", content: 'from . import data\n__all__ = ["data"]\n' },
  { path: "browsergrad_grad/utils/data.py", content: UTILS_DATA_PY },
  { path: "browsergrad_grad/__init__.py", content: INIT_PY },
];

/**
 * Mount root for the package's Python sources. Adding this to sys.path makes
 * `import browsergrad_grad` resolve to our files. Chosen to avoid collisions
 * with anything else a consumer might put under /home/pyodide.
 */
export const MOUNT_ROOT = "/lib/browsergrad_grad_src";
