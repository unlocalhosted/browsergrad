/**
 * `install_torch_alias()` — installs a `torch` namespace shim into
 * sys.modules so vanilla PyTorch user code runs unmodified against
 * browsergrad_grad.
 *
 * Mapping
 *   torch                       → browsergrad_grad
 *   torch.Tensor / torch.tensor → browsergrad_grad.Tensor
 *   torch.zeros / ones / randn  → browsergrad_grad equivalents
 *   torch.cat / stack           → browsergrad_grad equivalents
 *   torch.no_grad               → browsergrad_grad.no_grad
 *   torch.nn                    → browsergrad_grad.nn (copied into a new
 *                                 module so we can override .functional)
 *   torch.nn.functional         → browsergrad_grad.functional + PyTorch-name
 *                                 aliases (cross_entropy → cross_entropy_loss)
 *   torch.optim                 → browsergrad_grad.optim
 *
 * Limitations
 *   Only the subset of torch's API that browsergrad_grad implements is
 *   available. Anything else raises AttributeError. Notably absent:
 *   torch.cuda.*, torch.compile, torch.fx, torch.jit, dtype objects
 *   beyond the f32 we use internally.
 */
export const TORCH_COMPAT_PY = `
"""browsergrad_grad.torch_compat — install a torch-namespace shim."""

import sys as _bg_torch_sys
import types as _bg_torch_types


def install_torch_alias():
    """Register torch / torch.nn / torch.nn.functional / torch.optim in
    sys.modules so PyTorch user code runs against browsergrad_grad.

    Calling repeatedly is safe — the alias is re-installed each time, so
    if a user has reset sys.modules between calls it'll rebuild.
    """
    import browsergrad_grad as _bg

    torch_mod = _bg_torch_types.ModuleType("torch")
    torch_mod.__doc__ = "browsergrad_grad shim under the torch namespace"

    # Core tensor + constructors
    torch_mod.Tensor = _bg.Tensor
    torch_mod.tensor = _bg.Tensor          # torch.tensor(...) is a function
    torch_mod.zeros = _bg.zeros
    torch_mod.ones = _bg.ones
    torch_mod.randn = _bg.randn
    torch_mod.cat = _bg.cat
    torch_mod.stack = _bg.stack
    torch_mod.no_grad = _bg.no_grad

    # Numpy interop + reproducibility
    torch_mod.from_numpy = _bg.from_numpy
    torch_mod.manual_seed = _bg.manual_seed

    # Top-level math functions (PyTorch-style alternatives to methods/ops)
    torch_mod.matmul = _bg.matmul
    torch_mod.mm = _bg.mm
    torch_mod.bmm = _bg.bmm
    torch_mod.exp = _bg.exp
    torch_mod.log = _bg.log
    torch_mod.sum = _bg.sum
    torch_mod.mean = _bg.mean
    torch_mod.argmax = _bg.argmax
    torch_mod.einsum = _bg.einsum

    # Serialization
    torch_mod.save = _bg.save
    torch_mod.load = _bg.load

    # dtype tokens — strings are enough for the dtype= kwarg path we support
    torch_mod.float32 = "float32"
    torch_mod.float = "float32"
    torch_mod.int64 = "int64"
    torch_mod.long = "int64"

    # torch.nn — copy attributes into a fresh module so overriding
    # .functional below doesn't mutate browsergrad_grad.nn
    torch_nn = _bg_torch_types.ModuleType("torch.nn")
    for _name in dir(_bg.nn):
        if not _name.startswith("_"):
            setattr(torch_nn, _name, getattr(_bg.nn, _name))

    # torch.nn.functional — copy + PyTorch-name aliases for funcs we
    # name slightly differently internally.
    torch_F = _bg_torch_types.ModuleType("torch.nn.functional")
    for _name in dir(_bg.functional):
        if not _name.startswith("_"):
            setattr(torch_F, _name, getattr(_bg.functional, _name))
    # PyTorch name → browsergrad_grad name
    torch_F.cross_entropy = _bg.functional.cross_entropy_loss
    torch_F.nll = _bg.functional.nll_loss

    torch_nn.functional = torch_F
    torch_mod.nn = torch_nn

    # torch.optim — re-exposed directly (no name differences in v0)
    torch_mod.optim = _bg.optim

    # torch.utils.data — pass through to browsergrad_grad.utils.data
    torch_utils = _bg_torch_types.ModuleType("torch.utils")
    torch_utils.data = _bg.utils.data
    torch_mod.utils = torch_utils

    # Register
    _bg_torch_sys.modules["torch"] = torch_mod
    _bg_torch_sys.modules["torch.nn"] = torch_nn
    _bg_torch_sys.modules["torch.nn.functional"] = torch_F
    _bg_torch_sys.modules["torch.optim"] = _bg.optim
    _bg_torch_sys.modules["torch.utils"] = torch_utils
    _bg_torch_sys.modules["torch.utils.data"] = _bg.utils.data

    return torch_mod
`;
