"""Pile A — real op aliases under the torch namespace.

Maps everything browsergrad_grad actually implements (Tensor, ops, nn modules,
optim, utils.data, serialization) onto torch's namespace. See ARCHITECTURE.md
#2 and PROGRESS.md for the full Pile A taxonomy.

`install_real` attaches torch_mod.nn / torch_mod.nn.functional / torch_mod.optim /
torch_mod.utils so the orchestrator can register their sys.modules entries.
"""


def install_real(torch_mod, _bg, _types):
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

    # torch.nn — shallow-copy attributes into a fresh module so overriding
    # .functional below doesn't mutate browsergrad_grad.nn. CRITICAL: this is
    # a shallow copy of class references, so torch_nn.Module IS _bg.nn.Module.
    # The orchestrator asserts that invariant immediately after we return —
    # don't switch to a deep copy without removing the Pile B monkey-patch on
    # _bg.nn.Module.to.
    torch_nn = _types.ModuleType("torch.nn")
    for _name in dir(_bg.nn):
        if not _name.startswith("_"):
            setattr(torch_nn, _name, getattr(_bg.nn, _name))

    # torch.nn.functional — copy + PyTorch-name aliases for funcs we
    # name slightly differently internally.
    torch_F = _types.ModuleType("torch.nn.functional")
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
    torch_utils = _types.ModuleType("torch.utils")
    torch_utils.data = _bg.utils.data
    torch_mod.utils = torch_utils
