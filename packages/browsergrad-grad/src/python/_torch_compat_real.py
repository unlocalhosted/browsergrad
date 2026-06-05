"""Pile A — real op aliases under the torch namespace.

Maps everything browsergrad_grad actually implements (Tensor, ops, nn modules,
optim, utils.data, serialization) onto torch's namespace. See ARCHITECTURE.md
#2 and PROGRESS.md for the full Pile A taxonomy.

`install_real` attaches torch_mod.nn / torch_mod.nn.functional / torch_mod.optim /
torch_mod.utils so the orchestrator can register their sys.modules entries.
"""


def install_real(torch_mod, _bg, _types):
    import numpy as _np

    def _tensor_factory(data, dtype=None, requires_grad=False, device=None):
        """torch.tensor(): infer int64 for integer data, float32 for floats."""
        if dtype is not None:
            return _bg.Tensor(data, dtype=dtype, requires_grad=requires_grad)
        arr = _np.asarray(data)
        if _np.issubdtype(arr.dtype, _np.integer):
            return _bg.Tensor(arr, dtype="int64", requires_grad=requires_grad)
        return _bg.Tensor(arr.astype(_np.float32), requires_grad=requires_grad)

    # Core tensor + constructors
    torch_mod.Tensor = _bg.Tensor
    torch_mod.tensor = _tensor_factory
    torch_mod.zeros = _bg.zeros
    torch_mod.ones = _bg.ones
    torch_mod.randn = _bg.randn
    torch_mod.arange = _bg.arange
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
    torch_mod.where = _bg.where
    torch_mod.triu = _bg.triu
    torch_mod.tril = _bg.tril
    torch_mod.multinomial = _bg.multinomial
    torch_mod.topk = lambda input, k, dim=-1, largest=True: input.topk(k, dim=dim, largest=largest)
    torch_mod.tanh    = _bg.tanh
    torch_mod.sqrt    = _bg.sqrt
    torch_mod.pow     = _bg.pow
    torch_mod.rsqrt   = _bg.rsqrt
    torch_mod.cos     = _bg.cos
    torch_mod.sin     = _bg.sin
    torch_mod.cumsum  = _bg.cumsum
    torch_mod.sort    = _bg.sort
    torch_mod.minimum = _bg.minimum
    torch_mod.zeros_like = _bg.zeros_like
    torch_mod.ones_like  = _bg.ones_like
    torch_mod.std    = _bg.std
    torch_mod.prod   = _bg.prod
    torch_mod.gather = _bg.gather
    torch_mod.repeat_interleave = _bg.repeat_interleave
    torch_mod.softmax     = lambda input, dim=-1: _bg.functional.softmax(input, dim=dim)
    torch_mod.log_softmax = lambda input, dim=-1: _bg.functional.log_softmax(input, dim=dim)
    torch_mod.argmax      = _bg.argmax
    torch_mod.inference_mode = _bg.no_grad  # alias: same semantics as no_grad
    torch_mod.all = lambda input, dim=None, keepdim=False: _bg.Tensor(
        _np.all(input.data if isinstance(input, _bg.Tensor) else _np.asarray(input),
                axis=dim, keepdims=keepdim), dtype="bool"
    )

    # Serialization
    torch_mod.save = _bg.save
    torch_mod.load = _bg.load

    # dtype tokens — strings that map to our dtype aliases
    torch_mod.float32 = "float32"
    torch_mod.float  = "float32"
    torch_mod.float64 = "float64"
    torch_mod.double  = "float64"
    torch_mod.float16 = "float16"
    torch_mod.half    = "float16"
    torch_mod.bfloat16 = "float32"  # bf16 not supported in NumPy → fall back to f32
    torch_mod.int64   = "int64"
    torch_mod.long    = "int64"
    torch_mod.int32   = "int32"
    torch_mod.int     = "int32"
    torch_mod.bool    = "bool"

    # Math constants
    import math as _math
    torch_mod.pi  = _math.pi
    torch_mod.inf = float("inf")

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
    torch_F.silu = _bg.functional.silu
    torch_F.log_softmax = _bg.functional.log_softmax

    # torch.nn.utils
    torch_nn_utils = _types.ModuleType("torch.nn.utils")
    torch_nn_utils.clip_grad_norm_ = _bg.nn.clip_grad_norm_
    torch_nn.utils = torch_nn_utils

    torch_nn.functional = torch_F
    torch_mod.nn = torch_nn

    # torch.optim — re-exposed directly (no name differences in v0)
    torch_mod.optim = _bg.optim

    # torch.utils.data — pass through to browsergrad_grad.utils.data
    torch_utils = _types.ModuleType("torch.utils")
    torch_utils.data = _bg.utils.data
    torch_mod.utils = torch_utils
