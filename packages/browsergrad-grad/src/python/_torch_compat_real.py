"""Pile A — real op aliases under the torch namespace.

Maps everything browsergrad_grad actually implements (Tensor, ops, nn modules,
optim, utils.data, serialization) onto torch's namespace. See ARCHITECTURE.md
#2 and PROGRESS.md for the full Pile A taxonomy.

`install_real` attaches torch_mod.nn / torch_mod.nn.functional / torch_mod.optim /
torch_mod.utils so the orchestrator can register their sys.modules entries.
"""

from .tensor import _resolve_dtype


def install_real(torch_mod, _bg, _types):
    import numpy as _np

    def _tensor_factory_source(data):
        if isinstance(data, _bg.Tensor):
            return data.data, True
        if isinstance(data, (_np.ndarray, _np.generic)):
            return _np.asarray(data), True
        return _np.asarray(data), False

    def _tensor_factory(data, dtype=None, requires_grad=False, device=None):
        """Construct an owning leaf with bounded PyTorch-shaped dtype inference."""
        if type(requires_grad) is not bool:
            raise TypeError(
                "torch.tensor requires_grad must be bool; "
                f"got {type(requires_grad).__name__}"
            )
        if device is not None:
            if not isinstance(device, str):
                raise TypeError(
                    "torch.tensor device must be the string 'cpu'; "
                    f"got {type(device).__name__}"
                )
            if device != "cpu":
                raise NotImplementedError(
                    f"torch.tensor(device={device!r}) is unavailable: eager Grad "
                    "storage is CPU/Pyodide-backed and no device allocation occurred"
                )
        target_dtype = _resolve_dtype(dtype) if dtype is not None else None
        source, preserve_source_dtype = _tensor_factory_source(data)
        if source.dtype.kind not in "biuf":
            raise ValueError(
                f"torch.tensor does not support input dtype {source.dtype.name!r}; "
                "complex, object, string, structured, and datetime inputs are unavailable"
            )
        if target_dtype is None:
            if preserve_source_dtype:
                target_dtype = _resolve_dtype(source.dtype)
            elif _np.issubdtype(source.dtype, _np.bool_):
                target_dtype = _np.bool_
            elif _np.issubdtype(source.dtype, _np.integer):
                target_dtype = _np.int64
            else:
                target_dtype = _np.float32
        target_name = _np.dtype(target_dtype).name
        if requires_grad and target_name not in ("float16", "float32", "float64"):
            raise ValueError(
                "torch.tensor requires_grad=True requires float16, float32, "
                f"or float64 storage; got {target_name!r}"
            )
        copied = _np.array(source, dtype=target_dtype, order="K", copy=True)
        return _bg.Tensor(
            copied,
            dtype=target_dtype,
            requires_grad=requires_grad,
        )

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
    torch_mod.topk = _bg.topk
    torch_mod.scatter = _bg.scatter
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
    # Distinct unsupported token: construction/conversion rejects before
    # silently substituting float32 storage for bfloat16 semantics.
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.int64   = "int64"
    torch_mod.long    = "int64"
    torch_mod.int32   = "int32"
    torch_mod.int     = "int32"
    torch_mod.uint8   = "uint8"
    torch_mod.uint16  = "uint16"
    torch_mod.uint32  = "uint32"
    torch_mod.uint64  = "uint64"
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
