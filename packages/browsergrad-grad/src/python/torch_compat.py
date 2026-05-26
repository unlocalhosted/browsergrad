
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

    # ─── Pile B — possible but limited ─────────────────────────────
    # torch.amp.autocast — no-op context manager. We have no real fp16 path
    # in WASM/Pyodide; calling autocast() shouldn't break code that uses it.
    import contextlib as _bg_ctxlib
    import numpy as _bg_np_linalg
    torch_amp = _bg_torch_types.ModuleType("torch.amp")
    @_bg_ctxlib.contextmanager
    def _autocast(*args, **kwargs):
        yield
    torch_amp.autocast = _autocast
    torch_mod.amp = torch_amp

    # torch.linalg — wrap numpy.linalg, return Tensors.
    torch_linalg = _bg_torch_types.ModuleType("torch.linalg")
    def _linalg_norm(t, ord=None, dim=None, keepdim=False):
        arr = _bg_np_linalg.asarray(t)
        out = _bg_np_linalg.linalg.norm(arr, ord=ord, axis=dim, keepdims=keepdim)
        return _bg.Tensor(out.astype(_bg_np_linalg.float32))
    def _linalg_inv(t):
        return _bg.Tensor(_bg_np_linalg.linalg.inv(_bg_np_linalg.asarray(t)).astype(_bg_np_linalg.float32))
    def _linalg_det(t):
        return _bg.Tensor(_bg_np_linalg.linalg.det(_bg_np_linalg.asarray(t)).astype(_bg_np_linalg.float32))
    def _linalg_svd(t):
        u, s, vh = _bg_np_linalg.linalg.svd(_bg_np_linalg.asarray(t))
        return _bg.Tensor(u.astype(_bg_np_linalg.float32)), _bg.Tensor(s.astype(_bg_np_linalg.float32)), _bg.Tensor(vh.astype(_bg_np_linalg.float32))
    def _linalg_eigh(t):
        w, v = _bg_np_linalg.linalg.eigh(_bg_np_linalg.asarray(t))
        return _bg.Tensor(w.astype(_bg_np_linalg.float32)), _bg.Tensor(v.astype(_bg_np_linalg.float32))
    def _linalg_solve(a, b):
        out = _bg_np_linalg.linalg.solve(_bg_np_linalg.asarray(a), _bg_np_linalg.asarray(b))
        return _bg.Tensor(out.astype(_bg_np_linalg.float32))
    def _linalg_pinv(t):
        return _bg.Tensor(_bg_np_linalg.linalg.pinv(_bg_np_linalg.asarray(t)).astype(_bg_np_linalg.float32))
    torch_linalg.norm = _linalg_norm
    torch_linalg.inv = _linalg_inv
    torch_linalg.det = _linalg_det
    torch_linalg.svd = _linalg_svd
    torch_linalg.eigh = _linalg_eigh
    torch_linalg.solve = _linalg_solve
    torch_linalg.pinv = _linalg_pinv
    torch_mod.linalg = torch_linalg

    # Multi-GPU shim: nn.Module.to(device) accepts the call but no-ops.
    # The existing implementation already returns self, so we're good — but
    # be explicit in the alias for users who reach for module.to('cuda:0').
    def _module_to_shim(self, *args, **kwargs):
        return self
    _bg.nn.Module.to = _module_to_shim

    # ─── Pile C — physically impossible in browser ─────────────────
    # Loud, descriptive NotImplementedError. NEVER silent success (greed
    # made that mistake and it eats real labs alive).
    def _impossible(name, reason):
        def _raise(*args, **kwargs):
            raise NotImplementedError(f"{name}: {reason}")
        return _raise

    torch_mod.compile = _impossible(
        "torch.compile",
        "requires a compiler runtime not available in browser. Run your model uncompiled.",
    )

    torch_fx = _bg_torch_types.ModuleType("torch.fx")
    torch_fx.symbolic_trace = _impossible(
        "torch.fx.symbolic_trace",
        "FX tracing relies on Python introspection paths we don't model. Use the eager graph.",
    )
    torch_mod.fx = torch_fx

    torch_jit = _bg_torch_types.ModuleType("torch.jit")
    torch_jit.script = _impossible(
        "torch.jit.script",
        "no script compiler in browser. The eager Python path runs the same way.",
    )
    torch_jit.trace = _impossible(
        "torch.jit.trace",
        "no trace compiler in browser. The eager Python path runs the same way.",
    )
    torch_mod.jit = torch_jit

    torch_cuda = _bg_torch_types.ModuleType("torch.cuda")
    torch_cuda.is_available = lambda: False
    torch_cuda.device_count = lambda: 0
    torch_cuda.current_device = _impossible(
        "torch.cuda.current_device",
        "no CUDA runtime in browser; use WebGPU via @unlocalhosted/browsergrad-kernels.",
    )
    torch_mod.cuda = torch_cuda

    torch_distributed = _bg_torch_types.ModuleType("torch.distributed")
    torch_distributed.init_process_group = _impossible(
        "torch.distributed.init_process_group",
        "no multi-machine runtime in browser.",
    )
    torch_distributed.all_reduce = _impossible(
        "torch.distributed.all_reduce",
        "no multi-machine runtime in browser.",
    )
    torch_distributed.is_initialized = lambda: False
    torch_mod.distributed = torch_distributed

    torch_onnx = _bg_torch_types.ModuleType("torch.onnx")
    torch_onnx.export = _impossible(
        "torch.onnx.export",
        "ONNX exporter requires the C++ backend we don't ship.",
    )
    torch_mod.onnx = torch_onnx

    torch_quant = _bg_torch_types.ModuleType("torch.quantization")
    torch_quant.quantize = _impossible(
        "torch.quantization.quantize",
        "quantization toolchain requires backend kernels we don't ship in WASM.",
    )
    torch_mod.quantization = torch_quant

    # Register
    _bg_torch_sys.modules["torch"] = torch_mod
    _bg_torch_sys.modules["torch.nn"] = torch_nn
    _bg_torch_sys.modules["torch.nn.functional"] = torch_F
    _bg_torch_sys.modules["torch.optim"] = _bg.optim
    _bg_torch_sys.modules["torch.utils"] = torch_utils
    _bg_torch_sys.modules["torch.utils.data"] = _bg.utils.data
    _bg_torch_sys.modules["torch.amp"] = torch_amp
    _bg_torch_sys.modules["torch.linalg"] = torch_linalg
    _bg_torch_sys.modules["torch.fx"] = torch_fx
    _bg_torch_sys.modules["torch.jit"] = torch_jit
    _bg_torch_sys.modules["torch.cuda"] = torch_cuda
    _bg_torch_sys.modules["torch.distributed"] = torch_distributed
    _bg_torch_sys.modules["torch.onnx"] = torch_onnx
    _bg_torch_sys.modules["torch.quantization"] = torch_quant

    return torch_mod
