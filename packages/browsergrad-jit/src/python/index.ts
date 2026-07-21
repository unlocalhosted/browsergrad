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

import { FRAMEWORK_OPERATION_CONTRACTS_JSON } from "./framework-operation-contracts.v1.generated.js";
import { FRAMEWORK_CONTRACTS_PY } from "./_framework_contracts.generated.js";
import { IR_PY } from "./_ir.generated.js";
import { ERRORS_PY } from "./_errors.generated.js";
import { BUFFER_TABLE_PY } from "./_buffer_table.generated.js";
import { REALIZE_PY } from "./_realize.generated.js";
import { FUSION_PY } from "./_fusion.generated.js";
import { FUSION_CONFIG_PY } from "./_fusion_config.generated.js";
import { VJP_PY } from "./_vjp.generated.js";
import { TRACE_CACHE_PY } from "./_trace_cache.generated.js";
import { SAFETENSORS_PY } from "./_safetensors.generated.js";
import { CHECKPOINT_PY } from "./_checkpoint.generated.js";
import { UTILS_CHECKPOINT_PY } from "./_utils_checkpoint.generated.js";
import { UTILS_DATA_PY } from "./_utils_data.generated.js";
import { AMP_PY } from "./_amp.generated.js";
import { BRIDGE_PY } from "./_bridge.generated.js";
import { GPU_BUFFER_TABLE_PY } from "./_gpu_buffer_table.generated.js";
import { REALIZE_WEBGPU_PY } from "./_realize_webgpu.generated.js";
import { GPU_PLAN_PY } from "./_gpu_plan.generated.js";
import { FUNC_PY } from "./_func.generated.js";
import { VMAP_PY } from "./_vmap.generated.js";
import { WEBNN_PY } from "./_webnn.generated.js";
import { COST_MODEL_PY } from "./_cost_model.generated.js";
import { CUSTOM_KERNEL_PY } from "./_custom_kernel.generated.js";
import { ONNX_PY } from "./_onnx.generated.js";
import { LAB_PY } from "./_lab.generated.js";
import { TENSOR_PROXY_PY } from "./_tensor_proxy.generated.js";
import { NN_PY } from "./_nn.generated.js";
import { FUNCTIONAL_PY } from "./_functional.generated.js";
import { OPTIM_PY } from "./_optim.generated.js";
import { TORCH_COMPAT_PY } from "./_torch_compat.generated.js";
import pkg from "../../package.json" with { type: "json" };

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
    no_grad,
    inference_mode,
    from_numpy,
    tensor,
    zeros,
    ones,
    randn,
    arange,
    zeros_like,
    ones_like,
    cat,
    stack,
    where,
    minimum as _minimum,
    cumsum as _cumsum,
    sort as _sort,
    triu as _triu,
    tril as _tril,
    multinomial as _multinomial,
    einsum as _einsum,
    _from_buffer_id,
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
from . import _framework_contracts as _framework_contracts_mod
from . import _fusion as _fusion_mod
from . import _fusion_config as _fc
from . import _trace_cache as _tc
from ._safetensors import load_safetensors, save_safetensors
from . import _utils_checkpoint as _utils_ckpt
from . import _utils_data as _utils_data_mod
from . import _amp as _amp_mod
from . import _realize_webgpu as _webgpu_mod
from . import _gpu_plan as _gpu_plan_mod
from . import _func as _func_mod
from . import _custom_kernel as _custom_kernel_mod
from . import _onnx as _onnx_mod
from . import _lab as _lab_mod
from . import _webnn as _webnn_mod_exp
from . import _cost_model as _cost_model_mod
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
functional = _functional
F = _functional


def framework_operation_support():
    """Return executable typed-operation decisions used by every JIT boundary."""
    return _framework_contracts_mod.framework_operation_support()


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

# bg.utils.checkpoint matches torch.utils.checkpoint's shape:
#   from browsergrad_jit.utils.checkpoint import checkpoint
utils = _types.ModuleType("browsergrad_jit.utils")
utils_ckpt_mod = _types.ModuleType("browsergrad_jit.utils.checkpoint")
utils_ckpt_mod.checkpoint = _utils_ckpt.checkpoint
utils_data_mod = _types.ModuleType("browsergrad_jit.utils.data")
utils_data_mod.Dataset = _utils_data_mod.Dataset
utils_data_mod.DataLoader = _utils_data_mod.DataLoader
utils_data_mod.TensorDataset = _utils_data_mod.TensorDataset
utils.checkpoint = utils_ckpt_mod
utils.data = utils_data_mod
_sys.modules["browsergrad_jit.utils"] = utils
_sys.modules["browsergrad_jit.utils.checkpoint"] = utils_ckpt_mod
_sys.modules["browsergrad_jit.utils.data"] = utils_data_mod

# bg.amp — mixed precision (PRD-010). Matches torch.amp shape exactly:
#   with bg.amp.autocast(device_type="webgpu", dtype=torch.float16):
#       loss = ...
#   scaler = bg.amp.GradScaler()
amp = _types.ModuleType("browsergrad_jit.amp")
amp.autocast = _amp_mod.autocast
amp.GradScaler = _amp_mod.GradScaler
amp.is_available = _amp_mod.is_available
_sys.modules["browsergrad_jit.amp"] = amp


# bg.kernels — public surface for opt-in CUSTOM kernels (PRD-011.5).
# Today only flash_attention; PRD-012a auto-recognises the same pattern.
import math as _math


def flash_attention(q, k, v, mask=None, scale=None):
    """Build a CUSTOM(flash_attention) UOp; realize via bg.realize_webgpu.

    Inputs are TensorProxy. Shapes:
        Q: (B, H, Sq, D)
        K: (B, H, Sk, D)
        V: (B, H, Sk, D)
        mask (optional): (B or 1, H or 1, Sq, Sk) — additive logits mask
                         (use -inf for blocked positions, 0 for allowed)
    Returns a TensorProxy of shape (B, H, Sq, D).

    NOTE: backward is not implemented. The result has requires_grad=False
    and no autograd context, so it silently disconnects the input chain;
    NoBackwardError is not currently raised. Use only for forward/inference
    paths in v0.
    """
    if scale is None:
        scale = 1.0 / _math.sqrt(q.shape[-1])
    B, H, Sq, D = q.shape
    _, _, Sk, _ = k.shape
    out_shape = (B, H, Sq, D)
    inputs_uops = [q._uop, k._uop, v._uop]
    has_mask = mask is not None
    if has_mask:
        inputs_uops.append(mask._uop)
    from ._ir import UOp, OP_CUSTOM
    arg = {
        "op": "flash_attention",
        "b": int(B), "h": int(H), "sq": int(Sq), "sk": int(Sk), "d": int(D),
        "scale": float(scale),
        "has_mask": bool(has_mask),
    }
    uop = UOp(op=OP_CUSTOM, inputs=tuple(inputs_uops),
              shape=out_shape, dtype=q.dtype, arg=arg)
    return TensorProxy(uop, session=q._get_session(), requires_grad=False)


kernels = _types.ModuleType("browsergrad_jit.kernels")
kernels.flash_attention = flash_attention
_sys.modules["browsergrad_jit.kernels"] = kernels


# bg.func — functional transforms (PRD-014). Mirrors torch.func.
func = _types.ModuleType("browsergrad_jit.func")
func.grad = _func_mod.grad
func.vjp = _func_mod.vjp
func.functional_call = _func_mod.functional_call
func.vmap = _func_mod.vmap         # refuses with pointer in v0
func.jacrev = _func_mod.jacrev     # refuses with pointer in v0
_sys.modules["browsergrad_jit.func"] = func


# bg.custom_kernel — user-supplied WGSL (PRD-015). Forward-only;
# its result has requires_grad=False and disconnects autograd. Realize via
# bg.realize_webgpu.
custom_kernel = _custom_kernel_mod.custom_kernel


# bg.onnx — pure-Python ONNX export (PRD-016). Inference graphs only;
# caller declares which BUFFERs are inputs vs initializers.
def export_onnx_inference(tensor, *, input_buffers=(), output_name="output",
                          model_name="browsergrad_model", opset_version=17):
    """Serialize the tensor's IR as an ONNX ModelProto. Returns bytes.

    The tensor argument is a TensorProxy. The graph rooted at its UOp is
    exported. The input_buffers argument is a sequence of TensorProxy
    whose underlying BUFFER becomes a graph input (placeholder).
    Everything else reachable becomes an initializer (frozen weight).
    """
    sess = tensor._get_session()
    return _onnx_mod.export_inference(
        tensor._uop,
        buffer_table=sess.buffer_table,
        input_buffers=input_buffers,
        output_name=output_name,
        model_name=model_name,
        opset_version=opset_version,
    )

onnx = _types.ModuleType("browsergrad_jit.onnx")
onnx.export_inference = export_onnx_inference
onnx.OnnxUnmappableOp = _onnx_mod.OnnxUnmappableOp
_sys.modules["browsergrad_jit.onnx"] = onnx


# bg.lab — semantic harness primitives (PRD-013). Calls into the
# runtime's browsergrad module if available; structured-stdout fallback
# for plain-pyodide unit tests.
lab = _types.ModuleType("browsergrad_jit.lab")
lab.assert_pytorch_match = _lab_mod.assert_pytorch_match
lab.assert_shape_match = _lab_mod.assert_shape_match
lab.assert_no_nan_inf = _lab_mod.assert_no_nan_inf
_sys.modules["browsergrad_jit.lab"] = lab


# bg.experimental — unstable surfaces behind feature flags (PRD-011 spike).
experimental = _types.ModuleType("browsergrad_jit.experimental")
experimental_webnn = _types.ModuleType("browsergrad_jit.experimental.webnn")
experimental_webnn.is_available = _webnn_mod_exp.is_available
experimental_webnn.matmul = _webnn_mod_exp.matmul
experimental.webnn = experimental_webnn
_sys.modules["browsergrad_jit.experimental"] = experimental
_sys.modules["browsergrad_jit.experimental.webnn"] = experimental_webnn


# bg.jit.cost_model — tier selector (PRD-012b).
cost_model_mod = _types.ModuleType("browsergrad_jit.jit.cost_model")
cost_model_mod.estimate_flops = _cost_model_mod.estimate_flops
cost_model_mod.estimate_bytes = _cost_model_mod.estimate_bytes
cost_model_mod.pick_tier = _cost_model_mod.pick_tier
cost_model_mod.find_producer_consumer_pairs = _cost_model_mod.find_producer_consumer_pairs
cost_model_mod.cost_stats = _cost_model_mod.cost_stats
jit.cost_model = cost_model_mod
_sys.modules["browsergrad_jit.jit.cost_model"] = cost_model_mod


# bg.jit.gpu_plan - compiler-facing tensor-IR plan scaffold.
gpu_plan_mod = _types.ModuleType("browsergrad_jit.jit.gpu_plan")
gpu_plan_mod.build_gpu_execution_plan = _gpu_plan_mod.build_gpu_execution_plan
gpu_plan_mod.gpu_plan_summary = _gpu_plan_mod.gpu_plan_summary
gpu_plan_mod.GpuPlanUnsupported = _gpu_plan_mod.GpuPlanUnsupported
gpu_plan_mod.PRIMITIVE_GPU_IR_OPS = _gpu_plan_mod.PRIMITIVE_GPU_IR_OPS
jit.gpu_plan = gpu_plan_mod
_sys.modules["browsergrad_jit.jit.gpu_plan"] = gpu_plan_mod


def gpu_plan_summary(tensor, *, allow_custom=False):
    """Return compiler-facing GPU tensor-plan summary for a TensorProxy."""
    return _gpu_plan_mod.gpu_plan_summary(
        tensor._uop,
        allow_custom=allow_custom,
    )


# bg.kernels.transformer_block — opt-in megakernel constructor (PRD-012c).
# Forward-only; user explicitly opts in via this constructor. Builds an
# OP_CUSTOM tagged "transformer_block". This is constructor-only today:
# neither current realizer dispatches it and no automatic decomposition or
# fallback exists.
def transformer_block(x, w_qkv, w_o, w_ff1, w_ff2, *, num_heads=8, eps=1e-5):
    """Single transformer block: LayerNorm -> Attention -> Residual ->
    LayerNorm -> FFN -> Residual. Forward only.

    Shapes:
        x: (B, S, D) — input activations
        w_qkv: (D, 3*D) — fused Q/K/V projection
        w_o: (D, D) — output projection
        w_ff1: (D, 4*D), w_ff2: (4*D, D) — FFN
    Returns: (B, S, D)
    """
    if x.ndim != 3:
        raise TypeError(
            f"bg.kernels.transformer_block: x must be (B, S, D); got {x.shape}"
        )
    B, S, D = x.shape
    arg = {
        "op": "transformer_block",
        "b": int(B), "s": int(S), "d": int(D),
        "num_heads": int(num_heads),
        "eps": float(eps),
    }
    from ._ir import UOp, OP_CUSTOM
    uop = UOp(op=OP_CUSTOM,
              inputs=(x._uop, w_qkv._uop, w_o._uop, w_ff1._uop, w_ff2._uop),
              shape=(B, S, D), dtype=x.dtype, arg=arg)
    return TensorProxy(uop, session=x._get_session(), requires_grad=False)

kernels.transformer_block = transformer_block


# bg.realize_webgpu — explicit-realize entry point (PRD-011.5).
# Mirrors the .numpy() trigger but routes through the WebGPU bridge
# instead of the NumPy realizer. Raises if no bridge is registered.
def realize_webgpu(tensor):
    """Realize a TensorProxy through the registered WebGPU bridge.

    Returns a NumPy ndarray (the bridge materialises bytes back at the
    seam). Raises JitNotImplementedError if no bridge is registered or
    if the IR contains opcodes the WebGPU realizer doesn't support yet.
    """
    bridge = _webgpu_mod.get_registered_bridge()
    if bridge is None:
        raise JitNotImplementedError(
            "No WebGPU bridge registered. Call "
            "bg.register_webgpu_bridge(bridge) first — the bridge is "
            "constructed JS-side via createWebGpuRealizerBridge(device) "
            "from @unlocalhosted/browsergrad-kernels."
        )
    gbt = _webgpu_mod.get_registered_gpu_buffer_table()
    sess = tensor._get_session()
    return _webgpu_mod.realize_webgpu(
        tensor._uop,
        numpy_buffer_table=sess.buffer_table,
        gpu_buffer_table=gbt,
    )


def realize_tensor_plan_webgpu(tensor):
    """Realize via the canonical tensor GPU plan bridge.

    This path sends one primitive-IR execution plan to WebGPU instead of
    dispatching through legacy per-op bridge methods. CUSTOM ops are refused;
    explicit user/lab kernels stay on their own path.
    """
    bridge = _webgpu_mod.get_registered_bridge()
    if bridge is None:
        raise JitNotImplementedError(
            "No WebGPU bridge registered. Call "
            "bg.register_webgpu_bridge(bridge) first."
        )
    gbt = _webgpu_mod.get_registered_gpu_buffer_table()
    sess = tensor._get_session()
    return _webgpu_mod.realize_tensor_plan_webgpu(
        tensor._uop,
        numpy_buffer_table=sess.buffer_table,
        gpu_buffer_table=gbt,
    )


def realize_tensor_plan_webgpu_resident(tensor):
    """Realize via tensor-plan WebGPU and return a GPU-resident TensorProxy.

    CPU bytes are populated only when the caller later crosses an explicit
    host boundary such as \`.numpy()\` or \`.item()\`.
    """
    bridge = _webgpu_mod.get_registered_bridge()
    if bridge is None:
        raise JitNotImplementedError(
            "No WebGPU bridge registered. Call "
            "bg.register_webgpu_bridge(bridge) first."
        )
    gbt = _webgpu_mod.get_registered_gpu_buffer_table()
    sess = tensor._get_session()
    buffer_id = _webgpu_mod.realize_tensor_plan_webgpu_resident(
        tensor._uop,
        numpy_buffer_table=sess.buffer_table,
        gpu_buffer_table=gbt,
    )
    return _from_buffer_id(
        buffer_id,
        tensor.shape,
        tensor.dtype,
        session=sess,
        requires_grad=tensor.requires_grad,
    )


register_webgpu_bridge = _webgpu_mod.register_webgpu_bridge
unregister_webgpu_bridge = _webgpu_mod.unregister_webgpu_bridge
webgpu_is_available = _webgpu_mod.is_available
webgpu_supported_opcodes = _webgpu_mod.supported_opcodes


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


def sigmoid(x):
    return _functional.sigmoid(x)


def matmul(a, b):
    return a @ b


def mm(a, b):
    return a @ b


def bmm(a, b):
    return a @ b


def exp(x):
    return x.exp()


def log(x):
    return x.log()


def tanh(x):
    return _functional.tanh(x)


def abs(x):
    return x.abs()


def sqrt(x):
    return x.sqrt()


def pow(input, exponent):
    return input ** exponent


def sin(x):
    return x.sin()


def cos(x):
    return x.cos()


def rsqrt(x):
    return x ** -0.5


def minimum(a, b):
    return _minimum(a, b)


def sum(x, dim=None, axis=None, keepdim=False, keepdims=False):
    return x.sum(dim=dim, axis=axis, keepdim=keepdim, keepdims=keepdims)


def mean(x, dim=None, axis=None, keepdim=False, keepdims=False):
    return x.mean(dim=dim, axis=axis, keepdim=keepdim, keepdims=keepdims)


def argmax(x, dim=None, axis=None, keepdim=False, keepdims=False):
    return x.argmax(dim=dim, axis=axis, keepdim=keepdim, keepdims=keepdims)


def prod(x, dim=None, axis=None, keepdim=False, keepdims=False):
    return x.prod(dim=dim, axis=axis, keepdim=keepdim, keepdims=keepdims)


def std(x, dim=None, axis=None, keepdim=False, keepdims=False, unbiased=True):
    return x.std(
        dim=dim,
        axis=axis,
        keepdim=keepdim,
        keepdims=keepdims,
        unbiased=unbiased,
    )


def var(x, dim=None, axis=None, keepdim=False, keepdims=False, unbiased=True):
    return x.var(
        dim=dim,
        axis=axis,
        keepdim=keepdim,
        keepdims=keepdims,
        unbiased=unbiased,
    )


def gather(x, dim, index):
    return x.gather(dim, index)


def repeat_interleave(x, repeats, dim):
    return x.repeat_interleave(repeats, dim)


def cumsum(x, dim=0):
    return _cumsum(x, dim=dim)


def sort(x, dim=-1, descending=False):
    return _sort(x, dim=dim, descending=descending)


def triu(input, diagonal=0):
    return _triu(input, diagonal=diagonal)


def tril(input, diagonal=0):
    return _tril(input, diagonal=diagonal)


def multinomial(input, num_samples, replacement=True):
    return _multinomial(input, num_samples=num_samples, replacement=replacement)


def einsum(equation, *operands):
    return _einsum(equation, *operands)


def reshape(input, *shape):
    return input.reshape(*shape)


def view(input, *shape):
    return input.view(*shape)


def flatten(input, start_dim=0, end_dim=-1):
    return input.flatten(start_dim=start_dim, end_dim=end_dim)


def squeeze(input, dim=None):
    return input.squeeze(dim=dim)


def unsqueeze(input, dim):
    return input.unsqueeze(dim)


def transpose(input, dim0, dim1):
    return input.transpose(dim0, dim1)


def permute(input, *dims):
    return input.permute(*dims)


def softmax(input, dim=-1):
    return _functional.softmax(input, dim=dim)


def log_softmax(input, dim=-1):
    return _functional.log_softmax(input, dim=dim)


def save(obj, path):
    """Pickle a browser-safe object to Pyodide's filesystem.

    Intended for state_dict-shaped dicts containing NumPy arrays, scalars, and
    other pickle-safe educational checkpoint values.
    """
    import pickle as _pickle
    with open(path, "wb") as _f:
        _pickle.dump(obj, _f)


def load(path, **kwargs):
    """Load an object written by save(). Extra kwargs are accepted for
    PyTorch-shaped call sites and ignored."""
    import pickle as _pickle
    with open(path, "rb") as _f:
        return _pickle.load(_f)


__version__ = "${pkg.version}"

__all__ = [
    "TensorProxy", "Tensor",
    "tensor", "zeros", "ones", "randn", "arange", "from_numpy",
    "zeros_like", "ones_like", "cat", "stack", "where", "minimum",
    "triu", "tril", "multinomial", "einsum",
    "Session", "get_default_session", "set_default_session", "new_session",
    "manual_seed", "no_grad", "inference_mode",
    "sigmoid", "matmul", "mm", "bmm", "exp", "log", "tanh", "abs",
    "sqrt", "pow", "sin", "cos", "rsqrt",
    "sum", "mean", "argmax", "prod", "std", "var", "gather", "repeat_interleave",
    "cumsum", "sort", "reshape", "view", "flatten", "squeeze",
    "unsqueeze", "transpose", "permute", "softmax", "log_softmax",
    "save", "load",
    "functional", "F", "nn", "optim", "jit", "utils", "amp", "kernels", "func",
    "custom_kernel", "onnx", "lab", "experimental",
    "realize_webgpu", "realize_tensor_plan_webgpu", "realize_tensor_plan_webgpu_resident",
    "register_webgpu_bridge", "unregister_webgpu_bridge",
    "webgpu_is_available", "webgpu_supported_opcodes",
    "gpu_plan_summary",
    "framework_operation_support",
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
  { path: "browsergrad_jit/framework-operation-contracts.v1.json", content: FRAMEWORK_OPERATION_CONTRACTS_JSON },
  { path: "browsergrad_jit/_framework_contracts.py", content: FRAMEWORK_CONTRACTS_PY },
  { path: "browsergrad_jit/_ir.py", content: IR_PY },
  { path: "browsergrad_jit/_buffer_table.py", content: BUFFER_TABLE_PY },
  { path: "browsergrad_jit/_fusion_config.py", content: FUSION_CONFIG_PY },
  { path: "browsergrad_jit/_realize.py", content: REALIZE_PY },
  { path: "browsergrad_jit/_fusion.py", content: FUSION_PY },
  { path: "browsergrad_jit/_vjp.py", content: VJP_PY },
  { path: "browsergrad_jit/_trace_cache.py", content: TRACE_CACHE_PY },
  { path: "browsergrad_jit/_safetensors.py", content: SAFETENSORS_PY },
  { path: "browsergrad_jit/_checkpoint.py", content: CHECKPOINT_PY },
  { path: "browsergrad_jit/_utils_checkpoint.py", content: UTILS_CHECKPOINT_PY },
  { path: "browsergrad_jit/_utils_data.py", content: UTILS_DATA_PY },
  { path: "browsergrad_jit/_amp.py", content: AMP_PY },
  { path: "browsergrad_jit/_bridge.py", content: BRIDGE_PY },
  { path: "browsergrad_jit/_gpu_buffer_table.py", content: GPU_BUFFER_TABLE_PY },
  { path: "browsergrad_jit/_realize_webgpu.py", content: REALIZE_WEBGPU_PY },
  { path: "browsergrad_jit/_gpu_plan.py", content: GPU_PLAN_PY },
  { path: "browsergrad_jit/_func.py", content: FUNC_PY },
  { path: "browsergrad_jit/_vmap.py", content: VMAP_PY },
  { path: "browsergrad_jit/_webnn.py", content: WEBNN_PY },
  { path: "browsergrad_jit/_cost_model.py", content: COST_MODEL_PY },
  { path: "browsergrad_jit/_custom_kernel.py", content: CUSTOM_KERNEL_PY },
  { path: "browsergrad_jit/_onnx.py", content: ONNX_PY },
  { path: "browsergrad_jit/_lab.py", content: LAB_PY },
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
