"""browsergrad_jit._nn — nn.Module, nn.Linear, nn.Sequential, nn.ReLU,
nn.Dropout. The 0.1.0 MVP scope per PRD-005's revised plan.

INTERNAL module. Users import as `browsergrad_jit.nn`.

Conv/LayerNorm and BatchNorm1d use typed IR. MultiHeadAttention/Embedding/RNN
remain module-level coverage that can be promoted as GPU-native demand lands.
"""

from __future__ import annotations
from typing import Any, Iterator, Optional, Tuple
from collections import OrderedDict
import math
import uuid

import numpy as np

from ._tensor_proxy import (
    TensorProxy,
    _BackwardCtx,
    _should_track,
    from_numpy,
    zeros,
    randn,
)
from ._ir import (
    UOp,
    OP_BUFFER,
    OP_BATCH_NORM_1D,
    OP_BATCH_NORM_1D_STATS_UPDATE,
)
from ._errors import JitNotImplementedError, ShapeError
from ._framework_contracts import (
    BATCH_NORM_1D_STATE_EFFECT_KIND,
    execute_batch_norm_1d_vjp_array,
    infer_batch_norm_1d_contract,
    normalize_dropout_probability,
    normalize_batch_norm_1d_eps,
    normalize_batch_norm_1d_flag,
    normalize_batch_norm_1d_momentum,
    normalize_batch_norm_1d_num_features,
)
from . import _functional as F


# ---------------------------------------------------------------------------
# Parameter
# ---------------------------------------------------------------------------


class Parameter(TensorProxy):
    """A TensorProxy with `requires_grad=True` by default.

    Matches torch.nn.Parameter — it's structurally a Tensor, semantically
    a learnable. The only behavioral difference from TensorProxy is the
    default flag and the type that `Module.__setattr__` looks for to
    auto-register parameters."""

    def __init__(
        self,
        data: TensorProxy,
        requires_grad: bool = True,
    ) -> None:
        # Inherit the underlying UOp + session from `data`. We don't allow
        # constructing from raw arrays here — go through `from_numpy` first.
        super().__init__(
            uop=data._uop,
            session=data._session,
            requires_grad=requires_grad,
            ctx=None,  # leaf
        )

    def __repr__(self) -> str:
        return (
            f"Parameter(shape={self.shape}, dtype={self.dtype!r}, "
            f"requires_grad={self.requires_grad})"
        )


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class Module:
    """Base class. Auto-registers Parameters and Modules assigned as
    attributes. Mirrors torch.nn.Module's essentials: parameters(),
    buffers(), train(), eval(), state_dict(), __call__ → forward().

    Not a 1:1 PyTorch port — there are no register_* hooks yet. Those can be
    layered on if labs request them; the simpler base class keeps the teaching
    surface readable.
    """

    def __init__(self) -> None:
        # Use object.__setattr__ to bypass our custom __setattr__ on these
        # bootstrap attributes — they're storage, not user-assigned params.
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any) -> None:
        # Strip the old registration if `name` was a Parameter/Module before.
        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        buffers = self.__dict__.get("_buffers")
        if params is not None and name in params:
            del params[name]
        if modules is not None and name in modules:
            del modules[name]
        if buffers is not None and name in buffers:
            del buffers[name]

        if isinstance(value, Parameter):
            if params is None:
                raise RuntimeError(
                    f"{type(self).__name__}: assigned a Parameter before "
                    f"calling super().__init__()."
                )
            params[name] = value
        elif isinstance(value, Module):
            if modules is None:
                raise RuntimeError(
                    f"{type(self).__name__}: assigned a Module before "
                    f"calling super().__init__()."
                )
            modules[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Trace cache fast path: if the cache has an entry for this
        # (module instance, training mode, input signature), reuse the
        # cached IR with the new input BUFFERs rebound. Misses fall
        # through to the regular forward.
        #
        # We only consult the cache for positional-arg calls (the common
        # case for the `Module(x)` pattern). Kwargs disable the cache
        # because the signature would need to include kwarg identity and
        # ordering. Mutable buffers also disable it, including buffers owned
        # by descendants: bypassing forward would otherwise hide state
        # updates or reuse a graph built from an obsolete buffer snapshot.
        from . import _trace_cache
        cacheable = (
            not kwargs
            and next(self.buffers(recurse=True), None) is None
        )
        if cacheable and _trace_cache.is_enabled():
            cached = _trace_cache.maybe_cached_forward(
                module_id=id(self),
                training=bool(self.training),
                args=args,
            )
            if cached is not None:
                return cached
        out = self.forward(*args, **kwargs)
        if cacheable:
            _trace_cache.record(
                module_id=id(self),
                training=bool(self.training),
                args=args,
                output=out,
            )
        return out

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()"
        )

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Yield every Parameter owned by this module or its submodules."""
        for p in self._parameters.values():
            yield p
        if recurse:
            for m in self._modules.values():
                yield from m.parameters(recurse=True)

    def named_parameters(self, prefix: str = "",
                         recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        for name, p in self._parameters.items():
            yield (prefix + name if prefix else name), p
        if recurse:
            for mod_name, m in self._modules.items():
                child_prefix = (prefix + mod_name + ".") if prefix else (mod_name + ".")
                yield from m.named_parameters(prefix=child_prefix, recurse=True)

    def register_buffer(self, name: str, value: Any) -> None:
        if "." in name or not name:
            raise KeyError(f"invalid buffer name {name!r}")
        if name in self._parameters or name in self._modules:
            raise KeyError(f"attribute {name!r} is already registered")
        arr = np.asarray(value)
        self._buffers[name] = arr.copy()
        object.__setattr__(self, name, self._buffers[name])

    def buffers(self, recurse: bool = True) -> Iterator[np.ndarray]:
        for b in self._buffers.values():
            yield b
        if recurse:
            for m in self._modules.values():
                yield from m.buffers(recurse=True)

    def named_buffers(self, prefix: str = "",
                      recurse: bool = True) -> Iterator[Tuple[str, np.ndarray]]:
        for name, b in self._buffers.items():
            yield (prefix + name if prefix else name), b
        if recurse:
            for mod_name, m in self._modules.items():
                child_prefix = (prefix + mod_name + ".") if prefix else (mod_name + ".")
                yield from m.named_buffers(prefix=child_prefix, recurse=True)

    def train(self, mode: bool = True) -> "Module":
        object.__setattr__(self, "training", mode)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def state_dict(self) -> dict:
        self._synchronize_pending_state()
        state = {name: p.numpy() for name, p in self.named_parameters()}
        for name, b in self.named_buffers():
            state[name] = np.array(b, copy=True)
        return state

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        self._synchronize_pending_state()
        expected = {name for name, _ in self.named_parameters()}
        expected.update(name for name, _ in self.named_buffers())
        if strict:
            extra = sorted(set(state.keys()) - expected)
            if extra:
                raise KeyError(
                    f"state_dict has unexpected keys {extra!r}. Expected keys: {sorted(expected)}"
                )
        for name, p in self.named_parameters():
            if name not in state:
                raise KeyError(
                    f"state_dict missing key {name!r}. Have keys: {list(state)}"
                )
            arr = np.asarray(state[name])
            if arr.shape != p.shape:
                raise ShapeError(
                    f"state_dict[{name!r}] shape {arr.shape} != parameter shape {p.shape}"
                )
            # Write into the parameter's underlying buffer.
            session = p._get_session()
            session.buffer_table.update(p._uop.inputs[0].arg,
                                        arr.astype(np.dtype(p.dtype), copy=False))
        for name, b in self.named_buffers():
            if name not in state:
                raise KeyError(
                    f"state_dict missing key {name!r}. Have keys: {list(state)}"
                )
            arr = np.asarray(state[name])
            if arr.shape != b.shape:
                raise ShapeError(
                    f"state_dict[{name!r}] shape {arr.shape} != buffer shape {b.shape}"
                )
            b[...] = arr.astype(b.dtype, copy=False)

    def _synchronize_pending_state(self) -> None:
        """Commit lazy state effects before externally observing state."""
        for module in self._modules.values():
            module._synchronize_pending_state()

    def zero_grad(self) -> None:
        """Reset all parameter gradients. Mirrors optimizer.zero_grad()."""
        for p in self.parameters():
            p.grad = None


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------


class Linear(Module):
    """y = x @ weight.T + bias. Same convention as torch.nn.Linear."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Kaiming-uniform init (PyTorch default). For our v0, simpler:
        # uniform(-bound, bound) with bound = 1 / sqrt(in_features).
        # NB: use np.random.uniform (legacy global API) — NOT default_rng() —
        # because the latter creates a fresh generator per call and ignores
        # np.random.seed() / bg.manual_seed(). The educational use case
        # depends on deterministic init across runs given a fixed seed.
        bound = 1.0 / math.sqrt(in_features)
        weight_init = np.random.uniform(
            -bound, bound, size=(out_features, in_features)
        ).astype(np.float32)
        self.weight = Parameter(from_numpy(weight_init, requires_grad=True))
        if bias:
            bias_init = np.random.uniform(
                -bound, bound, size=(out_features,)
            ).astype(np.float32)
            self.bias: Optional[Parameter] = Parameter(from_numpy(bias_init, requires_grad=True))
        else:
            self.bias = None

    def forward(self, x: TensorProxy) -> TensorProxy:
        return F.linear(x, self.weight, self.bias)

    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None})"
        )


# ---------------------------------------------------------------------------
# Convolution modules
# ---------------------------------------------------------------------------


class Conv1d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if groups <= 0:
            raise ValueError("Conv1d: groups must be positive")
        if in_channels % groups != 0:
            raise ValueError("Conv1d: in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("Conv1d: out_channels must be divisible by groups")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.groups = int(groups)
        fan_in = (self.in_channels // self.groups) * self.kernel_size
        bound = 1.0 / math.sqrt(fan_in)
        weight = np.random.uniform(
            -bound,
            bound,
            size=(self.out_channels, self.in_channels // self.groups, self.kernel_size),
        ).astype(np.float32)
        self.weight = Parameter(from_numpy(weight, requires_grad=True))
        if bias:
            self.bias: Optional[Parameter] = Parameter(
                from_numpy(np.zeros((self.out_channels,), dtype=np.float32), requires_grad=True)
            )
        else:
            self.bias = None

    def forward(self, x: TensorProxy) -> TensorProxy:
        return F.conv1d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Any,
        stride: Any = 1,
        padding: Any = 0,
        dilation: Any = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if groups <= 0:
            raise ValueError("Conv2d: groups must be positive")
        if in_channels % groups != 0:
            raise ValueError("Conv2d: in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("Conv2d: out_channels must be divisible by groups")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = F._pair2(kernel_size, "kernel_size")
        self.stride = F._pair2(stride, "stride")
        self.padding = F._pair2(padding, "padding")
        self.dilation = F._pair2(dilation, "dilation")
        self.groups = int(groups)
        kh, kw = self.kernel_size
        fan_in = (self.in_channels // self.groups) * kh * kw
        bound = 1.0 / math.sqrt(fan_in)
        weight = np.random.uniform(
            -bound,
            bound,
            size=(self.out_channels, self.in_channels // self.groups, kh, kw),
        ).astype(np.float32)
        self.weight = Parameter(from_numpy(weight, requires_grad=True))
        if bias:
            self.bias: Optional[Parameter] = Parameter(
                from_numpy(np.zeros((self.out_channels,), dtype=np.float32), requires_grad=True)
            )
        else:
            self.bias = None

    def forward(self, x: TensorProxy) -> TensorProxy:
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def __repr__(self) -> str:
        return (
            f"Conv2d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, groups={self.groups})"
        )


class ConvTranspose2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Any,
        stride: Any = 1,
        padding: Any = 0,
        output_padding: Any = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Any = 1,
    ) -> None:
        super().__init__()
        if groups <= 0:
            raise ValueError("ConvTranspose2d: groups must be positive")
        if in_channels % groups != 0:
            raise ValueError("ConvTranspose2d: in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("ConvTranspose2d: out_channels must be divisible by groups")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = F._pair2(kernel_size, "kernel_size")
        self.stride = F._pair2(stride, "stride")
        self.padding = F._pair2(padding, "padding")
        self.output_padding = F._pair2(output_padding, "output_padding")
        self.dilation = F._pair2(dilation, "dilation")
        self.groups = int(groups)
        kh, kw = self.kernel_size
        fan_in = (self.out_channels // self.groups) * kh * kw
        bound = 1.0 / math.sqrt(fan_in)
        weight = np.random.uniform(
            -bound,
            bound,
            size=(self.in_channels, self.out_channels // self.groups, kh, kw),
        ).astype(np.float32)
        self.weight = Parameter(from_numpy(weight, requires_grad=True))
        if bias:
            self.bias: Optional[Parameter] = Parameter(
                from_numpy(np.zeros((self.out_channels,), dtype=np.float32), requires_grad=True)
            )
        else:
            self.bias = None

    def forward(self, x: TensorProxy) -> TensorProxy:
        return F.conv_transpose2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )


class Conv3d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Any,
        stride: Any = 1,
        padding: Any = 0,
        dilation: Any = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if groups <= 0:
            raise ValueError("Conv3d: groups must be positive")
        if in_channels % groups != 0:
            raise ValueError("Conv3d: in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("Conv3d: out_channels must be divisible by groups")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = F._triple3(kernel_size, "kernel_size")
        self.stride = F._triple3(stride, "stride")
        self.padding = F._triple3(padding, "padding")
        self.dilation = F._triple3(dilation, "dilation")
        self.groups = int(groups)
        kd, kh, kw = self.kernel_size
        fan_in = (self.in_channels // self.groups) * kd * kh * kw
        bound = 1.0 / math.sqrt(fan_in)
        weight = np.random.uniform(
            -bound,
            bound,
            size=(self.out_channels, self.in_channels // self.groups, kd, kh, kw),
        ).astype(np.float32)
        self.weight = Parameter(from_numpy(weight, requires_grad=True))
        if bias:
            self.bias: Optional[Parameter] = Parameter(
                from_numpy(np.zeros((self.out_channels,), dtype=np.float32), requires_grad=True)
            )
        else:
            self.bias = None

    def forward(self, x: TensorProxy) -> TensorProxy:
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


# ---------------------------------------------------------------------------
# Activation modules — thin wrappers around the functional equivalents
# ---------------------------------------------------------------------------


class ReLU(Module):
    def forward(self, x: TensorProxy) -> TensorProxy:
        return F.relu(x)


class Sigmoid(Module):
    def forward(self, x: TensorProxy) -> TensorProxy:
        return F.sigmoid(x)


class Tanh(Module):
    def forward(self, x: TensorProxy) -> TensorProxy:
        return F.tanh(x)


class GELU(Module):
    def forward(self, x: TensorProxy) -> TensorProxy:
        return F.gelu(x)


class SiLU(Module):
    def forward(self, x: TensorProxy) -> TensorProxy:
        return F.silu(x)


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope
    def forward(self, x: TensorProxy) -> TensorProxy:
        return F.leaky_relu(x, negative_slope=self.negative_slope)


class Softmax(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim
    def forward(self, x: TensorProxy) -> TensorProxy:
        return F.softmax(x, dim=self.dim)


class Dropout(Module):
    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        self.p = normalize_dropout_probability(p)
        if type(inplace) is not bool:
            raise ShapeError("Dropout: inplace must be an exact bool")
        if inplace:
            raise JitNotImplementedError(
                "Dropout(inplace=True) requires typed mutation semantics and "
                "is not supported"
            )
        self.inplace = inplace

    def forward(self, x: TensorProxy) -> TensorProxy:
        return F.dropout(
            x,
            p=self.p,
            training=self.training,
            inplace=self.inplace,
        )

    def __repr__(self) -> str:
        return f"Dropout(p={self.p}, inplace={self.inplace})"


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    def forward(self, x: TensorProxy) -> TensorProxy:
        return x.flatten(start_dim=self.start_dim, end_dim=self.end_dim)


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: Any,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            ns = (int(normalized_shape),)
        elif isinstance(normalized_shape, (tuple, list)) and len(normalized_shape) > 0:
            ns = tuple(int(v) for v in normalized_shape)
        else:
            raise ValueError("LayerNorm: normalized_shape must be int or non-empty tuple/list")
        self.normalized_shape = ns
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = Parameter(from_numpy(
                np.ones(ns, dtype=np.float32),
                requires_grad=True,
            ))
            self.bias = Parameter(from_numpy(
                np.zeros(ns, dtype=np.float32),
                requires_grad=True,
            ))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: TensorProxy) -> TensorProxy:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )

    def __repr__(self) -> str:
        return (
            f"LayerNorm({self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine})"
        )


class BatchNorm1d(Module):
    """Batch normalization over channel dim for (N, C) or (N, C, L)."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = normalize_batch_norm_1d_num_features(num_features)
        self.eps = normalize_batch_norm_1d_eps(eps)
        self.momentum = normalize_batch_norm_1d_momentum(momentum)
        self.affine = normalize_batch_norm_1d_flag(affine, "affine")
        self.track_running_stats = normalize_batch_norm_1d_flag(
            track_running_stats, "track_running_stats"
        )
        self._state_buffer_table = None
        self._state_session_token = None
        self._state_buffer_uops = None
        self._last_state_update = None
        self._last_state_effect_id = None
        self._state_name_token = uuid.uuid4().hex
        if self.affine:
            self.weight = Parameter(from_numpy(
                np.ones((self.num_features,), dtype=np.float32),
                requires_grad=True,
            ))
            self.bias = Parameter(from_numpy(
                np.zeros((self.num_features,), dtype=np.float32),
                requires_grad=True,
            ))
        else:
            self.weight = None
            self.bias = None
        if self.track_running_stats:
            self.register_buffer(
                "running_mean",
                np.zeros((self.num_features,), dtype=np.float32),
            )
            self.register_buffer(
                "running_var",
                np.ones((self.num_features,), dtype=np.float32),
            )
            self.register_buffer(
                "num_batches_tracked",
                np.asarray(0, dtype=np.int64),
            )
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def _ensure_state_buffers(self, session: Any) -> Tuple[UOp, UOp, UOp]:
        table = session.buffer_table
        if self._state_buffer_table is None:
            prefix = f"batch-norm-1d-{self._state_name_token}"
            mean_id = table.new_buffer(
                self.running_mean, name=f"{prefix}-running-mean"
            )
            var_id = table.new_buffer(
                self.running_var, name=f"{prefix}-running-var"
            )
            tracked_id = table.new_buffer(
                self.num_batches_tracked,
                name=f"{prefix}-num-batches-tracked",
            )
            self._state_buffer_table = table
            self._state_session_token = table.session_token
            self._state_buffer_uops = (
                UOp(
                    op=OP_BUFFER,
                    inputs=(),
                    shape=(self.num_features,),
                    dtype="float32",
                    arg=mean_id,
                ),
                UOp(
                    op=OP_BUFFER,
                    inputs=(),
                    shape=(self.num_features,),
                    dtype="float32",
                    arg=var_id,
                ),
                UOp(
                    op=OP_BUFFER,
                    inputs=(),
                    shape=(),
                    dtype="int64",
                    arg=tracked_id,
                ),
            )
        elif self._state_buffer_table is not table:
            raise ShapeError(
                "BatchNorm1d running state is bound to session "
                f"{self._state_session_token!r}; rebuild or load a state_dict "
                "into a module owned by the current session"
            )
        return self._state_buffer_uops

    def _drop_applied_state_tail(self) -> None:
        if (
            self._last_state_effect_id is not None
            and self._state_buffer_table is not None
            and self._state_buffer_table.effect_applied(
                self._last_state_effect_id,
                BATCH_NORM_1D_STATE_EFFECT_KIND,
            )
        ):
            self._last_state_update = None
            self._last_state_effect_id = None

    def _synchronize_pending_state(self) -> None:
        if (
            self._last_state_update is not None
            and self._last_state_effect_id is not None
        ):
            from ._realize import realize
            realize(self._last_state_update, self._state_buffer_table)
            self._drop_applied_state_tail()
        super()._synchronize_pending_state()

    def reset_running_stats(self) -> None:
        if not self.track_running_stats:
            return
        self._synchronize_pending_state()
        self.running_mean[...] = np.float32(0.0)
        self.running_var[...] = np.float32(1.0)
        self.num_batches_tracked[...] = np.int64(0)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            weight_session = self.weight._get_session()
            bias_session = self.bias._get_session()
            weight_session.buffer_table.update(
                self.weight._uop.inputs[0].arg,
                np.ones((self.num_features,), dtype=np.float32),
            )
            bias_session.buffer_table.update(
                self.bias._uop.inputs[0].arg,
                np.zeros((self.num_features,), dtype=np.float32),
            )

    def forward(self, x: TensorProxy) -> TensorProxy:
        if x.ndim not in (2, 3):
            raise ShapeError(
                f"BatchNorm1d expects 2D (N, C) or 3D (N, C, L); got shape {x.shape}"
            )
        if x.shape[1] != self.num_features:
            raise ShapeError(
                f"BatchNorm1d expected {self.num_features} channels, got {x.shape[1]}"
            )
        if x.dtype != "float32":
            raise ShapeError(
                f"BatchNorm1d v1 requires float32 input, got {x.dtype!r}"
            )
        sess = x._get_session()
        if self.affine and any(
            parameter._get_session().buffer_table is not sess.buffer_table
            for parameter in (self.weight, self.bias)
        ):
            raise ShapeError(
                "BatchNorm1d input and affine parameters must belong to the "
                "same session"
            )

        stats_proxy = None
        if self.training and self.track_running_stats:
            state_buffers = self._ensure_state_buffers(sess)
            self._drop_applied_state_tail()
            effect_id = sess.buffer_table.reserve_effect(
                BATCH_NORM_1D_STATE_EFFECT_KIND,
                tuple(buffer.arg for buffer in state_buffers),
            )
            state_inputs = []
            has_prior = self._last_state_update is not None
            if has_prior:
                state_inputs.append(self._last_state_update)
            state_inputs.extend((x._uop,) + state_buffers)
            state_update = UOp(
                op=OP_BATCH_NORM_1D_STATS_UPDATE,
                inputs=tuple(state_inputs),
                shape=(2, self.num_features),
                dtype="float32",
                arg={
                    "effect_id": effect_id,
                    "momentum": self.momentum,
                    "has_prior": has_prior,
                },
            )
            stats_proxy = TensorProxy(
                state_update,
                session=sess,
                requires_grad=False,
            )
            self._last_state_update = state_update
            self._last_state_effect_id = effect_id
            stats_mode = "batch-state"
        elif not self.training and self.track_running_stats:
            self._ensure_state_buffers(sess)
            self._synchronize_pending_state()
            snapshot = np.stack(
                (
                    np.array(self.running_mean, dtype=np.float32, copy=True),
                    np.array(self.running_var, dtype=np.float32, copy=True),
                ),
                axis=0,
            )
            stats_proxy = from_numpy(snapshot, session=sess)
            stats_mode = "running"
        else:
            stats_mode = "batch"

        input_proxies = [x]
        if stats_proxy is not None:
            input_proxies.append(stats_proxy)
        if self.affine:
            input_proxies.extend([self.weight, self.bias])
        input_uops = tuple(proxy._uop for proxy in input_proxies)
        contract = infer_batch_norm_1d_contract(
            input_uops,
            self.eps,
            self.affine,
            stats_mode,
        )
        uop = UOp(
            op=OP_BATCH_NORM_1D,
            inputs=input_uops,
            shape=x.shape,
            dtype="float32",
            arg={
                "eps": contract.eps,
                "affine": contract.affine,
                "stats_mode": contract.stats_mode,
            },
        )

        def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]):
            gradients = [
                execute_batch_norm_1d_vjp_array(
                    contract, ins, dy, "input"
                )
            ]
            if contract.stats_input_index is not None:
                gradients.append(None)
            if contract.affine:
                gradients.extend((
                    execute_batch_norm_1d_vjp_array(
                        contract, ins, dy, "weight"
                    ),
                    execute_batch_norm_1d_vjp_array(
                        contract, ins, dy, "bias"
                    ),
                ))
            return tuple(gradients)

        requires = _should_track(*input_proxies)
        ctx = _BackwardCtx(fn=_bw, input_proxies=tuple(input_proxies)) if requires else None
        return TensorProxy(uop, session=sess, requires_grad=requires, ctx=ctx)

    def __repr__(self) -> str:
        return (
            f"BatchNorm1d({self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.affine}, "
            f"track_running_stats={self.track_running_stats})"
        )


# ---------------------------------------------------------------------------
# Sequential — composition
# ---------------------------------------------------------------------------


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        for i, layer in enumerate(layers):
            # Use string keys so state_dict prefixes match PyTorch's: "0.weight"...
            setattr(self, str(i), layer)
        self._sequence_length = len(layers)

    def forward(self, x: Any) -> Any:
        for i in range(self._sequence_length):
            x = getattr(self, str(i))(x)
        return x

    def __len__(self) -> int:
        return self._sequence_length

    def __getitem__(self, idx: int) -> Module:
        if idx < 0:
            idx += self._sequence_length
        return getattr(self, str(idx))


# ---------------------------------------------------------------------------
# Loss modules
# ---------------------------------------------------------------------------


class MSELoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
    def forward(self, input: TensorProxy, target: TensorProxy) -> TensorProxy:
        return F.mse_loss(input, target, reduction=self.reduction)


class CrossEntropyLoss(Module):
    def __init__(
        self,
        weight: Optional[TensorProxy] = None,
        size_average: Optional[bool] = None,
        ignore_index: int = -100,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        validation_target = UOp(
            op=OP_BUFFER,
            inputs=(),
            shape=(),
            dtype="int64",
            arg="cross-entropy-weight-validation-target",
        )
        if weight is None:
            validation_input = UOp(
                op=OP_BUFFER,
                inputs=(),
                shape=(1,),
                dtype="float32",
                arg="cross-entropy-validation-input",
            )
            object.__setattr__(self, "weight", None)
            validation_inputs = (validation_input, validation_target)
        else:
            if not isinstance(weight, TensorProxy):
                raise TypeError(
                    "CrossEntropyLoss: weight must be a TensorProxy or None, "
                    f"got {type(weight).__name__}"
                )
            if weight.requires_grad:
                raise ValueError(
                    "CrossEntropyLoss: weight is non-differentiable and must "
                    "not require grad"
                )
            validation_inputs = (
                weight._uop,
                validation_target,
                weight._uop,
            )
        normalized_reduction = F._normalize_legacy_loss_reduction(
            "CrossEntropyLoss",
            reduction,
            size_average,
            reduce,
        )
        F.infer_cross_entropy_contract(
            validation_inputs,
            normalized_reduction,
            ignore_index,
            weight is not None,
            label_smoothing,
            "indices",
            0,
        )
        if weight is not None:
            self.register_buffer("weight", weight.numpy())
        self.ignore_index = ignore_index
        self.reduction = normalized_reduction
        self.label_smoothing = float(label_smoothing)
    def forward(self, input: TensorProxy, target: TensorProxy) -> TensorProxy:
        weight = (
            None
            if self.weight is None
            else from_numpy(
                np.array(self.weight, copy=True),
                session=input._get_session(),
            )
        )
        return F.cross_entropy(
            input,
            target,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class NLLLoss(Module):
    def __init__(
        self,
        weight: Optional[TensorProxy] = None,
        size_average: Optional[bool] = None,
        ignore_index: int = -100,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if weight is None:
            object.__setattr__(self, "weight", None)
        else:
            if not isinstance(weight, TensorProxy):
                raise TypeError(
                    "NLLLoss: weight must be a TensorProxy or None, got "
                    f"{type(weight).__name__}"
                )
            if weight.requires_grad:
                raise ValueError(
                    "NLLLoss: weight is non-differentiable and must not require grad"
                )
            validation_target = UOp(
                op=OP_BUFFER,
                inputs=(),
                shape=(),
                dtype="int64",
                arg="nll-loss-weight-validation-target",
            )
            F.infer_nll_loss_contract(
                (weight._uop, validation_target, weight._uop),
                "mean",
                ignore_index,
                True,
                0,
            )
            self.register_buffer("weight", weight.numpy())
        self.ignore_index = ignore_index
        self.reduction = F._normalize_legacy_loss_reduction(
            "NLLLoss",
            reduction,
            size_average,
            reduce,
        )
    def forward(self, input: TensorProxy, target: TensorProxy) -> TensorProxy:
        weight = (
            None
            if self.weight is None
            else from_numpy(
                np.array(self.weight, copy=True),
                session=input._get_session(),
            )
        )
        return F.nll_loss(
            input,
            target,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


class BCEWithLogitsLoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
    def forward(self, logits: TensorProxy, targets: TensorProxy) -> TensorProxy:
        return F.binary_cross_entropy_with_logits(
            logits, targets, reduction=self.reduction
        )


class BCELoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
    def forward(self, input: TensorProxy, target: TensorProxy) -> TensorProxy:
        return F.binary_cross_entropy(input, target, reduction=self.reduction)


class L1Loss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
    def forward(self, input: TensorProxy, target: TensorProxy) -> TensorProxy:
        return F.l1_loss(input, target, reduction=self.reduction)


class SmoothL1Loss(Module):
    def __init__(self, beta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    def forward(self, input: TensorProxy, target: TensorProxy) -> TensorProxy:
        return F.smooth_l1_loss(
            input, target, beta=self.beta, reduction=self.reduction
        )


class KLDivLoss(Module):
    def __init__(self, reduction: str = "mean", log_target: bool = False) -> None:
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target
    def forward(self, input: TensorProxy, target: TensorProxy) -> TensorProxy:
        return F.kl_div(input, target, reduction=self.reduction, log_target=self.log_target)


# Expose F at the package level so users can write `nn.functional.softmax`.
functional = F


__all__ = [
    "Module",
    "Parameter",
    "Linear",
    "Conv1d",
    "Conv2d",
    "ConvTranspose2d",
    "Conv3d",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "GELU",
    "SiLU",
    "LeakyReLU",
    "Softmax",
    "Dropout",
    "Flatten",
    "LayerNorm",
    "BatchNorm1d",
    "Sequential",
    "MSELoss",
    "CrossEntropyLoss",
    "NLLLoss",
    "BCEWithLogitsLoss",
    "BCELoss",
    "L1Loss",
    "SmoothL1Loss",
    "KLDivLoss",
    "functional",
]
