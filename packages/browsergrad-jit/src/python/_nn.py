"""browsergrad_jit._nn — nn.Module, nn.Linear, nn.Sequential, nn.ReLU,
nn.Dropout. The 0.1.0 MVP scope per PRD-005's revised plan.

INTERNAL module. Users import as `browsergrad_jit.nn`.

Conv/BatchNorm/LayerNorm/MultiHeadAttention/Embedding/RNN ship in 0.1.1+
patch releases — they need either real opcodes (PRD-006) or non-trivial
CUSTOM op wrappers that aren't load-bearing for the 0.1.0 conformance bar.
"""

from __future__ import annotations
from typing import Any, Iterator, Optional, Tuple
from collections import OrderedDict
import math

import numpy as np

from ._tensor_proxy import (
    TensorProxy,
    _BackwardCtx,
    _should_track,
    from_numpy,
    zeros,
    randn,
)
from ._ir import UOp, OP_CUSTOM
from ._errors import ShapeError
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
        # ordering — adds complexity without observed wins.
        from . import _trace_cache
        if not kwargs and _trace_cache.is_enabled():
            cached = _trace_cache.maybe_cached_forward(
                module_id=id(self),
                training=bool(self.training),
                args=args,
            )
            if cached is not None:
                return cached
        out = self.forward(*args, **kwargs)
        if not kwargs:
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
        state = {name: p.numpy() for name, p in self.named_parameters()}
        for name, b in self.named_buffers():
            state[name] = np.array(b, copy=True)
        return state

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
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


class Softmax(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim
    def forward(self, x: TensorProxy) -> TensorProxy:
        return F.softmax(x, dim=self.dim)


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
    def forward(self, x: TensorProxy) -> TensorProxy:
        return F.dropout(x, p=self.p, training=self.training)


class BatchNorm1d(Module):
    """Batch normalization over channel dim for (N, C) or (N, C, L)."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)
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
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x: TensorProxy) -> TensorProxy:
        if x.ndim not in (2, 3):
            raise ShapeError(
                f"BatchNorm1d expects 2D (N, C) or 3D (N, C, L); got shape {x.shape}"
            )
        if x.shape[1] != self.num_features:
            raise ShapeError(
                f"BatchNorm1d expected {self.num_features} channels, got {x.shape[1]}"
            )
        reduce_axes = (0,) if x.ndim == 2 else (0, 2)
        stat_shape = (1, self.num_features) if x.ndim == 2 else (1, self.num_features, 1)
        training_pass = self.training or not self.track_running_stats
        sess = x._get_session()

        captured: dict = {}

        def _forward(x_arr: np.ndarray, *affine_arrays: np.ndarray) -> np.ndarray:
            if training_pass:
                mean = x_arr.mean(axis=reduce_axes)
                var = x_arr.var(axis=reduce_axes)
                if self.training and self.track_running_stats:
                    m = self.momentum
                    self.running_mean[...] = (1.0 - m) * self.running_mean + m * mean
                    self.running_var[...] = (1.0 - m) * self.running_var + m * var
            else:
                mean = self.running_mean
                var = self.running_var
            inv_std = 1.0 / np.sqrt(var + self.eps)
            x_hat = (x_arr - mean.reshape(stat_shape)) * inv_std.reshape(stat_shape)
            captured["x_hat"] = x_hat
            captured["inv_std"] = inv_std
            out = x_hat
            if self.affine:
                weight_arr, bias_arr = affine_arrays
                out = out * weight_arr.reshape(stat_shape) + bias_arr.reshape(stat_shape)
            return out.astype(np.dtype(x.dtype), copy=False)

        input_uops = [x._uop]
        input_proxies = [x]
        if self.affine:
            input_uops.extend([self.weight._uop, self.bias._uop])
            input_proxies.extend([self.weight, self.bias])
        uop = UOp(
            op=OP_CUSTOM,
            inputs=tuple(input_uops),
            shape=x.shape,
            dtype=x.dtype,
            arg={"fn": _forward, "captures": (), "name": "batch_norm1d"},
        )

        def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]):
            x_arr = ins[0]
            x_hat = captured["x_hat"]
            inv_std = captured["inv_std"]
            if self.affine:
                weight_arr = ins[1]
                grad_x_hat = dy * weight_arr.reshape(stat_shape)
                grad_weight = (dy * x_hat).sum(axis=reduce_axes)
                grad_bias = dy.sum(axis=reduce_axes)
            else:
                grad_x_hat = dy
                grad_weight = None
                grad_bias = None
            if training_pass:
                n_total = float(np.prod([x_arr.shape[a] for a in reduce_axes]))
                sum_g = grad_x_hat.sum(axis=reduce_axes, keepdims=True)
                sum_g_xhat = (grad_x_hat * x_hat).sum(axis=reduce_axes, keepdims=True)
                grad_x = inv_std.reshape(stat_shape) * (
                    grad_x_hat - sum_g / n_total - x_hat * sum_g_xhat / n_total
                )
            else:
                grad_x = grad_x_hat * inv_std.reshape(stat_shape)
            if self.affine:
                return (grad_x, grad_weight, grad_bias)
            return (grad_x,)

        requires = _should_track(*input_proxies)
        ctx = _BackwardCtx(fn=_bw, input_proxies=tuple(input_proxies)) if requires else None
        return TensorProxy(uop, session=sess, requires_grad=requires, ctx=ctx)

    def __repr__(self) -> str:
        return (
            f"BatchNorm1d({self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.affine})"
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
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
    def forward(self, logits: TensorProxy, targets: TensorProxy) -> TensorProxy:
        return F.cross_entropy(logits, targets, reduction=self.reduction)


class NLLLoss(Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
    def forward(self, log_probs: TensorProxy, targets: TensorProxy) -> TensorProxy:
        return F.nll_loss(log_probs, targets, reduction=self.reduction)


# Expose F at the package level so users can write `nn.functional.softmax`.
functional = F


__all__ = [
    "Module",
    "Parameter",
    "Linear",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "GELU",
    "Softmax",
    "Dropout",
    "BatchNorm1d",
    "Sequential",
    "MSELoss",
    "CrossEntropyLoss",
    "NLLLoss",
    "functional",
]
