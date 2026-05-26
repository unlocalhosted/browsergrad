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
    from_numpy,
    zeros,
    randn,
)
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
    train(), eval(), state_dict(), __call__ → forward().

    Not a 1:1 PyTorch port — there's no _buffers tracking, no register_*
    hooks. Those can be layered on if labs request them; the simpler base
    class is enough for the 0.1.0 conformance bar.
    """

    def __init__(self) -> None:
        # Use object.__setattr__ to bypass our custom __setattr__ on these
        # bootstrap attributes — they're storage, not user-assigned params.
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any) -> None:
        # Strip the old registration if `name` was a Parameter/Module before.
        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        if params is not None and name in params:
            del params[name]
        if modules is not None and name in modules:
            del modules[name]

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
        return self.forward(*args, **kwargs)

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

    def train(self, mode: bool = True) -> "Module":
        object.__setattr__(self, "training", mode)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def state_dict(self) -> dict:
        return {name: p.numpy() for name, p in self.named_parameters()}

    def load_state_dict(self, state: dict) -> None:
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
        bound = 1.0 / math.sqrt(in_features)
        weight_init = np.random.default_rng().uniform(
            -bound, bound, size=(out_features, in_features)
        ).astype(np.float32)
        self.weight = Parameter(from_numpy(weight_init, requires_grad=True))
        if bias:
            bias_init = np.random.default_rng().uniform(
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
    "Sequential",
    "MSELoss",
    "CrossEntropyLoss",
    "NLLLoss",
    "functional",
]
