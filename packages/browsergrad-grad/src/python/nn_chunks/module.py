
"""browsergrad_grad.nn — neural network building blocks."""

import math
import numpy as np
from typing import Iterator, Optional, Tuple
from .tensor import Tensor, _build_ctx, stack
from . import functional as F


class Module:
    """Base class for everything with learnable parameters.

    Subclasses should:
      1. Assign Tensors with requires_grad=True as attributes (auto-tracked
         as parameters via __setattr__).
      2. Override .forward(*args, **kwargs).

    The default __call__ delegates to forward.
    """

    def __init__(self):
        self._modules: dict[str, "Module"] = {}
        self._parameters: dict[str, Tensor] = {}
        self._buffers: dict[str, Optional[Tensor]] = {}
        self._forward_hooks: list = []
        self.training: bool = True

    def __setattr__(self, name, value):
        if isinstance(value, Tensor) and value.requires_grad:
            self.__dict__.setdefault("_parameters", {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        object.__setattr__(self, name, value)

    def register_buffer(self, name: str, tensor, persistent: bool = True) -> None:
        """Register a non-trainable tensor as a buffer.

        Buffers are included in state_dict() / load_state_dict() but are
        excluded from parameters() — the optimizer never updates them.
        `persistent` is accepted for API compatibility but ignored.
        """
        self.__dict__.setdefault("_buffers", {})[name] = tensor
        object.__setattr__(self, name, tensor)

    def register_forward_hook(self, fn):
        """Register a hook fn(module, input, output) that runs after every
        forward pass. Returns a handle (the function itself) so the caller
        can pass it back to remove_forward_hook.
        """
        self.__dict__.setdefault("_forward_hooks", []).append(fn)
        return fn

    def remove_forward_hook(self, fn) -> None:
        hooks = self.__dict__.get("_forward_hooks", [])
        if fn in hooks:
            hooks.remove(fn)

    def parameters(self) -> Iterator[Tensor]:
        """Yield every parameter Tensor in this module and its submodules."""
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def state_dict(self, prefix: str = "") -> dict:
        """Return a flat dict[qualified-name → np.ndarray copy] of all
        parameters and buffers in this module and its submodules.

        Matches torch.nn.Module.state_dict() for the subset we model. Values
        are NumPy array copies — safe to serialize, mutate, or hand off
        without affecting the live parameters.
        """
        out: dict = {}
        for name, p in self._parameters.items():
            if p is None:
                continue
            out[prefix + name] = p.data.copy()
        for name, b in self.__dict__.get("_buffers", {}).items():
            if b is None:
                continue
            out[prefix + name] = b.data.copy()
        for name, m in self._modules.items():
            if m is None:
                continue
            out.update(m.state_dict(prefix=prefix + name + "."))
        return out

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        """Copy values from state (dict of name to ndarray or Tensor) into
        the module's parameters in place.

        Raises RuntimeError on shape mismatch. With strict=True (default),
        also raises on unknown or missing keys — matching PyTorch.
        """
        own = self.state_dict()
        own_keys = set(own.keys())
        given_keys = set(state.keys())
        if strict:
            unexpected = given_keys - own_keys
            missing = own_keys - given_keys
            if unexpected or missing:
                raise RuntimeError(
                    "load_state_dict mismatch: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
        self._assign_state(state, prefix="")

    def _assign_state(self, state: dict, prefix: str) -> None:
        for name, p in self._parameters.items():
            if p is None:
                continue
            key = prefix + name
            if key not in state:
                continue
            value = state[key]
            arr = value.data if isinstance(value, Tensor) else np.asarray(value)
            if arr.shape != p.data.shape:
                raise RuntimeError(
                    f"shape mismatch for {key}: "
                    f"checkpoint {tuple(arr.shape)} vs model {tuple(p.data.shape)}"
                )
            p.data[...] = arr.astype(p.data.dtype, copy=False)
        for name, b in self.__dict__.get("_buffers", {}).items():
            if b is None:
                continue
            key = prefix + name
            if key not in state:
                continue
            value = state[key]
            arr = value.data if isinstance(value, Tensor) else np.asarray(value)
            b.data[...] = arr.astype(b.data.dtype, copy=False)
        for name, m in self._modules.items():
            if m is None:
                continue
            m._assign_state(state, prefix=prefix + name + ".")

    def train(self, mode: bool = True) -> "Module":
        """Set this module and all submodules to training mode."""
        self.training = bool(mode)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        """Set this module and all submodules to evaluation mode."""
        return self.train(False)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()"
        )

    def __call__(self, *args, **kwargs):
        output = self.forward(*args, **kwargs)
        hooks = self.__dict__.get("_forward_hooks", [])
        for h in hooks:
            inp = args[0] if len(args) == 1 else args
            h(self, inp, output)
        return output


def Parameter(data, requires_grad: bool = True) -> Tensor:
    """Mark a tensor as a learnable parameter (requires_grad=True by default).

    Drop-in for torch.nn.Parameter. Module.__setattr__ automatically registers
    any Tensor with requires_grad=True in _parameters, so this is a thin factory.
    """
    if isinstance(data, Tensor):
        if data.requires_grad == requires_grad:
            return data
        return Tensor(data.data.copy(), requires_grad=requires_grad)
    return Tensor(data, requires_grad=requires_grad)


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    """Clip gradients by global norm. Matches torch.nn.utils.clip_grad_norm_.

    Returns the total pre-clip norm. Modifies .grad tensors in-place.
    """
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return 0.0
    total_norm = 0.0
    for p in params_with_grad:
        total_norm += float(np.linalg.norm(p.grad.data.flatten(), ord=norm_type)) ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    clip_coef = float(max_norm) / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params_with_grad:
            p.grad.data = (p.grad.data * clip_coef).astype(p.grad.data.dtype)
    return total_norm


# ─── Linear ────────────────────────────────────────────────
