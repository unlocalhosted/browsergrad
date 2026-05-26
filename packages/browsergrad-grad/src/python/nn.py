
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
        self._forward_hooks: list = []
        self.training: bool = True

    def __setattr__(self, name, value):
        if isinstance(value, Tensor) and value.requires_grad:
            self.__dict__.setdefault("_parameters", {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        object.__setattr__(self, name, value)

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
        parameters in this module and its submodules.

        Matches torch.nn.Module.state_dict() for the subset we model. Values
        are NumPy array copies — safe to serialize, mutate, or hand off
        without affecting the live parameters.
        """
        out: dict = {}
        for name, p in self._parameters.items():
            if p is None:
                continue
            out[prefix + name] = p.data.copy()
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


# ─── Linear ────────────────────────────────────────────────

class Linear(Module):
    """y = x @ W^T + b (PyTorch convention).

    W has shape (out_features, in_features); b has shape (out_features,).
    Initialized via Kaiming uniform on W and zeros on b, matching torch.nn.Linear.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        bound = 1.0 / math.sqrt(in_features)
        rng = np.random.default_rng()
        W_data = rng.uniform(-bound, bound, size=(out_features, in_features)).astype(np.float32)
        self.weight = Tensor(W_data, requires_grad=True)
        if bias:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # v0.2: bias broadcasts naturally via Tensor.__add__.
        out = x @ self.weight.transpose(0, 1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


# ─── Conv2d ────────────────────────────────────────────────

class Conv2d(Module):
    """2D convolution. v0.3: square kernel, isotropic stride/padding, dense.

    Forward shape:
        input:  (N, C_in, H, W)
        weight: (C_out, C_in, kernel_size, kernel_size)
        bias:   (C_out,)
        output: (N, C_out, H_out, W_out)
            H_out = (H + 2*padding - kernel_size) // stride + 1
            W_out same
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1.0 / math.sqrt(fan_in)
        rng = np.random.default_rng()
        W = rng.uniform(
            -bound, bound,
            size=(out_channels, in_channels, kernel_size, kernel_size),
        ).astype(np.float32)
        self.weight = Tensor(W, requires_grad=True)
        if bias:
            self.bias = Tensor(np.zeros(out_channels, dtype=np.float32), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # im2col-based correlation. Two outer loops over (i, j) gather each
        # K×K window into a column; the actual multiply-accumulate happens
        # via a batched matmul. ~10-100× faster than the naive 4-deep loops
        # for typical CNN shapes; same numerical result; same surface.
        N, C_in, H, W = x.data.shape
        K, S, P = self.kernel_size, self.stride, self.padding
        C_out = self.out_channels
        if P > 0:
            x_padded = np.pad(
                x.data, ((0, 0), (0, 0), (P, P), (P, P)), mode="constant",
            )
        else:
            x_padded = x.data
        H_pad = H + 2 * P
        W_pad = W + 2 * P
        H_out = (H_pad - K) // S + 1
        W_out = (W_pad - K) // S + 1
        L = H_out * W_out
        # cols: (N, C_in * K * K, H_out * W_out)
        cols = np.zeros((N, C_in * K * K, L), dtype=np.float32)
        for i in range(H_out):
            for j in range(W_out):
                h0, w0 = i * S, j * S
                # (N, C_in, K, K) → (N, C_in*K*K)
                cols[:, :, i * W_out + j] = (
                    x_padded[:, :, h0:h0+K, w0:w0+K].reshape(N, -1)
                )
        # weight_flat: (C_out, C_in * K * K)
        weight_flat = self.weight.data.reshape(C_out, -1)
        # out_flat: (N, C_out, L)  via broadcast matmul (C_out, K2) @ (N, K2, L)
        out_flat = weight_flat @ cols
        out_data = out_flat.reshape(N, C_out, H_out, W_out)
        if self.bias is not None:
            out_data = out_data + self.bias.data.reshape(1, C_out, 1, 1)

        out = Tensor(out_data.astype(np.float32))
        bias_t = self.bias
        if bias_t is not None:
            parents = (x, self.weight, bias_t)
        else:
            parents = (x, self.weight)
        cols_captured = cols
        weight_flat_captured = weight_flat
        weight_shape = self.weight.data.shape
        H_in, W_in = H, W
        in_padded_shape = x_padded.shape

        def backward(g):
            grad_out = g.data  # (N, C_out, H_out, W_out)
            grad_out_flat = grad_out.reshape(N, C_out, L)
            # grad_w_flat: (C_out, C_in*K*K) = sum over batch of (grad_out_flat[n] @ cols[n].T)
            # Using broadcast: (N, C_out, L) @ (N, L, K2) → sum over N → (C_out, K2)
            grad_w_flat = (grad_out_flat @ np.swapaxes(cols_captured, -1, -2)).sum(axis=0)
            grad_w = grad_w_flat.reshape(weight_shape)
            # grad_cols: (N, C_in*K*K, L) = weight_flat.T @ grad_out_flat
            grad_cols = np.swapaxes(weight_flat_captured, -1, -2) @ grad_out_flat
            # col2im: scatter each column back to its (h0:h0+K, w0:w0+K) window
            # with += accumulation.
            grad_x_padded = np.zeros(in_padded_shape, dtype=np.float32)
            for i in range(H_out):
                for j in range(W_out):
                    h0, w0 = i * S, j * S
                    grad_x_padded[:, :, h0:h0+K, w0:w0+K] += (
                        grad_cols[:, :, i * W_out + j].reshape(N, C_in, K, K)
                    )
            if P > 0:
                grad_x = grad_x_padded[:, :, P:P+H_in, P:P+W_in].copy()
            else:
                grad_x = grad_x_padded
            if bias_t is not None:
                grad_b = grad_out.sum(axis=(0, 2, 3))
                return (grad_x, grad_w, grad_b)
            return (grad_x, grad_w)

        return _build_ctx(out, parents, backward)

    def __repr__(self):
        return (
            f"Conv2d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
        )


# ─── BatchNorm ─────────────────────────────────────────────

class BatchNorm2d(Module):
    """2D batch normalization. Statistics computed per-channel over (N, H, W).

    See PyTorch's docs for the formula. v0 implements:
      - train-mode forward (batch statistics)
      - affine + non-affine variants
      - running mean/var update via momentum
      - eval-mode forward (running statistics)
      - full backward (fused formula like LayerNorm)
    Each capability lands in its own TDD cycle.
    """

    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True,
                 track_running_stats: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.weight = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
            self.bias = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        else:
            self.weight = None
            self.bias = None
        # Buffers (not parameters — no requires_grad). Stored as plain numpy
        # arrays so they don't show up in .parameters().
        if track_running_stats:
            self.running_mean = np.zeros(num_features, dtype=np.float32)
            self.running_var = np.ones(num_features, dtype=np.float32)
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x: Tensor) -> Tensor:
        xd = x.data
        N, C, H, W = xd.shape
        is_training_batch = self.training or not self.track_running_stats
        if is_training_batch:
            mean = xd.mean(axis=(0, 2, 3))
            var = xd.var(axis=(0, 2, 3))
            if self.training and self.track_running_stats:
                m = self.momentum
                self.running_mean = (1.0 - m) * self.running_mean + m * mean
                self.running_var = (1.0 - m) * self.running_var + m * var
        else:
            mean = self.running_mean
            var = self.running_var
        inv_std = 1.0 / np.sqrt(var + self.eps)
        mean_4d = mean.reshape(1, C, 1, 1)
        inv_std_4d = inv_std.reshape(1, C, 1, 1)
        x_hat = (xd - mean_4d) * inv_std_4d
        if self.affine:
            out_data = (
                x_hat * self.weight.data.reshape(1, C, 1, 1)
                + self.bias.data.reshape(1, C, 1, 1)
            )
        else:
            out_data = x_hat

        out = Tensor(out_data.astype(np.float32))

        # Build autograd context.
        # The fused BN backward formula assumes we're in train mode (batch
        # stats). In eval mode, mean/var are constants — backward only flows
        # through the affine path. We handle both.
        if self.affine:
            parents = (x, self.weight, self.bias)
        else:
            parents = (x,)
        affine_capture = self.affine
        weight_data = self.weight.data if self.affine else None
        N_total = float(N * H * W)
        x_hat_captured = x_hat
        inv_std_captured = inv_std
        training_pass = is_training_batch

        def backward(g):
            grad_out = g.data  # (N, C, H, W)
            if affine_capture:
                grad_x_hat = grad_out * weight_data.reshape(1, C, 1, 1)
                grad_weight = (grad_out * x_hat_captured).sum(axis=(0, 2, 3))
                grad_bias = grad_out.sum(axis=(0, 2, 3))
            else:
                grad_x_hat = grad_out
                grad_weight = None
                grad_bias = None
            if training_pass:
                # Fused batch-statistics backward (standard formula).
                sum_g = grad_x_hat.sum(axis=(0, 2, 3), keepdims=True)
                sum_g_xhat = (grad_x_hat * x_hat_captured).sum(axis=(0, 2, 3), keepdims=True)
                grad_x = inv_std_captured.reshape(1, C, 1, 1) * (
                    grad_x_hat
                    - sum_g / N_total
                    - x_hat_captured * sum_g_xhat / N_total
                )
            else:
                # In eval, mean/var are constants → grad_x = grad_x_hat * inv_std.
                grad_x = grad_x_hat * inv_std_captured.reshape(1, C, 1, 1)
            if affine_capture:
                return (grad_x, grad_weight, grad_bias)
            return (grad_x,)

        return _build_ctx(out, parents, backward)

    def __repr__(self):
        return (
            f"BatchNorm2d({self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.affine})"
        )


# ─── Conv1d ────────────────────────────────────────────────

class Conv1d(Module):
    """1D convolution. PyTorch-conformant correlation (no kernel flip).

    Forward shape:
        input:  (N, C_in, L)
        weight: (C_out, C_in, kernel_size)
        bias:   (C_out,)
        output: (N, C_out, L_out)
            L_out = (L + 2*padding - kernel_size) // stride + 1
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * kernel_size
        bound = 1.0 / math.sqrt(fan_in)
        rng = np.random.default_rng()
        W = rng.uniform(
            -bound, bound,
            size=(out_channels, in_channels, kernel_size),
        ).astype(np.float32)
        self.weight = Tensor(W, requires_grad=True)
        if bias:
            self.bias = Tensor(np.zeros(out_channels, dtype=np.float32), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        N, C_in, L = x.data.shape
        K, S, P = self.kernel_size, self.stride, self.padding
        C_out = self.out_channels
        if P > 0:
            x_padded = np.pad(x.data, ((0, 0), (0, 0), (P, P)), mode="constant")
        else:
            x_padded = x.data
        L_pad = L + 2 * P
        L_out = (L_pad - K) // S + 1
        out_data = np.zeros((N, C_out, L_out), dtype=np.float32)
        for n in range(N):
            for co in range(C_out):
                for i in range(L_out):
                    l0 = i * S
                    out_data[n, co, i] = (
                        self.weight.data[co] * x_padded[n, :, l0:l0+K]
                    ).sum()
        if self.bias is not None:
            out_data = out_data + self.bias.data.reshape(1, C_out, 1)

        out = Tensor(out_data)
        bias_t = self.bias
        if bias_t is not None:
            parents = (x, self.weight, bias_t)
        else:
            parents = (x, self.weight)
        x_padded_captured = x_padded
        weight_captured = self.weight.data
        weight_shape = self.weight.data.shape
        L_in = L

        def backward(g):
            grad_out = g.data
            grad_w = np.zeros(weight_shape, dtype=np.float32)
            grad_x_padded = np.zeros_like(x_padded_captured)
            for nn_ in range(N):
                for co in range(C_out):
                    for i in range(L_out):
                        l0 = i * S
                        go = grad_out[nn_, co, i]
                        grad_w[co] += go * x_padded_captured[nn_, :, l0:l0+K]
                        grad_x_padded[nn_, :, l0:l0+K] += go * weight_captured[co]
            grad_x = grad_x_padded[:, :, P:P+L_in].copy() if P > 0 else grad_x_padded
            if bias_t is not None:
                grad_b = grad_out.sum(axis=(0, 2))
                return (grad_x, grad_w, grad_b)
            return (grad_x, grad_w)

        return _build_ctx(out, parents, backward)

    def __repr__(self):
        return (
            f"Conv1d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
        )


# ─── BatchNorm1d ───────────────────────────────────────────

class BatchNorm1d(Module):
    """1D batch normalization. Accepts (N, C) or (N, C, L) input."""

    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True,
                 track_running_stats: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.weight = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
            self.bias = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        else:
            self.weight = None
            self.bias = None
        if track_running_stats:
            self.running_mean = np.zeros(num_features, dtype=np.float32)
            self.running_var = np.ones(num_features, dtype=np.float32)
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x: Tensor) -> Tensor:
        xd = x.data
        C = self.num_features
        if xd.ndim not in (2, 3):
            raise ValueError(f"BatchNorm1d expects 2D (N, C) or 3D (N, C, L); got {xd.ndim}D")
        # Reduction axes: (0,) for 2D, (0, 2) for 3D — everything except channel
        if xd.ndim == 2:
            reduce_axes = (0,)
            stat_shape = (1, C)
        else:
            reduce_axes = (0, 2)
            stat_shape = (1, C, 1)
        is_training_batch = self.training or not self.track_running_stats
        if is_training_batch:
            mean = xd.mean(axis=reduce_axes)
            var = xd.var(axis=reduce_axes)
            if self.training and self.track_running_stats:
                m = self.momentum
                self.running_mean = (1.0 - m) * self.running_mean + m * mean
                self.running_var = (1.0 - m) * self.running_var + m * var
        else:
            mean = self.running_mean
            var = self.running_var
        inv_std = 1.0 / np.sqrt(var + self.eps)
        mean_b = mean.reshape(stat_shape)
        inv_std_b = inv_std.reshape(stat_shape)
        x_hat = (xd - mean_b) * inv_std_b
        if self.affine:
            out_data = x_hat * self.weight.data.reshape(stat_shape) + self.bias.data.reshape(stat_shape)
        else:
            out_data = x_hat

        out = Tensor(out_data.astype(np.float32))
        if self.affine:
            parents = (x, self.weight, self.bias)
        else:
            parents = (x,)
        affine_capture = self.affine
        weight_data = self.weight.data if self.affine else None
        N_total = float(np.prod([xd.shape[a] for a in reduce_axes]))
        x_hat_cap = x_hat
        inv_std_cap = inv_std
        training_pass = is_training_batch

        def backward(g):
            grad_out = g.data
            if affine_capture:
                grad_x_hat = grad_out * weight_data.reshape(stat_shape)
                grad_weight = (grad_out * x_hat_cap).sum(axis=reduce_axes)
                grad_bias = grad_out.sum(axis=reduce_axes)
            else:
                grad_x_hat = grad_out
                grad_weight = None
                grad_bias = None
            if training_pass:
                sum_g = grad_x_hat.sum(axis=reduce_axes, keepdims=True)
                sum_g_xhat = (grad_x_hat * x_hat_cap).sum(axis=reduce_axes, keepdims=True)
                grad_x = inv_std_cap.reshape(stat_shape) * (
                    grad_x_hat - sum_g / N_total - x_hat_cap * sum_g_xhat / N_total
                )
            else:
                grad_x = grad_x_hat * inv_std_cap.reshape(stat_shape)
            if affine_capture:
                return (grad_x, grad_weight, grad_bias)
            return (grad_x,)

        return _build_ctx(out, parents, backward)

    def __repr__(self):
        return f"BatchNorm1d({self.num_features}, eps={self.eps}, momentum={self.momentum})"


# ─── Flatten ───────────────────────────────────────────────

class Flatten(Module):
    """Flattens dims from start_dim to end_dim (inclusive) into a single dim.

    Defaults match torch.nn.Flatten: start_dim=1, end_dim=-1 → preserves
    batch dim, collapses everything else.
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        shape = x.data.shape
        ndim = len(shape)
        sd = self.start_dim % ndim
        ed = self.end_dim % ndim
        # New shape: dims [0:sd] + [prod(dims[sd:ed+1])] + dims[ed+1:]
        prefix = shape[:sd]
        middle = 1
        for d in shape[sd:ed+1]:
            middle *= d
        suffix = shape[ed+1:]
        new_shape = prefix + (middle,) + suffix
        return x.reshape(new_shape)

    def __repr__(self):
        return f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"


# ─── Pooling ───────────────────────────────────────────────

class MaxPool2d(Module):
    """2D max pooling. Stride defaults to kernel_size (PyTorch convention)."""

    def __init__(self, kernel_size: int, stride=None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        # Naive windowed max. Handles any kernel_size, stride, padding=0.
        # Records argmax position per output cell for the backward pass.
        N, C, H, W = x.data.shape
        K, S = self.kernel_size, self.stride
        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1
        out_data = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        # argmax stores the (kh, kw) offset inside each window — int8 is enough
        # for K < 128 which covers any sane pooling layer.
        argmax_kh = np.zeros((N, C, H_out, W_out), dtype=np.int32)
        argmax_kw = np.zeros((N, C, H_out, W_out), dtype=np.int32)
        for i in range(H_out):
            for j in range(W_out):
                h0, w0 = i * S, j * S
                window = x.data[:, :, h0:h0+K, w0:w0+K]      # (N, C, K, K)
                flat = window.reshape(N, C, K * K)
                flat_idx = flat.argmax(axis=2)               # (N, C)
                out_data[:, :, i, j] = np.take_along_axis(
                    flat, flat_idx[..., None], axis=2
                ).squeeze(2)
                argmax_kh[:, :, i, j] = flat_idx // K
                argmax_kw[:, :, i, j] = flat_idx % K

        out = Tensor(out_data)
        in_shape = x.data.shape

        def backward(g):
            grad_x = np.zeros(in_shape, dtype=np.float32)
            for i in range(H_out):
                for j in range(W_out):
                    h0, w0 = i * S, j * S
                    # For each (n, c), the winning input position is
                    # (h0 + argmax_kh[n,c,i,j], w0 + argmax_kw[n,c,i,j]).
                    for n in range(N):
                        for c in range(C):
                            kh = int(argmax_kh[n, c, i, j])
                            kw = int(argmax_kw[n, c, i, j])
                            grad_x[n, c, h0 + kh, w0 + kw] += g.data[n, c, i, j]
            return (grad_x,)

        return _build_ctx(out, (x,), backward)

    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class AvgPool2d(Module):
    """2D average pooling. Stride defaults to kernel_size."""

    def __init__(self, kernel_size: int, stride=None, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.data.shape
        K, S = self.kernel_size, self.stride
        H_out = (H - K) // S + 1
        W_out = (W - K) // S + 1
        out_data = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        for i in range(H_out):
            for j in range(W_out):
                h0, w0 = i * S, j * S
                out_data[:, :, i, j] = x.data[:, :, h0:h0+K, w0:w0+K].mean(axis=(2, 3))

        out = Tensor(out_data)
        in_shape = x.data.shape
        inv_window = 1.0 / float(K * K)

        def backward(g):
            grad_x = np.zeros(in_shape, dtype=np.float32)
            for i in range(H_out):
                for j in range(W_out):
                    h0, w0 = i * S, j * S
                    # g[n,c,i,j] / K² broadcast over the K×K window.
                    grad_x[:, :, h0:h0+K, w0:w0+K] += (
                        g.data[:, :, i, j][:, :, None, None] * inv_window
                    )
            return (grad_x,)

        return _build_ctx(out, (x,), backward)

    def __repr__(self):
        return f"AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class AdaptiveAvgPool2d(Module):
    """Adaptive 2D average pooling matching PyTorch's formula.

    For target output (H_out, W_out) and input (H_in, W_in):
      start_a(i) = floor(i * dim_in / dim_out)
      end_a(i)   = ceil((i + 1) * dim_in / dim_out)
    Each output cell averages its variable-sized bin.

    output_size: int (square) or (H_out, W_out) tuple.
    """

    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = tuple(output_size)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H_in, W_in = x.data.shape
        H_out, W_out = self.output_size
        # Pre-compute bin boundaries (avoids repeated division in tight loop).
        h_starts = [(i * H_in) // H_out for i in range(H_out)]
        h_ends = [-(-((i + 1) * H_in) // H_out) for i in range(H_out)]
        w_starts = [(j * W_in) // W_out for j in range(W_out)]
        w_ends = [-(-((j + 1) * W_in) // W_out) for j in range(W_out)]

        out_data = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        for i in range(H_out):
            for j in range(W_out):
                out_data[:, :, i, j] = x.data[:, :, h_starts[i]:h_ends[i], w_starts[j]:w_ends[j]].mean(axis=(2, 3))

        out = Tensor(out_data)
        in_shape = x.data.shape

        def backward(g):
            grad_x = np.zeros(in_shape, dtype=np.float32)
            for i in range(H_out):
                for j in range(W_out):
                    sh, eh = h_starts[i], h_ends[i]
                    sw, ew = w_starts[j], w_ends[j]
                    area = float((eh - sh) * (ew - sw))
                    grad_x[:, :, sh:eh, sw:ew] += (
                        g.data[:, :, i, j][:, :, None, None] / area
                    )
            return (grad_x,)

        return _build_ctx(out, (x,), backward)

    def __repr__(self):
        return f"AdaptiveAvgPool2d(output_size={self.output_size})"


# ─── Sequential ────────────────────────────────────────────

class Sequential(Module):
    """Compose modules in sequence: out = mₙ(...m₂(m₁(x))...)."""

    def __init__(self, *modules: Module):
        super().__init__()
        for i, m in enumerate(modules):
            setattr(self, f"_seq_{i}", m)
        self._n = len(modules)

    def forward(self, x):
        for i in range(self._n):
            x = getattr(self, f"_seq_{i}")(x)
        return x

    def __repr__(self):
        parts = [repr(getattr(self, f"_seq_{i}")) for i in range(self._n)]
        return "Sequential(\\n  " + ",\\n  ".join(parts) + "\\n)"


# ─── LayerNorm ─────────────────────────────────────────────

class LayerNorm(Module):
    """Layer normalization over the last D dimensions.

    For 2D inputs (batch, features), `normalized_shape` should be a single
    int = features. The forward computes (x - mean) / sqrt(var + eps) along
    the last axis, then applies elementwise gamma and beta.
    """

    def __init__(self, normalized_shape, eps: float = 1e-5,
                 elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Tensor(np.ones(self.normalized_shape, dtype=np.float32), requires_grad=True)
            self.bias = Tensor(np.zeros(self.normalized_shape, dtype=np.float32), requires_grad=True)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # We hand-write the autograd for LayerNorm rather than chaining ops —
        # the standard formula is well-known and avoids accumulating numerical
        # noise through mean→var→sqrt→divide.
        nd = len(self.normalized_shape)
        if x.data.shape[-nd:] != self.normalized_shape:
            raise ValueError(
                f"LayerNorm: last {nd} dims of input shape {x.data.shape} "
                f"must equal normalized_shape {self.normalized_shape}"
            )
        axes = tuple(range(x.data.ndim - nd, x.data.ndim))
        xd = x.data
        mean = xd.mean(axis=axes, keepdims=True)
        centered = xd - mean
        var = (centered * centered).mean(axis=axes, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.eps)
        normed = centered * inv_std
        if self.weight is not None:
            out_data = normed * self.weight.data + self.bias.data
            parents: Tuple[Tensor, ...] = (x, self.weight, self.bias)
        else:
            out_data = normed
            parents = (x,)
        out = Tensor(out_data)

        N = float(np.prod([xd.shape[a] for a in axes]))
        weight_data = self.weight.data if self.weight is not None else None

        def backward(g):
            gd = g.data
            if weight_data is not None:
                g_normed = gd * weight_data
                gW = (gd * normed).sum(axis=tuple(range(gd.ndim - nd)), keepdims=False)
                gB = gd.sum(axis=tuple(range(gd.ndim - nd)), keepdims=False)
            else:
                g_normed = gd
                gW = None
                gB = None
            # Standard layernorm backward:
            #   dx = (1/N) * inv_std * (N*g_normed - sum(g_normed) - normed*sum(g_normed*normed))
            sum_g = g_normed.sum(axis=axes, keepdims=True)
            sum_g_normed = (g_normed * normed).sum(axis=axes, keepdims=True)
            dx = inv_std * (g_normed - sum_g / N - normed * sum_g_normed / N)
            if weight_data is not None:
                return (dx, gW, gB)
            return (dx,)

        return _build_ctx(out, parents, backward)

    def __repr__(self):
        return f"LayerNorm({self.normalized_shape}, eps={self.eps}, affine={self.elementwise_affine})"


# ─── Embedding ─────────────────────────────────────────────

class Embedding(Module):
    """Lookup table: indices → embedded vectors.

    `forward(indices)` accepts an integer numpy array (or list) of any shape;
    output shape is `indices.shape + (embedding_dim,)`. Gradient w.r.t. the
    weight is a scatter-add — non-indexed rows stay at zero.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Init U(-1/sqrt(D), 1/sqrt(D)) — standard for embeddings.
        bound = 1.0 / math.sqrt(embedding_dim)
        rng = np.random.default_rng()
        W = rng.uniform(-bound, bound, size=(num_embeddings, embedding_dim)).astype(np.float32)
        self.weight = Tensor(W, requires_grad=True)

    def forward(self, indices) -> Tensor:
        if isinstance(indices, Tensor):
            idx = indices.data.astype(np.int64)
        else:
            idx = np.asarray(indices, dtype=np.int64)
        if (idx < 0).any() or (idx >= self.num_embeddings).any():
            raise ValueError(
                f"Embedding: indices out of range [0, {self.num_embeddings})"
            )
        out_data = self.weight.data[idx]
        out = Tensor(out_data)
        weight_shape = self.weight.data.shape
        D = self.embedding_dim
        def backward(g):
            gw = np.zeros(weight_shape, dtype=np.float32)
            flat_idx = idx.flatten()
            flat_grad = g.data.reshape(-1, D)
            np.add.at(gw, flat_idx, flat_grad)
            return (gw,)
        return _build_ctx(out, (self.weight,), backward)

    def __repr__(self):
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"


# ─── Activation wrappers (for Sequential composition) ──────

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)
    def __repr__(self):
        return "ReLU()"


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, self.negative_slope)
    def __repr__(self):
        return f"LeakyReLU(negative_slope={self.negative_slope})"


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)
    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x)
    def __repr__(self):
        return "Tanh()"


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x)
    def __repr__(self):
        return "GELU()"


# ─── Loss modules (callable wrappers around functional losses) ─────

class CrossEntropyLoss(Module):
    """Module form of cross_entropy_loss. Matches torch.nn.CrossEntropyLoss."""
    def forward(self, logits: Tensor, targets) -> Tensor:
        return F.cross_entropy_loss(logits, targets)
    def __repr__(self):
        return "CrossEntropyLoss()"


class MSELoss(Module):
    """Module form of mse_loss. Matches torch.nn.MSELoss."""
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target)
    def __repr__(self):
        return "MSELoss()"


class NLLLoss(Module):
    """Module form of nll_loss. Matches torch.nn.NLLLoss."""
    def forward(self, log_probs: Tensor, targets) -> Tensor:
        return F.nll_loss(log_probs, targets)
    def __repr__(self):
        return "NLLLoss()"


class BCEWithLogitsLoss(Module):
    """Module form of binary cross-entropy from logits. Matches
    torch.nn.BCEWithLogitsLoss. Uses the stable formula
      max(x, 0) - x*y + log(1 + exp(-|x|))
    so very large/small logits don't overflow.
    """
    def forward(self, logits: Tensor, targets) -> Tensor:
        return F.bce_with_logits_loss(logits, targets)
    def __repr__(self):
        return "BCEWithLogitsLoss()"


class BCELoss(Module):
    """Binary cross-entropy from probabilities. Matches torch.nn.BCELoss.
    Use BCEWithLogitsLoss for raw logits — it's numerically stable.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.bce_loss(input, target, reduction=self.reduction)
    def __repr__(self):
        return f"BCELoss(reduction={self.reduction!r})"


class L1Loss(Module):
    """Mean absolute error. Matches torch.nn.L1Loss."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target, reduction=self.reduction)
    def __repr__(self):
        return f"L1Loss(reduction={self.reduction!r})"


class SmoothL1Loss(Module):
    """Smooth L1 (Huber with knee at beta). Matches torch.nn.SmoothL1Loss."""
    def __init__(self, beta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.smooth_l1_loss(input, target, beta=self.beta, reduction=self.reduction)
    def __repr__(self):
        return f"SmoothL1Loss(beta={self.beta}, reduction={self.reduction!r})"


class KLDivLoss(Module):
    """KL divergence loss. Input is log-prob (e.g. output of log_softmax).

    Note: PyTorch's KLDivLoss default reduction is 'mean' (over total elements)
    but recommends 'batchmean' for the mathematically conventional definition.
    We default to 'mean' to match PyTorch.
    """
    def __init__(self, reduction: str = "mean", log_target: bool = False):
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.kl_div_loss(input, target, reduction=self.reduction, log_target=self.log_target)
    def __repr__(self):
        return f"KLDivLoss(reduction={self.reduction!r}, log_target={self.log_target})"


# The MultiheadAttention lowercase alias is set at the BOTTOM of this module,
# after MultiHeadAttention is defined.


class Dropout(Module):
    """Inverted dropout (matches torch.nn.Dropout).

    train mode: each element kept with probability (1-p), kept values scaled
                by 1/(1-p) so E[y] = x.
    eval mode:  identity (no scaling, no zeroing).
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout: p must be in [0, 1), got {p}")
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = (np.random.rand(*x.data.shape) < keep).astype(np.float32) / keep
        out = Tensor((x.data * mask).astype(np.float32))

        def backward(g):
            return (g.data * mask,)

        return _build_ctx(out, (x,), backward)

    def __repr__(self):
        return f"Dropout(p={self.p})"


class MultiHeadAttention(Module):
    """Batch-first multi-head scaled dot-product attention.

    Inputs Q, K, V each of shape (N, S, embed_dim); output (N, S, embed_dim).
    `embed_dim` must be divisible by `num_heads`. No mask support in v0.

    Composes existing differentiable ops (Linear, matmul, softmax, transpose,
    reshape) so autograd works automatically — no hand-written backward.
    """

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"MultiHeadAttention: embed_dim {embed_dim} not divisible "
                f"by num_heads {num_heads}"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self._scale = float(1.0 / np.sqrt(self.head_dim))

    def _split_heads(self, x: Tensor, N: int, S: int) -> Tensor:
        # (N, S, D) → (N, S, H, d_k) → (N, H, S, d_k)
        return x.reshape(N, S, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor, N: int, S: int) -> Tensor:
        # (N, H, S, d_k) → (N, S, H, d_k) → (N, S, D)
        return x.transpose(1, 2).reshape(N, S, self.embed_dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        if query.data.ndim != 3:
            raise ValueError(
                f"MultiHeadAttention expects 3D query (N, S, D); got {query.data.ndim}D"
            )
        N, S, D = query.data.shape
        if D != self.embed_dim:
            raise ValueError(
                f"MultiHeadAttention: query last dim {D} ≠ embed_dim {self.embed_dim}"
            )
        q = self._split_heads(self.q_proj(query), N, query.data.shape[1])
        k = self._split_heads(self.k_proj(key), N, key.data.shape[1])
        v = self._split_heads(self.v_proj(value), N, value.data.shape[1])
        # scores: (N, H, S_q, S_k)
        scores = (q @ k.transpose(-1, -2)) * self._scale
        weights = F.softmax(scores, dim=-1)
        attn = weights @ v  # (N, H, S_q, d_k)
        merged = self._merge_heads(attn, N, S)
        return self.out_proj(merged)

    def __repr__(self):
        return (
            f"MultiHeadAttention(embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads})"
        )


class Dropout2d(Module):
    """Channel-wise dropout for (N, C, H, W) inputs.

    Whole channels are zeroed independently with probability p. Inverted-
    dropout scaling (kept channels scaled by 1/(1-p)) so expected values match.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout2d: p must be in [0, 1), got {p}")
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        if x.data.ndim != 4:
            raise ValueError(f"Dropout2d expects 4D (N, C, H, W); got {x.data.ndim}D")
        N, C = x.data.shape[:2]
        keep = 1.0 - self.p
        channel_mask = (np.random.rand(N, C) < keep).astype(np.float32) / keep
        # Broadcast (N, C) → (N, C, 1, 1)
        mask = channel_mask.reshape(N, C, 1, 1)
        out = Tensor((x.data * mask).astype(np.float32))

        def backward(g):
            return (g.data * mask,)

        return _build_ctx(out, (x,), backward)

    def __repr__(self):
        return f"Dropout2d(p={self.p})"


# ─── More norms (Pile A #17) ───────────────────────────────
#
# Forward implemented with NumPy reductions; backward flows through the
# linear (gamma, beta) path only. Mean/var are treated as data-independent
# constants in backward — close enough for typical inference and most
# training use cases. A fully fused backward through statistics can be
# added later if a use case needs it.

class GroupNorm(Module):
    """Group normalization (Wu & He, 2018). Splits channels into groups
    and normalizes within each group.
    """
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5,
                 affine: bool = True):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(f"GroupNorm: num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = float(eps)
        self.affine = affine
        if affine:
            self.weight = Tensor(np.ones(num_channels, dtype=np.float32), requires_grad=True)
            self.bias = Tensor(np.zeros(num_channels, dtype=np.float32), requires_grad=True)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        xd = x.data
        N, C = xd.shape[0], xd.shape[1]
        G = self.num_groups
        spatial = xd.shape[2:]
        # Reshape to (N, G, C/G, *spatial), reduce over (C/G, *spatial).
        grouped = xd.reshape(N, G, C // G, *spatial)
        axes = tuple(range(2, grouped.ndim))
        mean = grouped.mean(axis=axes, keepdims=True)
        var = grouped.var(axis=axes, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = (grouped - mean) * inv_std
        out_data = x_hat.reshape(xd.shape)
        if self.affine:
            w_shape = (1, C) + tuple(1 for _ in spatial)
            out_data = out_data * self.weight.data.reshape(w_shape) + self.bias.data.reshape(w_shape)
        out = Tensor(out_data.astype(np.float32))
        # Backward: only through affine path (gamma, beta) and an approximate
        # x gradient that treats mean/var as constants.
        parents = (x, self.weight, self.bias) if self.affine else (x,)
        x_hat_flat = (out_data - (self.bias.data.reshape(w_shape) if self.affine else 0.0)) / (self.weight.data.reshape(w_shape) if self.affine else 1.0) if self.affine else out_data
        affine = self.affine
        weight_data = self.weight.data if self.affine else None

        def backward(g):
            dx_hat = g.data * weight_data.reshape(w_shape) if affine else g.data
            inv_std_broadcast = inv_std.reshape(N, G, 1, *(1 for _ in spatial))
            dx_grouped = dx_hat.reshape(N, G, C // G, *spatial) * inv_std_broadcast
            dx = dx_grouped.reshape(xd.shape).astype(np.float32)
            if not affine:
                return (dx,)
            reduce_axes = tuple(i for i in range(g.data.ndim) if i != 1)
            dgamma = (g.data * x_hat_flat).sum(axis=reduce_axes).astype(np.float32)
            dbeta = g.data.sum(axis=reduce_axes).astype(np.float32)
            return (dx, dgamma, dbeta)

        return _build_ctx(out, parents, backward)

    def __repr__(self):
        return f"GroupNorm(num_groups={self.num_groups}, num_channels={self.num_channels})"


class InstanceNorm2d(Module):
    """Instance norm = GroupNorm with num_groups=num_channels. No running stats
    (we don't track them — most uses don't need them in browser-style labs).
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = float(eps)
        self.affine = affine
        if affine:
            self.weight = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
            self.bias = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        xd = x.data
        N, C, H, W = xd.shape
        mean = xd.mean(axis=(2, 3), keepdims=True)
        var = xd.var(axis=(2, 3), keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = (xd - mean) * inv_std
        if self.affine:
            out_data = x_hat * self.weight.data.reshape(1, C, 1, 1) + self.bias.data.reshape(1, C, 1, 1)
        else:
            out_data = x_hat
        out = Tensor(out_data.astype(np.float32))
        # Approximate backward through affine + scale (mean/var treated as constants).
        parents = (x, self.weight, self.bias) if self.affine else (x,)
        affine = self.affine
        weight_data = self.weight.data if self.affine else None

        def backward(g):
            dx_hat = g.data * weight_data.reshape(1, C, 1, 1) if affine else g.data
            dx = (dx_hat * inv_std).astype(np.float32)
            if not affine:
                return (dx,)
            dgamma = (g.data * x_hat).sum(axis=(0, 2, 3)).astype(np.float32)
            dbeta = g.data.sum(axis=(0, 2, 3)).astype(np.float32)
            return (dx, dgamma, dbeta)
        return _build_ctx(out, parents, backward)

    def __repr__(self):
        return f"InstanceNorm2d({self.num_features}, affine={self.affine})"


class BatchNorm3d(Module):
    """3D batch norm — per-channel stats across (N, D, H, W)."""
    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True,
                 track_running_stats: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.weight = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
            self.bias = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        else:
            self.weight = None
            self.bias = None
        if track_running_stats:
            self.running_mean = np.zeros(num_features, dtype=np.float32)
            self.running_var = np.ones(num_features, dtype=np.float32)
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x: Tensor) -> Tensor:
        xd = x.data
        N, C, D, H, W = xd.shape
        is_training_batch = self.training or not self.track_running_stats
        if is_training_batch:
            mean = xd.mean(axis=(0, 2, 3, 4))
            var = xd.var(axis=(0, 2, 3, 4))
            if self.training and self.track_running_stats:
                m = self.momentum
                self.running_mean = (1.0 - m) * self.running_mean + m * mean
                self.running_var = (1.0 - m) * self.running_var + m * var
        else:
            mean = self.running_mean
            var = self.running_var
        inv_std = 1.0 / np.sqrt(var + self.eps)
        bshape = (1, C, 1, 1, 1)
        x_hat = (xd - mean.reshape(bshape)) * inv_std.reshape(bshape)
        if self.affine:
            out_data = x_hat * self.weight.data.reshape(bshape) + self.bias.data.reshape(bshape)
        else:
            out_data = x_hat
        out = Tensor(out_data.astype(np.float32))
        parents = (x, self.weight, self.bias) if self.affine else (x,)
        affine = self.affine
        weight_data = self.weight.data if self.affine else None

        def backward(g):
            dx_hat = g.data * weight_data.reshape(bshape) if affine else g.data
            dx = (dx_hat * inv_std.reshape(bshape)).astype(np.float32)
            if not affine:
                return (dx,)
            dgamma = (g.data * x_hat).sum(axis=(0, 2, 3, 4)).astype(np.float32)
            dbeta = g.data.sum(axis=(0, 2, 3, 4)).astype(np.float32)
            return (dx, dgamma, dbeta)
        return _build_ctx(out, parents, backward)

    def __repr__(self):
        return f"BatchNorm3d({self.num_features})"


# ─── Recurrent layers (Pile A #15) ─────────────────────────
#
# Forward passes unroll the recurrence step-by-step over time using existing
# autograd primitives. BPTT happens automatically via the autograd graph —
# we don't write a manual backward. The cost: O(T) graph nodes per sequence,
# which is fine for educational sequence lengths.
#
# Initialization matches torch.nn: uniform on [-1/sqrt(hidden), 1/sqrt(hidden)].

def _rnn_init_uniform(shape, hidden_size, dtype=np.float32):
    bound = 1.0 / math.sqrt(hidden_size)
    return np.random.uniform(-bound, bound, size=shape).astype(dtype)


def _activation(name, x: Tensor) -> Tensor:
    if name == "tanh":
        return F.tanh(x)
    if name == "relu":
        return F.relu(x)
    raise ValueError(f"RNN: unsupported nonlinearity {name!r}")


class RNN(Module):
    """Vanilla Elman RNN. Supports single layer, single direction, batch_first.
    Multi-layer / bidirectional are out of scope for this educational impl.

    Shape (batch_first=False):
      input: (T, B, input_size)
      output: (T, B, hidden_size), h_n: (1, B, hidden_size)
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 batch_first: bool = False, nonlinearity: str = "tanh"):
        super().__init__()
        if num_layers != 1:
            raise NotImplementedError("RNN: num_layers>1 not supported yet")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.nonlinearity = nonlinearity
        self.weight_ih_l0 = Tensor(_rnn_init_uniform((hidden_size, input_size), hidden_size), requires_grad=True)
        self.weight_hh_l0 = Tensor(_rnn_init_uniform((hidden_size, hidden_size), hidden_size), requires_grad=True)
        self.bias_ih_l0 = Tensor(_rnn_init_uniform((hidden_size,), hidden_size), requires_grad=True)
        self.bias_hh_l0 = Tensor(_rnn_init_uniform((hidden_size,), hidden_size), requires_grad=True)

    def forward(self, x: Tensor, h_0: Optional[Tensor] = None):
        if self.batch_first:
            # Move T to dim 0.
            x = x.permute(1, 0, 2)
        T, B, _ = x.data.shape
        if h_0 is None:
            h = Tensor(np.zeros((B, self.hidden_size), dtype=np.float32))
        else:
            h = h_0[0] if h_0.data.ndim == 3 else h_0
        outs = []
        W_ih_T = self.weight_ih_l0.transpose(0, 1)
        W_hh_T = self.weight_hh_l0.transpose(0, 1)
        for t in range(T):
            x_t = x[t]  # (B, I)
            pre = x_t @ W_ih_T + self.bias_ih_l0 + h @ W_hh_T + self.bias_hh_l0
            h = _activation(self.nonlinearity, pre)
            outs.append(h)
        out = stack(outs, dim=0)  # (T, B, H)
        h_n = h.reshape(1, B, self.hidden_size)
        if self.batch_first:
            out = out.permute(1, 0, 2)
        return out, h_n

    def __repr__(self):
        return f"RNN({self.input_size}, {self.hidden_size}, batch_first={self.batch_first})"


class LSTM(Module):
    """Single-layer LSTM (Hochreiter & Schmidhuber). Matches torch.nn.LSTM
    parameter layout: weight_ih is (4*hidden, input) — gate order i, f, g, o.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 batch_first: bool = False):
        super().__init__()
        if num_layers != 1:
            raise NotImplementedError("LSTM: num_layers>1 not supported yet")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.weight_ih_l0 = Tensor(_rnn_init_uniform((4 * hidden_size, input_size), hidden_size), requires_grad=True)
        self.weight_hh_l0 = Tensor(_rnn_init_uniform((4 * hidden_size, hidden_size), hidden_size), requires_grad=True)
        self.bias_ih_l0 = Tensor(_rnn_init_uniform((4 * hidden_size,), hidden_size), requires_grad=True)
        self.bias_hh_l0 = Tensor(_rnn_init_uniform((4 * hidden_size,), hidden_size), requires_grad=True)

    def forward(self, x: Tensor, hc_0=None):
        if self.batch_first:
            x = x.permute(1, 0, 2)
        T, B, _ = x.data.shape
        H = self.hidden_size
        if hc_0 is None:
            h = Tensor(np.zeros((B, H), dtype=np.float32))
            c = Tensor(np.zeros((B, H), dtype=np.float32))
        else:
            h, c = hc_0
            if h.data.ndim == 3: h = h[0]
            if c.data.ndim == 3: c = c[0]
        W_ih_T = self.weight_ih_l0.transpose(0, 1)
        W_hh_T = self.weight_hh_l0.transpose(0, 1)
        outs = []
        for t in range(T):
            x_t = x[t]
            gates = x_t @ W_ih_T + self.bias_ih_l0 + h @ W_hh_T + self.bias_hh_l0
            i_g = F.sigmoid(gates[:, 0:H])
            f_g = F.sigmoid(gates[:, H:2*H])
            g_g = F.tanh(gates[:, 2*H:3*H])
            o_g = F.sigmoid(gates[:, 3*H:4*H])
            c = f_g * c + i_g * g_g
            h = o_g * F.tanh(c)
            outs.append(h)
        out = stack(outs, dim=0)
        h_n = h.reshape(1, B, H)
        c_n = c.reshape(1, B, H)
        if self.batch_first:
            out = out.permute(1, 0, 2)
        return out, (h_n, c_n)

    def __repr__(self):
        return f"LSTM({self.input_size}, {self.hidden_size}, batch_first={self.batch_first})"


class GRU(Module):
    """Single-layer GRU (Cho et al.). Matches torch.nn.GRU parameter layout:
    weight_ih is (3*hidden, input) — gate order r, z, n.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 batch_first: bool = False):
        super().__init__()
        if num_layers != 1:
            raise NotImplementedError("GRU: num_layers>1 not supported yet")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.weight_ih_l0 = Tensor(_rnn_init_uniform((3 * hidden_size, input_size), hidden_size), requires_grad=True)
        self.weight_hh_l0 = Tensor(_rnn_init_uniform((3 * hidden_size, hidden_size), hidden_size), requires_grad=True)
        self.bias_ih_l0 = Tensor(_rnn_init_uniform((3 * hidden_size,), hidden_size), requires_grad=True)
        self.bias_hh_l0 = Tensor(_rnn_init_uniform((3 * hidden_size,), hidden_size), requires_grad=True)

    def forward(self, x: Tensor, h_0: Optional[Tensor] = None):
        if self.batch_first:
            x = x.permute(1, 0, 2)
        T, B, _ = x.data.shape
        H = self.hidden_size
        if h_0 is None:
            h = Tensor(np.zeros((B, H), dtype=np.float32))
        else:
            h = h_0[0] if h_0.data.ndim == 3 else h_0
        W_ih_T = self.weight_ih_l0.transpose(0, 1)
        W_hh_T = self.weight_hh_l0.transpose(0, 1)
        outs = []
        for t in range(T):
            x_t = x[t]
            ih = x_t @ W_ih_T + self.bias_ih_l0
            hh = h   @ W_hh_T + self.bias_hh_l0
            r = F.sigmoid(ih[:, 0:H]   + hh[:, 0:H])
            z = F.sigmoid(ih[:, H:2*H] + hh[:, H:2*H])
            n = F.tanh(ih[:, 2*H:3*H] + r * hh[:, 2*H:3*H])
            h = (1.0 - z) * n + z * h
            outs.append(h)
        out = stack(outs, dim=0)
        h_n = h.reshape(1, B, H)
        if self.batch_first:
            out = out.permute(1, 0, 2)
        return out, h_n

    def __repr__(self):
        return f"GRU({self.input_size}, {self.hidden_size}, batch_first={self.batch_first})"


# ─── PyTorch lowercase-h alias ─────────────────────────────
# Defined here at module bottom so MultiHeadAttention is in scope.
MultiheadAttention = MultiHeadAttention


# ─── nn.init namespace ─────────────────────────────────────
# In-place initializers matching torch.nn.init.*. They mutate the tensor's
# .data buffer directly — the tensor identity (and any registered-param
# wiring) is preserved.

import math as _bg_init_math
import types as _bg_init_types
init = _bg_init_types.ModuleType("browsergrad_grad.nn.init")

def _init_zeros_(t):
    t.data[...] = 0.0
    return t
def _init_ones_(t):
    t.data[...] = 1.0
    return t
def _init_uniform_(t, a=0.0, b=1.0):
    t.data[...] = np.random.uniform(a, b, size=t.data.shape).astype(np.float32)
    return t
def _init_normal_(t, mean=0.0, std=1.0):
    t.data[...] = np.random.normal(mean, std, size=t.data.shape).astype(np.float32)
    return t
def _init_constant_(t, val):
    t.data[...] = float(val)
    return t
def _init_kaiming_uniform_(t, a=0.0, mode="fan_in", nonlinearity="relu"):
    # Compute fan_in from the tensor shape. For Linear-like (out, in)
    # or Conv (out, in, *spatial), fan_in = in * prod(spatial).
    if t.data.ndim < 2:
        fan_in = t.data.shape[0] if t.data.ndim > 0 else 1
    else:
        fan_in = int(np.prod(t.data.shape[1:]))
    gain = _bg_init_math.sqrt(2.0 / (1.0 + a * a)) if nonlinearity == "leaky_relu" else _bg_init_math.sqrt(2.0)
    std = gain / _bg_init_math.sqrt(fan_in)
    bound = _bg_init_math.sqrt(3.0) * std
    t.data[...] = np.random.uniform(-bound, bound, size=t.data.shape).astype(np.float32)
    return t
def _init_xavier_uniform_(t, gain=1.0):
    if t.data.ndim < 2:
        fan_in = fan_out = t.data.shape[0] if t.data.ndim > 0 else 1
    else:
        fan_out = t.data.shape[0]
        fan_in = int(np.prod(t.data.shape[1:]))
    bound = gain * _bg_init_math.sqrt(6.0 / (fan_in + fan_out))
    t.data[...] = np.random.uniform(-bound, bound, size=t.data.shape).astype(np.float32)
    return t

init.zeros_ = _init_zeros_
init.ones_ = _init_ones_
init.uniform_ = _init_uniform_
init.normal_ = _init_normal_
init.constant_ = _init_constant_
init.kaiming_uniform_ = _init_kaiming_uniform_
init.xavier_uniform_ = _init_xavier_uniform_

# Register under sys.modules so 'from browsergrad_grad.nn import init'
# works AND 'import browsergrad_grad.nn.init' resolves.
import sys as _bg_init_sys
_bg_init_sys.modules["browsergrad_grad.nn.init"] = init
# Leave _bg_init_math / _bg_init_types / _bg_init_sys in module globals —
# the init functions reference them as free vars at call time. The single
# underscore prefix already keeps them out of star-imports.
