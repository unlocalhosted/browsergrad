/**
 * nn — neural network modules with parameter management.
 *
 * v0.2: Linear loses its bias-broadcasting workaround now that the Tensor
 * supports real broadcasting. Adds LayerNorm, Embedding, and Module wrappers
 * for common activations so they slot into Sequential cleanly.
 */

export const NN_PY = `
"""browsergrad_grad.nn — neural network building blocks."""

import math
import numpy as np
from typing import Iterator, Tuple
from .tensor import Tensor, _build_ctx
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

    def __setattr__(self, name, value):
        if isinstance(value, Tensor) and value.requires_grad:
            self.__dict__.setdefault("_parameters", {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        object.__setattr__(self, name, value)

    def parameters(self) -> Iterator[Tensor]:
        """Yield every parameter Tensor in this module and its submodules."""
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()"
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


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
        # Naive direct-loop correlation. Handles kernel_size, stride, padding.
        # Refactor to im2col after backward + end-to-end tests are green.
        N, C_in, H, W = x.data.shape
        K, S, P = self.kernel_size, self.stride, self.padding
        C_out = self.out_channels
        if P > 0:
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (P, P), (P, P)),
                mode="constant",
            )
        else:
            x_padded = x.data
        H_pad = H + 2 * P
        W_pad = W + 2 * P
        H_out = (H_pad - K) // S + 1
        W_out = (W_pad - K) // S + 1
        out_data = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
        for n in range(N):
            for co in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        h0, w0 = i * S, j * S
                        out_data[n, co, i, j] = (
                            self.weight.data[co] * x_padded[n, :, h0:h0+K, w0:w0+K]
                        ).sum()
        if self.bias is not None:
            out_data = out_data + self.bias.data.reshape(1, C_out, 1, 1)

        out = Tensor(out_data)
        bias_t = self.bias
        if bias_t is not None:
            parents = (x, self.weight, bias_t)
        else:
            parents = (x, self.weight)
        # Capture only what backward needs — keeps the closure cheap and
        # makes it independent of subsequent mutations to weight/x.
        x_padded_captured = x_padded
        weight_captured = self.weight.data
        weight_shape = self.weight.data.shape
        H_in, W_in = H, W

        def backward(g):
            grad_out = g.data  # (N, C_out, H_out, W_out)
            # d/d(weight): accumulate grad_out * input-window per output cell.
            grad_w = np.zeros(weight_shape, dtype=np.float32)
            # d/d(x_padded): scatter-add grad_out * weight back to each window.
            grad_x_padded = np.zeros_like(x_padded_captured)
            for nn_ in range(N):
                for co in range(C_out):
                    for i in range(H_out):
                        for j in range(W_out):
                            h0, w0 = i * S, j * S
                            go = grad_out[nn_, co, i, j]
                            grad_w[co] += go * x_padded_captured[nn_, :, h0:h0+K, w0:w0+K]
                            grad_x_padded[nn_, :, h0:h0+K, w0:w0+K] += go * weight_captured[co]
            # Crop back to the input shape (no padding) if padding was applied.
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

    For 2D inputs (batch, features), \`normalized_shape\` should be a single
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

    \`forward(indices)\` accepts an integer numpy array (or list) of any shape;
    output shape is \`indices.shape + (embedding_dim,)\`. Gradient w.r.t. the
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
`;
