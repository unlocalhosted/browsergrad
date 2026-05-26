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
