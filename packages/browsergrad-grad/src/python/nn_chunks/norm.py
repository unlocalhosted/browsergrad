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
                 momentum: Optional[float] = 0.1, affine: bool = True,
                 track_running_stats: bool = True):
        super().__init__()
        if type(num_features) is not int or type(num_features) is bool:
            raise TypeError("BatchNorm1d: num_features must be an exact integer")
        if num_features <= 0 or num_features > (1 << 28):
            raise ValueError("BatchNorm1d: num_features must be in [1, 2**28]")
        if type(eps) not in (int, float, np.int8, np.int16, np.int32,
                             np.int64, np.uint8, np.uint16, np.uint32,
                             np.uint64, np.float16, np.float32, np.float64) or type(eps) is bool:
            raise TypeError("BatchNorm1d: eps must be an exact real scalar")
        eps = float(eps)
        if not math.isfinite(eps) or eps < 0.0:
            raise ValueError("BatchNorm1d: eps must be finite and non-negative")
        if momentum is not None:
            if type(momentum) not in (int, float, np.int8, np.int16, np.int32,
                                      np.int64, np.uint8, np.uint16, np.uint32,
                                      np.uint64, np.float16, np.float32, np.float64) or type(momentum) is bool:
                raise TypeError(
                    "BatchNorm1d: momentum must be None or an exact real scalar"
                )
            momentum = float(momentum)
            if not math.isfinite(momentum) or momentum < 0.0 or momentum > 1.0:
                raise ValueError(
                    "BatchNorm1d: momentum must be None or finite and in [0, 1]"
                )
        if type(affine) is not bool:
            raise TypeError("BatchNorm1d: affine must be an exact bool")
        if type(track_running_stats) is not bool:
            raise TypeError(
                "BatchNorm1d: track_running_stats must be an exact bool"
            )
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.weight = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
            self.bias = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
        else:
            self.weight = None
            self.bias = None
        if track_running_stats:
            self.register_buffer(
                "running_mean",
                Tensor(np.zeros(num_features, dtype=np.float32)),
            )
            self.register_buffer(
                "running_var",
                Tensor(np.ones(num_features, dtype=np.float32)),
            )
            self.register_buffer(
                "num_batches_tracked",
                Tensor(np.asarray(0, dtype=np.int64)),
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def reset_running_stats(self) -> None:
        if not self.track_running_stats:
            return
        self.running_mean.data[...] = np.float32(0.0)
        self.running_var.data[...] = np.float32(1.0)
        self.num_batches_tracked.data[...] = np.int64(0)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            self.weight.data[...] = np.float32(1.0)
            self.bias.data[...] = np.float32(0.0)

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("BatchNorm1d input must be a Tensor")
        xd = x.data
        C = self.num_features
        if xd.ndim not in (2, 3):
            raise ValueError(f"BatchNorm1d expects 2D (N, C) or 3D (N, C, L); got {xd.ndim}D")
        if xd.shape[1] != C:
            raise ValueError(
                f"BatchNorm1d expected {C} channels, got {xd.shape[1]}"
            )
        if xd.dtype.name != "float32":
            raise TypeError(
                f"BatchNorm1d v1 requires float32 input, got {xd.dtype.name}"
            )
        # Reduction axes: (0,) for 2D, (0, 2) for 3D — everything except channel
        if xd.ndim == 2:
            reduce_axes = (0,)
            stat_shape = (1, C)
        else:
            reduce_axes = (0, 2)
            stat_shape = (1, C, 1)
        is_training_batch = self.training or not self.track_running_stats
        if is_training_batch:
            sample_count = int(np.prod([xd.shape[axis] for axis in reduce_axes]))
            if sample_count <= 1:
                raise ValueError(
                    "BatchNorm1d batch-statistics mode requires more than "
                    f"one value per channel, got {sample_count}"
                )
            mean = xd.mean(axis=reduce_axes, dtype=np.float32)
            centered = xd - mean.reshape(stat_shape)
            var = (centered * centered).mean(
                axis=reduce_axes, dtype=np.float32
            )
            if self.training and self.track_running_stats:
                tracked = int(self.num_batches_tracked.data)
                if tracked < 0 or tracked >= (1 << 63) - 1:
                    raise OverflowError(
                        "BatchNorm1d num_batches_tracked reached int64 limit"
                    )
                tracked += 1
                factor = (
                    1.0 / float(tracked)
                    if self.momentum is None
                    else self.momentum
                )
                unbiased_var = var * np.float32(
                    sample_count / float(sample_count - 1)
                )
                self.running_mean.data[...] = np.asarray(
                    (1.0 - factor) * self.running_mean.data + factor * mean,
                    dtype=np.float32,
                )
                self.running_var.data[...] = np.asarray(
                    (1.0 - factor) * self.running_var.data
                    + factor * unbiased_var,
                    dtype=np.float32,
                )
                self.num_batches_tracked.data[...] = np.int64(tracked)
        else:
            mean = self.running_mean.data
            var = self.running_var.data
            sample_count = int(np.prod([xd.shape[axis] for axis in reduce_axes]))
        inv_std = np.asarray(
            1.0 / np.sqrt(np.asarray(var + self.eps, dtype=np.float32)),
            dtype=np.float32,
        )
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
        weight_data = self.weight.data.copy() if self.affine else None
        N_total = float(sample_count)
        x_hat_cap = np.array(x_hat, dtype=np.float32, copy=True)
        inv_std_cap = np.array(inv_std, dtype=np.float32, copy=True)
        training_pass = is_training_batch

        def backward(g):
            grad_out = g.data
            if affine_capture:
                grad_x_hat = grad_out * weight_data.reshape(stat_shape)
                grad_weight = (grad_out * x_hat_cap).sum(
                    axis=reduce_axes, dtype=np.float32
                )
                grad_bias = grad_out.sum(axis=reduce_axes, dtype=np.float32)
            else:
                grad_x_hat = grad_out
                grad_weight = None
                grad_bias = None
            if training_pass:
                sum_g = grad_x_hat.sum(
                    axis=reduce_axes, keepdims=True, dtype=np.float32
                )
                sum_g_xhat = (grad_x_hat * x_hat_cap).sum(
                    axis=reduce_axes, keepdims=True, dtype=np.float32
                )
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
        return (
            f"BatchNorm1d({self.num_features}, eps={self.eps}, "
            f"momentum={self.momentum}, affine={self.affine}, "
            f"track_running_stats={self.track_running_stats})"
        )


# ─── Flatten ───────────────────────────────────────────────

class LayerNorm(Module):
    """Layer normalization over the last D dimensions.

    For 2D inputs (batch, features), `normalized_shape` should be a single
    int = features. The forward computes (x - mean) / sqrt(var + eps) along
    the last axis, then applies elementwise gamma and beta.
    """

    def __init__(self, normalized_shape, eps: float = 1e-5,
                 elementwise_affine: bool = True, device=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = elementwise_affine
        self.device = device
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
        if self.device is not None and axes != (x.data.ndim - 1,):
            raise NotImplementedError(
                "LayerNorm(device=...): KernelDevice bridge supports normalization over last dim only"
            )
        if self.device is not None:
            out_data = _device.layernorm(
                self.device,
                xd,
                self.weight.data if self.weight is not None else None,
                self.bias.data if self.weight is not None else None,
                self.eps,
            )
        elif self.weight is not None:
            out_data = normed * self.weight.data + self.bias.data
            parents: Tuple[Tensor, ...] = (x, self.weight, self.bias)
        else:
            out_data = normed
            parents = (x,)
        if self.device is not None and self.weight is not None:
            parents = (x, self.weight, self.bias)
        elif self.device is not None:
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
        parents = (x, self.weight, self.bias) if self.affine else (x,)
        affine = self.affine
        weight_data = self.weight.data if self.affine else None
        group_size = float((C // G) * np.prod(spatial))

        def backward(g):
            dx_hat = g.data * weight_data.reshape(w_shape) if affine else g.data
            dx_hat_grouped = dx_hat.reshape(N, G, C // G, *spatial)
            sum_dx_hat = dx_hat_grouped.sum(axis=axes, keepdims=True)
            sum_dx_hat_xhat = (dx_hat_grouped * x_hat).sum(axis=axes, keepdims=True)
            dx_grouped = inv_std * (
                dx_hat_grouped
                - sum_dx_hat / group_size
                - x_hat * sum_dx_hat_xhat / group_size
            )
            dx = dx_grouped.reshape(xd.shape).astype(np.float32)
            if not affine:
                return (dx,)
            reduce_axes = tuple(i for i in range(g.data.ndim) if i != 1)
            dgamma = (g.data * x_hat.reshape(xd.shape)).sum(axis=reduce_axes).astype(np.float32)
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
        parents = (x, self.weight, self.bias) if self.affine else (x,)
        affine = self.affine
        weight_data = self.weight.data if self.affine else None
        norm_size = float(H * W)

        def backward(g):
            dx_hat = g.data * weight_data.reshape(1, C, 1, 1) if affine else g.data
            sum_dx_hat = dx_hat.sum(axis=(2, 3), keepdims=True)
            sum_dx_hat_xhat = (dx_hat * x_hat).sum(axis=(2, 3), keepdims=True)
            dx = inv_std * (
                dx_hat
                - sum_dx_hat / norm_size
                - x_hat * sum_dx_hat_xhat / norm_size
            )
            dx = dx.astype(np.float32)
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
        N_total = float(N * D * H * W)
        training_pass = is_training_batch

        def backward(g):
            dx_hat = g.data * weight_data.reshape(bshape) if affine else g.data
            if training_pass:
                sum_dx_hat = dx_hat.sum(axis=(0, 2, 3, 4), keepdims=True)
                sum_dx_hat_xhat = (dx_hat * x_hat).sum(axis=(0, 2, 3, 4), keepdims=True)
                dx = inv_std.reshape(bshape) * (
                    dx_hat
                    - sum_dx_hat / N_total
                    - x_hat * sum_dx_hat_xhat / N_total
                )
            else:
                dx = dx_hat * inv_std.reshape(bshape)
            dx = dx.astype(np.float32)
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
