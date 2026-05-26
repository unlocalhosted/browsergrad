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
