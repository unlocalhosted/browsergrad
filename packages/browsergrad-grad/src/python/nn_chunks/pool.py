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
