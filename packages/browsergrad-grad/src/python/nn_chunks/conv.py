def _pair2(value, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"{name} must be an int or a length-2 tuple")


def _triple3(value, name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    raise ValueError(f"{name} must be an int or a length-3 tuple")


class Conv2d(Module):
    """2D convolution. PyTorch-style correlation, including tuple shapes,
    dilation, and grouped/depthwise layouts.

    Forward shape:
        input:  (N, C_in, H, W)
        weight: (C_out, C_in/groups, kernel_h, kernel_w)
        bias:   (C_out,)
        output: (N, C_out, H_out, W_out)
            H_out = (H + 2*pad_h - dilation_h*(kernel_h-1) - 1) // stride_h + 1
            W_out = (W + 2*pad_w - dilation_w*(kernel_w-1) - 1) // stride_w + 1
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, dilation=1, groups: int = 1,
                 bias: bool = True):
        super().__init__()
        if groups <= 0:
            raise ValueError("Conv2d: groups must be positive")
        if in_channels % groups != 0:
            raise ValueError("Conv2d: in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("Conv2d: out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair2(kernel_size, "kernel_size")
        self.stride = _pair2(stride, "stride")
        self.padding = _pair2(padding, "padding")
        self.dilation = _pair2(dilation, "dilation")
        self.groups = int(groups)
        kh, kw = self.kernel_size
        fan_in = (in_channels // groups) * kh * kw
        bound = 1.0 / math.sqrt(fan_in)
        W = np.random.uniform(
            -bound, bound,
            size=(out_channels, in_channels // groups, kh, kw),
        ).astype(np.float32)
        self.weight = Tensor(W, requires_grad=True)
        if bias:
            self.bias = Tensor(np.zeros(out_channels, dtype=np.float32), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # im2col-based correlation. Two outer loops over (i, j) gather each
        # receptive field into a column; grouped multiply-accumulate happens
        # via batched matmul. Dilation changes only the gather/scatter stride.
        N, C_in, H, W = x.data.shape
        if C_in != self.in_channels:
            raise ValueError(f"Conv2d: expected {self.in_channels} input channels, got {C_in}")
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation
        groups = self.groups
        c_per_group = C_in // groups
        out_per_group = self.out_channels // groups
        C_out = self.out_channels
        if ph > 0 or pw > 0:
            x_padded = np.pad(
                x.data, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant",
            )
        else:
            x_padded = x.data
        H_pad = H + 2 * ph
        W_pad = W + 2 * pw
        eff_h = dh * (kh - 1) + 1
        eff_w = dw * (kw - 1) + 1
        H_out = (H_pad - eff_h) // sh + 1
        W_out = (W_pad - eff_w) // sw + 1
        L = H_out * W_out
        # cols: (N, C_in * kh * kw, H_out * W_out)
        cols = np.zeros((N, C_in * kh * kw, L), dtype=np.float32)
        for i in range(H_out):
            for j in range(W_out):
                h0, w0 = i * sh, j * sw
                # (N, C_in, kh, kw) -> (N, C_in*kh*kw)
                cols[:, :, i * W_out + j] = (
                    x_padded[:, :, h0:h0+eff_h:dh, w0:w0+eff_w:dw].reshape(N, -1)
                )
        out_flat = np.zeros((N, C_out, L), dtype=np.float32)
        weight_flats = []
        col_slices = []
        for g in range(groups):
            c0 = g * c_per_group
            c1 = (g + 1) * c_per_group
            o0 = g * out_per_group
            o1 = (g + 1) * out_per_group
            col_slice = slice(c0 * kh * kw, c1 * kh * kw)
            weight_flat_g = self.weight.data[o0:o1].reshape(out_per_group, -1)
            out_flat[:, o0:o1, :] = weight_flat_g @ cols[:, col_slice, :]
            weight_flats.append(weight_flat_g.copy())
            col_slices.append((o0, o1, c0, c1, col_slice))
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
        weight_flats_captured = tuple(weight_flats)
        col_slices_captured = tuple(col_slices)
        weight_shape = self.weight.data.shape
        H_in, W_in = H, W
        in_padded_shape = x_padded.shape

        def backward(g):
            grad_out = g.data  # (N, C_out, H_out, W_out)
            grad_out_flat = grad_out.reshape(N, C_out, L)
            grad_w = np.zeros(weight_shape, dtype=np.float32)
            grad_cols = np.zeros_like(cols_captured)
            for idx, (o0, o1, _c0, _c1, col_slice) in enumerate(col_slices_captured):
                grad_out_g = grad_out_flat[:, o0:o1, :]
                cols_g = cols_captured[:, col_slice, :]
                grad_w_g = (grad_out_g @ np.swapaxes(cols_g, -1, -2)).sum(axis=0)
                grad_w[o0:o1] = grad_w_g.reshape(out_per_group, c_per_group, kh, kw)
                grad_cols[:, col_slice, :] = (
                    np.swapaxes(weight_flats_captured[idx], -1, -2) @ grad_out_g
                )
            # col2im: scatter each column back to its (h0:h0+K, w0:w0+K) window
            # with += accumulation.
            grad_x_padded = np.zeros(in_padded_shape, dtype=np.float32)
            for i in range(H_out):
                for j in range(W_out):
                    h0, w0 = i * sh, j * sw
                    grad_x_padded[:, :, h0:h0+eff_h:dh, w0:w0+eff_w:dw] += (
                        grad_cols[:, :, i * W_out + j].reshape(N, C_in, kh, kw)
                    )
            if ph > 0 or pw > 0:
                grad_x = grad_x_padded[:, :, ph:ph+H_in, pw:pw+W_in].copy()
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
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, groups={self.groups})"
        )


class ConvTranspose2d(Module):
    """2D transposed convolution. Readable scatter implementation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, output_padding=0, groups: int = 1,
                 bias: bool = True, dilation=1):
        super().__init__()
        if groups <= 0:
            raise ValueError("ConvTranspose2d: groups must be positive")
        if in_channels % groups != 0:
            raise ValueError("ConvTranspose2d: in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("ConvTranspose2d: out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair2(kernel_size, "kernel_size")
        self.stride = _pair2(stride, "stride")
        self.padding = _pair2(padding, "padding")
        self.output_padding = _pair2(output_padding, "output_padding")
        self.dilation = _pair2(dilation, "dilation")
        self.groups = int(groups)
        kh, kw = self.kernel_size
        fan_in = (out_channels // groups) * kh * kw
        bound = 1.0 / math.sqrt(fan_in)
        W = np.random.uniform(
            -bound, bound,
            size=(in_channels, out_channels // groups, kh, kw),
        ).astype(np.float32)
        self.weight = Tensor(W, requires_grad=True)
        if bias:
            self.bias = Tensor(np.zeros(out_channels, dtype=np.float32), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        N, C_in, H, W = x.data.shape
        if C_in != self.in_channels:
            raise ValueError(f"ConvTranspose2d: expected {self.in_channels} input channels, got {C_in}")
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        oph, opw = self.output_padding
        dh, dw = self.dilation
        groups = self.groups
        in_per_group = C_in // groups
        out_per_group = self.out_channels // groups
        H_out = (H - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1
        W_out = (W - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1
        out_data = np.zeros((N, self.out_channels, H_out, W_out), dtype=np.float32)
        for n in range(N):
            for ci in range(C_in):
                group = ci // in_per_group
                co0 = group * out_per_group
                for ih in range(H):
                    for iw in range(W):
                        value = x.data[n, ci, ih, iw]
                        if value == 0:
                            continue
                        for r in range(kh):
                            oh = ih * sh - ph + r * dh
                            if oh < 0 or oh >= H_out:
                                continue
                            for c in range(kw):
                                ow = iw * sw - pw + c * dw
                                if 0 <= ow < W_out:
                                    out_data[n, co0:co0+out_per_group, oh, ow] += (
                                        value * self.weight.data[ci, :, r, c]
                                    )
        if self.bias is not None:
            out_data = out_data + self.bias.data.reshape(1, self.out_channels, 1, 1)

        out = Tensor(out_data)
        bias_t = self.bias
        parents = (x, self.weight, bias_t) if bias_t is not None else (x, self.weight)
        weight_data = self.weight.data

        def backward(g):
            grad_out = g.data
            grad_x = np.zeros_like(x.data, dtype=np.float32)
            grad_w = np.zeros_like(weight_data, dtype=np.float32)
            for n in range(N):
                for ci in range(C_in):
                    group = ci // in_per_group
                    co0 = group * out_per_group
                    for ih in range(H):
                        for iw in range(W):
                            x_val = x.data[n, ci, ih, iw]
                            for r in range(kh):
                                oh = ih * sh - ph + r * dh
                                if oh < 0 or oh >= H_out:
                                    continue
                                for c in range(kw):
                                    ow = iw * sw - pw + c * dw
                                    if 0 <= ow < W_out:
                                        go = grad_out[n, co0:co0+out_per_group, oh, ow]
                                        grad_x[n, ci, ih, iw] += (go * weight_data[ci, :, r, c]).sum()
                                        grad_w[ci, :, r, c] += go * x_val
            if bias_t is not None:
                grad_b = grad_out.sum(axis=(0, 2, 3))
                return (grad_x, grad_w, grad_b)
            return (grad_x, grad_w)

        return _build_ctx(out, parents, backward)

    def __repr__(self):
        return (
            f"ConvTranspose2d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, output_padding={self.output_padding}, "
            f"dilation={self.dilation}, groups={self.groups})"
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
        W = np.random.uniform(
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


class Conv3d(Module):
    """3D convolution. PyTorch-style correlation over (D, H, W)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, dilation=1, groups: int = 1,
                 bias: bool = True):
        super().__init__()
        if groups <= 0:
            raise ValueError("Conv3d: groups must be positive")
        if in_channels % groups != 0:
            raise ValueError("Conv3d: in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("Conv3d: out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple3(kernel_size, "kernel_size")
        self.stride = _triple3(stride, "stride")
        self.padding = _triple3(padding, "padding")
        self.dilation = _triple3(dilation, "dilation")
        self.groups = int(groups)
        kd, kh, kw = self.kernel_size
        fan_in = (in_channels // groups) * kd * kh * kw
        bound = 1.0 / math.sqrt(fan_in)
        W = np.random.uniform(
            -bound, bound,
            size=(out_channels, in_channels // groups, kd, kh, kw),
        ).astype(np.float32)
        self.weight = Tensor(W, requires_grad=True)
        if bias:
            self.bias = Tensor(np.zeros(out_channels, dtype=np.float32), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        N, C_in, D, H, W = x.data.shape
        if C_in != self.in_channels:
            raise ValueError(f"Conv3d: expected {self.in_channels} input channels, got {C_in}")
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        pd, ph, pw = self.padding
        dd, dh, dw = self.dilation
        groups = self.groups
        c_per_group = C_in // groups
        out_per_group = self.out_channels // groups
        C_out = self.out_channels
        if pd > 0 or ph > 0 or pw > 0:
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)),
                mode="constant",
            )
        else:
            x_padded = x.data
        D_pad = D + 2 * pd
        H_pad = H + 2 * ph
        W_pad = W + 2 * pw
        eff_d = dd * (kd - 1) + 1
        eff_h = dh * (kh - 1) + 1
        eff_w = dw * (kw - 1) + 1
        D_out = (D_pad - eff_d) // sd + 1
        H_out = (H_pad - eff_h) // sh + 1
        W_out = (W_pad - eff_w) // sw + 1
        L = D_out * H_out * W_out
        cols = np.zeros((N, C_in * kd * kh * kw, L), dtype=np.float32)
        col_idx = 0
        for od in range(D_out):
            for oh in range(H_out):
                for ow in range(W_out):
                    d0, h0, w0 = od * sd, oh * sh, ow * sw
                    cols[:, :, col_idx] = (
                        x_padded[
                            :,
                            :,
                            d0:d0+eff_d:dd,
                            h0:h0+eff_h:dh,
                            w0:w0+eff_w:dw,
                        ].reshape(N, -1)
                    )
                    col_idx += 1
        out_flat = np.zeros((N, C_out, L), dtype=np.float32)
        weight_flats = []
        col_slices = []
        for g in range(groups):
            c0 = g * c_per_group
            c1 = (g + 1) * c_per_group
            o0 = g * out_per_group
            o1 = (g + 1) * out_per_group
            col_slice = slice(c0 * kd * kh * kw, c1 * kd * kh * kw)
            weight_flat_g = self.weight.data[o0:o1].reshape(out_per_group, -1)
            out_flat[:, o0:o1, :] = weight_flat_g @ cols[:, col_slice, :]
            weight_flats.append(weight_flat_g.copy())
            col_slices.append((o0, o1, col_slice))
        out_data = out_flat.reshape(N, C_out, D_out, H_out, W_out)
        if self.bias is not None:
            out_data = out_data + self.bias.data.reshape(1, C_out, 1, 1, 1)

        out = Tensor(out_data.astype(np.float32))
        bias_t = self.bias
        parents = (x, self.weight, bias_t) if bias_t is not None else (x, self.weight)
        cols_captured = cols
        weight_flats_captured = tuple(weight_flats)
        col_slices_captured = tuple(col_slices)
        weight_shape = self.weight.data.shape
        in_padded_shape = x_padded.shape

        def backward(g):
            grad_out = g.data
            grad_out_flat = grad_out.reshape(N, C_out, L)
            grad_w = np.zeros(weight_shape, dtype=np.float32)
            grad_cols = np.zeros_like(cols_captured)
            for idx, (o0, o1, col_slice) in enumerate(col_slices_captured):
                grad_out_g = grad_out_flat[:, o0:o1, :]
                cols_g = cols_captured[:, col_slice, :]
                grad_w_g = (grad_out_g @ np.swapaxes(cols_g, -1, -2)).sum(axis=0)
                grad_w[o0:o1] = grad_w_g.reshape(out_per_group, c_per_group, kd, kh, kw)
                grad_cols[:, col_slice, :] = (
                    np.swapaxes(weight_flats_captured[idx], -1, -2) @ grad_out_g
                )
            grad_x_padded = np.zeros(in_padded_shape, dtype=np.float32)
            col_idx = 0
            for od in range(D_out):
                for oh in range(H_out):
                    for ow in range(W_out):
                        d0, h0, w0 = od * sd, oh * sh, ow * sw
                        grad_x_padded[
                            :,
                            :,
                            d0:d0+eff_d:dd,
                            h0:h0+eff_h:dh,
                            w0:w0+eff_w:dw,
                        ] += grad_cols[:, :, col_idx].reshape(N, C_in, kd, kh, kw)
                        col_idx += 1
            if pd > 0 or ph > 0 or pw > 0:
                grad_x = grad_x_padded[:, :, pd:pd+D, ph:ph+H, pw:pw+W].copy()
            else:
                grad_x = grad_x_padded
            if bias_t is not None:
                grad_b = grad_out.sum(axis=(0, 2, 3, 4))
                return (grad_x, grad_w, grad_b)
            return (grad_x, grad_w)

        return _build_ctx(out, parents, backward)

    def __repr__(self):
        return (
            f"Conv3d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, groups={self.groups})"
        )


# ─── BatchNorm1d ───────────────────────────────────────────
