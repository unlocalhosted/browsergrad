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