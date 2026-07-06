def _rnn_init_uniform(shape, hidden_size, dtype=np.float32):
    bound = 1.0 / math.sqrt(hidden_size)
    return np.random.uniform(-bound, bound, size=shape).astype(dtype)


def _activation(name, x: Tensor) -> Tensor:
    if name == "tanh":
        return F.tanh(x)
    if name == "relu":
        return F.relu(x)
    raise ValueError(f"RNN: unsupported nonlinearity {name!r}")


def _reverse_time(x: Tensor) -> Tensor:
    return x.flip(0)


def _direction_suffix(direction: int) -> str:
    return "" if direction == 0 else "_reverse"


def _dropout_between_layers(x: Tensor, p: float, training: bool) -> Tensor:
    if not training or p == 0.0:
        return x
    keep = 1.0 - p
    mask = (np.random.rand(*x.data.shape) < keep).astype(np.float32) / keep
    out = Tensor((x.data * mask).astype(np.float32))
    return _build_ctx(out, (x,), lambda g: (g.data * mask,))


def _init_hidden_stack(kind: str, state, B: int, H: int, count: int):
    if state is None:
        return [Tensor(np.zeros((B, H), dtype=np.float32)) for _ in range(count)]
    if isinstance(state, tuple):
        raise TypeError(f"{kind}: expected hidden Tensor, got tuple")
    if state.data.ndim == 2:
        if count != 1:
            raise ValueError(f"{kind}: h_0 rank-2 accepted only for single layer/direction")
        return [state]
    if state.data.shape[0] != count:
        raise ValueError(f"{kind}: h_0 first dim must be {count}, got {state.data.shape[0]}")
    return [state[i] for i in range(count)]


def _init_lstm_state(state, B: int, H: int, count: int):
    if state is None:
        zeros = [Tensor(np.zeros((B, H), dtype=np.float32)) for _ in range(count)]
        return zeros, [Tensor(np.zeros((B, H), dtype=np.float32)) for _ in range(count)]
    h, c = state
    hs = _init_hidden_stack("LSTM", h, B, H, count)
    if c.data.ndim == 2:
        if count != 1:
            raise ValueError("LSTM: c_0 rank-2 accepted only for single layer/direction")
        cs = [c]
    else:
        if c.data.shape[0] != count:
            raise ValueError(f"LSTM: c_0 first dim must be {count}, got {c.data.shape[0]}")
        cs = [c[i] for i in range(count)]
    return hs, cs


class RNN(Module):
    """Vanilla Elman RNN with PyTorch-style layer/direction state layout.

    Shape (batch_first=False):
      input: (T, B, input_size)
      output: (T, B, D*hidden_size), h_n: (D*num_layers, B, hidden_size)
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False,
                 nonlinearity: str = "tanh"):
        super().__init__()
        if num_layers < 1:
            raise ValueError("RNN: num_layers must be positive")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"RNN: dropout must be in [0, 1), got {dropout}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = int(num_layers)
        self.bias = bool(bias)
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        self.nonlinearity = nonlinearity
        directions = 2 if self.bidirectional else 1
        for layer in range(self.num_layers):
            layer_input = input_size if layer == 0 else hidden_size * directions
            for direction in range(directions):
                suffix = _direction_suffix(direction)
                setattr(self, f"weight_ih_l{layer}{suffix}",
                        Tensor(_rnn_init_uniform((hidden_size, layer_input), hidden_size), requires_grad=True))
                setattr(self, f"weight_hh_l{layer}{suffix}",
                        Tensor(_rnn_init_uniform((hidden_size, hidden_size), hidden_size), requires_grad=True))
                if self.bias:
                    setattr(self, f"bias_ih_l{layer}{suffix}",
                            Tensor(_rnn_init_uniform((hidden_size,), hidden_size), requires_grad=True))
                    setattr(self, f"bias_hh_l{layer}{suffix}",
                            Tensor(_rnn_init_uniform((hidden_size,), hidden_size), requires_grad=True))

    def _run_direction(self, x: Tensor, h: Tensor, layer: int, direction: int):
        suffix = _direction_suffix(direction)
        W_ih_T = getattr(self, f"weight_ih_l{layer}{suffix}").transpose(0, 1)
        W_hh_T = getattr(self, f"weight_hh_l{layer}{suffix}").transpose(0, 1)
        b_ih = getattr(self, f"bias_ih_l{layer}{suffix}", None)
        b_hh = getattr(self, f"bias_hh_l{layer}{suffix}", None)
        seq = _reverse_time(x) if direction == 1 else x
        outs = []
        for t in range(seq.data.shape[0]):
            pre = seq[t] @ W_ih_T + h @ W_hh_T
            if b_ih is not None:
                pre = pre + b_ih + b_hh
            h = _activation(self.nonlinearity, pre)
            outs.append(h)
        out = stack(outs, dim=0)
        if direction == 1:
            out = _reverse_time(out)
        return out, h

    def forward(self, x: Tensor, h_0: Optional[Tensor] = None):
        if self.batch_first:
            x = x.permute(1, 0, 2)
        T, B, _ = x.data.shape
        directions = 2 if self.bidirectional else 1
        hidden = _init_hidden_stack("RNN", h_0, B, self.hidden_size, self.num_layers * directions)
        layer_input = x
        final_h = []
        for layer in range(self.num_layers):
            dir_outs = []
            for direction in range(directions):
                idx = layer * directions + direction
                out_d, h_d = self._run_direction(layer_input, hidden[idx], layer, direction)
                dir_outs.append(out_d)
                final_h.append(h_d)
            out = dir_outs[0] if directions == 1 else cat(tuple(dir_outs), dim=2)
            if layer != self.num_layers - 1:
                out = _dropout_between_layers(out, self.dropout, self.training)
            layer_input = out
        h_n = stack(final_h, dim=0)
        if self.batch_first:
            out = out.permute(1, 0, 2)
        return out, h_n

    def __repr__(self):
        return (
            f"RNN({self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"batch_first={self.batch_first}, bidirectional={self.bidirectional})"
        )


class LSTM(Module):
    """Single-layer LSTM (Hochreiter & Schmidhuber). Matches torch.nn.LSTM
    parameter layout: weight_ih is (4*hidden, input) — gate order i, f, g, o.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        if num_layers < 1:
            raise ValueError("LSTM: num_layers must be positive")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"LSTM: dropout must be in [0, 1), got {dropout}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = int(num_layers)
        self.bias = bool(bias)
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        directions = 2 if self.bidirectional else 1
        for layer in range(self.num_layers):
            layer_input = input_size if layer == 0 else hidden_size * directions
            for direction in range(directions):
                suffix = _direction_suffix(direction)
                setattr(self, f"weight_ih_l{layer}{suffix}",
                        Tensor(_rnn_init_uniform((4 * hidden_size, layer_input), hidden_size), requires_grad=True))
                setattr(self, f"weight_hh_l{layer}{suffix}",
                        Tensor(_rnn_init_uniform((4 * hidden_size, hidden_size), hidden_size), requires_grad=True))
                if self.bias:
                    setattr(self, f"bias_ih_l{layer}{suffix}",
                            Tensor(_rnn_init_uniform((4 * hidden_size,), hidden_size), requires_grad=True))
                    setattr(self, f"bias_hh_l{layer}{suffix}",
                            Tensor(_rnn_init_uniform((4 * hidden_size,), hidden_size), requires_grad=True))

    def _run_direction(self, x: Tensor, h: Tensor, c: Tensor, layer: int, direction: int):
        suffix = _direction_suffix(direction)
        H = self.hidden_size
        W_ih_T = getattr(self, f"weight_ih_l{layer}{suffix}").transpose(0, 1)
        W_hh_T = getattr(self, f"weight_hh_l{layer}{suffix}").transpose(0, 1)
        b_ih = getattr(self, f"bias_ih_l{layer}{suffix}", None)
        b_hh = getattr(self, f"bias_hh_l{layer}{suffix}", None)
        seq = _reverse_time(x) if direction == 1 else x
        outs = []
        for t in range(seq.data.shape[0]):
            gates = seq[t] @ W_ih_T + h @ W_hh_T
            if b_ih is not None:
                gates = gates + b_ih + b_hh
            i_g = F.sigmoid(gates[:, 0:H])
            f_g = F.sigmoid(gates[:, H:2*H])
            g_g = F.tanh(gates[:, 2*H:3*H])
            o_g = F.sigmoid(gates[:, 3*H:4*H])
            c = f_g * c + i_g * g_g
            h = o_g * F.tanh(c)
            outs.append(h)
        out = stack(outs, dim=0)
        if direction == 1:
            out = _reverse_time(out)
        return out, h, c

    def forward(self, x: Tensor, hc_0=None):
        if self.batch_first:
            x = x.permute(1, 0, 2)
        T, B, _ = x.data.shape
        H = self.hidden_size
        directions = 2 if self.bidirectional else 1
        h0, c0 = _init_lstm_state(hc_0, B, H, self.num_layers * directions)
        layer_input = x
        final_h = []
        final_c = []
        for layer in range(self.num_layers):
            dir_outs = []
            for direction in range(directions):
                idx = layer * directions + direction
                out_d, h_d, c_d = self._run_direction(layer_input, h0[idx], c0[idx], layer, direction)
                dir_outs.append(out_d)
                final_h.append(h_d)
                final_c.append(c_d)
            out = dir_outs[0] if directions == 1 else cat(tuple(dir_outs), dim=2)
            if layer != self.num_layers - 1:
                out = _dropout_between_layers(out, self.dropout, self.training)
            layer_input = out
        h_n = stack(final_h, dim=0)
        c_n = stack(final_c, dim=0)
        if self.batch_first:
            out = out.permute(1, 0, 2)
        return out, (h_n, c_n)

    def __repr__(self):
        return (
            f"LSTM({self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"batch_first={self.batch_first}, bidirectional={self.bidirectional})"
        )


class GRU(Module):
    """Single-layer GRU (Cho et al.). Matches torch.nn.GRU parameter layout:
    weight_ih is (3*hidden, input) — gate order r, z, n.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        if num_layers < 1:
            raise ValueError("GRU: num_layers must be positive")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"GRU: dropout must be in [0, 1), got {dropout}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = int(num_layers)
        self.bias = bool(bias)
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)
        directions = 2 if self.bidirectional else 1
        for layer in range(self.num_layers):
            layer_input = input_size if layer == 0 else hidden_size * directions
            for direction in range(directions):
                suffix = _direction_suffix(direction)
                setattr(self, f"weight_ih_l{layer}{suffix}",
                        Tensor(_rnn_init_uniform((3 * hidden_size, layer_input), hidden_size), requires_grad=True))
                setattr(self, f"weight_hh_l{layer}{suffix}",
                        Tensor(_rnn_init_uniform((3 * hidden_size, hidden_size), hidden_size), requires_grad=True))
                if self.bias:
                    setattr(self, f"bias_ih_l{layer}{suffix}",
                            Tensor(_rnn_init_uniform((3 * hidden_size,), hidden_size), requires_grad=True))
                    setattr(self, f"bias_hh_l{layer}{suffix}",
                            Tensor(_rnn_init_uniform((3 * hidden_size,), hidden_size), requires_grad=True))

    def _run_direction(self, x: Tensor, h: Tensor, layer: int, direction: int):
        suffix = _direction_suffix(direction)
        H = self.hidden_size
        W_ih_T = getattr(self, f"weight_ih_l{layer}{suffix}").transpose(0, 1)
        W_hh_T = getattr(self, f"weight_hh_l{layer}{suffix}").transpose(0, 1)
        b_ih = getattr(self, f"bias_ih_l{layer}{suffix}", None)
        b_hh = getattr(self, f"bias_hh_l{layer}{suffix}", None)
        seq = _reverse_time(x) if direction == 1 else x
        outs = []
        for t in range(seq.data.shape[0]):
            ih = seq[t] @ W_ih_T
            hh = h @ W_hh_T
            if b_ih is not None:
                ih = ih + b_ih
                hh = hh + b_hh
            r = F.sigmoid(ih[:, 0:H] + hh[:, 0:H])
            z = F.sigmoid(ih[:, H:2*H] + hh[:, H:2*H])
            n = F.tanh(ih[:, 2*H:3*H] + r * hh[:, 2*H:3*H])
            h = (1.0 - z) * n + z * h
            outs.append(h)
        out = stack(outs, dim=0)
        if direction == 1:
            out = _reverse_time(out)
        return out, h

    def forward(self, x: Tensor, h_0: Optional[Tensor] = None):
        if self.batch_first:
            x = x.permute(1, 0, 2)
        T, B, _ = x.data.shape
        H = self.hidden_size
        directions = 2 if self.bidirectional else 1
        hidden = _init_hidden_stack("GRU", h_0, B, H, self.num_layers * directions)
        layer_input = x
        final_h = []
        for layer in range(self.num_layers):
            dir_outs = []
            for direction in range(directions):
                idx = layer * directions + direction
                out_d, h_d = self._run_direction(layer_input, hidden[idx], layer, direction)
                dir_outs.append(out_d)
                final_h.append(h_d)
            out = dir_outs[0] if directions == 1 else cat(tuple(dir_outs), dim=2)
            if layer != self.num_layers - 1:
                out = _dropout_between_layers(out, self.dropout, self.training)
            layer_input = out
        h_n = stack(final_h, dim=0)
        if self.batch_first:
            out = out.permute(1, 0, 2)
        return out, h_n

    def __repr__(self):
        return (
            f"GRU({self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"batch_first={self.batch_first}, bidirectional={self.bidirectional})"
        )


# ─── PyTorch lowercase-h alias ─────────────────────────────
# Defined here at module bottom so MultiHeadAttention is in scope.
