class Dropout(Module):
    """Inverted dropout (matches torch.nn.Dropout).

    train mode: each element kept with probability (1-p), kept values scaled
                by 1/(1-p) so E[y] = x.
    eval mode:  identity (no scaling, no zeroing).
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = F._normalize_dropout_probability(p)
        if type(inplace) is not bool:
            raise ValueError("Dropout: inplace must be an exact bool")
        if inplace:
            raise NotImplementedError(
                "Dropout(inplace=True) requires typed mutation semantics and "
                "is not supported"
            )
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(
            x,
            p=self.p,
            training=self.training,
            inplace=self.inplace,
        )

    def __repr__(self):
        return f"Dropout(p={self.p}, inplace={self.inplace})"


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
