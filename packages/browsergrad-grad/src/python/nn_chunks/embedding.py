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
