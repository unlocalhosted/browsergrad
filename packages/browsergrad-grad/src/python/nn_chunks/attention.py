class MultiHeadAttention(Module):
    """Batch-first multi-head scaled dot-product attention.

    Inputs Q, K, V each of shape (N, S, embed_dim); output (N, S, embed_dim).
    `embed_dim` must be divisible by `num_heads`. No mask support in v0.

    Composes existing differentiable ops (Linear, matmul, softmax, transpose,
    reshape) so autograd works automatically — no hand-written backward.
    """

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"MultiHeadAttention: embed_dim {embed_dim} not divisible "
                f"by num_heads {num_heads}"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self._scale = float(1.0 / np.sqrt(self.head_dim))

    def _split_heads(self, x: Tensor, N: int, S: int) -> Tensor:
        # (N, S, D) → (N, S, H, d_k) → (N, H, S, d_k)
        return x.reshape(N, S, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor, N: int, S: int) -> Tensor:
        # (N, H, S, d_k) → (N, S, H, d_k) → (N, S, D)
        return x.transpose(1, 2).reshape(N, S, self.embed_dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        if query.data.ndim != 3:
            raise ValueError(
                f"MultiHeadAttention expects 3D query (N, S, D); got {query.data.ndim}D"
            )
        N, S, D = query.data.shape
        if D != self.embed_dim:
            raise ValueError(
                f"MultiHeadAttention: query last dim {D} ≠ embed_dim {self.embed_dim}"
            )
        q = self._split_heads(self.q_proj(query), N, query.data.shape[1])
        k = self._split_heads(self.k_proj(key), N, key.data.shape[1])
        v = self._split_heads(self.v_proj(value), N, value.data.shape[1])
        # scores: (N, H, S_q, S_k)
        scores = (q @ k.transpose(-1, -2)) * self._scale
        weights = F.softmax(scores, dim=-1)
        attn = weights @ v  # (N, H, S_q, d_k)
        merged = self._merge_heads(attn, N, S)
        return self.out_proj(merged)

    def __repr__(self):
        return (
            f"MultiHeadAttention(embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads})"
        )


MultiheadAttention = MultiHeadAttention

