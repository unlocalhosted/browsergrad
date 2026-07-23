class Linear(Module):
    """y = x @ W^T + b (PyTorch convention).

    W has shape (out_features, in_features); b has shape (out_features,).
    Initialized via Kaiming uniform on W and zeros on b, matching torch.nn.Linear.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 dtype=None, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        storage_dtype = _parameter_storage_dtype(dtype, "nn.Linear")
        bound = 1.0 / math.sqrt(in_features)
        W_data = np.random.uniform(
            -bound, bound, size=(out_features, in_features)
        ).astype(storage_dtype)
        self.weight = Tensor(W_data, dtype=storage_dtype, requires_grad=True)
        if bias:
            self.bias = Tensor(
                np.zeros(out_features, dtype=storage_dtype),
                dtype=storage_dtype,
                requires_grad=True,
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # v0.2: bias broadcasts naturally via Tensor.__add__.
        out = x @ self.weight.transpose(0, 1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


# ─── Conv2d ────────────────────────────────────────────────
