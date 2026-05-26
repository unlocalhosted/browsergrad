class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)
    def __repr__(self):
        return "ReLU()"


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, self.negative_slope)
    def __repr__(self):
        return f"LeakyReLU(negative_slope={self.negative_slope})"


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)
    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x)
    def __repr__(self):
        return "Tanh()"


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x)
    def __repr__(self):
        return "GELU()"


# ─── Loss modules (callable wrappers around functional losses) ─────
