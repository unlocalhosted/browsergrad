class CrossEntropyLoss(Module):
    """Module form of cross_entropy_loss. Matches torch.nn.CrossEntropyLoss."""
    def forward(self, logits: Tensor, targets) -> Tensor:
        return F.cross_entropy_loss(logits, targets)
    def __repr__(self):
        return "CrossEntropyLoss()"


class MSELoss(Module):
    """Module form of mse_loss. Matches torch.nn.MSELoss."""
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target)
    def __repr__(self):
        return "MSELoss()"


class NLLLoss(Module):
    """Module form of nll_loss. Matches torch.nn.NLLLoss."""
    def forward(self, log_probs: Tensor, targets) -> Tensor:
        return F.nll_loss(log_probs, targets)
    def __repr__(self):
        return "NLLLoss()"


class BCEWithLogitsLoss(Module):
    """Module form of binary cross-entropy from logits. Matches
    torch.nn.BCEWithLogitsLoss. Uses the stable formula
      max(x, 0) - x*y + log(1 + exp(-|x|))
    so very large/small logits don't overflow.
    """
    def forward(self, logits: Tensor, targets) -> Tensor:
        return F.bce_with_logits_loss(logits, targets)
    def __repr__(self):
        return "BCEWithLogitsLoss()"


class BCELoss(Module):
    """Binary cross-entropy from probabilities. Matches torch.nn.BCELoss.
    Use BCEWithLogitsLoss for raw logits — it's numerically stable.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.bce_loss(input, target, reduction=self.reduction)
    def __repr__(self):
        return f"BCELoss(reduction={self.reduction!r})"


class L1Loss(Module):
    """Mean absolute error. Matches torch.nn.L1Loss."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target, reduction=self.reduction)
    def __repr__(self):
        return f"L1Loss(reduction={self.reduction!r})"


class SmoothL1Loss(Module):
    """Smooth L1 (Huber with knee at beta). Matches torch.nn.SmoothL1Loss."""
    def __init__(self, beta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.smooth_l1_loss(input, target, beta=self.beta, reduction=self.reduction)
    def __repr__(self):
        return f"SmoothL1Loss(beta={self.beta}, reduction={self.reduction!r})"


class KLDivLoss(Module):
    """KL divergence loss. Input is log-prob (e.g. output of log_softmax).

    Note: PyTorch's KLDivLoss default reduction is 'mean' (over total elements)
    but recommends 'batchmean' for the mathematically conventional definition.
    We default to 'mean' to match PyTorch.
    """
    def __init__(self, reduction: str = "mean", log_target: bool = False):
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.kl_div_loss(input, target, reduction=self.reduction, log_target=self.log_target)
    def __repr__(self):
        return f"KLDivLoss(reduction={self.reduction!r}, log_target={self.log_target})"


# The MultiheadAttention lowercase alias is set at the BOTTOM of this module,
# after MultiHeadAttention is defined.

