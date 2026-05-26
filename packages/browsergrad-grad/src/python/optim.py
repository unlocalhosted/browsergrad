
"""browsergrad_grad.optim — gradient-descent optimizers."""

import numpy as np
from typing import Iterable, List
from .tensor import Tensor


class Optimizer:
    """Base class. Subclasses implement step()."""

    def __init__(self, params: Iterable[Tensor], lr: float):
        self.params: List[Tensor] = [p for p in params]
        if any(not isinstance(p, Tensor) or not p.requires_grad for p in self.params):
            raise ValueError("All optimizer parameters must be Tensors with requires_grad=True")
        self.lr = float(lr)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum and weight decay.

    Update rule (matches PyTorch):
      g_t = grad + weight_decay * param          (if weight_decay)
      b_t = momentum * b_{t-1} + g_t              (if momentum)
      param = param - lr * (b_t if momentum else g_t)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self._velocity: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data
            if self.momentum != 0.0:
                self._velocity[i] = self.momentum * self._velocity[i] + g
                update = self._velocity[i]
            else:
                update = g
            p.data = p.data - self.lr * update

    def __repr__(self):
        return (
            f"SGD(lr={self.lr}, momentum={self.momentum}, "
            f"weight_decay={self.weight_decay}, params={len(self.params)})"
        )


class Adam(Optimizer):
    """Adam optimizer (Kingma & Ba, 2014).

    Update rule (matches torch.optim.Adam with default amsgrad=False):
      g_t        = grad + weight_decay * param                (L2 regularization)
      m_t        = beta1 * m_{t-1} + (1 - beta1) * g_t
      v_t        = beta2 * v_{t-1} + (1 - beta2) * g_t²
      m_hat      = m_t / (1 - beta1^t)
      v_hat      = v_t / (1 - beta2^t)
      param      = param - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._step_count = 0
        self._m: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]
        self._v: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self._step_count += 1
        bc1 = 1.0 - self.beta1 ** self._step_count
        bc2 = 1.0 - self.beta2 ** self._step_count
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2
            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def __repr__(self):
        return (
            f"Adam(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
            f"eps={self.eps}, weight_decay={self.weight_decay}, "
            f"params={len(self.params)})"
        )


class AdamW(Optimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2017).

    Difference from Adam: weight decay is applied directly to the parameter
    after the gradient update, NOT folded into the gradient. This is what
    transformer pre-training typically uses.

      m_t        = beta1 * m_{t-1} + (1 - beta1) * g_t
      v_t        = beta2 * v_{t-1} + (1 - beta2) * g_t²
      m_hat      = m_t / (1 - beta1^t)
      v_hat      = v_t / (1 - beta2^t)
      param      = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._step_count = 0
        self._m: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]
        self._v: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self._step_count += 1
        bc1 = 1.0 - self.beta1 ** self._step_count
        bc2 = 1.0 - self.beta2 ** self._step_count
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * (g * g)
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay != 0.0:
                p.data = p.data - self.lr * (update + self.weight_decay * p.data)
            else:
                p.data = p.data - self.lr * update

    def __repr__(self):
        return (
            f"AdamW(lr={self.lr}, betas=({self.beta1}, {self.beta2}), "
            f"eps={self.eps}, weight_decay={self.weight_decay}, "
            f"params={len(self.params)})"
        )


# ─── Learning-rate schedulers ──────────────────────────────

class _LRScheduler:
    """Base class. Subclasses override _compute_lr(step)."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = float(optimizer.lr)
        self.last_step = 0

    def step(self):
        """Advance one scheduler step; update optimizer.lr in place."""
        self.last_step += 1
        self.optimizer.lr = float(self._compute_lr(self.last_step))

    def _compute_lr(self, step: int) -> float:
        raise NotImplementedError


class StepLR(_LRScheduler):
    """Decay lr by `gamma` every `step_size` scheduler steps.

    Matches torch.optim.lr_scheduler.StepLR. At step N, the effective lr is
      base_lr * gamma ** (N // step_size)
    """

    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = int(step_size)
        self.gamma = float(gamma)

    def _compute_lr(self, step: int) -> float:
        return self.base_lr * (self.gamma ** (step // self.step_size))

    def __repr__(self):
        return f"StepLR(step_size={self.step_size}, gamma={self.gamma})"


class CosineAnnealingLR(_LRScheduler):
    """Cosine schedule from `base_lr` to `eta_min` over `T_max` steps.

    Matches torch.optim.lr_scheduler.CosineAnnealingLR. At step t:
      lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(t / T_max * pi))

    After t = T_max the cosine continues (PyTorch does NOT clamp).
    """

    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0):
        super().__init__(optimizer)
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)

    def _compute_lr(self, step: int) -> float:
        cos_val = np.cos(step / self.T_max * np.pi)
        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + cos_val)

    def __repr__(self):
        return f"CosineAnnealingLR(T_max={self.T_max}, eta_min={self.eta_min})"


# ─── More optimizers (Pile A #10) ──────────────────────────

class RMSprop(Optimizer):
    """RMSprop. Update rule (matches torch.optim.RMSprop, centered=False):
      v_t = alpha * v_{t-1} + (1 - alpha) * g_t^2
      param -= lr * g_t / (sqrt(v_t) + eps)
    """
    def __init__(self, params: Iterable[Tensor], lr: float = 0.01, alpha: float = 0.99,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._v: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data
            self._v[i] = self.alpha * self._v[i] + (1.0 - self.alpha) * (g * g)
            p.data = p.data - self.lr * g / (np.sqrt(self._v[i]) + self.eps)

    def __repr__(self):
        return f"RMSprop(lr={self.lr}, alpha={self.alpha}, eps={self.eps})"


class Adagrad(Optimizer):
    """Adagrad. Update rule:
      G_t = G_{t-1} + g_t^2
      param -= lr * g_t / (sqrt(G_t) + eps)
    """
    def __init__(self, params: Iterable[Tensor], lr: float = 0.01,
                 eps: float = 1e-10, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._G: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data
            self._G[i] = self._G[i] + g * g
            p.data = p.data - self.lr * g / (np.sqrt(self._G[i]) + self.eps)

    def __repr__(self):
        return f"Adagrad(lr={self.lr}, eps={self.eps})"


class Adadelta(Optimizer):
    """Adadelta. Update rule:
      Eg_t = rho * Eg_{t-1} + (1 - rho) * g_t^2
      dx_t = -(sqrt(Ex_{t-1} + eps) / sqrt(Eg_t + eps)) * g_t
      Ex_t = rho * Ex_{t-1} + (1 - rho) * dx_t^2
      param += lr * dx_t
    """
    def __init__(self, params: Iterable[Tensor], lr: float = 1.0, rho: float = 0.9,
                 eps: float = 1e-6, weight_decay: float = 0.0):
        super().__init__(params, lr)
        self.rho = float(rho)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self._Eg: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]
        self._Ex: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data
            self._Eg[i] = self.rho * self._Eg[i] + (1.0 - self.rho) * (g * g)
            dx = -(np.sqrt(self._Ex[i] + self.eps) / np.sqrt(self._Eg[i] + self.eps)) * g
            self._Ex[i] = self.rho * self._Ex[i] + (1.0 - self.rho) * (dx * dx)
            p.data = p.data + self.lr * dx

    def __repr__(self):
        return f"Adadelta(lr={self.lr}, rho={self.rho}, eps={self.eps})"


# ─── More LR schedulers (Pile A #11) ───────────────────────

class MultiStepLR(_LRScheduler):
    """Decay lr by gamma at each milestone. Matches torch.optim.lr_scheduler.MultiStepLR.
    At step N, effective lr = base_lr * gamma^(number_of_milestones_passed_by_N).
    """
    def __init__(self, optimizer, milestones, gamma: float = 0.1):
        super().__init__(optimizer)
        self.milestones = sorted(int(m) for m in milestones)
        self.gamma = float(gamma)

    def _compute_lr(self, step: int) -> float:
        passed = sum(1 for m in self.milestones if step >= m)
        return self.base_lr * (self.gamma ** passed)

    def __repr__(self):
        return f"MultiStepLR(milestones={self.milestones}, gamma={self.gamma})"


class ExponentialLR(_LRScheduler):
    """lr *= gamma each scheduler step. Matches torch.optim.lr_scheduler.ExponentialLR."""
    def __init__(self, optimizer, gamma: float):
        super().__init__(optimizer)
        self.gamma = float(gamma)

    def _compute_lr(self, step: int) -> float:
        return self.base_lr * (self.gamma ** step)

    def __repr__(self):
        return f"ExponentialLR(gamma={self.gamma})"


class ReduceLROnPlateau:
    """Reduce LR when a metric stops improving. Matches torch's class with the
    common subset of options: mode in {min, max}, factor, patience, threshold.

    Unlike _LRScheduler, this one takes a metric in step(metric).
    """
    def __init__(self, optimizer, mode: str = "min", factor: float = 0.1,
                 patience: int = 10, threshold: float = 1e-4, min_lr: float = 0.0):
        if mode not in ("min", "max"):
            raise ValueError(f"ReduceLROnPlateau: mode must be 'min' or 'max', got {mode!r}")
        if not (0.0 < factor < 1.0):
            raise ValueError(f"ReduceLROnPlateau: factor must be in (0, 1), got {factor}")
        self.optimizer = optimizer
        self.mode = mode
        self.factor = float(factor)
        self.patience = int(patience)
        self.threshold = float(threshold)
        self.min_lr = float(min_lr)
        self.best = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs = 0

    def _is_better(self, metric: float) -> bool:
        if self.mode == "min":
            return metric < self.best - self.threshold
        return metric > self.best + self.threshold

    def step(self, metric: float) -> None:
        m = float(metric)
        if self._is_better(m):
            self.best = m
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs > self.patience:
            new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
            self.optimizer.lr = float(new_lr)
            self.num_bad_epochs = 0

    def __repr__(self):
        return f"ReduceLROnPlateau(mode={self.mode!r}, factor={self.factor}, patience={self.patience})"


class OneCycleLR(_LRScheduler):
    """One-cycle policy: warm up from initial_lr to max_lr, then anneal to a
    very small final value. Matches torch.optim.lr_scheduler.OneCycleLR in
    its essentials: pct_start, anneal_strategy='cos', cosine warmup + anneal.

    Drops the more exotic momentum-cycling features (we don't model momentum
    in our optimizer base class).
    """
    def __init__(self, optimizer, max_lr: float, total_steps: int,
                 pct_start: float = 0.3, div_factor: float = 25.0,
                 final_div_factor: float = 1e4, anneal_strategy: str = "cos"):
        super().__init__(optimizer)
        if anneal_strategy not in ("cos", "linear"):
            raise ValueError(f"OneCycleLR: anneal_strategy must be 'cos' or 'linear'")
        self.max_lr = float(max_lr)
        self.total_steps = int(total_steps)
        self.pct_start = float(pct_start)
        self.div_factor = float(div_factor)
        self.final_div_factor = float(final_div_factor)
        self.anneal_strategy = anneal_strategy
        self.initial_lr = self.max_lr / self.div_factor
        self.final_lr = self.initial_lr / self.final_div_factor
        self.warmup_steps = max(1, int(round(self.pct_start * self.total_steps)))
        self.optimizer.lr = self.initial_lr
        self.base_lr = self.initial_lr  # for _LRScheduler interface compatibility

    def _compute_lr(self, step: int) -> float:
        if step <= self.warmup_steps:
            # Cosine warmup from initial_lr → max_lr.
            t = step / self.warmup_steps
            if self.anneal_strategy == "linear":
                return self.initial_lr + t * (self.max_lr - self.initial_lr)
            cos = 0.5 * (1.0 - np.cos(np.pi * t))
            return self.initial_lr + cos * (self.max_lr - self.initial_lr)
        # Anneal from max_lr → final_lr over remaining steps.
        remaining = self.total_steps - self.warmup_steps
        t = (step - self.warmup_steps) / max(remaining, 1)
        if self.anneal_strategy == "linear":
            return self.max_lr + t * (self.final_lr - self.max_lr)
        cos = 0.5 * (1.0 + np.cos(np.pi * min(t, 1.0)))
        return self.final_lr + cos * (self.max_lr - self.final_lr)

    def __repr__(self):
        return f"OneCycleLR(max_lr={self.max_lr}, total_steps={self.total_steps})"


# Expose schedulers under a sub-namespace matching torch.optim.lr_scheduler
import types as _bg_optim_types
import sys as _bg_optim_sys
lr_scheduler = _bg_optim_types.ModuleType("browsergrad_grad.optim.lr_scheduler")
lr_scheduler.StepLR = StepLR
lr_scheduler.CosineAnnealingLR = CosineAnnealingLR
lr_scheduler.MultiStepLR = MultiStepLR
lr_scheduler.ExponentialLR = ExponentialLR
lr_scheduler.ReduceLROnPlateau = ReduceLROnPlateau
lr_scheduler.OneCycleLR = OneCycleLR
_bg_optim_sys.modules["browsergrad_grad.optim.lr_scheduler"] = lr_scheduler
