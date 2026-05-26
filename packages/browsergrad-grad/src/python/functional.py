
"""browsergrad_grad.functional — stateless ops with autograd."""

import numpy as np
from .tensor import Tensor, _build_ctx


# ─── Activations ───────────────────────────────────────────

def relu(x: Tensor) -> Tensor:
    out_data = np.maximum(x.data, 0.0)
    out = Tensor(out_data)
    mask = (x.data > 0).astype(np.float32)
    return _build_ctx(out, (x,), lambda g: (g.data * mask,))


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    out_data = np.where(x.data > 0, x.data, x.data * negative_slope)
    out = Tensor(out_data)
    mask = np.where(x.data > 0, 1.0, negative_slope).astype(np.float32)
    return _build_ctx(out, (x,), lambda g: (g.data * mask,))


def sigmoid(x: Tensor) -> Tensor:
    s = 1.0 / (1.0 + np.exp(-x.data))
    out = Tensor(s)
    return _build_ctx(out, (x,), lambda g: (g.data * s * (1.0 - s),))


def tanh(x: Tensor) -> Tensor:
    t = np.tanh(x.data)
    out = Tensor(t)
    return _build_ctx(out, (x,), lambda g: (g.data * (1.0 - t * t),))


# Tanh-approximation GELU (GPT-2 / BERT variant)
_GELU_C = float(np.sqrt(2.0 / np.pi))

def gelu(x: Tensor) -> Tensor:
    xd = x.data
    inner = _GELU_C * (xd + 0.044715 * (xd ** 3))
    tanh_inner = np.tanh(inner)
    out_data = 0.5 * xd * (1.0 + tanh_inner)
    out = Tensor(out_data)
    # Derivative: 0.5*(1+tanh(inner)) + 0.5*x*sech²(inner) * d(inner)/dx
    # sech²(t) = 1 - tanh²(t); d(inner)/dx = C*(1 + 3*0.044715*x²)
    sech2 = 1.0 - tanh_inner * tanh_inner
    d_inner = _GELU_C * (1.0 + 3.0 * 0.044715 * (xd ** 2))
    deriv = 0.5 * (1.0 + tanh_inner) + 0.5 * xd * sech2 * d_inner
    return _build_ctx(out, (x,), lambda g: (g.data * deriv,))


# ─── Softmax family ────────────────────────────────────────

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Stable softmax along `dim`."""
    xd = x.data
    shifted = xd - xd.max(axis=dim, keepdims=True)
    exp_data = np.exp(shifted)
    sum_data = exp_data.sum(axis=dim, keepdims=True)
    s = exp_data / sum_data
    out = Tensor(s)
    # d(softmax)/dx_i for row r:  s_i * (g_i - sum_j(s_j * g_j))
    def backward(g):
        # weighted sum along dim
        ws = (g.data * s).sum(axis=dim, keepdims=True)
        return (s * (g.data - ws),)
    return _build_ctx(out, (x,), backward)


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable log_softmax along `dim`."""
    xd = x.data
    shifted = xd - xd.max(axis=dim, keepdims=True)
    log_sum = np.log(np.exp(shifted).sum(axis=dim, keepdims=True))
    out_data = shifted - log_sum
    out = Tensor(out_data)
    # d(log_softmax)/dx_i for row r: g_i - softmax_i * sum_j(g_j)
    s = np.exp(out_data)  # softmax recovered from log_softmax
    def backward(g):
        g_sum = g.data.sum(axis=dim, keepdims=True)
        return (g.data - s * g_sum,)
    return _build_ctx(out, (x,), backward)


# ─── Losses ────────────────────────────────────────────────

def mse_loss(y_hat: Tensor, y: Tensor) -> Tensor:
    """Mean squared error. Returns a scalar Tensor."""
    if y_hat.data.shape != y.data.shape:
        raise ValueError(
            f"mse_loss: shape mismatch {y_hat.data.shape} vs {y.data.shape}"
        )
    diff = y_hat.data - y.data
    loss_data = float(np.mean(diff * diff))
    out = Tensor(np.float32(loss_data))
    n = float(y.data.size)
    return _build_ctx(out, (y_hat,), lambda g: (g.data * (2.0 / n) * diff,))


def cross_entropy_loss(logits: Tensor, targets) -> Tensor:
    """Fused softmax + NLL: -mean(log_softmax(logits)[range(N), targets]).

    `logits`: shape (N, C). `targets`: int array shape (N,) — class indices.
    Returns a scalar Tensor. The fused form avoids both the numerical-stability
    pitfalls of naive log(softmax) and the need for differentiable indexing.
    """
    if isinstance(targets, Tensor):
        targets_np = targets.data.astype(np.int64)
    else:
        targets_np = np.asarray(targets, dtype=np.int64)
    if logits.data.ndim != 2:
        raise ValueError(f"cross_entropy_loss: logits must be 2D (N, C), got {logits.data.ndim}D")
    if targets_np.ndim != 1 or targets_np.shape[0] != logits.data.shape[0]:
        raise ValueError(
            f"cross_entropy_loss: targets shape {targets_np.shape} doesn't match logits batch {logits.data.shape[0]}"
        )
    N, C = logits.data.shape
    if (targets_np < 0).any() or (targets_np >= C).any():
        raise ValueError(f"cross_entropy_loss: targets out of range [0, {C})")

    # Numerically stable log_softmax.
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    log_sum = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probs = shifted - log_sum
    # Pick targets and average.
    loss_data = float(-log_probs[np.arange(N), targets_np].mean())
    out = Tensor(np.float32(loss_data))
    # Gradient w.r.t. logits: (softmax - one_hot(targets)) / N
    softmax_probs = np.exp(log_probs)
    one_hot = np.zeros_like(softmax_probs)
    one_hot[np.arange(N), targets_np] = 1.0
    grad_logits = (softmax_probs - one_hot) / N
    return _build_ctx(out, (logits,), lambda g: (g.data * grad_logits,))


def bce_with_logits_loss(logits: Tensor, targets) -> Tensor:
    """Binary cross-entropy from logits, numerically stable.

    Matches torch.nn.functional.binary_cross_entropy_with_logits with the
    default reduction='mean'. Stable formula:
      per_element = max(logits, 0) - logits * targets + log(1 + exp(-|logits|))
      loss = per_element.mean()

    Gradient (derived from sigmoid + cross-entropy):
      d(loss)/d(logit_i) = (sigmoid(logit_i) - target_i) / N
    """
    if isinstance(targets, Tensor):
        t_data = targets.data.astype(np.float32)
    else:
        t_data = np.asarray(targets, dtype=np.float32)
    x = logits.data
    if t_data.shape != x.shape:
        raise ValueError(
            f"bce_with_logits_loss: logits shape {x.shape} ≠ targets shape {t_data.shape}"
        )
    abs_x = np.abs(x)
    max_x_0 = np.maximum(x, 0.0)
    log1pexp = np.log1p(np.exp(-abs_x))
    per_element = max_x_0 - x * t_data + log1pexp
    loss_data = float(per_element.mean())
    out = Tensor(np.float32(loss_data))
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    n = float(x.size)
    grad_logits = (sigmoid - t_data) / n
    return _build_ctx(out, (logits,), lambda g: (g.data * grad_logits.astype(np.float32),))


def one_hot(indices, num_classes: int) -> Tensor:
    """One-hot encode integer indices.

    `indices`: numpy int array (or list / Tensor of integers). Output shape:
    indices.shape + (num_classes,). Non-differentiable; returns a float
    Tensor whose data is 0/1 for downstream f32 ops.
    """
    if isinstance(indices, Tensor):
        idx = indices.data.astype(np.int64)
    else:
        idx = np.asarray(indices, dtype=np.int64)
    if (idx < 0).any() or (idx >= num_classes).any():
        raise ValueError(f"one_hot: indices out of range [0, {num_classes})")
    out_shape = idx.shape + (num_classes,)
    out_data = np.zeros(out_shape, dtype=np.float32)
    flat_idx = idx.flatten()
    flat_out = out_data.reshape(-1, num_classes)
    flat_out[np.arange(flat_idx.size), flat_idx] = 1.0
    return Tensor(out_data)


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """Functional inverted dropout. Matches torch.nn.functional.dropout.

    When training=False or p==0, returns x unchanged.
    """
    if not training or p == 0.0:
        return x
    if not (0.0 <= p < 1.0):
        raise ValueError(f"dropout: p must be in [0, 1), got {p}")
    keep = 1.0 - p
    mask = (np.random.rand(*x.data.shape) < keep).astype(np.float32) / keep
    out = Tensor((x.data * mask).astype(np.float32))
    return _build_ctx(out, (x,), lambda g: (g.data * mask,))


def nll_loss(log_probs: Tensor, targets) -> Tensor:
    """Negative log-likelihood: -mean(log_probs[range(N), targets]).

    Expects `log_probs` to be the output of log_softmax already.
    Prefer `cross_entropy_loss` directly on logits — it's more stable.
    """
    if isinstance(targets, Tensor):
        targets_np = targets.data.astype(np.int64)
    else:
        targets_np = np.asarray(targets, dtype=np.int64)
    if log_probs.data.ndim != 2:
        raise ValueError(f"nll_loss: log_probs must be 2D (N, C), got {log_probs.data.ndim}D")
    N, C = log_probs.data.shape
    if (targets_np < 0).any() or (targets_np >= C).any():
        raise ValueError(f"nll_loss: targets out of range [0, {C})")
    loss_data = float(-log_probs.data[np.arange(N), targets_np].mean())
    out = Tensor(np.float32(loss_data))
    one_hot = np.zeros_like(log_probs.data)
    one_hot[np.arange(N), targets_np] = 1.0
    grad_log_probs = -one_hot / N
    return _build_ctx(out, (log_probs,), lambda g: (g.data * grad_log_probs,))


def _reduce_loss(per_elem: np.ndarray, grad_per_elem: np.ndarray, input_t: Tensor,
                 reduction: str, op_name: str, mean_denom=None) -> Tensor:
    """Shared reduction handler for losses.

    per_elem: per-element loss array.
    grad_per_elem: per-element dLoss/dInput before reduction-scale.
    reduction: 'mean' (scale loss by 1/N and grad by 1/N), 'sum', 'none', or
        any KL-style alias whose denominator is supplied via mean_denom.
    mean_denom: overrides the denominator for mean-style reductions (e.g.
        KL's 'batchmean' uses batch size instead of total element count).
        When None, uses per_elem.size.
    """
    if reduction in ("mean", "batchmean"):
        denom = float(per_elem.size if mean_denom is None else mean_denom)
        out = Tensor(np.float32(float(per_elem.sum()) / denom))
        scale = 1.0 / denom
        return _build_ctx(out, (input_t,), lambda g: ((g.data * grad_per_elem * scale).astype(np.float32),))
    if reduction == "sum":
        out = Tensor(np.float32(float(per_elem.sum())))
        return _build_ctx(out, (input_t,), lambda g: ((g.data * grad_per_elem).astype(np.float32),))
    if reduction == "none":
        out = Tensor(per_elem.astype(np.float32))
        return _build_ctx(out, (input_t,), lambda g: ((g.data * grad_per_elem).astype(np.float32),))
    raise ValueError(f"{op_name}: unknown reduction {reduction!r}")


def l1_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Mean absolute error.

    reduction: 'mean' (default), 'sum', or 'none'. Matches torch.nn.functional.l1_loss
    for the reductions we support.
    """
    if input.data.shape != target.data.shape:
        raise ValueError(f"l1_loss: shape mismatch {input.data.shape} vs {target.data.shape}")
    diff = input.data - target.data
    per_elem = np.abs(diff)
    grad_per_elem = np.sign(diff).astype(np.float32)
    return _reduce_loss(per_elem, grad_per_elem, input, reduction, "l1_loss")


def bce_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Binary cross-entropy from probabilities. Input must be in (0, 1).
    Use bce_with_logits_loss if you have raw logits — it's numerically stable.

    Forward: -(target * log(input) + (1-target) * log(1-input)).reduce()
    """
    if input.data.shape != target.data.shape:
        raise ValueError(f"bce_loss: shape mismatch {input.data.shape} vs {target.data.shape}")
    p = input.data.astype(np.float64)
    t = target.data.astype(np.float64)
    # clamp to avoid log(0) and division by zero in grad
    eps = 1e-12
    p_c = np.clip(p, eps, 1.0 - eps)
    per_elem = -(t * np.log(p_c) + (1.0 - t) * np.log(1.0 - p_c))
    grad_per_elem = ((1.0 - t) / (1.0 - p_c) - t / p_c).astype(np.float32)
    return _reduce_loss(per_elem, grad_per_elem, input, reduction, "bce_loss")


def smooth_l1_loss(input: Tensor, target: Tensor, beta: float = 1.0, reduction: str = "mean") -> Tensor:
    """Smooth L1 (Huber with knee at beta).

    Per element:
      0.5 * d^2 / beta              if |d| < beta
      |d| - 0.5 * beta              otherwise
    """
    if input.data.shape != target.data.shape:
        raise ValueError(f"smooth_l1_loss: shape mismatch {input.data.shape} vs {target.data.shape}")
    if beta <= 0:
        raise ValueError(f"smooth_l1_loss: beta must be > 0, got {beta}")
    d = input.data - target.data
    a = np.abs(d)
    quad = 0.5 * d * d / beta
    lin = a - 0.5 * beta
    per_elem = np.where(a < beta, quad, lin)
    grad_per_elem = np.where(a < beta, d / beta, np.sign(d)).astype(np.float32)
    return _reduce_loss(per_elem, grad_per_elem, input, reduction, "smooth_l1_loss")


def kl_div_loss(input: Tensor, target: Tensor, reduction: str = "mean", log_target: bool = False) -> Tensor:
    """KL divergence. input is expected to be a log-probability (output of log_softmax).
    target is a probability by default; pass log_target=True if you pre-logged it.

    Per element: target * (log(target) - input)
    """
    if input.data.shape != target.data.shape:
        raise ValueError(f"kl_div_loss: shape mismatch {input.data.shape} vs {target.data.shape}")
    if log_target:
        log_t = target.data.astype(np.float64)
        t = np.exp(log_t)
    else:
        t = target.data.astype(np.float64)
        # target * log(target), with the convention 0 * log(0) = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            log_t = np.where(t > 0, np.log(t), 0.0)
    per_elem = t * (log_t - input.data.astype(np.float64))
    grad_per_elem = (-t).astype(np.float32)
    mean_denom = float(input.data.shape[0]) if reduction == "batchmean" else None
    return _reduce_loss(per_elem, grad_per_elem, input, reduction, "kl_div_loss", mean_denom=mean_denom)


# ─── Spatial / shape ops ───────────────────────────────────

def pad(input: Tensor, pad, mode: str = "constant", value: float = 0.0) -> Tensor:
    """Pad input.

    pad: a sequence of even length, paired by dimension, last-dim-first
    (matching torch.nn.functional.pad). E.g. for a 2D input,
    pad=(left, right, top, bottom).

    Currently supports mode='constant' only — that covers nearly every
    real PyTorch lab. Add reflect / replicate when something needs them.
    """
    if mode != "constant":
        raise NotImplementedError(f"pad: mode {mode!r} not supported; only 'constant'")
    if len(pad) % 2 != 0:
        raise ValueError(f"pad: pad length must be even, got {len(pad)}")
    pairs_lastdim_first = list(zip(pad[0::2], pad[1::2]))
    # Convert to numpy's first-dim-first ordering, with zero-pad for any dims
    # that the user didn't specify.
    ndim = input.data.ndim
    npad = [(0, 0)] * ndim
    for k, (lo, hi) in enumerate(pairs_lastdim_first):
        dim = ndim - 1 - k
        npad[dim] = (int(lo), int(hi))
    out_data = np.pad(input.data, npad, mode="constant", constant_values=value)
    out = Tensor(out_data.astype(np.float32))

    def backward(g):
        slices = tuple(slice(lo, lo + s) for (lo, _), s in zip(npad, input.data.shape))
        return (g.data[slices].copy(),)

    return _build_ctx(out, (input,), backward)


def _interp_nearest_2d(x_data, out_h, out_w, scale_h, scale_w):
    H_in, W_in = x_data.shape[-2:]
    # Source-index map per output pixel.
    si = np.floor(np.arange(out_h) / scale_h).astype(np.int64)
    sj = np.floor(np.arange(out_w) / scale_w).astype(np.int64)
    si = np.clip(si, 0, H_in - 1)
    sj = np.clip(sj, 0, W_in - 1)
    return x_data[..., si[:, None], sj[None, :]], si, sj


def _interp_bilinear_2d(x_data, out_h, out_w, scale_h, scale_w, align_corners):
    H_in, W_in = x_data.shape[-2:]
    if align_corners:
        ih = np.linspace(0, H_in - 1, out_h).astype(np.float32) if out_h > 1 else np.zeros(out_h, dtype=np.float32)
        iw = np.linspace(0, W_in - 1, out_w).astype(np.float32) if out_w > 1 else np.zeros(out_w, dtype=np.float32)
    else:
        # Half-pixel-center mapping (PyTorch default).
        ih = (np.arange(out_h, dtype=np.float32) + 0.5) / scale_h - 0.5
        iw = (np.arange(out_w, dtype=np.float32) + 0.5) / scale_w - 0.5
    i0 = np.floor(ih).astype(np.int64); i1 = i0 + 1
    j0 = np.floor(iw).astype(np.int64); j1 = j0 + 1
    a = (ih - i0).astype(np.float32)
    b = (iw - j0).astype(np.float32)
    i0c = np.clip(i0, 0, H_in - 1); i1c = np.clip(i1, 0, H_in - 1)
    j0c = np.clip(j0, 0, W_in - 1); j1c = np.clip(j1, 0, W_in - 1)
    # Gather the 4 corners per output pixel.
    v00 = x_data[..., i0c[:, None], j0c[None, :]]
    v01 = x_data[..., i0c[:, None], j1c[None, :]]
    v10 = x_data[..., i1c[:, None], j0c[None, :]]
    v11 = x_data[..., i1c[:, None], j1c[None, :]]
    aw = a[:, None]; bw = b[None, :]
    out = (1 - aw) * ((1 - bw) * v00 + bw * v01) + aw * ((1 - bw) * v10 + bw * v11)
    return out.astype(np.float32)


def interpolate(input: Tensor, size=None, scale_factor=None, mode: str = "nearest", align_corners: bool = False) -> Tensor:
    """Resize 4D (N, C, H, W) feature maps.

    Supports mode in {'nearest', 'bilinear'}. Either size or scale_factor must
    be given. Backward path is implemented but slow — fine for educational use.
    """
    if input.data.ndim != 4:
        raise NotImplementedError(f"interpolate: only 4D input supported; got {input.data.ndim}D")
    H_in, W_in = input.data.shape[-2:]
    if size is not None:
        out_h, out_w = int(size[0]), int(size[1])
    elif scale_factor is not None:
        sf = scale_factor
        if isinstance(sf, (int, float)):
            sf = (sf, sf)
        out_h = int(round(H_in * sf[0]))
        out_w = int(round(W_in * sf[1]))
    else:
        raise ValueError("interpolate: provide size or scale_factor")
    scale_h = out_h / H_in
    scale_w = out_w / W_in

    if mode == "nearest":
        out_data, si, sj = _interp_nearest_2d(input.data, out_h, out_w, scale_h, scale_w)
        out = Tensor(out_data.astype(np.float32))
        def backward(g):
            # Scatter-add gradients back to source positions.
            dx = np.zeros_like(input.data)
            # g shape: (..., out_h, out_w); we add g[..., y, x] to dx[..., si[y], sj[x]].
            for y in range(out_h):
                for x in range(out_w):
                    dx[..., si[y], sj[x]] += g.data[..., y, x]
            return (dx,)
        return _build_ctx(out, (input,), backward)

    if mode == "bilinear":
        out_data = _interp_bilinear_2d(input.data, out_h, out_w, scale_h, scale_w, align_corners)
        out = Tensor(out_data)
        # Backward: numerical for the educational case (small enough).
        def backward(g):
            # Build the bilinear weight tensor and apply its transpose.
            if align_corners:
                ih = np.linspace(0, H_in - 1, out_h).astype(np.float32) if out_h > 1 else np.zeros(out_h, dtype=np.float32)
                iw = np.linspace(0, W_in - 1, out_w).astype(np.float32) if out_w > 1 else np.zeros(out_w, dtype=np.float32)
            else:
                ih = (np.arange(out_h, dtype=np.float32) + 0.5) / scale_h - 0.5
                iw = (np.arange(out_w, dtype=np.float32) + 0.5) / scale_w - 0.5
            i0 = np.floor(ih).astype(np.int64); i1 = i0 + 1
            j0 = np.floor(iw).astype(np.int64); j1 = j0 + 1
            a = (ih - i0).astype(np.float32)
            b = (iw - j0).astype(np.float32)
            i0c = np.clip(i0, 0, H_in - 1); i1c = np.clip(i1, 0, H_in - 1)
            j0c = np.clip(j0, 0, W_in - 1); j1c = np.clip(j1, 0, W_in - 1)
            dx = np.zeros_like(input.data)
            for y in range(out_h):
                for x in range(out_w):
                    g_yx = g.data[..., y, x]
                    aw = a[y]; bw = b[x]
                    dx[..., i0c[y], j0c[x]] += g_yx * (1 - aw) * (1 - bw)
                    dx[..., i0c[y], j1c[x]] += g_yx * (1 - aw) * bw
                    dx[..., i1c[y], j0c[x]] += g_yx * aw * (1 - bw)
                    dx[..., i1c[y], j1c[x]] += g_yx * aw * bw
            return (dx,)
        return _build_ctx(out, (input,), backward)

    raise NotImplementedError(f"interpolate: mode {mode!r} not supported")


def normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12) -> Tensor:
    """L_p normalize along dim. Default p=2, dim=1 (matches torch.nn.functional.normalize)."""
    if p != 2.0:
        raise NotImplementedError(f"normalize: only p=2 supported; got p={p}")
    norm = np.sqrt(np.sum(input.data * input.data, axis=dim, keepdims=True))
    norm = np.maximum(norm, eps)
    out = Tensor((input.data / norm).astype(np.float32))
    def backward(g):
        # d/dx_i of (x_i / ||x||) = 1/||x|| - x_i * x_i / ||x||^3 (along the normalized dim)
        # In vector form: (I - out * out^T) * g / ||x||
        x = input.data
        dot = np.sum(out.data * g.data, axis=dim, keepdims=True)
        dx = (g.data - out.data * dot) / norm
        return (dx.astype(np.float32),)
    return _build_ctx(out, (input,), backward)


def cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    """Cosine similarity along dim. Returns a tensor of one fewer dim."""
    a = x1.data
    b = x2.data
    na = np.sqrt(np.sum(a * a, axis=dim, keepdims=True))
    nb = np.sqrt(np.sum(b * b, axis=dim, keepdims=True))
    na = np.maximum(na, eps); nb = np.maximum(nb, eps)
    dot = np.sum(a * b, axis=dim, keepdims=True)
    out_data = (dot / (na * nb)).squeeze(axis=dim).astype(np.float32)
    out = Tensor(out_data)
    # Backward omitted (rarely needed); raise on call.
    def backward(g):
        raise NotImplementedError("cosine_similarity backward not implemented yet")
    return _build_ctx(out, (x1, x2), backward)


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor,
                                  attn_mask=None, dropout_p: float = 0.0,
                                  is_causal: bool = False, scale=None) -> Tensor:
    """Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V.

    Supports the subset of torch's API that matters most:
      - attn_mask: boolean (True = block) or float (added to scores).
      - is_causal: builds a triangular block mask.
      - scale: override 1/sqrt(d_k).

    Last dim of Q and K must agree; last-but-one of K and V must agree.
    """
    if dropout_p != 0.0:
        raise NotImplementedError("scaled_dot_product_attention: dropout_p > 0 not supported")
    Qd = query.data
    Kd = key.data
    Vd = value.data
    d_k = Qd.shape[-1]
    s = (1.0 / np.sqrt(d_k)) if scale is None else float(scale)
    # scores: (..., L_q, L_k)
    scores = np.matmul(Qd, np.swapaxes(Kd, -1, -2)) * s
    if is_causal:
        L_q, L_k = scores.shape[-2], scores.shape[-1]
        tri = np.triu(np.ones((L_q, L_k), dtype=bool), k=1)
        scores = np.where(tri, -np.inf, scores)
    if attn_mask is not None:
        m = np.asarray(attn_mask)
        if m.dtype == bool:
            scores = np.where(m, -np.inf, scores)
        else:
            scores = scores + m
    # softmax along last axis, stable.
    sm = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(sm)
    attn = e / e.sum(axis=-1, keepdims=True)
    out_data = np.matmul(attn, Vd).astype(np.float32)
    out = Tensor(out_data)
    def backward(g):
        # Educational backward through the explicit formula.
        # dV = attn^T @ g
        dV = np.matmul(np.swapaxes(attn, -1, -2), g.data)
        # dscores via softmax Jacobian: row-wise (diag(p) - pp^T) @ (g @ V^T)
        ga = np.matmul(g.data, np.swapaxes(Vd, -1, -2))
        row_sum = np.sum(ga * attn, axis=-1, keepdims=True)
        dscores = attn * (ga - row_sum)
        dscores = dscores * s
        dQ = np.matmul(dscores, Kd)
        dK = np.matmul(np.swapaxes(dscores, -1, -2), Qd)
        return (dQ.astype(np.float32), dK.astype(np.float32), dV.astype(np.float32))
    return _build_ctx(out, (query, key, value), backward)
