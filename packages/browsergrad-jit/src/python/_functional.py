"""browsergrad_jit._functional — functional API matching torch.nn.functional.

INTERNAL module. Users import as `browsergrad_jit.nn.functional as F`.

PRD-005 cut-down scope (per the critique): elementwise + MLP-style ops
needed for the 0.1.0 conformance bar. Conv, pool, attention, norm,
embedding, recurrent ship in 0.1.1/0.1.2 patches via the CUSTOM opcode
or as proper IR additions in PRD-006+.

The pattern across every op: build new UOps + new TensorProxies with the
right backward closures. NumPy-heavy logic that doesn't decompose into
the primitive opcode set lives behind a CUSTOM op so it still participates
in the IR (just opaque to fusion).
"""

from __future__ import annotations
from typing import Any, Optional, Tuple

import numpy as np

from ._ir import (
    UOp, OP_WHERE, OP_CONST, OP_CUSTOM, OP_REDUCE, OP_DIV,
)
from ._tensor_proxy import (
    TensorProxy, _BackwardCtx, _should_track, _to_proxy, from_numpy, where,
)
from ._errors import ShapeError


# ---------------------------------------------------------------------------
# ReLU & friends
# ---------------------------------------------------------------------------


def relu(x: TensorProxy) -> TensorProxy:
    """Standard ReLU: max(x, 0). Backward: dy * (x > 0)."""
    sess = x._get_session()
    # Express as WHERE(x > 0, x, 0). CMP+WHERE is one of our supported
    # primitive paths.
    zero = UOp(op=OP_CONST, inputs=(), shape=(), dtype=x.dtype, arg={"value": 0.0})
    cond_uop = UOp(op="CMP", inputs=(x._uop, zero), shape=x.shape, dtype="bool",
                   arg={"op": "gt"})
    uop = UOp(op=OP_WHERE, inputs=(cond_uop, x._uop, zero), shape=x.shape,
              dtype=x.dtype, arg=None)

    def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
        # ins is (x_arr,) — we passed `x` as the only input proxy.
        (x_arr,) = ins
        return (dy * (x_arr > 0).astype(dy.dtype),)

    requires = _should_track(x)
    ctx = _BackwardCtx(fn=_bw, input_proxies=(x,)) if requires else None
    return TensorProxy(uop, session=sess, requires_grad=requires, ctx=ctx)


def sigmoid(x: TensorProxy) -> TensorProxy:
    """sigmoid(x) = 1 / (1 + exp(-x)). Closed-form backward uses output."""
    # Reuse the existing ops; for numerical stability with large negative x
    # we use a CUSTOM-free form: (-x).exp(); 1.0 / (1.0 + e).
    e = (-x).exp()
    one = _to_proxy(1.0, x._get_session())
    return one / (one + e)


def tanh(x: TensorProxy) -> TensorProxy:
    """tanh(x) = (e^x - e^-x) / (e^x + e^-x)."""
    e_pos = x.exp()
    e_neg = (-x).exp()
    return (e_pos - e_neg) / (e_pos + e_neg)


def gelu(x: TensorProxy) -> TensorProxy:
    """GELU using the tanh approximation: matches PyTorch nn.functional.gelu."""
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = (2.0 / np.pi) ** 0.5
    return 0.5 * x * (_to_proxy(1.0, x._get_session()) + tanh(_to_proxy(c, x._get_session()) * (x + 0.044715 * x * x * x)))


def silu(x: TensorProxy) -> TensorProxy:
    """SiLU / swish: x * sigmoid(x)."""
    return x * sigmoid(x)


def leaky_relu(x: TensorProxy, negative_slope: float = 0.01) -> TensorProxy:
    """Leaky ReLU: x for positive values, negative_slope * x otherwise."""
    return where(x > 0, x, x * float(negative_slope))


# ---------------------------------------------------------------------------
# Softmax + cross-entropy
# ---------------------------------------------------------------------------


def softmax(x: TensorProxy, dim: int = -1) -> TensorProxy:
    """softmax along `dim`. Numerically stable (subtract row-max).

    Decomposed into primitive ops so fusion (PRD-006) can see it later.
    """
    # x_max = x.max(axis=dim, keepdims=True)
    # shifted = x - x_max
    # ex = shifted.exp()
    # s = ex.sum(axis=dim, keepdims=True)
    # softmax = ex / s
    x_max = x.max(axis=dim, keepdims=True)
    shifted = x - x_max
    e = shifted.exp()
    s = e.sum(axis=dim, keepdims=True)
    return e / s


def log_softmax(x: TensorProxy, dim: int = -1) -> TensorProxy:
    """log(softmax(x)). Computed in the numerically-stable form."""
    x_max = x.max(axis=dim, keepdims=True)
    shifted = x - x_max
    e = shifted.exp()
    s = e.sum(axis=dim, keepdims=True)
    return shifted - s.log()


def cross_entropy(logits: TensorProxy, targets: TensorProxy,
                  reduction: str = "mean") -> TensorProxy:
    """Cross-entropy loss matching torch.nn.functional.cross_entropy.

    `logits`: shape (N, C) — raw scores.
    `targets`: shape (N,) — integer class indices.
    """
    if logits.ndim != 2:
        raise ShapeError(
            f"cross_entropy: logits must be 2-D (N, C), got shape {logits.shape}"
        )
    if targets.ndim != 1:
        raise ShapeError(
            f"cross_entropy: targets must be 1-D (N,), got shape {targets.shape}"
        )

    # Build the loss as a CUSTOM op for v0 — the gather-by-targets pattern
    # is awkward to express without INDEX-with-batched-dim yet. PRD-006
    # decomposes this into primitive ops.
    sess = logits._get_session()
    N, C = logits.shape

    def _ce_forward(logits_arr: np.ndarray, targets_arr: np.ndarray) -> np.ndarray:
        # Standard numerically-stable CE.
        x_max = logits_arr.max(axis=-1, keepdims=True)
        shifted = logits_arr - x_max
        log_sum_exp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        log_probs = shifted - log_sum_exp  # (N, C)
        nll = -log_probs[np.arange(N), targets_arr.astype(np.int64)]
        if reduction == "mean":
            return np.asarray(nll.mean(), dtype=logits_arr.dtype)
        if reduction == "sum":
            return np.asarray(nll.sum(), dtype=logits_arr.dtype)
        if reduction == "none":
            return nll.astype(logits_arr.dtype, copy=False)
        raise ValueError(f"cross_entropy: unknown reduction {reduction!r}")

    out_shape: Tuple[int, ...]
    if reduction in ("mean", "sum"):
        out_shape = ()
    else:
        out_shape = (N,)

    uop = UOp(
        op=OP_CUSTOM,
        inputs=(logits._uop, targets._uop),
        shape=out_shape,
        dtype=logits.dtype,
        arg={"fn": _ce_forward, "captures": (), "name": "cross_entropy"},
    )

    def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
        logits_arr, targets_arr = ins
        # softmax probs - one_hot(targets)
        x_max = logits_arr.max(axis=-1, keepdims=True)
        shifted = logits_arr - x_max
        e = np.exp(shifted)
        probs = e / e.sum(axis=-1, keepdims=True)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(N), targets_arr.astype(np.int64)] = 1.0
        grad_logits = probs - one_hot
        if reduction == "mean":
            grad_logits *= (dy / N)
        elif reduction == "sum":
            grad_logits *= dy
        else:  # none — dy shape (N,)
            grad_logits *= dy[:, None]
        # targets has no gradient.
        return (grad_logits.astype(logits_arr.dtype, copy=False), None)

    requires = _should_track(logits)
    ctx = _BackwardCtx(fn=_bw, input_proxies=(logits, targets)) if requires else None
    return TensorProxy(uop, session=sess, requires_grad=requires, ctx=ctx)


def mse_loss(input: TensorProxy, target: TensorProxy,
             reduction: str = "mean") -> TensorProxy:
    """Mean-squared-error loss. Equivalent to ((input - target)**2).mean()."""
    diff = input - target
    sq = diff * diff
    if reduction == "mean":
        return sq.mean()
    if reduction == "sum":
        return sq.sum()
    if reduction == "none":
        return sq
    raise ValueError(f"mse_loss: unknown reduction {reduction!r}")


def nll_loss(log_probs: TensorProxy, targets: TensorProxy,
             reduction: str = "mean") -> TensorProxy:
    """Negative log likelihood: matches torch.nn.functional.nll_loss.

    `log_probs`: shape (N, C). `targets`: shape (N,)."""
    if log_probs.ndim != 2:
        raise ShapeError(
            f"nll_loss: log_probs must be 2-D, got shape {log_probs.shape}"
        )
    N, C = log_probs.shape
    sess = log_probs._get_session()

    def _nll_forward(lp_arr: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
        picked = -lp_arr[np.arange(N), t_arr.astype(np.int64)]
        if reduction == "mean":
            return np.asarray(picked.mean(), dtype=lp_arr.dtype)
        if reduction == "sum":
            return np.asarray(picked.sum(), dtype=lp_arr.dtype)
        return picked.astype(lp_arr.dtype, copy=False)

    out_shape: Tuple[int, ...] = () if reduction in ("mean", "sum") else (N,)
    uop = UOp(
        op=OP_CUSTOM,
        inputs=(log_probs._uop, targets._uop),
        shape=out_shape,
        dtype=log_probs.dtype,
        arg={"fn": _nll_forward, "captures": (), "name": "nll_loss"},
    )

    def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
        lp_arr, t_arr = ins
        grad_lp = np.zeros_like(lp_arr)
        rows = np.arange(N)
        cols = t_arr.astype(np.int64)
        if reduction == "mean":
            grad_lp[rows, cols] = -dy / N
        elif reduction == "sum":
            grad_lp[rows, cols] = -dy
        else:  # none
            grad_lp[rows, cols] = -dy
        return (grad_lp.astype(lp_arr.dtype, copy=False), None)

    requires = _should_track(log_probs)
    ctx = _BackwardCtx(fn=_bw, input_proxies=(log_probs, targets)) if requires else None
    return TensorProxy(uop, session=sess, requires_grad=requires, ctx=ctx)


def _custom_elementwise_loss(
    input: TensorProxy,
    target: TensorProxy,
    reduction: str,
    op_name: str,
    forward_fn: Any,
    grad_fn: Any,
    *,
    allow_batchmean: bool = False,
) -> TensorProxy:
    target = _to_proxy(target, input._get_session())
    if input.shape != target.shape:
        raise ShapeError(f"{op_name}: shape mismatch {input.shape} vs {target.shape}")
    valid_reductions = ("mean", "sum", "none", "batchmean") if allow_batchmean else (
        "mean", "sum", "none"
    )
    if reduction not in valid_reductions:
        raise ValueError(f"{op_name}: unknown reduction {reduction!r}")

    sess = input._get_session()
    out_shape: Tuple[int, ...] = () if reduction != "none" else input.shape

    def _forward(input_arr: np.ndarray, target_arr: np.ndarray) -> np.ndarray:
        per_elem = forward_fn(input_arr, target_arr).astype(np.float32, copy=False)
        if reduction == "none":
            return per_elem.astype(np.dtype(input.dtype), copy=False)
        if reduction == "sum":
            return np.asarray(per_elem.sum(), dtype=np.dtype(input.dtype))
        denom = float(input_arr.shape[0]) if reduction == "batchmean" else float(per_elem.size)
        return np.asarray(per_elem.sum() / denom, dtype=np.dtype(input.dtype))

    uop = UOp(
        op=OP_CUSTOM,
        inputs=(input._uop, target._uop),
        shape=out_shape,
        dtype=input.dtype,
        arg={"fn": _forward, "captures": (), "name": op_name},
    )

    def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
        input_arr, target_arr = ins
        grad = grad_fn(input_arr, target_arr).astype(np.float32, copy=False)
        if reduction == "mean":
            grad *= dy / float(input_arr.size)
        elif reduction == "sum":
            grad *= dy
        elif reduction == "batchmean":
            grad *= dy / float(input_arr.shape[0])
        else:
            grad *= dy
        return (grad.astype(input_arr.dtype, copy=False), None)

    requires = _should_track(input)
    ctx = _BackwardCtx(fn=_bw, input_proxies=(input, target)) if requires else None
    return TensorProxy(uop, session=sess, requires_grad=requires, ctx=ctx)


def l1_loss(
    input: TensorProxy,
    target: TensorProxy,
    reduction: str = "mean",
) -> TensorProxy:
    def _forward(input_arr: np.ndarray, target_arr: np.ndarray) -> np.ndarray:
        return np.abs(input_arr - target_arr)

    def _grad(input_arr: np.ndarray, target_arr: np.ndarray) -> np.ndarray:
        return np.sign(input_arr - target_arr)

    return _custom_elementwise_loss(input, target, reduction, "l1_loss", _forward, _grad)


def binary_cross_entropy(
    input: TensorProxy,
    target: TensorProxy,
    reduction: str = "mean",
) -> TensorProxy:
    def _forward(input_arr: np.ndarray, target_arr: np.ndarray) -> np.ndarray:
        p = input_arr.astype(np.float64, copy=False)
        t = target_arr.astype(np.float64, copy=False)
        eps = 1e-12
        p_clamped = np.clip(p, eps, 1.0 - eps)
        return -(t * np.log(p_clamped) + (1.0 - t) * np.log(1.0 - p_clamped))

    def _grad(input_arr: np.ndarray, target_arr: np.ndarray) -> np.ndarray:
        p = input_arr.astype(np.float64, copy=False)
        t = target_arr.astype(np.float64, copy=False)
        eps = 1e-12
        p_clamped = np.clip(p, eps, 1.0 - eps)
        return (1.0 - t) / (1.0 - p_clamped) - t / p_clamped

    return _custom_elementwise_loss(
        input, target, reduction, "binary_cross_entropy", _forward, _grad
    )


def smooth_l1_loss(
    input: TensorProxy,
    target: TensorProxy,
    beta: float = 1.0,
    reduction: str = "mean",
) -> TensorProxy:
    if beta <= 0:
        raise ValueError(f"smooth_l1_loss: beta must be > 0, got {beta}")

    def _forward(input_arr: np.ndarray, target_arr: np.ndarray) -> np.ndarray:
        diff = input_arr - target_arr
        abs_diff = np.abs(diff)
        return np.where(abs_diff < beta, 0.5 * diff * diff / beta, abs_diff - 0.5 * beta)

    def _grad(input_arr: np.ndarray, target_arr: np.ndarray) -> np.ndarray:
        diff = input_arr - target_arr
        abs_diff = np.abs(diff)
        return np.where(abs_diff < beta, diff / beta, np.sign(diff))

    return _custom_elementwise_loss(
        input, target, reduction, "smooth_l1_loss", _forward, _grad
    )


def kl_div(
    input: TensorProxy,
    target: TensorProxy,
    reduction: str = "mean",
    log_target: bool = False,
) -> TensorProxy:
    def _forward(input_arr: np.ndarray, target_arr: np.ndarray) -> np.ndarray:
        if log_target:
            log_t = target_arr.astype(np.float64, copy=False)
            t = np.exp(log_t)
        else:
            t = target_arr.astype(np.float64, copy=False)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_t = np.where(t > 0, np.log(t), 0.0)
        return t * (log_t - input_arr.astype(np.float64, copy=False))

    def _grad(input_arr: np.ndarray, target_arr: np.ndarray) -> np.ndarray:
        if log_target:
            return -np.exp(target_arr.astype(np.float64, copy=False))
        return -target_arr

    return _custom_elementwise_loss(
        input,
        target,
        reduction,
        "kl_div",
        _forward,
        _grad,
        allow_batchmean=True,
    )


def kl_div_loss(
    input: TensorProxy,
    target: TensorProxy,
    reduction: str = "mean",
    log_target: bool = False,
) -> TensorProxy:
    return kl_div(input, target, reduction=reduction, log_target=log_target)


def binary_cross_entropy_with_logits(
    logits: TensorProxy,
    targets: TensorProxy,
    reduction: str = "mean",
) -> TensorProxy:
    """Binary cross-entropy from raw logits.

    Stable formula:
      max(x, 0) - x*y + log(1 + exp(-abs(x)))
    Backward:
      sigmoid(x) - y, scaled by the reduction.
    """
    targets = _to_proxy(targets, logits._get_session())
    if logits.shape != targets.shape:
        raise ShapeError(
            "binary_cross_entropy_with_logits: logits and targets must have "
            f"the same shape, got {logits.shape} vs {targets.shape}"
        )
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(
            f"binary_cross_entropy_with_logits: unknown reduction {reduction!r}"
        )
    sess = logits._get_session()

    def _bce_forward(logits_arr: np.ndarray, targets_arr: np.ndarray) -> np.ndarray:
        x = logits_arr.astype(np.float32, copy=False)
        y = targets_arr.astype(np.float32, copy=False)
        per_elem = np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))
        if reduction == "mean":
            return np.asarray(per_elem.mean(), dtype=x.dtype)
        if reduction == "sum":
            return np.asarray(per_elem.sum(), dtype=x.dtype)
        return per_elem.astype(x.dtype, copy=False)

    out_shape: Tuple[int, ...] = () if reduction in ("mean", "sum") else logits.shape
    uop = UOp(
        op=OP_CUSTOM,
        inputs=(logits._uop, targets._uop),
        shape=out_shape,
        dtype=logits.dtype,
        arg={
            "fn": _bce_forward,
            "captures": (),
            "name": "binary_cross_entropy_with_logits",
        },
    )

    def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
        logits_arr, targets_arr = ins
        sigmoid = np.empty_like(logits_arr, dtype=np.float32)
        positive = logits_arr >= 0
        sigmoid[positive] = 1.0 / (1.0 + np.exp(-logits_arr[positive]))
        exp_x = np.exp(logits_arr[~positive])
        sigmoid[~positive] = exp_x / (1.0 + exp_x)
        grad_logits = sigmoid - targets_arr
        if reduction == "mean":
            grad_logits *= dy / float(logits_arr.size)
        elif reduction == "sum":
            grad_logits *= dy
        else:
            grad_logits *= dy
        return (grad_logits.astype(logits_arr.dtype, copy=False), None)

    requires = _should_track(logits)
    ctx = _BackwardCtx(fn=_bw, input_proxies=(logits, targets)) if requires else None
    return TensorProxy(uop, session=sess, requires_grad=requires, ctx=ctx)


def one_hot(indices: Any, num_classes: int) -> TensorProxy:
    """One-hot encode integer indices.

    Mirrors browsergrad_grad's teaching-friendly choice: output is float32
    so it composes naturally with downstream f32 tensor ops.
    """
    sess = None
    if isinstance(indices, TensorProxy):
        sess = indices._get_session()
        idx = indices._realize_array().astype(np.int64)
    else:
        idx = np.asarray(indices, dtype=np.int64)
    if (idx < 0).any() or (idx >= num_classes).any():
        raise ValueError(f"one_hot: indices out of range [0, {num_classes})")
    out_shape = idx.shape + (int(num_classes),)
    out_data = np.zeros(out_shape, dtype=np.float32)
    flat_idx = idx.reshape(-1)
    flat_out = out_data.reshape(-1, int(num_classes))
    flat_out[np.arange(flat_idx.size), flat_idx] = 1.0
    return from_numpy(out_data, session=sess)


def bce_with_logits_loss(
    logits: TensorProxy,
    targets: TensorProxy,
    reduction: str = "mean",
) -> TensorProxy:
    return binary_cross_entropy_with_logits(logits, targets, reduction=reduction)


def bce_loss(
    input: TensorProxy,
    target: TensorProxy,
    reduction: str = "mean",
) -> TensorProxy:
    return binary_cross_entropy(input, target, reduction=reduction)


def cross_entropy_loss(
    logits: TensorProxy,
    targets: TensorProxy,
    reduction: str = "mean",
) -> TensorProxy:
    return cross_entropy(logits, targets, reduction=reduction)


# ---------------------------------------------------------------------------
# Linear (the workhorse of MLPs)
# ---------------------------------------------------------------------------


def linear(x: TensorProxy, weight: TensorProxy,
           bias: Optional[TensorProxy] = None) -> TensorProxy:
    """y = x @ weight.T + bias. Same contract as torch.nn.functional.linear."""
    # PyTorch's nn.Linear stores weight as (out_features, in_features), so
    # the math is x @ W.T. Matching that convention here means we can swap
    # state_dicts with PyTorch later.
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


# ---------------------------------------------------------------------------
# Dropout — needs RANDOM but for v0 we keep it eager (PRD-005 critique
# documents this as acceptable; the RANDOM opcode lands when we wire it
# into the IR-cached path).
# ---------------------------------------------------------------------------


def dropout(x: TensorProxy, p: float = 0.5, training: bool = True) -> TensorProxy:
    """Standard dropout. At training time, zero each element independently
    with probability `p`; scale survivors by 1/(1-p). At eval time, identity.
    """
    if not training or p == 0.0:
        return x
    if p < 0.0 or p > 1.0:
        raise ValueError(f"dropout p must be in [0, 1], got {p}")
    if p == 1.0:
        return x * _to_proxy(0.0, x._get_session())
    # CUSTOM op: we sample the mask at forward time; backward uses the same
    # mask via the closure capture.
    sess = x._get_session()

    def _drop_forward(x_arr: np.ndarray) -> np.ndarray:
        mask = (np.random.rand(*x_arr.shape) > p).astype(x_arr.dtype)
        # Persist mask on the closure via mutable container — captured below.
        captured["mask"] = mask
        return (x_arr * mask) / (1.0 - p)

    captured: dict = {}
    uop = UOp(
        op=OP_CUSTOM,
        inputs=(x._uop,),
        shape=x.shape,
        dtype=x.dtype,
        arg={"fn": _drop_forward, "captures": (), "name": "dropout"},
    )

    def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
        mask = captured.get("mask")
        if mask is None:
            # Forward never ran — shouldn't happen on a normal backward.
            return (dy.copy(),)
        return ((dy * mask) / (1.0 - p),)

    requires = _should_track(x)
    ctx = _BackwardCtx(fn=_bw, input_proxies=(x,)) if requires else None
    return TensorProxy(uop, session=sess, requires_grad=requires, ctx=ctx)


def pad(input: TensorProxy, pad, mode: str = "constant", value: float = 0.0) -> TensorProxy:
    if mode != "constant":
        raise NotImplementedError(f"pad: mode {mode!r} not supported; only 'constant'")
    if len(pad) % 2 != 0:
        raise ValueError(f"pad: pad length must be even, got {len(pad)}")
    pairs_lastdim_first = list(zip(pad[0::2], pad[1::2]))
    ndim = input.ndim
    npad = [(0, 0)] * ndim
    for k, (lo, hi) in enumerate(pairs_lastdim_first):
        dim = ndim - 1 - k
        npad[dim] = (int(lo), int(hi))
    out_shape = tuple(input.shape[i] + lo + hi for i, (lo, hi) in enumerate(npad))

    def _pad_forward(x_arr: np.ndarray) -> np.ndarray:
        return np.pad(x_arr, npad, mode="constant", constant_values=value)

    uop = UOp(
        op=OP_CUSTOM,
        inputs=(input._uop,),
        shape=out_shape,
        dtype=input.dtype,
        arg={"fn": _pad_forward, "captures": (), "name": "pad"},
    )

    def _bw(dy: np.ndarray, _ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
        slices = tuple(slice(lo, lo + size) for (lo, _), size in zip(npad, input.shape))
        return (dy[slices].copy().astype(np.dtype(input.dtype), copy=False),)

    requires = _should_track(input)
    ctx = _BackwardCtx(fn=_bw, input_proxies=(input,)) if requires else None
    return TensorProxy(uop, session=input._get_session(), requires_grad=requires, ctx=ctx)


def normalize(input: TensorProxy, p: float = 2.0, dim: int = 1, eps: float = 1e-12) -> TensorProxy:
    if p != 2.0:
        raise NotImplementedError(f"normalize: only p=2 supported; got p={p}")
    norm = (input * input).sum(dim=dim, keepdim=True).sqrt().clamp_min(eps)
    return input / norm


def cosine_similarity(x1: TensorProxy, x2: TensorProxy, dim: int = 1, eps: float = 1e-8) -> TensorProxy:
    x2 = _to_proxy(x2, x1._get_session())
    dot = (x1 * x2).sum(dim=dim, keepdim=True)
    n1 = (x1 * x1).sum(dim=dim, keepdim=True).sqrt().clamp_min(eps)
    n2 = (x2 * x2).sum(dim=dim, keepdim=True).sqrt().clamp_min(eps)
    return (dot / (n1 * n2)).squeeze(dim=dim)


def _interp_nearest_2d(x_data: np.ndarray, out_h: int, out_w: int, scale_h: float, scale_w: float):
    h_in, w_in = x_data.shape[-2:]
    si = np.floor(np.arange(out_h) / scale_h).astype(np.int64)
    sj = np.floor(np.arange(out_w) / scale_w).astype(np.int64)
    si = np.clip(si, 0, h_in - 1)
    sj = np.clip(sj, 0, w_in - 1)
    return x_data[..., si[:, None], sj[None, :]], si, sj


def _interp_bilinear_2d(
    x_data: np.ndarray,
    out_h: int,
    out_w: int,
    scale_h: float,
    scale_w: float,
    align_corners: bool,
) -> np.ndarray:
    h_in, w_in = x_data.shape[-2:]
    if align_corners:
        ih = np.linspace(0, h_in - 1, out_h).astype(np.float32) if out_h > 1 else np.zeros(out_h, dtype=np.float32)
        iw = np.linspace(0, w_in - 1, out_w).astype(np.float32) if out_w > 1 else np.zeros(out_w, dtype=np.float32)
    else:
        ih = (np.arange(out_h, dtype=np.float32) + 0.5) / scale_h - 0.5
        iw = (np.arange(out_w, dtype=np.float32) + 0.5) / scale_w - 0.5
    i0 = np.floor(ih).astype(np.int64)
    i1 = i0 + 1
    j0 = np.floor(iw).astype(np.int64)
    j1 = j0 + 1
    a = (ih - i0).astype(np.float32)
    b = (iw - j0).astype(np.float32)
    i0c = np.clip(i0, 0, h_in - 1)
    i1c = np.clip(i1, 0, h_in - 1)
    j0c = np.clip(j0, 0, w_in - 1)
    j1c = np.clip(j1, 0, w_in - 1)
    v00 = x_data[..., i0c[:, None], j0c[None, :]]
    v01 = x_data[..., i0c[:, None], j1c[None, :]]
    v10 = x_data[..., i1c[:, None], j0c[None, :]]
    v11 = x_data[..., i1c[:, None], j1c[None, :]]
    aw = a[:, None]
    bw = b[None, :]
    out = (1 - aw) * ((1 - bw) * v00 + bw * v01) + aw * ((1 - bw) * v10 + bw * v11)
    return out.astype(np.float32)


def interpolate(
    input: TensorProxy,
    size=None,
    scale_factor=None,
    mode: str = "nearest",
    align_corners: bool = False,
) -> TensorProxy:
    if input.ndim != 4:
        raise NotImplementedError(f"interpolate: only 4D input supported; got {input.ndim}D")
    h_in, w_in = input.shape[-2:]
    if size is not None:
        out_h, out_w = int(size[0]), int(size[1])
    elif scale_factor is not None:
        sf = scale_factor if isinstance(scale_factor, (tuple, list)) else (scale_factor, scale_factor)
        out_h = int(round(h_in * sf[0]))
        out_w = int(round(w_in * sf[1]))
    else:
        raise ValueError("interpolate: provide size or scale_factor")
    scale_h = out_h / h_in
    scale_w = out_w / w_in
    out_shape = input.shape[:-2] + (out_h, out_w)

    captured: dict = {}

    def _interpolate_forward(x_arr: np.ndarray) -> np.ndarray:
        if mode == "nearest":
            out, si, sj = _interp_nearest_2d(x_arr, out_h, out_w, scale_h, scale_w)
            captured["nearest"] = (si, sj)
            return out.astype(np.dtype(input.dtype), copy=False)
        if mode == "bilinear":
            return _interp_bilinear_2d(x_arr, out_h, out_w, scale_h, scale_w, align_corners).astype(np.dtype(input.dtype), copy=False)
        raise NotImplementedError(f"interpolate: mode {mode!r} not supported")

    uop = UOp(
        op=OP_CUSTOM,
        inputs=(input._uop,),
        shape=out_shape,
        dtype=input.dtype,
        arg={"fn": _interpolate_forward, "captures": (), "name": "interpolate"},
    )

    def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
        (x_arr,) = ins
        dx = np.zeros_like(x_arr)
        if mode == "nearest":
            si, sj = captured["nearest"]
            for y in range(out_h):
                for x in range(out_w):
                    dx[..., si[y], sj[x]] += dy[..., y, x]
            return (dx.astype(x_arr.dtype, copy=False),)
        raise NotImplementedError("interpolate: bilinear backward not implemented in JIT yet")

    requires = _should_track(input)
    ctx = _BackwardCtx(fn=_bw, input_proxies=(input,)) if requires else None
    return TensorProxy(uop, session=input._get_session(), requires_grad=requires, ctx=ctx)


def scaled_dot_product_attention(
    query: TensorProxy,
    key: TensorProxy,
    value: TensorProxy,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale=None,
) -> TensorProxy:
    if dropout_p != 0.0:
        raise NotImplementedError("scaled_dot_product_attention: dropout_p > 0 not supported")
    s = (1.0 / np.sqrt(query.shape[-1])) if scale is None else float(scale)
    scores = (query @ key.transpose(-1, -2)) * s
    if is_causal:
        l_q, l_k = scores.shape[-2], scores.shape[-1]
        mask = np.triu(np.ones((l_q, l_k), dtype=bool), k=1)
        scores = scores.masked_fill(from_numpy(mask, session=query._get_session()), -np.inf)
    if attn_mask is not None:
        mask_proxy = _to_proxy(attn_mask, query._get_session())
        if mask_proxy.dtype == "bool":
            scores = scores.masked_fill(mask_proxy, -np.inf)
        else:
            scores = scores + mask_proxy
    attn = softmax(scores, dim=-1)
    return attn @ value


__all__ = [
    "relu",
    "sigmoid",
    "tanh",
    "gelu",
    "silu",
    "leaky_relu",
    "softmax",
    "log_softmax",
    "cross_entropy",
    "cross_entropy_loss",
    "mse_loss",
    "nll_loss",
    "l1_loss",
    "binary_cross_entropy",
    "bce_loss",
    "smooth_l1_loss",
    "kl_div",
    "kl_div_loss",
    "binary_cross_entropy_with_logits",
    "bce_with_logits_loss",
    "one_hot",
    "linear",
    "dropout",
    "pad",
    "interpolate",
    "normalize",
    "cosine_similarity",
    "scaled_dot_product_attention",
]
