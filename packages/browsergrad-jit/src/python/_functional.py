"""browsergrad_jit._functional — functional API matching torch.nn.functional.

INTERNAL module. Users import as `browsergrad_jit.nn.functional as F`.

PRD-005 cut-down scope (per the critique): elementwise + MLP-style ops
needed for the 0.1.0 conformance bar. Conv and LayerNorm have since moved to
primitive IR ops; pool, attention, embedding, recurrent, and other heavy ops
still use CUSTOM until promoted.

The pattern across every op: build new UOps + new TensorProxies with the
right backward closures. NumPy-heavy logic that doesn't decompose into
the primitive opcode set lives behind a CUSTOM op so it still participates
in the IR (just opaque to fusion).
"""

from __future__ import annotations
from typing import Any, Optional, Tuple

import numpy as np

from ._ir import (
    UOp, OP_WHERE, OP_CONST, OP_CUSTOM, OP_CONV1D, OP_CONV2D,
    OP_CONV_TRANSPOSE2D, OP_CONV3D,
    OP_LAYER_NORM, OP_REDUCE, OP_DIV, OP_PAD, OP_L1_LOSS, OP_SMOOTH_L1_LOSS,
)
from ._tensor_proxy import (
    TensorProxy, _BackwardCtx, _should_track, _to_proxy, from_numpy, where,
)
from ._errors import ShapeError
from ._framework_contracts import (
    execute_l1_loss_vjp_array,
    execute_smooth_l1_loss_vjp_array,
    infer_l1_loss_contract,
    infer_smooth_l1_loss_contract,
    normalize_pad_request,
)


def _pair2(value: Any, name: str) -> Tuple[int, int]:
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"{name} must be an int or a length-2 tuple")


def _triple3(value: Any, name: str) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (int(value), int(value), int(value))
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    raise ValueError(f"{name} must be an int or a length-3 tuple")


def _check_groups(
    op_name: str,
    in_channels: int,
    out_channels: int,
    weight_in_channels: int,
    groups: int,
) -> None:
    if groups <= 0:
        raise ValueError(f"{op_name}: groups must be positive")
    if in_channels % groups != 0:
        raise ValueError(f"{op_name}: input channels must be divisible by groups")
    if out_channels % groups != 0:
        raise ValueError(f"{op_name}: output channels must be divisible by groups")
    if weight_in_channels != in_channels // groups:
        raise ShapeError(
            f"{op_name}: weight expects {weight_in_channels * groups} input "
            f"channels across {groups} groups, got {in_channels}"
        )


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
    if not isinstance(input, TensorProxy):
        raise TypeError(
            f"l1_loss: input must be a TensorProxy, got {type(input).__name__}"
        )
    if not isinstance(target, TensorProxy):
        raise TypeError(
            f"l1_loss: target must be a TensorProxy, got {type(target).__name__}"
        )
    if target._get_session() is not input._get_session():
        raise ShapeError("l1_loss: input and target must belong to the same session")
    contract = infer_l1_loss_contract(
        (input._uop, target._uop),
        reduction,
        0,
    )
    uop = UOp(
        op=OP_L1_LOSS,
        inputs=(input._uop, target._uop),
        shape=contract.output_shape,
        dtype=contract.output_dtype,
        arg={"reduction": contract.reduction, "batch_rank": contract.batch_rank},
    )

    def _bw(
        dy: np.ndarray,
        ins: Tuple[np.ndarray, ...],
    ) -> Tuple[Optional[np.ndarray], ...]:
        arrays = (ins[0], ins[1])
        return (
            execute_l1_loss_vjp_array(contract, 0, dy, arrays),
            execute_l1_loss_vjp_array(contract, 1, dy, arrays),
        )

    requires = _should_track(input, target)
    ctx = _BackwardCtx(fn=_bw, input_proxies=(input, target)) if requires else None
    return TensorProxy(
        uop,
        session=input._get_session(),
        requires_grad=requires,
        ctx=ctx,
    )


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
    if not isinstance(input, TensorProxy):
        raise TypeError(
            f"smooth_l1_loss: input must be a TensorProxy, got {type(input).__name__}"
        )
    if not isinstance(target, TensorProxy):
        raise TypeError(
            f"smooth_l1_loss: target must be a TensorProxy, got {type(target).__name__}"
        )
    if target._get_session() is not input._get_session():
        raise ShapeError(
            "smooth_l1_loss: input and target must belong to the same session"
        )
    contract = infer_smooth_l1_loss_contract(
        (input._uop, target._uop),
        beta,
        reduction,
        0,
    )
    uop = UOp(
        op=OP_SMOOTH_L1_LOSS,
        inputs=(input._uop, target._uop),
        shape=contract.output_shape,
        dtype=contract.output_dtype,
        arg={
            "reduction": contract.reduction,
            "batch_rank": contract.batch_rank,
            "beta": contract.beta,
        },
    )

    def _bw(
        dy: np.ndarray,
        ins: Tuple[np.ndarray, ...],
    ) -> Tuple[Optional[np.ndarray], ...]:
        arrays = (ins[0], ins[1])
        return (
            execute_smooth_l1_loss_vjp_array(contract, 0, dy, arrays),
            execute_smooth_l1_loss_vjp_array(contract, 1, dy, arrays),
        )

    requires = _should_track(input, target)
    ctx = _BackwardCtx(fn=_bw, input_proxies=(input, target)) if requires else None
    return TensorProxy(
        uop,
        session=input._get_session(),
        requires_grad=requires,
        ctx=ctx,
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


def _normalized_shape_tuple(normalized_shape: Any) -> Tuple[int, ...]:
    if isinstance(normalized_shape, int):
        return (int(normalized_shape),)
    if isinstance(normalized_shape, (tuple, list)):
        if len(normalized_shape) == 0:
            raise ValueError("layer_norm: normalized_shape must be non-empty")
        return tuple(int(v) for v in normalized_shape)
    raise TypeError("layer_norm: normalized_shape must be int or tuple/list of ints")


def layer_norm(
    input: TensorProxy,
    normalized_shape: Any,
    weight: Optional[TensorProxy] = None,
    bias: Optional[TensorProxy] = None,
    eps: float = 1e-5,
) -> TensorProxy:
    ns = _normalized_shape_tuple(normalized_shape)
    if len(ns) > input.ndim:
        raise ShapeError(
            f"layer_norm: normalized_shape {ns} longer than input shape {input.shape}"
        )
    if input.shape[-len(ns):] != ns:
        raise ShapeError(
            f"layer_norm: input trailing shape {input.shape[-len(ns):]} "
            f"does not match normalized_shape {ns}"
        )
    cols = int(np.prod(ns))
    rows = int(np.prod(input.shape) // cols)
    sess = input._get_session()
    weight_proxy = (
        _to_proxy(weight, sess)
        if weight is not None
        else from_numpy(np.ones(ns, dtype=np.float32), session=sess)
    )
    bias_proxy = (
        _to_proxy(bias, sess)
        if bias is not None
        else from_numpy(np.zeros(ns, dtype=np.float32), session=sess)
    )
    if weight_proxy.shape != ns:
        raise ShapeError(
            f"layer_norm: weight shape {weight_proxy.shape} must match normalized_shape {ns}"
        )
    if bias_proxy.shape != ns:
        raise ShapeError(
            f"layer_norm: bias shape {bias_proxy.shape} must match normalized_shape {ns}"
        )
    arg = {
        "normalized_shape": ns,
        "rows": rows,
        "cols": cols,
        "eps": float(eps),
    }
    uop = UOp(
        op=OP_LAYER_NORM,
        inputs=(input._uop, weight_proxy._uop, bias_proxy._uop),
        shape=input.shape,
        dtype=input.dtype,
        arg=arg,
    )

    def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
        x_arr, weight_arr, bias_arr = ins
        x2 = x_arr.reshape(rows, cols).astype(np.float32, copy=False)
        dy2 = dy.reshape(rows, cols).astype(np.float32, copy=False)
        w = weight_arr.reshape(cols).astype(np.float32, copy=False)
        mean = x2.mean(axis=1, keepdims=True)
        centered = x2 - mean
        var = (centered * centered).mean(axis=1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + float(eps))
        x_hat = centered * inv_std
        grad_x_hat = dy2 * w.reshape(1, cols)
        sum_g = grad_x_hat.sum(axis=1, keepdims=True)
        sum_g_xhat = (grad_x_hat * x_hat).sum(axis=1, keepdims=True)
        grad_x = (inv_std / float(cols)) * (
            float(cols) * grad_x_hat - sum_g - x_hat * sum_g_xhat
        )
        grad_w = (dy2 * x_hat).sum(axis=0).reshape(ns)
        grad_b = dy2.sum(axis=0).reshape(ns)
        return (
            grad_x.reshape(input.shape).astype(x_arr.dtype, copy=False),
            grad_w.astype(weight_arr.dtype, copy=False),
            grad_b.astype(bias_arr.dtype, copy=False),
        )

    requires = _should_track(input, weight_proxy, bias_proxy)
    ctx = _BackwardCtx(
        fn=_bw,
        input_proxies=(input, weight_proxy, bias_proxy),
    ) if requires else None
    return TensorProxy(uop, session=sess, requires_grad=requires, ctx=ctx)


# ---------------------------------------------------------------------------
# Convolution family — CUSTOM UOps with explicit NumPy VJPs.
# ---------------------------------------------------------------------------


def conv1d(
    input: TensorProxy,
    weight: TensorProxy,
    bias: Optional[TensorProxy] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> TensorProxy:
    if input.ndim != 3:
        raise ShapeError(f"conv1d: input must be 3-D (N, C, L), got {input.shape}")
    if weight.ndim != 3:
        raise ShapeError(
            f"conv1d: weight must be 3-D (C_out, C_in/groups, K), got {weight.shape}"
        )
    stride = int(stride)
    padding = int(padding)
    dilation = int(dilation)
    groups = int(groups)
    N, C_in, L_in = input.shape
    C_out, C_per_group, K = weight.shape
    _check_groups("conv1d", C_in, C_out, C_per_group, groups)
    if bias is not None and bias.shape != (C_out,):
        raise ShapeError(f"conv1d: bias shape must be {(C_out,)}, got {bias.shape}")
    eff_k = dilation * (K - 1) + 1
    L_out = (L_in + 2 * padding - eff_k) // stride + 1
    if L_out <= 0:
        raise ShapeError(f"conv1d: output length must be positive, got {L_out}")
    out_shape = (N, C_out, L_out)
    out_per_group = C_out // groups

    input_uops = [input._uop, weight._uop]
    input_proxies = [input, weight]
    if bias is not None:
        input_uops.append(bias._uop)
        input_proxies.append(bias)
    uop = UOp(
        op=OP_CONV1D,
        inputs=tuple(input_uops),
        shape=out_shape,
        dtype=input.dtype,
        arg={
            "n": N,
            "c_in": C_in,
            "l_in": L_in,
            "c_out": C_out,
            "k": K,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "l_out": L_out,
            "has_bias": bias is not None,
        },
    )

    def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
        x_arr = ins[0]
        w_arr = ins[1]
        if padding > 0:
            x_pad = np.pad(x_arr, ((0, 0), (0, 0), (padding, padding)), mode="constant")
        else:
            x_pad = x_arr
        grad_w = np.zeros_like(w_arr, dtype=np.float32)
        grad_x_pad = np.zeros_like(x_pad, dtype=np.float32)
        for n in range(N):
            for g in range(groups):
                c0 = g * C_per_group
                o0 = g * out_per_group
                for co in range(out_per_group):
                    for i in range(L_out):
                        l0 = i * stride
                        go = dy[n, o0 + co, i]
                        window = x_pad[n, c0:c0+C_per_group, l0:l0+eff_k:dilation]
                        grad_w[o0 + co] += go * window
                        grad_x_pad[n, c0:c0+C_per_group, l0:l0+eff_k:dilation] += (
                            go * w_arr[o0 + co]
                        )
        grad_x = (
            grad_x_pad[:, :, padding:padding+L_in].copy()
            if padding > 0 else grad_x_pad
        )
        if bias is not None:
            grad_b = dy.sum(axis=(0, 2))
            return (grad_x, grad_w, grad_b)
        return (grad_x, grad_w)

    requires = _should_track(*input_proxies)
    ctx = _BackwardCtx(fn=_bw, input_proxies=tuple(input_proxies)) if requires else None
    return TensorProxy(uop, session=input._get_session(), requires_grad=requires, ctx=ctx)


def conv2d(
    input: TensorProxy,
    weight: TensorProxy,
    bias: Optional[TensorProxy] = None,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> TensorProxy:
    if input.ndim != 4:
        raise ShapeError(f"conv2d: input must be 4-D (N, C, H, W), got {input.shape}")
    if weight.ndim != 4:
        raise ShapeError(
            "conv2d: weight must be 4-D "
            f"(C_out, C_in/groups, KH, KW), got {weight.shape}"
        )
    sh, sw = _pair2(stride, "stride")
    ph, pw = _pair2(padding, "padding")
    dh, dw = _pair2(dilation, "dilation")
    groups = int(groups)
    N, C_in, H, W = input.shape
    C_out, C_per_group, kh, kw = weight.shape
    _check_groups("conv2d", C_in, C_out, C_per_group, groups)
    if bias is not None and bias.shape != (C_out,):
        raise ShapeError(f"conv2d: bias shape must be {(C_out,)}, got {bias.shape}")
    eff_h = dh * (kh - 1) + 1
    eff_w = dw * (kw - 1) + 1
    H_out = (H + 2 * ph - eff_h) // sh + 1
    W_out = (W + 2 * pw - eff_w) // sw + 1
    if H_out <= 0 or W_out <= 0:
        raise ShapeError(f"conv2d: output spatial shape must be positive, got {(H_out, W_out)}")
    out_shape = (N, C_out, H_out, W_out)
    out_per_group = C_out // groups
    L = H_out * W_out

    def _im2col(x_arr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
        if ph > 0 or pw > 0:
            x_pad = np.pad(x_arr, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
        else:
            x_pad = x_arr
        cols = np.zeros((N, C_in * kh * kw, L), dtype=np.float32)
        for i in range(H_out):
            for j in range(W_out):
                h0, w0 = i * sh, j * sw
                cols[:, :, i * W_out + j] = (
                    x_pad[:, :, h0:h0+eff_h:dh, w0:w0+eff_w:dw].reshape(N, -1)
                )
        return cols, x_pad.shape

    input_uops = [input._uop, weight._uop]
    input_proxies = [input, weight]
    if bias is not None:
        input_uops.append(bias._uop)
        input_proxies.append(bias)
    uop = UOp(
        op=OP_CONV2D,
        inputs=tuple(input_uops),
        shape=out_shape,
        dtype=input.dtype,
        arg={
            "n": N,
            "c_in": C_in,
            "h": H,
            "w": W,
            "c_out": C_out,
            "kh": kh,
            "kw": kw,
            "stride_h": sh,
            "stride_w": sw,
            "pad_h": ph,
            "pad_w": pw,
            "dilation_h": dh,
            "dilation_w": dw,
            "groups": groups,
            "out_h": H_out,
            "out_w": W_out,
            "has_bias": bias is not None,
        },
    )

    def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
        cols, x_pad_shape = _im2col(ins[0])
        grad_out_flat = dy.reshape(N, C_out, L)
        grad_w = np.zeros_like(ins[1], dtype=np.float32)
        grad_cols = np.zeros_like(cols)
        weight_flats = []
        col_slices = []
        for g in range(groups):
            c0 = g * C_per_group
            c1 = (g + 1) * C_per_group
            o0 = g * out_per_group
            o1 = (g + 1) * out_per_group
            col_slice = slice(c0 * kh * kw, c1 * kh * kw)
            weight_flats.append(ins[1][o0:o1].reshape(out_per_group, -1).copy())
            col_slices.append((o0, o1, col_slice))
        for idx, (o0, o1, col_slice) in enumerate(col_slices):
            grad_out_g = grad_out_flat[:, o0:o1, :]
            cols_g = cols[:, col_slice, :]
            grad_w_g = (grad_out_g @ np.swapaxes(cols_g, -1, -2)).sum(axis=0)
            grad_w[o0:o1] = grad_w_g.reshape(out_per_group, C_per_group, kh, kw)
            grad_cols[:, col_slice, :] = (
                np.swapaxes(weight_flats[idx], -1, -2) @ grad_out_g
            )
        grad_x_pad = np.zeros(x_pad_shape, dtype=np.float32)
        for i in range(H_out):
            for j in range(W_out):
                h0, w0 = i * sh, j * sw
                grad_x_pad[:, :, h0:h0+eff_h:dh, w0:w0+eff_w:dw] += (
                    grad_cols[:, :, i * W_out + j].reshape(N, C_in, kh, kw)
                )
        grad_x = (
            grad_x_pad[:, :, ph:ph+H, pw:pw+W].copy()
            if ph > 0 or pw > 0 else grad_x_pad
        )
        if bias is not None:
            grad_b = dy.sum(axis=(0, 2, 3))
            return (grad_x, grad_w, grad_b)
        return (grad_x, grad_w)

    requires = _should_track(*input_proxies)
    ctx = _BackwardCtx(fn=_bw, input_proxies=tuple(input_proxies)) if requires else None
    return TensorProxy(uop, session=input._get_session(), requires_grad=requires, ctx=ctx)


def conv_transpose2d(
    input: TensorProxy,
    weight: TensorProxy,
    bias: Optional[TensorProxy] = None,
    stride: Any = 1,
    padding: Any = 0,
    output_padding: Any = 0,
    groups: int = 1,
    dilation: Any = 1,
) -> TensorProxy:
    if input.ndim != 4:
        raise ShapeError(
            f"conv_transpose2d: input must be 4-D (N, C, H, W), got {input.shape}"
        )
    if weight.ndim != 4:
        raise ShapeError(
            "conv_transpose2d: weight must be 4-D "
            f"(C_in, C_out/groups, KH, KW), got {weight.shape}"
        )
    sh, sw = _pair2(stride, "stride")
    ph, pw = _pair2(padding, "padding")
    oph, opw = _pair2(output_padding, "output_padding")
    dh, dw = _pair2(dilation, "dilation")
    groups = int(groups)
    N, C_in, H, W = input.shape
    W_C_in, C_out_per_group, kh, kw = weight.shape
    if W_C_in != C_in:
        raise ShapeError(
            f"conv_transpose2d: weight first dim {W_C_in} must equal input channels {C_in}"
        )
    if groups <= 0 or C_in % groups != 0:
        raise ValueError("conv_transpose2d: groups must divide input channels")
    C_out = C_out_per_group * groups
    if bias is not None and bias.shape != (C_out,):
        raise ShapeError(
            f"conv_transpose2d: bias shape must be {(C_out,)}, got {bias.shape}"
        )
    H_out = (H - 1) * sh - 2 * ph + dh * (kh - 1) + oph + 1
    W_out = (W - 1) * sw - 2 * pw + dw * (kw - 1) + opw + 1
    if H_out <= 0 or W_out <= 0:
        raise ShapeError(
            f"conv_transpose2d: output spatial shape must be positive, got {(H_out, W_out)}"
        )
    out_shape = (N, C_out, H_out, W_out)
    in_per_group = C_in // groups

    input_uops = [input._uop, weight._uop]
    input_proxies = [input, weight]
    if bias is not None:
        input_uops.append(bias._uop)
        input_proxies.append(bias)
    uop = UOp(
        op=OP_CONV_TRANSPOSE2D,
        inputs=tuple(input_uops),
        shape=out_shape,
        dtype=input.dtype,
        arg={
            "n": N,
            "c_in": C_in,
            "h": H,
            "w": W,
            "c_out": C_out,
            "c_out_per_group": C_out_per_group,
            "kh": kh,
            "kw": kw,
            "stride_h": sh,
            "stride_w": sw,
            "pad_h": ph,
            "pad_w": pw,
            "output_pad_h": oph,
            "output_pad_w": opw,
            "dilation_h": dh,
            "dilation_w": dw,
            "groups": groups,
            "out_h": H_out,
            "out_w": W_out,
            "has_bias": bias is not None,
        },
    )

    def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
        x_arr = ins[0]
        w_arr = ins[1]
        grad_x = np.zeros_like(x_arr, dtype=np.float32)
        grad_w = np.zeros_like(w_arr, dtype=np.float32)
        for n in range(N):
            for ci in range(C_in):
                group = ci // in_per_group
                co0 = group * C_out_per_group
                for ih in range(H):
                    for iw in range(W):
                        x_val = x_arr[n, ci, ih, iw]
                        for r in range(kh):
                            oh = ih * sh - ph + r * dh
                            if oh < 0 or oh >= H_out:
                                continue
                            for c in range(kw):
                                ow = iw * sw - pw + c * dw
                                if 0 <= ow < W_out:
                                    go = dy[n, co0:co0+C_out_per_group, oh, ow]
                                    grad_x[n, ci, ih, iw] += (go * w_arr[ci, :, r, c]).sum()
                                    grad_w[ci, :, r, c] += go * x_val
        if bias is not None:
            grad_b = dy.sum(axis=(0, 2, 3))
            return (grad_x, grad_w, grad_b)
        return (grad_x, grad_w)

    requires = _should_track(*input_proxies)
    ctx = _BackwardCtx(fn=_bw, input_proxies=tuple(input_proxies)) if requires else None
    return TensorProxy(uop, session=input._get_session(), requires_grad=requires, ctx=ctx)


def conv3d(
    input: TensorProxy,
    weight: TensorProxy,
    bias: Optional[TensorProxy] = None,
    stride: Any = 1,
    padding: Any = 0,
    dilation: Any = 1,
    groups: int = 1,
) -> TensorProxy:
    if input.ndim != 5:
        raise ShapeError(f"conv3d: input must be 5-D (N, C, D, H, W), got {input.shape}")
    if weight.ndim != 5:
        raise ShapeError(
            "conv3d: weight must be 5-D "
            f"(C_out, C_in/groups, KD, KH, KW), got {weight.shape}"
        )
    sd, sh, sw = _triple3(stride, "stride")
    pd, ph, pw = _triple3(padding, "padding")
    dd, dh, dw = _triple3(dilation, "dilation")
    groups = int(groups)
    N, C_in, D, H, W = input.shape
    C_out, C_per_group, kd, kh, kw = weight.shape
    _check_groups("conv3d", C_in, C_out, C_per_group, groups)
    if bias is not None and bias.shape != (C_out,):
        raise ShapeError(f"conv3d: bias shape must be {(C_out,)}, got {bias.shape}")
    eff_d = dd * (kd - 1) + 1
    eff_h = dh * (kh - 1) + 1
    eff_w = dw * (kw - 1) + 1
    D_out = (D + 2 * pd - eff_d) // sd + 1
    H_out = (H + 2 * ph - eff_h) // sh + 1
    W_out = (W + 2 * pw - eff_w) // sw + 1
    if D_out <= 0 or H_out <= 0 or W_out <= 0:
        raise ShapeError(
            f"conv3d: output spatial shape must be positive, got {(D_out, H_out, W_out)}"
        )
    out_shape = (N, C_out, D_out, H_out, W_out)
    input_uops = [input._uop, weight._uop]
    input_proxies = [input, weight]
    if bias is not None:
        input_uops.append(bias._uop)
        input_proxies.append(bias)
    uop = UOp(
        op=OP_CONV3D,
        inputs=tuple(input_uops),
        shape=out_shape,
        dtype=input.dtype,
        arg={
            "n": N,
            "c_in": C_in,
            "d": D,
            "h": H,
            "w": W,
            "c_out": C_out,
            "kd": kd,
            "kh": kh,
            "kw": kw,
            "stride_d": sd,
            "stride_h": sh,
            "stride_w": sw,
            "pad_d": pd,
            "pad_h": ph,
            "pad_w": pw,
            "dilation_d": dd,
            "dilation_h": dh,
            "dilation_w": dw,
            "groups": groups,
            "out_d": D_out,
            "out_h": H_out,
            "out_w": W_out,
            "has_bias": bias is not None,
        },
    )

    def _bw(dy: np.ndarray, ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
        x_arr = ins[0]
        w_arr = ins[1]
        if pd > 0 or ph > 0 or pw > 0:
            x_pad = np.pad(
                x_arr,
                ((0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)),
                mode="constant",
            )
        else:
            x_pad = x_arr
        L = D_out * H_out * W_out
        cols = np.zeros((N, C_in * kd * kh * kw, L), dtype=np.float32)
        col_idx = 0
        for od in range(D_out):
            for oh in range(H_out):
                for ow in range(W_out):
                    d0, h0, w0 = od * sd, oh * sh, ow * sw
                    cols[:, :, col_idx] = (
                        x_pad[
                            :,
                            :,
                            d0:d0+eff_d:dd,
                            h0:h0+eff_h:dh,
                            w0:w0+eff_w:dw,
                        ].reshape(N, -1)
                    )
                    col_idx += 1
        grad_out_flat = dy.reshape(N, C_out, L)
        grad_w = np.zeros_like(ins[1], dtype=np.float32)
        grad_cols = np.zeros_like(cols)
        out_per_group = C_out // groups
        for g in range(groups):
            c0 = g * C_per_group
            c1 = (g + 1) * C_per_group
            o0 = g * out_per_group
            o1 = (g + 1) * out_per_group
            col_slice = slice(c0 * kd * kh * kw, c1 * kd * kh * kw)
            grad_out_g = grad_out_flat[:, o0:o1, :]
            cols_g = cols[:, col_slice, :]
            grad_w_g = (grad_out_g @ np.swapaxes(cols_g, -1, -2)).sum(axis=0)
            grad_w[o0:o1] = grad_w_g.reshape(out_per_group, C_per_group, kd, kh, kw)
            grad_cols[:, col_slice, :] = (
                np.swapaxes(w_arr[o0:o1].reshape(out_per_group, -1), -1, -2) @ grad_out_g
            )
        grad_x_pad = np.zeros(x_pad.shape, dtype=np.float32)
        col_idx = 0
        for od in range(D_out):
            for oh in range(H_out):
                for ow in range(W_out):
                    d0, h0, w0 = od * sd, oh * sh, ow * sw
                    grad_x_pad[
                        :,
                        :,
                        d0:d0+eff_d:dd,
                        h0:h0+eff_h:dh,
                        w0:w0+eff_w:dw,
                    ] += grad_cols[:, :, col_idx].reshape(N, C_in, kd, kh, kw)
                    col_idx += 1
        grad_x = (
            grad_x_pad[:, :, pd:pd+D, ph:ph+H, pw:pw+W].copy()
            if pd > 0 or ph > 0 or pw > 0 else grad_x_pad
        )
        if bias is not None:
            grad_b = dy.sum(axis=(0, 2, 3, 4))
            return (grad_x, grad_w, grad_b)
        return (grad_x, grad_w)

    requires = _should_track(*input_proxies)
    ctx = _BackwardCtx(fn=_bw, input_proxies=tuple(input_proxies)) if requires else None
    return TensorProxy(uop, session=input._get_session(), requires_grad=requires, ctx=ctx)


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


def pad(input: TensorProxy, pad, mode: str = "constant", value=None) -> TensorProxy:
    if type(input) is not TensorProxy:
        raise TypeError(f"pad: input must be a TensorProxy, got {type(input).__name__}")
    pad_width, normalized_value, out_shape = normalize_pad_request(
        input._uop,
        pad,
        mode,
        value,
    )
    uop = UOp(
        op=OP_PAD,
        inputs=(input._uop,),
        shape=out_shape,
        dtype=input.dtype,
        arg={
            "pad_width": pad_width,
            "mode": "constant",
            "value": normalized_value,
        },
    )

    def _bw(dy: np.ndarray, _ins: Tuple[np.ndarray, ...]) -> Tuple[Optional[np.ndarray], ...]:
        slices = tuple(
            slice(lower, lower + size)
            for (lower, _), size in zip(pad_width, input.shape)
        )
        return (dy[slices].copy().astype(np.dtype(input.dtype), copy=False),)

    requires = _should_track(input) and input.dtype in ("float16", "float32", "float64")
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
    scores = query @ key.transpose(-1, -2)
    scale_tensor = from_numpy(
        np.full(scores.shape, s, dtype=np.float32),
        session=query._get_session(),
    )
    scores = scores * scale_tensor
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
    "layer_norm",
    "conv1d",
    "conv2d",
    "conv_transpose2d",
    "conv3d",
    "dropout",
    "pad",
    "interpolate",
    "normalize",
    "cosine_similarity",
    "scaled_dot_product_attention",
]
