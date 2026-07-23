
"""browsergrad_grad.functional — stateless ops with autograd."""

import numpy as np
from .tensor import Tensor, _build_ctx, _normalize_pad_contract
from . import _device


# ─── Activations ───────────────────────────────────────────

def silu(x: Tensor) -> Tensor:
    """Sigmoid Linear Unit: x * sigmoid(x). Used in SwiGLU feed-forward."""
    s = 1.0 / (1.0 + np.exp(-x.data))
    out = Tensor((x.data * s).astype(np.float32))
    x_data = x.data
    def backward(g):
        s_bw = 1.0 / (1.0 + np.exp(-x_data))
        dsilu = (s_bw + x_data * s_bw * (1.0 - s_bw)).astype(np.float32)
        return (g.data * dsilu,)
    return _build_ctx(out, (x,), backward)


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

def softmax(x: Tensor, dim: int = -1, device=None) -> Tensor:
    """Stable softmax along `dim`."""
    xd = x.data
    axis = dim if dim >= 0 else xd.ndim + dim
    if device is not None and axis != xd.ndim - 1:
        raise NotImplementedError(
            f"softmax(device=...): KernelDevice bridge supports only last dim (got dim={dim})"
        )
    if device is not None:
        s = _device.softmax(device, xd)
    else:
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


def bce_with_logits_loss(
    logits: Tensor,
    targets: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Typed, numerically stable binary cross entropy from raw logits."""
    if not isinstance(logits, Tensor):
        raise TypeError(
            "binary_cross_entropy_with_logits: logits must be a Tensor, got "
            f"{type(logits).__name__}"
        )
    if not isinstance(targets, Tensor):
        raise TypeError(
            "binary_cross_entropy_with_logits: targets must be a Tensor, got "
            f"{type(targets).__name__}"
        )
    contract = _elementwise_loss_contract(
        logits,
        targets,
        reduction,
        "binary_cross_entropy_with_logits",
        _BINARY_CROSS_ENTROPY_WITH_LOGITS_WORK_VISIT_FACTOR,
        compute_buffers=4,
        mask_buffers=1,
    )
    shape, _, _, _, compute_dtype, _ = contract
    dtype = np.dtype(compute_dtype)
    logits_array = logits.data.astype(dtype, copy=False)
    targets_array = targets.data.astype(dtype, copy=False)

    softplus_negative = np.empty(shape, dtype=dtype)
    np.abs(logits_array, out=softplus_negative)
    np.negative(softplus_negative, out=softplus_negative)
    with np.errstate(over="ignore", invalid="ignore"):
        np.exp(softplus_negative, out=softplus_negative)
        np.log1p(softplus_negative, out=softplus_negative)
    negative_logits = np.empty(shape, dtype=dtype)
    np.negative(logits_array, out=negative_logits)
    np.maximum(negative_logits, 0.0, out=negative_logits)
    np.add(softplus_negative, negative_logits, out=softplus_negative)

    per_element = np.empty(shape, dtype=dtype)
    np.subtract(1.0, targets_array, out=per_element)
    np.multiply(per_element, logits_array, out=per_element)
    np.add(per_element, softplus_negative, out=per_element)
    per_element[per_element == 0.0] = 0.0

    # Reuse the forward temporaries to retain exact stable derivatives for
    # eager backward without holding another copy of either source tensor.
    input_derivative = softplus_negative
    np.abs(logits_array, out=input_derivative)
    np.negative(input_derivative, out=input_derivative)
    with np.errstate(over="ignore", invalid="ignore"):
        np.exp(input_derivative, out=input_derivative)
    denominator = negative_logits
    np.add(1.0, input_derivative, out=denominator)
    np.divide(input_derivative, denominator, out=input_derivative)
    nonnegative = logits_array >= 0.0
    np.divide(1.0, denominator, out=input_derivative, where=nonnegative)
    np.subtract(input_derivative, targets_array, out=input_derivative)
    target_derivative = np.empty(shape, dtype=dtype)
    np.negative(logits_array, out=target_derivative)
    return _finish_typed_elementwise_loss(
        logits,
        targets,
        reduction,
        contract,
        per_element,
        input_derivative,
        target_derivative,
    )


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


def _normalize_legacy_loss_reduction(
    operation,
    reduction,
    size_average,
    reduce,
):
    if size_average is None and reduce is None:
        return reduction
    if size_average is not None and type(size_average) is not bool:
        raise ValueError(
            f"{operation}: size_average must be an exact bool or None"
        )
    if reduce is not None and type(reduce) is not bool:
        raise ValueError(f"{operation}: reduce must be an exact bool or None")
    effective_size_average = True if size_average is None else size_average
    effective_reduce = True if reduce is None else reduce
    if not effective_reduce:
        return "none"
    return "mean" if effective_size_average else "sum"


def _nll_loss_contract(input, target, weight, reduction, ignore_index):
    if not isinstance(input, Tensor):
        raise TypeError(
            f"nll_loss: input must be a Tensor, got {type(input).__name__}"
        )
    if not isinstance(target, Tensor):
        raise TypeError(
            f"nll_loss: target must be a Tensor, got {type(target).__name__}"
        )
    if type(input.data) is not np.ndarray:
        raise TypeError("nll_loss: input data must be an exact ndarray")
    if type(target.data) is not np.ndarray:
        raise TypeError("nll_loss: target data must be an exact ndarray")
    if type(reduction) is not str:
        raise ValueError(
            "nll_loss: reduction must be a string, got "
            f"{type(reduction).__name__}"
        )
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(
            "nll_loss: reduction must be 'none', 'sum', or 'mean', got "
            f"{reduction!r}"
        )
    integer_types = (
        int,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    )
    if type(ignore_index) not in integer_types or type(ignore_index) is bool:
        raise ValueError("nll_loss: ignore_index must be an exact integer")
    normalized_ignore_index = int(ignore_index)
    if (
        normalized_ignore_index < -(1 << 63)
        or normalized_ignore_index > (1 << 63) - 1
    ):
        raise ValueError("nll_loss: ignore_index must fit signed int64")

    input_shape = tuple(input.data.shape)
    target_shape = tuple(target.data.shape)
    if len(input_shape) < 1:
        raise ValueError(
            "nll_loss: batch_rank 0 leaves no user input dimension "
            f"in shape {input_shape}"
        )
    shapes = [input_shape, target_shape]
    arrays = [input.data, target.data]
    if weight is not None:
        if not isinstance(weight, Tensor):
            raise TypeError(
                f"nll_loss: weight must be a Tensor or None, got "
                f"{type(weight).__name__}"
            )
        if weight.requires_grad:
            raise ValueError(
                "nll_loss: weight is non-differentiable and must not require grad"
            )
        if type(weight.data) is not np.ndarray:
            raise TypeError("nll_loss: weight data must be an exact ndarray")
        shapes.append(tuple(weight.data.shape))
        arrays.append(weight.data)
    for index, shape in enumerate(shapes):
        if len(shape) > _L1_LOSS_RANK_MAX:
            raise ValueError(
                f"nll_loss input {index} rank {len(shape)} exceeds the "
                f"{_L1_LOSS_RANK_MAX}-rank ceiling"
            )
        for axis, extent in enumerate(shape):
            if extent > _L1_LOSS_OUTPUT_EXTENT_MAX:
                raise ValueError(
                    f"nll_loss input {index} extent {extent} on axis {axis} "
                    f"exceeds the {_L1_LOSS_OUTPUT_EXTENT_MAX}-element "
                    "per-axis ceiling"
                )

    input_dtype = input.data.dtype.name
    if input_dtype not in _L1_LOSS_FLOATING_DTYPES:
        raise ValueError(
            f"nll_loss input dtype {input_dtype!r} is not supported; expected "
            "float16, float32, or float64"
        )
    if target.data.dtype.name != "int64":
        raise ValueError(
            f"nll_loss target dtype {target.data.dtype.name!r} is not "
            "supported; expected int64"
        )
    class_axis = 0 if len(input_shape) == 1 else 1
    class_count = input_shape[class_axis]
    expected_target_shape = (
        input_shape[:class_axis] + input_shape[class_axis + 1:]
    )
    if target_shape != expected_target_shape:
        raise ValueError(
            f"nll_loss: target shape {target_shape} must equal input shape "
            f"{input_shape} with class axis {class_axis} removed "
            f"({expected_target_shape})"
        )
    weight_shape = None
    if weight is not None:
        weight_shape = tuple(weight.data.shape)
        if weight_shape != (class_count,):
            raise ValueError(
                f"nll_loss: weight shape {weight_shape} must equal class "
                f"count ({class_count},)"
            )
        if weight.data.dtype.name != input_dtype:
            raise ValueError(
                f"nll_loss: weight dtype {weight.data.dtype.name!r} must "
                f"equal input dtype {input_dtype!r}"
            )

    input_elements = _l1_loss_checked_product(
        input_shape, _L1_LOSS_WORK_ELEMENT_MAX
    )
    target_elements = _l1_loss_checked_product(
        target_shape, _L1_LOSS_WORK_ELEMENT_MAX
    )
    weight_elements = (
        _l1_loss_checked_product(weight_shape, _L1_LOSS_WORK_ELEMENT_MAX)
        if weight_shape is not None
        else 0
    )
    capacity_elements = max(
        _l1_loss_checked_product(
            tuple(max(1, extent) for extent in shape),
            _L1_LOSS_WORK_ELEMENT_MAX,
        )
        for shape in shapes
    )
    if (
        capacity_elements
        > _L1_LOSS_WORK_ELEMENT_MAX // _NLL_LOSS_WORK_VISIT_FACTOR
    ):
        work_elements = _L1_LOSS_WORK_ELEMENT_MAX + 1
    else:
        work_elements = capacity_elements * _NLL_LOSS_WORK_VISIT_FACTOR
    if work_elements > _L1_LOSS_WORK_ELEMENT_MAX:
        raise ValueError(
            "nll_loss: projected work exceeds the "
            f"{_L1_LOSS_WORK_ELEMENT_MAX}-element-visit ceiling"
        )
    output_shape = target_shape if reduction == "none" else ()
    output_elements = _l1_loss_checked_product(
        output_shape, _L1_LOSS_OUTPUT_BYTE_MAX
    )
    output_bytes = output_elements * input.data.dtype.itemsize
    if output_bytes > _L1_LOSS_OUTPUT_BYTE_MAX:
        raise ValueError(
            f"nll_loss: output requires {output_bytes} bytes, exceeding the "
            f"{_L1_LOSS_OUTPUT_BYTE_MAX}-byte ceiling"
        )
    compute_dtype = "float32" if input_dtype == "float16" else input_dtype
    compute_bytes = np.dtype(compute_dtype).itemsize
    cast_bytes = 0
    if input_dtype != compute_dtype:
        cast_bytes += input_elements * compute_bytes
        cast_bytes += weight_elements * compute_bytes
    workspace_bytes = (
        output_bytes
        + cast_bytes
        + target_elements * (2 * compute_bytes + 8 + 1)
        + input_elements * compute_bytes
    )
    if workspace_bytes > _L1_LOSS_WORKSPACE_BYTE_MAX:
        raise ValueError(
            "nll_loss: projected output/cast/gather/gradient workspace "
            f"requires {workspace_bytes} bytes, exceeding the "
            f"{_L1_LOSS_WORKSPACE_BYTE_MAX}-byte ceiling"
        )
    return {
        "input_shape": input_shape,
        "target_shape": target_shape,
        "output_shape": output_shape,
        "input_dtype": input_dtype,
        "compute_dtype": compute_dtype,
        "class_axis": class_axis,
        "class_count": class_count,
        "ignore_index": normalized_ignore_index,
        "reduction": reduction,
        "has_weight": weight is not None,
    }


def nll_loss(
    input: Tensor,
    target: Tensor,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
) -> Tensor:
    """Typed negative log likelihood with PyTorch class-axis semantics."""
    normalized_reduction = _normalize_legacy_loss_reduction(
        "nll_loss",
        reduction,
        size_average,
        reduce,
    )
    contract = _nll_loss_contract(
        input,
        target,
        weight,
        normalized_reduction,
        ignore_index,
    )
    dtype = np.dtype(contract["compute_dtype"])
    targets = target.data
    valid = targets != contract["ignore_index"]
    if bool(valid.any()):
        minimum = int(np.min(targets, where=valid, initial=0))
        maximum = int(np.max(targets, where=valid, initial=0))
        if minimum < 0 or maximum >= contract["class_count"]:
            raise ValueError(
                "nll_loss: target values must be in "
                f"[0, {contract['class_count']}) or equal ignore_index "
                f"{contract['ignore_index']}"
            )
    safe_targets = np.where(valid, targets, 0)
    if contract["has_weight"]:
        weights = weight.data.astype(dtype, copy=False)
        if contract["class_count"] == 0:
            selected_weight = np.zeros(contract["target_shape"], dtype=dtype)
        else:
            selected_weight = weights[safe_targets]
        selected_weight = np.where(valid, selected_weight, 0.0)
    else:
        selected_weight = valid.astype(dtype, copy=False)

    input_array = input.data.astype(dtype, copy=False)
    if contract["class_count"] == 0:
        per_target = np.zeros(contract["target_shape"], dtype=dtype)
    else:
        class_last = np.moveaxis(input_array, contract["class_axis"], -1)
        per_target = np.take_along_axis(
            class_last,
            safe_targets[..., None],
            axis=-1,
        )[..., 0]
        np.negative(per_target, out=per_target)
        if contract["has_weight"]:
            np.multiply(per_target, selected_weight, out=per_target)
        np.copyto(per_target, 0.0, where=~valid)

    if contract["reduction"] == "none":
        result = per_target
    else:
        numerator = per_target.sum(dtype=dtype)
        if contract["reduction"] == "sum":
            result = numerator
        else:
            denominator = (
                selected_weight.sum(dtype=dtype)
                if contract["has_weight"]
                else valid.sum()
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.divide(numerator, denominator)
    out = Tensor(
        np.array(result, dtype=np.dtype(contract["input_dtype"]), copy=True),
        dtype=contract["input_dtype"],
    )

    saved_valid = np.array(valid, dtype=np.bool_, copy=True)
    saved_targets = np.array(safe_targets, dtype=np.int64, copy=True)
    saved_weight = np.array(selected_weight, dtype=dtype, copy=True)

    def backward(g):
        gradient = np.zeros(contract["input_shape"], dtype=dtype)
        if contract["class_count"] == 0 or not bool(saved_valid.any()):
            return (
                gradient.astype(
                    np.dtype(contract["input_dtype"]),
                    copy=False,
                ),
            )
        upstream = g.data.astype(dtype, copy=False)
        if contract["reduction"] != "none":
            upstream = np.broadcast_to(upstream, contract["target_shape"])
            if contract["reduction"] == "mean":
                denominator = (
                    saved_weight.sum(dtype=dtype)
                    if contract["has_weight"]
                    else saved_valid.sum()
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    upstream = np.divide(upstream, denominator)
        selected_gradient = np.empty(contract["target_shape"], dtype=dtype)
        np.negative(upstream, out=selected_gradient)
        if contract["has_weight"]:
            np.multiply(
                selected_gradient,
                saved_weight,
                out=selected_gradient,
            )
        np.copyto(selected_gradient, 0.0, where=~saved_valid)
        class_last_gradient = np.moveaxis(
            gradient,
            contract["class_axis"],
            -1,
        )
        np.put_along_axis(
            class_last_gradient,
            saved_targets[..., None],
            selected_gradient[..., None],
            axis=-1,
        )
        return (
            gradient.astype(
                np.dtype(contract["input_dtype"]),
                copy=gradient.dtype.name != contract["input_dtype"],
            ),
        )

    return _build_ctx(out, (input,), backward)


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


_L1_LOSS_FLOATING_DTYPES = frozenset({"float16", "float32", "float64"})
_L1_LOSS_RANK_MAX = 32
_L1_LOSS_OUTPUT_BYTE_MAX = 1 << 28
_L1_LOSS_OUTPUT_EXTENT_MAX = _L1_LOSS_OUTPUT_BYTE_MAX
_L1_LOSS_WORK_ELEMENT_MAX = 1 << 28
_L1_LOSS_WORKSPACE_BYTE_MAX = 1 << 28
_L1_LOSS_WORK_VISIT_FACTOR = 10
_SMOOTH_L1_LOSS_WORK_VISIT_FACTOR = 32
_BINARY_CROSS_ENTROPY_WORK_VISIT_FACTOR = 48
_BINARY_CROSS_ENTROPY_LOG_FLOOR = -100.0
_BINARY_CROSS_ENTROPY_GRAD_EPSILON = 1e-12
_BINARY_CROSS_ENTROPY_WITH_LOGITS_WORK_VISIT_FACTOR = 36
_KL_DIV_WORK_VISIT_FACTOR = 48
_NLL_LOSS_WORK_VISIT_FACTOR = 32
_SMOOTH_L1_BETA_TYPES = (
    int,
    float,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
)


def _l1_loss_checked_product(extents, ceiling: int) -> int:
    product = 1
    for extent in extents:
        if extent == 0:
            return 0
        if product > ceiling // extent:
            return ceiling + 1
        product *= extent
    return product


def _elementwise_loss_contract(
    input: Tensor,
    target: Tensor,
    reduction,
    operation: str,
    work_visit_factor: int,
    compute_buffers: int = 3,
    mask_buffers: int = 1,
    allow_batchmean: bool = False,
):
    if type(compute_buffers) is not int or compute_buffers < 0:
        raise ValueError(f"{operation}: compute buffer count must be non-negative")
    if type(mask_buffers) is not int or mask_buffers < 0:
        raise ValueError(f"{operation}: mask buffer count must be non-negative")
    if not isinstance(input, Tensor):
        raise TypeError(
            f"{operation}: input must be a Tensor, got {type(input).__name__}"
        )
    if not isinstance(target, Tensor):
        raise TypeError(
            f"{operation}: target must be a Tensor, got {type(target).__name__}"
        )
    if type(input.data) is not np.ndarray:
        raise TypeError(f"{operation}: input data must be an exact ndarray")
    if type(target.data) is not np.ndarray:
        raise TypeError(f"{operation}: target data must be an exact ndarray")
    if type(reduction) is not str:
        raise ValueError(
            f"{operation}: reduction must be a string, got {type(reduction).__name__}"
        )
    valid_reductions = (
        ("none", "sum", "mean", "batchmean")
        if allow_batchmean
        else ("none", "sum", "mean")
    )
    if reduction not in valid_reductions:
        expected = (
            "'none', 'sum', 'mean', or 'batchmean'"
            if allow_batchmean
            else "'none', 'sum', or 'mean'"
        )
        raise ValueError(
            f"{operation}: reduction must be {expected}, got {reduction!r}"
        )
    if input.data.shape != target.data.shape:
        raise ValueError(
            f"{operation}: input shape {input.data.shape} must equal target shape "
            f"{target.data.shape}"
        )
    shape = tuple(input.data.shape)
    if len(shape) > _L1_LOSS_RANK_MAX:
        raise ValueError(
            f"{operation}: input rank {len(shape)} exceeds the "
            f"{_L1_LOSS_RANK_MAX}-rank ceiling"
        )
    for axis, extent in enumerate(shape):
        if extent > _L1_LOSS_OUTPUT_EXTENT_MAX:
            raise ValueError(
                f"{operation}: input extent {extent} on axis {axis} exceeds the "
                f"{_L1_LOSS_OUTPUT_EXTENT_MAX}-element per-axis ceiling"
            )
    input_dtype = input.data.dtype.name
    target_dtype = target.data.dtype.name
    if input_dtype not in _L1_LOSS_FLOATING_DTYPES:
        raise ValueError(
            f"{operation}: input dtype {input_dtype!r} is not supported; "
            "expected float16, float32, or float64"
        )
    if target_dtype not in _L1_LOSS_FLOATING_DTYPES:
        raise ValueError(
            f"{operation}: target dtype {target_dtype!r} is not supported; "
            "expected float16, float32, or float64"
        )
    output_dtype = np.promote_types(input.data.dtype, target.data.dtype).name
    compute_dtype = "float32" if output_dtype == "float16" else output_dtype
    input_elements = _l1_loss_checked_product(shape, _L1_LOSS_WORK_ELEMENT_MAX)
    capacity_elements = _l1_loss_checked_product(
        tuple(max(1, extent) for extent in shape),
        _L1_LOSS_WORK_ELEMENT_MAX,
    )
    if capacity_elements > _L1_LOSS_WORK_ELEMENT_MAX // work_visit_factor:
        work_elements = _L1_LOSS_WORK_ELEMENT_MAX + 1
    else:
        work_elements = capacity_elements * work_visit_factor
    if work_elements > _L1_LOSS_WORK_ELEMENT_MAX:
        raise ValueError(
            f"{operation}: projected work exceeds the "
            f"{_L1_LOSS_WORK_ELEMENT_MAX}-element-visit ceiling"
        )
    output_elements = input_elements if reduction == "none" else 1
    output_bytes = output_elements * np.dtype(output_dtype).itemsize
    if output_bytes > _L1_LOSS_OUTPUT_BYTE_MAX:
        raise ValueError(
            f"{operation}: output requires {output_bytes} bytes, exceeding the "
            f"{_L1_LOSS_OUTPUT_BYTE_MAX}-byte ceiling"
        )
    compute_bytes = np.dtype(compute_dtype).itemsize
    cast_bytes = sum(
        input_elements * compute_bytes
        for dtype in (input_dtype, target_dtype)
        if dtype != compute_dtype
    )
    workspace_bytes = (
        output_bytes
        + cast_bytes
        + input_elements * (
            compute_bytes * compute_buffers + mask_buffers
        )
        + input_elements * (
            np.dtype(input_dtype).itemsize + np.dtype(target_dtype).itemsize
        )
    )
    if workspace_bytes > _L1_LOSS_WORKSPACE_BYTE_MAX:
        raise ValueError(
            f"{operation}: projected output/cast/intermediate/gradient workspace "
            f"requires {workspace_bytes} bytes, exceeding the "
            f"{_L1_LOSS_WORKSPACE_BYTE_MAX}-byte ceiling"
        )
    return shape, input_dtype, target_dtype, output_dtype, compute_dtype, input_elements


def _l1_loss_contract(input: Tensor, target: Tensor, reduction):
    return _elementwise_loss_contract(
        input,
        target,
        reduction,
        "l1_loss",
        _L1_LOSS_WORK_VISIT_FACTOR,
    )


def _normalize_smooth_l1_beta(beta) -> float:
    if type(beta) not in _SMOOTH_L1_BETA_TYPES or type(beta) is bool:
        raise ValueError("smooth_l1_loss: beta must be an exact real scalar")
    try:
        normalized = float(beta)
    except (OverflowError, ValueError) as exc:
        raise ValueError("smooth_l1_loss: beta must be finite") from exc
    if not np.isfinite(normalized):
        raise ValueError("smooth_l1_loss: beta must be finite")
    if normalized < 0.0:
        raise ValueError(
            f"smooth_l1_loss: beta must be non-negative, got {normalized}"
        )
    return 0.0 if normalized == 0.0 else normalized


def _finish_typed_elementwise_loss(
    input: Tensor,
    target: Tensor,
    reduction: str,
    contract,
    per_element: np.ndarray,
    input_derivative: np.ndarray,
    target_derivative: np.ndarray | None = None,
    *,
    batch_denominator: int | None = None,
) -> Tensor:
    (
        shape,
        input_dtype,
        target_dtype,
        output_dtype,
        compute_dtype,
        input_elements,
    ) = contract
    if reduction == "none":
        result = per_element
    elif reduction == "sum":
        result = per_element.sum(dtype=np.dtype(compute_dtype))
    elif reduction == "mean" and input_elements == 0:
        result = np.asarray(np.nan, dtype=np.dtype(compute_dtype))
    elif reduction == "mean":
        result = per_element.sum(dtype=np.dtype(compute_dtype)) / float(input_elements)
    else:
        if batch_denominator is None:
            raise ValueError("batchmean loss requires an exact batch denominator")
        numerator = per_element.sum(dtype=np.dtype(compute_dtype))
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.divide(numerator, float(batch_denominator))
    out = Tensor(
        np.array(result, dtype=np.dtype(output_dtype), copy=True),
        dtype=output_dtype,
    )

    def backward(g):
        if input_elements == 0:
            return (
                np.zeros(shape, dtype=np.dtype(input_dtype)),
                np.zeros(shape, dtype=np.dtype(target_dtype)),
            )
        upstream = g.data.astype(np.dtype(compute_dtype), copy=False)
        if reduction != "none":
            upstream = np.broadcast_to(upstream, shape)
            if reduction == "mean":
                upstream = upstream / float(input_elements)
            elif reduction == "batchmean":
                if batch_denominator is None:
                    raise ValueError(
                        "batchmean loss requires an exact batch denominator"
                    )
                with np.errstate(divide="ignore", invalid="ignore"):
                    upstream = upstream / float(batch_denominator)
        working_gradient = np.empty(shape, dtype=np.dtype(compute_dtype))
        np.multiply(input_derivative, upstream, out=working_gradient)
        input_gradient = working_gradient.astype(np.dtype(input_dtype), copy=True)
        if target_derivative is None:
            np.negative(working_gradient, out=working_gradient)
            working_gradient[input_derivative == 0] = 0.0
        else:
            np.multiply(target_derivative, upstream, out=working_gradient)
        target_gradient = working_gradient.astype(np.dtype(target_dtype), copy=True)
        return input_gradient, target_gradient

    return _build_ctx(out, (input, target), backward)


def l1_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Typed same-shape absolute error with exact reduction semantics."""
    contract = _l1_loss_contract(input, target, reduction)
    shape, _, _, _, compute_dtype, _ = contract
    dtype = np.dtype(compute_dtype)
    left = input.data.astype(dtype, copy=False)
    right = target.data.astype(dtype, copy=False)
    difference = np.empty(shape, dtype=dtype)
    np.subtract(left, right, out=difference)
    per_element = np.empty(shape, dtype=dtype)
    np.abs(difference, out=per_element)
    np.sign(difference, out=difference)
    return _finish_typed_elementwise_loss(
        input,
        target,
        reduction,
        contract,
        per_element,
        difference,
    )


def bce_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Typed BCE over probabilities with PyTorch's endpoint semantics."""
    contract = _elementwise_loss_contract(
        input,
        target,
        reduction,
        "binary_cross_entropy",
        _BINARY_CROSS_ENTROPY_WORK_VISIT_FACTOR,
        compute_buffers=4,
        mask_buffers=1,
    )
    shape, _, _, _, compute_dtype, _ = contract
    for tensor, label in ((input, "input"), (target, "target")):
        array = tensor.data
        if array.size == 0:
            continue
        if not bool(np.isfinite(array).all()):
            raise ValueError(
                f"binary_cross_entropy: all elements of {label} must be finite "
                "and between 0 and 1"
            )
        if bool(array.min() < 0.0) or bool(array.max() > 1.0):
            raise ValueError(
                f"binary_cross_entropy: all elements of {label} must be between 0 and 1"
            )

    dtype = np.dtype(compute_dtype)
    probabilities = input.data.astype(dtype, copy=False)
    targets = target.data.astype(dtype, copy=False)
    log_probability = np.empty(shape, dtype=dtype)
    log_one_minus_probability = np.empty(shape, dtype=dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.log(probabilities, out=log_probability)
        np.subtract(1.0, probabilities, out=log_one_minus_probability)
        np.log(log_one_minus_probability, out=log_one_minus_probability)
    np.maximum(
        log_probability,
        _BINARY_CROSS_ENTROPY_LOG_FLOOR,
        out=log_probability,
    )
    np.maximum(
        log_one_minus_probability,
        _BINARY_CROSS_ENTROPY_LOG_FLOOR,
        out=log_one_minus_probability,
    )
    per_element = np.empty(shape, dtype=dtype)
    np.multiply(targets, log_probability, out=per_element)
    target_derivative = np.empty(shape, dtype=dtype)
    np.subtract(1.0, targets, out=target_derivative)
    np.multiply(
        target_derivative,
        log_one_minus_probability,
        out=target_derivative,
    )
    np.add(per_element, target_derivative, out=per_element)
    np.negative(per_element, out=per_element)
    per_element[per_element == 0.0] = 0.0

    # PyTorch differentiates the target through the unclamped logit even
    # though the forward logs are clamped at -100.
    with np.errstate(divide="ignore", invalid="ignore"):
        np.log(probabilities, out=log_probability)
        np.subtract(1.0, probabilities, out=target_derivative)
        np.log(target_derivative, out=target_derivative)
    np.subtract(target_derivative, log_probability, out=target_derivative)

    input_derivative = log_probability
    np.subtract(probabilities, targets, out=input_derivative)
    np.subtract(1.0, probabilities, out=log_one_minus_probability)
    np.multiply(
        log_one_minus_probability,
        probabilities,
        out=log_one_minus_probability,
    )
    epsilon = np.asarray(
        _BINARY_CROSS_ENTROPY_GRAD_EPSILON,
        dtype=dtype,
    ).item()
    np.maximum(log_one_minus_probability, epsilon, out=log_one_minus_probability)
    np.divide(input_derivative, log_one_minus_probability, out=input_derivative)
    return _finish_typed_elementwise_loss(
        input,
        target,
        reduction,
        contract,
        per_element,
        input_derivative,
        target_derivative,
    )


def smooth_l1_loss(input: Tensor, target: Tensor, beta: float = 1.0, reduction: str = "mean") -> Tensor:
    """Typed same-shape piecewise quadratic/linear error."""
    normalized_beta = _normalize_smooth_l1_beta(beta)
    contract = _elementwise_loss_contract(
        input,
        target,
        reduction,
        "smooth_l1_loss",
        _SMOOTH_L1_LOSS_WORK_VISIT_FACTOR,
    )
    shape, _, _, _, compute_dtype, _ = contract
    dtype = np.dtype(compute_dtype)
    with np.errstate(over="ignore", under="ignore"):
        compute_beta = float(np.asarray(normalized_beta, dtype=dtype).item())
    if normalized_beta > 0.0 and (compute_beta == 0.0 or not np.isfinite(compute_beta)):
        raise ValueError(
            f"smooth_l1_loss: beta {normalized_beta} is not representable as "
            f"a finite nonzero {compute_dtype} scalar"
        )
    normalized_beta = compute_beta
    left = input.data.astype(dtype, copy=False)
    right = target.data.astype(dtype, copy=False)
    difference = np.empty(shape, dtype=dtype)
    np.subtract(left, right, out=difference)
    per_element = np.empty(shape, dtype=dtype)
    np.abs(difference, out=per_element)
    if normalized_beta == 0.0:
        np.sign(difference, out=difference)
    else:
        quadratic_mask = per_element < normalized_beta
        np.subtract(per_element, normalized_beta * 0.5, out=per_element)
        quadratic = np.empty(shape, dtype=dtype)
        np.multiply(difference, difference, out=quadratic)
        np.divide(quadratic, normalized_beta, out=quadratic)
        np.multiply(quadratic, 0.5, out=quadratic)
        np.copyto(per_element, quadratic, where=quadratic_mask)
        np.divide(difference, normalized_beta, out=quadratic)
        np.sign(difference, out=difference)
        np.copyto(difference, quadratic, where=quadratic_mask)
        del quadratic, quadratic_mask
    return _finish_typed_elementwise_loss(
        input,
        target,
        reduction,
        contract,
        per_element,
        difference,
    )


def kl_div_loss(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    log_target: bool = False,
) -> Tensor:
    """Typed KL divergence with native zero-target and derivative semantics."""
    if type(log_target) is not bool:
        raise ValueError(
            "kl_div: log_target must be an exact bool, got "
            f"{type(log_target).__name__}"
        )
    contract = _elementwise_loss_contract(
        input,
        target,
        reduction,
        "kl_div",
        _KL_DIV_WORK_VISIT_FACTOR,
        compute_buffers=4,
        mask_buffers=1,
        allow_batchmean=True,
    )
    shape, _, _, _, compute_dtype, _ = contract
    dtype = np.dtype(compute_dtype)
    input_array = input.data.astype(dtype, copy=False)
    target_array = target.data.astype(dtype, copy=False)
    per_element = np.empty(shape, dtype=dtype)
    target_derivative = np.empty(shape, dtype=dtype)
    input_derivative = np.empty(shape, dtype=dtype)
    if log_target:
        with np.errstate(over="ignore", invalid="ignore"):
            np.exp(target_array, out=input_derivative)
        np.subtract(target_array, input_array, out=per_element)
        np.multiply(input_derivative, per_element, out=per_element)
        np.add(per_element, input_derivative, out=target_derivative)
        np.negative(input_derivative, out=input_derivative)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            np.log(target_array, out=target_derivative)
            np.multiply(target_array, target_derivative, out=per_element)
        np.copyto(per_element, 0.0, where=target_array == 0.0)
        np.multiply(target_array, input_array, out=input_derivative)
        np.subtract(per_element, input_derivative, out=per_element)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(target_array, target_array, out=input_derivative)
        np.copyto(target_derivative, 0.0, where=target_array == 0.0)
        np.add(target_derivative, input_derivative, out=target_derivative)
        np.subtract(target_derivative, input_array, out=target_derivative)
        np.negative(target_array, out=input_derivative)
    per_element[per_element == 0.0] = 0.0
    batch_denominator = 1 if len(shape) == 0 else shape[0]
    return _finish_typed_elementwise_loss(
        input,
        target,
        reduction,
        contract,
        per_element,
        input_derivative,
        target_derivative,
        batch_denominator=batch_denominator,
    )


# ─── Spatial / shape ops ───────────────────────────────────

def pad(input: Tensor, pad, mode: str = "constant", value=None) -> Tensor:
    """Pad input.

    pad: a sequence of even length, paired by dimension, last-dim-first
    (matching torch.nn.functional.pad). E.g. for a 2D input,
    pad=(left, right, top, bottom).

    Currently supports mode='constant' only — that covers nearly every
    real PyTorch lab. Add reflect / replicate when something needs them.
    """
    pad_width, normalized_value, _ = _normalize_pad_contract(
        input,
        pad,
        mode,
        value,
    )
    out_data = np.pad(
        input.data,
        pad_width,
        mode="constant",
        constant_values=normalized_value,
    )
    out = Tensor(
        np.array(out_data, dtype=input.data.dtype, copy=True),
        dtype=input.dtype,
    )

    def backward(g):
        slices = tuple(
            slice(lower, lower + size)
            for (lower, _), size in zip(pad_width, input.data.shape)
        )
        return (g.data[slices].copy().astype(input.data.dtype, copy=False),)

    if input.requires_grad and input.dtype in ("float16", "float32", "float64"):
        return _build_ctx(out, (input,), backward)
    return out


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
                                  is_causal: bool = False, scale=None,
                                  device=None) -> Tensor:
    """Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V.

    Supports the subset of torch's API that matters most:
      - attn_mask: boolean (True = block) or float (added to scores).
      - is_causal: builds a triangular block mask.
      - scale: override 1/sqrt(d_k).

    Last dim of Q and K must agree; last-but-one of K and V must agree.
    """
    if dropout_p != 0.0:
        raise NotImplementedError("scaled_dot_product_attention: dropout_p > 0 not supported")
    if device is not None and (attn_mask is not None or is_causal or scale is not None):
        raise NotImplementedError(
            "scaled_dot_product_attention(device=...): KernelDevice bridge supports only "
            "unmasked default-scale attention"
        )
    Qd = query.data
    Kd = key.data
    Vd = value.data
    d_k = Qd.shape[-1]
    s = (1.0 / np.sqrt(d_k)) if scale is None else float(scale)
    if device is not None:
        if Qd.ndim != 2 or Kd.ndim != 2 or Vd.ndim != 2:
            raise NotImplementedError(
                "scaled_dot_product_attention(device=...): KernelDevice bridge supports "
                f"2D Q/K/V only (got ranks {Qd.ndim}/{Kd.ndim}/{Vd.ndim})"
            )
        out_data = _device.attention(device, Qd, Kd, Vd)
        scores = np.matmul(Qd, np.swapaxes(Kd, -1, -2)) * s
        sm = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(sm)
        attn = e / e.sum(axis=-1, keepdims=True)
    else:
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


# ─────────────────────────────────────────────────────────────────────
# PyTorch-name aliases. The native names use *_loss suffixes for the loss
# family (cross_entropy_loss, mse_loss, ...); PyTorch's torch.nn.functional
# drops the suffix for some (cross_entropy, nll, bce_with_logits). Expose
# both so user code copied from PyTorch tutorials works against the grad
# namespace without going through install_torch_alias().
cross_entropy = cross_entropy_loss
nll = nll_loss
bce_with_logits = bce_with_logits_loss
binary_cross_entropy_with_logits = bce_with_logits_loss
kl_div = kl_div_loss
