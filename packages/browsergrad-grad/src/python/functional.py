
"""browsergrad_grad.functional — stateless ops with autograd."""

import math

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


def cross_entropy_loss(
    input: Tensor,
    target: Tensor,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
) -> Tensor:
    if not isinstance(target, Tensor):
        target = Tensor(np.asarray(target, dtype=np.int64), dtype="int64")
    return cross_entropy(
        input,
        target,
        weight=weight,
        size_average=size_average,
        ignore_index=ignore_index,
        reduce=reduce,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


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


_DROPOUT_SCALAR_TYPES = (
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


def _normalize_dropout_probability(p) -> float:
    if type(p) not in _DROPOUT_SCALAR_TYPES or type(p) is bool:
        raise ValueError("dropout: p must be an exact real scalar")
    normalized = float(p)
    if not np.isfinite(normalized) or normalized < 0.0 or normalized > 1.0:
        raise ValueError(
            f"dropout: p must be finite and in [0, 1], got {normalized!r}"
        )
    return normalized


def dropout(
    input: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    """Functional inverted dropout. Matches torch.nn.functional.dropout.

    When training=False or p==0, returns x unchanged.
    """
    if not isinstance(input, Tensor):
        raise TypeError(
            f"dropout: input must be a Tensor, got {type(input).__name__}"
        )
    normalized_p = _normalize_dropout_probability(p)
    if type(training) is not bool:
        raise ValueError("dropout: training must be an exact bool")
    if type(inplace) is not bool:
        raise ValueError("dropout: inplace must be an exact bool")
    if inplace:
        raise NotImplementedError(
            "dropout(inplace=True) requires typed mutation semantics and is "
            "not supported"
        )
    if input.data.ndim > 32:
        raise ValueError("dropout: input rank exceeds the 32-rank ceiling")
    if any(extent > (1 << 28) for extent in input.data.shape):
        raise ValueError("dropout: input extent exceeds the per-axis ceiling")
    if input.data.nbytes > (1 << 28):
        raise ValueError("dropout: output exceeds the 268435456-byte ceiling")
    if not training or normalized_p == 0.0 or input.data.size == 0:
        return input
    if normalized_p == 1.0:
        result = np.zeros_like(input.data)
        out = Tensor(result, dtype=result.dtype.name)
        return _build_ctx(
            out,
            (input,),
            lambda g: (np.zeros_like(input.data),),
        )
    if input.data.dtype.name not in ("float16", "float32", "float64"):
        raise ValueError(
            "dropout: stochastic training requires float16, float32, or "
            f"float64 input, got {input.data.dtype.name!r}"
        )
    elements = int(input.data.size)
    if elements > (1 << 28) // 8:
        raise ValueError("dropout: projected work exceeds the element-visit ceiling")
    if input.data.nbytes + elements * 9 > (1 << 28):
        raise ValueError("dropout: projected workspace exceeds the byte ceiling")

    mask = np.random.random(input.data.shape) >= normalized_p
    result = np.zeros_like(input.data)
    scale = np.asarray(1.0 / (1.0 - normalized_p), dtype=input.data.dtype)
    np.multiply(input.data, scale, out=result, where=mask)
    out = Tensor(result, dtype=result.dtype.name)

    def backward(g):
        gradient = np.zeros_like(input.data)
        np.multiply(g.data, scale, out=gradient, where=mask)
        return (gradient,)

    return _build_ctx(out, (input,), backward)


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


def _cross_entropy_contract(
    input,
    target,
    weight,
    reduction,
    ignore_index,
    label_smoothing,
):
    if not isinstance(input, Tensor):
        raise TypeError(
            f"cross_entropy: input must be a Tensor, got {type(input).__name__}"
        )
    if not isinstance(target, Tensor):
        raise TypeError(
            f"cross_entropy: target must be a Tensor, got {type(target).__name__}"
        )
    if type(input.data) is not np.ndarray:
        raise TypeError("cross_entropy: input data must be an exact ndarray")
    if type(target.data) is not np.ndarray:
        raise TypeError("cross_entropy: target data must be an exact ndarray")
    if type(reduction) is not str:
        raise ValueError(
            "cross_entropy: reduction must be a string, got "
            f"{type(reduction).__name__}"
        )
    if reduction not in ("none", "sum", "mean"):
        raise ValueError(
            "cross_entropy: reduction must be 'none', 'sum', or 'mean', got "
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
        raise ValueError(
            "cross_entropy: ignore_index must be an exact integer"
        )
    normalized_ignore_index = int(ignore_index)
    if (
        normalized_ignore_index < -(1 << 63)
        or normalized_ignore_index > (1 << 63) - 1
    ):
        raise ValueError("cross_entropy: ignore_index must fit signed int64")
    if (
        type(label_smoothing) not in _SMOOTH_L1_BETA_TYPES
        or type(label_smoothing) is bool
    ):
        raise ValueError(
            "cross_entropy: label_smoothing must be an exact real scalar"
        )
    normalized_smoothing = float(label_smoothing)
    if (
        not np.isfinite(normalized_smoothing)
        or normalized_smoothing < 0.0
        or normalized_smoothing > 1.0
    ):
        raise ValueError(
            "cross_entropy: label_smoothing must be finite and in [0, 1], "
            f"got {normalized_smoothing!r}"
        )

    input_shape = tuple(input.data.shape)
    target_shape = tuple(target.data.shape)
    if len(input_shape) < 1:
        raise ValueError(
            "cross_entropy: batch_rank 0 leaves no user input dimension "
            f"in shape {input_shape}"
        )
    shapes = [input_shape, target_shape]
    if weight is not None:
        if not isinstance(weight, Tensor):
            raise TypeError(
                "cross_entropy: weight must be a Tensor or None, got "
                f"{type(weight).__name__}"
            )
        if weight.requires_grad:
            raise ValueError(
                "cross_entropy: weight is non-differentiable and must not "
                "require grad"
            )
        if type(weight.data) is not np.ndarray:
            raise TypeError("cross_entropy: weight data must be an exact ndarray")
        shapes.append(tuple(weight.data.shape))
    for index, shape in enumerate(shapes):
        if len(shape) > _L1_LOSS_RANK_MAX:
            raise ValueError(
                f"cross_entropy input {index} rank {len(shape)} exceeds the "
                f"{_L1_LOSS_RANK_MAX}-rank ceiling"
            )
        for axis, extent in enumerate(shape):
            if extent > _L1_LOSS_OUTPUT_EXTENT_MAX:
                raise ValueError(
                    f"cross_entropy input {index} extent {extent} on axis "
                    f"{axis} exceeds the {_L1_LOSS_OUTPUT_EXTENT_MAX}-element "
                    "per-axis ceiling"
                )

    input_dtype = input.data.dtype.name
    target_dtype = target.data.dtype.name
    if input_dtype not in _L1_LOSS_FLOATING_DTYPES:
        raise ValueError(
            f"cross_entropy input dtype {input_dtype!r} is not supported; "
            "expected float16, float32, or float64"
        )
    class_axis = 0 if len(input_shape) == 1 else 1
    class_count = input_shape[class_axis]
    if class_count == 0:
        raise ValueError(
            "cross_entropy: class dimension must contain at least one class"
        )
    position_shape = (
        input_shape[:class_axis] + input_shape[class_axis + 1:]
    )
    target_mode = (
        "probabilities" if target_shape == input_shape else "indices"
    )
    if target_mode == "probabilities":
        if target_dtype != input_dtype:
            raise ValueError(
                f"cross_entropy probability target dtype {target_dtype!r} "
                f"must equal input dtype {input_dtype!r}"
            )
        if normalized_ignore_index >= 0:
            raise ValueError(
                "cross_entropy: ignore_index is not supported for floating "
                "point targets unless it is negative"
            )
    else:
        if target_shape != position_shape:
            raise ValueError(
                f"cross_entropy: index target shape {target_shape} must equal "
                f"input shape {input_shape} with class axis {class_axis} "
                f"removed ({position_shape})"
            )
        if target_dtype != "int64":
            raise ValueError(
                f"cross_entropy index target dtype {target_dtype!r} is not "
                "supported; expected int64"
            )

    weight_shape = None
    if weight is not None:
        weight_shape = tuple(weight.data.shape)
        if weight_shape != (class_count,):
            raise ValueError(
                f"cross_entropy: weight shape {weight_shape} must equal class "
                f"count ({class_count},)"
            )
        if weight.data.dtype.name != input_dtype:
            raise ValueError(
                f"cross_entropy: weight dtype {weight.data.dtype.name!r} "
                f"must equal input dtype {input_dtype!r}"
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
        > _L1_LOSS_WORK_ELEMENT_MAX // _CROSS_ENTROPY_WORK_VISIT_FACTOR
    ):
        work_elements = _L1_LOSS_WORK_ELEMENT_MAX + 1
    else:
        work_elements = (
            capacity_elements * _CROSS_ENTROPY_WORK_VISIT_FACTOR
        )
    if work_elements > _L1_LOSS_WORK_ELEMENT_MAX:
        raise ValueError(
            "cross_entropy: projected work exceeds the "
            f"{_L1_LOSS_WORK_ELEMENT_MAX}-element-visit ceiling"
        )
    output_shape = position_shape if reduction == "none" else ()
    output_elements = _l1_loss_checked_product(
        output_shape, _L1_LOSS_OUTPUT_BYTE_MAX
    )
    output_bytes = output_elements * input.data.dtype.itemsize
    if output_bytes > _L1_LOSS_OUTPUT_BYTE_MAX:
        raise ValueError(
            f"cross_entropy: output requires {output_bytes} bytes, exceeding "
            f"the {_L1_LOSS_OUTPUT_BYTE_MAX}-byte ceiling"
        )
    compute_dtype = "float32" if input_dtype == "float16" else input_dtype
    compute_bytes = np.dtype(compute_dtype).itemsize
    cast_bytes = 0
    if input_dtype != compute_dtype:
        cast_bytes += input_elements * compute_bytes
        if target_mode == "probabilities":
            cast_bytes += target_elements * compute_bytes
        cast_bytes += weight_elements * compute_bytes
    position_elements = _l1_loss_checked_product(
        position_shape, _L1_LOSS_WORK_ELEMENT_MAX
    )
    workspace_bytes = (
        output_bytes
        + cast_bytes
        + input_elements * compute_bytes * 6
        + position_elements * (2 * compute_bytes + 8 + 1)
    )
    if workspace_bytes > _L1_LOSS_WORKSPACE_BYTE_MAX:
        raise ValueError(
            "cross_entropy: projected stable-softmax/target/gradient "
            f"workspace requires {workspace_bytes} bytes, exceeding the "
            f"{_L1_LOSS_WORKSPACE_BYTE_MAX}-byte ceiling"
        )
    return {
        "input_shape": input_shape,
        "target_shape": target_shape,
        "position_shape": position_shape,
        "output_shape": output_shape,
        "input_dtype": input_dtype,
        "target_dtype": target_dtype,
        "compute_dtype": compute_dtype,
        "class_axis": class_axis,
        "class_count": class_count,
        "ignore_index": normalized_ignore_index,
        "reduction": reduction,
        "target_mode": target_mode,
        "has_weight": weight is not None,
        "label_smoothing": normalized_smoothing,
    }


def cross_entropy(
    input: Tensor,
    target: Tensor,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction="mean",
    label_smoothing=0.0,
) -> Tensor:
    """Stable cross entropy for class indices or class probabilities."""
    normalized_reduction = _normalize_legacy_loss_reduction(
        "cross_entropy",
        reduction,
        size_average,
        reduce,
    )
    contract = _cross_entropy_contract(
        input,
        target,
        weight,
        normalized_reduction,
        ignore_index,
        label_smoothing,
    )
    dtype = np.dtype(contract["compute_dtype"])
    logits = input.data.astype(dtype, copy=False)
    maximum = np.max(
        logits,
        axis=contract["class_axis"],
        keepdims=True,
    )
    log_probabilities = np.empty(contract["input_shape"], dtype=dtype)
    np.subtract(logits, maximum, out=log_probabilities)
    exponentials = np.empty(contract["input_shape"], dtype=dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        np.exp(log_probabilities, out=exponentials)
    normalizer = exponentials.sum(
        axis=contract["class_axis"],
        keepdims=True,
        dtype=dtype,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        np.log(normalizer, out=normalizer)
    np.subtract(log_probabilities, normalizer, out=log_probabilities)

    weight_view = None
    if contract["has_weight"]:
        view_shape = [1] * len(contract["input_shape"])
        view_shape[contract["class_axis"]] = contract["class_count"]
        weight_view = weight.data.astype(dtype, copy=False).reshape(view_shape)

    valid = None
    safe_targets = None
    selected_weight = None
    if contract["target_mode"] == "probabilities":
        coefficient = target.data.astype(dtype, copy=False)
        if contract["label_smoothing"] != 0.0:
            coefficient = np.array(coefficient, dtype=dtype, copy=True)
            np.multiply(
                coefficient,
                1.0 - contract["label_smoothing"],
                out=coefficient,
            )
            np.add(
                coefficient,
                contract["label_smoothing"] / contract["class_count"],
                out=coefficient,
            )
        if weight_view is not None:
            coefficient = np.multiply(coefficient, weight_view)
        per_class = np.multiply(log_probabilities, coefficient)
        per_position = per_class.sum(
            axis=contract["class_axis"],
            dtype=dtype,
        )
        np.negative(per_position, out=per_position)
    else:
        targets = target.data
        valid = targets != contract["ignore_index"]
        if bool(valid.any()):
            minimum = int(np.min(targets, where=valid, initial=0))
            maximum = int(np.max(targets, where=valid, initial=0))
            if minimum < 0 or maximum >= contract["class_count"]:
                raise ValueError(
                    "cross_entropy: target values must be in "
                    f"[0, {contract['class_count']}) or equal ignore_index "
                    f"{contract['ignore_index']}"
                )
        safe_targets = np.where(valid, targets, 0)
        if contract["has_weight"]:
            weights = weight.data.astype(dtype, copy=False)
            selected_weight = weights[safe_targets]
            selected_weight = np.where(valid, selected_weight, 0.0)
        else:
            selected_weight = valid.astype(dtype, copy=False)
        class_last = np.moveaxis(
            log_probabilities, contract["class_axis"], -1
        )
        per_position = np.take_along_axis(
            class_last,
            safe_targets[..., None],
            axis=-1,
        )[..., 0]
        np.negative(per_position, out=per_position)
        if contract["has_weight"]:
            np.multiply(per_position, selected_weight, out=per_position)
        if contract["label_smoothing"] != 0.0:
            smooth_source = log_probabilities
            if weight_view is not None:
                smooth_source = np.multiply(smooth_source, weight_view)
            smooth_loss = smooth_source.sum(
                axis=contract["class_axis"],
                dtype=dtype,
            )
            np.negative(smooth_loss, out=smooth_loss)
            np.multiply(
                per_position,
                1.0 - contract["label_smoothing"],
                out=per_position,
            )
            np.multiply(
                smooth_loss,
                contract["label_smoothing"] / contract["class_count"],
                out=smooth_loss,
            )
            np.add(per_position, smooth_loss, out=per_position)
        np.copyto(per_position, 0.0, where=~valid)
        coefficient = np.zeros(contract["input_shape"], dtype=dtype)
        if contract["label_smoothing"] != 0.0:
            if weight_view is None:
                coefficient.fill(
                    contract["label_smoothing"] / contract["class_count"]
                )
            else:
                coefficient = np.array(
                    np.broadcast_to(weight_view, contract["input_shape"]),
                    dtype=dtype,
                    copy=True,
                )
                np.multiply(
                    coefficient,
                    contract["label_smoothing"] / contract["class_count"],
                    out=coefficient,
                )
        coefficient_last = np.moveaxis(
            coefficient, contract["class_axis"], -1
        )
        selected = (
            1.0 - contract["label_smoothing"]
        ) * selected_weight
        previous = np.take_along_axis(
            coefficient_last,
            safe_targets[..., None],
            axis=-1,
        )[..., 0]
        np.add(previous, selected, out=previous)
        np.put_along_axis(
            coefficient_last,
            safe_targets[..., None],
            previous[..., None],
            axis=-1,
        )
        invalid = np.broadcast_to(
            np.expand_dims(~valid, axis=contract["class_axis"]),
            contract["input_shape"],
        )
        np.copyto(coefficient, 0.0, where=invalid)

    if contract["reduction"] == "none":
        result = per_position
    else:
        numerator = per_position.sum(dtype=dtype)
        if contract["reduction"] == "sum":
            result = numerator
        else:
            if contract["target_mode"] == "indices":
                denominator = (
                    selected_weight.sum(dtype=dtype)
                    if contract["has_weight"]
                    else valid.sum()
                )
            else:
                denominator = per_position.size
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.divide(numerator, denominator)
    out = Tensor(
        np.array(
            result,
            dtype=np.dtype(contract["input_dtype"]),
            copy=True,
        ),
        dtype=contract["input_dtype"],
    )

    saved_log_probabilities = np.array(
        log_probabilities, dtype=dtype, copy=True
    )
    saved_coefficient = np.array(coefficient, dtype=dtype, copy=True)
    saved_valid = (
        None if valid is None else np.array(valid, dtype=np.bool_, copy=True)
    )
    saved_selected_weight = (
        None
        if selected_weight is None
        else np.array(selected_weight, dtype=dtype, copy=True)
    )
    saved_weight_view = (
        None
        if weight_view is None
        else np.array(weight_view, dtype=dtype, copy=True)
    )

    def backward(g):
        upstream = g.data.astype(dtype, copy=False)
        if contract["reduction"] != "none":
            upstream = np.broadcast_to(upstream, contract["position_shape"])
            if contract["reduction"] == "mean":
                if contract["target_mode"] == "indices":
                    denominator = (
                        saved_selected_weight.sum(dtype=dtype)
                        if contract["has_weight"]
                        else saved_valid.sum()
                    )
                else:
                    denominator = int(np.prod(contract["position_shape"]))
                with np.errstate(divide="ignore", invalid="ignore"):
                    upstream = np.divide(upstream, denominator)
        upstream = np.expand_dims(
            upstream,
            axis=contract["class_axis"],
        )
        probabilities = np.empty(contract["input_shape"], dtype=dtype)
        with np.errstate(over="ignore", invalid="ignore"):
            np.exp(saved_log_probabilities, out=probabilities)
        coefficient_sum = saved_coefficient.sum(
            axis=contract["class_axis"],
            keepdims=True,
            dtype=dtype,
        )
        np.multiply(probabilities, coefficient_sum, out=probabilities)
        np.subtract(probabilities, saved_coefficient, out=probabilities)
        np.multiply(probabilities, upstream, out=probabilities)
        if saved_valid is not None:
            invalid = np.broadcast_to(
                np.expand_dims(
                    ~saved_valid,
                    axis=contract["class_axis"],
                ),
                contract["input_shape"],
            )
            np.copyto(probabilities, 0.0, where=invalid)
        input_gradient = probabilities.astype(
            np.dtype(contract["input_dtype"]),
            copy=probabilities.dtype.name != contract["input_dtype"],
        )
        if contract["target_mode"] != "probabilities":
            return (input_gradient,)
        if contract["label_smoothing"] == 1.0:
            target_gradient = np.zeros(contract["target_shape"], dtype=dtype)
        else:
            target_gradient = np.empty(contract["target_shape"], dtype=dtype)
            np.negative(saved_log_probabilities, out=target_gradient)
            if saved_weight_view is not None:
                np.multiply(
                    target_gradient,
                    saved_weight_view,
                    out=target_gradient,
                )
            if contract["label_smoothing"] != 0.0:
                np.multiply(
                    target_gradient,
                    1.0 - contract["label_smoothing"],
                    out=target_gradient,
                )
            np.multiply(target_gradient, upstream, out=target_gradient)
        return (
            input_gradient,
            target_gradient.astype(
                np.dtype(contract["target_dtype"]),
                copy=target_gradient.dtype.name != contract["target_dtype"],
            ),
        )

    parents = (
        (input, target)
        if contract["target_mode"] == "probabilities"
        else (input,)
    )
    return _build_ctx(out, parents, backward)


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
_CROSS_ENTROPY_WORK_VISIT_FACTOR = 64
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


_INTERPOLATE_2D_FLOATING_DTYPES = frozenset(
    {"float16", "float32", "float64"}
)
_INTERPOLATE_2D_DTYPE_BYTES = {
    "float16": 2,
    "float32": 4,
    "float64": 8,
}
_INTERPOLATE_2D_INTEGER_TYPES = (
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
_INTERPOLATE_2D_REAL_TYPES = _INTERPOLATE_2D_INTEGER_TYPES + (
    float,
    np.float16,
    np.float32,
    np.float64,
)
_INTERPOLATE_2D_OUTPUT_BYTE_MAX = 1 << 28
_INTERPOLATE_2D_OUTPUT_EXTENT_MAX = 1 << 20
_INTERPOLATE_2D_WORK_ELEMENT_MAX = 1 << 28
_INTERPOLATE_2D_WORKSPACE_BYTE_MAX = 1 << 28
_INTERPOLATE_2D_NEAREST_WORK_VISIT_FACTOR = 4
_INTERPOLATE_2D_BILINEAR_WORK_VISIT_FACTOR = 32


def _interpolate_2d_checked_product(extents):
    product = 1
    for extent in extents:
        if extent == 0:
            return 0
        if product > _INTERPOLATE_2D_WORK_ELEMENT_MAX // extent:
            return _INTERPOLATE_2D_WORK_ELEMENT_MAX + 1
        product *= extent
    return product


def _interpolate_2d_size(value):
    if type(value) in _INTERPOLATE_2D_INTEGER_TYPES:
        raw = (value, value)
    elif type(value) in (tuple, list) and len(value) == 2:
        raw = (value[0], value[1])
    else:
        raise ValueError(
            "interpolate: size must be an integer or a length-2 tuple/list"
        )
    normalized = []
    for axis, extent in enumerate(raw):
        if type(extent) not in _INTERPOLATE_2D_INTEGER_TYPES:
            raise ValueError(
                f"interpolate: size[{axis}] must be a built-in or NumPy integer"
            )
        canonical = int(extent)
        if (
            canonical <= 0
            or canonical > _INTERPOLATE_2D_OUTPUT_EXTENT_MAX
        ):
            raise ValueError(
                f"interpolate: size[{axis}] must be in "
                f"[1, {_INTERPOLATE_2D_OUTPUT_EXTENT_MAX}], got {canonical}"
            )
        normalized.append(canonical)
    return normalized[0], normalized[1]


def _interpolate_2d_scale_factors(value):
    if type(value) in _INTERPOLATE_2D_REAL_TYPES:
        raw = (value, value)
    elif type(value) in (tuple, list) and len(value) == 2:
        raw = (value[0], value[1])
    else:
        raise ValueError(
            "interpolate: scale_factor must be a real scalar or a "
            "length-2 tuple/list"
        )
    normalized = []
    for axis, scale in enumerate(raw):
        if type(scale) not in _INTERPOLATE_2D_REAL_TYPES:
            raise ValueError(
                f"interpolate: scale_factor[{axis}] must be an exact real scalar"
            )
        canonical = float(scale)
        if not math.isfinite(canonical) or canonical <= 0.0:
            raise ValueError(
                f"interpolate: scale_factor[{axis}] must be finite and positive"
            )
        normalized.append(canonical)
    return normalized[0], normalized[1]


def _interpolate_2d_scaled_extent(input_extent, scale):
    if scale > _INTERPOLATE_2D_OUTPUT_EXTENT_MAX / input_extent:
        raise ValueError(
            "interpolate: scaled output extent exceeds the "
            f"{_INTERPOLATE_2D_OUTPUT_EXTENT_MAX}-element ceiling"
        )
    output_extent = math.floor(input_extent * scale)
    if (
        output_extent <= 0
        or output_extent > _INTERPOLATE_2D_OUTPUT_EXTENT_MAX
    ):
        raise ValueError(
            "interpolate: scale_factor produces a non-positive or oversized "
            f"output extent {output_extent}"
        )
    return output_extent


def _normalize_interpolate_2d_contract(
    input,
    size,
    scale_factor,
    mode,
    align_corners,
    recompute_scale_factor,
    antialias,
):
    if type(input) is not Tensor:
        raise TypeError(
            f"interpolate: input must be a Tensor, got {type(input).__name__}"
        )
    shape = tuple(input.data.shape)
    dtype = input.dtype
    if len(shape) != 4:
        raise ValueError(
            "interpolate: v1 requires rank-4 (N,C,H,W) input, "
            f"got shape {shape!r}"
        )
    if dtype not in _INTERPOLATE_2D_FLOATING_DTYPES:
        raise ValueError(
            "interpolate: v1 requires float16, float32, or float64 input, "
            f"got {dtype!r}"
        )
    for axis, extent in enumerate(shape):
        if extent > _INTERPOLATE_2D_OUTPUT_EXTENT_MAX:
            raise ValueError(
                f"interpolate: input extent {extent} on axis {axis} exceeds the "
                f"{_INTERPOLATE_2D_OUTPUT_EXTENT_MAX}-element ceiling"
            )
    if shape[-2] == 0 or shape[-1] == 0:
        raise ValueError("interpolate: input spatial extents must be positive")
    input_elements = _interpolate_2d_checked_product(shape)
    dtype_bytes = _INTERPOLATE_2D_DTYPE_BYTES[dtype]
    if input_elements * dtype_bytes > _INTERPOLATE_2D_OUTPUT_BYTE_MAX:
        raise ValueError(
            "interpolate: input bytes exceed the "
            f"{_INTERPOLATE_2D_OUTPUT_BYTE_MAX}-byte ceiling"
        )
    if type(mode) is not str or mode not in ("nearest", "bilinear"):
        raise ValueError(
            "interpolate: v1 mode must be exactly 'nearest' or 'bilinear'"
        )
    if mode == "nearest":
        if align_corners is not None:
            raise ValueError(
                "interpolate: align_corners must be None for nearest mode"
            )
        normalized_align_corners = False
    elif align_corners is None:
        normalized_align_corners = False
    elif type(align_corners) is bool:
        normalized_align_corners = align_corners
    else:
        raise ValueError(
            "interpolate: align_corners must be None or an exact bool"
        )
    if type(antialias) is not bool:
        raise ValueError("interpolate: antialias must be an exact bool")
    if antialias:
        raise ValueError(
            "interpolate: antialias=True is outside the typed v1 profile"
        )
    if (size is None) == (scale_factor is None):
        raise ValueError(
            "interpolate: exactly one of size or scale_factor must be provided"
        )
    if (
        recompute_scale_factor is not None
        and type(recompute_scale_factor) is not bool
    ):
        raise ValueError(
            "interpolate: recompute_scale_factor must be None or an exact bool"
        )
    if size is not None:
        if recompute_scale_factor is not None:
            raise ValueError(
                "interpolate: recompute_scale_factor is invalid with explicit size"
            )
        output_size = _interpolate_2d_size(size)
        scale_factors = None
        recompute = False
    else:
        scale_factors = _interpolate_2d_scale_factors(scale_factor)
        output_size = (
            _interpolate_2d_scaled_extent(shape[-2], scale_factors[0]),
            _interpolate_2d_scaled_extent(shape[-1], scale_factors[1]),
        )
        recompute = (
            False
            if recompute_scale_factor is None
            else recompute_scale_factor
        )
    if normalized_align_corners:
        coordinate_scales = tuple(
            0.0
            if output_extent == 1
            else (input_extent - 1) / (output_extent - 1)
            for input_extent, output_extent in zip(shape[-2:], output_size)
        )
    elif scale_factors is not None and not recompute:
        coordinate_scales = tuple(1.0 / scale for scale in scale_factors)
    else:
        coordinate_scales = tuple(
            input_extent / output_extent
            for input_extent, output_extent in zip(shape[-2:], output_size)
        )
    output_shape = shape[:-2] + output_size
    output_elements = _interpolate_2d_checked_product(output_shape)
    output_bytes = output_elements * dtype_bytes
    if output_bytes > _INTERPOLATE_2D_OUTPUT_BYTE_MAX:
        raise ValueError(
            f"interpolate: output requires {output_bytes} bytes, exceeding the "
            f"{_INTERPOLATE_2D_OUTPUT_BYTE_MAX}-byte ceiling"
        )
    visit_factor = (
        _INTERPOLATE_2D_NEAREST_WORK_VISIT_FACTOR
        if mode == "nearest"
        else _INTERPOLATE_2D_BILINEAR_WORK_VISIT_FACTOR
    )
    if (
        output_elements
        > _INTERPOLATE_2D_WORK_ELEMENT_MAX // visit_factor
    ):
        raise ValueError(
            "interpolate: projected work exceeds the "
            f"{_INTERPOLATE_2D_WORK_ELEMENT_MAX}-element-visit ceiling"
        )
    input_bytes = input_elements * dtype_bytes
    spatial_output_elements = output_size[0] * output_size[1]
    if mode == "nearest":
        workspace_bytes = (
            input_bytes + output_bytes * 3 + spatial_output_elements * 8
        )
    else:
        workspace_bytes = (
            input_bytes
            + output_bytes * 8
            + spatial_output_elements * 56
            + (output_size[0] + output_size[1]) * 64
        )
    if workspace_bytes > _INTERPOLATE_2D_WORKSPACE_BYTE_MAX:
        raise ValueError(
            f"interpolate: projected workspace requires {workspace_bytes} bytes, "
            f"exceeding the {_INTERPOLATE_2D_WORKSPACE_BYTE_MAX}-byte ceiling"
        )
    return {
        "input_shape": shape,
        "output_shape": output_shape,
        "output_dtype": dtype,
        "output_size": output_size,
        "mode": mode,
        "align_corners": normalized_align_corners,
        "scale_factors": scale_factors,
        "recompute_scale_factor": recompute,
        "coordinate_scales": coordinate_scales,
    }


def _interpolate_2d_axis_geometry(
    input_extent,
    output_extent,
    coordinate_scale,
    mode,
    align_corners,
    compute_dtype,
):
    positions = np.arange(output_extent, dtype=compute_dtype)
    if mode == "nearest":
        indices = np.floor(positions * coordinate_scale).astype(np.int64)
        return (np.clip(indices, 0, input_extent - 1),)
    if align_corners:
        coordinates = positions * coordinate_scale
    else:
        coordinates = (
            (positions + compute_dtype.type(0.5)) * coordinate_scale - 0.5
        )
    lower = np.floor(coordinates).astype(np.int64)
    upper = lower + 1
    fraction = np.asarray(coordinates - lower, dtype=compute_dtype)
    return (
        np.clip(lower, 0, input_extent - 1),
        np.clip(upper, 0, input_extent - 1),
        fraction,
    )


def _interpolate_2d_geometry(contract):
    compute_dtype = np.dtype(
        "float32"
        if contract["output_dtype"] == "float16"
        else contract["output_dtype"]
    )
    h_in, w_in = contract["input_shape"][-2:]
    out_h, out_w = contract["output_size"]
    return (
        compute_dtype,
        _interpolate_2d_axis_geometry(
            h_in,
            out_h,
            contract["coordinate_scales"][0],
            contract["mode"],
            contract["align_corners"],
            compute_dtype,
        ),
        _interpolate_2d_axis_geometry(
            w_in,
            out_w,
            contract["coordinate_scales"][1],
            contract["mode"],
            contract["align_corners"],
            compute_dtype,
        ),
    )


def _execute_interpolate_2d(contract, source):
    compute_dtype, h_geometry, w_geometry = _interpolate_2d_geometry(contract)
    if contract["mode"] == "nearest":
        result = source[
            ...,
            h_geometry[0][:, None],
            w_geometry[0][None, :],
        ]
    else:
        source_compute = source.astype(compute_dtype, copy=False)
        h0, h1, h_fraction = h_geometry
        w0, w1, w_fraction = w_geometry
        v00 = source_compute[..., h0[:, None], w0[None, :]]
        v01 = source_compute[..., h0[:, None], w1[None, :]]
        v10 = source_compute[..., h1[:, None], w0[None, :]]
        v11 = source_compute[..., h1[:, None], w1[None, :]]
        h_weight = h_fraction[:, None]
        w_weight = w_fraction[None, :]
        one = compute_dtype.type(1.0)
        result = (
            (one - h_weight)
            * ((one - w_weight) * v00 + w_weight * v01)
            + h_weight * ((one - w_weight) * v10 + w_weight * v11)
        )
    return np.array(
        result,
        dtype=np.dtype(contract["output_dtype"]),
        copy=True,
    )


def _execute_interpolate_2d_vjp(contract, dy):
    compute_dtype, h_geometry, w_geometry = _interpolate_2d_geometry(contract)
    h_in, w_in = contract["input_shape"][-2:]
    out_h, out_w = contract["output_size"]
    prefix_elements = _interpolate_2d_checked_product(
        contract["input_shape"][:-2]
    )
    dy_flat = dy.astype(compute_dtype, copy=False).reshape(
        (prefix_elements, out_h * out_w)
    )
    dx_flat = np.zeros(
        (prefix_elements, h_in * w_in),
        dtype=compute_dtype,
    )
    if contract["mode"] == "nearest":
        source_indices = (
            h_geometry[0][:, None] * w_in + w_geometry[0][None, :]
        ).reshape(-1)
        np.add.at(dx_flat, (slice(None), source_indices), dy_flat)
    else:
        h0, h1, h_fraction = h_geometry
        w0, w1, w_fraction = w_geometry
        h_weight = h_fraction[:, None]
        w_weight = w_fraction[None, :]
        one = compute_dtype.type(1.0)
        corners = (
            (
                (h0[:, None] * w_in + w0[None, :]).reshape(-1),
                ((one - h_weight) * (one - w_weight)).reshape(-1),
            ),
            (
                (h0[:, None] * w_in + w1[None, :]).reshape(-1),
                ((one - h_weight) * w_weight).reshape(-1),
            ),
            (
                (h1[:, None] * w_in + w0[None, :]).reshape(-1),
                (h_weight * (one - w_weight)).reshape(-1),
            ),
            (
                (h1[:, None] * w_in + w1[None, :]).reshape(-1),
                (h_weight * w_weight).reshape(-1),
            ),
        )
        for source_indices, weights in corners:
            np.add.at(
                dx_flat,
                (slice(None), source_indices),
                dy_flat * weights,
            )
    return np.array(
        dx_flat.reshape(contract["input_shape"]),
        dtype=np.dtype(contract["output_dtype"]),
        copy=True,
    )


def interpolate(
    input: Tensor,
    size=None,
    scale_factor=None,
    mode: str = "nearest",
    align_corners=None,
    recompute_scale_factor=None,
    antialias: bool = False,
) -> Tensor:
    """Resize a 4D (N, C, H, W) floating tensor with a bounded typed contract."""
    contract = _normalize_interpolate_2d_contract(
        input,
        size,
        scale_factor,
        mode,
        align_corners,
        recompute_scale_factor,
        antialias,
    )
    out = Tensor(
        _execute_interpolate_2d(contract, input.data),
        dtype=contract["output_dtype"],
    )

    def backward(g):
        if (
            tuple(g.data.shape) != contract["output_shape"]
            or g.dtype != contract["output_dtype"]
        ):
            raise ValueError(
                "interpolate: upstream gradient must match output shape and dtype"
            )
        return (_execute_interpolate_2d_vjp(contract, g.data),)

    return _build_ctx(out, (input,), backward)


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
nll = nll_loss
bce_with_logits = bce_with_logits_loss
binary_cross_entropy_with_logits = bce_with_logits_loss
kl_div = kl_div_loss
