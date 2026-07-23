"""browsergrad_jit._realize — NumPy realization of a UOp graph.

INTERNAL. Realization is the moment a lazy UOp graph turns into concrete
np.ndarray values. The realize() function below is the single entry point;
every realization trigger on TensorProxy (.numpy, .tolist, .item, __bool__,
__float__, __int__, __iter__, .backward, optimizer.step) goes through it.

Algorithm — one topological walk + one dispatch-table call per node:

  1. topo = toposort(root)
  2. for node in topo:
       value_table[node] = _DISPATCH[node.op](node, value_table, buffer_table)
  3. return value_table[root]

The value_table is a per-realization dict keyed by id(uop) (UOps are hashable
but we use id() because two structurally-equal UOps that happen to be
different objects should be cached separately — the cost of structural
re-hashing on a 10K-node graph isn't worth saving the slot).

Dispatch handlers are pure functions: they receive the node, the in-flight
value_table, and the BufferTable, and return an ndarray. They never mutate
the input arrays — gradient accumulation is the ONE legitimate path that
mutates a BufferTable entry, and it goes through STORE, not through a
handler's return value.
"""

from __future__ import annotations
import warnings
from typing import Any, Callable

import numpy as np

from ._ir import (
    UOp, ALL_OPS, toposort,
    OP_BUFFER, OP_LOAD, OP_STORE, OP_CONST, OP_RANDOM,
    OP_CAST, OP_ADD, OP_MUL, OP_DIV, OP_NEG,
    OP_EXP, OP_LOG, OP_ABS, OP_CLAMP, OP_COS, OP_FLIP, OP_CUMSUM, OP_CONCAT, OP_STACK, OP_NARROW, OP_TRIL, OP_TRIU, OP_PROD, OP_VAR, OP_REPEAT,
    OP_REPEAT_INTERLEAVE, OP_SIGN, OP_SIN, OP_CMP, OP_MATMUL,
    OP_CONV1D, OP_CONV1D_BACKWARD_INPUT, OP_CONV1D_BACKWARD_WEIGHT,
    OP_CONV1D_BACKWARD_BIAS, OP_CONV2D,
    OP_CONV2D_BACKWARD_INPUT, OP_CONV2D_BACKWARD_WEIGHT,
    OP_CONV2D_BACKWARD_BIAS, OP_CONV_TRANSPOSE2D,
    OP_CONV_TRANSPOSE2D_BACKWARD_INPUT,
    OP_CONV_TRANSPOSE2D_BACKWARD_WEIGHT,
    OP_CONV_TRANSPOSE2D_BACKWARD_BIAS, OP_CONV3D,
    OP_CONV3D_BACKWARD_INPUT, OP_CONV3D_BACKWARD_WEIGHT,
    OP_CONV3D_BACKWARD_BIAS, OP_LAYER_NORM,
    OP_LAYER_NORM_BACKWARD_INPUT, OP_LAYER_NORM_BACKWARD_WEIGHT,
    OP_LAYER_NORM_BACKWARD_BIAS, OP_REDUCE,
    OP_RESHAPE, OP_PERMUTE, OP_SLICE, OP_PAD, OP_SORT_INDICES, OP_SORT_VALUES,
    OP_TOPK_INDICES, OP_TOPK_VALUES, OP_SCATTER, OP_EINSUM, OP_L1_LOSS,
    OP_SMOOTH_L1_LOSS, OP_BINARY_CROSS_ENTROPY,
    OP_BINARY_CROSS_ENTROPY_WITH_LOGITS,
    OP_KL_DIV,
    OP_NLL_LOSS,
    OP_CROSS_ENTROPY,
    OP_DROPOUT,
    OP_BATCH_NORM_1D,
    OP_BATCH_NORM_1D_STATS_UPDATE,
    OP_INTERPOLATE_2D,
    OP_WHERE, OP_INDEX, OP_MASK, OP_CUSTOM,
    OP_FUSED_ELEMENTWISE, OP_FUSED_SOFTMAX,
    OP_SCATTER_ADD, OP_BROADCAST_TO, OP_EINSUM_VJP, OP_L1_LOSS_VJP,
    OP_SMOOTH_L1_LOSS_VJP, OP_BINARY_CROSS_ENTROPY_VJP,
    OP_BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP,
    OP_KL_DIV_VJP,
    OP_NLL_LOSS_VJP,
    OP_CROSS_ENTROPY_VJP,
    OP_DROPOUT_VJP,
    OP_BATCH_NORM_1D_VJP,
    OP_INTERPOLATE_2D_VJP,
    OP_ISNAN, OP_SGD_UPDATE, OP_ADAMW_UPDATE_M, OP_ADAMW_UPDATE_V,
    OP_ADAMW_UPDATE_PARAM, OP_ADAM_UPDATE_M, OP_ADAM_UPDATE_V,
    OP_ADAM_UPDATE_PARAM,
)
from ._buffer_table import BufferTable
from ._errors import RealizationError, ShapeError
from ._framework_contracts import (
    validate_broadcast_to_contract,
    validate_cat_contract,
    validate_stack_contract,
    validate_pad_contract,
    validate_sort_indices_contract,
    validate_sort_values_contract,
    validate_topk_indices_contract,
    validate_topk_values_contract,
    validate_scatter_contract,
    validate_einsum_contract,
    validate_einsum_vjp_contract,
    validate_l1_loss_contract,
    validate_l1_loss_vjp_contract,
    validate_smooth_l1_loss_contract,
    validate_smooth_l1_loss_vjp_contract,
    validate_binary_cross_entropy_contract,
    validate_binary_cross_entropy_vjp_contract,
    validate_binary_cross_entropy_with_logits_contract,
    validate_binary_cross_entropy_with_logits_vjp_contract,
    validate_kl_div_contract,
    validate_kl_div_vjp_contract,
    validate_nll_loss_contract,
    validate_nll_loss_vjp_contract,
    validate_cross_entropy_contract,
    validate_cross_entropy_vjp_contract,
    validate_dropout_contract,
    validate_dropout_vjp_contract,
    validate_batch_norm_1d_contract,
    validate_batch_norm_1d_stats_update_contract,
    validate_batch_norm_1d_vjp_contract,
    validate_interpolate_2d_contract,
    validate_interpolate_2d_vjp_contract,
    validate_clamp_contract,
    validate_cumsum_contract,
    validate_flip_contract,
    validate_tril_contract,
    validate_triu_contract,
    validate_gather_contract,
    validate_gather_scatter_add_contract,
    validate_narrow_contract,
    validate_prod_contract,
    validate_var_contract,
    validate_where_contract,
    validate_repeat_contract,
    validate_repeat_interleave_contract,
    validate_real_numeric_unary_contract,
    validate_typed_unary_contract,
    stable_sort_indices_array,
    partial_topk_indices_array,
    scatter_index_violation,
    execute_einsum_arrays,
    execute_einsum_vjp_array,
    execute_l1_loss_arrays,
    execute_l1_loss_vjp_array,
    execute_smooth_l1_loss_arrays,
    execute_smooth_l1_loss_vjp_array,
    execute_binary_cross_entropy_arrays,
    execute_binary_cross_entropy_vjp_array,
    execute_binary_cross_entropy_with_logits_arrays,
    execute_binary_cross_entropy_with_logits_vjp_array,
    execute_kl_div_arrays,
    execute_kl_div_vjp_array,
    execute_nll_loss_arrays,
    execute_nll_loss_vjp_array,
    execute_cross_entropy_arrays,
    execute_cross_entropy_vjp_array,
    execute_dropout_array,
    execute_dropout_vjp_array,
    infer_batch_norm_1d_contract,
    batch_norm_1d_batch_stats_array,
    execute_batch_norm_1d_array,
    execute_batch_norm_1d_vjp_array,
    BATCH_NORM_1D_STATE_EFFECT_KIND,
    execute_interpolate_2d_array,
    execute_interpolate_2d_vjp_array,
)


# A dispatch handler. Receives the node, the in-flight value table, and the
# BufferTable. Returns a concrete np.ndarray. Must not mutate inputs.
Handler = Callable[[UOp, dict, BufferTable], np.ndarray]


# ---------------------------------------------------------------------------
# Per-opcode handlers
# ---------------------------------------------------------------------------


def _h_buffer(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    arr = bt.get(node.arg)
    # Defense in depth: confirm shape/dtype agree with what the IR promised.
    if tuple(arr.shape) != tuple(node.shape):
        raise RealizationError(
            f"BUFFER {node.arg!r} on the table has shape {arr.shape} but the "
            f"IR declared {node.shape}"
        )
    return arr


def _h_load(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    # LOAD just unwraps the BUFFER it wraps. The BUFFER handler already
    # ran (topological order), so we have the value cached.
    return vt[id(node.inputs[0])]


def _h_const(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    value = node.arg["value"]
    return np.asarray(value, dtype=np.dtype(node.dtype))


def _h_random(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    arg = node.arg
    rng = np.random.default_rng(arg["seed_key"])
    dist = arg["dist"]
    if dist == "uniform":
        # Optional 'low'/'high' overrides default [0, 1).
        low = arg.get("low", 0.0)
        high = arg.get("high", 1.0)
        out = rng.uniform(low=low, high=high, size=node.shape)
    elif dist == "normal":
        mean = arg.get("mean", 0.0)
        std = arg.get("std", 1.0)
        out = rng.normal(loc=mean, scale=std, size=node.shape)
    else:
        raise RealizationError(f"RANDOM: unknown dist {dist!r}")
    return out.astype(np.dtype(node.dtype), copy=False)


def _h_cast(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    return x.astype(np.dtype(node.arg["dtype"]), copy=False)


def _h_add(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    a = vt[id(node.inputs[0])]
    b = vt[id(node.inputs[1])]
    return a + b


def _h_mul(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    a = vt[id(node.inputs[0])]
    b = vt[id(node.inputs[1])]
    return a * b


def _h_div(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    a = vt[id(node.inputs[0])]
    b = vt[id(node.inputs[1])]
    # True division. For integer dtypes NumPy would normally float-promote;
    # the IR promises the output dtype matches the broadcast of the inputs,
    # so we honor that here.
    out = a / b
    return out.astype(np.dtype(node.dtype), copy=False)


def _h_neg(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    return -vt[id(node.inputs[0])]


def _h_exp(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    return np.exp(vt[id(node.inputs[0])])


def _h_log(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    return np.log(vt[id(node.inputs[0])])


def _h_abs(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    validate_real_numeric_unary_contract(node)
    return np.abs(vt[id(node.inputs[0])]).astype(np.dtype(node.dtype), copy=False)


def _h_clamp(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    minimum, maximum = validate_clamp_contract(node)
    return np.clip(
        vt[id(node.inputs[0])],
        minimum,
        maximum,
    ).astype(np.dtype(node.dtype), copy=False)


def _h_cos(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    validate_typed_unary_contract(node)
    return np.cos(vt[id(node.inputs[0])]).astype(np.dtype(node.dtype), copy=False)


def _h_flip(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    axis = validate_flip_contract(node)
    return np.flip(vt[id(node.inputs[0])], axis=axis).copy()


def _h_cumsum(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    axis, reverse = validate_cumsum_contract(node)
    source = vt[id(node.inputs[0])]
    if reverse:
        source = np.flip(source, axis=axis)
    scanned = np.cumsum(
        source,
        axis=axis,
        dtype=np.dtype(node.dtype),
    )
    if reverse:
        scanned = np.flip(scanned, axis=axis)
    return np.array(scanned, dtype=np.dtype(node.dtype), copy=True)


def _h_concat(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    axis, _, legacy_empty = validate_cat_contract(node)
    arrays = []
    for source_node, empty in zip(node.inputs, legacy_empty):
        if empty and len(node.shape) != 1:
            continue
        arrays.append(np.asarray(vt[id(source_node)], dtype=np.dtype(node.dtype)))
    concatenated = np.concatenate(arrays, axis=axis)
    return np.array(concatenated, dtype=np.dtype(node.dtype), copy=True)


def _h_stack(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    axis = validate_stack_contract(node)
    arrays = [
        np.asarray(vt[id(source)], dtype=np.dtype(node.dtype))
        for source in node.inputs
    ]
    stacked = np.stack(arrays, axis=axis)
    return np.array(stacked, dtype=np.dtype(node.dtype), copy=True)


def _h_narrow(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    axis, start, length = validate_narrow_contract(node)
    slices = [slice(None)] * len(node.shape)
    slices[axis] = slice(start, start + length)
    narrowed = vt[id(node.inputs[0])][tuple(slices)]
    return np.array(narrowed, dtype=np.dtype(node.dtype), copy=True)


def _h_tril(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    diagonal = validate_tril_contract(node)
    selected = np.tril(vt[id(node.inputs[0])], k=diagonal)
    return np.array(selected, dtype=np.dtype(node.dtype), copy=True)


def _h_triu(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    diagonal = validate_triu_contract(node)
    selected = np.triu(vt[id(node.inputs[0])], k=diagonal)
    return np.array(selected, dtype=np.dtype(node.dtype), copy=True)


def _h_repeat(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    repeats, _ = validate_repeat_contract(node)
    tiled = np.tile(vt[id(node.inputs[0])], repeats)
    return np.array(tiled, dtype=np.dtype(node.dtype), copy=True)


def _h_prod(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    axes, keepdims, _ = validate_prod_contract(node)
    product = np.prod(
        vt[id(node.inputs[0])],
        axis=axes,
        keepdims=keepdims,
        dtype=np.dtype(node.dtype),
    )
    return np.array(product, dtype=np.dtype(node.dtype), copy=True).reshape(node.shape)


def _h_var(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    axes, correction, keepdims, _, _ = validate_var_contract(node)
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        variance = np.var(
            vt[id(node.inputs[0])],
            axis=axes,
            keepdims=keepdims,
            ddof=correction,
            dtype=np.dtype(node.dtype),
        )
    return np.array(variance, dtype=np.dtype(node.dtype), copy=True).reshape(node.shape)


def _h_repeat_interleave(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    repeats, axis = validate_repeat_interleave_contract(node)
    repeated = np.repeat(vt[id(node.inputs[0])], repeats, axis=axis)
    return np.array(repeated, dtype=np.dtype(node.dtype), copy=True)


def _h_sign(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    validate_real_numeric_unary_contract(node)
    return np.sign(vt[id(node.inputs[0])]).astype(np.dtype(node.dtype), copy=False)


def _h_sin(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    validate_typed_unary_contract(node)
    return np.sin(vt[id(node.inputs[0])]).astype(np.dtype(node.dtype), copy=False)


_CMP_OPS = {
    "eq": np.equal,
    "ne": np.not_equal,
    "lt": np.less,
    "le": np.less_equal,
    "gt": np.greater,
    "ge": np.greater_equal,
}


def _h_cmp(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    a = vt[id(node.inputs[0])]
    b = vt[id(node.inputs[1])]
    op = node.arg["op"]
    if op not in _CMP_OPS:
        raise RealizationError(f"CMP: unknown comparison op {op!r}")
    return _CMP_OPS[op](a, b)


def _h_matmul(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    """Matmul with the tensor-core fp32-accumulator semantics for fp16
    inputs (PRD-010). Real GPU hardware does fp16 × fp16 → fp32 accumulate
    → fp16 store; NumPy's `a @ b` with f16 inputs would accumulate in
    f16, which underflows on long reductions and diverges from any
    eventual WGSL kernel. Match the WGSL/tensor-core path here so the
    educational story holds across backends.
    """
    a = vt[id(node.inputs[0])]
    b = vt[id(node.inputs[1])]
    if a.dtype == np.float16 or b.dtype == np.float16:
        # Upcast → accumulate → downcast. Mimics WGSL's
        # `var<workgroup> acc : f32` pattern.
        out = a.astype(np.float32) @ b.astype(np.float32)
        return out.astype(np.dtype(node.dtype), copy=False)
    return a @ b


def _h_conv1d(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    w_arr = vt[id(node.inputs[1])]
    bias_arr = vt[id(node.inputs[2])] if bool(node.arg.get("has_bias", False)) else None
    arg = node.arg
    n = int(arg["n"])
    c_in = int(arg["c_in"])
    l_in = int(arg["l_in"])
    c_out = int(arg["c_out"])
    k = int(arg["k"])
    stride = int(arg["stride"])
    padding = int(arg["padding"])
    dilation = int(arg["dilation"])
    groups = int(arg["groups"])
    l_out = int(arg["l_out"])
    c_per_group = c_in // groups
    out_per_group = c_out // groups
    eff_k = dilation * (k - 1) + 1
    if padding > 0:
        x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode="constant")
    else:
        x_pad = x
    out = np.zeros((n, c_out, l_out), dtype=np.float32)
    for nn in range(n):
        for g in range(groups):
            c0 = g * c_per_group
            o0 = g * out_per_group
            for co in range(out_per_group):
                for i in range(l_out):
                    l0 = i * stride
                    out[nn, o0 + co, i] = (
                        x_pad[nn, c0:c0+c_per_group, l0:l0+eff_k:dilation]
                        * w_arr[o0 + co]
                    ).sum()
    if bias_arr is not None:
        out += bias_arr.reshape(1, c_out, 1)
    return out.astype(np.dtype(node.dtype), copy=False)


def _h_conv1d_backward_input(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    dy = vt[id(node.inputs[0])]
    weight = vt[id(node.inputs[1])]
    arg = node.arg
    n = int(arg["n"])
    c_in = int(arg["c_in"])
    l_in = int(arg["l_in"])
    c_out = int(arg["c_out"])
    k = int(arg["k"])
    stride = int(arg["stride"])
    padding = int(arg["padding"])
    dilation = int(arg["dilation"])
    groups = int(arg["groups"])
    l_out = int(arg["l_out"])
    c_per_group = c_in // groups
    out_per_group = c_out // groups
    grad_x_pad = np.zeros((n, c_in, l_in + 2 * padding), dtype=np.float32)
    for nn in range(n):
        for g in range(groups):
            c0 = g * c_per_group
            o0 = g * out_per_group
            for co in range(out_per_group):
                out_ch = o0 + co
                for i in range(l_out):
                    grad_val = dy[nn, out_ch, i]
                    base = i * stride
                    for ci in range(c_per_group):
                        in_ch = c0 + ci
                        for r in range(k):
                            li = base + r * dilation
                            grad_x_pad[nn, in_ch, li] += grad_val * weight[out_ch, ci, r]
    grad_x = (
        grad_x_pad[:, :, padding:padding+l_in].copy()
        if padding > 0 else grad_x_pad
    )
    return grad_x.astype(np.dtype(node.dtype), copy=False)


def _h_conv1d_backward_weight(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    dy = vt[id(node.inputs[0])]
    x = vt[id(node.inputs[1])]
    arg = node.arg
    n = int(arg["n"])
    c_in = int(arg["c_in"])
    c_out = int(arg["c_out"])
    k = int(arg["k"])
    stride = int(arg["stride"])
    padding = int(arg["padding"])
    dilation = int(arg["dilation"])
    groups = int(arg["groups"])
    l_out = int(arg["l_out"])
    c_per_group = c_in // groups
    out_per_group = c_out // groups
    if padding > 0:
        x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode="constant")
    else:
        x_pad = x
    grad_w = np.zeros((c_out, c_per_group, k), dtype=np.float32)
    for nn in range(n):
        for g in range(groups):
            c0 = g * c_per_group
            o0 = g * out_per_group
            for co in range(out_per_group):
                out_ch = o0 + co
                for ci in range(c_per_group):
                    in_ch = c0 + ci
                    for r in range(k):
                        acc = 0.0
                        for i in range(l_out):
                            li = i * stride + r * dilation
                            acc += dy[nn, out_ch, i] * x_pad[nn, in_ch, li]
                        grad_w[out_ch, ci, r] += acc
    return grad_w.astype(np.dtype(node.dtype), copy=False)


def _h_conv1d_backward_bias(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    dy = vt[id(node.inputs[0])]
    return dy.sum(axis=(0, 2)).astype(np.dtype(node.dtype), copy=False)


def _h_conv2d(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    w_arr = vt[id(node.inputs[1])]
    bias_arr = vt[id(node.inputs[2])] if bool(node.arg.get("has_bias", False)) else None
    arg = node.arg
    n = int(arg["n"])
    c_in = int(arg["c_in"])
    h = int(arg["h"])
    w = int(arg["w"])
    c_out = int(arg["c_out"])
    kh = int(arg["kh"])
    kw = int(arg["kw"])
    stride_h = int(arg["stride_h"])
    stride_w = int(arg["stride_w"])
    pad_h = int(arg["pad_h"])
    pad_w = int(arg["pad_w"])
    dilation_h = int(arg["dilation_h"])
    dilation_w = int(arg["dilation_w"])
    groups = int(arg["groups"])
    out_h = int(arg["out_h"])
    out_w = int(arg["out_w"])
    c_per_group = c_in // groups
    out_per_group = c_out // groups
    eff_h = dilation_h * (kh - 1) + 1
    eff_w = dilation_w * (kw - 1) + 1
    if pad_h > 0 or pad_w > 0:
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    else:
        x_pad = x
    out = np.zeros((n, c_out, out_h, out_w), dtype=np.float32)
    for nn in range(n):
        for g in range(groups):
            c0 = g * c_per_group
            o0 = g * out_per_group
            for co in range(out_per_group):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h0 = oh * stride_h
                        w0 = ow * stride_w
                        out[nn, o0 + co, oh, ow] = (
                            x_pad[
                                nn,
                                c0:c0+c_per_group,
                                h0:h0+eff_h:dilation_h,
                                w0:w0+eff_w:dilation_w,
                            ]
                            * w_arr[o0 + co]
                        ).sum()
    if bias_arr is not None:
        out += bias_arr.reshape(1, c_out, 1, 1)
    return out.astype(np.dtype(node.dtype), copy=False)


def _h_conv2d_backward_input(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    dy = vt[id(node.inputs[0])]
    weight = vt[id(node.inputs[1])]
    arg = node.arg
    n = int(arg["n"])
    c_in = int(arg["c_in"])
    h = int(arg["h"])
    w = int(arg["w"])
    c_out = int(arg["c_out"])
    kh = int(arg["kh"])
    kw = int(arg["kw"])
    stride_h = int(arg["stride_h"])
    stride_w = int(arg["stride_w"])
    pad_h = int(arg["pad_h"])
    pad_w = int(arg["pad_w"])
    dilation_h = int(arg["dilation_h"])
    dilation_w = int(arg["dilation_w"])
    groups = int(arg["groups"])
    out_h = int(arg["out_h"])
    out_w = int(arg["out_w"])
    c_per_group = c_in // groups
    out_per_group = c_out // groups
    grad_x_pad = np.zeros(
        (n, c_in, h + 2 * pad_h, w + 2 * pad_w),
        dtype=np.float32,
    )
    for nn in range(n):
        for g in range(groups):
            c0 = g * c_per_group
            o0 = g * out_per_group
            for co in range(out_per_group):
                out_ch = o0 + co
                for oh in range(out_h):
                    for ow in range(out_w):
                        grad_val = dy[nn, out_ch, oh, ow]
                        h_base = oh * stride_h
                        w_base = ow * stride_w
                        for ci in range(c_per_group):
                            in_ch = c0 + ci
                            for r in range(kh):
                                ih = h_base + r * dilation_h
                                for s in range(kw):
                                    iw = w_base + s * dilation_w
                                    grad_x_pad[nn, in_ch, ih, iw] += (
                                        grad_val * weight[out_ch, ci, r, s]
                                    )
    grad_x = (
        grad_x_pad[:, :, pad_h:pad_h+h, pad_w:pad_w+w].copy()
        if pad_h > 0 or pad_w > 0 else grad_x_pad
    )
    return grad_x.astype(np.dtype(node.dtype), copy=False)


def _h_conv2d_backward_weight(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    dy = vt[id(node.inputs[0])]
    x = vt[id(node.inputs[1])]
    arg = node.arg
    n = int(arg["n"])
    c_in = int(arg["c_in"])
    c_out = int(arg["c_out"])
    kh = int(arg["kh"])
    kw = int(arg["kw"])
    stride_h = int(arg["stride_h"])
    stride_w = int(arg["stride_w"])
    pad_h = int(arg["pad_h"])
    pad_w = int(arg["pad_w"])
    dilation_h = int(arg["dilation_h"])
    dilation_w = int(arg["dilation_w"])
    groups = int(arg["groups"])
    out_h = int(arg["out_h"])
    out_w = int(arg["out_w"])
    c_per_group = c_in // groups
    out_per_group = c_out // groups
    if pad_h > 0 or pad_w > 0:
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    else:
        x_pad = x
    grad_w = np.zeros((c_out, c_per_group, kh, kw), dtype=np.float32)
    for nn in range(n):
        for g in range(groups):
            c0 = g * c_per_group
            o0 = g * out_per_group
            for co in range(out_per_group):
                out_ch = o0 + co
                for ci in range(c_per_group):
                    in_ch = c0 + ci
                    for r in range(kh):
                        for s in range(kw):
                            acc = 0.0
                            for oh in range(out_h):
                                ih = oh * stride_h + r * dilation_h
                                for ow in range(out_w):
                                    iw = ow * stride_w + s * dilation_w
                                    acc += dy[nn, out_ch, oh, ow] * x_pad[nn, in_ch, ih, iw]
                            grad_w[out_ch, ci, r, s] += acc
    return grad_w.astype(np.dtype(node.dtype), copy=False)


def _h_conv2d_backward_bias(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    dy = vt[id(node.inputs[0])]
    return dy.sum(axis=(0, 2, 3)).astype(np.dtype(node.dtype), copy=False)


def _conv_transpose2d_arg(arg: dict) -> tuple[int, ...]:
    return (
        int(arg["n"]),
        int(arg["c_in"]),
        int(arg["h"]),
        int(arg["w"]),
        int(arg["c_out"]),
        int(arg["c_out_per_group"]),
        int(arg["kh"]),
        int(arg["kw"]),
        int(arg["stride_h"]),
        int(arg["stride_w"]),
        int(arg["pad_h"]),
        int(arg["pad_w"]),
        int(arg["output_pad_h"]),
        int(arg["output_pad_w"]),
        int(arg["dilation_h"]),
        int(arg["dilation_w"]),
        int(arg["groups"]),
        int(arg["out_h"]),
        int(arg["out_w"]),
    )


def _h_conv_transpose2d(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    weight = vt[id(node.inputs[1])]
    bias = vt[id(node.inputs[2])] if len(node.inputs) > 2 else None
    (
        n, c_in, h, w, c_out, c_out_per_group, kh, kw,
        stride_h, stride_w, pad_h, pad_w, _oph, _opw,
        dilation_h, dilation_w, groups, out_h, out_w,
    ) = _conv_transpose2d_arg(node.arg)
    in_per_group = c_in // groups
    out = np.zeros((n, c_out, out_h, out_w), dtype=np.float32)
    for nn in range(n):
        for ci in range(c_in):
            group = ci // in_per_group
            co0 = group * c_out_per_group
            for ih in range(h):
                for iw in range(w):
                    value = x[nn, ci, ih, iw]
                    if value == 0:
                        continue
                    for r in range(kh):
                        oh = ih * stride_h - pad_h + r * dilation_h
                        if oh < 0 or oh >= out_h:
                            continue
                        for c in range(kw):
                            ow = iw * stride_w - pad_w + c * dilation_w
                            if 0 <= ow < out_w:
                                out[nn, co0:co0+c_out_per_group, oh, ow] += (
                                    value * weight[ci, :, r, c]
                                )
    if bias is not None:
        out += bias.reshape(1, c_out, 1, 1)
    return out.astype(np.dtype(node.dtype), copy=False)


def _h_conv_transpose2d_backward_input(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    dy = vt[id(node.inputs[0])]
    weight = vt[id(node.inputs[1])]
    (
        n, c_in, h, w, _c_out, c_out_per_group, kh, kw,
        stride_h, stride_w, pad_h, pad_w, _oph, _opw,
        dilation_h, dilation_w, groups, out_h, out_w,
    ) = _conv_transpose2d_arg(node.arg)
    in_per_group = c_in // groups
    grad_x = np.zeros((n, c_in, h, w), dtype=np.float32)
    for nn in range(n):
        for ci in range(c_in):
            group = ci // in_per_group
            co0 = group * c_out_per_group
            for ih in range(h):
                for iw in range(w):
                    acc = 0.0
                    for r in range(kh):
                        oh = ih * stride_h - pad_h + r * dilation_h
                        if oh < 0 or oh >= out_h:
                            continue
                        for c in range(kw):
                            ow = iw * stride_w - pad_w + c * dilation_w
                            if 0 <= ow < out_w:
                                acc += (
                                    dy[nn, co0:co0+c_out_per_group, oh, ow]
                                    * weight[ci, :, r, c]
                                ).sum()
                    grad_x[nn, ci, ih, iw] = acc
    return grad_x.astype(np.dtype(node.dtype), copy=False)


def _h_conv_transpose2d_backward_weight(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    dy = vt[id(node.inputs[0])]
    x = vt[id(node.inputs[1])]
    (
        n, c_in, h, w, _c_out, c_out_per_group, kh, kw,
        stride_h, stride_w, pad_h, pad_w, _oph, _opw,
        dilation_h, dilation_w, groups, out_h, out_w,
    ) = _conv_transpose2d_arg(node.arg)
    in_per_group = c_in // groups
    grad_w = np.zeros((c_in, c_out_per_group, kh, kw), dtype=np.float32)
    for nn in range(n):
        for ci in range(c_in):
            group = ci // in_per_group
            co0 = group * c_out_per_group
            for ih in range(h):
                for iw in range(w):
                    x_val = x[nn, ci, ih, iw]
                    for r in range(kh):
                        oh = ih * stride_h - pad_h + r * dilation_h
                        if oh < 0 or oh >= out_h:
                            continue
                        for c in range(kw):
                            ow = iw * stride_w - pad_w + c * dilation_w
                            if 0 <= ow < out_w:
                                grad_w[ci, :, r, c] += (
                                    dy[nn, co0:co0+c_out_per_group, oh, ow]
                                    * x_val
                                )
    return grad_w.astype(np.dtype(node.dtype), copy=False)


def _h_conv_transpose2d_backward_bias(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    dy = vt[id(node.inputs[0])]
    return dy.sum(axis=(0, 2, 3)).astype(np.dtype(node.dtype), copy=False)


def _h_conv3d(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    w_arr = vt[id(node.inputs[1])]
    bias_arr = vt[id(node.inputs[2])] if bool(node.arg.get("has_bias", False)) else None
    arg = node.arg
    n = int(arg["n"])
    c_in = int(arg["c_in"])
    d = int(arg["d"])
    h = int(arg["h"])
    w = int(arg["w"])
    c_out = int(arg["c_out"])
    kd = int(arg["kd"])
    kh = int(arg["kh"])
    kw = int(arg["kw"])
    stride_d = int(arg["stride_d"])
    stride_h = int(arg["stride_h"])
    stride_w = int(arg["stride_w"])
    pad_d = int(arg["pad_d"])
    pad_h = int(arg["pad_h"])
    pad_w = int(arg["pad_w"])
    dilation_d = int(arg["dilation_d"])
    dilation_h = int(arg["dilation_h"])
    dilation_w = int(arg["dilation_w"])
    groups = int(arg["groups"])
    out_d = int(arg["out_d"])
    out_h = int(arg["out_h"])
    out_w = int(arg["out_w"])
    c_per_group = c_in // groups
    out_per_group = c_out // groups
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        x_pad = np.pad(
            x,
            ((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
        )
    else:
        x_pad = x
    out = np.zeros((n, c_out, out_d, out_h, out_w), dtype=np.float32)
    for nn in range(n):
        for g in range(groups):
            c0 = g * c_per_group
            o0 = g * out_per_group
            for co in range(out_per_group):
                out_ch = o0 + co
                for od in range(out_d):
                    d0 = od * stride_d
                    for oh in range(out_h):
                        h0 = oh * stride_h
                        for ow in range(out_w):
                            w0 = ow * stride_w
                            out[nn, out_ch, od, oh, ow] = (
                                x_pad[
                                    nn,
                                    c0:c0+c_per_group,
                                    d0:d0+dilation_d*(kd-1)+1:dilation_d,
                                    h0:h0+dilation_h*(kh-1)+1:dilation_h,
                                    w0:w0+dilation_w*(kw-1)+1:dilation_w,
                                ]
                                * w_arr[out_ch]
                            ).sum()
    if bias_arr is not None:
        out += bias_arr.reshape(1, c_out, 1, 1, 1)
    return out.astype(np.dtype(node.dtype), copy=False)


def _h_conv3d_backward_input(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    dy = vt[id(node.inputs[0])]
    weight = vt[id(node.inputs[1])]
    arg = node.arg
    n = int(arg["n"])
    c_in = int(arg["c_in"])
    d = int(arg["d"])
    h = int(arg["h"])
    w = int(arg["w"])
    c_out = int(arg["c_out"])
    kd = int(arg["kd"])
    kh = int(arg["kh"])
    kw = int(arg["kw"])
    stride_d = int(arg["stride_d"])
    stride_h = int(arg["stride_h"])
    stride_w = int(arg["stride_w"])
    pad_d = int(arg["pad_d"])
    pad_h = int(arg["pad_h"])
    pad_w = int(arg["pad_w"])
    dilation_d = int(arg["dilation_d"])
    dilation_h = int(arg["dilation_h"])
    dilation_w = int(arg["dilation_w"])
    groups = int(arg["groups"])
    out_d = int(arg["out_d"])
    out_h = int(arg["out_h"])
    out_w = int(arg["out_w"])
    c_per_group = c_in // groups
    out_per_group = c_out // groups
    grad_x_pad = np.zeros(
        (n, c_in, d + 2 * pad_d, h + 2 * pad_h, w + 2 * pad_w),
        dtype=np.float32,
    )
    for nn in range(n):
        for g in range(groups):
            c0 = g * c_per_group
            o0 = g * out_per_group
            for co in range(out_per_group):
                out_ch = o0 + co
                for od in range(out_d):
                    d_base = od * stride_d
                    for oh in range(out_h):
                        h_base = oh * stride_h
                        for ow in range(out_w):
                            grad_val = dy[nn, out_ch, od, oh, ow]
                            w_base = ow * stride_w
                            for ci in range(c_per_group):
                                in_ch = c0 + ci
                                for rd in range(kd):
                                    di = d_base + rd * dilation_d
                                    for rh in range(kh):
                                        hi = h_base + rh * dilation_h
                                        for rw in range(kw):
                                            wi = w_base + rw * dilation_w
                                            grad_x_pad[nn, in_ch, di, hi, wi] += (
                                                grad_val * weight[out_ch, ci, rd, rh, rw]
                                            )
    grad_x = (
        grad_x_pad[:, :, pad_d:pad_d+d, pad_h:pad_h+h, pad_w:pad_w+w].copy()
        if pad_d > 0 or pad_h > 0 or pad_w > 0 else grad_x_pad
    )
    return grad_x.astype(np.dtype(node.dtype), copy=False)


def _h_conv3d_backward_weight(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    dy = vt[id(node.inputs[0])]
    x = vt[id(node.inputs[1])]
    arg = node.arg
    n = int(arg["n"])
    c_in = int(arg["c_in"])
    c_out = int(arg["c_out"])
    kd = int(arg["kd"])
    kh = int(arg["kh"])
    kw = int(arg["kw"])
    stride_d = int(arg["stride_d"])
    stride_h = int(arg["stride_h"])
    stride_w = int(arg["stride_w"])
    pad_d = int(arg["pad_d"])
    pad_h = int(arg["pad_h"])
    pad_w = int(arg["pad_w"])
    dilation_d = int(arg["dilation_d"])
    dilation_h = int(arg["dilation_h"])
    dilation_w = int(arg["dilation_w"])
    groups = int(arg["groups"])
    out_d = int(arg["out_d"])
    out_h = int(arg["out_h"])
    out_w = int(arg["out_w"])
    c_per_group = c_in // groups
    out_per_group = c_out // groups
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        x_pad = np.pad(
            x,
            ((0, 0), (0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
        )
    else:
        x_pad = x
    grad_w = np.zeros((c_out, c_per_group, kd, kh, kw), dtype=np.float32)
    for nn in range(n):
        for g in range(groups):
            c0 = g * c_per_group
            o0 = g * out_per_group
            for co in range(out_per_group):
                out_ch = o0 + co
                for ci in range(c_per_group):
                    in_ch = c0 + ci
                    for rd in range(kd):
                        for rh in range(kh):
                            for rw in range(kw):
                                acc = 0.0
                                for od in range(out_d):
                                    di = od * stride_d + rd * dilation_d
                                    for oh in range(out_h):
                                        hi = oh * stride_h + rh * dilation_h
                                        for ow in range(out_w):
                                            wi = ow * stride_w + rw * dilation_w
                                            acc += (
                                                dy[nn, out_ch, od, oh, ow]
                                                * x_pad[nn, in_ch, di, hi, wi]
                                            )
                                grad_w[out_ch, ci, rd, rh, rw] += acc
    return grad_w.astype(np.dtype(node.dtype), copy=False)


def _h_conv3d_backward_bias(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    dy = vt[id(node.inputs[0])]
    return dy.sum(axis=(0, 2, 3, 4)).astype(np.dtype(node.dtype), copy=False)


def _layer_norm_stats(
    x: np.ndarray,
    weight: np.ndarray,
    arg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = int(arg["rows"])
    cols = int(arg["cols"])
    eps = float(arg.get("eps", 1e-5))
    x2 = x.reshape(rows, cols).astype(np.float32, copy=False)
    w = weight.reshape(cols).astype(np.float32, copy=False)
    mean = x2.mean(axis=1, keepdims=True)
    centered = x2 - mean
    var = (centered * centered).mean(axis=1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = centered * inv_std
    return x2, w, inv_std, x_hat


def _h_layer_norm(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    weight = vt[id(node.inputs[1])]
    bias = vt[id(node.inputs[2])]
    cols = int(node.arg["cols"])
    _, w, _, x_hat = _layer_norm_stats(x, weight, node.arg)
    b = bias.reshape(cols).astype(np.float32, copy=False)
    out = x_hat * w.reshape(1, cols) + b.reshape(1, cols)
    return out.reshape(x.shape).astype(np.dtype(node.dtype), copy=False)


def _h_layer_norm_backward_input(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    dy = vt[id(node.inputs[0])]
    x = vt[id(node.inputs[1])]
    weight = vt[id(node.inputs[2])]
    rows = int(node.arg["rows"])
    cols = int(node.arg["cols"])
    _, w, inv_std, x_hat = _layer_norm_stats(x, weight, node.arg)
    dy2 = dy.reshape(rows, cols).astype(np.float32, copy=False)
    grad_x_hat = dy2 * w.reshape(1, cols)
    sum_g = grad_x_hat.sum(axis=1, keepdims=True)
    sum_g_xhat = (grad_x_hat * x_hat).sum(axis=1, keepdims=True)
    grad_x = (inv_std / float(cols)) * (
        float(cols) * grad_x_hat - sum_g - x_hat * sum_g_xhat
    )
    return grad_x.reshape(x.shape).astype(np.dtype(node.dtype), copy=False)


def _h_layer_norm_backward_weight(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    dy = vt[id(node.inputs[0])]
    x = vt[id(node.inputs[1])]
    rows = int(node.arg["rows"])
    cols = int(node.arg["cols"])
    dummy_weight = np.ones((cols,), dtype=np.float32)
    _, _, _, x_hat = _layer_norm_stats(x, dummy_weight, node.arg)
    dy2 = dy.reshape(rows, cols).astype(np.float32, copy=False)
    grad_w = (dy2 * x_hat).sum(axis=0)
    return grad_w.reshape(node.shape).astype(np.dtype(node.dtype), copy=False)


def _h_layer_norm_backward_bias(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    dy = vt[id(node.inputs[0])]
    rows = int(node.arg["rows"])
    cols = int(node.arg["cols"])
    grad_b = dy.reshape(rows, cols).sum(axis=0)
    return grad_b.reshape(node.shape).astype(np.dtype(node.dtype), copy=False)


def _h_isnan(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    """Element-wise NaN check. Used by GradScaler's overflow detection
    (PRD-010); the boolean result feeds REDUCE(any) to produce the
    "any gradient overflowed?" scalar."""
    x = vt[id(node.inputs[0])]
    return np.isnan(x)


_REDUCE_OPS = {
    "sum": np.sum,
    "max": np.max,
    "min": np.min,
    "mean": np.mean,
    "argmax": np.argmax,
    "argmin": np.argmin,
    "any": np.any,
    "all": np.all,
}


def _h_reduce(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    op = node.arg["op"]
    axis = node.arg.get("axis", None)
    keepdims = node.arg.get("keepdims", False)
    fn = _REDUCE_OPS.get(op)
    if fn is None:
        raise RealizationError(f"REDUCE: unknown op {op!r}")
    if op in ("argmax", "argmin"):
        # NumPy's argmax/argmin take axis but not keepdims.
        out = fn(x, axis=axis)
        if keepdims and axis is not None:
            out = np.expand_dims(out, axis=axis)
    elif op in ("any", "all"):
        out = fn(x, axis=axis, keepdims=keepdims)
    else:
        out = fn(x, axis=axis, keepdims=keepdims)
    return np.asarray(out, dtype=np.dtype(node.dtype))


def _h_reshape(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    return x.reshape(node.arg["new_shape"])


def _h_permute(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    return np.transpose(x, axes=node.arg["axes"])


def _h_slice(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    slices = node.arg["slices"]
    # Cast to numpy slicing tuple.
    return x[tuple(slices)]


def _h_pad(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    pad_width, value = validate_pad_contract(node)
    padded = np.pad(
        x,
        pad_width=pad_width,
        mode="constant",
        constant_values=value,
    )
    return np.array(padded, dtype=np.dtype(node.dtype), copy=True)


def _h_sort_indices(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    axis, descending, _ = validate_sort_indices_contract(node)
    source = vt[id(node.inputs[0])]
    return stable_sort_indices_array(source, axis, descending)


def _h_sort_values(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    axis, _, _ = validate_sort_values_contract(node)
    source = vt[id(node.inputs[0])]
    indices = vt[id(node.inputs[1])]
    if source.ndim == 0:
        return np.array(source, dtype=np.dtype(node.dtype), copy=True)
    values = np.take_along_axis(source, indices, axis=axis)
    return np.array(values, dtype=np.dtype(node.dtype), copy=True)


def _h_topk_indices(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    axis, k, largest, sorted_output = validate_topk_indices_contract(node)
    source = vt[id(node.inputs[0])]
    return partial_topk_indices_array(
        source,
        axis,
        k,
        largest,
        sorted_output,
    )


def _h_topk_values(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    axis, _, _, _ = validate_topk_values_contract(node)
    source = vt[id(node.inputs[0])]
    indices = vt[id(node.inputs[1])]
    values = np.take_along_axis(source, indices, axis=axis)
    return np.array(values, dtype=np.dtype(node.dtype), copy=True)


def _h_scatter(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    axis = validate_scatter_contract(node)
    target = vt[id(node.inputs[0])]
    index = vt[id(node.inputs[1])]
    source = vt[id(node.inputs[2])]
    violation = scatter_index_violation(index, axis, target.shape[axis])
    if violation is not None:
        raise RealizationError(violation)
    output = np.array(target, dtype=np.dtype(node.dtype), copy=True)
    np.put_along_axis(output, index, source, axis=axis)
    return output


def _h_einsum(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    contract = validate_einsum_contract(node)
    arrays = tuple(vt[id(source)] for source in node.inputs)
    return execute_einsum_arrays(contract, arrays)


def _h_einsum_vjp(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    contract, operand = validate_einsum_vjp_contract(node)
    dy = vt[id(node.inputs[0])]
    arrays = tuple(vt[id(source)] for source in node.inputs[1:])
    return execute_einsum_vjp_array(contract, operand, dy, arrays)


def _h_l1_loss(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    contract = validate_l1_loss_contract(node)
    arrays = tuple(vt[id(source)] for source in node.inputs)
    return execute_l1_loss_arrays(contract, arrays)


def _h_l1_loss_vjp(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    contract, operand = validate_l1_loss_vjp_contract(node)
    dy = vt[id(node.inputs[0])]
    arrays = tuple(vt[id(source)] for source in node.inputs[1:])
    return execute_l1_loss_vjp_array(contract, operand, dy, arrays)


def _h_smooth_l1_loss(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    contract = validate_smooth_l1_loss_contract(node)
    arrays = tuple(vt[id(source)] for source in node.inputs)
    return execute_smooth_l1_loss_arrays(contract, arrays)


def _h_smooth_l1_loss_vjp(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    contract, operand = validate_smooth_l1_loss_vjp_contract(node)
    dy = vt[id(node.inputs[0])]
    arrays = tuple(vt[id(source)] for source in node.inputs[1:])
    return execute_smooth_l1_loss_vjp_array(contract, operand, dy, arrays)


def _h_binary_cross_entropy(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    contract = validate_binary_cross_entropy_contract(node)
    arrays = tuple(vt[id(source)] for source in node.inputs)
    return execute_binary_cross_entropy_arrays(contract, arrays)


def _h_binary_cross_entropy_vjp(
    node: UOp,
    vt: dict,
    bt: BufferTable,
) -> np.ndarray:
    contract, operand = validate_binary_cross_entropy_vjp_contract(node)
    dy = vt[id(node.inputs[0])]
    arrays = tuple(vt[id(source)] for source in node.inputs[1:])
    return execute_binary_cross_entropy_vjp_array(contract, operand, dy, arrays)


def _h_binary_cross_entropy_with_logits(
    node: UOp,
    vt: dict,
    bt: BufferTable,
) -> np.ndarray:
    contract = validate_binary_cross_entropy_with_logits_contract(node)
    arrays = tuple(vt[id(source)] for source in node.inputs)
    return execute_binary_cross_entropy_with_logits_arrays(contract, arrays)


def _h_binary_cross_entropy_with_logits_vjp(
    node: UOp,
    vt: dict,
    bt: BufferTable,
) -> np.ndarray:
    contract, operand = validate_binary_cross_entropy_with_logits_vjp_contract(node)
    dy = vt[id(node.inputs[0])]
    arrays = tuple(vt[id(source)] for source in node.inputs[1:])
    return execute_binary_cross_entropy_with_logits_vjp_array(
        contract, operand, dy, arrays
    )


def _h_kl_div(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    contract = validate_kl_div_contract(node)
    arrays = tuple(vt[id(source)] for source in node.inputs)
    return execute_kl_div_arrays(contract, arrays)


def _h_kl_div_vjp(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    contract, operand = validate_kl_div_vjp_contract(node)
    dy = vt[id(node.inputs[0])]
    arrays = tuple(vt[id(source)] for source in node.inputs[1:])
    return execute_kl_div_vjp_array(contract, operand, dy, arrays)


def _h_nll_loss(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    contract = validate_nll_loss_contract(node)
    arrays = tuple(vt[id(source)] for source in node.inputs)
    return execute_nll_loss_arrays(contract, arrays)


def _h_nll_loss_vjp(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    contract = validate_nll_loss_vjp_contract(node)
    dy = vt[id(node.inputs[0])]
    arrays = tuple(vt[id(source)] for source in node.inputs[1:])
    return execute_nll_loss_vjp_array(contract, dy, arrays)


def _h_cross_entropy(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    contract = validate_cross_entropy_contract(node)
    arrays = tuple(vt[id(source)] for source in node.inputs)
    return execute_cross_entropy_arrays(contract, arrays)


def _h_cross_entropy_vjp(
    node: UOp,
    vt: dict,
    bt: BufferTable,
) -> np.ndarray:
    contract, operand = validate_cross_entropy_vjp_contract(node)
    dy = vt[id(node.inputs[0])]
    arrays = tuple(vt[id(source)] for source in node.inputs[1:])
    return execute_cross_entropy_vjp_array(
        contract, operand, dy, arrays
    )


def _h_dropout(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    contract = validate_dropout_contract(node)
    return execute_dropout_array(contract, vt[id(node.inputs[0])])


def _h_dropout_vjp(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    contract = validate_dropout_vjp_contract(node)
    return execute_dropout_vjp_array(
        contract,
        vt[id(node.inputs[0])],
        vt[id(node.inputs[1])],
    )


def _h_batch_norm_1d(
    node: UOp,
    vt: dict,
    bt: BufferTable,
) -> np.ndarray:
    contract = validate_batch_norm_1d_contract(node)
    arrays = tuple(vt[id(source)] for source in node.inputs)
    return execute_batch_norm_1d_array(contract, arrays)


def _h_batch_norm_1d_stats_update(
    node: UOp,
    vt: dict,
    bt: BufferTable,
) -> np.ndarray:
    contract = validate_batch_norm_1d_stats_update_contract(node)
    source = vt[id(node.inputs[contract.source_input_index])]
    source_contract = infer_batch_norm_1d_contract(
        (node.inputs[contract.source_input_index],),
        0.0,
        False,
        "batch",
    )
    mean, biased_var = batch_norm_1d_batch_stats_array(
        source, source_contract
    )
    running_mean = bt.get(contract.running_mean_buffer_id)
    running_var = bt.get(contract.running_var_buffer_id)
    tracked = bt.get(contract.num_batches_tracked_buffer_id)
    if (
        tracked.shape != ()
        or tracked.dtype.name != "int64"
        or int(tracked) < 0
        or int(tracked) >= (1 << 63) - 1
    ):
        raise ShapeError(
            "BatchNorm1d num_batches_tracked must be a non-negative int64 "
            "below its overflow boundary"
        )
    next_count = int(tracked) + 1
    factor = (
        1.0 / float(next_count)
        if contract.momentum is None
        else contract.momentum
    )
    unbiased_var = np.asarray(
        biased_var
        * np.float32(
            contract.sample_count / float(contract.sample_count - 1)
        ),
        dtype=np.float32,
    )
    next_mean = np.asarray(
        (1.0 - factor) * running_mean + factor * mean,
        dtype=np.float32,
    )
    next_var = np.asarray(
        (1.0 - factor) * running_var + factor * unbiased_var,
        dtype=np.float32,
    )
    next_tracked = np.asarray(next_count, dtype=np.int64)
    bt.apply_updates_once(
        contract.effect_id,
        BATCH_NORM_1D_STATE_EFFECT_KIND,
        (
            (contract.running_mean_buffer_id, next_mean),
            (contract.running_var_buffer_id, next_var),
            (contract.num_batches_tracked_buffer_id, next_tracked),
        ),
    )
    return np.stack((mean, biased_var), axis=0).astype(
        np.float32, copy=False
    )


def _h_batch_norm_1d_vjp(
    node: UOp,
    vt: dict,
    bt: BufferTable,
) -> np.ndarray:
    contract, operand = validate_batch_norm_1d_vjp_contract(node)
    dy = vt[id(node.inputs[0])]
    arrays = tuple(vt[id(source)] for source in node.inputs[1:])
    return execute_batch_norm_1d_vjp_array(
        contract, arrays, dy, operand
    )


def _h_interpolate_2d(
    node: UOp,
    vt: dict,
    bt: BufferTable,
) -> np.ndarray:
    contract = validate_interpolate_2d_contract(node)
    return execute_interpolate_2d_array(
        contract,
        vt[id(node.inputs[0])],
    )


def _h_interpolate_2d_vjp(
    node: UOp,
    vt: dict,
    bt: BufferTable,
) -> np.ndarray:
    contract = validate_interpolate_2d_vjp_contract(node)
    return execute_interpolate_2d_vjp_array(
        contract,
        vt[id(node.inputs[0])],
        vt[id(node.inputs[1])],
    )


def _h_where(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    validate_where_contract(node)
    cond = vt[id(node.inputs[0])]
    a = vt[id(node.inputs[1])]
    b = vt[id(node.inputs[2])]
    selected = np.where(cond, a, b)
    return np.array(selected, dtype=np.dtype(node.dtype), copy=True).reshape(node.shape)


def _gather_index_tuple(index: np.ndarray, axis: int) -> tuple[np.ndarray, ...]:
    coordinates = []
    for dimension in range(index.ndim):
        if dimension == axis:
            coordinates.append(index)
        else:
            extent = index.shape[dimension]
            shape = [1] * index.ndim
            shape[dimension] = extent
            coordinate = np.arange(extent).reshape(shape)
            coordinates.append(np.broadcast_to(coordinate, index.shape))
    return tuple(coordinates)


def _h_index(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    idx = vt[id(node.inputs[1])]
    dim = validate_gather_contract(node)
    if idx.size and (bool(np.any(idx < 0)) or bool(np.any(idx >= x.shape[dim]))):
        raise RealizationError(
            f"gather: index values must be in [0, {x.shape[dim]})"
        )
    gathered = x[_gather_index_tuple(idx, dim)]
    return np.array(gathered, dtype=np.dtype(node.dtype), copy=True)


def _h_mask(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    x = vt[id(node.inputs[0])]
    mask = vt[id(node.inputs[1])]
    return x[mask]


def _h_custom(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    """Opaque escape hatch. The `arg` carries a callable that receives the
    input ndarrays and returns the output ndarray.

    Kept for the reviewed legacy NumPy-callback inventory while those public
    operations migrate to typed IR. Nodes without `arg["fn"]` fail here."""
    fn = node.arg.get("fn")
    if fn is None:
        raise RealizationError(
            f"CUSTOM UOp: arg must carry a 'fn' callable; got {node.arg!r}"
        )
    inputs = tuple(vt[id(inp)] for inp in node.inputs)
    captures = node.arg.get("captures", ())
    out = fn(*inputs, *captures)
    if not isinstance(out, np.ndarray):
        raise RealizationError(
            f"CUSTOM UOp: fn returned {type(out).__name__}, expected np.ndarray"
        )
    return out


def _h_broadcast_to(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    """Broadcast `x` to `arg["shape"]`. The realizer's metadata-shape
    contract doesn't coerce — ADD or any other op produces NumPy's
    natural broadcast result, which equals the smaller side's shape.
    This opcode makes the broadcast explicit.

    Calls `np.broadcast_to(...)` which returns a view (read-only by
    default); we `.copy()` so downstream ops see a writable, owning
    ndarray. The copy cost is the price of "the IR is the truth"
    semantics — without it, a buggy downstream op could mutate the
    broadcast source in-place.
    """
    x = vt[id(node.inputs[0])]
    target_shape = validate_broadcast_to_contract(node)
    return np.broadcast_to(x, target_shape).copy()


def _h_scatter_add(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    """Scatter-add the source values into a copy of `target` at positions
    given by `idx` along `dim`. The inverse of INDEX / GATHER.

    Inputs: (target, idx, src). The target carries the receiving shape and
    starting values (typically zeros built by the autograd builder); the
    realizer copies it so the original BUFFER stays untouched.

    `np.add.at` is the deterministic-by-construction NumPy call — same
    output every run. When PRD-012 lowers this to WGSL, the kernel must
    preserve that determinism (sort-and-segment-reduce by default).
    """
    target = vt[id(node.inputs[0])]
    idx = vt[id(node.inputs[1])]
    src = vt[id(node.inputs[2])]
    dim = validate_gather_scatter_add_contract(node)
    if idx.size and (bool(np.any(idx < 0)) or bool(np.any(idx >= target.shape[dim]))):
        raise RealizationError(
            f"gather: index values must be in [0, {target.shape[dim]})"
        )
    out = target.copy()
    np.add.at(out, _gather_index_tuple(idx, dim), src)
    return out


def _h_fused_elementwise(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    """Realize an OP_FUSED_ELEMENTWISE chain in a single Python loop.

    The arg carries `ops`: a tuple of `(opcode, lhs_ref, rhs_ref_or_None)`.
    Refs are integers — negative means "external input N" (using
    `node.inputs[-ref - 1]`), non-negative means "the i-th step's output."

    Compared to dispatching each op through the realizer's main loop:
      * No `value_table` insertion + lookup overhead per intermediate.
      * No np.ndarray retained for any non-terminal step (Python's GC
        reclaims each step once the next one consumes it).
      * One Python frame instead of N.

    On a 3-op chain over (B=64, hidden=512) f32: peak intermediate memory
    drops from 384 KB (3 × 128 KB) to 128 KB (one in-flight).
    """
    ops = node.arg["ops"]
    externals = [vt[id(inp)] for inp in node.inputs]
    steps: list[np.ndarray] = []

    def _resolve(ref: int) -> np.ndarray:
        if ref < 0:
            return externals[-ref - 1]
        return steps[ref]

    for opcode, lhs_ref, rhs_ref in ops:
        a = _resolve(lhs_ref)
        if opcode == OP_ADD:
            steps.append(a + _resolve(rhs_ref))
        elif opcode == OP_MUL:
            steps.append(a * _resolve(rhs_ref))
        elif opcode == OP_DIV:
            steps.append(a / _resolve(rhs_ref))
        elif opcode == OP_NEG:
            steps.append(-a)
        elif opcode == OP_EXP:
            steps.append(np.exp(a))
        elif opcode == OP_LOG:
            steps.append(np.log(a))
        else:
            raise RealizationError(
                f"FUSED_ELEMENTWISE: unsupported inner opcode {opcode!r}"
            )
    return steps[-1]


def _h_fused_softmax(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    """Realize an OP_FUSED_SOFTMAX in three NumPy calls.

    Numerically stable: subtract the row-max before exp. Equivalent to
    the unfused 7-node decomposition the matcher absorbed, but emits
    fewer intermediate ndarrays (`x - m` doesn't survive past the
    subsequent `np.exp`)."""
    x = vt[id(node.inputs[0])]
    axis = node.arg["axis"]
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _h_store(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    """STORE writes its source into the BUFFER referenced by inputs[0].

    arg: {accumulate: bool}. With accumulate=True, the existing buffer
    content is added; otherwise overwritten. Returns the stored value so
    callers can chain — though typically STORE is a sink.
    """
    buf_uop = node.inputs[0]
    src = vt[id(node.inputs[1])]
    if buf_uop.op != OP_BUFFER:
        raise RealizationError(
            f"STORE: first input must be a BUFFER, got {buf_uop.op}"
        )
    target = bt.get(buf_uop.arg)
    accumulate = node.arg.get("accumulate", False) if node.arg else False
    new_value = (target + src) if accumulate else src
    bt.update(buf_uop.arg, new_value.astype(target.dtype, copy=False))
    return new_value


def _h_sgd_update(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    param = vt[id(node.inputs[0])]
    grad = vt[id(node.inputs[1])]
    lr = float(node.arg.get("lr", 0.0))
    weight_decay = float(node.arg.get("weight_decay", 0.0))
    update = grad + weight_decay * param if weight_decay != 0.0 else grad
    return (param - lr * update).astype(np.dtype(node.dtype), copy=False)


def _h_adamw_update_m(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    m = vt[id(node.inputs[0])]
    grad = vt[id(node.inputs[1])]
    beta1 = float(node.arg["beta1"])
    return (beta1 * m + (1.0 - beta1) * grad).astype(np.dtype(node.dtype), copy=False)


def _h_adamw_update_v(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    v = vt[id(node.inputs[0])]
    grad = vt[id(node.inputs[1])]
    beta2 = float(node.arg["beta2"])
    return (beta2 * v + (1.0 - beta2) * (grad * grad)).astype(np.dtype(node.dtype), copy=False)


def _h_adamw_update_param(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    param = vt[id(node.inputs[0])]
    m_new = vt[id(node.inputs[2])]
    v_new = vt[id(node.inputs[3])]
    arg = node.arg
    lr = float(arg["lr"])
    beta1 = float(arg["beta1"])
    beta2 = float(arg["beta2"])
    eps = float(arg["eps"])
    step = int(arg["step"])
    weight_decay = float(arg.get("weight_decay", 0.0))
    m_hat = m_new / (1.0 - beta1 ** step)
    v_hat = v_new / (1.0 - beta2 ** step)
    update = m_hat / (np.sqrt(v_hat) + eps)
    return (param - lr * update - lr * weight_decay * param).astype(
        np.dtype(node.dtype),
        copy=False,
    )


def _h_adam_update_m(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    param = vt[id(node.inputs[0])]
    grad = vt[id(node.inputs[1])]
    m = vt[id(node.inputs[2])]
    beta1 = float(node.arg["beta1"])
    weight_decay = float(node.arg.get("weight_decay", 0.0))
    grad_eff = grad + weight_decay * param if weight_decay != 0.0 else grad
    return (beta1 * m + (1.0 - beta1) * grad_eff).astype(
        np.dtype(node.dtype),
        copy=False,
    )


def _h_adam_update_v(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    param = vt[id(node.inputs[0])]
    grad = vt[id(node.inputs[1])]
    v = vt[id(node.inputs[2])]
    beta2 = float(node.arg["beta2"])
    weight_decay = float(node.arg.get("weight_decay", 0.0))
    grad_eff = grad + weight_decay * param if weight_decay != 0.0 else grad
    return (beta2 * v + (1.0 - beta2) * (grad_eff * grad_eff)).astype(
        np.dtype(node.dtype),
        copy=False,
    )


def _h_adam_update_param(node: UOp, vt: dict, bt: BufferTable) -> np.ndarray:
    param = vt[id(node.inputs[0])]
    m_new = vt[id(node.inputs[1])]
    v_new = vt[id(node.inputs[2])]
    arg = node.arg
    lr = float(arg["lr"])
    beta1 = float(arg["beta1"])
    beta2 = float(arg["beta2"])
    eps = float(arg["eps"])
    step = int(arg["step"])
    m_hat = m_new / (1.0 - beta1 ** step)
    v_hat = v_new / (1.0 - beta2 ** step)
    return (param - lr * (m_hat / (np.sqrt(v_hat) + eps))).astype(
        np.dtype(node.dtype),
        copy=False,
    )


# Dispatch table. Adding a new opcode here is also an update to ALL_OPS in
# _ir.py — the sanity check at module-import time below would fire if we
# forget.
_DISPATCH: dict[str, Handler] = {
    OP_BUFFER:  _h_buffer,
    OP_LOAD:    _h_load,
    OP_STORE:   _h_store,
    OP_CONST:   _h_const,
    OP_RANDOM:  _h_random,
    OP_CAST:    _h_cast,
    OP_ADD:     _h_add,
    OP_MUL:     _h_mul,
    OP_DIV:     _h_div,
    OP_NEG:     _h_neg,
    OP_EXP:     _h_exp,
    OP_LOG:     _h_log,
    OP_ABS:     _h_abs,
    OP_CLAMP:   _h_clamp,
    OP_COS:     _h_cos,
    OP_FLIP:    _h_flip,
    OP_CUMSUM:  _h_cumsum,
    OP_CONCAT:  _h_concat,
    OP_STACK:   _h_stack,
    OP_NARROW:  _h_narrow,
    OP_TRIL:    _h_tril,
    OP_TRIU:    _h_triu,
    OP_PROD:    _h_prod,
    OP_VAR:     _h_var,
    OP_REPEAT:  _h_repeat,
    OP_REPEAT_INTERLEAVE: _h_repeat_interleave,
    OP_SIGN:    _h_sign,
    OP_SIN:     _h_sin,
    OP_CMP:     _h_cmp,
    OP_MATMUL:  _h_matmul,
    OP_CONV1D:  _h_conv1d,
    OP_CONV1D_BACKWARD_INPUT: _h_conv1d_backward_input,
    OP_CONV1D_BACKWARD_WEIGHT: _h_conv1d_backward_weight,
    OP_CONV1D_BACKWARD_BIAS: _h_conv1d_backward_bias,
    OP_CONV2D:  _h_conv2d,
    OP_CONV2D_BACKWARD_INPUT: _h_conv2d_backward_input,
    OP_CONV2D_BACKWARD_WEIGHT: _h_conv2d_backward_weight,
    OP_CONV2D_BACKWARD_BIAS: _h_conv2d_backward_bias,
    OP_CONV_TRANSPOSE2D: _h_conv_transpose2d,
    OP_CONV_TRANSPOSE2D_BACKWARD_INPUT: _h_conv_transpose2d_backward_input,
    OP_CONV_TRANSPOSE2D_BACKWARD_WEIGHT: _h_conv_transpose2d_backward_weight,
    OP_CONV_TRANSPOSE2D_BACKWARD_BIAS: _h_conv_transpose2d_backward_bias,
    OP_CONV3D:  _h_conv3d,
    OP_CONV3D_BACKWARD_INPUT: _h_conv3d_backward_input,
    OP_CONV3D_BACKWARD_WEIGHT: _h_conv3d_backward_weight,
    OP_CONV3D_BACKWARD_BIAS: _h_conv3d_backward_bias,
    OP_LAYER_NORM: _h_layer_norm,
    OP_LAYER_NORM_BACKWARD_INPUT: _h_layer_norm_backward_input,
    OP_LAYER_NORM_BACKWARD_WEIGHT: _h_layer_norm_backward_weight,
    OP_LAYER_NORM_BACKWARD_BIAS: _h_layer_norm_backward_bias,
    OP_REDUCE:  _h_reduce,
    OP_RESHAPE: _h_reshape,
    OP_PERMUTE: _h_permute,
    OP_SLICE:   _h_slice,
    OP_PAD:     _h_pad,
    OP_SORT_INDICES: _h_sort_indices,
    OP_SORT_VALUES: _h_sort_values,
    OP_TOPK_INDICES: _h_topk_indices,
    OP_TOPK_VALUES: _h_topk_values,
    OP_SCATTER: _h_scatter,
    OP_EINSUM: _h_einsum,
    OP_L1_LOSS: _h_l1_loss,
    OP_SMOOTH_L1_LOSS: _h_smooth_l1_loss,
    OP_BINARY_CROSS_ENTROPY: _h_binary_cross_entropy,
    OP_BINARY_CROSS_ENTROPY_WITH_LOGITS: _h_binary_cross_entropy_with_logits,
    OP_KL_DIV: _h_kl_div,
    OP_NLL_LOSS: _h_nll_loss,
    OP_CROSS_ENTROPY: _h_cross_entropy,
    OP_DROPOUT: _h_dropout,
    OP_BATCH_NORM_1D: _h_batch_norm_1d,
    OP_BATCH_NORM_1D_STATS_UPDATE: _h_batch_norm_1d_stats_update,
    OP_INTERPOLATE_2D: _h_interpolate_2d,
    OP_WHERE:   _h_where,
    OP_INDEX:   _h_index,
    OP_MASK:    _h_mask,
    OP_CUSTOM:  _h_custom,
    # Fusion (PRD-006)
    OP_FUSED_ELEMENTWISE: _h_fused_elementwise,
    OP_FUSED_SOFTMAX:     _h_fused_softmax,
    # Autograd (PRD-007)
    OP_SCATTER_ADD:       _h_scatter_add,
    OP_BROADCAST_TO:      _h_broadcast_to,
    OP_EINSUM_VJP:        _h_einsum_vjp,
    OP_L1_LOSS_VJP:       _h_l1_loss_vjp,
    OP_SMOOTH_L1_LOSS_VJP: _h_smooth_l1_loss_vjp,
    OP_BINARY_CROSS_ENTROPY_VJP: _h_binary_cross_entropy_vjp,
    OP_BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP: _h_binary_cross_entropy_with_logits_vjp,
    OP_KL_DIV_VJP: _h_kl_div_vjp,
    OP_NLL_LOSS_VJP: _h_nll_loss_vjp,
    OP_CROSS_ENTROPY_VJP: _h_cross_entropy_vjp,
    OP_DROPOUT_VJP: _h_dropout_vjp,
    OP_BATCH_NORM_1D_VJP: _h_batch_norm_1d_vjp,
    OP_INTERPOLATE_2D_VJP: _h_interpolate_2d_vjp,
    # Mixed precision (PRD-010)
    OP_ISNAN:             _h_isnan,
    # Optimizer/update IR
    OP_SGD_UPDATE:        _h_sgd_update,
    OP_ADAMW_UPDATE_M:    _h_adamw_update_m,
    OP_ADAMW_UPDATE_V:    _h_adamw_update_v,
    OP_ADAMW_UPDATE_PARAM: _h_adamw_update_param,
    OP_ADAM_UPDATE_M:     _h_adam_update_m,
    OP_ADAM_UPDATE_V:     _h_adam_update_v,
    OP_ADAM_UPDATE_PARAM: _h_adam_update_param,
}


# Module-load assertion: every opcode in ALL_OPS has a dispatch handler.
# Forgetting one would otherwise surface as a confusing KeyError mid-realization.
_missing = ALL_OPS - set(_DISPATCH)
if _missing:
    raise RuntimeError(
        f"_realize: opcodes lack dispatch handlers: {sorted(_missing)}"
    )
_extra = set(_DISPATCH) - ALL_OPS
if _extra:
    raise RuntimeError(
        f"_realize: dispatch table has handlers for unknown opcodes: {sorted(_extra)}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def realize(
    root: UOp,
    buffer_table: BufferTable,
    *,
    autograd_holdout: "set[int] | None" = None,
) -> np.ndarray:
    """Walk the IR rooted at `root` in topological order, dispatching each
    UOp to its NumPy handler. Returns a fresh np.ndarray; safe to mutate.

    When fusion is enabled (`_fusion_config.is_enabled()`), the pass runs
    first and replaces `root` with a rewritten graph containing
    `OP_FUSED_*` nodes. The dispatch table picks them up via their
    handlers transparently.

    `autograd_holdout`: the set of UOp ids that must not be absorbed as
    non-terminal nodes in any fused group. The caller (typically a
    backward walker that has access to per-proxy `_ctx.input_proxies`)
    computes this via `_fusion.collect_autograd_holdout` and passes it
    in. Default `None` ≡ "no constraints," which is correct for forward-
    only realization where no closures will later read intermediates.

    Raises `RealizationError` if any handler fails. The error carries the
    offending UOp's opcode + shape in the message so debugging is local.
    """
    from . import _fusion_config, _amp
    # AMP cast-insertion runs BEFORE fusion (PRD-010). Reason: the cast
    # pass walks per-UOp `autocast_hint` tags stamped by TensorProxy
    # builders inside `with autocast(...)`. Fusion's FUSED_SOFTMAX
    # matcher consumes EXP/REDUCE/DIV decompositions — running the cast
    # pass first guarantees those ops are wrapped in CAST(f32) before
    # any fusion rewrite, keeping softmax numerically stable. A
    # consequence: cast-wrapped softmax chains will not match the
    # FUSED_SOFTMAX pattern in NumPy v0; the wins flow through the
    # f32-accumulated MATMUL handler instead (the load-bearing piece).
    # No-op precheck inside `insert_cast_pass` makes this free when
    # autocast is inactive.
    root = _amp.insert_cast_pass(root)
    if _fusion_config.is_enabled():
        from ._fusion import fuse
        root = fuse(root, holdout=autograd_holdout or set())
    else:
        # Keep introspection state consistent: a realize() with fusion
        # disabled should not leave a stale report from a prior fuse()
        # call lying around for debug_fused_kernels() to find.
        from . import _fusion as _f
        _f._LAST_REPORT = _f.FusionReport()

    value_table: dict[int, np.ndarray] = {}
    order = toposort(root)
    for node in order:
        handler = _DISPATCH.get(node.op)
        if handler is None:
            # Should be unreachable thanks to the module-load assertion; if
            # it ever fires, the dispatch table drifted from ALL_OPS without
            # raising at import. Surface the bug clearly.
            raise RealizationError(
                f"no dispatch handler for opcode {node.op!r}"
            )
        try:
            value_table[id(node)] = handler(node, value_table, buffer_table)
        except Exception as e:
            # Wrap with context. Keep the cause chain so debuggers can
            # walk the original traceback.
            raise RealizationError(
                f"{node.op} (shape={node.shape}, dtype={node.dtype}) failed: {e}"
            ) from e
    return value_table[id(root)]


__all__ = ["realize"]
