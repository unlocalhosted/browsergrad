"""browsergrad_jit._onnx — ONNX export (PRD-016).

INTERNAL. Public surface: `bg.onnx.export_inference(root, ...)`.

Hand-rolled pure-Python proto3 encoder per the DL/GPU review. Pyodide
does not ship the `onnx` package (it depends on C-ext protobuf), and
a JS-bridge round-trip is overkill for educational scope. Writing the
~30 ONNX proto fields we touch from scratch is ~200 LOC and pays back
immediately (no wheel, no build step, runs in browser and Node alike).

Scope (v0):
  * Forward-only. Backward export is undefined in the ONNX inference
    spec — emit the forward subgraph reachable from `output_uops`.
  * Caller declares which BUFFERs are inputs vs initializers via
    `input_buffers`. Everything else reachable becomes an initializer.
  * Core arithmetic/shape export covers ADD, MUL, DIV, NEG, EXP, LOG, MATMUL,
    REDUCE (sum/mean/max), RESHAPE, PERMUTE, CAST, WHERE, CMP
    (→Equal/Greater/Less), and BROADCAST_TO (→Expand). Registry-admitted
    framework operations additionally cover ABS, SIGN, SIN, COS, CLAMP
    (→Clip), FLIP (→Slice), INDEX (→GatherElements), PROD (→ReduceProd),
    VAR (→ReduceMean/Sub/Mul/ReduceSum/Div), TRIL/TRIU (→Trilu), REPEAT (→Tile), and
    REPEAT_INTERLEAVE (→Unsqueeze/Tile/Reshape), constant PAD (→Pad),
    SCATTER (→ScatterElements), EINSUM (→Einsum), L1_LOSS
    (→Sub/Abs/ReduceSum or ReduceMean), SMOOTH_L1_LOSS
    (→Sub/Abs/Less/Mul/Where/reduce), BCE with logits
    (→Neg/Softplus/Sub/Mul/Add/reduce), KL divergence
    (→Log/Exp/Equal/Where/Mul/Sub/reduce), and paired sort and selected top-k
    indices/values (→TopK/GatherElements).
    Plus lifecycle
    (BUFFER/LOAD/CONST).
  * Opset 17 (axes as attribute on ReduceSum/Mean/Max — opset 18 made
    axes a runtime input, which would require initializer plumbing).
  * The encoder represents bool, uint8, signed int8/16/32/64, unsigned
    int16/32/64, and float16/32/64. Each operation contract narrows that graph
    dtype profile explicitly.

Refusals (typed `OnnxUnmappableOp`):
  * OP_RANDOM (no runtime randomness in ONNX inference)
  * OP_CONV1D/OP_CONV2D/OP_CONV_TRANSPOSE2D/OP_CONV3D and OP_CONV*_BACKWARD_*
    (tensor compiler/export mapping deferred)
  * OP_CUSTOM (opaque)
  * OP_MASK, OP_SCATTER_ADD (initializer-tensor plumbing
    deferred to a follow-on)
  * OP_ISNAN, OP_SLICE (same — needs initializer wiring or
    opset-specific shapes)
  * OP_FUSED_ELEMENTWISE, OP_FUSED_SOFTMAX (export on pre-fusion IR
    via `_fusion_config.use_fusion(False)`; the export path enforces
    this by disabling fusion for the duration of the call)

Why this is the load-bearing core, not the polish:
  * The PRD's "PyTorch-shaped wrapper, dynamic axes, verify-by-default,
    training-graph branch, browser download UX" are layers above this.
    Each can land independently once the encoder is real.
"""

from __future__ import annotations
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ._ir import (
    UOp, toposort,
    OP_BUFFER, OP_LOAD, OP_CONST, OP_CAST,
    OP_ADD, OP_MUL, OP_DIV, OP_NEG, OP_EXP, OP_LOG,
    OP_ABS, OP_CLAMP, OP_COS, OP_FLIP, OP_CUMSUM, OP_CONCAT, OP_STACK, OP_NARROW, OP_TRIL, OP_TRIU, OP_PROD, OP_VAR, OP_REPEAT, OP_REPEAT_INTERLEAVE,
    OP_SIGN, OP_SIN, OP_CMP,
    OP_MATMUL, OP_CONV1D, OP_CONV1D_BACKWARD_INPUT,
    OP_CONV1D_BACKWARD_WEIGHT, OP_CONV1D_BACKWARD_BIAS,
    OP_CONV2D, OP_CONV2D_BACKWARD_INPUT,
    OP_CONV2D_BACKWARD_WEIGHT, OP_CONV2D_BACKWARD_BIAS,
    OP_CONV_TRANSPOSE2D, OP_CONV_TRANSPOSE2D_BACKWARD_INPUT,
    OP_CONV_TRANSPOSE2D_BACKWARD_WEIGHT, OP_CONV_TRANSPOSE2D_BACKWARD_BIAS,
    OP_CONV3D, OP_CONV3D_BACKWARD_INPUT,
    OP_CONV3D_BACKWARD_WEIGHT, OP_CONV3D_BACKWARD_BIAS,
    OP_LAYER_NORM, OP_LAYER_NORM_BACKWARD_INPUT,
    OP_LAYER_NORM_BACKWARD_WEIGHT, OP_LAYER_NORM_BACKWARD_BIAS,
    OP_REDUCE, OP_RESHAPE, OP_PERMUTE, OP_PAD,
    OP_SORT_INDICES, OP_SORT_VALUES,
    OP_TOPK_INDICES, OP_TOPK_VALUES, OP_SCATTER, OP_EINSUM, OP_L1_LOSS,
    OP_SMOOTH_L1_LOSS, OP_BINARY_CROSS_ENTROPY,
    OP_BINARY_CROSS_ENTROPY_WITH_LOGITS,
    OP_KL_DIV,
    OP_NLL_LOSS,
    OP_CROSS_ENTROPY,
    OP_WHERE, OP_BROADCAST_TO, OP_INDEX, OP_SGD_UPDATE,
    OP_ADAMW_UPDATE_M, OP_ADAMW_UPDATE_V, OP_ADAMW_UPDATE_PARAM,
    OP_ADAM_UPDATE_M, OP_ADAM_UPDATE_V, OP_ADAM_UPDATE_PARAM,
)
from ._errors import JitError
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
    validate_l1_loss_contract,
    validate_smooth_l1_loss_contract,
    validate_binary_cross_entropy_contract,
    validate_binary_cross_entropy_with_logits_contract,
    validate_kl_div_contract,
    validate_nll_loss_contract,
    validate_cross_entropy_contract,
    validate_clamp_contract,
    validate_cumsum_contract,
    validate_flip_contract,
    validate_narrow_contract,
    validate_tril_contract,
    validate_triu_contract,
    validate_gather_contract,
    validate_prod_contract,
    validate_var_contract,
    validate_where_contract,
    validate_repeat_contract,
    validate_repeat_interleave_contract,
    validate_typed_unary_contract,
    einsum_onnx_equation,
)


class OnnxUnmappableOp(JitError):
    """An opcode in the IR has no ONNX equivalent (or none we ship in v0)."""


# --------------------------------------------------------------------------
# Proto3 wire-format primitives.
#
# Field encoding: (tag, wire_type) varint header + payload.
# Wire types:
#   0 = VARINT (int32, int64, uint32, uint64, bool, enum)
#   1 = FIXED64
#   2 = LENGTH-DELIMITED (string, bytes, embedded message, packed)
#   5 = FIXED32
# --------------------------------------------------------------------------


def _varint(n: int) -> bytes:
    out = bytearray()
    if n < 0:
        # Two's complement 64-bit for negative ints.
        n &= (1 << 64) - 1
    while n >= 0x80:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n & 0x7F)
    return bytes(out)


def _tag(field_no: int, wire_type: int) -> bytes:
    return _varint((field_no << 3) | wire_type)


def _emit_int64(field_no: int, v: int) -> bytes:
    return _tag(field_no, 0) + _varint(v)


def _emit_int32(field_no: int, v: int) -> bytes:
    return _tag(field_no, 0) + _varint(v)


def _emit_string(field_no: int, s: str) -> bytes:
    data = s.encode("utf-8")
    return _tag(field_no, 2) + _varint(len(data)) + data


def _emit_bytes(field_no: int, b: bytes) -> bytes:
    return _tag(field_no, 2) + _varint(len(b)) + b


def _emit_message(field_no: int, body: bytes) -> bytes:
    return _tag(field_no, 2) + _varint(len(body)) + body


def _emit_packed_int64(field_no: int, values: Sequence[int]) -> bytes:
    inner = b"".join(_varint(v if v >= 0 else (v & ((1 << 64) - 1)))
                     for v in values)
    return _tag(field_no, 2) + _varint(len(inner)) + inner


# --------------------------------------------------------------------------
# ONNX field numbers (from onnx.proto3, schema-as-of opset 17).
#
# Hand-curated to keep the dependency footprint zero. We only touch ~30
# fields; the full proto has hundreds. If a future op needs a new field,
# add it here with its number.
# --------------------------------------------------------------------------


# TensorProto.DataType enum
DT_FLOAT = 1
DT_UINT8 = 2
DT_INT8 = 3
DT_UINT16 = 4
DT_INT16 = 5
DT_INT32 = 6
DT_INT64 = 7
DT_BOOL = 9
DT_FLOAT16 = 10
DT_DOUBLE = 11
DT_UINT32 = 12
DT_UINT64 = 13


_DTYPE_TO_ONNX: Dict[str, int] = {
    "float32": DT_FLOAT,
    "float16": DT_FLOAT16,
    "float64": DT_DOUBLE,
    "uint8": DT_UINT8,
    "int8": DT_INT8,
    "int16": DT_INT16,
    "int32": DT_INT32,
    "int64": DT_INT64,
    "uint16": DT_UINT16,
    "uint32": DT_UINT32,
    "uint64": DT_UINT64,
    "bool": DT_BOOL,
}

# The existing exporter contract predates typed EINSUM and intentionally
# exposes only this conservative graph-wide dtype profile.  Keep the broader
# TensorProto encoder private to EINSUM and L1_LOSS so their explicit records
# cannot silently widen unrelated operation contracts.
_LEGACY_GRAPH_DTYPES = frozenset({"float32", "int32", "int64", "bool"})


# AttributeProto.AttributeType enum
AT_INT = 2
AT_STRING = 3
AT_INTS = 7


# Field numbers
# TensorProto: dims=1, data_type=2, name=8, raw_data=9
# ValueInfoProto: name=1, type=2
# TypeProto: tensor_type=1
# TypeProto.Tensor: elem_type=1, shape=2
# TensorShapeProto: dim=1
# TensorShapeProto.Dimension: dim_value=1, dim_param=2
# NodeProto: input=1, output=2, name=3, op_type=4, attribute=5
# AttributeProto: name=1, type=20, i=3, ints=8 (in opset 17)
# GraphProto: node=1, name=2, initializer=5, input=11, output=12, value_info=13
# ModelProto: ir_version=1, producer_name=2, producer_version=3,
#             opset_import=8, graph=7


def _emit_tensor_shape(shape: Sequence[int]) -> bytes:
    """TensorShapeProto with each dim as dim_value."""
    dims = []
    for d in shape:
        # Dimension.dim_value (field 1, varint)
        inner = _emit_int64(1, int(d))
        # Wrap as TensorShapeProto.dim (field 1 of TensorShapeProto, message)
        dims.append(_emit_message(1, inner))
    return b"".join(dims)


def _emit_type_proto_tensor(elem_type: int, shape: Sequence[int]) -> bytes:
    """TypeProto wrapping a Tensor sub-message."""
    # TypeProto.Tensor: elem_type=1 (int32), shape=2 (TensorShapeProto)
    inner = _emit_int32(1, elem_type) + _emit_message(2, _emit_tensor_shape(shape))
    # TypeProto.tensor_type = field 1
    return _emit_message(1, inner)


def _emit_value_info(name: str, elem_type: int, shape: Sequence[int]) -> bytes:
    """ValueInfoProto: name + type."""
    return _emit_string(1, name) + _emit_message(2, _emit_type_proto_tensor(elem_type, shape))


def _emit_tensor_proto(
    name: str,
    elem_type: int,
    shape: Sequence[int],
    raw_data: bytes,
) -> bytes:
    """TensorProto with raw_data. We always use raw_data (not float_data /
    int64_data) because raw_data is cheaper to write (one length-prefixed
    blob) and accepted by every conformant ONNX runtime."""
    parts = []
    for d in shape:
        parts.append(_emit_int64(1, int(d)))  # dims = field 1, repeated int64
    parts.append(_emit_int32(2, elem_type))   # data_type = field 2
    parts.append(_emit_string(8, name))        # name = field 8
    parts.append(_emit_bytes(9, raw_data))     # raw_data = field 9
    return b"".join(parts)


def _emit_attr_int(name: str, value: int) -> bytes:
    """AttributeProto with a single int64 attribute."""
    parts = [
        _emit_string(1, name),       # name = field 1
        _emit_int32(20, AT_INT),     # type = field 20 (AttributeType.INT)
        _emit_int64(3, value),        # i = field 3
    ]
    return b"".join(parts)


def _emit_attr_ints(name: str, values: Sequence[int]) -> bytes:
    """AttributeProto with a repeated int64 attribute."""
    parts = [
        _emit_string(1, name),
        _emit_int32(20, AT_INTS),
        _emit_packed_int64(8, values),  # ints = field 8, packed
    ]
    return b"".join(parts)


def _emit_attr_string(name: str, value: str) -> bytes:
    """AttributeProto with one strict UTF-8 string attribute."""
    parts = [
        _emit_string(1, name),
        _emit_int32(20, AT_STRING),
        _emit_string(4, value),
    ]
    return b"".join(parts)


def _emit_node(
    inputs: Sequence[str],
    outputs: Sequence[str],
    name: str,
    op_type: str,
    attributes: Sequence[bytes] = (),
) -> bytes:
    """NodeProto."""
    parts = []
    for inp in inputs:
        parts.append(_emit_string(1, inp))     # input = field 1
    for out in outputs:
        parts.append(_emit_string(2, out))     # output = field 2
    parts.append(_emit_string(3, name))         # name = field 3
    parts.append(_emit_string(4, op_type))      # op_type = field 4
    for attr in attributes:
        parts.append(_emit_message(5, attr))   # attribute = field 5
    return b"".join(parts)


def _emit_opset_import(domain: str, version: int) -> bytes:
    """OperatorSetIdProto: domain=1, version=2."""
    return _emit_string(1, domain) + _emit_int64(2, version)


# --------------------------------------------------------------------------
# Op mapping
# --------------------------------------------------------------------------


# Direct 1:1 op-name mapping for ops that take inputs and produce outputs
# with no attribute plumbing required.
_SIMPLE_OPS: Dict[str, str] = {
    OP_ADD: "Add",
    OP_MUL: "Mul",
    OP_DIV: "Div",
    OP_NEG: "Neg",
    OP_EXP: "Exp",
    OP_LOG: "Log",
    OP_MATMUL: "MatMul",
    OP_WHERE: "Where",
}

_TYPED_FRAMEWORK_SIMPLE_OPS: Dict[str, str] = {
    OP_ABS: "Abs",
    OP_COS: "Cos",
    OP_SIGN: "Sign",
    OP_SIN: "Sin",
}

_CMP_OP_MAP: Dict[str, str] = {
    "eq": "Equal",
    "lt": "Less",
    "le": "LessOrEqual",
    "gt": "Greater",
    "ge": "GreaterOrEqual",
    "ne": "Equal",  # Wrapped with Not below if needed
}

_REDUCE_OP_MAP: Dict[str, str] = {
    "sum": "ReduceSum",
    "mean": "ReduceMean",
    "max": "ReduceMax",
    "min": "ReduceMin",
}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


@dataclass
class _NamedTensor:
    """A BUFFER that becomes either a graph input or initializer."""
    buffer_id: str
    name: str
    is_input: bool   # True → graph.input; False → graph.initializer


def export_inference(
    root_uop: UOp,
    *,
    buffer_table: Any,
    input_buffers: Sequence[Any] = (),
    output_name: str = "output",
    model_name: str = "browsergrad_model",
    opset_version: int = 17,
) -> bytes:
    """Serialize the IR rooted at `root_uop` as an ONNX ModelProto.

    Parameters
    ----------
    root_uop
        The output UOp of the graph to export. The graph reachable from
        this root (via .inputs) becomes the ONNX model.
    buffer_table
        The session's BufferTable. Used to fetch initializer bytes.
    input_buffers
        TensorProxy or BUFFER-id-string entries whose BUFFER becomes a
        graph input (placeholder). Everything else reachable becomes
        an initializer (frozen weight). At the IR level we can't tell
        weights from placeholders — the caller must.
    output_name
        Name of the graph's single output.
    model_name
        Model name written to the producer.
    opset_version
        ONNX opset to declare. v0 supports 17.

    Returns the serialized ModelProto bytes.
    """
    if opset_version != 17:
        raise NotImplementedError(
            f"export_inference: only opset_version=17 supported in v0 "
            f"(got {opset_version}). Opset 18+ moves Reduce axes to a runtime "
            f"input which needs additional initializer plumbing."
        )

    # Resolve input_buffers to a set of BUFFER ids.
    input_buffer_ids: set = set()
    for b in input_buffers:
        if isinstance(b, str):
            input_buffer_ids.add(b)
        elif hasattr(b, "_uop"):
            u = b._uop
            # TensorProxy wraps LOAD(BUFFER(...)). Walk to the BUFFER.
            if u.op == OP_LOAD:
                u = u.inputs[0]
            if u.op != OP_BUFFER:
                raise ValueError(
                    f"export_inference: input_buffers entry doesn't resolve "
                    f"to a BUFFER (got op {u.op})"
                )
            input_buffer_ids.add(u.arg)
        else:
            raise TypeError(
                f"export_inference: input_buffers entry has unexpected type "
                f"{type(b).__name__}"
            )

    order = toposort(root_uop)

    # Assign names to every UOp on the chain. Reachable BUFFERs become
    # inputs or initializers; intermediate ops get "node{N}" names.
    uop_to_name: Dict[int, str] = {}
    nodes: List[bytes] = []
    initializers: List[bytes] = []
    graph_inputs: List[bytes] = []
    next_node_id = 0

    def name_for(u: UOp) -> str:
        if id(u) in uop_to_name:
            return uop_to_name[id(u)]
        return ""  # filled in below

    # First pass: name BUFFER and CONST leaves.
    const_counter = 0
    for node in order:
        if node.op == OP_BUFFER:
            buffer_id = node.arg
            if buffer_id in input_buffer_ids:
                nm = f"input_{buffer_id.split(':')[-1]}"
                uop_to_name[id(node)] = nm
                graph_inputs.append(
                    _emit_value_info(nm, _dtype_or_die(node.dtype), node.shape)
                )
            else:
                nm = f"weight_{buffer_id.split(':')[-1]}"
                uop_to_name[id(node)] = nm
                arr = buffer_table.get(buffer_id)
                raw = arr.tobytes()
                initializers.append(
                    _emit_tensor_proto(nm, _dtype_or_die(node.dtype), node.shape, raw)
                )
        elif node.op == OP_CONST:
            value = node.arg["value"]
            import numpy as np
            arr = _np_array_for_const(value, node.dtype)
            nm = f"const_{const_counter}"
            const_counter += 1
            uop_to_name[id(node)] = nm
            initializers.append(
                _emit_tensor_proto(nm, _dtype_or_die(node.dtype), node.shape or (1,), arr.tobytes())
            )

    # Second pass: emit compute nodes in topo order.
    for node in order:
        if node.op in (OP_BUFFER, OP_CONST):
            continue
        if node.op == OP_LOAD:
            # LOAD is a pass-through over BUFFER; reuse the BUFFER's name.
            uop_to_name[id(node)] = uop_to_name[id(node.inputs[0])]
            continue
        if node.op not in (
            OP_EINSUM,
            OP_L1_LOSS,
            OP_SMOOTH_L1_LOSS,
            OP_BINARY_CROSS_ENTROPY,
            OP_BINARY_CROSS_ENTROPY_WITH_LOGITS,
            OP_KL_DIV,
            OP_NLL_LOSS,
            OP_CROSS_ENTROPY,
        ) and node.dtype not in _LEGACY_GRAPH_DTYPES:
            raise OnnxUnmappableOp(
                f"export_inference: {node.op} dtype {node.dtype!r} is not "
                "exportable; the legacy graph profile supports float32, "
                "int32, int64, and bool"
            )
        nm = f"node_{next_node_id}_{node.op}"
        next_node_id += 1
        out_name = f"out_{next_node_id - 1}"
        uop_to_name[id(node)] = out_name

        input_names = [uop_to_name[id(inp)] for inp in node.inputs]

        if node.op in _TYPED_FRAMEWORK_SIMPLE_OPS:
            validate_typed_unary_contract(node)
            nodes.append(
                _emit_node(
                    input_names,
                    [out_name],
                    nm,
                    _TYPED_FRAMEWORK_SIMPLE_OPS[node.op],
                )
            )
        elif node.op in _SIMPLE_OPS:
            if node.op == OP_WHERE:
                validate_where_contract(node)
                _dtype_or_die(node.dtype)
            nodes.append(_emit_node(input_names, [out_name], nm, _SIMPLE_OPS[node.op]))
        elif node.op == OP_CAST:
            attrs = [_emit_attr_int("to", _dtype_or_die(node.dtype))]
            nodes.append(_emit_node(input_names, [out_name], nm, "Cast", attrs))
        elif node.op == OP_RESHAPE:
            # ONNX Reshape takes the target shape as a runtime input (initializer).
            shape_arr = _i64_initializer_for_shape(node.arg["new_shape"])
            shape_const_name = f"const_shape_{next_node_id - 1}"
            initializers.append(
                _emit_tensor_proto(shape_const_name, DT_INT64,
                                   (len(node.arg["new_shape"]),), shape_arr)
            )
            nodes.append(_emit_node(input_names + [shape_const_name], [out_name], nm, "Reshape"))
        elif node.op == OP_PERMUTE:
            attrs = [_emit_attr_ints("perm", node.arg["axes"])]
            nodes.append(_emit_node(input_names, [out_name], nm, "Transpose", attrs))
        elif node.op == OP_REDUCE:
            op = node.arg["op"]
            axis = node.arg.get("axis")
            keepdims = node.arg.get("keepdims", False)
            if op not in _REDUCE_OP_MAP:
                raise OnnxUnmappableOp(
                    f"export_inference: REDUCE op {op!r} is not exportable in v0. "
                    f"Supported reduce ops: {sorted(_REDUCE_OP_MAP)}."
                )
            attrs: List[bytes] = []
            if axis is not None:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                attrs.append(_emit_attr_ints("axes", axes))
            attrs.append(_emit_attr_int("keepdims", 1 if keepdims else 0))
            nodes.append(_emit_node(input_names, [out_name], nm, _REDUCE_OP_MAP[op], attrs))
        elif node.op == OP_BROADCAST_TO:
            target_shape = validate_broadcast_to_contract(node)
            shape_arr = _i64_initializer_for_shape(target_shape)
            shape_const_name = f"const_expand_{next_node_id - 1}"
            initializers.append(
                _emit_tensor_proto(shape_const_name, DT_INT64,
                                   (len(target_shape),), shape_arr)
            )
            nodes.append(_emit_node(input_names + [shape_const_name], [out_name], nm, "Expand"))
        elif node.op == OP_CLAMP:
            minimum, maximum = validate_clamp_contract(node)
            onnx_dtype = _dtype_or_die(node.dtype)
            clip_inputs = list(input_names)
            if minimum is not None:
                min_name = f"const_clamp_min_{next_node_id - 1}"
                initializers.append(_emit_tensor_proto(
                    min_name,
                    onnx_dtype,
                    (1,),
                    _initializer_bytes_for_scalar(minimum, node.dtype),
                ))
                clip_inputs.append(min_name)
            elif maximum is not None:
                clip_inputs.append("")
            if maximum is not None:
                max_name = f"const_clamp_max_{next_node_id - 1}"
                initializers.append(_emit_tensor_proto(
                    max_name,
                    onnx_dtype,
                    (1,),
                    _initializer_bytes_for_scalar(maximum, node.dtype),
                ))
                clip_inputs.append(max_name)
            nodes.append(_emit_node(clip_inputs, [out_name], nm, "Clip"))
        elif node.op == OP_FLIP:
            axis = validate_flip_contract(node)
            slice_inputs = list(input_names)
            for suffix, value in (
                ("starts", -1),
                ("ends", -(1 << 63)),
                ("axes", axis),
                ("steps", -1),
            ):
                initializer_name = f"const_flip_{suffix}_{next_node_id - 1}"
                initializers.append(_emit_tensor_proto(
                    initializer_name,
                    DT_INT64,
                    (1,),
                    _i64_initializer_for_shape((value,)),
                ))
                slice_inputs.append(initializer_name)
            nodes.append(_emit_node(slice_inputs, [out_name], nm, "Slice"))
        elif node.op == OP_EINSUM:
            contract = validate_einsum_contract(node)
            if node.dtype == "bool":
                raise OnnxUnmappableOp(
                    "export_inference: ONNX Einsum does not admit bool tensors"
                )
            _dtype_or_die(node.dtype)
            equation = einsum_onnx_equation(contract)
            suffix = next_node_id - 1
            einsum_inputs = []
            for index, (source, source_name) in enumerate(zip(node.inputs, input_names)):
                einsum_input = source_name
                if source.dtype != node.dtype:
                    cast_name = f"einsum_cast_{suffix}_{index}"
                    nodes.append(_emit_node(
                        [source_name],
                        [cast_name],
                        f"{nm}_cast_{index}",
                        "Cast",
                        [_emit_attr_int("to", _dtype_or_die(node.dtype))],
                    ))
                    einsum_input = cast_name
                einsum_inputs.append(einsum_input)
            nodes.append(_emit_node(
                einsum_inputs,
                [out_name],
                nm,
                "Einsum",
                [_emit_attr_string("equation", equation)],
            ))
        elif node.op == OP_L1_LOSS:
            contract = validate_l1_loss_contract(node)
            suffix = next_node_id - 1
            compute_dtype = "float32" if node.dtype == "float16" else node.dtype
            loss_inputs = []
            for index, (source, source_name) in enumerate(zip(node.inputs, input_names)):
                loss_input = source_name
                if source.dtype != compute_dtype:
                    cast_name = f"l1_loss_cast_{suffix}_{index}"
                    nodes.append(_emit_node(
                        [source_name],
                        [cast_name],
                        f"{nm}_cast_{index}",
                        "Cast",
                        [_emit_attr_int("to", _dtype_or_die(compute_dtype))],
                    ))
                    loss_input = cast_name
                loss_inputs.append(loss_input)
            difference_name = f"l1_loss_difference_{suffix}"
            nodes.append(_emit_node(
                loss_inputs,
                [difference_name],
                f"{nm}_subtract",
                "Sub",
            ))
            reduction_axes = tuple(range(contract.batch_rank, len(contract.input_shape)))
            compute_result_name = (
                out_name if compute_dtype == node.dtype else f"l1_loss_compute_result_{suffix}"
            )
            absolute_name = (
                compute_result_name
                if contract.reduction == "none"
                else f"l1_loss_absolute_{suffix}"
            )
            nodes.append(_emit_node(
                [difference_name],
                [absolute_name],
                f"{nm}_absolute",
                "Abs",
            ))
            if contract.reduction != "none":
                if reduction_axes:
                    reduction_op = "ReduceSum" if contract.reduction == "sum" else "ReduceMean"
                    nodes.append(_emit_node(
                        [absolute_name],
                        [compute_result_name],
                        f"{nm}_reduce",
                        reduction_op,
                        [
                            _emit_attr_ints("axes", reduction_axes),
                            _emit_attr_int("keepdims", 0),
                        ],
                    ))
                else:
                    nodes.append(_emit_node(
                        [absolute_name],
                        [compute_result_name],
                        f"{nm}_identity",
                        "Identity",
                    ))
            if compute_dtype != node.dtype:
                nodes.append(_emit_node(
                    [compute_result_name],
                    [out_name],
                    f"{nm}_output_cast",
                    "Cast",
                    [_emit_attr_int("to", _dtype_or_die(node.dtype))],
                ))
        elif node.op == OP_SMOOTH_L1_LOSS:
            contract = validate_smooth_l1_loss_contract(node)
            suffix = next_node_id - 1
            compute_dtype = "float32" if node.dtype == "float16" else node.dtype
            loss_inputs = []
            for index, (source, source_name) in enumerate(zip(node.inputs, input_names)):
                loss_input = source_name
                if source.dtype != compute_dtype:
                    cast_name = f"smooth_l1_loss_cast_{suffix}_{index}"
                    nodes.append(_emit_node(
                        [source_name],
                        [cast_name],
                        f"{nm}_cast_{index}",
                        "Cast",
                        [_emit_attr_int("to", _dtype_or_die(compute_dtype))],
                    ))
                    loss_input = cast_name
                loss_inputs.append(loss_input)

            difference_name = f"smooth_l1_loss_difference_{suffix}"
            absolute_name = f"smooth_l1_loss_absolute_{suffix}"
            nodes.append(_emit_node(
                loss_inputs,
                [difference_name],
                f"{nm}_subtract",
                "Sub",
            ))
            nodes.append(_emit_node(
                [difference_name],
                [absolute_name],
                f"{nm}_absolute",
                "Abs",
            ))

            if contract.beta == 0.0:
                per_element_name = absolute_name
            else:
                beta_name = f"smooth_l1_loss_beta_{suffix}"
                half_beta_name = f"smooth_l1_loss_half_beta_{suffix}"
                half_name = f"smooth_l1_loss_half_{suffix}"
                for constant_name, constant_value in (
                    (beta_name, contract.beta),
                    (half_beta_name, contract.beta * 0.5),
                    (half_name, 0.5),
                ):
                    initializers.append(_emit_tensor_proto(
                        constant_name,
                        _dtype_or_die(compute_dtype),
                        (1,),
                        _initializer_bytes_for_scalar(constant_value, compute_dtype),
                    ))
                quadratic_mask_name = f"smooth_l1_loss_quadratic_mask_{suffix}"
                squared_name = f"smooth_l1_loss_squared_{suffix}"
                divided_name = f"smooth_l1_loss_divided_{suffix}"
                quadratic_name = f"smooth_l1_loss_quadratic_{suffix}"
                linear_name = f"smooth_l1_loss_linear_{suffix}"
                per_element_name = f"smooth_l1_loss_piecewise_{suffix}"
                nodes.append(_emit_node(
                    [absolute_name, beta_name],
                    [quadratic_mask_name],
                    f"{nm}_quadratic_mask",
                    "Less",
                ))
                nodes.append(_emit_node(
                    [difference_name, difference_name],
                    [squared_name],
                    f"{nm}_square",
                    "Mul",
                ))
                nodes.append(_emit_node(
                    [squared_name, beta_name],
                    [divided_name],
                    f"{nm}_divide_beta",
                    "Div",
                ))
                nodes.append(_emit_node(
                    [divided_name, half_name],
                    [quadratic_name],
                    f"{nm}_quadratic",
                    "Mul",
                ))
                nodes.append(_emit_node(
                    [absolute_name, half_beta_name],
                    [linear_name],
                    f"{nm}_linear",
                    "Sub",
                ))
                nodes.append(_emit_node(
                    [quadratic_mask_name, quadratic_name, linear_name],
                    [per_element_name],
                    f"{nm}_piecewise",
                    "Where",
                ))

            reduction_axes = tuple(range(contract.batch_rank, len(contract.input_shape)))
            compute_result_name = (
                out_name
                if compute_dtype == node.dtype
                else f"smooth_l1_loss_compute_result_{suffix}"
            )
            if contract.reduction == "none":
                if per_element_name != compute_result_name:
                    nodes.append(_emit_node(
                        [per_element_name],
                        [compute_result_name],
                        f"{nm}_identity",
                        "Identity",
                    ))
            elif reduction_axes:
                reduction_op = "ReduceSum" if contract.reduction == "sum" else "ReduceMean"
                nodes.append(_emit_node(
                    [per_element_name],
                    [compute_result_name],
                    f"{nm}_reduce",
                    reduction_op,
                    [
                        _emit_attr_ints("axes", reduction_axes),
                        _emit_attr_int("keepdims", 0),
                    ],
                ))
            else:
                nodes.append(_emit_node(
                    [per_element_name],
                    [compute_result_name],
                    f"{nm}_identity",
                    "Identity",
                ))
            if compute_dtype != node.dtype:
                nodes.append(_emit_node(
                    [compute_result_name],
                    [out_name],
                    f"{nm}_output_cast",
                    "Cast",
                    [_emit_attr_int("to", _dtype_or_die(node.dtype))],
                ))
        elif node.op == OP_BINARY_CROSS_ENTROPY:
            validate_binary_cross_entropy_contract(node)
            raise OnnxUnmappableOp(
                "export_inference: BINARY_CROSS_ENTROPY requires fail-closed runtime "
                "validation that every input and target probability is finite and in "
                "[0, 1]; ONNX opset 17 cannot represent that rejection contract"
            )
        elif node.op == OP_BINARY_CROSS_ENTROPY_WITH_LOGITS:
            contract = validate_binary_cross_entropy_with_logits_contract(node)
            suffix = next_node_id - 1
            compute_dtype = "float32" if node.dtype == "float16" else node.dtype
            loss_inputs = []
            for index, (source, source_name) in enumerate(zip(node.inputs, input_names)):
                loss_input = source_name
                if source.dtype != compute_dtype:
                    cast_name = f"bce_with_logits_cast_{suffix}_{index}"
                    nodes.append(_emit_node(
                        [source_name],
                        [cast_name],
                        f"{nm}_cast_{index}",
                        "Cast",
                        [_emit_attr_int("to", _dtype_or_die(compute_dtype))],
                    ))
                    loss_input = cast_name
                loss_inputs.append(loss_input)

            one_name = f"bce_with_logits_one_{suffix}"
            initializers.append(_emit_tensor_proto(
                one_name,
                _dtype_or_die(compute_dtype),
                (),
                _initializer_bytes_for_scalar(1.0, compute_dtype),
            ))
            negative_logits_name = f"bce_with_logits_negative_{suffix}"
            softplus_name = f"bce_with_logits_softplus_{suffix}"
            one_minus_target_name = f"bce_with_logits_one_minus_target_{suffix}"
            linear_name = f"bce_with_logits_linear_{suffix}"
            per_element_name = f"bce_with_logits_per_element_{suffix}"
            nodes.append(_emit_node(
                [loss_inputs[0]],
                [negative_logits_name],
                f"{nm}_negative",
                "Neg",
            ))
            nodes.append(_emit_node(
                [negative_logits_name],
                [softplus_name],
                f"{nm}_softplus",
                "Softplus",
            ))
            nodes.append(_emit_node(
                [one_name, loss_inputs[1]],
                [one_minus_target_name],
                f"{nm}_one_minus_target",
                "Sub",
            ))
            nodes.append(_emit_node(
                [one_minus_target_name, loss_inputs[0]],
                [linear_name],
                f"{nm}_linear",
                "Mul",
            ))
            nodes.append(_emit_node(
                [linear_name, softplus_name],
                [per_element_name],
                f"{nm}_per_element",
                "Add",
            ))

            reduction_axes = tuple(range(contract.batch_rank, len(contract.input_shape)))
            compute_result_name = (
                out_name
                if compute_dtype == node.dtype
                else f"bce_with_logits_compute_result_{suffix}"
            )
            if contract.reduction == "none":
                nodes.append(_emit_node(
                    [per_element_name],
                    [compute_result_name],
                    f"{nm}_identity",
                    "Identity",
                ))
            elif reduction_axes:
                reduction_op = "ReduceSum" if contract.reduction == "sum" else "ReduceMean"
                nodes.append(_emit_node(
                    [per_element_name],
                    [compute_result_name],
                    f"{nm}_reduce",
                    reduction_op,
                    [
                        _emit_attr_ints("axes", reduction_axes),
                        _emit_attr_int("keepdims", 0),
                    ],
                ))
            else:
                nodes.append(_emit_node(
                    [per_element_name],
                    [compute_result_name],
                    f"{nm}_identity",
                    "Identity",
                ))
            if compute_dtype != node.dtype:
                nodes.append(_emit_node(
                    [compute_result_name],
                    [out_name],
                    f"{nm}_output_cast",
                    "Cast",
                    [_emit_attr_int("to", _dtype_or_die(node.dtype))],
                ))
        elif node.op == OP_KL_DIV:
            contract = validate_kl_div_contract(node)
            suffix = next_node_id - 1
            compute_dtype = "float32" if node.dtype == "float16" else node.dtype
            loss_inputs = []
            for index, (source, source_name) in enumerate(zip(node.inputs, input_names)):
                loss_input = source_name
                if source.dtype != compute_dtype:
                    cast_name = f"kl_div_cast_{suffix}_{index}"
                    nodes.append(_emit_node(
                        [source_name],
                        [cast_name],
                        f"{nm}_cast_{index}",
                        "Cast",
                        [_emit_attr_int("to", _dtype_or_die(compute_dtype))],
                    ))
                    loss_input = cast_name
                loss_inputs.append(loss_input)

            if contract.log_target:
                probability_name = f"kl_div_probability_{suffix}"
                difference_name = f"kl_div_difference_{suffix}"
                per_element_name = f"kl_div_per_element_{suffix}"
                nodes.append(_emit_node(
                    [loss_inputs[1]],
                    [probability_name],
                    f"{nm}_target_exp",
                    "Exp",
                ))
                nodes.append(_emit_node(
                    [loss_inputs[1], loss_inputs[0]],
                    [difference_name],
                    f"{nm}_difference",
                    "Sub",
                ))
                nodes.append(_emit_node(
                    [probability_name, difference_name],
                    [per_element_name],
                    f"{nm}_per_element",
                    "Mul",
                ))
            else:
                zero_name = f"kl_div_zero_{suffix}"
                initializers.append(_emit_tensor_proto(
                    zero_name,
                    _dtype_or_die(compute_dtype),
                    (),
                    _initializer_bytes_for_scalar(0.0, compute_dtype),
                ))
                log_name = f"kl_div_target_log_{suffix}"
                xlogy_raw_name = f"kl_div_xlogy_raw_{suffix}"
                zero_mask_name = f"kl_div_zero_mask_{suffix}"
                xlogy_name = f"kl_div_xlogy_{suffix}"
                input_product_name = f"kl_div_input_product_{suffix}"
                per_element_name = f"kl_div_per_element_{suffix}"
                nodes.append(_emit_node(
                    [loss_inputs[1]],
                    [log_name],
                    f"{nm}_target_log",
                    "Log",
                ))
                nodes.append(_emit_node(
                    [loss_inputs[1], log_name],
                    [xlogy_raw_name],
                    f"{nm}_xlogy_raw",
                    "Mul",
                ))
                nodes.append(_emit_node(
                    [loss_inputs[1], zero_name],
                    [zero_mask_name],
                    f"{nm}_zero_mask",
                    "Equal",
                ))
                nodes.append(_emit_node(
                    [zero_mask_name, zero_name, xlogy_raw_name],
                    [xlogy_name],
                    f"{nm}_xlogy",
                    "Where",
                ))
                nodes.append(_emit_node(
                    [loss_inputs[1], loss_inputs[0]],
                    [input_product_name],
                    f"{nm}_input_product",
                    "Mul",
                ))
                nodes.append(_emit_node(
                    [xlogy_name, input_product_name],
                    [per_element_name],
                    f"{nm}_per_element",
                    "Sub",
                ))

            reduction_axes = tuple(range(contract.batch_rank, len(contract.input_shape)))
            compute_result_name = (
                out_name
                if compute_dtype == node.dtype
                else f"kl_div_compute_result_{suffix}"
            )
            if contract.reduction == "none" or not reduction_axes:
                nodes.append(_emit_node(
                    [per_element_name],
                    [compute_result_name],
                    f"{nm}_identity",
                    "Identity",
                ))
            elif contract.reduction in ("sum", "mean"):
                reduction_op = (
                    "ReduceSum" if contract.reduction == "sum" else "ReduceMean"
                )
                nodes.append(_emit_node(
                    [per_element_name],
                    [compute_result_name],
                    f"{nm}_reduce",
                    reduction_op,
                    [
                        _emit_attr_ints("axes", reduction_axes),
                        _emit_attr_int("keepdims", 0),
                    ],
                ))
            else:
                sum_name = f"kl_div_batch_sum_{suffix}"
                denominator_name = f"kl_div_batch_denominator_{suffix}"
                initializers.append(_emit_tensor_proto(
                    denominator_name,
                    _dtype_or_die(compute_dtype),
                    (),
                    _initializer_bytes_for_scalar(
                        float(contract.batch_denominator), compute_dtype
                    ),
                ))
                nodes.append(_emit_node(
                    [per_element_name],
                    [sum_name],
                    f"{nm}_batch_sum",
                    "ReduceSum",
                    [
                        _emit_attr_ints("axes", reduction_axes),
                        _emit_attr_int("keepdims", 0),
                    ],
                ))
                nodes.append(_emit_node(
                    [sum_name, denominator_name],
                    [compute_result_name],
                    f"{nm}_batchmean",
                    "Div",
                ))
            if compute_dtype != node.dtype:
                nodes.append(_emit_node(
                    [compute_result_name],
                    [out_name],
                    f"{nm}_output_cast",
                    "Cast",
                    [_emit_attr_int("to", _dtype_or_die(node.dtype))],
                ))
        elif node.op == OP_NLL_LOSS:
            contract = validate_nll_loss_contract(node)
            if contract.batch_rank != 0:
                raise OnnxUnmappableOp(
                    "export_inference: NLL_LOSS after vmap is not exportable "
                    "because ONNX fixes the class axis at 1"
                )
            attributes = [
                _emit_attr_string("reduction", contract.reduction),
                _emit_attr_int("ignore_index", contract.ignore_index),
            ]
            if len(contract.input_shape) == 1:
                suffix = next_node_id - 1
                axes_name = f"nll_loss_batch_axis_{suffix}"
                initializers.append(_emit_tensor_proto(
                    axes_name,
                    DT_INT64,
                    (1,),
                    _i64_initializer_for_shape((0,)),
                ))
                input_name = f"nll_loss_input_batched_{suffix}"
                target_name = f"nll_loss_target_batched_{suffix}"
                nodes.append(_emit_node(
                    [input_names[0], axes_name],
                    [input_name],
                    f"{nm}_unsqueeze_input",
                    "Unsqueeze",
                ))
                nodes.append(_emit_node(
                    [input_names[1], axes_name],
                    [target_name],
                    f"{nm}_unsqueeze_target",
                    "Unsqueeze",
                ))
                nll_inputs = [input_name, target_name] + input_names[2:]
                nll_output = (
                    f"nll_loss_batched_output_{suffix}"
                    if contract.reduction == "none"
                    else out_name
                )
                nodes.append(_emit_node(
                    nll_inputs,
                    [nll_output],
                    nm,
                    "NegativeLogLikelihoodLoss",
                    attributes,
                ))
                if contract.reduction == "none":
                    nodes.append(_emit_node(
                        [nll_output, axes_name],
                        [out_name],
                        f"{nm}_squeeze_output",
                        "Squeeze",
                    ))
            else:
                nodes.append(_emit_node(
                    input_names,
                    [out_name],
                    nm,
                    "NegativeLogLikelihoodLoss",
                    attributes,
                ))
        elif node.op == OP_CROSS_ENTROPY:
            contract = validate_cross_entropy_contract(node)
            if contract.batch_rank != 0:
                raise OnnxUnmappableOp(
                    "export_inference: CROSS_ENTROPY after vmap is not "
                    "exportable because ONNX fixes the class axis at 1"
                )
            if contract.target_mode != "indices":
                raise OnnxUnmappableOp(
                    "export_inference: CROSS_ENTROPY probability targets are "
                    "not representable by ONNX SoftmaxCrossEntropyLoss"
                )
            if contract.label_smoothing != 0.0:
                raise OnnxUnmappableOp(
                    "export_inference: CROSS_ENTROPY label_smoothing is not "
                    "representable by ONNX SoftmaxCrossEntropyLoss opset 17"
                )
            attributes = [
                _emit_attr_string("reduction", contract.reduction),
                _emit_attr_int("ignore_index", contract.ignore_index),
            ]
            if len(contract.input_shape) == 1:
                suffix = next_node_id - 1
                axes_name = f"cross_entropy_batch_axis_{suffix}"
                initializers.append(_emit_tensor_proto(
                    axes_name,
                    DT_INT64,
                    (1,),
                    _i64_initializer_for_shape((0,)),
                ))
                input_name = f"cross_entropy_input_batched_{suffix}"
                target_name = f"cross_entropy_target_batched_{suffix}"
                nodes.append(_emit_node(
                    [input_names[0], axes_name],
                    [input_name],
                    f"{nm}_unsqueeze_input",
                    "Unsqueeze",
                ))
                nodes.append(_emit_node(
                    [input_names[1], axes_name],
                    [target_name],
                    f"{nm}_unsqueeze_target",
                    "Unsqueeze",
                ))
                cross_entropy_inputs = [
                    input_name,
                    target_name,
                    *input_names[2:],
                ]
                cross_entropy_output = (
                    f"cross_entropy_batched_output_{suffix}"
                    if contract.reduction == "none"
                    else out_name
                )
                nodes.append(_emit_node(
                    cross_entropy_inputs,
                    [cross_entropy_output],
                    nm,
                    "SoftmaxCrossEntropyLoss",
                    attributes,
                ))
                if contract.reduction == "none":
                    nodes.append(_emit_node(
                        [cross_entropy_output, axes_name],
                        [out_name],
                        f"{nm}_squeeze_output",
                        "Squeeze",
                    ))
            else:
                nodes.append(_emit_node(
                    input_names,
                    [out_name],
                    nm,
                    "SoftmaxCrossEntropyLoss",
                    attributes,
                ))
        elif node.op == OP_CONCAT:
            axis, _, legacy_empty = validate_cat_contract(node)
            if node.dtype not in ("float32", "int32", "int64", "bool"):
                raise OnnxUnmappableOp(
                    f"export_inference: CONCAT dtype {node.dtype!r} is not exportable; "
                    "supported output dtypes are float32, int32, int64, and bool"
                )
            suffix = next_node_id - 1
            concat_inputs = []
            for index, (source, source_name, empty) in enumerate(
                zip(node.inputs, input_names, legacy_empty)
            ):
                if empty and len(node.shape) != 1:
                    continue
                concat_input = source_name
                if source.dtype != node.dtype:
                    cast_name = f"concat_cast_{suffix}_{index}"
                    nodes.append(_emit_node(
                        [source_name],
                        [cast_name],
                        f"{nm}_cast_{index}",
                        "Cast",
                        [_emit_attr_int("to", _dtype_or_die(node.dtype))],
                    ))
                    concat_input = cast_name
                concat_inputs.append(concat_input)
            nodes.append(_emit_node(
                concat_inputs,
                [out_name],
                nm,
                "Concat",
                [_emit_attr_int("axis", axis)],
            ))
        elif node.op == OP_STACK:
            axis = validate_stack_contract(node)
            if node.dtype not in ("float32", "int32", "int64", "bool"):
                raise OnnxUnmappableOp(
                    f"export_inference: STACK dtype {node.dtype!r} is not exportable; "
                    "supported output dtypes are float32, int32, int64, and bool"
                )
            suffix = next_node_id - 1
            axes_name = f"const_stack_axes_{suffix}"
            initializers.append(_emit_tensor_proto(
                axes_name,
                DT_INT64,
                (1,),
                _i64_initializer_for_shape((axis,)),
            ))
            stacked_inputs = []
            for index, (source, source_name) in enumerate(zip(node.inputs, input_names)):
                stack_input = source_name
                if source.dtype != node.dtype:
                    cast_name = f"stack_cast_{suffix}_{index}"
                    nodes.append(_emit_node(
                        [source_name],
                        [cast_name],
                        f"{nm}_cast_{index}",
                        "Cast",
                        [_emit_attr_int("to", _dtype_or_die(node.dtype))],
                    ))
                    stack_input = cast_name
                unsqueezed_name = f"stack_unsqueezed_{suffix}_{index}"
                nodes.append(_emit_node(
                    [stack_input, axes_name],
                    [unsqueezed_name],
                    f"{nm}_unsqueeze_{index}",
                    "Unsqueeze",
                ))
                stacked_inputs.append(unsqueezed_name)
            nodes.append(_emit_node(
                stacked_inputs,
                [out_name],
                nm,
                "Concat",
                [_emit_attr_int("axis", axis)],
            ))
        elif node.op == OP_PAD:
            pad_width, value = validate_pad_contract(node)
            if node.dtype not in ("float32", "int32", "int64"):
                raise OnnxUnmappableOp(
                    f"export_inference: PAD dtype {node.dtype!r} is not exportable; "
                    "supported output dtypes are float32, int32, and int64"
                )
            suffix = next_node_id - 1
            pads_name = f"const_pad_width_{suffix}"
            pads = tuple(pair[0] for pair in pad_width) + tuple(
                pair[1] for pair in pad_width
            )
            initializers.append(_emit_tensor_proto(
                pads_name,
                DT_INT64,
                (len(pads),),
                _i64_initializer_for_shape(pads),
            ))
            value_name = f"const_pad_value_{suffix}"
            initializers.append(_emit_tensor_proto(
                value_name,
                _dtype_or_die(node.dtype),
                (),
                _initializer_bytes_for_scalar(value, node.dtype),
            ))
            nodes.append(_emit_node(
                input_names + [pads_name, value_name],
                [out_name],
                nm,
                "Pad",
            ))
        elif node.op == OP_NARROW:
            validate_narrow_contract(node)
            raise OnnxUnmappableOp(
                "export_inference: gradient-only NARROW is not exportable in v0"
            )
        elif node.op == OP_CUMSUM:
            axis, reverse = validate_cumsum_contract(node)
            if node.dtype not in ("float32", "int32", "int64"):
                raise OnnxUnmappableOp(
                    f"export_inference: CUMSUM dtype {node.dtype!r} is not exportable; "
                    "supported output dtypes are float32, int32, and int64"
                )
            suffix = next_node_id - 1
            scan_input = input_names[0]
            if node.inputs[0].dtype != node.dtype:
                cast_name = f"cumsum_cast_{suffix}"
                nodes.append(_emit_node(
                    [scan_input],
                    [cast_name],
                    f"{nm}_cast",
                    "Cast",
                    [_emit_attr_int("to", _dtype_or_die(node.dtype))],
                ))
                scan_input = cast_name
            axis_name = f"const_cumsum_axis_{suffix}"
            initializers.append(_emit_tensor_proto(
                axis_name,
                DT_INT64,
                (),
                _i64_initializer_for_shape((axis,)),
            ))
            nodes.append(_emit_node(
                [scan_input, axis_name],
                [out_name],
                nm,
                "CumSum",
                [
                    _emit_attr_int("exclusive", 0),
                    _emit_attr_int("reverse", 1 if reverse else 0),
                ],
            ))
        elif node.op in (OP_TRIL, OP_TRIU):
            upper = node.op == OP_TRIU
            diagonal = (
                validate_triu_contract(node)
                if upper
                else validate_tril_contract(node)
            )
            _dtype_or_die(node.dtype)
            diagonal_name = f"const_{node.op.lower()}_diagonal_{next_node_id - 1}"
            initializers.append(_emit_tensor_proto(
                diagonal_name,
                DT_INT64,
                (),
                _i64_initializer_for_shape((diagonal,)),
            ))
            nodes.append(_emit_node(
                input_names + [diagonal_name],
                [out_name],
                nm,
                "Trilu",
                [_emit_attr_int("upper", int(upper))],
            ))
        elif node.op == OP_INDEX:
            axis = validate_gather_contract(node)
            _dtype_or_die(node.dtype)
            nodes.append(_emit_node(
                input_names,
                [out_name],
                nm,
                "GatherElements",
                [_emit_attr_int("axis", axis)],
            ))
        elif node.op == OP_SORT_INDICES:
            axis, descending, _ = validate_sort_indices_contract(node)
            source = node.inputs[0]
            if not source.shape or source.shape[axis] == 0:
                raise OnnxUnmappableOp(
                    "export_inference: SORT_INDICES requires a non-scalar, "
                    "nonempty selected axis for ONNX TopK"
                )
            if source.dtype not in ("float32", "int32", "int64"):
                raise OnnxUnmappableOp(
                    f"export_inference: sort source dtype {source.dtype!r} is not "
                    "exportable; supported dtypes are float32, int32, and int64"
                )
            suffix = next_node_id - 1
            k_name = f"const_sort_k_{suffix}"
            initializers.append(_emit_tensor_proto(
                k_name,
                DT_INT64,
                (1,),
                _i64_initializer_for_shape((source.shape[axis],)),
            ))
            unused_values_name = f"unused_sort_values_{suffix}"
            nodes.append(_emit_node(
                [input_names[0], k_name],
                [unused_values_name, out_name],
                nm,
                "TopK",
                [
                    _emit_attr_int("axis", axis),
                    _emit_attr_int("largest", 1 if descending else 0),
                    _emit_attr_int("sorted", 1),
                ],
            ))
        elif node.op == OP_SORT_VALUES:
            axis, _, _ = validate_sort_values_contract(node)
            source = node.inputs[0]
            if not source.shape or source.shape[axis] == 0:
                raise OnnxUnmappableOp(
                    "export_inference: SORT_VALUES requires a non-scalar, "
                    "nonempty selected axis for ONNX TopK/GatherElements"
                )
            if node.dtype not in ("float32", "int32", "int64"):
                raise OnnxUnmappableOp(
                    f"export_inference: SORT_VALUES dtype {node.dtype!r} is not "
                    "exportable; supported dtypes are float32, int32, and int64"
                )
            nodes.append(_emit_node(
                input_names,
                [out_name],
                nm,
                "GatherElements",
                [_emit_attr_int("axis", axis)],
            ))
        elif node.op == OP_TOPK_INDICES:
            axis, k, largest, sorted_output = validate_topk_indices_contract(node)
            source = node.inputs[0]
            if k == 0:
                raise OnnxUnmappableOp(
                    "export_inference: TOPK_INDICES with k=0 is not exportable "
                    "because ONNX TopK requires positive K"
                )
            if source.dtype not in ("float32", "int32", "int64"):
                raise OnnxUnmappableOp(
                    f"export_inference: topk source dtype {source.dtype!r} is not "
                    "exportable; supported dtypes are float32, int32, and int64"
                )
            suffix = next_node_id - 1
            k_name = f"const_topk_k_{suffix}"
            initializers.append(_emit_tensor_proto(
                k_name,
                DT_INT64,
                (1,),
                _i64_initializer_for_shape((k,)),
            ))
            unused_values_name = f"unused_topk_values_{suffix}"
            nodes.append(_emit_node(
                [input_names[0], k_name],
                [unused_values_name, out_name],
                nm,
                "TopK",
                [
                    _emit_attr_int("axis", axis),
                    _emit_attr_int("largest", 1 if largest else 0),
                    _emit_attr_int("sorted", 1 if sorted_output else 0),
                ],
            ))
        elif node.op == OP_TOPK_VALUES:
            axis, k, _, _ = validate_topk_values_contract(node)
            if k == 0:
                raise OnnxUnmappableOp(
                    "export_inference: TOPK_VALUES with k=0 is not exportable "
                    "because ONNX TopK requires positive K"
                )
            if node.dtype not in ("float32", "int32", "int64"):
                raise OnnxUnmappableOp(
                    f"export_inference: TOPK_VALUES dtype {node.dtype!r} is not "
                    "exportable; supported dtypes are float32, int32, and int64"
                )
            nodes.append(_emit_node(
                input_names,
                [out_name],
                nm,
                "GatherElements",
                [_emit_attr_int("axis", axis)],
            ))
        elif node.op == OP_SCATTER:
            axis = validate_scatter_contract(node)
            if node.dtype not in ("float32", "int32", "int64", "bool"):
                raise OnnxUnmappableOp(
                    f"export_inference: SCATTER dtype {node.dtype!r} is not "
                    "exportable; supported dtypes are float32, int32, int64, and bool"
                )
            updates_name = input_names[2]
            if node.inputs[2].shape == ():
                suffix = next_node_id - 1
                shape_name = f"const_scatter_shape_{suffix}"
                updates_name = f"expanded_scatter_source_{suffix}"
                initializers.append(_emit_tensor_proto(
                    shape_name,
                    DT_INT64,
                    (len(node.inputs[1].shape),),
                    _i64_initializer_for_shape(node.inputs[1].shape),
                ))
                nodes.append(_emit_node(
                    [input_names[2], shape_name],
                    [updates_name],
                    f"{nm}_expand_source",
                    "Expand",
                ))
            nodes.append(_emit_node(
                [input_names[0], input_names[1], updates_name],
                [out_name],
                nm,
                "ScatterElements",
                [_emit_attr_int("axis", axis)],
            ))
        elif node.op == OP_PROD:
            axes, keepdims, _ = validate_prod_contract(node)
            if node.dtype not in ("float32", "int32", "int64"):
                raise OnnxUnmappableOp(
                    f"export_inference: PROD dtype {node.dtype!r} is not exportable; "
                    "supported dtypes are float32, int32, and int64"
                )
            attrs = [
                _emit_attr_ints("axes", axes),
                _emit_attr_int("keepdims", 1 if keepdims else 0),
            ]
            nodes.append(_emit_node(input_names, [out_name], nm, "ReduceProd", attrs))
        elif node.op == OP_VAR:
            axes, correction, keepdims, _, reduced_elements = validate_var_contract(node)
            if node.dtype != "float32":
                raise OnnxUnmappableOp(
                    f"export_inference: VAR dtype {node.dtype!r} is not exportable; "
                    "supported dtype is float32"
                )
            suffix = next_node_id - 1
            mean_name = f"var_mean_{suffix}"
            centered_name = f"var_centered_{suffix}"
            squared_name = f"var_squared_{suffix}"
            sum_name = f"var_sum_{suffix}"
            denominator_name = f"const_var_denominator_{suffix}"
            axes_attr = _emit_attr_ints("axes", axes)
            initializers.append(_emit_tensor_proto(
                denominator_name,
                DT_FLOAT,
                (1,),
                _initializer_bytes_for_scalar(
                    float(max(0, reduced_elements - correction)),
                    node.dtype,
                ),
            ))
            nodes.extend([
                _emit_node(
                    input_names,
                    [mean_name],
                    f"{nm}_mean",
                    "ReduceMean",
                    [axes_attr, _emit_attr_int("keepdims", 1)],
                ),
                _emit_node(
                    [input_names[0], mean_name],
                    [centered_name],
                    f"{nm}_center",
                    "Sub",
                ),
                _emit_node(
                    [centered_name, centered_name],
                    [squared_name],
                    f"{nm}_square",
                    "Mul",
                ),
                _emit_node(
                    [squared_name],
                    [sum_name],
                    f"{nm}_sum",
                    "ReduceSum",
                    [axes_attr, _emit_attr_int("keepdims", 1 if keepdims else 0)],
                ),
                _emit_node(
                    [sum_name, denominator_name],
                    [out_name],
                    f"{nm}_divide",
                    "Div",
                ),
            ])
        elif node.op == OP_REPEAT:
            repeats, _ = validate_repeat_contract(node)
            _dtype_or_die(node.dtype)
            repeats_name = f"const_repeat_{next_node_id - 1}"
            initializers.append(_emit_tensor_proto(
                repeats_name,
                DT_INT64,
                (len(repeats),),
                _i64_initializer_for_shape(repeats),
            ))
            nodes.append(_emit_node(input_names + [repeats_name], [out_name], nm, "Tile"))
        elif node.op == OP_REPEAT_INTERLEAVE:
            repeats, axis = validate_repeat_interleave_contract(node)
            _dtype_or_die(node.dtype)
            suffix = next_node_id - 1
            axes_name = f"const_repeat_interleave_axes_{suffix}"
            repeats_name = f"const_repeat_interleave_repeats_{suffix}"
            shape_name = f"const_repeat_interleave_shape_{suffix}"
            unsqueezed_name = f"repeat_interleave_unsqueezed_{suffix}"
            tiled_name = f"repeat_interleave_tiled_{suffix}"
            tile_repeats = [1] * (len(node.inputs[0].shape) + 1)
            tile_repeats[axis + 1] = repeats
            for name, values in (
                (axes_name, (axis + 1,)),
                (repeats_name, tuple(tile_repeats)),
                (shape_name, node.shape),
            ):
                initializers.append(_emit_tensor_proto(
                    name,
                    DT_INT64,
                    (len(values),),
                    _i64_initializer_for_shape(values),
                ))
            nodes.append(_emit_node(
                input_names + [axes_name],
                [unsqueezed_name],
                f"{nm}_unsqueeze",
                "Unsqueeze",
            ))
            nodes.append(_emit_node(
                [unsqueezed_name, repeats_name],
                [tiled_name],
                f"{nm}_tile",
                "Tile",
            ))
            nodes.append(_emit_node(
                [tiled_name, shape_name],
                [out_name],
                f"{nm}_reshape",
                "Reshape",
            ))
        elif node.op == OP_CMP:
            cmp_op = node.arg["op"]
            onnx_op = _CMP_OP_MAP.get(cmp_op)
            if onnx_op is None:
                raise OnnxUnmappableOp(
                    f"export_inference: CMP op {cmp_op!r} is not exportable. "
                    f"Supported: {sorted(_CMP_OP_MAP)}"
                )
            nodes.append(_emit_node(input_names, [out_name], nm, onnx_op))
        else:
            raise OnnxUnmappableOp(
                f"export_inference: opcode {node.op!r} is not exportable in v0. "
                f"Supported ops: {sorted(set(_SIMPLE_OPS) | {OP_CAST, OP_RESHAPE, OP_PERMUTE, OP_REDUCE, OP_BROADCAST_TO, OP_CLAMP, OP_FLIP, OP_CUMSUM, OP_CONCAT, OP_STACK, OP_PAD, OP_SORT_INDICES, OP_SORT_VALUES, OP_TOPK_INDICES, OP_TOPK_VALUES, OP_SCATTER, OP_EINSUM, OP_L1_LOSS, OP_SMOOTH_L1_LOSS, OP_TRIL, OP_TRIU, OP_INDEX, OP_PROD, OP_VAR, OP_REPEAT, OP_REPEAT_INTERLEAVE, OP_CMP, OP_LOAD, OP_BUFFER, OP_CONST})}. "
                f"Unsupported tensor IR ops such as {OP_CONV1D!r}, "
                f"{OP_CONV1D_BACKWARD_INPUT!r}, "
                f"{OP_CONV1D_BACKWARD_WEIGHT!r}, "
                f"{OP_CONV1D_BACKWARD_BIAS!r}, {OP_CONV2D!r}, "
                f"{OP_CONV2D_BACKWARD_INPUT!r}, and "
                f"{OP_CONV2D_BACKWARD_WEIGHT!r}, and "
                f"{OP_CONV2D_BACKWARD_BIAS!r}, {OP_CONV_TRANSPOSE2D!r}, "
                f"{OP_CONV_TRANSPOSE2D_BACKWARD_INPUT!r}, "
                f"{OP_CONV_TRANSPOSE2D_BACKWARD_WEIGHT!r}, "
                f"{OP_CONV_TRANSPOSE2D_BACKWARD_BIAS!r}, {OP_CONV3D!r}, "
                f"{OP_CONV3D_BACKWARD_INPUT!r}, "
                f"{OP_CONV3D_BACKWARD_WEIGHT!r}, "
                f"{OP_CONV3D_BACKWARD_BIAS!r}, {OP_SGD_UPDATE!r}, "
                f"{OP_LAYER_NORM!r}, {OP_LAYER_NORM_BACKWARD_INPUT!r}, "
                f"{OP_LAYER_NORM_BACKWARD_WEIGHT!r}, "
                f"{OP_LAYER_NORM_BACKWARD_BIAS!r}, "
                f"{OP_ADAMW_UPDATE_M!r}, {OP_ADAMW_UPDATE_V!r}, and "
                f"{OP_ADAMW_UPDATE_PARAM!r}, {OP_ADAM_UPDATE_M!r}, "
                f"{OP_ADAM_UPDATE_V!r}, and {OP_ADAM_UPDATE_PARAM!r} "
                f"need explicit ONNX mappings "
                f"before export."
            )

    # Rename the root's output edge to `output_name`.
    root_internal = uop_to_name[id(root_uop)]
    # Re-emit the last node to rename its output, or add an Identity. Adding
    # Identity is cleaner (one fewer string-rewrite path).
    nodes.append(
        _emit_node([root_internal], [output_name], "output_alias", "Identity")
    )

    # graph.output ValueInfoProto.
    graph_outputs = [_emit_value_info(output_name, _dtype_or_die(root_uop.dtype), root_uop.shape)]

    # GraphProto: node=1 (repeated), name=2, initializer=5 (repeated),
    #             input=11 (repeated), output=12 (repeated).
    graph_parts = []
    for n in nodes:
        graph_parts.append(_emit_message(1, n))
    graph_parts.append(_emit_string(2, "graph"))
    for init in initializers:
        graph_parts.append(_emit_message(5, init))
    for inp in graph_inputs:
        graph_parts.append(_emit_message(11, inp))
    for out in graph_outputs:
        graph_parts.append(_emit_message(12, out))
    graph_proto = b"".join(graph_parts)

    # ModelProto: ir_version=1, producer_name=2, producer_version=3,
    #             opset_import=8 (repeated), graph=7.
    model_parts = [
        _emit_int64(1, 8),  # ir_version = 8 (matches opset 17 era)
        _emit_string(2, "browsergrad-jit"),
        _emit_string(3, "0.7.0"),
        _emit_message(7, graph_proto),
        _emit_message(8, _emit_opset_import("", opset_version)),
    ]
    return b"".join(model_parts)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _dtype_or_die(dt: str) -> int:
    onnx_dt = _DTYPE_TO_ONNX.get(dt)
    if onnx_dt is None:
        raise OnnxUnmappableOp(
            f"export_inference: dtype {dt!r} has no ONNX equivalent in v0. "
            f"Supported: {sorted(_DTYPE_TO_ONNX)}"
        )
    return onnx_dt


def _i64_initializer_for_shape(shape: Sequence[int]) -> bytes:
    """Pack a shape tuple as a little-endian int64 array for an
    ONNX shape initializer."""
    return b"".join(struct.pack("<q", int(d)) for d in shape)


def _initializer_bytes_for_scalar(value: float, dtype: str) -> bytes:
    import numpy as np
    return np.asarray([value], dtype=np.dtype(dtype)).tobytes()


def _np_array_for_const(value: Any, dtype: str) -> Any:
    import numpy as np
    arr = np.asarray(value, dtype=np.dtype(dtype))
    if arr.ndim == 0:
        # ONNX requires non-scalar Tensors for initializers; expand to 1-D.
        arr = arr.reshape((1,))
    return arr


__all__ = ["export_inference", "OnnxUnmappableOp"]
