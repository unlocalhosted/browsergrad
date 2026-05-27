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
  * 14 ops: ADD, MUL, DIV, NEG, EXP, LOG, MATMUL, REDUCE (sum/mean/max),
    RESHAPE, PERMUTE, CAST, WHERE, CMP (→Equal/Greater/Less),
    BROADCAST_TO (→Expand). Plus lifecycle (BUFFER/LOAD/CONST).
  * Opset 17 (axes as attribute on ReduceSum/Mean/Max — opset 18 made
    axes a runtime input, which would require initializer plumbing).
  * fp32 + int64 dtypes only.

Refusals (typed `OnnxUnmappableOp`):
  * OP_RANDOM (no runtime randomness in ONNX inference)
  * OP_CUSTOM (opaque)
  * OP_MASK, OP_INDEX, OP_SCATTER_ADD (initializer-tensor plumbing
    deferred to a follow-on)
  * OP_ISNAN, OP_SLICE, OP_PAD (same — needs initializer wiring or
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
    OP_ADD, OP_MUL, OP_DIV, OP_NEG, OP_EXP, OP_LOG, OP_CMP,
    OP_MATMUL, OP_REDUCE, OP_RESHAPE, OP_PERMUTE,
    OP_WHERE, OP_BROADCAST_TO,
)
from ._errors import JitError


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
DT_INT32 = 6
DT_INT64 = 7
DT_BOOL = 9


_DTYPE_TO_ONNX: Dict[str, int] = {
    "float32": DT_FLOAT,
    "int32": DT_INT32,
    "int64": DT_INT64,
    "bool": DT_BOOL,
}


# AttributeProto.AttributeType enum
AT_INT = 2
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
        nm = f"node_{next_node_id}_{node.op}"
        next_node_id += 1
        out_name = f"out_{next_node_id - 1}"
        uop_to_name[id(node)] = out_name

        input_names = [uop_to_name[id(inp)] for inp in node.inputs]

        if node.op in _SIMPLE_OPS:
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
            shape_arr = _i64_initializer_for_shape(node.arg["shape"])
            shape_const_name = f"const_expand_{next_node_id - 1}"
            initializers.append(
                _emit_tensor_proto(shape_const_name, DT_INT64,
                                   (len(node.arg["shape"]),), shape_arr)
            )
            nodes.append(_emit_node(input_names + [shape_const_name], [out_name], nm, "Expand"))
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
                f"Supported ops: {sorted(set(_SIMPLE_OPS) | {OP_CAST, OP_RESHAPE, OP_PERMUTE, OP_REDUCE, OP_BROADCAST_TO, OP_CMP, OP_LOAD, OP_BUFFER, OP_CONST})}"
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


def _np_array_for_const(value: Any, dtype: str) -> Any:
    import numpy as np
    arr = np.asarray(value, dtype=np.dtype(dtype))
    if arr.ndim == 0:
        # ONNX requires non-scalar Tensors for initializers; expand to 1-D.
        arr = arr.reshape((1,))
    return arr


__all__ = ["export_inference", "OnnxUnmappableOp"]
