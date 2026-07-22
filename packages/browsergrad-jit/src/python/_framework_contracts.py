"""Executable framework-operation contracts and public support reporting.

The package-owned JSON registry is the single machine-readable decision table
for migrated framework operations. Import validates its complete schema before
binding each record to an executable node validator. CPU, transform, export,
and plan boundaries call the bound validator; the public support table is a
detached projection of those same admitted records.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Tuple

import numpy as np

from ._errors import ShapeError


FRAMEWORK_OPERATION_SUPPORT_SCHEMA = "browsergrad.jit.framework-operation-contracts"
FRAMEWORK_OPERATION_SUPPORT_VERSION = (1, 0)
REPEAT_FACTOR_MAX = 1 << 30
REPEAT_RANK_MAX = 32
VAR_CORRECTION_MIN = -(1 << 31)
VAR_CORRECTION_MAX = (1 << 31) - 1
VARIADIC_INPUT_MAX = 1024
VARIADIC_OUTPUT_BYTE_MAX = 1 << 28
PAD_RANK_MAX = 32
PAD_OUTPUT_BYTE_MAX = 1 << 28
PAD_OUTPUT_EXTENT_MAX = PAD_OUTPUT_BYTE_MAX
SORT_RANK_MAX = 32
SORT_AXIS_MAX = 1 << 20
SORT_OUTPUT_BYTE_MAX = 1 << 28
SORT_OUTPUT_EXTENT_MAX = SORT_OUTPUT_BYTE_MAX
SORT_WORKSPACE_BYTE_MAX = 1 << 28
TOPK_RANK_MAX = SORT_RANK_MAX
TOPK_AXIS_MAX = SORT_AXIS_MAX
TOPK_OUTPUT_BYTE_MAX = SORT_OUTPUT_BYTE_MAX
TOPK_OUTPUT_EXTENT_MAX = SORT_OUTPUT_EXTENT_MAX
TOPK_WORKSPACE_BYTE_MAX = SORT_WORKSPACE_BYTE_MAX
SCATTER_RANK_MAX = 32
SCATTER_OUTPUT_BYTE_MAX = 1 << 28
SCATTER_OUTPUT_EXTENT_MAX = SCATTER_OUTPUT_BYTE_MAX
SCATTER_WORKSPACE_BYTE_MAX = 1 << 28
EINSUM_INPUT_MAX = 64
EINSUM_EQUATION_BYTE_MAX = 4096
EINSUM_RANK_MAX = 32
EINSUM_LABEL_MAX = 52
EINSUM_OUTPUT_BYTE_MAX = 1 << 28
EINSUM_OUTPUT_EXTENT_MAX = EINSUM_OUTPUT_BYTE_MAX
EINSUM_WORK_ELEMENT_MAX = 1 << 28
EINSUM_WORKSPACE_BYTE_MAX = 1 << 28
MASKED_FILL_CONTRACT_ID = "browsergrad.jit.framework.tensor.masked-fill.v1"
_REGISTRY_FILENAME = "framework-operation-contracts.v1.json"
_REGISTRY_BYTE_LIMIT = 32 * 1024
_ROOT_FIELDS = frozenset({"schema", "version", "operations"})
_VERSION_FIELDS = frozenset({"major", "minor"})
_OPERATION_FIELDS = frozenset({
    "contractId",
    "publicSurface",
    "opcode",
    "semanticState",
    "shapeContract",
    "dtypeContract",
    "decisions",
    "retiredOpaqueOperationId",
})
_DECISION_FIELDS = (
    "cpu",
    "closureAutograd",
    "symbolicVjp",
    "functionalGrad",
    "vmap",
    "onnxExport",
    "tensorPlan",
    "webgpu",
    "residency",
    "materialization",
)
_DECISION_FIELD_SET = frozenset(_DECISION_FIELDS)
_ENUMS = {
    "semanticState": frozenset({"typed"}),
    "shapeContract": frozenset({
        "preserve-unary-input",
        "preserve-single-axis-reverse",
        "preserve-batched-lower-triangular",
        "preserve-batched-upper-triangular",
        "preserve-single-axis-inclusive-scan",
        "variadic-existing-axis-concatenation-with-legacy-empty",
        "variadic-new-axis-stacking",
        "same-rank-index-shaped-gather",
        "preserve-source-with-broadcast-bool-mask",
        "trailing-dimension-constant-padding",
        "same-shape-axis-ordering",
        "selected-axis-becomes-exact-k",
        "same-rank-unique-index-overwrite-scatter",
        "canonical-general-einstein-contraction",
        "static-broadcast-with-existing-dim-minus-one",
        "selected-axis-times-repeat-count",
        "static-product-reduction",
        "static-variance-reduction",
        "tile-multipliers-with-left-rank-padding",
    }),
    "dtypeContract": frozenset({
        "preserve-floating-input",
        "preserve-input",
        "preserve-supported-input-with-exact-fill",
        "values-preserve-input-indices-int64",
        "preserve-input-require-bool-mask",
        "preserve-real-numeric-input",
        "preserve-source-require-int64-index",
        "preserve-target-require-int64-index-matching-source",
        "dimensioned-tensor-promotion-with-fp32-half-accumulator",
        "promote-integral-default-or-explicit-scan-dtype",
        "pytorch-dimensioned-tensor-promotion",
    }),
    "cpu": frozenset({
        "supported-numpy-dtype-preserving",
        "supported-numpy-owning-copy",
        "supported-numpy-owning-copy-with-range-check",
        "supported-numpy-owning-scan-copy",
        "supported-numpy-owning-concatenation-copy",
        "supported-numpy-owning-stack-copy",
        "supported-numpy-owning-constant-pad-copy",
        "supported-numpy-owning-stable-sort-indices",
        "supported-numpy-owning-sort-gather",
        "supported-numpy-owning-partial-topk-indices",
        "supported-numpy-owning-topk-gather",
        "supported-numpy-owning-unique-overwrite-scatter",
        "supported-numpy-owning-greedy-einsum",
    }),
    "closureAutograd": frozenset({
        "supported-cos-derivative",
        "supported-inclusive-bound-mask",
        "supported-involutive-flip",
        "supported-idempotent-triangular-selection",
        "supported-deterministic-scatter-add",
        "supported-negative-sin-derivative",
        "supported-sign-derivative",
        "supported-selected-axis-block-sum",
        "supported-mask-complement-selection",
        "supported-centered-variance-rule",
        "supported-tile-block-sum",
        "supported-unbroadcast-sum",
        "supported-zero-aware-product-rule",
        "supported-zero-derivative",
        "supported-opposite-direction-inclusive-scan-for-floating-source-and-output",
        "supported-static-axis-split",
        "supported-static-axis-index",
        "supported-static-interior-slice",
        "not-applicable-discrete-indices",
        "supported-permutation-scatter",
        "supported-unique-overwrite-scatter",
        "supported-general-einsum-vjp",
    }),
    "symbolicVjp": frozenset({
        "supported-cos-derivative",
        "supported-inclusive-bound-mask",
        "supported-involutive-flip",
        "supported-idempotent-triangular-selection",
        "supported-deterministic-scatter-add",
        "supported-negative-sin-derivative",
        "supported-sign-derivative",
        "supported-selected-axis-block-sum",
        "supported-mask-complement-selection",
        "supported-centered-variance-rule",
        "supported-tile-block-sum",
        "supported-unbroadcast-sum",
        "supported-zero-aware-product-rule",
        "supported-zero-derivative",
        "supported-opposite-direction-inclusive-scan-for-floating-source-and-output",
        "supported-static-axis-split",
        "supported-static-axis-index",
        "supported-static-interior-slice",
        "not-applicable-discrete-indices",
        "supported-permutation-scatter",
        "supported-unique-overwrite-scatter",
        "supported-general-einsum-vjp",
    }),
    "functionalGrad": frozenset({
        "supported-via-symbolic-vjp",
        "supported-for-floating-source-and-output-via-symbolic-vjp",
        "supported-for-floating-output-via-symbolic-vjp",
        "supported-for-floating-input-via-symbolic-vjp",
        "supported-for-floating-target-and-source-via-symbolic-vjp",
        "not-applicable-discrete-output",
    }),
    "vmap": frozenset({
        "supported-leading-batch-axis",
        "supported-leading-batch-axis-with-axis-shift",
        "supported-leading-batch-axis-with-index-axis-shift",
        "supported-leading-batch-axis-with-mask-broadcast",
        "supported-leading-batch-axis-preserve-matrix-axes",
        "supported-leading-batch-axis-with-unit-repeat",
        "supported-leading-batch-axis-with-scan-axis-shift",
        "supported-leading-batch-axis-with-axis-shift-and-captured-broadcast",
        "supported-leading-batch-axis-with-scatter-captured-broadcast",
        "supported-leading-batch-axis-with-einsum-captured-broadcast",
        "supported-leading-batch-axis-preserving-pad",
    }),
    "onnxExport": frozenset({
        "supported-opset17-clip-export-dtypes",
        "supported-opset17-direct-unary-export-dtypes",
        "supported-opset17-expand",
        "supported-opset17-gather-elements-float32-int32-int64-bool",
        "supported-opset17-where-float32-int32-int64-bool",
        "supported-opset17-reduce-prod-float32-int32-int64",
        "supported-opset17-variance-decomposition-float32",
        "supported-opset17-slice-float32-int32-int64-bool",
        "supported-opset17-trilu-float32-int32-int64-bool",
        "supported-opset17-tile-float32-int32-int64-bool",
        "supported-opset17-unsqueeze-tile-reshape-float32-int32-int64-bool",
        "supported-opset17-cumsum-with-cast-float32-int32-int64",
        "supported-opset17-concat-with-casts-float32-int32-int64-bool",
        "supported-opset17-unsqueeze-concat-with-casts-float32-int32-int64-bool",
        "supported-opset17-pad-float32-int32-int64",
        "supported-opset17-full-axis-topk-gather-float32-int32-int64",
        "supported-opset17-selected-k-topk-gather-float32-int32-int64",
        "supported-opset17-scatter-elements-float32-int32-int64-bool",
        "supported-opset17-resolved-einsum-numeric-dtypes",
    }),
    "tensorPlan": frozenset({
        "refused-negative-stride-profile",
        "refused-no-deterministic-index-lowering",
        "refused-no-portable-masked-selection",
        "refused-no-portable-triangular-selection",
        "refused-no-canonical-variadic-copy-lowering",
        "refused-no-canonical-pad-lowering",
        "refused-no-canonical-sort-lowering",
        "refused-no-canonical-topk-lowering",
        "refused-no-canonical-scatter-overwrite-lowering",
        "refused-no-canonical-contraction-lowering",
        "refused-no-canonical-tile-layout-profile",
        "refused-no-canonical-selected-axis-replication-profile",
        "refused-no-portable-lowering",
        "supported-primitive",
    }),
    "webgpu": frozenset({
        "profile-nonempty-f32-rank-at-most-4",
        "refused-no-deterministic-index-kernel",
        "refused-negative-stride-profile",
        "refused-no-canonical-tile-layout-profile",
        "refused-no-canonical-selected-axis-replication-profile",
        "refused-no-tensor-plan-kernel",
    }),
    "residency": frozenset({"host-materialized", "supported-materializing-and-resident"}),
    "materialization": frozenset({"cpu-owning-array", "cpu-owning-copy"}),
}

_REAL_NUMERIC_DTYPES = frozenset({
    "float16", "float32", "float64",
    "int8", "int16", "int32", "int64", "uint8",
})
_FLOATING_DTYPES = frozenset({"float16", "float32", "float64"})
_MASKED_FILL_DTYPES = _REAL_NUMERIC_DTYPES | frozenset({"bool"})
_TRIANGULAR_DTYPES = _MASKED_FILL_DTYPES
_CUMSUM_DTYPES = _MASKED_FILL_DTYPES
_VARIADIC_DTYPES = _MASKED_FILL_DTYPES
_VARIADIC_FLOATING_RANK = MappingProxyType({
    "float16": 0,
    "float32": 1,
    "float64": 2,
})
_VARIADIC_SIGNED_BITS = MappingProxyType({
    "int8": 8,
    "int16": 16,
    "int32": 32,
    "int64": 64,
})
_VARIADIC_DTYPE_BYTES = MappingProxyType({
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int16": 2,
    "float16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
})
_PAD_DTYPES = _VARIADIC_DTYPES
_SORT_DTYPES = _VARIADIC_DTYPES
_TOPK_DTYPES = _SORT_DTYPES
_SCATTER_DTYPES = _VARIADIC_DTYPES
_PAD_INTEGER_TYPES = (
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
_PAD_VALUE_TYPES = _PAD_INTEGER_TYPES + (
    float,
    np.float16,
    np.float32,
    np.float64,
)
_UNARY_DTYPE_PROFILES = MappingProxyType({
    "preserve-floating-input": (_FLOATING_DTYPES, "floating"),
    "preserve-real-numeric-input": (_REAL_NUMERIC_DTYPES, "real numeric"),
})


@dataclass(frozen=True, slots=True)
class FrameworkOperationContract:
    contract_id: str
    public_surface: str
    opcode: str
    semantic_state: str
    shape_contract: str
    dtype_contract: str
    decisions: Mapping[str, str]
    retired_opaque_operation_id: str


@dataclass(frozen=True, slots=True)
class _ExecutableFrameworkOperationContract:
    record: FrameworkOperationContract
    validator: Callable[[Any, FrameworkOperationContract], Any]


def _decode_closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in pairs:
        if key in decoded:
            raise ValueError(f"framework operation registry duplicates field {key!r}")
        decoded[key] = value
    return decoded


def _require_exact_fields(label: str, value: Any, fields: frozenset[str]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be a plain object")
    actual = frozenset(value)
    if actual != fields:
        raise ValueError(
            f"{label} fields changed; expected {sorted(fields)!r}, got {sorted(actual)!r}"
        )
    return value


def _require_string(label: str, value: Any) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{label} must be a non-empty canonical string")
    return value


def _require_enum(label: str, field: str, value: Any) -> str:
    string = _require_string(label, value)
    if string not in _ENUMS[field]:
        raise ValueError(f"{label} value {string!r} is not registered")
    return string


def _parse_registry_payload(payload: bytes) -> Tuple[FrameworkOperationContract, ...]:
    if type(payload) is not bytes:
        raise ValueError("framework operation registry payload must be immutable bytes")
    if not payload or len(payload) > _REGISTRY_BYTE_LIMIT:
        raise ValueError(
            f"framework operation registry must contain 1..{_REGISTRY_BYTE_LIMIT} bytes"
        )
    try:
        text = payload.decode("utf-8", errors="strict")
        root = json.loads(text, object_pairs_hook=_decode_closed_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("framework operation registry must be canonical UTF-8 JSON") from exc

    root = _require_exact_fields("framework operation registry", root, _ROOT_FIELDS)
    if root["schema"] != FRAMEWORK_OPERATION_SUPPORT_SCHEMA:
        raise ValueError("framework operation registry schema is not supported")
    version = _require_exact_fields(
        "framework operation registry version", root["version"], _VERSION_FIELDS
    )
    if (
        type(version["major"]) is not int
        or type(version["minor"]) is not int
        or (version["major"], version["minor"]) != FRAMEWORK_OPERATION_SUPPORT_VERSION
    ):
        raise ValueError("framework operation registry version must be exactly 1.0")
    operations = root["operations"]
    if type(operations) is not list or not operations:
        raise ValueError("framework operation registry operations must be a non-empty list")

    records = []
    contract_ids: set[str] = set()
    opcodes: set[str] = set()
    retired_ids: set[str] = set()
    for index, raw in enumerate(operations):
        label = f"framework operation registry operations[{index}]"
        raw = _require_exact_fields(label, raw, _OPERATION_FIELDS)
        decisions_raw = _require_exact_fields(
            f"{label}.decisions", raw["decisions"], _DECISION_FIELD_SET
        )
        contract_id = _require_string(f"{label}.contractId", raw["contractId"])
        public_surface = _require_string(f"{label}.publicSurface", raw["publicSurface"])
        opcode = _require_string(f"{label}.opcode", raw["opcode"])
        retired_id = _require_string(
            f"{label}.retiredOpaqueOperationId", raw["retiredOpaqueOperationId"]
        )
        if not contract_id.startswith("browsergrad.jit.framework."):
            raise ValueError(f"{label}.contractId is outside the BrowserGrad JIT namespace")
        if not retired_id.startswith("jit.custom."):
            raise ValueError(f"{label}.retiredOpaqueOperationId is outside the frozen namespace")
        for field_name, field_value, seen in (
            ("contractId", contract_id, contract_ids),
            ("opcode", opcode, opcodes),
            ("retiredOpaqueOperationId", retired_id, retired_ids),
        ):
            if field_value in seen:
                raise ValueError(f"{label}.{field_name} duplicates {field_value!r}")
            seen.add(field_value)
        decisions = MappingProxyType({
            field: _require_enum(
                f"{label}.decisions.{field}", field, decisions_raw[field]
            )
            for field in _DECISION_FIELDS
        })
        records.append(FrameworkOperationContract(
            contract_id=contract_id,
            public_surface=public_surface,
            opcode=opcode,
            semantic_state=_require_enum(
                f"{label}.semanticState", "semanticState", raw["semanticState"]
            ),
            shape_contract=_require_enum(
                f"{label}.shapeContract", "shapeContract", raw["shapeContract"]
            ),
            dtype_contract=_require_enum(
                f"{label}.dtypeContract", "dtypeContract", raw["dtypeContract"]
            ),
            decisions=decisions,
            retired_opaque_operation_id=retired_id,
        ))
    return tuple(records)


def _load_registry() -> Tuple[FrameworkOperationContract, ...]:
    path = Path(__file__).with_name(_REGISTRY_FILENAME)
    return _parse_registry_payload(path.read_bytes())


def _require_single_input_tuple(node: Any, opcode: str) -> tuple[Any, ...]:
    inputs = getattr(node, "inputs", ())
    if type(inputs) is not tuple:
        raise ShapeError(f"{opcode} inputs must be a plain tuple")
    if len(inputs) != 1:
        raise ShapeError(f"{opcode} requires exactly one input, got {len(inputs)}")
    return inputs


def _validate_broadcast_to(node: Any, contract: FrameworkOperationContract) -> Tuple[int, ...]:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = _require_single_input_tuple(node, contract.opcode)
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("BROADCAST_TO arg must be a plain dict")
    fields = set(arg)
    if "shape" not in fields or not fields.issubset({"shape", "vjp_of"}):
        raise ShapeError(
            "BROADCAST_TO arg fields must be exactly 'shape' plus optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("BROADCAST_TO arg.vjp_of must reference a UOp")
    target = arg["shape"]
    if type(target) is not tuple:
        raise ShapeError("BROADCAST_TO arg.shape must be a tuple")
    for axis, extent in enumerate(target):
        if type(extent) is not int or extent < 0:
            raise ShapeError(
                f"BROADCAST_TO arg.shape[{axis}] must be a non-negative int, got {extent!r}"
            )
    if target != getattr(node, "shape", None):
        raise ShapeError(
            f"BROADCAST_TO arg.shape {target} does not match node shape "
            f"{getattr(node, 'shape', None)}"
        )
    source = inputs[0]
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError(
            f"BROADCAST_TO must preserve dtype {getattr(source, 'dtype', None)!r}, "
            f"got {getattr(node, 'dtype', None)!r}"
        )
    source_shape = getattr(source, "shape", ())
    if type(source_shape) is not tuple:
        raise ShapeError("BROADCAST_TO input shape must be a tuple")
    if len(source_shape) > len(target):
        raise ShapeError(
            f"BROADCAST_TO cannot reduce rank {len(source_shape)} to {len(target)}"
        )
    leading = len(target) - len(source_shape)
    for axis, source_extent in enumerate(source_shape):
        target_extent = target[axis + leading]
        if source_extent != 1 and source_extent != target_extent:
            raise ShapeError(
                "BROADCAST_TO incompatible dimension "
                f"{axis}: {source_extent} -> {target_extent}"
            )
    return target


def _validate_typed_unary(node: Any, contract: FrameworkOperationContract) -> None:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = _require_single_input_tuple(node, contract.opcode)
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError(f"{contract.opcode} arg must be a plain dict")
    fields = set(arg)
    if not fields.issubset({"vjp_of"}):
        raise ShapeError(f"{contract.opcode} arg fields must be empty plus optional 'vjp_of'")
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError(f"{contract.opcode} arg.vjp_of must reference a UOp")
    source = inputs[0]
    if getattr(node, "shape", None) != getattr(source, "shape", None):
        raise ShapeError(f"{contract.opcode} must preserve its input shape")
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError(f"{contract.opcode} must preserve its input dtype")
    dtype_profile = _UNARY_DTYPE_PROFILES.get(contract.dtype_contract)
    if dtype_profile is None:
        raise ShapeError(
            f"{contract.opcode} has no unary dtype profile for {contract.dtype_contract!r}"
        )
    allowed_dtypes, dtype_label = dtype_profile
    if getattr(node, "dtype", None) not in allowed_dtypes:
        raise ShapeError(
            f"{contract.opcode} supports {dtype_label} dtypes only, got "
            f"{getattr(node, 'dtype', None)!r}"
        )


def _validate_clamp(
    node: Any,
    contract: FrameworkOperationContract,
) -> Tuple[Optional[float], Optional[float]]:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = _require_single_input_tuple(node, contract.opcode)
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("CLAMP arg must be a plain dict")
    fields = set(arg)
    if not {"min", "max"}.issubset(fields) or not fields.issubset({"min", "max", "vjp_of"}):
        raise ShapeError("CLAMP arg fields must be exactly 'min' and 'max' plus optional 'vjp_of'")
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("CLAMP arg.vjp_of must reference a UOp")
    source = inputs[0]
    if getattr(node, "shape", None) != getattr(source, "shape", None):
        raise ShapeError("CLAMP must preserve its input shape")
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError("CLAMP must preserve its input dtype")
    if getattr(node, "dtype", None) not in _FLOATING_DTYPES:
        raise ShapeError(
            f"CLAMP supports floating dtypes only, got {getattr(node, 'dtype', None)!r}"
        )
    bounds: list[Optional[float]] = []
    for name in ("min", "max"):
        bound = arg[name]
        if bound is not None and (type(bound) is not float or not math.isfinite(bound)):
            raise ShapeError(f"CLAMP arg.{name} must be None or a finite float")
        bounds.append(bound)
    minimum, maximum = bounds
    if minimum is None and maximum is None:
        raise ShapeError("CLAMP requires at least one bound")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ShapeError(f"CLAMP min {minimum} must be <= max {maximum}")
    return minimum, maximum


def _validate_flip(node: Any, contract: FrameworkOperationContract) -> int:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = _require_single_input_tuple(node, contract.opcode)
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("FLIP arg must be a plain dict")
    fields = set(arg)
    if "axis" not in fields or not fields.issubset({"axis", "vjp_of"}):
        raise ShapeError("FLIP arg fields must be exactly 'axis' plus optional 'vjp_of'")
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("FLIP arg.vjp_of must reference a UOp")
    source = inputs[0]
    if getattr(node, "shape", None) != getattr(source, "shape", None):
        raise ShapeError("FLIP must preserve its input shape")
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError("FLIP must preserve its input dtype")
    axis = arg["axis"]
    if type(axis) is not int:
        raise ShapeError("FLIP arg.axis must be a normalized integer")
    rank = len(getattr(source, "shape", ()))
    if axis < 0 or axis >= rank:
        raise ShapeError(f"FLIP arg.axis {axis} out of range for rank {rank}")
    return axis


def promote_variadic_dtypes(dtypes: tuple[str, ...], operation: str) -> str:
    if type(dtypes) is not tuple or not dtypes:
        raise ShapeError(f"{operation} dtype promotion requires a non-empty tuple")
    for index, dtype in enumerate(dtypes):
        if dtype not in _VARIADIC_DTYPES:
            raise ShapeError(
                f"{operation} input {index} has unsupported dtype {dtype!r}"
            )
    floating = [dtype for dtype in dtypes if dtype in _VARIADIC_FLOATING_RANK]
    if floating:
        return max(floating, key=_VARIADIC_FLOATING_RANK.__getitem__)
    signed = [dtype for dtype in dtypes if dtype in _VARIADIC_SIGNED_BITS]
    has_uint8 = "uint8" in dtypes
    if signed:
        widest = max(signed, key=_VARIADIC_SIGNED_BITS.__getitem__)
        if has_uint8 and _VARIADIC_SIGNED_BITS[widest] == 8:
            return "int16"
        return widest
    if has_uint8:
        return "uint8"
    return "bool"


_EINSUM_LABEL_ORDER = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_EINSUM_LABEL_RANK = MappingProxyType({
    label: index for index, label in enumerate(_EINSUM_LABEL_ORDER)
})
_EINSUM_ELLIPSIS = "..."


@dataclass(frozen=True, slots=True)
class EinsumContract:
    equation: str
    input_shapes: Tuple[Tuple[int, ...], ...]
    input_dtypes: Tuple[str, ...]
    input_labels: Tuple[Tuple[int, ...], ...]
    output_labels: Tuple[int, ...]
    label_sizes: Tuple[int, ...]
    batch_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    output_dtype: str
    batch_rank: int
    work_elements: int


def _parse_einsum_term(term: str, location: str) -> Tuple[str, ...]:
    tokens = []
    saw_ellipsis = False
    index = 0
    while index < len(term):
        label = term[index]
        if label in _EINSUM_LABEL_RANK:
            tokens.append(label)
            index += 1
            continue
        if term.startswith(_EINSUM_ELLIPSIS, index):
            if saw_ellipsis:
                raise ShapeError(f"einsum: {location} contains more than one ellipsis")
            tokens.append(_EINSUM_ELLIPSIS)
            saw_ellipsis = True
            index += 3
            continue
        raise ShapeError(
            f"einsum: invalid subscript {label!r} in {location}; "
            "expected ASCII letters or one ellipsis"
        )
    return tuple(tokens)


def _einsum_checked_product(
    extents: Tuple[int, ...],
    ceiling: int,
) -> int:
    product = 1
    for extent in extents:
        if extent == 0:
            return 0
        if product > ceiling // extent:
            return ceiling + 1
        product *= extent
    return product


def _merge_einsum_extent(
    label: str,
    current: int,
    incoming: int,
) -> int:
    if current != 1 and incoming != 1 and current != incoming:
        raise ShapeError(
            f"einsum: subscript {label!r} has incompatible broadcast "
            f"extents {current} and {incoming}"
        )
    if current == 1:
        return incoming
    return current


def infer_einsum_contract(
    inputs: tuple[Any, ...],
    equation: Any,
    batch_rank: int = 0,
) -> EinsumContract:
    if type(inputs) is not tuple or not inputs:
        raise ShapeError("einsum: operands must be a non-empty plain tuple")
    if len(inputs) > EINSUM_INPUT_MAX:
        raise ShapeError(
            f"einsum: operand count {len(inputs)} exceeds the "
            f"{EINSUM_INPUT_MAX}-operand ceiling"
        )
    if type(equation) is not str:
        raise ShapeError("einsum: equation must be an exact string")
    if len(equation.encode("utf-8")) > EINSUM_EQUATION_BYTE_MAX:
        raise ShapeError(
            f"einsum: equation exceeds the {EINSUM_EQUATION_BYTE_MAX}-byte ceiling"
        )
    if type(batch_rank) is not int or batch_rank < 0 or batch_rank > EINSUM_RANK_MAX:
        raise ShapeError(
            f"einsum: batch_rank must be in [0, {EINSUM_RANK_MAX}]"
        )

    compact = equation.replace(" ", "")
    if compact.count("->") > 1:
        raise ShapeError("einsum: equation may contain at most one '->'")
    explicit_output = "->" in compact
    if explicit_output:
        lhs, output_term = compact.split("->")
    else:
        lhs = compact
        output_term = ""
    input_terms = tuple(lhs.split(","))
    if len(input_terms) != len(inputs):
        raise ShapeError(
            f"einsum: equation has {len(input_terms)} input terms for "
            f"{len(inputs)} operands"
        )
    parsed_terms = tuple(
        _parse_einsum_term(term, f"operand {index}")
        for index, term in enumerate(input_terms)
    )

    input_shapes = []
    input_dtypes = []
    batch_shape: Tuple[int, ...] = ()
    base_shapes = []
    ellipsis_ranks = []
    for index, source in enumerate(inputs):
        shape = getattr(source, "shape", None)
        if type(shape) is not tuple:
            raise ShapeError(f"einsum: operand {index} shape must be a tuple")
        if len(shape) > EINSUM_RANK_MAX:
            raise ShapeError(
                f"einsum: operand {index} rank {len(shape)} exceeds the "
                f"{EINSUM_RANK_MAX}-rank ceiling"
            )
        for axis, extent in enumerate(shape):
            if type(extent) is not int or extent < 0:
                raise ShapeError(
                    f"einsum: operand {index} shape[{axis}] must be a "
                    "non-negative integer"
                )
            if extent > EINSUM_OUTPUT_EXTENT_MAX:
                raise ShapeError(
                    f"einsum: operand {index} extent {extent} exceeds the "
                    f"{EINSUM_OUTPUT_EXTENT_MAX}-extent ceiling"
                )
        if len(shape) < batch_rank:
            raise ShapeError(
                f"einsum: operand {index} rank {len(shape)} is smaller than "
                f"batch_rank {batch_rank}"
            )
        prefix = shape[:batch_rank]
        if index == 0:
            batch_shape = prefix
        elif prefix != batch_shape:
            raise ShapeError(
                f"einsum: mapped batch prefix {prefix} for operand {index} "
                f"does not match {batch_shape}"
            )
        base_shape = shape[batch_rank:]
        tokens = parsed_terms[index]
        explicit_labels = sum(token != _EINSUM_ELLIPSIS for token in tokens)
        has_ellipsis = _EINSUM_ELLIPSIS in tokens
        if has_ellipsis:
            if explicit_labels > len(base_shape):
                raise ShapeError(
                    f"einsum: operand {index} has {explicit_labels} explicit "
                    f"subscripts for rank {len(base_shape)}"
                )
            ellipsis_rank = len(base_shape) - explicit_labels
        else:
            if explicit_labels != len(base_shape):
                raise ShapeError(
                    f"einsum: operand {index} has {explicit_labels} subscripts "
                    f"for rank {len(base_shape)}"
                )
            ellipsis_rank = 0
        input_shapes.append(shape)
        base_shapes.append(base_shape)
        ellipsis_ranks.append(ellipsis_rank)
        input_dtypes.append(getattr(source, "dtype", None))

    ellipsis_rank = max(ellipsis_ranks, default=0)
    named_counts: dict[str, int] = {}
    named_sizes: dict[str, int] = {}
    ellipsis_sizes = [1] * ellipsis_rank
    expanded_keys = []
    for operand_index, (tokens, shape, local_ellipsis_rank) in enumerate(
        zip(parsed_terms, base_shapes, ellipsis_ranks)
    ):
        cursor = 0
        local_named_extents: dict[str, int] = {}
        operand_keys = []
        for token in tokens:
            if token == _EINSUM_ELLIPSIS:
                offset = ellipsis_rank - local_ellipsis_rank
                for local_axis in range(local_ellipsis_rank):
                    extent = shape[cursor + local_axis]
                    ellipsis_axis = offset + local_axis
                    ellipsis_sizes[ellipsis_axis] = _merge_einsum_extent(
                        f"ellipsis[{ellipsis_axis}]",
                        ellipsis_sizes[ellipsis_axis],
                        extent,
                    )
                    operand_keys.append(("ellipsis", ellipsis_axis))
                cursor += local_ellipsis_rank
                continue
            extent = shape[cursor]
            previous_local = local_named_extents.get(token)
            if previous_local is not None and previous_local != extent:
                raise ShapeError(
                    f"einsum: repeated subscript {token!r} in operand "
                    f"{operand_index} has unequal extents {previous_local} and {extent}"
                )
            local_named_extents[token] = extent
            named_counts[token] = named_counts.get(token, 0) + 1
            named_sizes[token] = _merge_einsum_extent(
                token,
                named_sizes.get(token, 1),
                extent,
            )
            operand_keys.append(("named", token))
            cursor += 1
        expanded_keys.append(tuple(operand_keys))

    has_input_ellipsis = any(
        _EINSUM_ELLIPSIS in tokens for tokens in parsed_terms
    )
    if explicit_output:
        parsed_output = _parse_einsum_term(output_term, "output")
        seen_output = set()
        output_keys = []
        for token in parsed_output:
            if token == _EINSUM_ELLIPSIS:
                if token in seen_output:
                    raise ShapeError("einsum: output contains more than one ellipsis")
                seen_output.add(token)
                output_keys.extend(
                    ("ellipsis", axis) for axis in range(ellipsis_rank)
                )
                continue
            if token in seen_output:
                raise ShapeError(
                    f"einsum: output subscript {token!r} appears more than once"
                )
            if token not in named_counts:
                raise ShapeError(
                    f"einsum: output subscript {token!r} does not appear in any operand"
                )
            seen_output.add(token)
            output_keys.append(("named", token))
        canonical_output = output_term
    else:
        implicit_labels = tuple(
            label
            for label in _EINSUM_LABEL_ORDER
            if named_counts.get(label, 0) == 1
        )
        output_keys = []
        if has_input_ellipsis:
            output_keys.extend(("ellipsis", axis) for axis in range(ellipsis_rank))
        output_keys.extend(("named", label) for label in implicit_labels)
        canonical_output = (
            (_EINSUM_ELLIPSIS if has_input_ellipsis else "")
            + "".join(implicit_labels)
        )

    named_key_order = [
        ("named", label)
        for label in _EINSUM_LABEL_ORDER
        if label in named_counts
    ]
    ellipsis_key_order = [
        ("ellipsis", axis) for axis in range(ellipsis_rank)
    ]
    key_order = tuple(named_key_order + ellipsis_key_order)
    if len(key_order) > EINSUM_LABEL_MAX:
        raise ShapeError(
            f"einsum: {len(key_order)} resolved labels exceed the "
            f"{EINSUM_LABEL_MAX}-label NumPy execution ceiling"
        )
    key_to_id = {key: index for index, key in enumerate(key_order)}
    input_labels = tuple(
        tuple(key_to_id[key] for key in keys) for keys in expanded_keys
    )
    output_labels = tuple(key_to_id[key] for key in output_keys)
    label_sizes = tuple(
        named_sizes[key[1]] if key[0] == "named" else ellipsis_sizes[key[1]]
        for key in key_order
    )
    base_output_shape = tuple(label_sizes[label] for label in output_labels)
    output_shape = batch_shape + base_output_shape
    if len(output_shape) > EINSUM_RANK_MAX:
        raise ShapeError(
            f"einsum: output rank {len(output_shape)} exceeds the "
            f"{EINSUM_RANK_MAX}-rank ceiling"
        )
    output_dtype = promote_variadic_dtypes(tuple(input_dtypes), "EINSUM")
    output_elements = _einsum_checked_product(
        output_shape,
        EINSUM_OUTPUT_BYTE_MAX,
    )
    output_bytes = output_elements * _VARIADIC_DTYPE_BYTES[output_dtype]
    if output_bytes > EINSUM_OUTPUT_BYTE_MAX:
        raise ShapeError(
            f"einsum: output requires {output_bytes} bytes, exceeding the "
            f"{EINSUM_OUTPUT_BYTE_MAX}-byte ceiling"
        )

    domain_elements = _einsum_checked_product(
        batch_shape + label_sizes,
        EINSUM_WORK_ELEMENT_MAX,
    )
    if domain_elements > EINSUM_WORK_ELEMENT_MAX // max(1, len(inputs)):
        work_elements = EINSUM_WORK_ELEMENT_MAX + 1
    else:
        work_elements = domain_elements * max(1, len(inputs))
    if work_elements > EINSUM_WORK_ELEMENT_MAX:
        raise ShapeError(
            f"einsum: projected contraction work exceeds the "
            f"{EINSUM_WORK_ELEMENT_MAX}-element ceiling"
        )

    accumulation_dtype = "float32" if output_dtype == "float16" else output_dtype
    accumulation_bytes = _VARIADIC_DTYPE_BYTES[accumulation_dtype]
    cast_bytes = 0
    for shape, dtype in zip(input_shapes, input_dtypes):
        if dtype != accumulation_dtype:
            cast_elements = _einsum_checked_product(
                shape,
                EINSUM_WORKSPACE_BYTE_MAX,
            )
            cast_bytes += cast_elements * accumulation_bytes
    gradient_bytes = 0
    for dtype, labels in zip(input_dtypes, input_labels):
        if dtype not in _FLOATING_DTYPES:
            continue
        unique_labels = tuple(dict.fromkeys(labels))
        unique_shape = batch_shape + tuple(label_sizes[label] for label in unique_labels)
        elements = _einsum_checked_product(
            unique_shape,
            EINSUM_WORKSPACE_BYTE_MAX,
        )
        gradient_bytes = max(gradient_bytes, elements * accumulation_bytes)
    contraction_bytes = domain_elements * accumulation_bytes
    workspace_bytes = output_bytes + cast_bytes + contraction_bytes + gradient_bytes
    if workspace_bytes > EINSUM_WORKSPACE_BYTE_MAX:
        raise ShapeError(
            f"einsum: projected output/cast/contraction/gradient workspace requires "
            f"{workspace_bytes} bytes, exceeding the "
            f"{EINSUM_WORKSPACE_BYTE_MAX}-byte ceiling"
        )

    return EinsumContract(
        equation=",".join(input_terms) + "->" + canonical_output,
        input_shapes=tuple(input_shapes),
        input_dtypes=tuple(input_dtypes),
        input_labels=input_labels,
        output_labels=output_labels,
        label_sizes=label_sizes,
        batch_shape=batch_shape,
        output_shape=output_shape,
        output_dtype=output_dtype,
        batch_rank=batch_rank,
        work_elements=work_elements,
    )


def _validate_einsum_runtime_arrays(
    contract: EinsumContract,
    arrays: tuple[np.ndarray, ...],
) -> None:
    if type(arrays) is not tuple or len(arrays) != len(contract.input_shapes):
        raise ShapeError("einsum: runtime arrays do not match the operand contract")
    for index, (array, shape, dtype) in enumerate(
        zip(arrays, contract.input_shapes, contract.input_dtypes)
    ):
        if type(array) is not np.ndarray:
            raise ShapeError(f"einsum: runtime operand {index} must be an ndarray")
        if tuple(array.shape) != shape or array.dtype.name != dtype:
            raise ShapeError(
                f"einsum: runtime operand {index} metadata "
                f"{tuple(array.shape)}/{array.dtype.name} does not match "
                f"{shape}/{dtype}"
            )


def execute_einsum_arrays(
    contract: EinsumContract,
    arrays: tuple[np.ndarray, ...],
) -> np.ndarray:
    _validate_einsum_runtime_arrays(contract, arrays)
    compute_dtype = "float32" if contract.output_dtype == "float16" else contract.output_dtype
    arguments = []
    for array, labels in zip(arrays, contract.input_labels):
        runtime_labels = list(labels)
        if contract.batch_rank:
            runtime_labels.insert(0, Ellipsis)
        arguments.extend([
            array.astype(np.dtype(compute_dtype), copy=False),
            runtime_labels,
        ])
    output_labels = list(contract.output_labels)
    if contract.batch_rank:
        output_labels.insert(0, Ellipsis)
    arguments.append(output_labels)
    result = np.einsum(*arguments, optimize="greedy")
    reshaped = np.asarray(
        result,
        dtype=np.dtype(contract.output_dtype),
    ).reshape(contract.output_shape)
    return np.array(reshaped, dtype=np.dtype(contract.output_dtype), copy=True)


def execute_einsum_vjp_array(
    contract: EinsumContract,
    operand_index: int,
    dy: np.ndarray,
    arrays: tuple[np.ndarray, ...],
) -> np.ndarray:
    _validate_einsum_runtime_arrays(contract, arrays)
    if type(operand_index) is not int or operand_index < 0 or operand_index >= len(arrays):
        raise ShapeError("EINSUM_VJP operand index is outside the input range")
    target_dtype = contract.input_dtypes[operand_index]
    if target_dtype not in _FLOATING_DTYPES:
        raise ShapeError("EINSUM_VJP target must have a floating dtype")
    if type(dy) is not np.ndarray:
        raise ShapeError("EINSUM_VJP upstream gradient must be an ndarray")
    if tuple(dy.shape) != contract.output_shape or dy.dtype.name != contract.output_dtype:
        raise ShapeError(
            "EINSUM_VJP upstream gradient metadata does not match the forward output"
        )

    compute_dtype = "float32" if contract.output_dtype == "float16" else contract.output_dtype
    target_labels = contract.input_labels[operand_index]
    unique_target_labels = tuple(dict.fromkeys(target_labels))
    present_labels = set(contract.output_labels)
    for index, labels in enumerate(contract.input_labels):
        if index != operand_index:
            present_labels.update(labels)
    requested_labels = tuple(
        label for label in unique_target_labels if label in present_labels
    )
    arguments = []
    dy_labels = list(contract.output_labels)
    if contract.batch_rank:
        dy_labels.insert(0, Ellipsis)
    arguments.extend([
        dy.astype(np.dtype(compute_dtype), copy=False),
        dy_labels,
    ])
    for index, (array, labels) in enumerate(zip(arrays, contract.input_labels)):
        if index == operand_index:
            continue
        runtime_labels = list(labels)
        if contract.batch_rank:
            runtime_labels.insert(0, Ellipsis)
        arguments.extend([
            array.astype(np.dtype(compute_dtype), copy=False),
            runtime_labels,
        ])
    result_labels = list(requested_labels)
    if contract.batch_rank:
        result_labels.insert(0, Ellipsis)
    arguments.append(result_labels)
    contracted = np.einsum(*arguments, optimize="greedy")

    requested = set(requested_labels)
    expanded_shape = contract.batch_shape + tuple(
        contract.label_sizes[label] if label in requested else 1
        for label in unique_target_labels
    )
    full_unique_shape = contract.batch_shape + tuple(
        contract.label_sizes[label] for label in unique_target_labels
    )
    gradient = np.asarray(contracted, dtype=np.dtype(compute_dtype)).reshape(expanded_shape)
    gradient = np.broadcast_to(gradient, full_unique_shape)

    target_base_shape = contract.input_shapes[operand_index][contract.batch_rank:]
    target_label_extents: dict[int, int] = {}
    for label, extent in zip(target_labels, target_base_shape):
        target_label_extents.setdefault(label, extent)
    for position, label in enumerate(unique_target_labels):
        target_extent = target_label_extents[label]
        if target_extent == 1 and contract.label_sizes[label] != 1:
            gradient = gradient.sum(
                axis=contract.batch_rank + position,
                keepdims=True,
                dtype=np.dtype(compute_dtype),
            )
    gradient = np.asarray(gradient, dtype=np.dtype(target_dtype))
    if len(unique_target_labels) == len(target_labels):
        return np.array(
            gradient,
            dtype=np.dtype(target_dtype),
            copy=True,
        ).reshape(contract.input_shapes[operand_index])

    output = np.zeros(
        contract.input_shapes[operand_index],
        dtype=np.dtype(target_dtype),
    )
    unique_positions = {
        label: position for position, label in enumerate(unique_target_labels)
    }
    grids = []
    unique_rank = len(unique_target_labels)
    for label in target_labels:
        position = unique_positions[label]
        extent = target_label_extents[label]
        grid_shape = [1] * unique_rank
        grid_shape[position] = extent
        grids.append(np.arange(extent, dtype=np.intp).reshape(tuple(grid_shape)))
    if contract.batch_rank:
        batch_elements = _einsum_checked_product(
            contract.batch_shape,
            EINSUM_WORK_ELEMENT_MAX,
        )
        output_flat = output.reshape((batch_elements,) + target_base_shape)
        gradient_flat = gradient.reshape(
            (batch_elements,) + tuple(
                target_label_extents[label] for label in unique_target_labels
            )
        )
        for batch_index in range(batch_elements):
            output_flat[batch_index][tuple(grids)] = gradient_flat[batch_index]
    else:
        output[tuple(grids)] = gradient
    return output


def einsum_onnx_equation(contract: EinsumContract) -> str:
    resolved_count = contract.batch_rank + len(contract.label_sizes)
    if resolved_count > 26:
        raise ShapeError(
            f"einsum: ONNX export requires at most 26 resolved labels, got "
            f"{resolved_count}"
        )
    labels = "abcdefghijklmnopqrstuvwxyz"
    batch_prefix = labels[:contract.batch_rank]
    offset = contract.batch_rank
    input_terms = []
    for input_labels in contract.input_labels:
        input_terms.append(
            batch_prefix
            + "".join(labels[offset + label] for label in input_labels)
        )
    output_term = batch_prefix + "".join(
        labels[offset + label] for label in contract.output_labels
    )
    return ",".join(input_terms) + "->" + output_term


def _variadic_input_metadata(
    inputs: tuple[Any, ...],
    operation: str,
) -> Tuple[Tuple[Tuple[int, ...], ...], str]:
    if type(inputs) is not tuple:
        raise ShapeError(f"{operation} inputs must be a plain tuple")
    if not inputs:
        raise ShapeError(f"{operation} requires at least one input")
    if len(inputs) > VARIADIC_INPUT_MAX:
        raise ShapeError(
            f"{operation} input count {len(inputs)} exceeds the "
            f"{VARIADIC_INPUT_MAX}-input ceiling"
        )
    shapes = []
    dtypes = []
    for index, source in enumerate(inputs):
        shape = getattr(source, "shape", None)
        if type(shape) is not tuple:
            raise ShapeError(f"{operation} input {index} shape must be a tuple")
        for axis, extent in enumerate(shape):
            if type(extent) is not int or extent < 0:
                raise ShapeError(
                    f"{operation} input {index} shape[{axis}] must be a "
                    "non-negative integer"
                )
        shapes.append(shape)
        dtypes.append(getattr(source, "dtype", None))
    return tuple(shapes), promote_variadic_dtypes(tuple(dtypes), operation)


def _validate_variadic_output_resource(
    shape: Tuple[int, ...],
    dtype: str,
    operation: str,
) -> None:
    elements = 1
    for extent in shape:
        elements *= extent
    output_bytes = elements * _VARIADIC_DTYPE_BYTES[dtype]
    if output_bytes > VARIADIC_OUTPUT_BYTE_MAX:
        raise ShapeError(
            f"{operation} output requires {output_bytes} bytes, exceeding the "
            f"{VARIADIC_OUTPUT_BYTE_MAX}-byte ceiling"
        )


def infer_cat_contract(
    inputs: tuple[Any, ...],
    axis: int,
) -> Tuple[int, Tuple[int, ...], str, Tuple[int, ...], Tuple[bool, ...]]:
    if type(axis) is not int:
        raise ShapeError("CONCAT arg.axis must be a normalized integer")
    shapes, output_dtype = _variadic_input_metadata(inputs, "CONCAT")
    legacy_empty = tuple(shape == (0,) for shape in shapes)

    substantive = [shape for shape, empty in zip(shapes, legacy_empty) if not empty]
    reference = substantive[0] if substantive else (0,)
    rank = len(reference)
    if rank == 0:
        raise ShapeError("CONCAT does not accept scalar inputs")
    if axis < 0 or axis >= rank:
        raise ShapeError(f"CONCAT arg.axis {axis} out of range for rank {rank}")

    output_shape = list(reference)
    output_shape[axis] = 0
    sizes = []
    for index, (shape, empty) in enumerate(zip(shapes, legacy_empty)):
        if empty:
            sizes.append(0)
            continue
        if len(shape) != rank:
            raise ShapeError(
                f"CONCAT input {index} rank {len(shape)} does not match rank {rank}"
            )
        for shape_axis, (actual, expected) in enumerate(zip(shape, reference)):
            if shape_axis != axis and actual != expected:
                raise ShapeError(
                    f"CONCAT input {index} shape {shape} does not match {reference} "
                    f"outside axis {axis}"
                )
        sizes.append(shape[axis])
        output_shape[axis] += shape[axis]

    output_shape = tuple(output_shape)
    _validate_variadic_output_resource(output_shape, output_dtype, "CONCAT")
    return (
        axis,
        output_shape,
        output_dtype,
        tuple(sizes),
        legacy_empty,
    )


def infer_stack_contract(
    inputs: tuple[Any, ...],
    axis: int,
) -> Tuple[int, Tuple[int, ...], str]:
    if type(axis) is not int:
        raise ShapeError("STACK arg.axis must be a normalized integer")
    shapes, output_dtype = _variadic_input_metadata(inputs, "STACK")
    reference = shapes[0]
    rank = len(reference)
    if axis < 0 or axis > rank:
        raise ShapeError(f"STACK arg.axis {axis} out of range for output rank {rank + 1}")
    for index, shape in enumerate(shapes[1:], start=1):
        if shape != reference:
            raise ShapeError(
                f"STACK input {index} shape {shape} does not match {reference}"
            )
    output_shape = reference[:axis] + (len(inputs),) + reference[axis:]
    _validate_variadic_output_resource(output_shape, output_dtype, "STACK")
    return axis, output_shape, output_dtype


def _pad_source_metadata(source: Any) -> Tuple[Tuple[int, ...], str]:
    shape = getattr(source, "shape", None)
    if type(shape) is not tuple:
        raise ShapeError("PAD source shape must be a tuple")
    if len(shape) > PAD_RANK_MAX:
        raise ShapeError(
            f"PAD source rank {len(shape)} exceeds the {PAD_RANK_MAX}-rank ceiling"
        )
    for axis, extent in enumerate(shape):
        if type(extent) is not int or extent < 0:
            raise ShapeError(
                f"PAD source shape[{axis}] must be a non-negative integer"
            )
    dtype = getattr(source, "dtype", None)
    if dtype not in _PAD_DTYPES:
        raise ShapeError(f"PAD source dtype {dtype!r} is not supported")
    return shape, dtype


def normalize_pad_value(value: Any, dtype: str) -> Any:
    if value is None:
        value = False if dtype == "bool" else 0
    if dtype == "bool":
        if type(value) not in (bool, np.bool_):
            raise ShapeError(
                "pad: a boolean input requires a built-in or NumPy boolean fill value"
            )
        return bool(value)
    if type(value) not in _PAD_VALUE_TYPES:
        raise ShapeError(
            "pad: value must be a built-in or NumPy real scalar, "
            f"got {type(value).__name__}"
        )
    if dtype.startswith("float"):
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = float(np.asarray(value, dtype=np.dtype(dtype)).item())
        if not math.isfinite(normalized):
            raise ShapeError("pad: floating fill value must remain finite in the input dtype")
        return 0.0 if normalized == 0.0 else normalized
    if type(value) in (float, np.float16, np.float32, np.float64):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ShapeError(
                f"pad: value {numeric!r} is not an exact finite integer for {dtype}"
            )
        normalized = int(numeric)
    else:
        normalized = int(value)
    bounds = np.iinfo(np.dtype(dtype))
    if normalized < int(bounds.min) or normalized > int(bounds.max):
        raise ShapeError(f"pad: value {normalized} is out of range for {dtype}")
    return normalized


def _pad_output_shape(
    source_shape: Tuple[int, ...],
    pad_width: Tuple[Tuple[int, int], ...],
    dtype: str,
) -> Tuple[int, ...]:
    output_shape = []
    for axis, (extent, (lower, upper)) in enumerate(zip(source_shape, pad_width)):
        output_extent = extent + lower + upper
        if output_extent > PAD_OUTPUT_EXTENT_MAX:
            raise ShapeError(
                f"PAD output extent {output_extent} on axis {axis} exceeds the "
                f"{PAD_OUTPUT_EXTENT_MAX}-element per-axis ceiling"
            )
        output_shape.append(output_extent)
    canonical_shape = tuple(output_shape)
    elements = 1
    for extent in canonical_shape:
        elements *= extent
    output_bytes = elements * _VARIADIC_DTYPE_BYTES[dtype]
    if output_bytes > PAD_OUTPUT_BYTE_MAX:
        raise ShapeError(
            f"PAD output requires {output_bytes} bytes, exceeding the "
            f"{PAD_OUTPUT_BYTE_MAX}-byte ceiling"
        )
    return canonical_shape


def normalize_pad_request(
    source: Any,
    pad: Any,
    mode: Any,
    value: Any,
) -> Tuple[Tuple[Tuple[int, int], ...], Any, Tuple[int, ...]]:
    source_shape, dtype = _pad_source_metadata(source)
    if type(mode) is not str:
        raise ShapeError(f"pad: mode must be a string, got {type(mode).__name__}")
    if mode != "constant":
        raise ShapeError(f"pad: mode {mode!r} is not supported; expected 'constant'")
    if type(pad) not in (tuple, list):
        raise ShapeError("pad: pad must be a plain tuple or list")
    if len(pad) % 2 != 0:
        raise ShapeError(f"pad: pad length must be even, got {len(pad)}")
    pair_count = len(pad) // 2
    if pair_count > len(source_shape):
        raise ShapeError(
            f"pad: {pair_count} padded dimensions exceed input rank {len(source_shape)}"
        )
    pad_width = [(0, 0)] * len(source_shape)
    for index in range(pair_count):
        lower_raw = pad[2 * index]
        upper_raw = pad[2 * index + 1]
        if type(lower_raw) not in _PAD_INTEGER_TYPES or type(upper_raw) not in _PAD_INTEGER_TYPES:
            raise ShapeError(
                f"pad: pair {index} must contain built-in or NumPy integer scalars"
            )
        lower = int(lower_raw)
        upper = int(upper_raw)
        if lower < 0 or upper < 0:
            raise ShapeError("pad: negative padding is outside the constant-pad v1 profile")
        pad_width[len(source_shape) - 1 - index] = (lower, upper)
    canonical = tuple(pad_width)
    normalized_value = normalize_pad_value(value, dtype)
    return canonical, normalized_value, _pad_output_shape(source_shape, canonical, dtype)


def infer_pad_contract(
    source: Any,
    pad_width: Any,
    mode: Any,
    value: Any,
) -> Tuple[Tuple[Tuple[int, int], ...], Any, Tuple[int, ...]]:
    source_shape, dtype = _pad_source_metadata(source)
    if type(mode) is not str or mode != "constant":
        raise ShapeError("PAD arg.mode must be exactly 'constant'")
    if type(pad_width) is not tuple or len(pad_width) != len(source_shape):
        raise ShapeError("PAD arg.pad_width must be one canonical pair per source axis")
    canonical = []
    for axis, pair in enumerate(pad_width):
        if type(pair) is not tuple or len(pair) != 2:
            raise ShapeError(f"PAD arg.pad_width[{axis}] must be an exact pair")
        lower, upper = pair
        if type(lower) is not int or type(upper) is not int or lower < 0 or upper < 0:
            raise ShapeError(
                f"PAD arg.pad_width[{axis}] must contain non-negative normalized integers"
            )
        canonical.append((lower, upper))
    canonical_pad = tuple(canonical)
    normalized_value = normalize_pad_value(value, dtype)
    if type(value) is not type(normalized_value) or value != normalized_value:
        raise ShapeError("PAD arg.value must be canonical for the source dtype")
    return (
        canonical_pad,
        normalized_value,
        _pad_output_shape(source_shape, canonical_pad, dtype),
    )


def _sort_source_metadata(source: Any) -> Tuple[Tuple[int, ...], str]:
    shape = getattr(source, "shape", None)
    if type(shape) is not tuple:
        raise ShapeError("sort: source shape must be a tuple")
    if len(shape) > SORT_RANK_MAX:
        raise ShapeError(
            f"sort: source rank {len(shape)} exceeds the {SORT_RANK_MAX}-rank ceiling"
        )
    elements = 1
    for axis, extent in enumerate(shape):
        if type(extent) is not int or extent < 0:
            raise ShapeError(
                f"sort: source shape[{axis}] must be a non-negative integer"
            )
        if extent > SORT_OUTPUT_EXTENT_MAX:
            raise ShapeError(
                f"sort: source extent {extent} on axis {axis} exceeds the "
                f"{SORT_OUTPUT_EXTENT_MAX}-element per-axis ceiling"
            )
        elements *= extent
    dtype = getattr(source, "dtype", None)
    if dtype not in _SORT_DTYPES:
        raise ShapeError(f"sort: source dtype {dtype!r} is not supported")
    output_bytes = elements * (_VARIADIC_DTYPE_BYTES[dtype] + 8)
    if output_bytes > SORT_OUTPUT_BYTE_MAX:
        raise ShapeError(
            f"sort: paired outputs require {output_bytes} bytes, exceeding the "
            f"{SORT_OUTPUT_BYTE_MAX}-byte ceiling"
        )
    workspace_bytes = elements * 24
    if workspace_bytes > SORT_WORKSPACE_BYTE_MAX:
        raise ShapeError(
            f"sort: conservative ordering workspace requires {workspace_bytes} bytes, "
            f"exceeding the {SORT_WORKSPACE_BYTE_MAX}-byte ceiling"
        )
    return shape, dtype


def normalize_sort_request(
    source: Any,
    dim: Any,
    descending: Any,
    stable: Any,
) -> Tuple[int, bool, bool]:
    shape, _ = _sort_source_metadata(source)
    if type(dim) not in _PAD_INTEGER_TYPES:
        raise ShapeError(
            f"sort: dim must be a built-in or NumPy integer scalar, got {type(dim).__name__}"
        )
    if type(descending) is not bool:
        raise ShapeError(
            f"sort: descending must be a boolean, got {type(descending).__name__}"
        )
    if type(stable) is not bool:
        raise ShapeError(f"sort: stable must be a boolean, got {type(stable).__name__}")
    raw_axis = int(dim)
    if not shape:
        if raw_axis not in (-1, 0):
            raise ShapeError(f"sort: dim {raw_axis} out of range for a scalar input")
        return 0, descending, stable
    axis = raw_axis + len(shape) if raw_axis < 0 else raw_axis
    if axis < 0 or axis >= len(shape):
        raise ShapeError(f"sort: dim {raw_axis} out of range for rank {len(shape)}")
    if shape[axis] > SORT_AXIS_MAX:
        raise ShapeError(
            f"sort: selected axis extent {shape[axis]} exceeds the "
            f"{SORT_AXIS_MAX}-element sorting ceiling"
        )
    return axis, descending, stable


def infer_sort_contract(
    source: Any,
    axis: Any,
    descending: Any,
    stable: Any,
) -> Tuple[int, bool, bool]:
    shape, _ = _sort_source_metadata(source)
    if type(axis) is not int:
        raise ShapeError("sort: IR axis must be a normalized integer")
    if type(descending) is not bool or type(stable) is not bool:
        raise ShapeError("sort: IR descending and stable flags must be booleans")
    if not shape:
        if axis != 0:
            raise ShapeError("sort: scalar IR axis must be canonical zero")
    elif axis < 0 or axis >= len(shape):
        raise ShapeError(f"sort: IR axis {axis} out of range for rank {len(shape)}")
    elif shape[axis] > SORT_AXIS_MAX:
        raise ShapeError(
            f"sort: selected axis extent {shape[axis]} exceeds the "
            f"{SORT_AXIS_MAX}-element sorting ceiling"
        )
    return axis, descending, stable


def stable_sort_indices_array(
    array: np.ndarray,
    axis: int,
    descending: bool,
) -> np.ndarray:
    """Return deterministic stable indices without dtype-changing negation."""
    if array.ndim == 0:
        return np.asarray(0, dtype=np.int64)
    if descending:
        reversed_array = np.flip(array, axis=axis)
        reversed_indices = np.argsort(reversed_array, axis=axis, kind="stable")
        descending_indices = np.flip(reversed_indices, axis=axis)
        result = array.shape[axis] - 1 - descending_indices
    else:
        result = np.argsort(array, axis=axis, kind="stable")
    return np.array(result, dtype=np.int64, copy=True)


def _topk_source_metadata(source: Any) -> Tuple[Tuple[int, ...], str, int]:
    shape = getattr(source, "shape", None)
    if type(shape) is not tuple:
        raise ShapeError("topk: source shape must be a tuple")
    if not shape:
        raise ShapeError("topk: scalar inputs are outside the typed topk v1 profile")
    if len(shape) > TOPK_RANK_MAX:
        raise ShapeError(
            f"topk: source rank {len(shape)} exceeds the {TOPK_RANK_MAX}-rank ceiling"
        )
    elements = 1
    for axis, extent in enumerate(shape):
        if type(extent) is not int or extent < 0:
            raise ShapeError(
                f"topk: source shape[{axis}] must be a non-negative integer"
            )
        if extent > TOPK_OUTPUT_EXTENT_MAX:
            raise ShapeError(
                f"topk: source extent {extent} on axis {axis} exceeds the "
                f"{TOPK_OUTPUT_EXTENT_MAX}-element per-axis ceiling"
            )
        elements *= extent
    dtype = getattr(source, "dtype", None)
    if dtype not in _TOPK_DTYPES:
        raise ShapeError(f"topk: source dtype {dtype!r} is not supported")
    return shape, dtype, elements


def _topk_output_contract(
    shape: Tuple[int, ...],
    dtype: str,
    elements: int,
    axis: int,
    k: int,
) -> Tuple[int, ...]:
    if shape[axis] > TOPK_AXIS_MAX:
        raise ShapeError(
            f"topk: selected axis extent {shape[axis]} exceeds the "
            f"{TOPK_AXIS_MAX}-element selection ceiling"
        )
    if k < 0 or k > shape[axis]:
        raise ShapeError(
            f"topk: k must be between 0 and selected-axis extent {shape[axis]}, got {k}"
        )
    output_shape_list = list(shape)
    output_shape_list[axis] = k
    output_shape = tuple(output_shape_list)
    output_elements = 1
    for extent in output_shape:
        output_elements *= extent
    paired_output_bytes = output_elements * (_VARIADIC_DTYPE_BYTES[dtype] + 8)
    if paired_output_bytes > TOPK_OUTPUT_BYTE_MAX:
        raise ShapeError(
            f"topk: paired outputs require {paired_output_bytes} bytes, exceeding the "
            f"{TOPK_OUTPUT_BYTE_MAX}-byte ceiling"
        )
    # NumPy argpartition owns one full-size int64 permutation. Sorted output
    # additionally owns selected values and bounded int64 ordering/remap buffers.
    workspace_bytes = elements * 8 + output_elements * (
        _VARIADIC_DTYPE_BYTES[dtype] + 32
    )
    if workspace_bytes > TOPK_WORKSPACE_BYTE_MAX:
        raise ShapeError(
            f"topk: conservative selection workspace requires {workspace_bytes} bytes, "
            f"exceeding the {TOPK_WORKSPACE_BYTE_MAX}-byte ceiling"
        )
    return output_shape


def normalize_topk_request(
    source: Any,
    k: Any,
    dim: Any,
    largest: Any,
    sorted_output: Any,
) -> Tuple[int, int, bool, bool, Tuple[int, ...]]:
    shape, dtype, elements = _topk_source_metadata(source)
    if type(k) not in _PAD_INTEGER_TYPES:
        raise ShapeError(
            f"topk: k must be a built-in or NumPy integer scalar, got {type(k).__name__}"
        )
    if dim is None:
        raw_axis = -1
    elif type(dim) in _PAD_INTEGER_TYPES:
        raw_axis = int(dim)
    else:
        raise ShapeError(
            f"topk: dim must be None or a built-in/NumPy integer scalar, got "
            f"{type(dim).__name__}"
        )
    if type(largest) is not bool:
        raise ShapeError(
            f"topk: largest must be a boolean, got {type(largest).__name__}"
        )
    if type(sorted_output) is not bool:
        raise ShapeError(
            f"topk: sorted must be a boolean, got {type(sorted_output).__name__}"
        )
    axis = raw_axis + len(shape) if raw_axis < 0 else raw_axis
    if axis < 0 or axis >= len(shape):
        raise ShapeError(f"topk: dim {raw_axis} out of range for rank {len(shape)}")
    k_value = int(k)
    output_shape = _topk_output_contract(shape, dtype, elements, axis, k_value)
    return axis, k_value, largest, sorted_output, output_shape


def infer_topk_contract(
    source: Any,
    axis: Any,
    k: Any,
    largest: Any,
    sorted_output: Any,
) -> Tuple[int, int, bool, bool, Tuple[int, ...]]:
    shape, dtype, elements = _topk_source_metadata(source)
    if type(axis) is not int or axis < 0 or axis >= len(shape):
        raise ShapeError(f"topk: IR axis {axis!r} is invalid for rank {len(shape)}")
    if type(k) is not int:
        raise ShapeError("topk: IR k must be a normalized integer")
    if type(largest) is not bool or type(sorted_output) is not bool:
        raise ShapeError("topk: IR largest and sorted flags must be booleans")
    output_shape = _topk_output_contract(shape, dtype, elements, axis, k)
    return axis, k, largest, sorted_output, output_shape


def partial_topk_indices_array(
    array: np.ndarray,
    axis: int,
    k: int,
    largest: bool,
    sorted_output: bool,
) -> np.ndarray:
    """Select top-k with bounded partial ordering and no dtype-changing negation."""
    output_shape = list(array.shape)
    output_shape[axis] = k
    if k == 0:
        return np.empty(tuple(output_shape), dtype=np.int64)
    axis_extent = array.shape[axis]
    kth = axis_extent - k if largest else k - 1
    partition = np.argpartition(array, kth=kth, axis=axis)
    slices = [slice(None)] * array.ndim
    slices[axis] = slice(axis_extent - k, None) if largest else slice(0, k)
    selected = np.array(partition[tuple(slices)], dtype=np.int64, copy=True)
    if sorted_output:
        selected_values = np.take_along_axis(array, selected, axis=axis)
        local_order = stable_sort_indices_array(
            selected_values,
            axis,
            descending=largest,
        )
        selected = np.take_along_axis(selected, local_order, axis=axis)
    return np.array(selected, dtype=np.int64, copy=True)


def _validate_cat(
    node: Any,
    contract: FrameworkOperationContract,
) -> Tuple[int, Tuple[int, ...], Tuple[bool, ...]]:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("CONCAT arg must be a plain dict")
    fields = set(arg)
    if "axis" not in fields or not fields.issubset({"axis", "vjp_of"}):
        raise ShapeError("CONCAT arg fields must be exactly 'axis' plus optional 'vjp_of'")
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("CONCAT arg.vjp_of must reference a UOp")
    _, output_shape, output_dtype, sizes, legacy_empty = infer_cat_contract(
        getattr(node, "inputs", ()),
        arg["axis"],
    )
    if getattr(node, "shape", None) != output_shape:
        raise ShapeError(
            f"CONCAT output shape must be {output_shape}, got {getattr(node, 'shape', None)}"
        )
    if getattr(node, "dtype", None) != output_dtype:
        raise ShapeError(
            f"CONCAT output dtype must be {output_dtype!r}, got {getattr(node, 'dtype', None)!r}"
        )
    return arg["axis"], sizes, legacy_empty


def _validate_stack(
    node: Any,
    contract: FrameworkOperationContract,
) -> int:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("STACK arg must be a plain dict")
    fields = set(arg)
    if "axis" not in fields or not fields.issubset({"axis", "vjp_of"}):
        raise ShapeError("STACK arg fields must be exactly 'axis' plus optional 'vjp_of'")
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("STACK arg.vjp_of must reference a UOp")
    axis, output_shape, output_dtype = infer_stack_contract(
        getattr(node, "inputs", ()),
        arg["axis"],
    )
    if getattr(node, "shape", None) != output_shape:
        raise ShapeError(
            f"STACK output shape must be {output_shape}, got {getattr(node, 'shape', None)}"
        )
    if getattr(node, "dtype", None) != output_dtype:
        raise ShapeError(
            f"STACK output dtype must be {output_dtype!r}, got {getattr(node, 'dtype', None)!r}"
        )
    return axis


def _validate_pad(
    node: Any,
    contract: FrameworkOperationContract,
) -> Tuple[Tuple[Tuple[int, int], ...], Any]:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = _require_single_input_tuple(node, "PAD")
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("PAD arg must be a plain dict")
    fields = set(arg)
    required = {"pad_width", "mode", "value"}
    if not required.issubset(fields) or not fields.issubset(required | {"vjp_of"}):
        raise ShapeError(
            "PAD arg fields must be exactly 'pad_width', 'mode', and 'value' "
            "plus optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("PAD arg.vjp_of must reference a UOp")
    pad_width, value, output_shape = infer_pad_contract(
        inputs[0],
        arg["pad_width"],
        arg["mode"],
        arg["value"],
    )
    if getattr(node, "shape", None) != output_shape:
        raise ShapeError(
            f"PAD output shape must be {output_shape}, got {getattr(node, 'shape', None)}"
        )
    if getattr(node, "dtype", None) != getattr(inputs[0], "dtype", None):
        raise ShapeError("PAD must preserve the source dtype")
    return pad_width, value


def _sort_arg(node: Any, opcode: str) -> Tuple[int, bool, bool]:
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError(f"{opcode} arg must be a plain dict")
    required = {"axis", "descending", "stable"}
    fields = set(arg)
    if not required.issubset(fields) or not fields.issubset(required | {"vjp_of"}):
        raise ShapeError(
            f"{opcode} arg fields must be exactly 'axis', 'descending', and "
            "'stable' plus optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError(f"{opcode} arg.vjp_of must reference a UOp")
    inputs = getattr(node, "inputs", None)
    if type(inputs) is not tuple or not inputs:
        raise ShapeError(f"{opcode} must have a source input")
    return infer_sort_contract(
        inputs[0],
        arg["axis"],
        arg["descending"],
        arg["stable"],
    )


def _validate_sort_indices(
    node: Any,
    contract: FrameworkOperationContract,
) -> Tuple[int, bool, bool]:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = _require_single_input_tuple(node, "SORT_INDICES")
    source = inputs[0]
    normalized = _sort_arg(node, "SORT_INDICES")
    if getattr(node, "shape", None) != getattr(source, "shape", None):
        raise ShapeError("SORT_INDICES must preserve the source shape")
    if getattr(node, "dtype", None) != "int64":
        raise ShapeError("SORT_INDICES output dtype must be int64")
    return normalized


def _validate_sort_values(
    node: Any,
    contract: FrameworkOperationContract,
) -> Tuple[int, bool, bool]:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = getattr(node, "inputs", None)
    if type(inputs) is not tuple or len(inputs) != 2:
        raise ShapeError("SORT_VALUES must have source and SORT_INDICES inputs")
    source, indices = inputs
    normalized = _sort_arg(node, "SORT_VALUES")
    if getattr(node, "shape", None) != getattr(source, "shape", None):
        raise ShapeError("SORT_VALUES must preserve the source shape")
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError("SORT_VALUES must preserve the source dtype")
    if getattr(indices, "op", None) != "SORT_INDICES":
        raise ShapeError("SORT_VALUES second input must be typed SORT_INDICES")
    index_inputs = getattr(indices, "inputs", None)
    if type(index_inputs) is not tuple or len(index_inputs) != 1 or index_inputs[0] is not source:
        raise ShapeError("SORT_VALUES indices must derive from the exact same source")
    index_normalized = validate_sort_indices_contract(indices)
    if index_normalized != normalized:
        raise ShapeError("SORT_VALUES and SORT_INDICES ordering arguments must match")
    return normalized


def _topk_arg(
    node: Any,
    opcode: str,
) -> Tuple[int, int, bool, bool, Tuple[int, ...]]:
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError(f"{opcode} arg must be a plain dict")
    required = {"axis", "k", "largest", "sorted"}
    fields = set(arg)
    if not required.issubset(fields) or not fields.issubset(required | {"vjp_of"}):
        raise ShapeError(
            f"{opcode} arg fields must be exactly 'axis', 'k', 'largest', and "
            "'sorted' plus optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError(f"{opcode} arg.vjp_of must reference a UOp")
    inputs = getattr(node, "inputs", None)
    if type(inputs) is not tuple or not inputs:
        raise ShapeError(f"{opcode} must have a source input")
    return infer_topk_contract(
        inputs[0],
        arg["axis"],
        arg["k"],
        arg["largest"],
        arg["sorted"],
    )


def _validate_topk_indices(
    node: Any,
    contract: FrameworkOperationContract,
) -> Tuple[int, int, bool, bool]:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = _require_single_input_tuple(node, "TOPK_INDICES")
    axis, k, largest, sorted_output, output_shape = _topk_arg(
        node,
        "TOPK_INDICES",
    )
    if getattr(node, "shape", None) != output_shape:
        raise ShapeError(
            f"TOPK_INDICES output shape must be {output_shape}, got "
            f"{getattr(node, 'shape', None)}"
        )
    if getattr(node, "dtype", None) != "int64":
        raise ShapeError("TOPK_INDICES output dtype must be int64")
    return axis, k, largest, sorted_output


def _validate_topk_values(
    node: Any,
    contract: FrameworkOperationContract,
) -> Tuple[int, int, bool, bool]:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = getattr(node, "inputs", None)
    if type(inputs) is not tuple or len(inputs) != 2:
        raise ShapeError("TOPK_VALUES must have source and TOPK_INDICES inputs")
    source, indices = inputs
    axis, k, largest, sorted_output, output_shape = _topk_arg(
        node,
        "TOPK_VALUES",
    )
    if getattr(node, "shape", None) != output_shape:
        raise ShapeError(
            f"TOPK_VALUES output shape must be {output_shape}, got "
            f"{getattr(node, 'shape', None)}"
        )
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError("TOPK_VALUES must preserve the source dtype")
    if getattr(indices, "op", None) != "TOPK_INDICES":
        raise ShapeError("TOPK_VALUES second input must be typed TOPK_INDICES")
    index_inputs = getattr(indices, "inputs", None)
    if type(index_inputs) is not tuple or len(index_inputs) != 1 or index_inputs[0] is not source:
        raise ShapeError("TOPK_VALUES indices must derive from the exact same source")
    index_normalized = validate_topk_indices_contract(indices)
    if index_normalized != (axis, k, largest, sorted_output):
        raise ShapeError("TOPK_VALUES and TOPK_INDICES selection arguments must match")
    return axis, k, largest, sorted_output


def _validate_narrow(node: Any) -> Tuple[int, int, int]:
    inputs = _require_single_input_tuple(node, "NARROW")
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("NARROW arg must be a plain dict")
    fields = set(arg)
    if not {"axis", "start", "length"}.issubset(fields) or not fields.issubset(
        {"axis", "start", "length", "vjp_of"}
    ):
        raise ShapeError(
            "NARROW arg fields must be exactly 'axis', 'start', and 'length' "
            "plus optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("NARROW arg.vjp_of must reference a UOp")
    source = inputs[0]
    source_shape = getattr(source, "shape", None)
    if type(source_shape) is not tuple or not source_shape:
        raise ShapeError("NARROW requires an input with rank at least one")
    axis = arg["axis"]
    start = arg["start"]
    length = arg["length"]
    if type(axis) is not int or axis < 0 or axis >= len(source_shape):
        raise ShapeError(f"NARROW arg.axis {axis!r} is invalid for rank {len(source_shape)}")
    if type(start) is not int or start < 0:
        raise ShapeError("NARROW arg.start must be a non-negative integer")
    if type(length) is not int or length < 0:
        raise ShapeError("NARROW arg.length must be a non-negative integer")
    if start + length > source_shape[axis]:
        raise ShapeError(
            f"NARROW range [{start}, {start + length}) exceeds axis extent {source_shape[axis]}"
        )
    expected_shape = list(source_shape)
    expected_shape[axis] = length
    if getattr(node, "shape", None) != tuple(expected_shape):
        raise ShapeError(f"NARROW output shape must be {tuple(expected_shape)}")
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError("NARROW must preserve source dtype")
    return axis, start, length


def _validate_cumsum(
    node: Any,
    contract: FrameworkOperationContract,
) -> Tuple[int, bool]:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = _require_single_input_tuple(node, contract.opcode)
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("CUMSUM arg must be a plain dict")
    fields = set(arg)
    if not {"axis", "reverse"}.issubset(fields) or not fields.issubset(
        {"axis", "reverse", "vjp_of"}
    ):
        raise ShapeError(
            "CUMSUM arg fields must be exactly 'axis' and 'reverse' plus optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("CUMSUM arg.vjp_of must reference a UOp")
    source = inputs[0]
    source_shape = getattr(source, "shape", None)
    if type(source_shape) is not tuple:
        raise ShapeError("CUMSUM source shape must be a tuple")
    if not source_shape:
        raise ShapeError("CUMSUM requires an input with rank at least one")
    if getattr(node, "shape", None) != source_shape:
        raise ShapeError("CUMSUM must preserve its input shape")
    source_dtype = getattr(source, "dtype", None)
    output_dtype = getattr(node, "dtype", None)
    if source_dtype not in _CUMSUM_DTYPES:
        raise ShapeError(f"CUMSUM does not support source dtype {source_dtype!r}")
    if output_dtype not in _CUMSUM_DTYPES:
        raise ShapeError(f"CUMSUM does not support output dtype {output_dtype!r}")
    axis = arg["axis"]
    reverse = arg["reverse"]
    if type(axis) is not int:
        raise ShapeError("CUMSUM arg.axis must be a normalized integer")
    if axis < 0 or axis >= len(source_shape):
        raise ShapeError(
            f"CUMSUM arg.axis {axis} out of range for rank {len(source_shape)}"
        )
    if type(reverse) is not bool:
        raise ShapeError("CUMSUM arg.reverse must be a boolean")
    return axis, reverse


def _validate_triangular(
    node: Any,
    contract: FrameworkOperationContract,
    *,
    upper: bool,
) -> int:
    opcode = contract.opcode
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = _require_single_input_tuple(node, opcode)
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError(f"{opcode} arg must be a plain dict")
    fields = set(arg)
    if "diagonal" not in fields or not fields.issubset({"diagonal", "vjp_of"}):
        raise ShapeError(
            f"{opcode} arg fields must be exactly 'diagonal' plus optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError(f"{opcode} arg.vjp_of must reference a UOp")
    source = inputs[0]
    source_shape = getattr(source, "shape", None)
    if type(source_shape) is not tuple or len(source_shape) < 2:
        raise ShapeError(f"{opcode} requires an input with rank at least two")
    if getattr(node, "shape", None) != source_shape:
        raise ShapeError(f"{opcode} must preserve its input shape")
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError(f"{opcode} must preserve its input dtype")
    if getattr(node, "dtype", None) not in _TRIANGULAR_DTYPES:
        raise ShapeError(
            f"{opcode} does not support dtype {getattr(node, 'dtype', None)!r}"
        )
    diagonal = arg["diagonal"]
    if type(diagonal) is not int:
        raise ShapeError(f"{opcode} arg.diagonal must be a normalized integer")
    rows, columns = source_shape[-2:]
    if rows == 0 or columns == 0:
        minimum, maximum = 0, 0
    elif upper:
        minimum, maximum = 1 - rows, columns
    else:
        minimum, maximum = -rows, columns - 1
    if diagonal < minimum or diagonal > maximum:
        raise ShapeError(
            f"{opcode} arg.diagonal {diagonal} is outside canonical range "
            f"[{minimum}, {maximum}] for matrix shape {(rows, columns)}"
        )
    return diagonal


def _validate_tril(node: Any, contract: FrameworkOperationContract) -> int:
    return _validate_triangular(node, contract, upper=False)


def _validate_triu(node: Any, contract: FrameworkOperationContract) -> int:
    return _validate_triangular(node, contract, upper=True)


def _infer_scatter_contract(
    target: Any,
    index: Any,
    source: Any,
    axis: Any,
) -> int:
    target_shape = getattr(target, "shape", None)
    index_shape = getattr(index, "shape", None)
    source_shape = getattr(source, "shape", None)
    if type(target_shape) is not tuple or type(index_shape) is not tuple:
        raise ShapeError("scatter: target and index shapes must be tuples")
    rank = len(target_shape)
    if rank == 0 or rank > SCATTER_RANK_MAX:
        raise ShapeError(
            f"scatter: target rank must be in [1, {SCATTER_RANK_MAX}], got {rank}"
        )
    if len(index_shape) != rank:
        raise ShapeError(
            "scatter: target and index must have the same nonzero rank, "
            f"got {rank} and {len(index_shape)}"
        )
    if type(axis) is not int or axis < 0 or axis >= rank:
        raise ShapeError(
            f"scatter: normalized dim must be in [0, {rank}), got {axis!r}"
        )
    target_dtype = getattr(target, "dtype", None)
    if target_dtype not in _SCATTER_DTYPES:
        raise ShapeError(f"scatter: target dtype {target_dtype!r} is not supported")
    if getattr(index, "dtype", None) != "int64":
        raise ShapeError(
            f"scatter: index dtype must be 'int64', got {getattr(index, 'dtype', None)!r}"
        )
    if getattr(source, "dtype", None) != target_dtype:
        raise ShapeError("scatter: tensor source dtype must equal the target dtype")
    if source_shape not in ((), index_shape):
        raise ShapeError(
            "scatter: source must be a canonical scalar or have exactly the index shape"
        )
    target_elements = 1
    for dimension, extent in enumerate(target_shape):
        if type(extent) is not int or extent < 0:
            raise ShapeError(
                f"scatter: target shape[{dimension}] must be a non-negative integer"
            )
        if extent > SCATTER_OUTPUT_EXTENT_MAX:
            raise ShapeError(
                f"scatter: target extent {extent} on axis {dimension} exceeds the "
                f"{SCATTER_OUTPUT_EXTENT_MAX}-element per-axis ceiling"
            )
        target_elements *= extent
    index_elements = 1
    for dimension, (target_extent, index_extent) in enumerate(
        zip(target_shape, index_shape)
    ):
        if type(index_extent) is not int or index_extent < 0:
            raise ShapeError(
                f"scatter: index shape[{dimension}] must be a non-negative integer"
            )
        if index_extent > target_extent:
            qualifier = "selected" if dimension == axis else "non-scatter"
            raise ShapeError(
                f"scatter: index extent {index_extent} exceeds target extent "
                f"{target_extent} at {qualifier} dimension {dimension}; the typed "
                "overwrite profile requires unique destinations"
            )
        index_elements *= index_extent
    output_bytes = target_elements * _VARIADIC_DTYPE_BYTES[target_dtype]
    if output_bytes > SCATTER_OUTPUT_BYTE_MAX:
        raise ShapeError(
            f"scatter: output requires {output_bytes} bytes, exceeding the "
            f"{SCATTER_OUTPUT_BYTE_MAX}-byte ceiling"
        )
    # CPU owns one output copy and, for duplicate rejection, one sorted int64
    # index copy plus one adjacent-equality boolean temporary.
    workspace_bytes = output_bytes + index_elements * 9
    if workspace_bytes > SCATTER_WORKSPACE_BYTE_MAX:
        raise ShapeError(
            f"scatter: conservative overwrite workspace requires {workspace_bytes} "
            f"bytes, exceeding the {SCATTER_WORKSPACE_BYTE_MAX}-byte ceiling"
        )
    return axis


def normalize_scatter_request(
    target: Any,
    index: Any,
    source: Any,
    dim: Any,
) -> int:
    target_shape = getattr(target, "shape", None)
    if type(target_shape) is not tuple:
        raise ShapeError("scatter: target shape must be a tuple")
    if type(dim) not in _PAD_INTEGER_TYPES:
        raise ShapeError(
            "scatter: dim must be a built-in or NumPy integer scalar, "
            f"got {type(dim).__name__}"
        )
    raw_axis = int(dim)
    axis = raw_axis + len(target_shape) if raw_axis < 0 else raw_axis
    if axis < 0 or axis >= len(target_shape):
        raise ShapeError(
            f"scatter: dim {raw_axis} out of range for rank {len(target_shape)}"
        )
    return _infer_scatter_contract(target, index, source, axis)


def infer_scatter_contract(
    target: Any,
    index: Any,
    source: Any,
    axis: Any,
) -> int:
    return _infer_scatter_contract(target, index, source, axis)


def scatter_index_violation(
    index: np.ndarray,
    axis: int,
    target_axis_extent: int,
) -> Optional[str]:
    if index.size == 0:
        return None
    if bool(np.any(index < 0)) or bool(np.any(index >= target_axis_extent)):
        return f"scatter: index values must be in [0, {target_axis_extent})"
    if index.shape[axis] <= 1:
        return None
    ordered = np.sort(index, axis=axis)
    lower = [slice(None)] * index.ndim
    upper = [slice(None)] * index.ndim
    lower[axis] = slice(None, -1)
    upper[axis] = slice(1, None)
    if bool(np.any(ordered[tuple(lower)] == ordered[tuple(upper)])):
        return (
            "scatter: duplicate destination indices are outside the deterministic "
            "overwrite profile"
        )
    return None


def _validate_scatter(node: Any, contract: FrameworkOperationContract) -> int:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode "
            f"{getattr(node, 'op', None)!r}"
        )
    inputs = getattr(node, "inputs", None)
    if type(inputs) is not tuple or len(inputs) != 3:
        raise ShapeError("SCATTER must have target, index, and source inputs")
    target, index, source = inputs
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("SCATTER arg must be a plain dict")
    fields = set(arg)
    if "dim" not in fields or not fields.issubset({"dim", "vjp_of"}):
        raise ShapeError(
            "SCATTER arg fields must be exactly 'dim' plus optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("SCATTER arg.vjp_of must reference a UOp")
    axis = infer_scatter_contract(target, index, source, arg["dim"])
    if getattr(node, "shape", None) != getattr(target, "shape", None):
        raise ShapeError("SCATTER output shape must equal its target shape")
    if getattr(node, "dtype", None) != getattr(target, "dtype", None):
        raise ShapeError("SCATTER output dtype must equal its target dtype")
    return axis


def _validate_einsum(
    node: Any,
    contract: FrameworkOperationContract,
) -> EinsumContract:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode "
            f"{getattr(node, 'op', None)!r}"
        )
    inputs = getattr(node, "inputs", None)
    if type(inputs) is not tuple or not inputs:
        raise ShapeError("EINSUM must have a non-empty plain input tuple")
    arg = getattr(node, "arg", None)
    if type(arg) is not dict or set(arg) != {"equation", "batch_rank"}:
        raise ShapeError(
            "EINSUM arg fields must be exactly 'equation' and 'batch_rank'"
        )
    normalized = infer_einsum_contract(
        inputs,
        arg["equation"],
        arg["batch_rank"],
    )
    if arg["equation"] != normalized.equation:
        raise ShapeError(
            f"EINSUM equation must be canonical {normalized.equation!r}"
        )
    if getattr(node, "shape", None) != normalized.output_shape:
        raise ShapeError(
            f"EINSUM declared shape {getattr(node, 'shape', None)!r} does not "
            f"match derived shape {normalized.output_shape!r}"
        )
    if getattr(node, "dtype", None) != normalized.output_dtype:
        raise ShapeError(
            f"EINSUM declared dtype {getattr(node, 'dtype', None)!r} does not "
            f"match promoted dtype {normalized.output_dtype!r}"
        )
    return normalized


def _validate_einsum_vjp(node: Any) -> Tuple[EinsumContract, int]:
    if getattr(node, "op", None) != "EINSUM_VJP":
        raise ShapeError("EINSUM_VJP validator received the wrong opcode")
    inputs = getattr(node, "inputs", None)
    if type(inputs) is not tuple or len(inputs) < 2:
        raise ShapeError(
            "EINSUM_VJP inputs must be upstream gradient plus original operands"
        )
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("EINSUM_VJP arg must be a plain dict")
    fields = set(arg)
    if not {"equation", "batch_rank", "operand"}.issubset(fields) or not fields.issubset({
        "equation", "batch_rank", "operand", "vjp_of"
    }):
        raise ShapeError(
            "EINSUM_VJP arg fields must be exactly 'equation', 'batch_rank', "
            "and 'operand' plus optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("EINSUM_VJP arg.vjp_of must reference a UOp")
    normalized = infer_einsum_contract(
        inputs[1:],
        arg["equation"],
        arg["batch_rank"],
    )
    if arg["equation"] != normalized.equation:
        raise ShapeError(
            f"EINSUM_VJP equation must be canonical {normalized.equation!r}"
        )
    operand = arg["operand"]
    if type(operand) is not int or operand < 0 or operand >= len(inputs) - 1:
        raise ShapeError("EINSUM_VJP operand must be a normalized input index")
    upstream = inputs[0]
    if (
        getattr(upstream, "shape", None) != normalized.output_shape
        or getattr(upstream, "dtype", None) != normalized.output_dtype
    ):
        raise ShapeError(
            "EINSUM_VJP upstream gradient must match the forward output metadata"
        )
    target = inputs[operand + 1]
    if getattr(target, "dtype", None) not in _FLOATING_DTYPES:
        raise ShapeError("EINSUM_VJP target must have a floating dtype")
    if getattr(node, "shape", None) != getattr(target, "shape", None):
        raise ShapeError("EINSUM_VJP output shape must match its target operand")
    if getattr(node, "dtype", None) != getattr(target, "dtype", None):
        raise ShapeError("EINSUM_VJP output dtype must match its target operand")
    return normalized, operand


def _validate_gather(node: Any, contract: FrameworkOperationContract) -> int:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = getattr(node, "inputs", None)
    if type(inputs) is not tuple or len(inputs) != 2:
        raise ShapeError("INDEX must have exactly source and index inputs")
    source, index = inputs
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("INDEX arg must be a plain dict")
    fields = set(arg)
    if "dim" not in fields or not fields.issubset({"dim", "vjp_of"}):
        raise ShapeError("INDEX arg fields must be exactly 'dim' plus optional 'vjp_of'")
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("INDEX arg.vjp_of must reference a UOp")
    source_shape = getattr(source, "shape", None)
    index_shape = getattr(index, "shape", None)
    if type(source_shape) is not tuple or type(index_shape) is not tuple:
        raise ShapeError("INDEX source and index shapes must be tuples")
    rank = len(source_shape)
    if rank == 0 or len(index_shape) != rank:
        raise ShapeError(
            f"INDEX source and index must have the same nonzero rank, got {rank} and {len(index_shape)}"
        )
    axis = arg["dim"]
    if type(axis) is not int or axis < 0 or axis >= rank:
        raise ShapeError(f"INDEX arg.dim must be normalized into [0, {rank}), got {axis!r}")
    if getattr(index, "dtype", None) != "int64":
        raise ShapeError(f"INDEX index dtype must be 'int64', got {getattr(index, 'dtype', None)!r}")
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError("INDEX must preserve its source dtype")
    if getattr(node, "shape", None) != index_shape:
        raise ShapeError("INDEX output shape must equal its index shape")
    for dimension, (source_extent, index_extent) in enumerate(zip(source_shape, index_shape)):
        if dimension != axis and index_extent > source_extent:
            raise ShapeError(
                f"INDEX index extent {index_extent} exceeds source extent {source_extent} "
                f"at non-gather dimension {dimension}"
            )
    return axis


def validate_gather_scatter_add_contract(node: Any) -> int:
    if getattr(node, "op", None) != "SCATTER_ADD":
        raise ShapeError("gather scatter-add validator requires SCATTER_ADD")
    inputs = getattr(node, "inputs", None)
    if type(inputs) is not tuple or len(inputs) != 3:
        raise ShapeError("SCATTER_ADD must have target, index, and source inputs")
    target, index, source = inputs
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("SCATTER_ADD arg must be a plain dict")
    fields = set(arg)
    if "dim" not in fields or not fields.issubset({"dim", "vjp_of"}):
        raise ShapeError("SCATTER_ADD arg fields must be exactly 'dim' plus optional 'vjp_of'")
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("SCATTER_ADD arg.vjp_of must reference a UOp")
    target_shape = getattr(target, "shape", None)
    index_shape = getattr(index, "shape", None)
    if type(target_shape) is not tuple or type(index_shape) is not tuple:
        raise ShapeError("SCATTER_ADD target and index shapes must be tuples")
    rank = len(target_shape)
    axis = arg["dim"]
    if rank == 0 or len(index_shape) != rank:
        raise ShapeError("SCATTER_ADD target and index must have the same nonzero rank")
    if type(axis) is not int or axis < 0 or axis >= rank:
        raise ShapeError(f"SCATTER_ADD arg.dim must be normalized into [0, {rank}), got {axis!r}")
    if getattr(index, "dtype", None) != "int64":
        raise ShapeError("SCATTER_ADD index dtype must be int64")
    if getattr(source, "shape", None) != index_shape:
        raise ShapeError("SCATTER_ADD source shape must equal its index shape")
    if getattr(source, "dtype", None) != getattr(target, "dtype", None):
        raise ShapeError("SCATTER_ADD source dtype must equal its target dtype")
    if getattr(node, "shape", None) != target_shape or getattr(node, "dtype", None) != getattr(target, "dtype", None):
        raise ShapeError("SCATTER_ADD output shape and dtype must equal its target")
    for dimension, (target_extent, index_extent) in enumerate(zip(target_shape, index_shape)):
        if dimension != axis and index_extent > target_extent:
            raise ShapeError(
                f"SCATTER_ADD index extent {index_extent} exceeds target extent {target_extent} "
                f"at non-gather dimension {dimension}"
            )
    return axis


def _closed_broadcast_shape(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    rank = max((len(shape) for shape in shapes), default=0)
    result = []
    for reverse_axis in range(1, rank + 1):
        extents = [
            shape[-reverse_axis] if reverse_axis <= len(shape) else 1
            for shape in shapes
        ]
        nonunit = {extent for extent in extents if extent != 1}
        if len(nonunit) > 1:
            raise ShapeError(f"WHERE input shapes {shapes!r} are not broadcastable")
        result.append(next(iter(nonunit), 1))
    return tuple(reversed(result))


def _validate_where_common(node: Any) -> Tuple[Any, Any, Any]:
    if getattr(node, "op", None) != "WHERE":
        raise ShapeError(
            f"WHERE validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = getattr(node, "inputs", None)
    if type(inputs) is not tuple or len(inputs) != 3:
        raise ShapeError("WHERE must have exactly three inputs")
    condition, lhs, rhs = inputs
    if getattr(condition, "dtype", None) != "bool":
        raise ShapeError("WHERE condition dtype must be bool")
    shapes = tuple(getattr(value, "shape", None) for value in inputs)
    if any(type(shape) is not tuple for shape in shapes):
        raise ShapeError("WHERE input shapes must be tuples")
    expected_shape = _closed_broadcast_shape(*shapes)
    if getattr(node, "shape", None) != expected_shape:
        raise ShapeError(
            f"WHERE declared shape {getattr(node, 'shape', None)!r} does not match "
            f"derived shape {expected_shape!r}"
        )
    arg = getattr(node, "arg", None)
    if arg is not None and arg != MASKED_FILL_CONTRACT_ID:
        if type(arg) is not dict or set(arg) != {"vjp_of"}:
            raise ShapeError(
                "WHERE arg must be None, the masked-fill contract ID, or exact VJP provenance"
            )
        if type(arg["vjp_of"]) is not type(node):
            raise ShapeError("WHERE arg.vjp_of must reference a UOp")
    return condition, lhs, rhs


def _validate_masked_fill(
    node: Any,
    contract: FrameworkOperationContract,
) -> Any:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    mask, fill, source = _validate_where_common(node)
    if getattr(node, "arg", None) != contract.contract_id:
        return None
    if getattr(node, "shape", None) != getattr(source, "shape", None):
        raise ShapeError("masked_fill must preserve its source shape")
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError("masked_fill must preserve its source dtype")
    if getattr(source, "dtype", None) not in _MASKED_FILL_DTYPES:
        raise ShapeError(
            f"masked_fill does not support source dtype {getattr(source, 'dtype', None)!r}"
        )
    if getattr(fill, "op", None) != "CONST" or getattr(fill, "shape", None) != ():
        raise ShapeError("masked_fill fill input must be a scalar CONST")
    if getattr(fill, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError("masked_fill fill CONST dtype must equal its source dtype")
    fill_arg = getattr(fill, "arg", None)
    if type(fill_arg) is not dict or set(fill_arg) != {"value"}:
        raise ShapeError("masked_fill fill CONST arg must contain exactly 'value'")
    value = fill_arg["value"]
    source_dtype = getattr(source, "dtype", None)
    if source_dtype == "bool":
        if type(value) is not bool:
            raise ShapeError("masked_fill boolean fill value must be normalized to bool")
    elif isinstance(source_dtype, str) and source_dtype.startswith("float"):
        if type(value) is not float:
            raise ShapeError("masked_fill floating fill value must be normalized to float")
    elif type(value) is not int or type(value) is bool:
        raise ShapeError("masked_fill integer fill value must be normalized to int")
    return value


def _validate_repeat(
    node: Any,
    contract: FrameworkOperationContract,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = _require_single_input_tuple(node, contract.opcode)
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("REPEAT arg must be a plain dict")
    fields = set(arg)
    if "repeats" not in fields or not fields.issubset({"repeats", "vjp_of"}):
        raise ShapeError("REPEAT arg fields must be exactly 'repeats' plus optional 'vjp_of'")
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("REPEAT arg.vjp_of must reference a UOp")
    source = inputs[0]
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError("REPEAT must preserve its input dtype")
    repeats = arg["repeats"]
    if type(repeats) is not tuple:
        raise ShapeError("REPEAT arg.repeats must be a canonical tuple")
    source_shape = tuple(getattr(source, "shape", ()))
    if not repeats:
        raise ShapeError("REPEAT requires at least one repeat factor")
    if len(repeats) < len(source_shape):
        raise ShapeError(
            f"REPEAT repeat rank {len(repeats)} cannot be shorter than input rank {len(source_shape)}"
        )
    if len(repeats) > REPEAT_RANK_MAX:
        raise ShapeError(
            f"REPEAT repeat rank {len(repeats)} exceeds the {REPEAT_RANK_MAX}-axis ceiling"
        )
    for axis, factor in enumerate(repeats):
        if type(factor) is not int:
            raise ShapeError(f"REPEAT arg.repeats[{axis}] must be a normalized integer")
        if factor < 0 or factor > REPEAT_FACTOR_MAX:
            raise ShapeError(
                f"REPEAT arg.repeats[{axis}] must be in [0, {REPEAT_FACTOR_MAX}], got {factor}"
            )
    padded_shape = (1,) * (len(repeats) - len(source_shape)) + source_shape
    expected_shape = tuple(
        factor * extent for factor, extent in zip(repeats, padded_shape)
    )
    if getattr(node, "shape", None) != expected_shape:
        raise ShapeError(
            f"REPEAT declared shape {getattr(node, 'shape', None)!r} does not match "
            f"derived shape {expected_shape!r}"
        )
    return repeats, padded_shape


def _validate_prod(
    node: Any,
    contract: FrameworkOperationContract,
) -> Tuple[Tuple[int, ...], bool, Tuple[int, ...]]:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = _require_single_input_tuple(node, contract.opcode)
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("PROD arg must be a plain dict")
    fields = set(arg)
    if not {"axes", "keepdims"}.issubset(fields) or not fields.issubset(
        {"axes", "keepdims", "vjp_of"}
    ):
        raise ShapeError(
            "PROD arg fields must be exactly 'axes' and 'keepdims' plus optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("PROD arg.vjp_of must reference a UOp")
    source = inputs[0]
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError("PROD must preserve its input dtype")
    source_shape = tuple(getattr(source, "shape", ()))
    axes = arg["axes"]
    keepdims = arg["keepdims"]
    if type(axes) is not tuple:
        raise ShapeError("PROD arg.axes must be a canonical tuple")
    if type(keepdims) is not bool:
        raise ShapeError("PROD arg.keepdims must be a boolean")
    if source_shape and not axes:
        raise ShapeError("PROD arg.axes must be non-empty for a non-scalar input")
    if tuple(sorted(axes)) != axes or len(set(axes)) != len(axes):
        raise ShapeError("PROD arg.axes must be strictly increasing and unique")
    for axis in axes:
        if type(axis) is not int:
            raise ShapeError("PROD arg.axes must contain normalized integers")
        if axis < 0 or axis >= len(source_shape):
            raise ShapeError(
                f"PROD arg axis {axis} out of range for rank {len(source_shape)}"
            )
    expected_shape = tuple(
        1 if keepdims and axis in axes else extent
        for axis, extent in enumerate(source_shape)
        if keepdims or axis not in axes
    )
    if getattr(node, "shape", None) != expected_shape:
        raise ShapeError(
            f"PROD declared shape {getattr(node, 'shape', None)!r} does not match "
            f"derived shape {expected_shape!r}"
        )
    expanded_shape = tuple(
        1 if axis in axes else extent
        for axis, extent in enumerate(source_shape)
    )
    return axes, keepdims, expanded_shape


def _validate_var(
    node: Any,
    contract: FrameworkOperationContract,
) -> Tuple[Tuple[int, ...], int, bool, Tuple[int, ...], int]:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = _require_single_input_tuple(node, contract.opcode)
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("VAR arg must be a plain dict")
    fields = set(arg)
    if not {"axes", "correction", "keepdims"}.issubset(fields) or not fields.issubset(
        {"axes", "correction", "keepdims", "vjp_of"}
    ):
        raise ShapeError(
            "VAR arg fields must be exactly 'axes', 'correction', and 'keepdims' "
            "plus optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("VAR arg.vjp_of must reference a UOp")
    source = inputs[0]
    source_shape = getattr(source, "shape", None)
    if type(source_shape) is not tuple:
        raise ShapeError("VAR source shape must be a tuple")
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError("VAR must preserve its input dtype")
    if getattr(node, "dtype", None) not in _FLOATING_DTYPES:
        raise ShapeError(
            f"VAR supports floating dtypes only, got {getattr(node, 'dtype', None)!r}"
        )
    axes = arg["axes"]
    correction = arg["correction"]
    keepdims = arg["keepdims"]
    if type(axes) is not tuple:
        raise ShapeError("VAR arg.axes must be a canonical tuple")
    if type(correction) is not int:
        raise ShapeError("VAR arg.correction must be a normalized integer")
    if correction < VAR_CORRECTION_MIN or correction > VAR_CORRECTION_MAX:
        raise ShapeError(
            f"VAR arg.correction must be in [{VAR_CORRECTION_MIN}, {VAR_CORRECTION_MAX}], "
            f"got {correction}"
        )
    if type(keepdims) is not bool:
        raise ShapeError("VAR arg.keepdims must be a boolean")
    if source_shape and not axes:
        raise ShapeError("VAR arg.axes must be non-empty for a non-scalar input")
    if tuple(sorted(axes)) != axes or len(set(axes)) != len(axes):
        raise ShapeError("VAR arg.axes must be strictly increasing and unique")
    reduced_elements = 1
    for axis in axes:
        if type(axis) is not int:
            raise ShapeError("VAR arg.axes must contain normalized integers")
        if axis < 0 or axis >= len(source_shape):
            raise ShapeError(
                f"VAR arg axis {axis} out of range for rank {len(source_shape)}"
            )
        reduced_elements *= source_shape[axis]
    expected_shape = tuple(
        1 if keepdims and axis in axes else extent
        for axis, extent in enumerate(source_shape)
        if keepdims or axis not in axes
    )
    if getattr(node, "shape", None) != expected_shape:
        raise ShapeError(
            f"VAR declared shape {getattr(node, 'shape', None)!r} does not match "
            f"derived shape {expected_shape!r}"
        )
    expanded_shape = tuple(
        1 if axis in axes else extent
        for axis, extent in enumerate(source_shape)
    )
    return axes, correction, keepdims, expanded_shape, reduced_elements


def _validate_repeat_interleave(
    node: Any,
    contract: FrameworkOperationContract,
) -> Tuple[int, int]:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = _require_single_input_tuple(node, contract.opcode)
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("REPEAT_INTERLEAVE arg must be a plain dict")
    fields = set(arg)
    if not {"axis", "repeats"}.issubset(fields) or not fields.issubset(
        {"axis", "repeats", "vjp_of"}
    ):
        raise ShapeError(
            "REPEAT_INTERLEAVE arg fields must be exactly 'axis' and 'repeats' "
            "plus optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("REPEAT_INTERLEAVE arg.vjp_of must reference a UOp")
    source = inputs[0]
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError("REPEAT_INTERLEAVE must preserve its input dtype")
    source_shape = tuple(getattr(source, "shape", ()))
    axis = arg["axis"]
    repeats = arg["repeats"]
    if type(axis) is not int:
        raise ShapeError("REPEAT_INTERLEAVE arg.axis must be a normalized integer")
    if axis < 0 or axis >= len(source_shape):
        raise ShapeError(
            f"REPEAT_INTERLEAVE arg.axis {axis} out of range for rank {len(source_shape)}"
        )
    if type(repeats) is not int:
        raise ShapeError("REPEAT_INTERLEAVE arg.repeats must be a normalized integer")
    if repeats < 0 or repeats > REPEAT_FACTOR_MAX:
        raise ShapeError(
            f"REPEAT_INTERLEAVE arg.repeats must be in [0, {REPEAT_FACTOR_MAX}], "
            f"got {repeats}"
        )
    expected_shape = list(source_shape)
    expected_shape[axis] *= repeats
    expected_shape_tuple = tuple(expected_shape)
    if getattr(node, "shape", None) != expected_shape_tuple:
        raise ShapeError(
            f"REPEAT_INTERLEAVE declared shape {getattr(node, 'shape', None)!r} "
            f"does not match derived shape {expected_shape_tuple!r}"
        )
    return repeats, axis


_VALIDATORS: Mapping[str, Callable[[Any, FrameworkOperationContract], Any]] = MappingProxyType({
    "browsergrad.jit.framework.tensor.abs.v1": _validate_typed_unary,
    "browsergrad.jit.framework.tensor.cat.v1": _validate_cat,
    "browsergrad.jit.framework.tensor.clamp.v1": _validate_clamp,
    "browsergrad.jit.framework.tensor.cos.v1": _validate_typed_unary,
    "browsergrad.jit.framework.tensor.cumsum.v1": _validate_cumsum,
    "browsergrad.jit.framework.tensor.einsum.v1": _validate_einsum,
    "browsergrad.jit.framework.tensor.expand.v1": _validate_broadcast_to,
    "browsergrad.jit.framework.tensor.flip.v1": _validate_flip,
    "browsergrad.jit.framework.tensor.gather.v1": _validate_gather,
    "browsergrad.jit.framework.tensor.masked-fill.v1": _validate_masked_fill,
    "browsergrad.jit.framework.tensor.pad.v1": _validate_pad,
    "browsergrad.jit.framework.tensor.prod.v1": _validate_prod,
    "browsergrad.jit.framework.tensor.var.v1": _validate_var,
    "browsergrad.jit.framework.tensor.repeat.v1": _validate_repeat,
    "browsergrad.jit.framework.tensor.repeat-interleave.v1": _validate_repeat_interleave,
    "browsergrad.jit.framework.tensor.scatter.v1": _validate_scatter,
    "browsergrad.jit.framework.tensor.sort-indices.v1": _validate_sort_indices,
    "browsergrad.jit.framework.tensor.sort-values.v1": _validate_sort_values,
    "browsergrad.jit.framework.tensor.sign.v1": _validate_typed_unary,
    "browsergrad.jit.framework.tensor.sin.v1": _validate_typed_unary,
    "browsergrad.jit.framework.tensor.stack.v1": _validate_stack,
    "browsergrad.jit.framework.tensor.topk-indices.v1": _validate_topk_indices,
    "browsergrad.jit.framework.tensor.topk-values.v1": _validate_topk_values,
    "browsergrad.jit.framework.tensor.tril.v1": _validate_tril,
    "browsergrad.jit.framework.tensor.triu.v1": _validate_triu,
})
_RECORDS = _load_registry()
if frozenset(_VALIDATORS) != frozenset(record.contract_id for record in _RECORDS):
    raise ValueError("framework operation registry and executable validators differ")
_BY_OPCODE: Mapping[str, _ExecutableFrameworkOperationContract] = MappingProxyType({
    record.opcode: _ExecutableFrameworkOperationContract(
        record=record,
        validator=_VALIDATORS[record.contract_id],
    )
    for record in _RECORDS
})
_INTERNAL_VALIDATORS: Mapping[str, Callable[[Any], Any]] = MappingProxyType({
    "NARROW": _validate_narrow,
    "EINSUM_VJP": _validate_einsum_vjp,
})


def has_framework_operation_contract(opcode: str) -> bool:
    return opcode in _BY_OPCODE


def has_internal_operation_contract(opcode: str) -> bool:
    return opcode in _INTERNAL_VALIDATORS


def validate_internal_operation_contract(node: Any) -> Any:
    opcode = getattr(node, "op", None)
    validator = _INTERNAL_VALIDATORS.get(opcode)
    if validator is None:
        raise ShapeError(f"no typed internal-operation contract for opcode {opcode!r}")
    return validator(node)


def validate_framework_operation_contract(
    node: Any,
) -> tuple[FrameworkOperationContract, Any]:
    opcode = getattr(node, "op", None)
    executable = _BY_OPCODE.get(opcode)
    if executable is None:
        raise ShapeError(f"no typed framework-operation contract for opcode {opcode!r}")
    normalized = executable.validator(node, executable.record)
    return executable.record, normalized


def validate_broadcast_to_contract(node: Any) -> Tuple[int, ...]:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.expand.v1":
        raise ShapeError("BROADCAST_TO resolved to the wrong framework-operation contract")
    return normalized


def validate_clamp_contract(node: Any) -> Tuple[Optional[float], Optional[float]]:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.clamp.v1":
        raise ShapeError("CLAMP resolved to the wrong framework-operation contract")
    return normalized


def validate_cat_contract(
    node: Any,
) -> Tuple[int, Tuple[int, ...], Tuple[bool, ...]]:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.cat.v1":
        raise ShapeError("CONCAT resolved to the wrong framework-operation contract")
    return normalized


def validate_stack_contract(node: Any) -> int:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.stack.v1":
        raise ShapeError("STACK resolved to the wrong framework-operation contract")
    return normalized


def validate_pad_contract(node: Any) -> Tuple[Tuple[Tuple[int, int], ...], Any]:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.pad.v1":
        raise ShapeError("PAD resolved to the wrong framework-operation contract")
    return normalized


def validate_sort_indices_contract(node: Any) -> Tuple[int, bool, bool]:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.sort-indices.v1":
        raise ShapeError("SORT_INDICES resolved to the wrong framework-operation contract")
    return normalized


def validate_sort_values_contract(node: Any) -> Tuple[int, bool, bool]:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.sort-values.v1":
        raise ShapeError("SORT_VALUES resolved to the wrong framework-operation contract")
    return normalized


def validate_topk_indices_contract(node: Any) -> Tuple[int, int, bool, bool]:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.topk-indices.v1":
        raise ShapeError("TOPK_INDICES resolved to the wrong framework-operation contract")
    return normalized


def validate_topk_values_contract(node: Any) -> Tuple[int, int, bool, bool]:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.topk-values.v1":
        raise ShapeError("TOPK_VALUES resolved to the wrong framework-operation contract")
    return normalized


def validate_scatter_contract(node: Any) -> int:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.scatter.v1":
        raise ShapeError("SCATTER resolved to the wrong framework-operation contract")
    return normalized


def validate_einsum_contract(node: Any) -> EinsumContract:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.einsum.v1":
        raise ShapeError("EINSUM resolved to the wrong framework-operation contract")
    return normalized


def validate_einsum_vjp_contract(node: Any) -> Tuple[EinsumContract, int]:
    return validate_internal_operation_contract(node)


def validate_narrow_contract(node: Any) -> Tuple[int, int, int]:
    return validate_internal_operation_contract(node)


def validate_flip_contract(node: Any) -> int:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.flip.v1":
        raise ShapeError("FLIP resolved to the wrong framework-operation contract")
    return normalized


def validate_cumsum_contract(node: Any) -> Tuple[int, bool]:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.cumsum.v1":
        raise ShapeError("CUMSUM resolved to the wrong framework-operation contract")
    return normalized


def validate_gather_contract(node: Any) -> int:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.gather.v1":
        raise ShapeError("INDEX resolved to the wrong framework-operation contract")
    return normalized


def validate_tril_contract(node: Any) -> int:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.tril.v1":
        raise ShapeError("TRIL resolved to the wrong framework-operation contract")
    return normalized


def validate_triu_contract(node: Any) -> int:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.triu.v1":
        raise ShapeError("TRIU resolved to the wrong framework-operation contract")
    return normalized


def validate_masked_fill_contract(node: Any) -> Any:
    if getattr(node, "arg", None) != MASKED_FILL_CONTRACT_ID:
        raise ShapeError("masked_fill WHERE arg must be its exact framework contract ID")
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != MASKED_FILL_CONTRACT_ID:
        raise ShapeError("WHERE resolved to the wrong masked-fill contract")
    return normalized


def validate_where_contract(node: Any) -> bool:
    _validate_where_common(node)
    if getattr(node, "arg", None) == MASKED_FILL_CONTRACT_ID:
        validate_masked_fill_contract(node)
        return True
    return False


def validate_repeat_contract(node: Any) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.repeat.v1":
        raise ShapeError("REPEAT resolved to the wrong framework-operation contract")
    return normalized


def validate_prod_contract(node: Any) -> Tuple[Tuple[int, ...], bool, Tuple[int, ...]]:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.prod.v1":
        raise ShapeError("PROD resolved to the wrong framework-operation contract")
    return normalized


def validate_var_contract(
    node: Any,
) -> Tuple[Tuple[int, ...], int, bool, Tuple[int, ...], int]:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.var.v1":
        raise ShapeError("VAR resolved to the wrong framework-operation contract")
    return normalized


def validate_repeat_interleave_contract(node: Any) -> Tuple[int, int]:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.tensor.repeat-interleave.v1":
        raise ShapeError(
            "REPEAT_INTERLEAVE resolved to the wrong framework-operation contract"
        )
    return normalized


_TYPED_UNARY_CONTRACT_IDS = frozenset({
    "browsergrad.jit.framework.tensor.abs.v1",
    "browsergrad.jit.framework.tensor.cos.v1",
    "browsergrad.jit.framework.tensor.sign.v1",
    "browsergrad.jit.framework.tensor.sin.v1",
})


def validate_typed_unary_contract(node: Any) -> FrameworkOperationContract:
    record, _ = validate_framework_operation_contract(node)
    if record.contract_id not in _TYPED_UNARY_CONTRACT_IDS:
        raise ShapeError("unary node resolved to the wrong framework-operation contract")
    return record


def validate_real_numeric_unary_contract(node: Any) -> FrameworkOperationContract:
    record = validate_typed_unary_contract(node)
    if record.dtype_contract != "preserve-real-numeric-input":
        raise ShapeError("real-numeric unary node resolved to the wrong dtype contract")
    return record


def framework_operation_support() -> dict[str, Any]:
    """Return a detached deterministic table of executable framework decisions."""
    operations = []
    for record in sorted(_RECORDS, key=lambda item: item.contract_id):
        operations.append({
            "contractId": record.contract_id,
            "publicSurface": record.public_surface,
            "opcode": record.opcode,
            "semanticState": record.semantic_state,
            "shapeContract": record.shape_contract,
            "dtypeContract": record.dtype_contract,
            "decisions": dict(record.decisions),
            "retiredOpaqueOperationId": record.retired_opaque_operation_id,
        })
    return {
        "schema": FRAMEWORK_OPERATION_SUPPORT_SCHEMA,
        "version": {
            "major": FRAMEWORK_OPERATION_SUPPORT_VERSION[0],
            "minor": FRAMEWORK_OPERATION_SUPPORT_VERSION[1],
        },
        "operations": operations,
    }


__all__ = [
    "FRAMEWORK_OPERATION_SUPPORT_SCHEMA",
    "FRAMEWORK_OPERATION_SUPPORT_VERSION",
    "REPEAT_FACTOR_MAX",
    "REPEAT_RANK_MAX",
    "VAR_CORRECTION_MIN",
    "VAR_CORRECTION_MAX",
    "VARIADIC_INPUT_MAX",
    "VARIADIC_OUTPUT_BYTE_MAX",
    "PAD_RANK_MAX",
    "PAD_OUTPUT_BYTE_MAX",
    "PAD_OUTPUT_EXTENT_MAX",
    "SORT_RANK_MAX",
    "SORT_AXIS_MAX",
    "SORT_OUTPUT_BYTE_MAX",
    "SORT_OUTPUT_EXTENT_MAX",
    "SORT_WORKSPACE_BYTE_MAX",
    "TOPK_RANK_MAX",
    "TOPK_AXIS_MAX",
    "TOPK_OUTPUT_BYTE_MAX",
    "TOPK_OUTPUT_EXTENT_MAX",
    "TOPK_WORKSPACE_BYTE_MAX",
    "SCATTER_RANK_MAX",
    "SCATTER_OUTPUT_BYTE_MAX",
    "SCATTER_OUTPUT_EXTENT_MAX",
    "SCATTER_WORKSPACE_BYTE_MAX",
    "EINSUM_INPUT_MAX",
    "EINSUM_EQUATION_BYTE_MAX",
    "EINSUM_RANK_MAX",
    "EINSUM_LABEL_MAX",
    "EINSUM_OUTPUT_BYTE_MAX",
    "EINSUM_OUTPUT_EXTENT_MAX",
    "EINSUM_WORK_ELEMENT_MAX",
    "EINSUM_WORKSPACE_BYTE_MAX",
    "MASKED_FILL_CONTRACT_ID",
    "FrameworkOperationContract",
    "EinsumContract",
    "framework_operation_support",
    "has_framework_operation_contract",
    "has_internal_operation_contract",
    "infer_cat_contract",
    "infer_stack_contract",
    "infer_pad_contract",
    "infer_sort_contract",
    "infer_topk_contract",
    "infer_scatter_contract",
    "infer_einsum_contract",
    "normalize_pad_request",
    "normalize_pad_value",
    "normalize_sort_request",
    "normalize_topk_request",
    "normalize_scatter_request",
    "scatter_index_violation",
    "execute_einsum_arrays",
    "execute_einsum_vjp_array",
    "einsum_onnx_equation",
    "stable_sort_indices_array",
    "partial_topk_indices_array",
    "promote_variadic_dtypes",
    "validate_framework_operation_contract",
    "validate_internal_operation_contract",
    "validate_broadcast_to_contract",
    "validate_cat_contract",
    "validate_stack_contract",
    "validate_pad_contract",
    "validate_sort_indices_contract",
    "validate_sort_values_contract",
    "validate_topk_indices_contract",
    "validate_topk_values_contract",
    "validate_scatter_contract",
    "validate_einsum_contract",
    "validate_einsum_vjp_contract",
    "validate_clamp_contract",
    "validate_cumsum_contract",
    "validate_flip_contract",
    "validate_gather_contract",
    "validate_gather_scatter_add_contract",
    "validate_tril_contract",
    "validate_triu_contract",
    "validate_masked_fill_contract",
    "validate_narrow_contract",
    "validate_where_contract",
    "validate_prod_contract",
    "validate_var_contract",
    "validate_repeat_contract",
    "validate_repeat_interleave_contract",
    "validate_real_numeric_unary_contract",
    "validate_typed_unary_contract",
]
