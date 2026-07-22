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

from ._errors import ShapeError


FRAMEWORK_OPERATION_SUPPORT_SCHEMA = "browsergrad.jit.framework-operation-contracts"
FRAMEWORK_OPERATION_SUPPORT_VERSION = (1, 0)
REPEAT_FACTOR_MAX = 1 << 30
REPEAT_RANK_MAX = 32
VAR_CORRECTION_MIN = -(1 << 31)
VAR_CORRECTION_MAX = (1 << 31) - 1
MASKED_FILL_CONTRACT_ID = "browsergrad.jit.framework.tensor.masked-fill.v1"
_REGISTRY_FILENAME = "framework-operation-contracts.v1.json"
_REGISTRY_BYTE_LIMIT = 16 * 1024
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
        "same-rank-index-shaped-gather",
        "preserve-source-with-broadcast-bool-mask",
        "static-broadcast-with-existing-dim-minus-one",
        "selected-axis-times-repeat-count",
        "static-product-reduction",
        "static-variance-reduction",
        "tile-multipliers-with-left-rank-padding",
    }),
    "dtypeContract": frozenset({
        "preserve-floating-input",
        "preserve-input",
        "preserve-input-require-bool-mask",
        "preserve-real-numeric-input",
        "preserve-source-require-int64-index",
        "promote-integral-default-or-explicit-scan-dtype",
    }),
    "cpu": frozenset({
        "supported-numpy-dtype-preserving",
        "supported-numpy-owning-copy",
        "supported-numpy-owning-copy-with-range-check",
        "supported-numpy-owning-scan-copy",
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
    }),
    "functionalGrad": frozenset({
        "supported-via-symbolic-vjp",
        "supported-for-floating-source-and-output-via-symbolic-vjp",
    }),
    "vmap": frozenset({
        "supported-leading-batch-axis",
        "supported-leading-batch-axis-with-axis-shift",
        "supported-leading-batch-axis-with-index-axis-shift",
        "supported-leading-batch-axis-with-mask-broadcast",
        "supported-leading-batch-axis-preserve-matrix-axes",
        "supported-leading-batch-axis-with-unit-repeat",
        "supported-leading-batch-axis-with-scan-axis-shift",
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
    }),
    "tensorPlan": frozenset({
        "refused-negative-stride-profile",
        "refused-no-deterministic-index-lowering",
        "refused-no-portable-masked-selection",
        "refused-no-portable-triangular-selection",
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
    "browsergrad.jit.framework.tensor.clamp.v1": _validate_clamp,
    "browsergrad.jit.framework.tensor.cos.v1": _validate_typed_unary,
    "browsergrad.jit.framework.tensor.cumsum.v1": _validate_cumsum,
    "browsergrad.jit.framework.tensor.expand.v1": _validate_broadcast_to,
    "browsergrad.jit.framework.tensor.flip.v1": _validate_flip,
    "browsergrad.jit.framework.tensor.gather.v1": _validate_gather,
    "browsergrad.jit.framework.tensor.masked-fill.v1": _validate_masked_fill,
    "browsergrad.jit.framework.tensor.prod.v1": _validate_prod,
    "browsergrad.jit.framework.tensor.var.v1": _validate_var,
    "browsergrad.jit.framework.tensor.repeat.v1": _validate_repeat,
    "browsergrad.jit.framework.tensor.repeat-interleave.v1": _validate_repeat_interleave,
    "browsergrad.jit.framework.tensor.sign.v1": _validate_typed_unary,
    "browsergrad.jit.framework.tensor.sin.v1": _validate_typed_unary,
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


def has_framework_operation_contract(opcode: str) -> bool:
    return opcode in _BY_OPCODE


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
    "MASKED_FILL_CONTRACT_ID",
    "FrameworkOperationContract",
    "framework_operation_support",
    "has_framework_operation_contract",
    "validate_framework_operation_contract",
    "validate_broadcast_to_contract",
    "validate_clamp_contract",
    "validate_cumsum_contract",
    "validate_flip_contract",
    "validate_gather_contract",
    "validate_gather_scatter_add_contract",
    "validate_tril_contract",
    "validate_triu_contract",
    "validate_masked_fill_contract",
    "validate_where_contract",
    "validate_prod_contract",
    "validate_var_contract",
    "validate_repeat_contract",
    "validate_repeat_interleave_contract",
    "validate_real_numeric_unary_contract",
    "validate_typed_unary_contract",
]
