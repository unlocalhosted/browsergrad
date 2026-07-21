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
        "static-broadcast-with-existing-dim-minus-one",
    }),
    "dtypeContract": frozenset({
        "preserve-floating-input",
        "preserve-input",
        "preserve-real-numeric-input",
    }),
    "cpu": frozenset({
        "supported-numpy-dtype-preserving",
        "supported-numpy-owning-copy",
    }),
    "closureAutograd": frozenset({
        "supported-cos-derivative",
        "supported-inclusive-bound-mask",
        "supported-involutive-flip",
        "supported-negative-sin-derivative",
        "supported-sign-derivative",
        "supported-unbroadcast-sum",
        "supported-zero-derivative",
    }),
    "symbolicVjp": frozenset({
        "supported-cos-derivative",
        "supported-inclusive-bound-mask",
        "supported-involutive-flip",
        "supported-negative-sin-derivative",
        "supported-sign-derivative",
        "supported-unbroadcast-sum",
        "supported-zero-derivative",
    }),
    "functionalGrad": frozenset({"supported-via-symbolic-vjp"}),
    "vmap": frozenset({
        "supported-leading-batch-axis",
        "supported-leading-batch-axis-with-axis-shift",
    }),
    "onnxExport": frozenset({
        "supported-opset17-clip-export-dtypes",
        "supported-opset17-direct-unary-export-dtypes",
        "supported-opset17-expand",
        "supported-opset17-slice-float32-int32-int64-bool",
    }),
    "tensorPlan": frozenset({
        "refused-negative-stride-profile",
        "refused-no-portable-lowering",
        "supported-primitive",
    }),
    "webgpu": frozenset({
        "profile-nonempty-f32-rank-at-most-4",
        "refused-negative-stride-profile",
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


_VALIDATORS: Mapping[str, Callable[[Any, FrameworkOperationContract], Any]] = MappingProxyType({
    "browsergrad.jit.framework.tensor.abs.v1": _validate_typed_unary,
    "browsergrad.jit.framework.tensor.clamp.v1": _validate_clamp,
    "browsergrad.jit.framework.tensor.cos.v1": _validate_typed_unary,
    "browsergrad.jit.framework.tensor.expand.v1": _validate_broadcast_to,
    "browsergrad.jit.framework.tensor.flip.v1": _validate_flip,
    "browsergrad.jit.framework.tensor.sign.v1": _validate_typed_unary,
    "browsergrad.jit.framework.tensor.sin.v1": _validate_typed_unary,
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
    "FrameworkOperationContract",
    "framework_operation_support",
    "has_framework_operation_contract",
    "validate_framework_operation_contract",
    "validate_broadcast_to_contract",
    "validate_clamp_contract",
    "validate_flip_contract",
    "validate_real_numeric_unary_contract",
    "validate_typed_unary_contract",
]
