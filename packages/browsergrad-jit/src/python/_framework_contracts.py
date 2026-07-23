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
L1_LOSS_RANK_MAX = 32
L1_LOSS_OUTPUT_BYTE_MAX = 1 << 28
L1_LOSS_OUTPUT_EXTENT_MAX = L1_LOSS_OUTPUT_BYTE_MAX
L1_LOSS_WORK_ELEMENT_MAX = 1 << 28
L1_LOSS_WORKSPACE_BYTE_MAX = 1 << 28
L1_LOSS_WORK_VISIT_FACTOR = 10
SMOOTH_L1_LOSS_WORK_VISIT_FACTOR = 32
BINARY_CROSS_ENTROPY_WORK_VISIT_FACTOR = 48
BINARY_CROSS_ENTROPY_LOG_FLOOR = -100.0
BINARY_CROSS_ENTROPY_GRAD_EPSILON = 1e-12
BINARY_CROSS_ENTROPY_WITH_LOGITS_WORK_VISIT_FACTOR = 36
KL_DIV_WORK_VISIT_FACTOR = 48
NLL_LOSS_WORK_VISIT_FACTOR = 32
CROSS_ENTROPY_WORK_VISIT_FACTOR = 64
MASKED_FILL_CONTRACT_ID = "browsergrad.jit.framework.tensor.masked-fill.v1"
_REGISTRY_FILENAME = "framework-operation-contracts.v1.json"
_REGISTRY_BYTE_LIMIT = 64 * 1024
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
        "same-shape-elementwise-loss-with-batched-reduction",
        "class-axis-index-loss-with-batched-reduction",
        "class-axis-logits-loss-with-index-or-probability-target-and-batched-reduction",
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
        "promote-floating-inputs-with-fp32-half-accumulator",
        "preserve-floating-input-require-int64-target-and-optional-matching-weight",
        "preserve-floating-input-require-index-or-matching-floating-target-and-optional-matching-weight",
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
        "supported-numpy-owning-loss-reduction",
        "supported-numpy-owning-bounded-bce-reduction",
        "supported-numpy-owning-stable-bce-with-logits-reduction",
        "supported-numpy-owning-kl-div-reduction",
        "supported-numpy-owning-bounded-nll-reduction",
        "supported-numpy-owning-stable-cross-entropy-reduction",
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
        "supported-signed-difference-for-both-inputs",
        "supported-piecewise-difference-for-both-inputs",
        "supported-clamped-bce-derivatives-for-both-inputs",
        "supported-stable-bce-with-logits-derivatives-for-both-inputs",
        "supported-native-kl-div-derivatives-for-both-inputs",
        "supported-selected-class-negative-weight-gradient",
        "supported-stable-logits-and-probability-target-gradients",
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
        "supported-signed-difference-for-both-inputs",
        "supported-piecewise-difference-for-both-inputs",
        "supported-clamped-bce-derivatives-for-both-inputs",
        "supported-stable-bce-with-logits-derivatives-for-both-inputs",
        "supported-native-kl-div-derivatives-for-both-inputs",
        "supported-selected-class-negative-weight-gradient",
        "supported-stable-logits-and-probability-target-gradients",
    }),
    "functionalGrad": frozenset({
        "supported-via-symbolic-vjp",
        "supported-for-floating-source-and-output-via-symbolic-vjp",
        "supported-for-floating-output-via-symbolic-vjp",
        "supported-for-floating-input-via-symbolic-vjp",
        "supported-for-floating-target-and-source-via-symbolic-vjp",
        "supported-for-both-floating-inputs-via-symbolic-vjp",
        "supported-for-floating-input-and-probability-target-via-symbolic-vjp",
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
        "supported-leading-batch-axis-with-per-example-reduction",
        "supported-leading-batch-axis-with-class-axis-shift-and-captured-weight",
        "supported-leading-batch-axis-with-target-mode-class-axis-shift-and-captured-weight",
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
        "supported-opset17-sub-abs-reduce-float16-float32-float64",
        "supported-opset17-piecewise-smooth-l1-float16-float32-float64",
        "refused-runtime-probability-domain-cannot-fail-closed",
        "supported-opset17-stable-bce-with-logits-float16-float32-float64",
        "supported-opset17-kl-div-float16-float32-float64",
        "supported-opset17-negative-log-likelihood-loss-unmapped-profile",
        "supported-opset17-softmax-cross-entropy-loss-unmapped-index-profile",
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
        "refused-no-canonical-loss-reduction-lowering",
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
_SMOOTH_L1_BETA_TYPES = _PAD_VALUE_TYPES
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


@dataclass(frozen=True, slots=True)
class _ElementwiseLossGeometry:
    input_shape: Tuple[int, ...]
    input_dtypes: Tuple[str, str]
    output_shape: Tuple[int, ...]
    output_dtype: str
    reduction: str
    batch_rank: int
    reduced_elements: int
    work_elements: int
    workspace_bytes: int


@dataclass(frozen=True, slots=True)
class L1LossContract:
    input_shape: Tuple[int, ...]
    input_dtypes: Tuple[str, str]
    output_shape: Tuple[int, ...]
    output_dtype: str
    reduction: str
    batch_rank: int
    reduced_elements: int
    work_elements: int
    workspace_bytes: int


@dataclass(frozen=True, slots=True)
class SmoothL1LossContract:
    input_shape: Tuple[int, ...]
    input_dtypes: Tuple[str, str]
    output_shape: Tuple[int, ...]
    output_dtype: str
    reduction: str
    batch_rank: int
    reduced_elements: int
    work_elements: int
    workspace_bytes: int
    beta: float


@dataclass(frozen=True, slots=True)
class BinaryCrossEntropyContract:
    input_shape: Tuple[int, ...]
    input_dtypes: Tuple[str, str]
    output_shape: Tuple[int, ...]
    output_dtype: str
    reduction: str
    batch_rank: int
    reduced_elements: int
    work_elements: int
    workspace_bytes: int


@dataclass(frozen=True, slots=True)
class BinaryCrossEntropyWithLogitsContract:
    input_shape: Tuple[int, ...]
    input_dtypes: Tuple[str, str]
    output_shape: Tuple[int, ...]
    output_dtype: str
    reduction: str
    batch_rank: int
    reduced_elements: int
    work_elements: int
    workspace_bytes: int


@dataclass(frozen=True, slots=True)
class KlDivContract:
    input_shape: Tuple[int, ...]
    input_dtypes: Tuple[str, str]
    output_shape: Tuple[int, ...]
    output_dtype: str
    reduction: str
    batch_rank: int
    reduced_elements: int
    work_elements: int
    workspace_bytes: int
    log_target: bool
    batch_denominator: int


@dataclass(frozen=True, slots=True)
class NllLossContract:
    input_shape: Tuple[int, ...]
    target_shape: Tuple[int, ...]
    weight_shape: Tuple[int, ...] | None
    output_shape: Tuple[int, ...]
    output_dtype: str
    reduction: str
    batch_rank: int
    class_axis: int
    class_count: int
    ignore_index: int
    has_weight: bool
    input_elements: int
    target_elements: int
    work_elements: int
    workspace_bytes: int


@dataclass(frozen=True, slots=True)
class CrossEntropyContract:
    input_shape: Tuple[int, ...]
    target_shape: Tuple[int, ...]
    weight_shape: Tuple[int, ...] | None
    output_shape: Tuple[int, ...]
    output_dtype: str
    target_dtype: str
    target_mode: str
    reduction: str
    batch_rank: int
    class_axis: int
    class_count: int
    ignore_index: int
    has_weight: bool
    label_smoothing: float
    input_elements: int
    target_elements: int
    work_elements: int
    workspace_bytes: int


def _elementwise_loss_checked_product(extents: Tuple[int, ...], ceiling: int) -> int:
    product = 1
    for extent in extents:
        if extent == 0:
            return 0
        if product > ceiling // extent:
            return ceiling + 1
        product *= extent
    return product


def _infer_elementwise_loss_geometry(
    inputs: tuple[Any, ...],
    reduction: Any,
    batch_rank: Any,
    operation: str,
    work_visit_factor: int,
    compute_buffers: int = 3,
    mask_buffers: int = 1,
    allow_batchmean: bool = False,
) -> _ElementwiseLossGeometry:
    if type(compute_buffers) is not int or compute_buffers < 0:
        raise ShapeError(f"{operation}: compute buffer count must be non-negative")
    if type(mask_buffers) is not int or mask_buffers < 0:
        raise ShapeError(f"{operation}: mask buffer count must be non-negative")
    if type(inputs) is not tuple or len(inputs) != 2:
        raise ShapeError(f"{operation} requires exactly two tensor inputs")
    if type(reduction) is not str:
        raise ShapeError(
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
        raise ShapeError(
            f"{operation}: reduction must be {expected}, got {reduction!r}"
        )
    if type(batch_rank) is not int:
        raise ShapeError(f"{operation}: batch_rank must be a normalized integer")

    shapes = []
    dtypes = []
    for index, source in enumerate(inputs):
        shape = getattr(source, "shape", None)
        if type(shape) is not tuple:
            raise ShapeError(f"{operation} input {index} shape must be a tuple")
        if len(shape) > L1_LOSS_RANK_MAX:
            raise ShapeError(
                f"{operation} input {index} rank {len(shape)} exceeds the "
                f"{L1_LOSS_RANK_MAX}-rank ceiling"
            )
        for axis, extent in enumerate(shape):
            if type(extent) is not int or extent < 0:
                raise ShapeError(
                    f"{operation} input {index} shape[{axis}] must be a non-negative integer"
                )
            if extent > L1_LOSS_OUTPUT_EXTENT_MAX:
                raise ShapeError(
                    f"{operation} input {index} extent {extent} on axis {axis} exceeds the "
                    f"{L1_LOSS_OUTPUT_EXTENT_MAX}-element per-axis ceiling"
                )
        dtype = getattr(source, "dtype", None)
        if dtype not in _FLOATING_DTYPES:
            raise ShapeError(
                f"{operation} input {index} dtype {dtype!r} is not supported; "
                "expected float16, float32, or float64"
            )
        shapes.append(shape)
        dtypes.append(dtype)

    input_shape = shapes[0]
    if shapes[1] != input_shape:
        raise ShapeError(
            f"{operation}: input shape {input_shape} must equal target shape {shapes[1]}"
        )
    if batch_rank < 0 or batch_rank > len(input_shape):
        raise ShapeError(
            f"{operation}: batch_rank {batch_rank} is out of range for rank {len(input_shape)}"
        )

    output_dtype = promote_variadic_dtypes(tuple(dtypes), operation.upper())
    output_shape = input_shape if reduction == "none" else input_shape[:batch_rank]
    reduced_elements = _elementwise_loss_checked_product(
        input_shape[batch_rank:],
        L1_LOSS_WORK_ELEMENT_MAX,
    )
    input_elements = _elementwise_loss_checked_product(
        input_shape,
        L1_LOSS_WORK_ELEMENT_MAX,
    )
    capacity_elements = _elementwise_loss_checked_product(
        tuple(max(1, extent) for extent in input_shape),
        L1_LOSS_WORK_ELEMENT_MAX,
    )
    if capacity_elements > L1_LOSS_WORK_ELEMENT_MAX // work_visit_factor:
        work_elements = L1_LOSS_WORK_ELEMENT_MAX + 1
    else:
        work_elements = capacity_elements * work_visit_factor
    if work_elements > L1_LOSS_WORK_ELEMENT_MAX:
        raise ShapeError(
            f"{operation}: projected work exceeds the "
            f"{L1_LOSS_WORK_ELEMENT_MAX}-element-visit ceiling"
        )

    output_elements = _elementwise_loss_checked_product(
        output_shape,
        L1_LOSS_OUTPUT_BYTE_MAX,
    )
    output_bytes = output_elements * _VARIADIC_DTYPE_BYTES[output_dtype]
    if output_bytes > L1_LOSS_OUTPUT_BYTE_MAX:
        raise ShapeError(
            f"{operation}: output requires {output_bytes} bytes, exceeding the "
            f"{L1_LOSS_OUTPUT_BYTE_MAX}-byte ceiling"
        )

    compute_dtype = "float32" if output_dtype == "float16" else output_dtype
    compute_bytes = _VARIADIC_DTYPE_BYTES[compute_dtype]
    cast_bytes = sum(
        input_elements * compute_bytes for dtype in dtypes if dtype != compute_dtype
    )
    # The peak closure lifetime retains both source cotangents, input casts,
    # the output, and the operation-specific compute/mask buffers. Explicit
    # counts prevent a new loss from silently inheriting a smaller operation's
    # allocation proof.
    intermediate_bytes = input_elements * (
        compute_bytes * compute_buffers + mask_buffers
    )
    gradient_bytes = input_elements * sum(
        _VARIADIC_DTYPE_BYTES[dtype] for dtype in dtypes
    )
    workspace_bytes = output_bytes + cast_bytes + intermediate_bytes + gradient_bytes
    if workspace_bytes > L1_LOSS_WORKSPACE_BYTE_MAX:
        raise ShapeError(
            f"{operation}: projected output/cast/intermediate/gradient workspace "
            f"requires {workspace_bytes} bytes, exceeding the "
            f"{L1_LOSS_WORKSPACE_BYTE_MAX}-byte ceiling"
        )

    return _ElementwiseLossGeometry(
        input_shape=input_shape,
        input_dtypes=(dtypes[0], dtypes[1]),
        output_shape=output_shape,
        output_dtype=output_dtype,
        reduction=reduction,
        batch_rank=batch_rank,
        reduced_elements=reduced_elements,
        work_elements=work_elements,
        workspace_bytes=workspace_bytes,
    )


def infer_l1_loss_contract(
    inputs: tuple[Any, ...],
    reduction: Any,
    batch_rank: Any = 0,
) -> L1LossContract:
    geometry = _infer_elementwise_loss_geometry(
        inputs,
        reduction,
        batch_rank,
        "l1_loss",
        L1_LOSS_WORK_VISIT_FACTOR,
    )
    return L1LossContract(**{
        field: getattr(geometry, field)
        for field in L1LossContract.__dataclass_fields__
    })


def normalize_smooth_l1_beta(beta: Any) -> float:
    if type(beta) not in _SMOOTH_L1_BETA_TYPES or type(beta) is bool:
        raise ShapeError(
            "smooth_l1_loss: beta must be an exact real scalar"
        )
    try:
        normalized = float(beta)
    except (OverflowError, ValueError) as exc:
        raise ShapeError("smooth_l1_loss: beta must be finite") from exc
    if not math.isfinite(normalized):
        raise ShapeError("smooth_l1_loss: beta must be finite")
    if normalized < 0.0:
        raise ShapeError(
            f"smooth_l1_loss: beta must be non-negative, got {normalized}"
        )
    return 0.0 if normalized == 0.0 else normalized


def infer_smooth_l1_loss_contract(
    inputs: tuple[Any, ...],
    beta: Any,
    reduction: Any,
    batch_rank: Any = 0,
) -> SmoothL1LossContract:
    normalized_beta = normalize_smooth_l1_beta(beta)
    geometry = _infer_elementwise_loss_geometry(
        inputs,
        reduction,
        batch_rank,
        "smooth_l1_loss",
        SMOOTH_L1_LOSS_WORK_VISIT_FACTOR,
    )
    compute_dtype = "float32" if geometry.output_dtype == "float16" else geometry.output_dtype
    with np.errstate(over="ignore", under="ignore"):
        compute_beta = float(
            np.asarray(normalized_beta, dtype=np.dtype(compute_dtype)).item()
        )
    if normalized_beta > 0.0 and (compute_beta == 0.0 or not math.isfinite(compute_beta)):
        raise ShapeError(
            f"smooth_l1_loss: beta {normalized_beta} is not representable as "
            f"a finite nonzero {compute_dtype} scalar"
        )
    values = {
        field: getattr(geometry, field)
        for field in _ElementwiseLossGeometry.__dataclass_fields__
    }
    return SmoothL1LossContract(**values, beta=compute_beta)


def infer_binary_cross_entropy_contract(
    inputs: tuple[Any, ...],
    reduction: Any,
    batch_rank: Any = 0,
) -> BinaryCrossEntropyContract:
    geometry = _infer_elementwise_loss_geometry(
        inputs,
        reduction,
        batch_rank,
        "binary_cross_entropy",
        BINARY_CROSS_ENTROPY_WORK_VISIT_FACTOR,
        compute_buffers=4,
        mask_buffers=1,
    )
    return BinaryCrossEntropyContract(**{
        field: getattr(geometry, field)
        for field in BinaryCrossEntropyContract.__dataclass_fields__
    })


def infer_binary_cross_entropy_with_logits_contract(
    inputs: tuple[Any, ...],
    reduction: Any,
    batch_rank: Any = 0,
) -> BinaryCrossEntropyWithLogitsContract:
    geometry = _infer_elementwise_loss_geometry(
        inputs,
        reduction,
        batch_rank,
        "binary_cross_entropy_with_logits",
        BINARY_CROSS_ENTROPY_WITH_LOGITS_WORK_VISIT_FACTOR,
        compute_buffers=4,
        mask_buffers=1,
    )
    return BinaryCrossEntropyWithLogitsContract(**{
        field: getattr(geometry, field)
        for field in BinaryCrossEntropyWithLogitsContract.__dataclass_fields__
    })


def infer_kl_div_contract(
    inputs: tuple[Any, ...],
    reduction: Any,
    log_target: Any,
    batch_rank: Any = 0,
) -> KlDivContract:
    if type(log_target) is not bool:
        raise ShapeError(
            "kl_div: log_target must be an exact bool, got "
            f"{type(log_target).__name__}"
        )
    geometry = _infer_elementwise_loss_geometry(
        inputs,
        reduction,
        batch_rank,
        "kl_div",
        KL_DIV_WORK_VISIT_FACTOR,
        compute_buffers=4,
        mask_buffers=1,
        allow_batchmean=True,
    )
    user_rank = len(geometry.input_shape) - geometry.batch_rank
    batch_denominator = (
        1 if user_rank == 0 else geometry.input_shape[geometry.batch_rank]
    )
    values = {
        field: getattr(geometry, field)
        for field in _ElementwiseLossGeometry.__dataclass_fields__
    }
    return KlDivContract(
        **values,
        log_target=log_target,
        batch_denominator=batch_denominator,
    )


def _normalize_nll_ignore_index(ignore_index: Any) -> int:
    if type(ignore_index) not in _PAD_INTEGER_TYPES or type(ignore_index) is bool:
        raise ShapeError("nll_loss: ignore_index must be an exact integer")
    normalized = int(ignore_index)
    if normalized < -(1 << 63) or normalized > (1 << 63) - 1:
        raise ShapeError("nll_loss: ignore_index must fit signed int64")
    return normalized


def infer_nll_loss_contract(
    inputs: tuple[Any, ...],
    reduction: Any,
    ignore_index: Any,
    has_weight: Any,
    batch_rank: Any = 0,
) -> NllLossContract:
    if type(has_weight) is not bool:
        raise ShapeError("nll_loss: has_weight must be an exact bool")
    expected_arity = 3 if has_weight else 2
    if type(inputs) is not tuple or len(inputs) != expected_arity:
        raise ShapeError(
            f"nll_loss requires exactly {expected_arity} tensor inputs for "
            f"has_weight={has_weight}"
        )
    if type(reduction) is not str:
        raise ShapeError(
            "nll_loss: reduction must be a string, got "
            f"{type(reduction).__name__}"
        )
    if reduction not in ("none", "sum", "mean"):
        raise ShapeError(
            "nll_loss: reduction must be 'none', 'sum', or 'mean', got "
            f"{reduction!r}"
        )
    if type(batch_rank) is not int:
        raise ShapeError("nll_loss: batch_rank must be a normalized integer")
    normalized_ignore_index = _normalize_nll_ignore_index(ignore_index)

    shapes: list[Tuple[int, ...]] = []
    dtypes: list[str] = []
    for index, source in enumerate(inputs):
        shape = getattr(source, "shape", None)
        if type(shape) is not tuple:
            raise ShapeError(f"nll_loss input {index} shape must be a tuple")
        if len(shape) > L1_LOSS_RANK_MAX:
            raise ShapeError(
                f"nll_loss input {index} rank {len(shape)} exceeds the "
                f"{L1_LOSS_RANK_MAX}-rank ceiling"
            )
        for axis, extent in enumerate(shape):
            if type(extent) is not int or extent < 0:
                raise ShapeError(
                    f"nll_loss input {index} shape[{axis}] must be a "
                    "non-negative integer"
                )
            if extent > L1_LOSS_OUTPUT_EXTENT_MAX:
                raise ShapeError(
                    f"nll_loss input {index} extent {extent} on axis {axis} "
                    f"exceeds the {L1_LOSS_OUTPUT_EXTENT_MAX}-element "
                    "per-axis ceiling"
                )
        shapes.append(shape)
        dtypes.append(getattr(source, "dtype", None))

    input_shape, target_shape = shapes[:2]
    input_dtype, target_dtype = dtypes[:2]
    if input_dtype not in _FLOATING_DTYPES:
        raise ShapeError(
            f"nll_loss input dtype {input_dtype!r} is not supported; expected "
            "float16, float32, or float64"
        )
    if target_dtype != "int64":
        raise ShapeError(
            f"nll_loss target dtype {target_dtype!r} is not supported; "
            "expected int64"
        )
    if batch_rank < 0 or batch_rank >= len(input_shape):
        raise ShapeError(
            f"nll_loss: batch_rank {batch_rank} leaves no user input "
            f"dimension in shape {input_shape}"
        )
    user_rank = len(input_shape) - batch_rank
    class_axis = batch_rank if user_rank == 1 else batch_rank + 1
    class_count = input_shape[class_axis]
    expected_target_shape = (
        input_shape[:class_axis] + input_shape[class_axis + 1:]
    )
    if target_shape != expected_target_shape:
        raise ShapeError(
            f"nll_loss: target shape {target_shape} must equal input shape "
            f"{input_shape} with class axis {class_axis} removed "
            f"({expected_target_shape})"
        )

    weight_shape: Tuple[int, ...] | None = None
    if has_weight:
        weight_shape = shapes[2]
        expected_weight_shape = input_shape[:batch_rank] + (class_count,)
        if weight_shape != expected_weight_shape:
            raise ShapeError(
                f"nll_loss: weight shape {weight_shape} must equal mapped "
                f"batch prefix plus class count {expected_weight_shape}"
            )
        if dtypes[2] != input_dtype:
            raise ShapeError(
                f"nll_loss: weight dtype {dtypes[2]!r} must equal input "
                f"dtype {input_dtype!r}"
            )

    output_shape = target_shape if reduction == "none" else input_shape[:batch_rank]
    input_elements = _elementwise_loss_checked_product(
        input_shape, L1_LOSS_WORK_ELEMENT_MAX
    )
    target_elements = _elementwise_loss_checked_product(
        target_shape, L1_LOSS_WORK_ELEMENT_MAX
    )
    weight_elements = (
        _elementwise_loss_checked_product(
            weight_shape, L1_LOSS_WORK_ELEMENT_MAX
        )
        if weight_shape is not None
        else 0
    )
    capacity_elements = max(
        _elementwise_loss_checked_product(
            tuple(max(1, extent) for extent in shape),
            L1_LOSS_WORK_ELEMENT_MAX,
        )
        for shape in shapes
    )
    if capacity_elements > L1_LOSS_WORK_ELEMENT_MAX // NLL_LOSS_WORK_VISIT_FACTOR:
        work_elements = L1_LOSS_WORK_ELEMENT_MAX + 1
    else:
        work_elements = capacity_elements * NLL_LOSS_WORK_VISIT_FACTOR
    if work_elements > L1_LOSS_WORK_ELEMENT_MAX:
        raise ShapeError(
            "nll_loss: projected work exceeds the "
            f"{L1_LOSS_WORK_ELEMENT_MAX}-element-visit ceiling"
        )

    output_elements = _elementwise_loss_checked_product(
        output_shape, L1_LOSS_OUTPUT_BYTE_MAX
    )
    output_bytes = output_elements * _VARIADIC_DTYPE_BYTES[input_dtype]
    if output_bytes > L1_LOSS_OUTPUT_BYTE_MAX:
        raise ShapeError(
            f"nll_loss: output requires {output_bytes} bytes, exceeding the "
            f"{L1_LOSS_OUTPUT_BYTE_MAX}-byte ceiling"
        )
    compute_dtype = "float32" if input_dtype == "float16" else input_dtype
    compute_bytes = _VARIADIC_DTYPE_BYTES[compute_dtype]
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
    if workspace_bytes > L1_LOSS_WORKSPACE_BYTE_MAX:
        raise ShapeError(
            "nll_loss: projected output/cast/gather/gradient workspace "
            f"requires {workspace_bytes} bytes, exceeding the "
            f"{L1_LOSS_WORKSPACE_BYTE_MAX}-byte ceiling"
        )

    return NllLossContract(
        input_shape=input_shape,
        target_shape=target_shape,
        weight_shape=weight_shape,
        output_shape=output_shape,
        output_dtype=input_dtype,
        reduction=reduction,
        batch_rank=batch_rank,
        class_axis=class_axis,
        class_count=class_count,
        ignore_index=normalized_ignore_index,
        has_weight=has_weight,
        input_elements=input_elements,
        target_elements=target_elements,
        work_elements=work_elements,
        workspace_bytes=workspace_bytes,
    )


def _normalize_cross_entropy_ignore_index(ignore_index: Any) -> int:
    if type(ignore_index) not in _PAD_INTEGER_TYPES or type(ignore_index) is bool:
        raise ShapeError("cross_entropy: ignore_index must be an exact integer")
    normalized = int(ignore_index)
    if normalized < -(1 << 63) or normalized > (1 << 63) - 1:
        raise ShapeError("cross_entropy: ignore_index must fit signed int64")
    return normalized


def _normalize_cross_entropy_label_smoothing(label_smoothing: Any) -> float:
    if type(label_smoothing) not in _PAD_VALUE_TYPES or type(label_smoothing) is bool:
        raise ShapeError(
            "cross_entropy: label_smoothing must be an exact real scalar"
        )
    normalized = float(label_smoothing)
    if not math.isfinite(normalized) or normalized < 0.0 or normalized > 1.0:
        raise ShapeError(
            "cross_entropy: label_smoothing must be finite and in [0, 1], "
            f"got {normalized!r}"
        )
    return normalized


def infer_cross_entropy_contract(
    inputs: tuple[Any, ...],
    reduction: Any,
    ignore_index: Any,
    has_weight: Any,
    label_smoothing: Any,
    target_mode: Any,
    batch_rank: Any = 0,
) -> CrossEntropyContract:
    if type(has_weight) is not bool:
        raise ShapeError("cross_entropy: has_weight must be an exact bool")
    expected_arity = 3 if has_weight else 2
    if type(inputs) is not tuple or len(inputs) != expected_arity:
        raise ShapeError(
            f"cross_entropy requires exactly {expected_arity} tensor inputs "
            f"for has_weight={has_weight}"
        )
    if type(reduction) is not str:
        raise ShapeError(
            "cross_entropy: reduction must be a string, got "
            f"{type(reduction).__name__}"
        )
    if reduction not in ("none", "sum", "mean"):
        raise ShapeError(
            "cross_entropy: reduction must be 'none', 'sum', or 'mean', got "
            f"{reduction!r}"
        )
    if type(batch_rank) is not int:
        raise ShapeError("cross_entropy: batch_rank must be a normalized integer")
    if type(target_mode) is not str or target_mode not in (
        "indices",
        "probabilities",
    ):
        raise ShapeError(
            "cross_entropy: target_mode must be 'indices' or 'probabilities'"
        )
    normalized_ignore_index = _normalize_cross_entropy_ignore_index(ignore_index)
    normalized_smoothing = _normalize_cross_entropy_label_smoothing(
        label_smoothing
    )

    shapes: list[Tuple[int, ...]] = []
    dtypes: list[str] = []
    for index, source in enumerate(inputs):
        shape = getattr(source, "shape", None)
        if type(shape) is not tuple:
            raise ShapeError(f"cross_entropy input {index} shape must be a tuple")
        if len(shape) > L1_LOSS_RANK_MAX:
            raise ShapeError(
                f"cross_entropy input {index} rank {len(shape)} exceeds the "
                f"{L1_LOSS_RANK_MAX}-rank ceiling"
            )
        for axis, extent in enumerate(shape):
            if type(extent) is not int or extent < 0:
                raise ShapeError(
                    f"cross_entropy input {index} shape[{axis}] must be a "
                    "non-negative integer"
                )
            if extent > L1_LOSS_OUTPUT_EXTENT_MAX:
                raise ShapeError(
                    f"cross_entropy input {index} extent {extent} on axis "
                    f"{axis} exceeds the {L1_LOSS_OUTPUT_EXTENT_MAX}-element "
                    "per-axis ceiling"
                )
        shapes.append(shape)
        dtypes.append(getattr(source, "dtype", None))

    input_shape, target_shape = shapes[:2]
    input_dtype, target_dtype = dtypes[:2]
    if input_dtype not in _FLOATING_DTYPES:
        raise ShapeError(
            f"cross_entropy input dtype {input_dtype!r} is not supported; "
            "expected float16, float32, or float64"
        )
    if batch_rank < 0 or batch_rank >= len(input_shape):
        raise ShapeError(
            f"cross_entropy: batch_rank {batch_rank} leaves no user input "
            f"dimension in shape {input_shape}"
        )
    user_rank = len(input_shape) - batch_rank
    class_axis = batch_rank if user_rank == 1 else batch_rank + 1
    class_count = input_shape[class_axis]
    if class_count == 0:
        raise ShapeError(
            "cross_entropy: class dimension must contain at least one class"
        )
    position_shape = (
        input_shape[:class_axis] + input_shape[class_axis + 1:]
    )
    derived_target_mode = (
        "probabilities" if target_shape == input_shape else "indices"
    )
    if target_mode != derived_target_mode:
        raise ShapeError(
            f"cross_entropy: target_mode {target_mode!r} does not match "
            f"shape-derived mode {derived_target_mode!r}"
        )
    if target_mode == "probabilities":
        if target_dtype != input_dtype:
            raise ShapeError(
                f"cross_entropy probability target dtype {target_dtype!r} "
                f"must equal input dtype {input_dtype!r}"
            )
        if normalized_ignore_index >= 0:
            raise ShapeError(
                "cross_entropy: ignore_index is not supported for floating "
                "point targets unless it is negative"
            )
    else:
        if target_shape != position_shape:
            raise ShapeError(
                f"cross_entropy: index target shape {target_shape} must equal "
                f"input shape {input_shape} with class axis {class_axis} "
                f"removed ({position_shape})"
            )
        if target_dtype != "int64":
            raise ShapeError(
                f"cross_entropy index target dtype {target_dtype!r} is not "
                "supported; expected int64"
            )

    weight_shape: Tuple[int, ...] | None = None
    if has_weight:
        weight_shape = shapes[2]
        expected_weight_shape = input_shape[:batch_rank] + (class_count,)
        if weight_shape != expected_weight_shape:
            raise ShapeError(
                f"cross_entropy: weight shape {weight_shape} must equal "
                f"mapped batch prefix plus class count {expected_weight_shape}"
            )
        if dtypes[2] != input_dtype:
            raise ShapeError(
                f"cross_entropy: weight dtype {dtypes[2]!r} must equal input "
                f"dtype {input_dtype!r}"
            )

    output_shape = position_shape if reduction == "none" else input_shape[:batch_rank]
    input_elements = _elementwise_loss_checked_product(
        input_shape, L1_LOSS_WORK_ELEMENT_MAX
    )
    target_elements = _elementwise_loss_checked_product(
        target_shape, L1_LOSS_WORK_ELEMENT_MAX
    )
    weight_elements = (
        _elementwise_loss_checked_product(
            weight_shape, L1_LOSS_WORK_ELEMENT_MAX
        )
        if weight_shape is not None
        else 0
    )
    capacity_elements = max(
        _elementwise_loss_checked_product(
            tuple(max(1, extent) for extent in shape),
            L1_LOSS_WORK_ELEMENT_MAX,
        )
        for shape in shapes
    )
    if (
        capacity_elements
        > L1_LOSS_WORK_ELEMENT_MAX // CROSS_ENTROPY_WORK_VISIT_FACTOR
    ):
        work_elements = L1_LOSS_WORK_ELEMENT_MAX + 1
    else:
        work_elements = capacity_elements * CROSS_ENTROPY_WORK_VISIT_FACTOR
    if work_elements > L1_LOSS_WORK_ELEMENT_MAX:
        raise ShapeError(
            "cross_entropy: projected work exceeds the "
            f"{L1_LOSS_WORK_ELEMENT_MAX}-element-visit ceiling"
        )

    output_elements = _elementwise_loss_checked_product(
        output_shape, L1_LOSS_OUTPUT_BYTE_MAX
    )
    output_bytes = output_elements * _VARIADIC_DTYPE_BYTES[input_dtype]
    if output_bytes > L1_LOSS_OUTPUT_BYTE_MAX:
        raise ShapeError(
            f"cross_entropy: output requires {output_bytes} bytes, exceeding "
            f"the {L1_LOSS_OUTPUT_BYTE_MAX}-byte ceiling"
        )
    compute_dtype = "float32" if input_dtype == "float16" else input_dtype
    compute_bytes = _VARIADIC_DTYPE_BYTES[compute_dtype]
    cast_bytes = 0
    if input_dtype != compute_dtype:
        cast_bytes += input_elements * compute_bytes
        if target_mode == "probabilities":
            cast_bytes += target_elements * compute_bytes
        cast_bytes += weight_elements * compute_bytes
    position_elements = _elementwise_loss_checked_product(
        position_shape, L1_LOSS_WORK_ELEMENT_MAX
    )
    workspace_bytes = (
        output_bytes
        + cast_bytes
        + input_elements * compute_bytes * 6
        + position_elements * (2 * compute_bytes + 8 + 1)
    )
    if workspace_bytes > L1_LOSS_WORKSPACE_BYTE_MAX:
        raise ShapeError(
            "cross_entropy: projected stable-softmax/target/gradient "
            f"workspace requires {workspace_bytes} bytes, exceeding the "
            f"{L1_LOSS_WORKSPACE_BYTE_MAX}-byte ceiling"
        )

    return CrossEntropyContract(
        input_shape=input_shape,
        target_shape=target_shape,
        weight_shape=weight_shape,
        output_shape=output_shape,
        output_dtype=input_dtype,
        target_dtype=target_dtype,
        target_mode=target_mode,
        reduction=reduction,
        batch_rank=batch_rank,
        class_axis=class_axis,
        class_count=class_count,
        ignore_index=normalized_ignore_index,
        has_weight=has_weight,
        label_smoothing=normalized_smoothing,
        input_elements=input_elements,
        target_elements=target_elements,
        work_elements=work_elements,
        workspace_bytes=workspace_bytes,
    )


def _validate_elementwise_loss_runtime_arrays(
    contract: Any,
    arrays: tuple[np.ndarray, ...],
    operation: str,
) -> None:
    if type(arrays) is not tuple or len(arrays) != 2:
        raise ShapeError(f"{operation} execution requires exactly two arrays")
    for index, (array, dtype) in enumerate(zip(arrays, contract.input_dtypes)):
        if type(array) is not np.ndarray:
            raise ShapeError(
                f"{operation} runtime input {index} must be an exact ndarray"
            )
        if tuple(array.shape) != contract.input_shape:
            raise ShapeError(
                f"{operation} runtime input {index} shape {tuple(array.shape)} does not "
                f"match {contract.input_shape}"
            )
        if array.dtype.name != dtype:
            raise ShapeError(
                f"{operation} runtime input {index} dtype {array.dtype.name!r} does not "
                f"match {dtype!r}"
            )


def _elementwise_loss_compute_dtype(contract: Any) -> str:
    return "float32" if contract.output_dtype == "float16" else contract.output_dtype


def _execute_elementwise_loss_reduction(
    contract: Any,
    per_element: np.ndarray,
) -> np.ndarray:
    compute_dtype = _elementwise_loss_compute_dtype(contract)
    reduction_axes = tuple(range(contract.batch_rank, len(contract.input_shape)))
    if contract.reduction == "none" or not reduction_axes:
        result = per_element
    elif contract.reduction == "sum":
        result = per_element.sum(axis=reduction_axes, dtype=np.dtype(compute_dtype))
    elif contract.reduction == "mean" and contract.reduced_elements == 0:
        result = np.full(contract.output_shape, np.nan, dtype=np.dtype(compute_dtype))
    elif contract.reduction == "mean":
        result = per_element.sum(
            axis=reduction_axes,
            dtype=np.dtype(compute_dtype),
        ) / float(contract.reduced_elements)
    else:
        numerator = per_element.sum(
            axis=reduction_axes,
            dtype=np.dtype(compute_dtype),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.divide(numerator, float(contract.batch_denominator))
    return np.array(result, dtype=np.dtype(contract.output_dtype), copy=True)


def _elementwise_loss_upstream(
    contract: Any,
    dy: np.ndarray,
    operation: str,
) -> np.ndarray:
    if type(dy) is not np.ndarray:
        raise ShapeError(f"{operation} VJP cotangent must be an exact ndarray")
    if tuple(dy.shape) != contract.output_shape or dy.dtype.name != contract.output_dtype:
        raise ShapeError(
            f"{operation} VJP cotangent must have shape {contract.output_shape} and "
            f"dtype {contract.output_dtype!r}"
        )
    compute_dtype = _elementwise_loss_compute_dtype(contract)
    upstream = dy.astype(np.dtype(compute_dtype), copy=False)
    if contract.reduction != "none":
        user_rank = len(contract.input_shape) - contract.batch_rank
        upstream = upstream.reshape(contract.output_shape + (1,) * user_rank)
        upstream = np.broadcast_to(upstream, contract.input_shape)
        if contract.reduction == "mean":
            upstream = upstream / float(contract.reduced_elements)
        elif contract.reduction == "batchmean":
            with np.errstate(divide="ignore", invalid="ignore"):
                upstream = upstream / float(contract.batch_denominator)
    return upstream


def execute_l1_loss_arrays(
    contract: L1LossContract,
    arrays: tuple[np.ndarray, ...],
) -> np.ndarray:
    _validate_elementwise_loss_runtime_arrays(contract, arrays, "l1_loss")
    compute_dtype = _elementwise_loss_compute_dtype(contract)
    left = arrays[0].astype(np.dtype(compute_dtype), copy=False)
    right = arrays[1].astype(np.dtype(compute_dtype), copy=False)
    per_element = np.abs(left - right)
    return _execute_elementwise_loss_reduction(contract, per_element)


def execute_l1_loss_vjp_array(
    contract: L1LossContract,
    operand: int,
    dy: np.ndarray,
    arrays: tuple[np.ndarray, ...],
) -> np.ndarray:
    _validate_elementwise_loss_runtime_arrays(contract, arrays, "l1_loss")
    if type(operand) is not int or operand not in (0, 1):
        raise ShapeError("l1_loss VJP operand must be 0 or 1")
    upstream = _elementwise_loss_upstream(contract, dy, "l1_loss")
    target_dtype = contract.input_dtypes[operand]
    if contract.reduced_elements == 0:
        return np.zeros(contract.input_shape, dtype=np.dtype(target_dtype))

    compute_dtype = _elementwise_loss_compute_dtype(contract)
    left = arrays[0].astype(np.dtype(compute_dtype), copy=False)
    right = arrays[1].astype(np.dtype(compute_dtype), copy=False)
    signed = np.empty(contract.input_shape, dtype=np.dtype(compute_dtype))
    np.subtract(left, right, out=signed)
    np.sign(signed, out=signed)
    gradient = np.empty(contract.input_shape, dtype=np.dtype(compute_dtype))
    np.multiply(signed, upstream, out=gradient)
    if operand == 1:
        np.negative(gradient, out=gradient)
        gradient[signed == 0] = 0.0
    if gradient.dtype.name == target_dtype:
        return gradient
    return gradient.astype(np.dtype(target_dtype), copy=True)


def execute_smooth_l1_loss_arrays(
    contract: SmoothL1LossContract,
    arrays: tuple[np.ndarray, ...],
) -> np.ndarray:
    _validate_elementwise_loss_runtime_arrays(contract, arrays, "smooth_l1_loss")
    compute_dtype = _elementwise_loss_compute_dtype(contract)
    dtype = np.dtype(compute_dtype)
    left = arrays[0].astype(dtype, copy=False)
    right = arrays[1].astype(dtype, copy=False)
    difference = np.empty(contract.input_shape, dtype=dtype)
    np.subtract(left, right, out=difference)
    if contract.beta == 0.0:
        np.abs(difference, out=difference)
        per_element = difference
    else:
        per_element = np.empty(contract.input_shape, dtype=dtype)
        np.abs(difference, out=per_element)
        quadratic_mask = per_element < contract.beta
        np.subtract(per_element, contract.beta * 0.5, out=per_element)
        quadratic = np.empty(contract.input_shape, dtype=dtype)
        np.multiply(difference, difference, out=quadratic)
        np.divide(quadratic, contract.beta, out=quadratic)
        np.multiply(quadratic, 0.5, out=quadratic)
        np.copyto(per_element, quadratic, where=quadratic_mask)
    return _execute_elementwise_loss_reduction(contract, per_element)


def execute_smooth_l1_loss_vjp_array(
    contract: SmoothL1LossContract,
    operand: int,
    dy: np.ndarray,
    arrays: tuple[np.ndarray, ...],
) -> np.ndarray:
    _validate_elementwise_loss_runtime_arrays(contract, arrays, "smooth_l1_loss")
    if type(operand) is not int or operand not in (0, 1):
        raise ShapeError("smooth_l1_loss VJP operand must be 0 or 1")
    upstream = _elementwise_loss_upstream(contract, dy, "smooth_l1_loss")
    target_dtype = contract.input_dtypes[operand]
    if contract.reduced_elements == 0:
        return np.zeros(contract.input_shape, dtype=np.dtype(target_dtype))

    compute_dtype = _elementwise_loss_compute_dtype(contract)
    dtype = np.dtype(compute_dtype)
    left = arrays[0].astype(dtype, copy=False)
    right = arrays[1].astype(dtype, copy=False)
    difference = np.empty(contract.input_shape, dtype=dtype)
    np.subtract(left, right, out=difference)
    if contract.beta == 0.0:
        derivative = difference
        np.sign(derivative, out=derivative)
    else:
        derivative = np.empty(contract.input_shape, dtype=dtype)
        np.abs(difference, out=derivative)
        quadratic_mask = derivative < contract.beta
        np.sign(difference, out=derivative)
        np.divide(difference, contract.beta, out=difference)
        np.copyto(derivative, difference, where=quadratic_mask)
    np.multiply(derivative, upstream, out=difference)
    if operand == 1:
        np.negative(difference, out=difference)
        difference[derivative == 0] = 0.0
    if difference.dtype.name == target_dtype:
        return difference
    return difference.astype(np.dtype(target_dtype), copy=True)


def _validate_binary_cross_entropy_domain(
    arrays: tuple[np.ndarray, ...],
) -> None:
    for index, label in ((0, "input"), (1, "target")):
        array = arrays[index]
        if array.size == 0:
            continue
        if not bool(np.isfinite(array).all()):
            raise ShapeError(
                f"binary_cross_entropy: all elements of {label} must be finite "
                "and between 0 and 1"
            )
        minimum = array.min()
        maximum = array.max()
        if bool(minimum < 0.0) or bool(maximum > 1.0):
            raise ShapeError(
                f"binary_cross_entropy: all elements of {label} must be between 0 and 1"
            )


def execute_binary_cross_entropy_arrays(
    contract: BinaryCrossEntropyContract,
    arrays: tuple[np.ndarray, ...],
) -> np.ndarray:
    _validate_elementwise_loss_runtime_arrays(
        contract, arrays, "binary_cross_entropy"
    )
    _validate_binary_cross_entropy_domain(arrays)
    compute_dtype = _elementwise_loss_compute_dtype(contract)
    dtype = np.dtype(compute_dtype)
    probabilities = arrays[0].astype(dtype, copy=False)
    targets = arrays[1].astype(dtype, copy=False)

    log_probability = np.empty(contract.input_shape, dtype=dtype)
    log_one_minus_probability = np.empty(contract.input_shape, dtype=dtype)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.log(probabilities, out=log_probability)
        np.subtract(1.0, probabilities, out=log_one_minus_probability)
        np.log(log_one_minus_probability, out=log_one_minus_probability)
    np.maximum(
        log_probability,
        BINARY_CROSS_ENTROPY_LOG_FLOOR,
        out=log_probability,
    )
    np.maximum(
        log_one_minus_probability,
        BINARY_CROSS_ENTROPY_LOG_FLOOR,
        out=log_one_minus_probability,
    )

    per_element = np.empty(contract.input_shape, dtype=dtype)
    np.multiply(targets, log_probability, out=per_element)
    np.subtract(1.0, targets, out=log_probability)
    np.multiply(
        log_probability,
        log_one_minus_probability,
        out=log_probability,
    )
    np.add(per_element, log_probability, out=per_element)
    np.negative(per_element, out=per_element)
    per_element[per_element == 0.0] = 0.0
    return _execute_elementwise_loss_reduction(contract, per_element)


def execute_binary_cross_entropy_vjp_array(
    contract: BinaryCrossEntropyContract,
    operand: int,
    dy: np.ndarray,
    arrays: tuple[np.ndarray, ...],
) -> np.ndarray:
    _validate_elementwise_loss_runtime_arrays(
        contract, arrays, "binary_cross_entropy"
    )
    _validate_binary_cross_entropy_domain(arrays)
    if type(operand) is not int or operand not in (0, 1):
        raise ShapeError("binary_cross_entropy VJP operand must be 0 or 1")
    upstream = _elementwise_loss_upstream(
        contract, dy, "binary_cross_entropy"
    )
    target_dtype = contract.input_dtypes[operand]
    if contract.reduced_elements == 0:
        return np.zeros(contract.input_shape, dtype=np.dtype(target_dtype))

    compute_dtype = _elementwise_loss_compute_dtype(contract)
    dtype = np.dtype(compute_dtype)
    probabilities = arrays[0].astype(dtype, copy=False)
    targets = arrays[1].astype(dtype, copy=False)
    gradient = np.empty(contract.input_shape, dtype=dtype)
    scratch = np.empty(contract.input_shape, dtype=dtype)
    if operand == 0:
        np.subtract(probabilities, targets, out=gradient)
        np.subtract(1.0, probabilities, out=scratch)
        np.multiply(scratch, probabilities, out=scratch)
        epsilon = np.asarray(
            BINARY_CROSS_ENTROPY_GRAD_EPSILON,
            dtype=dtype,
        ).item()
        np.maximum(scratch, epsilon, out=scratch)
        np.divide(gradient, scratch, out=gradient)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            np.log(probabilities, out=gradient)
            np.subtract(1.0, probabilities, out=scratch)
            np.log(scratch, out=scratch)
        np.subtract(scratch, gradient, out=gradient)
    np.multiply(gradient, upstream, out=gradient)
    if gradient.dtype.name == target_dtype:
        return gradient
    return gradient.astype(np.dtype(target_dtype), copy=True)


def execute_binary_cross_entropy_with_logits_arrays(
    contract: BinaryCrossEntropyWithLogitsContract,
    arrays: tuple[np.ndarray, ...],
) -> np.ndarray:
    _validate_elementwise_loss_runtime_arrays(
        contract, arrays, "binary_cross_entropy_with_logits"
    )
    compute_dtype = _elementwise_loss_compute_dtype(contract)
    dtype = np.dtype(compute_dtype)
    logits = arrays[0].astype(dtype, copy=False)
    targets = arrays[1].astype(dtype, copy=False)

    # Match ATen's stable (1 - target) * logits - log_sigmoid(logits)
    # formulation without evaluating exp(logits) for large positive values.
    softplus_negative = np.empty(contract.input_shape, dtype=dtype)
    np.abs(logits, out=softplus_negative)
    np.negative(softplus_negative, out=softplus_negative)
    with np.errstate(over="ignore", invalid="ignore"):
        np.exp(softplus_negative, out=softplus_negative)
        np.log1p(softplus_negative, out=softplus_negative)
    negative_logits = np.empty(contract.input_shape, dtype=dtype)
    np.negative(logits, out=negative_logits)
    np.maximum(negative_logits, 0.0, out=negative_logits)
    np.add(softplus_negative, negative_logits, out=softplus_negative)

    per_element = np.empty(contract.input_shape, dtype=dtype)
    np.subtract(1.0, targets, out=per_element)
    np.multiply(per_element, logits, out=per_element)
    np.add(per_element, softplus_negative, out=per_element)
    per_element[per_element == 0.0] = 0.0
    return _execute_elementwise_loss_reduction(contract, per_element)


def execute_binary_cross_entropy_with_logits_vjp_array(
    contract: BinaryCrossEntropyWithLogitsContract,
    operand: int,
    dy: np.ndarray,
    arrays: tuple[np.ndarray, ...],
) -> np.ndarray:
    _validate_elementwise_loss_runtime_arrays(
        contract, arrays, "binary_cross_entropy_with_logits"
    )
    if type(operand) is not int or operand not in (0, 1):
        raise ShapeError(
            "binary_cross_entropy_with_logits VJP operand must be 0 or 1"
        )
    upstream = _elementwise_loss_upstream(
        contract, dy, "binary_cross_entropy_with_logits"
    )
    target_dtype = contract.input_dtypes[operand]
    if contract.reduced_elements == 0:
        return np.zeros(contract.input_shape, dtype=np.dtype(target_dtype))

    compute_dtype = _elementwise_loss_compute_dtype(contract)
    dtype = np.dtype(compute_dtype)
    logits = arrays[0].astype(dtype, copy=False)
    targets = arrays[1].astype(dtype, copy=False)
    gradient = np.empty(contract.input_shape, dtype=dtype)
    if operand == 0:
        magnitude = np.empty(contract.input_shape, dtype=dtype)
        np.abs(logits, out=magnitude)
        np.negative(magnitude, out=magnitude)
        with np.errstate(over="ignore", invalid="ignore"):
            np.exp(magnitude, out=magnitude)
        denominator = np.empty(contract.input_shape, dtype=dtype)
        np.add(1.0, magnitude, out=denominator)
        np.divide(magnitude, denominator, out=gradient)
        nonnegative = logits >= 0.0
        np.divide(1.0, denominator, out=gradient, where=nonnegative)
        np.subtract(gradient, targets, out=gradient)
    else:
        np.negative(logits, out=gradient)
    np.multiply(gradient, upstream, out=gradient)
    if gradient.dtype.name == target_dtype:
        return gradient
    return gradient.astype(np.dtype(target_dtype), copy=True)


def execute_kl_div_arrays(
    contract: KlDivContract,
    arrays: tuple[np.ndarray, ...],
) -> np.ndarray:
    _validate_elementwise_loss_runtime_arrays(contract, arrays, "kl_div")
    compute_dtype = _elementwise_loss_compute_dtype(contract)
    dtype = np.dtype(compute_dtype)
    input_array = arrays[0].astype(dtype, copy=False)
    target_array = arrays[1].astype(dtype, copy=False)
    per_element = np.empty(contract.input_shape, dtype=dtype)
    auxiliary = np.empty(contract.input_shape, dtype=dtype)
    if contract.log_target:
        with np.errstate(over="ignore", invalid="ignore"):
            np.exp(target_array, out=auxiliary)
        np.subtract(target_array, input_array, out=per_element)
        np.multiply(auxiliary, per_element, out=per_element)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            np.log(target_array, out=auxiliary)
            np.multiply(target_array, auxiliary, out=per_element)
        np.copyto(per_element, 0.0, where=target_array == 0.0)
        np.multiply(target_array, input_array, out=auxiliary)
        np.subtract(per_element, auxiliary, out=per_element)
    per_element[per_element == 0.0] = 0.0
    return _execute_elementwise_loss_reduction(contract, per_element)


def execute_kl_div_vjp_array(
    contract: KlDivContract,
    operand: int,
    dy: np.ndarray,
    arrays: tuple[np.ndarray, ...],
) -> np.ndarray:
    _validate_elementwise_loss_runtime_arrays(contract, arrays, "kl_div")
    if type(operand) is not int or operand not in (0, 1):
        raise ShapeError("kl_div VJP operand must be 0 or 1")
    upstream = _elementwise_loss_upstream(contract, dy, "kl_div")
    target_dtype = contract.input_dtypes[operand]
    if contract.reduced_elements == 0:
        return np.zeros(contract.input_shape, dtype=np.dtype(target_dtype))

    compute_dtype = _elementwise_loss_compute_dtype(contract)
    dtype = np.dtype(compute_dtype)
    input_array = arrays[0].astype(dtype, copy=False)
    target_array = arrays[1].astype(dtype, copy=False)
    gradient = np.empty(contract.input_shape, dtype=dtype)
    if operand == 0:
        if contract.log_target:
            with np.errstate(over="ignore", invalid="ignore"):
                np.exp(target_array, out=gradient)
            np.negative(gradient, out=gradient)
        else:
            np.negative(target_array, out=gradient)
    elif contract.log_target:
        auxiliary = np.empty(contract.input_shape, dtype=dtype)
        with np.errstate(over="ignore", invalid="ignore"):
            np.exp(target_array, out=auxiliary)
        np.subtract(target_array, input_array, out=gradient)
        np.add(gradient, 1.0, out=gradient)
        np.multiply(auxiliary, gradient, out=gradient)
    else:
        auxiliary = np.empty(contract.input_shape, dtype=dtype)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.log(target_array, out=gradient)
            np.divide(target_array, target_array, out=auxiliary)
        np.copyto(gradient, 0.0, where=target_array == 0.0)
        np.add(gradient, auxiliary, out=gradient)
        np.subtract(gradient, input_array, out=gradient)
    np.multiply(gradient, upstream, out=gradient)
    if gradient.dtype.name == target_dtype:
        return gradient
    return gradient.astype(np.dtype(target_dtype), copy=True)


def _validate_nll_loss_runtime_arrays(
    contract: NllLossContract,
    arrays: tuple[np.ndarray, ...],
) -> None:
    expected_arity = 3 if contract.has_weight else 2
    if type(arrays) is not tuple or len(arrays) != expected_arity:
        raise ShapeError(
            f"nll_loss execution requires exactly {expected_arity} arrays"
        )
    expected = (
        (contract.input_shape, contract.output_dtype),
        (contract.target_shape, "int64"),
    )
    if contract.has_weight:
        expected += ((contract.weight_shape, contract.output_dtype),)
    for index, (array, (shape, dtype)) in enumerate(zip(arrays, expected)):
        if type(array) is not np.ndarray:
            raise ShapeError(
                f"nll_loss runtime input {index} must be an exact ndarray"
            )
        if tuple(array.shape) != shape:
            raise ShapeError(
                f"nll_loss runtime input {index} shape {tuple(array.shape)} "
                f"does not match {shape}"
            )
        if array.dtype.name != dtype:
            raise ShapeError(
                f"nll_loss runtime input {index} dtype {array.dtype.name!r} "
                f"does not match {dtype!r}"
            )


def _nll_loss_selection(
    contract: NllLossContract,
    arrays: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = arrays[1]
    valid = targets != contract.ignore_index
    if bool(valid.any()):
        minimum = int(np.min(targets, where=valid, initial=0))
        maximum = int(np.max(targets, where=valid, initial=0))
        if minimum < 0 or maximum >= contract.class_count:
            raise ShapeError(
                "nll_loss: target values must be in "
                f"[0, {contract.class_count}) or equal ignore_index "
                f"{contract.ignore_index}"
            )
    safe_targets = np.where(valid, targets, 0)
    compute_dtype = (
        "float32" if contract.output_dtype == "float16" else contract.output_dtype
    )
    dtype = np.dtype(compute_dtype)
    if contract.has_weight:
        weight = arrays[2].astype(dtype, copy=False)
        singleton_count = len(contract.target_shape) - contract.batch_rank
        weight_view = weight.reshape(
            contract.input_shape[:contract.batch_rank]
            + (1,) * singleton_count
            + (contract.class_count,)
        )
        if contract.class_count == 0:
            selected_weight = np.zeros(contract.target_shape, dtype=dtype)
        else:
            selected_weight = np.take_along_axis(
                weight_view,
                safe_targets[..., None],
                axis=-1,
            )[..., 0]
        selected_weight = np.where(valid, selected_weight, 0.0)
    else:
        selected_weight = valid.astype(dtype, copy=False)
    return valid, safe_targets, selected_weight


def _nll_loss_reduction_axes(
    contract: NllLossContract,
) -> tuple[int, ...]:
    return tuple(range(contract.batch_rank, len(contract.target_shape)))


def _nll_loss_denominator(
    contract: NllLossContract,
    valid: np.ndarray,
    selected_weight: np.ndarray,
) -> np.ndarray:
    reduction_axes = _nll_loss_reduction_axes(contract)
    source = selected_weight if contract.has_weight else valid
    if reduction_axes:
        return source.sum(axis=reduction_axes)
    return np.asarray(source)


def execute_nll_loss_arrays(
    contract: NllLossContract,
    arrays: tuple[np.ndarray, ...],
) -> np.ndarray:
    _validate_nll_loss_runtime_arrays(contract, arrays)
    valid, safe_targets, selected_weight = _nll_loss_selection(contract, arrays)
    compute_dtype = (
        "float32" if contract.output_dtype == "float16" else contract.output_dtype
    )
    dtype = np.dtype(compute_dtype)
    input_array = arrays[0].astype(dtype, copy=False)
    if contract.class_count == 0:
        per_target = np.zeros(contract.target_shape, dtype=dtype)
    else:
        class_last = np.moveaxis(input_array, contract.class_axis, -1)
        per_target = np.take_along_axis(
            class_last,
            safe_targets[..., None],
            axis=-1,
        )[..., 0]
        np.negative(per_target, out=per_target)
        if contract.has_weight:
            np.multiply(per_target, selected_weight, out=per_target)
        np.copyto(per_target, 0.0, where=~valid)
    reduction_axes = _nll_loss_reduction_axes(contract)
    if contract.reduction == "none":
        result = per_target
    elif reduction_axes:
        numerator = per_target.sum(axis=reduction_axes, dtype=dtype)
        if contract.reduction == "sum":
            result = numerator
        else:
            denominator = _nll_loss_denominator(
                contract, valid, selected_weight
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.divide(numerator, denominator)
    elif contract.reduction == "sum":
        result = per_target
    else:
        denominator = _nll_loss_denominator(contract, valid, selected_weight)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.divide(per_target, denominator)
    return np.array(result, dtype=np.dtype(contract.output_dtype), copy=True)


def execute_nll_loss_vjp_array(
    contract: NllLossContract,
    dy: np.ndarray,
    arrays: tuple[np.ndarray, ...],
) -> np.ndarray:
    _validate_nll_loss_runtime_arrays(contract, arrays)
    if type(dy) is not np.ndarray:
        raise ShapeError("nll_loss VJP cotangent must be an exact ndarray")
    if tuple(dy.shape) != contract.output_shape or dy.dtype.name != contract.output_dtype:
        raise ShapeError(
            f"nll_loss VJP cotangent must have shape {contract.output_shape} "
            f"and dtype {contract.output_dtype!r}"
        )
    compute_dtype = (
        "float32" if contract.output_dtype == "float16" else contract.output_dtype
    )
    dtype = np.dtype(compute_dtype)
    gradient = np.zeros(contract.input_shape, dtype=dtype)
    valid, safe_targets, selected_weight = _nll_loss_selection(contract, arrays)
    if contract.class_count == 0 or not bool(valid.any()):
        return gradient.astype(np.dtype(contract.output_dtype), copy=False)

    upstream = dy.astype(dtype, copy=False)
    if contract.reduction != "none":
        target_user_rank = len(contract.target_shape) - contract.batch_rank
        upstream = upstream.reshape(
            contract.output_shape + (1,) * target_user_rank
        )
        upstream = np.broadcast_to(upstream, contract.target_shape)
        if contract.reduction == "mean":
            denominator = _nll_loss_denominator(
                contract, valid, selected_weight
            )
            denominator = np.asarray(denominator, dtype=dtype).reshape(
                contract.output_shape + (1,) * target_user_rank
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                upstream = np.divide(upstream, denominator)

    selected_gradient = np.empty(contract.target_shape, dtype=dtype)
    np.negative(upstream, out=selected_gradient)
    if contract.has_weight:
        np.multiply(selected_gradient, selected_weight, out=selected_gradient)
    np.copyto(selected_gradient, 0.0, where=~valid)
    class_last_gradient = np.moveaxis(gradient, contract.class_axis, -1)
    np.put_along_axis(
        class_last_gradient,
        safe_targets[..., None],
        selected_gradient[..., None],
        axis=-1,
    )
    if gradient.dtype.name == contract.output_dtype:
        return gradient
    return gradient.astype(np.dtype(contract.output_dtype), copy=True)


def _validate_cross_entropy_runtime_arrays(
    contract: CrossEntropyContract,
    arrays: tuple[np.ndarray, ...],
) -> None:
    expected_arity = 3 if contract.has_weight else 2
    if type(arrays) is not tuple or len(arrays) != expected_arity:
        raise ShapeError(
            f"cross_entropy execution requires exactly {expected_arity} arrays"
        )
    expected = (
        (contract.input_shape, contract.output_dtype),
        (contract.target_shape, contract.target_dtype),
    )
    if contract.has_weight:
        expected += ((contract.weight_shape, contract.output_dtype),)
    for index, (array, (shape, dtype)) in enumerate(zip(arrays, expected)):
        if type(array) is not np.ndarray:
            raise ShapeError(
                f"cross_entropy runtime input {index} must be an exact ndarray"
            )
        if tuple(array.shape) != shape:
            raise ShapeError(
                f"cross_entropy runtime input {index} shape "
                f"{tuple(array.shape)} does not match {shape}"
            )
        if array.dtype.name != dtype:
            raise ShapeError(
                f"cross_entropy runtime input {index} dtype "
                f"{array.dtype.name!r} does not match {dtype!r}"
            )


def _cross_entropy_compute_dtype(contract: CrossEntropyContract) -> np.dtype:
    return np.dtype(
        "float32" if contract.output_dtype == "float16" else contract.output_dtype
    )


def _cross_entropy_position_shape(
    contract: CrossEntropyContract,
) -> Tuple[int, ...]:
    return (
        contract.input_shape[:contract.class_axis]
        + contract.input_shape[contract.class_axis + 1:]
    )


def _cross_entropy_weight_view(
    contract: CrossEntropyContract,
    arrays: tuple[np.ndarray, ...],
    dtype: np.dtype,
) -> np.ndarray | None:
    if not contract.has_weight:
        return None
    shape = (
        contract.input_shape[:contract.batch_rank]
        + (1,) * (contract.class_axis - contract.batch_rank)
        + (contract.class_count,)
        + (1,) * (len(contract.input_shape) - contract.class_axis - 1)
    )
    return arrays[2].astype(dtype, copy=False).reshape(shape)


def _cross_entropy_log_probabilities(
    contract: CrossEntropyContract,
    arrays: tuple[np.ndarray, ...],
    dtype: np.dtype,
) -> np.ndarray:
    logits = arrays[0].astype(dtype, copy=False)
    maximum = np.max(logits, axis=contract.class_axis, keepdims=True)
    shifted = np.empty(contract.input_shape, dtype=dtype)
    np.subtract(logits, maximum, out=shifted)
    exponentials = np.empty(contract.input_shape, dtype=dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        np.exp(shifted, out=exponentials)
    normalizer = exponentials.sum(
        axis=contract.class_axis,
        keepdims=True,
        dtype=dtype,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        np.log(normalizer, out=normalizer)
    np.subtract(shifted, normalizer, out=shifted)
    return shifted


def _cross_entropy_index_selection(
    contract: CrossEntropyContract,
    arrays: tuple[np.ndarray, ...],
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    targets = arrays[1]
    valid = targets != contract.ignore_index
    if bool(valid.any()):
        minimum = int(np.min(targets, where=valid, initial=0))
        maximum = int(np.max(targets, where=valid, initial=0))
        if minimum < 0 or maximum >= contract.class_count:
            raise ShapeError(
                "cross_entropy: target values must be in "
                f"[0, {contract.class_count}) or equal ignore_index "
                f"{contract.ignore_index}"
            )
    safe_targets = np.where(valid, targets, 0)
    if contract.has_weight:
        weight = arrays[2].astype(dtype, copy=False)
        singleton_count = (
            len(_cross_entropy_position_shape(contract)) - contract.batch_rank
        )
        selection_view = weight.reshape(
            contract.input_shape[:contract.batch_rank]
            + (1,) * singleton_count
            + (contract.class_count,)
        )
        selected_weight = np.take_along_axis(
            selection_view,
            safe_targets[..., None],
            axis=-1,
        )[..., 0]
        selected_weight = np.where(valid, selected_weight, 0.0)
    else:
        selected_weight = valid.astype(dtype, copy=False)
    return valid, safe_targets, selected_weight


def _cross_entropy_reduction_axes(
    contract: CrossEntropyContract,
) -> tuple[int, ...]:
    return tuple(
        range(contract.batch_rank, len(_cross_entropy_position_shape(contract)))
    )


def _cross_entropy_index_denominator(
    contract: CrossEntropyContract,
    valid: np.ndarray,
    selected_weight: np.ndarray,
) -> np.ndarray:
    source = selected_weight if contract.has_weight else valid
    axes = _cross_entropy_reduction_axes(contract)
    if axes:
        return source.sum(axis=axes)
    return np.asarray(source)


def _cross_entropy_reduce(
    contract: CrossEntropyContract,
    per_position: np.ndarray,
    valid: np.ndarray | None,
    selected_weight: np.ndarray | None,
    dtype: np.dtype,
) -> np.ndarray:
    if contract.reduction == "none":
        return per_position
    axes = _cross_entropy_reduction_axes(contract)
    numerator = (
        per_position.sum(axis=axes, dtype=dtype)
        if axes
        else np.asarray(per_position)
    )
    if contract.reduction == "sum":
        return numerator
    if contract.target_mode == "indices":
        assert valid is not None and selected_weight is not None
        denominator = _cross_entropy_index_denominator(
            contract, valid, selected_weight
        )
    else:
        denominator = _elementwise_loss_checked_product(
            _cross_entropy_position_shape(contract)[contract.batch_rank:],
            L1_LOSS_WORK_ELEMENT_MAX,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(numerator, denominator)


def execute_cross_entropy_arrays(
    contract: CrossEntropyContract,
    arrays: tuple[np.ndarray, ...],
) -> np.ndarray:
    _validate_cross_entropy_runtime_arrays(contract, arrays)
    dtype = _cross_entropy_compute_dtype(contract)
    log_probabilities = _cross_entropy_log_probabilities(
        contract, arrays, dtype
    )
    weight_view = _cross_entropy_weight_view(contract, arrays, dtype)
    valid: np.ndarray | None = None
    selected_weight: np.ndarray | None = None

    if contract.target_mode == "probabilities":
        coefficient = arrays[1].astype(dtype, copy=False)
        if contract.label_smoothing != 0.0:
            coefficient = np.array(coefficient, dtype=dtype, copy=True)
            np.multiply(
                coefficient,
                1.0 - contract.label_smoothing,
                out=coefficient,
            )
            np.add(
                coefficient,
                contract.label_smoothing / contract.class_count,
                out=coefficient,
            )
        if weight_view is not None:
            coefficient = np.multiply(coefficient, weight_view)
        per_class = np.multiply(log_probabilities, coefficient)
        per_position = per_class.sum(
            axis=contract.class_axis,
            dtype=dtype,
        )
        np.negative(per_position, out=per_position)
    else:
        valid, safe_targets, selected_weight = (
            _cross_entropy_index_selection(contract, arrays, dtype)
        )
        class_last = np.moveaxis(
            log_probabilities, contract.class_axis, -1
        )
        per_position = np.take_along_axis(
            class_last,
            safe_targets[..., None],
            axis=-1,
        )[..., 0]
        np.negative(per_position, out=per_position)
        if contract.has_weight:
            np.multiply(per_position, selected_weight, out=per_position)
        if contract.label_smoothing != 0.0:
            smooth_source = log_probabilities
            if weight_view is not None:
                smooth_source = np.multiply(smooth_source, weight_view)
            smooth_loss = smooth_source.sum(
                axis=contract.class_axis,
                dtype=dtype,
            )
            np.negative(smooth_loss, out=smooth_loss)
            np.multiply(
                per_position,
                1.0 - contract.label_smoothing,
                out=per_position,
            )
            np.multiply(
                smooth_loss,
                contract.label_smoothing / contract.class_count,
                out=smooth_loss,
            )
            np.add(per_position, smooth_loss, out=per_position)
        np.copyto(per_position, 0.0, where=~valid)

    result = _cross_entropy_reduce(
        contract,
        per_position,
        valid,
        selected_weight,
        dtype,
    )
    return np.array(result, dtype=np.dtype(contract.output_dtype), copy=True)


def _cross_entropy_vjp_upstream(
    contract: CrossEntropyContract,
    dy: np.ndarray,
    valid: np.ndarray | None,
    selected_weight: np.ndarray | None,
    dtype: np.dtype,
) -> np.ndarray:
    if type(dy) is not np.ndarray:
        raise ShapeError("cross_entropy VJP cotangent must be an exact ndarray")
    if (
        tuple(dy.shape) != contract.output_shape
        or dy.dtype.name != contract.output_dtype
    ):
        raise ShapeError(
            "cross_entropy VJP cotangent must have shape "
            f"{contract.output_shape} and dtype {contract.output_dtype!r}"
        )
    upstream = dy.astype(dtype, copy=False)
    position_shape = _cross_entropy_position_shape(contract)
    if contract.reduction != "none":
        user_position_rank = len(position_shape) - contract.batch_rank
        upstream = upstream.reshape(
            contract.output_shape + (1,) * user_position_rank
        )
        upstream = np.broadcast_to(upstream, position_shape)
        if contract.reduction == "mean":
            if contract.target_mode == "indices":
                assert valid is not None and selected_weight is not None
                denominator = _cross_entropy_index_denominator(
                    contract, valid, selected_weight
                )
                denominator = np.asarray(denominator, dtype=dtype).reshape(
                    contract.output_shape + (1,) * user_position_rank
                )
            else:
                denominator = _elementwise_loss_checked_product(
                    position_shape[contract.batch_rank:],
                    L1_LOSS_WORK_ELEMENT_MAX,
                )
            with np.errstate(divide="ignore", invalid="ignore"):
                upstream = np.divide(upstream, denominator)
    return np.expand_dims(upstream, axis=contract.class_axis)


def execute_cross_entropy_vjp_array(
    contract: CrossEntropyContract,
    operand: int,
    dy: np.ndarray,
    arrays: tuple[np.ndarray, ...],
) -> np.ndarray:
    _validate_cross_entropy_runtime_arrays(contract, arrays)
    allowed = (0, 1) if contract.target_mode == "probabilities" else (0,)
    if type(operand) is not int or operand not in allowed:
        raise ShapeError(
            f"cross_entropy VJP operand must be one of {allowed} for "
            f"{contract.target_mode} targets"
        )
    dtype = _cross_entropy_compute_dtype(contract)
    log_probabilities = _cross_entropy_log_probabilities(
        contract, arrays, dtype
    )
    weight_view = _cross_entropy_weight_view(contract, arrays, dtype)
    valid: np.ndarray | None = None
    selected_weight: np.ndarray | None = None

    if contract.target_mode == "indices":
        valid, safe_targets, selected_weight = (
            _cross_entropy_index_selection(contract, arrays, dtype)
        )
    upstream = _cross_entropy_vjp_upstream(
        contract,
        dy,
        valid,
        selected_weight,
        dtype,
    )

    if operand == 1:
        if contract.label_smoothing == 1.0:
            gradient = np.zeros(contract.target_shape, dtype=dtype)
        else:
            gradient = np.empty(contract.target_shape, dtype=dtype)
            np.negative(log_probabilities, out=gradient)
            if weight_view is not None:
                np.multiply(gradient, weight_view, out=gradient)
            if contract.label_smoothing != 0.0:
                np.multiply(
                    gradient,
                    1.0 - contract.label_smoothing,
                    out=gradient,
                )
            np.multiply(gradient, upstream, out=gradient)
        if gradient.dtype.name == contract.target_dtype:
            return gradient
        return gradient.astype(np.dtype(contract.target_dtype), copy=True)

    if contract.target_mode == "probabilities":
        coefficient = arrays[1].astype(dtype, copy=False)
        if contract.label_smoothing != 0.0:
            coefficient = np.array(coefficient, dtype=dtype, copy=True)
            np.multiply(
                coefficient,
                1.0 - contract.label_smoothing,
                out=coefficient,
            )
            np.add(
                coefficient,
                contract.label_smoothing / contract.class_count,
                out=coefficient,
            )
        if weight_view is not None:
            coefficient = np.multiply(coefficient, weight_view)
    else:
        coefficient = np.zeros(contract.input_shape, dtype=dtype)
        if contract.label_smoothing != 0.0:
            if weight_view is None:
                coefficient.fill(
                    contract.label_smoothing / contract.class_count
                )
            else:
                coefficient = np.array(
                    np.broadcast_to(weight_view, contract.input_shape),
                    dtype=dtype,
                    copy=True,
                )
                np.multiply(
                    coefficient,
                    contract.label_smoothing / contract.class_count,
                    out=coefficient,
                )
        class_last = np.moveaxis(coefficient, contract.class_axis, -1)
        selected = (1.0 - contract.label_smoothing) * selected_weight
        previous = np.take_along_axis(
            class_last,
            safe_targets[..., None],
            axis=-1,
        )[..., 0]
        np.add(previous, selected, out=previous)
        np.put_along_axis(
            class_last,
            safe_targets[..., None],
            previous[..., None],
            axis=-1,
        )
        invalid = np.broadcast_to(
            np.expand_dims(~valid, axis=contract.class_axis),
            contract.input_shape,
        )
        np.copyto(coefficient, 0.0, where=invalid)

    probabilities = np.empty(contract.input_shape, dtype=dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        np.exp(log_probabilities, out=probabilities)
    normalizer = coefficient.sum(
        axis=contract.class_axis,
        keepdims=True,
        dtype=dtype,
    )
    np.multiply(probabilities, normalizer, out=probabilities)
    np.subtract(probabilities, coefficient, out=probabilities)
    np.multiply(probabilities, upstream, out=probabilities)
    if contract.target_mode == "indices":
        invalid = np.broadcast_to(
            np.expand_dims(~valid, axis=contract.class_axis),
            contract.input_shape,
        )
        np.copyto(probabilities, 0.0, where=invalid)
    if probabilities.dtype.name == contract.output_dtype:
        return probabilities
    return probabilities.astype(np.dtype(contract.output_dtype), copy=True)


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


def _validate_l1_loss(
    node: Any,
    contract: FrameworkOperationContract,
) -> L1LossContract:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    arg = getattr(node, "arg", None)
    if type(arg) is not dict or set(arg) != {"reduction", "batch_rank"}:
        raise ShapeError("L1_LOSS arg fields must be exactly 'reduction' and 'batch_rank'")
    normalized = infer_l1_loss_contract(
        getattr(node, "inputs", None),
        arg["reduction"],
        arg["batch_rank"],
    )
    if getattr(node, "shape", None) != normalized.output_shape:
        raise ShapeError(
            f"L1_LOSS declared shape {getattr(node, 'shape', None)!r} does not "
            f"match derived shape {normalized.output_shape!r}"
        )
    if getattr(node, "dtype", None) != normalized.output_dtype:
        raise ShapeError(
            f"L1_LOSS declared dtype {getattr(node, 'dtype', None)!r} does not "
            f"match promoted dtype {normalized.output_dtype!r}"
        )
    return normalized


def _validate_l1_loss_vjp(node: Any) -> Tuple[L1LossContract, int]:
    if getattr(node, "op", None) != "L1_LOSS_VJP":
        raise ShapeError(
            f"L1_LOSS_VJP validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = getattr(node, "inputs", None)
    if type(inputs) is not tuple or len(inputs) != 3:
        raise ShapeError("L1_LOSS_VJP must have exactly dy, input, and target inputs")
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("L1_LOSS_VJP arg must be a plain dict")
    fields = set(arg)
    if not {"reduction", "batch_rank", "operand"}.issubset(fields) or not fields.issubset(
        {"reduction", "batch_rank", "operand", "vjp_of"}
    ):
        raise ShapeError(
            "L1_LOSS_VJP arg fields must be exactly 'reduction', 'batch_rank', "
            "and 'operand' plus optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("L1_LOSS_VJP arg.vjp_of must reference a UOp")
    operand = arg["operand"]
    if type(operand) is not int or operand not in (0, 1):
        raise ShapeError("L1_LOSS_VJP arg.operand must be 0 or 1")
    normalized = infer_l1_loss_contract(
        inputs[1:],
        arg["reduction"],
        arg["batch_rank"],
    )
    dy = inputs[0]
    if (
        getattr(dy, "shape", None) != normalized.output_shape
        or getattr(dy, "dtype", None) != normalized.output_dtype
    ):
        raise ShapeError(
            f"L1_LOSS_VJP dy must have shape {normalized.output_shape!r} and "
            f"dtype {normalized.output_dtype!r}"
        )
    source = inputs[operand + 1]
    if getattr(node, "shape", None) != getattr(source, "shape", None):
        raise ShapeError("L1_LOSS_VJP must preserve its selected operand shape")
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError("L1_LOSS_VJP must preserve its selected operand dtype")
    return normalized, operand


def _validate_smooth_l1_loss(
    node: Any,
    contract: FrameworkOperationContract,
) -> SmoothL1LossContract:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    arg = getattr(node, "arg", None)
    if type(arg) is not dict or set(arg) != {"reduction", "batch_rank", "beta"}:
        raise ShapeError(
            "SMOOTH_L1_LOSS arg fields must be exactly 'reduction', 'batch_rank', and 'beta'"
        )
    if type(arg["beta"]) is not float:
        raise ShapeError("SMOOTH_L1_LOSS arg.beta must be a canonical float")
    normalized = infer_smooth_l1_loss_contract(
        getattr(node, "inputs", None),
        arg["beta"],
        arg["reduction"],
        arg["batch_rank"],
    )
    if normalized.beta != arg["beta"] or (
        arg["beta"] == 0.0 and math.copysign(1.0, arg["beta"]) < 0.0
    ):
        raise ShapeError("SMOOTH_L1_LOSS arg.beta must be canonical")
    if getattr(node, "shape", None) != normalized.output_shape:
        raise ShapeError(
            f"SMOOTH_L1_LOSS declared shape {getattr(node, 'shape', None)!r} does not "
            f"match derived shape {normalized.output_shape!r}"
        )
    if getattr(node, "dtype", None) != normalized.output_dtype:
        raise ShapeError(
            f"SMOOTH_L1_LOSS declared dtype {getattr(node, 'dtype', None)!r} does not "
            f"match promoted dtype {normalized.output_dtype!r}"
        )
    return normalized


def _validate_smooth_l1_loss_vjp(node: Any) -> Tuple[SmoothL1LossContract, int]:
    if getattr(node, "op", None) != "SMOOTH_L1_LOSS_VJP":
        raise ShapeError(
            "SMOOTH_L1_LOSS_VJP validator received opcode "
            f"{getattr(node, 'op', None)!r}"
        )
    inputs = getattr(node, "inputs", None)
    if type(inputs) is not tuple or len(inputs) != 3:
        raise ShapeError(
            "SMOOTH_L1_LOSS_VJP must have exactly dy, input, and target inputs"
        )
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("SMOOTH_L1_LOSS_VJP arg must be a plain dict")
    fields = set(arg)
    required = {"reduction", "batch_rank", "beta", "operand"}
    if not required.issubset(fields) or not fields.issubset(required | {"vjp_of"}):
        raise ShapeError(
            "SMOOTH_L1_LOSS_VJP arg fields must be exactly 'reduction', "
            "'batch_rank', 'beta', and 'operand' plus optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("SMOOTH_L1_LOSS_VJP arg.vjp_of must reference a UOp")
    if type(arg["beta"]) is not float:
        raise ShapeError("SMOOTH_L1_LOSS_VJP arg.beta must be a canonical float")
    operand = arg["operand"]
    if type(operand) is not int or operand not in (0, 1):
        raise ShapeError("SMOOTH_L1_LOSS_VJP arg.operand must be 0 or 1")
    normalized = infer_smooth_l1_loss_contract(
        inputs[1:],
        arg["beta"],
        arg["reduction"],
        arg["batch_rank"],
    )
    if normalized.beta != arg["beta"] or (
        arg["beta"] == 0.0 and math.copysign(1.0, arg["beta"]) < 0.0
    ):
        raise ShapeError("SMOOTH_L1_LOSS_VJP arg.beta must be canonical")
    dy = inputs[0]
    if (
        getattr(dy, "shape", None) != normalized.output_shape
        or getattr(dy, "dtype", None) != normalized.output_dtype
    ):
        raise ShapeError(
            f"SMOOTH_L1_LOSS_VJP dy must have shape {normalized.output_shape!r} "
            f"and dtype {normalized.output_dtype!r}"
        )
    source = inputs[operand + 1]
    if getattr(node, "shape", None) != getattr(source, "shape", None):
        raise ShapeError(
            "SMOOTH_L1_LOSS_VJP must preserve its selected operand shape"
        )
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError(
            "SMOOTH_L1_LOSS_VJP must preserve its selected operand dtype"
        )
    return normalized, operand


def _validate_binary_cross_entropy(
    node: Any,
    contract: FrameworkOperationContract,
) -> BinaryCrossEntropyContract:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    arg = getattr(node, "arg", None)
    if type(arg) is not dict or set(arg) != {"reduction", "batch_rank"}:
        raise ShapeError(
            "BINARY_CROSS_ENTROPY arg fields must be exactly 'reduction' and 'batch_rank'"
        )
    normalized = infer_binary_cross_entropy_contract(
        getattr(node, "inputs", None),
        arg["reduction"],
        arg["batch_rank"],
    )
    if getattr(node, "shape", None) != normalized.output_shape:
        raise ShapeError(
            "BINARY_CROSS_ENTROPY declared shape "
            f"{getattr(node, 'shape', None)!r} does not match derived shape "
            f"{normalized.output_shape!r}"
        )
    if getattr(node, "dtype", None) != normalized.output_dtype:
        raise ShapeError(
            "BINARY_CROSS_ENTROPY declared dtype "
            f"{getattr(node, 'dtype', None)!r} does not match promoted dtype "
            f"{normalized.output_dtype!r}"
        )
    return normalized


def _validate_binary_cross_entropy_vjp(
    node: Any,
) -> Tuple[BinaryCrossEntropyContract, int]:
    if getattr(node, "op", None) != "BINARY_CROSS_ENTROPY_VJP":
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_VJP validator received opcode "
            f"{getattr(node, 'op', None)!r}"
        )
    inputs = getattr(node, "inputs", None)
    if type(inputs) is not tuple or len(inputs) != 3:
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_VJP must have exactly dy, input, and target inputs"
        )
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("BINARY_CROSS_ENTROPY_VJP arg must be a plain dict")
    fields = set(arg)
    required = {"reduction", "batch_rank", "operand"}
    if not required.issubset(fields) or not fields.issubset(required | {"vjp_of"}):
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_VJP arg fields must be exactly 'reduction', "
            "'batch_rank', and 'operand' plus optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_VJP arg.vjp_of must reference a UOp"
        )
    operand = arg["operand"]
    if type(operand) is not int or operand not in (0, 1):
        raise ShapeError("BINARY_CROSS_ENTROPY_VJP arg.operand must be 0 or 1")
    normalized = infer_binary_cross_entropy_contract(
        inputs[1:],
        arg["reduction"],
        arg["batch_rank"],
    )
    dy = inputs[0]
    if (
        getattr(dy, "shape", None) != normalized.output_shape
        or getattr(dy, "dtype", None) != normalized.output_dtype
    ):
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_VJP dy must have shape "
            f"{normalized.output_shape!r} and dtype {normalized.output_dtype!r}"
        )
    source = inputs[operand + 1]
    if getattr(node, "shape", None) != getattr(source, "shape", None):
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_VJP must preserve its selected operand shape"
        )
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_VJP must preserve its selected operand dtype"
        )
    return normalized, operand


def _validate_binary_cross_entropy_with_logits(
    node: Any,
    contract: FrameworkOperationContract,
) -> BinaryCrossEntropyWithLogitsContract:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    arg = getattr(node, "arg", None)
    if type(arg) is not dict or set(arg) != {"reduction", "batch_rank"}:
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_WITH_LOGITS arg fields must be exactly "
            "'reduction' and 'batch_rank'"
        )
    normalized = infer_binary_cross_entropy_with_logits_contract(
        getattr(node, "inputs", None),
        arg["reduction"],
        arg["batch_rank"],
    )
    if getattr(node, "shape", None) != normalized.output_shape:
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_WITH_LOGITS declared shape "
            f"{getattr(node, 'shape', None)!r} does not match derived shape "
            f"{normalized.output_shape!r}"
        )
    if getattr(node, "dtype", None) != normalized.output_dtype:
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_WITH_LOGITS declared dtype "
            f"{getattr(node, 'dtype', None)!r} does not match promoted dtype "
            f"{normalized.output_dtype!r}"
        )
    return normalized


def _validate_binary_cross_entropy_with_logits_vjp(
    node: Any,
) -> Tuple[BinaryCrossEntropyWithLogitsContract, int]:
    if getattr(node, "op", None) != "BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP":
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP validator received opcode "
            f"{getattr(node, 'op', None)!r}"
        )
    inputs = getattr(node, "inputs", None)
    if type(inputs) is not tuple or len(inputs) != 3:
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP must have exactly dy, logits, "
            "and target inputs"
        )
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP arg must be a plain dict"
        )
    fields = set(arg)
    required = {"reduction", "batch_rank", "operand"}
    if not required.issubset(fields) or not fields.issubset(required | {"vjp_of"}):
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP arg fields must be exactly "
            "'reduction', 'batch_rank', and 'operand' plus optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP arg.vjp_of must reference a UOp"
        )
    operand = arg["operand"]
    if type(operand) is not int or operand not in (0, 1):
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP arg.operand must be 0 or 1"
        )
    normalized = infer_binary_cross_entropy_with_logits_contract(
        inputs[1:],
        arg["reduction"],
        arg["batch_rank"],
    )
    dy = inputs[0]
    if (
        getattr(dy, "shape", None) != normalized.output_shape
        or getattr(dy, "dtype", None) != normalized.output_dtype
    ):
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP dy must have shape "
            f"{normalized.output_shape!r} and dtype {normalized.output_dtype!r}"
        )
    source = inputs[operand + 1]
    if getattr(node, "shape", None) != getattr(source, "shape", None):
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP must preserve its selected "
            "operand shape"
        )
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP must preserve its selected "
            "operand dtype"
        )
    return normalized, operand


def _validate_kl_div(
    node: Any,
    contract: FrameworkOperationContract,
) -> KlDivContract:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode {getattr(node, 'op', None)!r}"
        )
    arg = getattr(node, "arg", None)
    if type(arg) is not dict or set(arg) != {
        "reduction", "batch_rank", "log_target"
    }:
        raise ShapeError(
            "KL_DIV arg fields must be exactly 'reduction', 'batch_rank', and "
            "'log_target'"
        )
    normalized = infer_kl_div_contract(
        getattr(node, "inputs", None),
        arg["reduction"],
        arg["log_target"],
        arg["batch_rank"],
    )
    if getattr(node, "shape", None) != normalized.output_shape:
        raise ShapeError(
            f"KL_DIV declared shape {getattr(node, 'shape', None)!r} does not "
            f"match derived shape {normalized.output_shape!r}"
        )
    if getattr(node, "dtype", None) != normalized.output_dtype:
        raise ShapeError(
            f"KL_DIV declared dtype {getattr(node, 'dtype', None)!r} does not "
            f"match promoted dtype {normalized.output_dtype!r}"
        )
    return normalized


def _validate_kl_div_vjp(node: Any) -> Tuple[KlDivContract, int]:
    if getattr(node, "op", None) != "KL_DIV_VJP":
        raise ShapeError(
            f"KL_DIV_VJP validator received opcode {getattr(node, 'op', None)!r}"
        )
    inputs = getattr(node, "inputs", None)
    if type(inputs) is not tuple or len(inputs) != 3:
        raise ShapeError("KL_DIV_VJP must have exactly dy, input, and target inputs")
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("KL_DIV_VJP arg must be a plain dict")
    fields = set(arg)
    required = {"reduction", "batch_rank", "log_target", "operand"}
    if not required.issubset(fields) or not fields.issubset(required | {"vjp_of"}):
        raise ShapeError(
            "KL_DIV_VJP arg fields must be exactly 'reduction', 'batch_rank', "
            "'log_target', and 'operand' plus optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("KL_DIV_VJP arg.vjp_of must reference a UOp")
    operand = arg["operand"]
    if type(operand) is not int or operand not in (0, 1):
        raise ShapeError("KL_DIV_VJP arg.operand must be 0 or 1")
    normalized = infer_kl_div_contract(
        inputs[1:],
        arg["reduction"],
        arg["log_target"],
        arg["batch_rank"],
    )
    dy = inputs[0]
    if (
        getattr(dy, "shape", None) != normalized.output_shape
        or getattr(dy, "dtype", None) != normalized.output_dtype
    ):
        raise ShapeError(
            f"KL_DIV_VJP dy must have shape {normalized.output_shape!r} and "
            f"dtype {normalized.output_dtype!r}"
        )
    source = inputs[operand + 1]
    if getattr(node, "shape", None) != getattr(source, "shape", None):
        raise ShapeError("KL_DIV_VJP must preserve its selected operand shape")
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError("KL_DIV_VJP must preserve its selected operand dtype")
    return normalized, operand


def _validate_nll_loss(
    node: Any,
    contract: FrameworkOperationContract,
) -> NllLossContract:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode "
            f"{getattr(node, 'op', None)!r}"
        )
    arg = getattr(node, "arg", None)
    required = {"reduction", "batch_rank", "ignore_index", "has_weight"}
    if type(arg) is not dict or set(arg) != required:
        raise ShapeError(
            "NLL_LOSS arg fields must be exactly 'reduction', 'batch_rank', "
            "'ignore_index', and 'has_weight'"
        )
    normalized = infer_nll_loss_contract(
        getattr(node, "inputs", None),
        arg["reduction"],
        arg["ignore_index"],
        arg["has_weight"],
        arg["batch_rank"],
    )
    if getattr(node, "shape", None) != normalized.output_shape:
        raise ShapeError(
            f"NLL_LOSS declared shape {getattr(node, 'shape', None)!r} does "
            f"not match derived shape {normalized.output_shape!r}"
        )
    if getattr(node, "dtype", None) != normalized.output_dtype:
        raise ShapeError(
            f"NLL_LOSS declared dtype {getattr(node, 'dtype', None)!r} does "
            f"not match input dtype {normalized.output_dtype!r}"
        )
    return normalized


def _validate_nll_loss_vjp(node: Any) -> NllLossContract:
    if getattr(node, "op", None) != "NLL_LOSS_VJP":
        raise ShapeError(
            "NLL_LOSS_VJP validator received opcode "
            f"{getattr(node, 'op', None)!r}"
        )
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("NLL_LOSS_VJP arg must be a plain dict")
    required = {"reduction", "batch_rank", "ignore_index", "has_weight"}
    fields = set(arg)
    if not required.issubset(fields) or not fields.issubset(required | {"vjp_of"}):
        raise ShapeError(
            "NLL_LOSS_VJP arg fields must be exactly 'reduction', "
            "'batch_rank', 'ignore_index', and 'has_weight' plus optional "
            "'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("NLL_LOSS_VJP arg.vjp_of must reference a UOp")
    inputs = getattr(node, "inputs", None)
    expected_arity = 4 if arg["has_weight"] is True else 3
    if type(inputs) is not tuple or len(inputs) != expected_arity:
        raise ShapeError(
            f"NLL_LOSS_VJP must have exactly {expected_arity} inputs for "
            f"has_weight={arg['has_weight']!r}"
        )
    normalized = infer_nll_loss_contract(
        inputs[1:],
        arg["reduction"],
        arg["ignore_index"],
        arg["has_weight"],
        arg["batch_rank"],
    )
    dy = inputs[0]
    if (
        getattr(dy, "shape", None) != normalized.output_shape
        or getattr(dy, "dtype", None) != normalized.output_dtype
    ):
        raise ShapeError(
            f"NLL_LOSS_VJP dy must have shape {normalized.output_shape!r} "
            f"and dtype {normalized.output_dtype!r}"
        )
    source = inputs[1]
    if getattr(node, "shape", None) != getattr(source, "shape", None):
        raise ShapeError("NLL_LOSS_VJP must preserve the input shape")
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError("NLL_LOSS_VJP must preserve the input dtype")
    return normalized


def _validate_cross_entropy(
    node: Any,
    contract: FrameworkOperationContract,
) -> CrossEntropyContract:
    if getattr(node, "op", None) != contract.opcode:
        raise ShapeError(
            f"{contract.contract_id} validator received opcode "
            f"{getattr(node, 'op', None)!r}"
        )
    arg = getattr(node, "arg", None)
    required = {
        "reduction",
        "batch_rank",
        "ignore_index",
        "has_weight",
        "label_smoothing",
        "target_mode",
    }
    if type(arg) is not dict or set(arg) != required:
        raise ShapeError(
            "CROSS_ENTROPY arg fields must be exactly 'reduction', "
            "'batch_rank', 'ignore_index', 'has_weight', "
            "'label_smoothing', and 'target_mode'"
        )
    normalized = infer_cross_entropy_contract(
        getattr(node, "inputs", None),
        arg["reduction"],
        arg["ignore_index"],
        arg["has_weight"],
        arg["label_smoothing"],
        arg["target_mode"],
        arg["batch_rank"],
    )
    if getattr(node, "shape", None) != normalized.output_shape:
        raise ShapeError(
            f"CROSS_ENTROPY declared shape {getattr(node, 'shape', None)!r} "
            f"does not match derived shape {normalized.output_shape!r}"
        )
    if getattr(node, "dtype", None) != normalized.output_dtype:
        raise ShapeError(
            f"CROSS_ENTROPY declared dtype {getattr(node, 'dtype', None)!r} "
            f"does not match input dtype {normalized.output_dtype!r}"
        )
    return normalized


def _validate_cross_entropy_vjp(
    node: Any,
) -> tuple[CrossEntropyContract, int]:
    if getattr(node, "op", None) != "CROSS_ENTROPY_VJP":
        raise ShapeError(
            "CROSS_ENTROPY_VJP validator received opcode "
            f"{getattr(node, 'op', None)!r}"
        )
    arg = getattr(node, "arg", None)
    if type(arg) is not dict:
        raise ShapeError("CROSS_ENTROPY_VJP arg must be a plain dict")
    required = {
        "reduction",
        "batch_rank",
        "ignore_index",
        "has_weight",
        "label_smoothing",
        "target_mode",
        "operand",
    }
    fields = set(arg)
    if not required.issubset(fields) or not fields.issubset(
        required | {"vjp_of"}
    ):
        raise ShapeError(
            "CROSS_ENTROPY_VJP arg fields must be the forward contract "
            "fields plus 'operand' and optional 'vjp_of'"
        )
    if "vjp_of" in arg and type(arg["vjp_of"]) is not type(node):
        raise ShapeError("CROSS_ENTROPY_VJP arg.vjp_of must reference a UOp")
    inputs = getattr(node, "inputs", None)
    expected_arity = 4 if arg["has_weight"] is True else 3
    if type(inputs) is not tuple or len(inputs) != expected_arity:
        raise ShapeError(
            f"CROSS_ENTROPY_VJP must have exactly {expected_arity} inputs "
            f"for has_weight={arg['has_weight']!r}"
        )
    normalized = infer_cross_entropy_contract(
        inputs[1:],
        arg["reduction"],
        arg["ignore_index"],
        arg["has_weight"],
        arg["label_smoothing"],
        arg["target_mode"],
        arg["batch_rank"],
    )
    dy = inputs[0]
    if (
        getattr(dy, "shape", None) != normalized.output_shape
        or getattr(dy, "dtype", None) != normalized.output_dtype
    ):
        raise ShapeError(
            "CROSS_ENTROPY_VJP dy must have shape "
            f"{normalized.output_shape!r} and dtype "
            f"{normalized.output_dtype!r}"
        )
    operand = arg["operand"]
    allowed = (0, 1) if normalized.target_mode == "probabilities" else (0,)
    if type(operand) is not int or operand not in allowed:
        raise ShapeError(
            f"CROSS_ENTROPY_VJP operand must be one of {allowed} for "
            f"{normalized.target_mode} targets"
        )
    source = inputs[operand + 1]
    if getattr(node, "shape", None) != getattr(source, "shape", None):
        raise ShapeError(
            "CROSS_ENTROPY_VJP must preserve its selected operand shape"
        )
    if getattr(node, "dtype", None) != getattr(source, "dtype", None):
        raise ShapeError(
            "CROSS_ENTROPY_VJP must preserve its selected operand dtype"
        )
    return normalized, operand


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
    "browsergrad.jit.framework.functional.l1-loss.v1": _validate_l1_loss,
    "browsergrad.jit.framework.functional.smooth-l1-loss.v1": _validate_smooth_l1_loss,
    "browsergrad.jit.framework.functional.binary-cross-entropy.v1": _validate_binary_cross_entropy,
    "browsergrad.jit.framework.functional.binary-cross-entropy-with-logits.v1": _validate_binary_cross_entropy_with_logits,
    "browsergrad.jit.framework.functional.kl-div.v1": _validate_kl_div,
    "browsergrad.jit.framework.functional.nll-loss.v1": _validate_nll_loss,
    "browsergrad.jit.framework.functional.cross-entropy.v1": _validate_cross_entropy,
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
    "L1_LOSS_VJP": _validate_l1_loss_vjp,
    "SMOOTH_L1_LOSS_VJP": _validate_smooth_l1_loss_vjp,
    "BINARY_CROSS_ENTROPY_VJP": _validate_binary_cross_entropy_vjp,
    "BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP": _validate_binary_cross_entropy_with_logits_vjp,
    "KL_DIV_VJP": _validate_kl_div_vjp,
    "NLL_LOSS_VJP": _validate_nll_loss_vjp,
    "CROSS_ENTROPY_VJP": _validate_cross_entropy_vjp,
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


def validate_l1_loss_contract(node: Any) -> L1LossContract:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.functional.l1-loss.v1":
        raise ShapeError("L1_LOSS resolved to the wrong framework-operation contract")
    return normalized


def validate_l1_loss_vjp_contract(node: Any) -> Tuple[L1LossContract, int]:
    return validate_internal_operation_contract(node)


def validate_smooth_l1_loss_contract(node: Any) -> SmoothL1LossContract:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.functional.smooth-l1-loss.v1":
        raise ShapeError(
            "SMOOTH_L1_LOSS resolved to the wrong framework-operation contract"
        )
    return normalized


def validate_smooth_l1_loss_vjp_contract(
    node: Any,
) -> Tuple[SmoothL1LossContract, int]:
    return validate_internal_operation_contract(node)


def validate_binary_cross_entropy_contract(
    node: Any,
) -> BinaryCrossEntropyContract:
    record, normalized = validate_framework_operation_contract(node)
    if (
        record.contract_id
        != "browsergrad.jit.framework.functional.binary-cross-entropy.v1"
    ):
        raise ShapeError(
            "BINARY_CROSS_ENTROPY resolved to the wrong framework-operation contract"
        )
    return normalized


def validate_binary_cross_entropy_vjp_contract(
    node: Any,
) -> Tuple[BinaryCrossEntropyContract, int]:
    return validate_internal_operation_contract(node)


def validate_binary_cross_entropy_with_logits_contract(
    node: Any,
) -> BinaryCrossEntropyWithLogitsContract:
    record, normalized = validate_framework_operation_contract(node)
    if (
        record.contract_id
        != "browsergrad.jit.framework.functional.binary-cross-entropy-with-logits.v1"
    ):
        raise ShapeError(
            "BINARY_CROSS_ENTROPY_WITH_LOGITS resolved to the wrong "
            "framework-operation contract"
        )
    return normalized


def validate_binary_cross_entropy_with_logits_vjp_contract(
    node: Any,
) -> Tuple[BinaryCrossEntropyWithLogitsContract, int]:
    return validate_internal_operation_contract(node)


def validate_kl_div_contract(node: Any) -> KlDivContract:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.functional.kl-div.v1":
        raise ShapeError("KL_DIV resolved to the wrong framework-operation contract")
    return normalized


def validate_kl_div_vjp_contract(node: Any) -> Tuple[KlDivContract, int]:
    return validate_internal_operation_contract(node)


def validate_nll_loss_contract(node: Any) -> NllLossContract:
    record, normalized = validate_framework_operation_contract(node)
    if record.contract_id != "browsergrad.jit.framework.functional.nll-loss.v1":
        raise ShapeError(
            "NLL_LOSS resolved to the wrong framework-operation contract"
        )
    return normalized


def validate_nll_loss_vjp_contract(node: Any) -> NllLossContract:
    return validate_internal_operation_contract(node)


def validate_cross_entropy_contract(node: Any) -> CrossEntropyContract:
    record, normalized = validate_framework_operation_contract(node)
    if (
        record.contract_id
        != "browsergrad.jit.framework.functional.cross-entropy.v1"
    ):
        raise ShapeError(
            "CROSS_ENTROPY resolved to the wrong framework-operation contract"
        )
    return normalized


def validate_cross_entropy_vjp_contract(
    node: Any,
) -> tuple[CrossEntropyContract, int]:
    return validate_internal_operation_contract(node)


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
    "L1_LOSS_RANK_MAX",
    "L1_LOSS_OUTPUT_BYTE_MAX",
    "L1_LOSS_OUTPUT_EXTENT_MAX",
    "L1_LOSS_WORK_ELEMENT_MAX",
    "L1_LOSS_WORKSPACE_BYTE_MAX",
    "L1_LOSS_WORK_VISIT_FACTOR",
    "SMOOTH_L1_LOSS_WORK_VISIT_FACTOR",
    "BINARY_CROSS_ENTROPY_WORK_VISIT_FACTOR",
    "BINARY_CROSS_ENTROPY_LOG_FLOOR",
    "BINARY_CROSS_ENTROPY_GRAD_EPSILON",
    "BINARY_CROSS_ENTROPY_WITH_LOGITS_WORK_VISIT_FACTOR",
    "KL_DIV_WORK_VISIT_FACTOR",
    "NLL_LOSS_WORK_VISIT_FACTOR",
    "CROSS_ENTROPY_WORK_VISIT_FACTOR",
    "MASKED_FILL_CONTRACT_ID",
    "FrameworkOperationContract",
    "EinsumContract",
    "L1LossContract",
    "SmoothL1LossContract",
    "BinaryCrossEntropyContract",
    "BinaryCrossEntropyWithLogitsContract",
    "KlDivContract",
    "NllLossContract",
    "CrossEntropyContract",
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
    "infer_l1_loss_contract",
    "infer_smooth_l1_loss_contract",
    "infer_binary_cross_entropy_contract",
    "infer_binary_cross_entropy_with_logits_contract",
    "infer_kl_div_contract",
    "infer_nll_loss_contract",
    "infer_cross_entropy_contract",
    "normalize_smooth_l1_beta",
    "normalize_pad_request",
    "normalize_pad_value",
    "normalize_sort_request",
    "normalize_topk_request",
    "normalize_scatter_request",
    "scatter_index_violation",
    "execute_einsum_arrays",
    "execute_einsum_vjp_array",
    "execute_l1_loss_arrays",
    "execute_l1_loss_vjp_array",
    "execute_smooth_l1_loss_arrays",
    "execute_smooth_l1_loss_vjp_array",
    "execute_binary_cross_entropy_arrays",
    "execute_binary_cross_entropy_vjp_array",
    "execute_binary_cross_entropy_with_logits_arrays",
    "execute_binary_cross_entropy_with_logits_vjp_array",
    "execute_kl_div_arrays",
    "execute_kl_div_vjp_array",
    "execute_nll_loss_arrays",
    "execute_nll_loss_vjp_array",
    "execute_cross_entropy_arrays",
    "execute_cross_entropy_vjp_array",
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
    "validate_l1_loss_contract",
    "validate_l1_loss_vjp_contract",
    "validate_smooth_l1_loss_contract",
    "validate_smooth_l1_loss_vjp_contract",
    "validate_binary_cross_entropy_contract",
    "validate_binary_cross_entropy_vjp_contract",
    "validate_binary_cross_entropy_with_logits_contract",
    "validate_binary_cross_entropy_with_logits_vjp_contract",
    "validate_kl_div_contract",
    "validate_kl_div_vjp_contract",
    "validate_nll_loss_contract",
    "validate_nll_loss_vjp_contract",
    "validate_cross_entropy_contract",
    "validate_cross_entropy_vjp_contract",
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
