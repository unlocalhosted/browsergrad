"""browsergrad_jit._gpu_plan - tensor-IR execution plan scaffold.

INTERNAL. This is the first compiler-facing layer for the GPU-native path:
UOp graph in, backend-neutral tensor execution plan out. It deliberately does
not call the legacy per-op WebGPU bridge and it refuses CUSTOM by default.

The plan is still conservative, but it owns the first scheduler/codegen
choices: linear elementwise chains lower to one FUSED_ELEMENTWISE primitive,
and canonical softmax DAGs lower to one FUSED_SOFTMAX primitive.
Its job is to pin the contract that future WebGPU lowering must satisfy:
primitive tensor IR, explicit liveness/materialization, CPU reference parity,
no hidden readbacks.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ._errors import JitNotImplementedError
from ._ir import (
    UOp, toposort,
    OP_BUFFER, OP_LOAD, OP_CONST, OP_CAST,
    OP_ADD, OP_MUL, OP_DIV, OP_NEG, OP_EXP, OP_LOG, OP_CMP,
    OP_MATMUL,
    OP_CONV1D, OP_CONV1D_BACKWARD_INPUT, OP_CONV1D_BACKWARD_WEIGHT,
    OP_CONV1D_BACKWARD_BIAS,
    OP_CONV2D, OP_CONV2D_BACKWARD_INPUT, OP_CONV2D_BACKWARD_WEIGHT,
    OP_CONV2D_BACKWARD_BIAS,
    OP_CONV_TRANSPOSE2D, OP_CONV_TRANSPOSE2D_BACKWARD_INPUT,
    OP_CONV_TRANSPOSE2D_BACKWARD_WEIGHT, OP_CONV_TRANSPOSE2D_BACKWARD_BIAS,
    OP_CONV3D, OP_CONV3D_BACKWARD_INPUT, OP_CONV3D_BACKWARD_WEIGHT,
    OP_CONV3D_BACKWARD_BIAS,
    OP_LAYER_NORM, OP_LAYER_NORM_BACKWARD_INPUT,
    OP_LAYER_NORM_BACKWARD_WEIGHT, OP_LAYER_NORM_BACKWARD_BIAS,
    OP_REDUCE, OP_RESHAPE, OP_PERMUTE, OP_SLICE, OP_PAD,
    OP_WHERE, OP_INDEX, OP_MASK, OP_SCATTER_ADD, OP_BROADCAST_TO,
    OP_ISNAN, OP_SGD_UPDATE, OP_ADAMW_UPDATE_M, OP_ADAMW_UPDATE_V,
    OP_ADAMW_UPDATE_PARAM, OP_ADAM_UPDATE_M, OP_ADAM_UPDATE_V,
    OP_ADAM_UPDATE_PARAM, OP_FUSED_ELEMENTWISE, OP_FUSED_SOFTMAX, OP_CUSTOM,
    validate_broadcast_to_contract,
)


_DTYPE_BYTES: Dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "int64": 8,
    "int32": 4,
    "bool": 1,
}


TENSOR_PLAN_SEMANTIC_REQUEST_SCHEMA = "browsergrad.jit.tensor-plan-semantic-requests"
TENSOR_PLAN_SEMANTIC_REQUEST_VERSION = (1, 0)
DENSE_PERMUTATION_VIEW_COPY_REQUEST = "dense-permutation-view-copy"
_WIRE_I64_MAX = (1 << 63) - 1


PRIMITIVE_GPU_IR_OPS = frozenset({
    OP_BUFFER, OP_LOAD, OP_CONST, OP_CAST,
    OP_ADD, OP_MUL, OP_DIV, OP_NEG, OP_EXP, OP_LOG, OP_CMP,
    OP_MATMUL,
    OP_CONV1D, OP_CONV1D_BACKWARD_INPUT, OP_CONV1D_BACKWARD_WEIGHT,
    OP_CONV1D_BACKWARD_BIAS,
    OP_CONV2D, OP_CONV2D_BACKWARD_INPUT, OP_CONV2D_BACKWARD_WEIGHT,
    OP_CONV2D_BACKWARD_BIAS,
    OP_CONV_TRANSPOSE2D, OP_CONV_TRANSPOSE2D_BACKWARD_INPUT,
    OP_CONV_TRANSPOSE2D_BACKWARD_WEIGHT, OP_CONV_TRANSPOSE2D_BACKWARD_BIAS,
    OP_CONV3D, OP_CONV3D_BACKWARD_INPUT, OP_CONV3D_BACKWARD_WEIGHT,
    OP_CONV3D_BACKWARD_BIAS,
    OP_LAYER_NORM, OP_LAYER_NORM_BACKWARD_INPUT,
    OP_LAYER_NORM_BACKWARD_WEIGHT, OP_LAYER_NORM_BACKWARD_BIAS,
    OP_REDUCE, OP_RESHAPE, OP_PERMUTE, OP_SLICE, OP_PAD,
    OP_WHERE, OP_INDEX, OP_MASK, OP_SCATTER_ADD, OP_BROADCAST_TO,
    OP_ISNAN, OP_SGD_UPDATE, OP_ADAMW_UPDATE_M, OP_ADAMW_UPDATE_V,
    OP_ADAMW_UPDATE_PARAM, OP_ADAM_UPDATE_M, OP_ADAM_UPDATE_V,
    OP_ADAM_UPDATE_PARAM, OP_FUSED_ELEMENTWISE, OP_FUSED_SOFTMAX,
})


@dataclass(frozen=True)
class PlanBuffer:
    """Device-resident value lifetime for one UOp result."""

    value_id: int
    op: str
    shape: Tuple[int, ...]
    dtype: str
    bytes: int
    first_step: int
    last_step: int
    materialize: bool


@dataclass(frozen=True)
class PlanStep:
    """One scheduled tensor-IR operation."""

    step: int
    value_id: int
    op: str
    input_ids: Tuple[int, ...]
    shape: Tuple[int, ...]
    dtype: str
    arg: Any


@dataclass(frozen=True)
class GpuExecutionPlan:
    """Backend-neutral plan consumed by future WGSL lowering/runtime."""

    steps: Tuple[PlanStep, ...]
    buffers: Tuple[PlanBuffer, ...]
    root_id: int
    materialization_boundary: str
    peak_live_bytes: int
    has_custom_ops: bool

    def summary(self) -> Dict[str, Any]:
        return {
            "step_count": len(self.steps),
            "root_id": self.root_id,
            "materialization_boundary": self.materialization_boundary,
            "peak_live_bytes": self.peak_live_bytes,
            "has_custom_ops": self.has_custom_ops,
            "ops": [step.op for step in self.steps],
            "steps": [
                {
                    "step": step.step,
                    "value_id": step.value_id,
                    "op": step.op,
                    "input_ids": list(step.input_ids),
                    "shape": list(step.shape),
                    "dtype": step.dtype,
                    "arg": step.arg,
                }
                for step in self.steps
            ],
            "buffers": [
                {
                    "value_id": buf.value_id,
                    "op": buf.op,
                    "shape": list(buf.shape),
                    "dtype": buf.dtype,
                    "bytes": buf.bytes,
                    "first_step": buf.first_step,
                    "last_step": buf.last_step,
                    "materialize": buf.materialize,
                }
                for buf in self.buffers
            ],
        }


@dataclass(frozen=True)
class DensePermutationViewCopyRequest:
    """Semantic-core construction request plus plan-local routing identity.

    ``value_id`` correlates the verified side-table entry with one frozen plan
    step. It is deliberately excluded from the semantic constructor inputs.
    """

    value_id: int
    input_shape: Tuple[str, ...]
    axes: Tuple[int, ...]
    dtype: str

    def summary(self) -> Dict[str, Any]:
        return {
            "kind": DENSE_PERMUTATION_VIEW_COPY_REQUEST,
            "valueId": self.value_id,
            "inputShape": list(self.input_shape),
            "axes": list(self.axes),
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class GpuExecutionSubmission:
    """One frozen legacy plan and its separately versioned semantic requests."""

    plan: GpuExecutionPlan
    semantic_requests: Tuple[DensePermutationViewCopyRequest, ...]

    def plan_summary(self) -> Dict[str, Any]:
        summary = self.plan.summary()
        # Semantic-route PERMUTE meaning lives only in the side table. Erase
        # legacy axes and non-serializable VJP annotations from this compatible
        # scheduling/liveness projection; gpu_plan_summary() stays frozen.
        for step in summary["steps"]:
            if step["op"] == OP_PERMUTE:
                step["arg"] = None
        return summary

    def semantic_request_summary(self) -> Dict[str, Any]:
        return {
            "schema": TENSOR_PLAN_SEMANTIC_REQUEST_SCHEMA,
            "version": {
                "major": TENSOR_PLAN_SEMANTIC_REQUEST_VERSION[0],
                "minor": TENSOR_PLAN_SEMANTIC_REQUEST_VERSION[1],
            },
            "requests": [request.summary() for request in self.semantic_requests],
        }


class GpuPlanUnsupported(JitNotImplementedError):
    """Raised when a graph cannot enter the canonical GPU tensor plan."""


def _numel(shape: Tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= max(int(d), 1)
    return n


def _bytes_for(shape: Tuple[int, ...], dtype: str) -> int:
    return _numel(shape) * _DTYPE_BYTES.get(dtype, 4)


def _find_custom(node: UOp) -> Optional[UOp]:
    for n in toposort(node):
        if n.op == OP_CUSTOM:
            return n
    return None


def build_gpu_execution_plan(root: UOp, *, allow_custom: bool = False) -> GpuExecutionPlan:
    """Build backend-neutral GPU plan from primitive tensor IR.

    `allow_custom=False` is intentional. Core framework GPU lowering must not
    hide behind Python CUSTOM callbacks. User-authored/lab kernels can opt into
    CUSTOM-specific paths outside this core planner.
    """
    if not allow_custom:
        from ._fusion import fuse
        root = fuse(root)

    custom = _find_custom(root)
    if custom is not None and not allow_custom:
        label = None
        if isinstance(custom.arg, dict):
            label = custom.arg.get("op") or custom.arg.get("kernel_name")
        raise GpuPlanUnsupported(
            "GPU tensor plan refuses CUSTOM op "
            f"{label!r}. Promote framework ops to primitive IR; use the "
            "custom-kernel path only for explicit lab/user kernels."
        )

    order = toposort(root)
    step_index: Dict[int, int] = {id(node): i for i, node in enumerate(order)}
    last_use: Dict[int, int] = {id(node): i for i, node in enumerate(order)}
    for i, node in enumerate(order):
        if node.op == OP_BROADCAST_TO:
            validate_broadcast_to_contract(node)
        if node.op not in PRIMITIVE_GPU_IR_OPS and not (allow_custom and node.op == OP_CUSTOM):
            raise GpuPlanUnsupported(
                f"GPU tensor plan does not support opcode {node.op!r}. "
                "Add primitive IR lowering/refusal before GPU codegen."
            )
        for inp in node.inputs:
            last_use[id(inp)] = i

    steps: List[PlanStep] = []
    buffers: List[PlanBuffer] = []
    for node in order:
        sid = step_index[id(node)]
        steps.append(
            PlanStep(
                step=sid,
                value_id=sid,
                op=node.op,
                input_ids=tuple(step_index[id(inp)] for inp in node.inputs),
                shape=tuple(node.shape),
                dtype=node.dtype,
                arg=node.arg,
            )
        )
        buffers.append(
            PlanBuffer(
                value_id=sid,
                op=node.op,
                shape=tuple(node.shape),
                dtype=node.dtype,
                bytes=_bytes_for(tuple(node.shape), node.dtype),
                first_step=sid,
                last_step=last_use[id(node)],
                materialize=node is root,
            )
        )

    peak = 0
    for i in range(len(order)):
        live = 0
        for buf in buffers:
            if buf.first_step <= i <= buf.last_step:
                live += buf.bytes
        peak = max(peak, live)

    return GpuExecutionPlan(
        steps=tuple(steps),
        buffers=tuple(buffers),
        root_id=step_index[id(root)],
        materialization_boundary="root",
        peak_live_bytes=peak,
        has_custom_ops=custom is not None,
    )


def gpu_plan_summary(root: UOp, *, allow_custom: bool = False) -> Dict[str, Any]:
    """Small dict wrapper for tests/docs/UI."""
    return build_gpu_execution_plan(root, allow_custom=allow_custom).summary()


def build_gpu_execution_submission(
    root: UOp,
    *,
    allow_custom: bool = False,
) -> GpuExecutionSubmission:
    """Build the frozen plan once and emit semantic requests beside it.

    The side table is produced at the JIT boundary, before the kernels package
    sees the legacy tensor plan. Kernels must use the verified artifacts built
    from these requests as semantic authority; plan fields remain scheduling
    and liveness compatibility data only.
    """
    plan = build_gpu_execution_plan(root, allow_custom=allow_custom)
    steps_by_value_id = {step.value_id: step for step in plan.steps}
    requests = tuple(
        _dense_permutation_request(step, steps_by_value_id)
        for step in plan.steps
        if step.op == OP_PERMUTE
    )
    return GpuExecutionSubmission(plan=plan, semantic_requests=requests)


def _dense_permutation_request(
    step: PlanStep,
    steps_by_value_id: Dict[int, PlanStep],
) -> DensePermutationViewCopyRequest:
    path = f"PERMUTE value_id={step.value_id}"
    if len(step.input_ids) != 1:
        _semantic_request_unsupported(path, "requires exactly one input")
    source_value_id = step.input_ids[0]
    source = steps_by_value_id.get(source_value_id)
    if source is None:
        _semantic_request_unsupported(
            path,
            f"references missing source value_id={source_value_id}",
        )
    if step.dtype != "float32" or source.dtype != "float32":
        _semantic_request_unsupported(
            path,
            "initial semantic view-copy lowering requires float32 source and destination",
        )

    input_shape = tuple(source.shape)
    rank = len(input_shape)
    if rank not in (2, 3):
        _semantic_request_unsupported(path, f"initial semantic view-copy lowering requires rank 2 or 3, got {rank}")
    for axis, extent in enumerate(input_shape):
        if type(extent) is not int or extent <= 0:
            _semantic_request_unsupported(
                path,
                f"input shape axis {axis} must be a positive static integer, got {extent!r}",
            )
        if extent > _WIRE_I64_MAX:
            _semantic_request_unsupported(
                path,
                f"input shape axis {axis} exceeds canonical signed-64 wire range",
            )

    if type(step.arg) is not dict or "axes" not in step.arg:
        _semantic_request_unsupported(path, "arg must contain canonical tuple axes")
    unexpected_arg_fields = set(step.arg) - {"axes", "vjp_of"}
    if unexpected_arg_fields:
        _semantic_request_unsupported(
            path,
            f"arg contains unsupported fields {sorted(unexpected_arg_fields)!r}",
        )
    raw_axes = step.arg["axes"]
    if type(raw_axes) is not tuple:
        _semantic_request_unsupported(path, "axes must be a canonical tuple")
    axes = tuple(raw_axes)
    if (
        len(axes) != rank
        or any(type(axis) is not int or axis < 0 or axis >= rank for axis in axes)
        or len(set(axes)) != rank
    ):
        _semantic_request_unsupported(path, f"axes must be an exact permutation of [0, {rank})")

    expected_output_shape = tuple(input_shape[axis] for axis in axes)
    if tuple(step.shape) != expected_output_shape:
        _semantic_request_unsupported(
            path,
            f"declared output shape {tuple(step.shape)!r} does not match derived shape {expected_output_shape!r}",
        )
    return DensePermutationViewCopyRequest(
        value_id=step.value_id,
        input_shape=tuple(str(extent) for extent in input_shape),
        axes=axes,
        dtype="f32",
    )


def _semantic_request_unsupported(path: str, message: str) -> None:
    raise GpuPlanUnsupported(f"GPU semantic request {path}: {message}")


__all__ = [
    "PRIMITIVE_GPU_IR_OPS",
    "PlanBuffer",
    "PlanStep",
    "GpuExecutionPlan",
    "DensePermutationViewCopyRequest",
    "GpuExecutionSubmission",
    "GpuPlanUnsupported",
    "build_gpu_execution_plan",
    "build_gpu_execution_submission",
    "gpu_plan_summary",
    "TENSOR_PLAN_SEMANTIC_REQUEST_SCHEMA",
    "TENSOR_PLAN_SEMANTIC_REQUEST_VERSION",
    "DENSE_PERMUTATION_VIEW_COPY_REQUEST",
]
