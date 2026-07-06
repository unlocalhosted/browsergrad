"""browsergrad_jit._gpu_plan - tensor-IR execution plan scaffold.

INTERNAL. This is the first compiler-facing layer for the GPU-native path:
UOp graph in, backend-neutral tensor execution plan out. It deliberately does
not call the legacy per-op WebGPU bridge and it refuses CUSTOM by default.

The plan is still conservative: one schedule item per primitive IR node, no
WGSL emission, no fusion/tiling choices yet. Its job is to pin the contract
that future WebGPU lowering must satisfy: primitive tensor IR, explicit
liveness/materialization, CPU reference parity, no hidden readbacks.
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
)


_DTYPE_BYTES: Dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "int64": 8,
    "int32": 4,
    "bool": 1,
}


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


__all__ = [
    "PRIMITIVE_GPU_IR_OPS",
    "PlanBuffer",
    "PlanStep",
    "GpuExecutionPlan",
    "GpuPlanUnsupported",
    "build_gpu_execution_plan",
    "gpu_plan_summary",
]
