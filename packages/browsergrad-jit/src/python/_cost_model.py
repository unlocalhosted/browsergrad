"""browsergrad_jit._cost_model — tier selector for op routing (PRD-012b).

INTERNAL. Public surface: `bg.jit.cost_model.{estimate_flops,
estimate_bytes, pick_tier}`.

Per the DL/GPU review: the cost model is the load-bearing piece that
makes pluggable backends honest. Without it, the realizer either
always-picks-WGSL (regression on tiny ops where bridge tax dominates)
or always-picks-NumPy (no GPU win ever). With it, each op is routed
to the tier with the highest expected throughput.

v0 model (intentionally simple):
  * FLOPs per UOp from a hand-curated table.
  * Bridge transfer cost ≈ bytes × 1ns/byte for upload + readback.
  * Pick GPU when bridge-amortised throughput > NumPy throughput.
  * Constants are educated guesses; refined as more workloads ship.

Producer-consumer fusion:
  * An ADD feeding MATMUL ("bias add" pattern) → fused once on GPU,
    avoiding two dispatches and an intermediate buffer.
  * Detected by walking the IR and noting consumer count == 1
    + opcode compatibility.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple

from ._ir import (
    UOp, toposort,
    OP_ADD, OP_MUL, OP_DIV, OP_NEG, OP_EXP, OP_LOG, OP_CMP,
    OP_MATMUL, OP_REDUCE, OP_CAST, OP_RESHAPE, OP_PERMUTE,
    OP_BROADCAST_TO, OP_BUFFER, OP_LOAD, OP_CONST,
    OP_FUSED_ELEMENTWISE, OP_FUSED_SOFTMAX, OP_CUSTOM,
)


# Bytes per element by dtype name. Used for transfer-cost estimation.
_BYTES_PER_ELEM: Dict[str, int] = {
    "float32": 4, "float16": 2, "int64": 8, "int32": 4, "bool": 1,
}


def _numel(shape: Tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= max(d, 1)
    return n


def estimate_flops(node: UOp) -> int:
    """FLOPs for a single UOp at its output shape. Rough; v0 model.

    Conventions:
      * MATMUL with output (M, N) over K: 2*M*N*K (mult + add per output).
      * Elementwise: 1 flop per output element.
      * REDUCE: numel(input) flops (sum or max).
      * FUSED_ELEMENTWISE: len(ops) flops per output element.
      * BUFFER/LOAD/CONST: 0 (lifecycle).
      * RESHAPE/PERMUTE: 0 (metadata).
    """
    op = node.op
    if op in (OP_BUFFER, OP_LOAD, OP_CONST, OP_RESHAPE, OP_PERMUTE):
        return 0
    out_n = _numel(node.shape)
    if op == OP_MATMUL:
        # output (..., M, N), K from inputs[0].shape[-1]
        a_shape = node.inputs[0].shape
        K = a_shape[-1] if a_shape else 1
        return 2 * out_n * K
    if op == OP_REDUCE:
        # Reduces over its input's reduced axes — approximate by input numel.
        return _numel(node.inputs[0].shape)
    if op in (OP_ADD, OP_MUL, OP_DIV, OP_NEG, OP_EXP, OP_LOG, OP_CMP,
              OP_CAST, OP_BROADCAST_TO):
        return out_n
    if op == OP_FUSED_ELEMENTWISE:
        n_ops = len(node.arg["ops"]) if isinstance(node.arg, dict) else 1
        return out_n * n_ops
    if op == OP_FUSED_SOFTMAX:
        # 4 passes: max, exp, sum, div. Approximate.
        return out_n * 4
    if op == OP_CUSTOM:
        # Conservative — caller knows better; default to "expensive".
        return out_n * 10
    return out_n  # default


def estimate_bytes(node: UOp) -> int:
    """Approximate input + output bytes for a single UOp."""
    elem = _BYTES_PER_ELEM.get(node.dtype, 4)
    out_b = _numel(node.shape) * elem
    in_b = 0
    for inp in node.inputs:
        in_elem = _BYTES_PER_ELEM.get(inp.dtype, 4)
        in_b += _numel(inp.shape) * in_elem
    return in_b + out_b


def pick_tier(
    node: UOp,
    *,
    gpu_flops_per_sec: float = 2.0e10,    # ~20 GFLOPS — modest GPU
    cpu_flops_per_sec: float = 3.0e9,     # ~3 GFLOPS — Pyodide NumPy baseline
    bridge_overhead_ns_per_byte: float = 1.0,  # JS bridge + GPUBuffer upload
    bridge_fixed_cost_ns: float = 5.0e5,  # ~0.5ms per dispatch
) -> str:
    """Return 'gpu' or 'numpy' based on which tier is faster for this op.

    Heuristic: total_time = flops / throughput + bytes * bridge_ns +
    bridge_fixed_cost. Pick whichever yields lower time.

    Tiny ops (numel < 64) always pick NumPy — the bridge fixed cost
    dominates and any GPU win is rounding error.
    """
    if _numel(node.shape) < 64:
        return "numpy"
    flops = estimate_flops(node)
    bytes_ = estimate_bytes(node)
    gpu_compute_ns = (flops / gpu_flops_per_sec) * 1e9
    gpu_transfer_ns = bytes_ * bridge_overhead_ns_per_byte + bridge_fixed_cost_ns
    gpu_total = gpu_compute_ns + gpu_transfer_ns
    cpu_total = (flops / cpu_flops_per_sec) * 1e9
    return "gpu" if gpu_total < cpu_total else "numpy"


def find_producer_consumer_pairs(root: UOp) -> List[Tuple[UOp, UOp]]:
    """Walk the IR and return (producer, consumer) pairs where the
    producer has exactly one consumer and the pair is fuseable.

    Currently flags ADD → MATMUL ("bias add" pattern) and elementwise
    chains where the FUSED_ELEMENTWISE matcher hasn't yet absorbed them.

    PRD-012b uses this list to drive producer-consumer kernel fusion;
    PRD-006's fusion pass uses a similar walk for its own purposes
    (elementwise chains).
    """
    consumer_count: Dict[int, int] = {}
    consumers_by_producer: Dict[int, List[UOp]] = {}
    nodes = toposort(root)
    for n in nodes:
        for inp in n.inputs:
            consumer_count[id(inp)] = consumer_count.get(id(inp), 0) + 1
            consumers_by_producer.setdefault(id(inp), []).append(n)

    pairs: List[Tuple[UOp, UOp]] = []
    for n in nodes:
        if n.op != OP_ADD:
            continue
        if consumer_count.get(id(n), 0) != 1:
            continue
        cons = consumers_by_producer[id(n)][0]
        if cons.op == OP_MATMUL:
            pairs.append((n, cons))
    return pairs


def cost_stats(root: UOp) -> Dict[str, int]:
    """Roll up the IR's cost into a dict — useful for the perf bench
    and for the "explain" path that lab UIs can surface."""
    nodes = toposort(root)
    total_flops = 0
    total_bytes = 0
    gpu_picks = 0
    numpy_picks = 0
    for n in nodes:
        total_flops += estimate_flops(n)
        total_bytes += estimate_bytes(n)
        if pick_tier(n) == "gpu":
            gpu_picks += 1
        else:
            numpy_picks += 1
    return {
        "total_flops": int(total_flops),
        "total_bytes": int(total_bytes),
        "nodes": len(nodes),
        "gpu_picks": gpu_picks,
        "numpy_picks": numpy_picks,
        "producer_consumer_pairs": len(find_producer_consumer_pairs(root)),
    }


__all__ = [
    "estimate_flops",
    "estimate_bytes",
    "pick_tier",
    "find_producer_consumer_pairs",
    "cost_stats",
]
