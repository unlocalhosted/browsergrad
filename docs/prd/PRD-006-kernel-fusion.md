# PRD-006 — Kernel Fusion (Elementwise + Reduce/Softmax)

**Status**: Draft
**Author**: vijayksingh
**Created**: 2026-05-26
**Target milestone**: P1 (Months 4–9)
**Roadmap ref**: [PRD.md](../../PRD.md) §7 P1.2 + P1.3
**Companion PRDs**: PRD-005 (JIT foundation — prerequisite), PRD-007 (symbolic backward), PRD-012 (megakernel codegen for transformer blocks)
**Implementation estimate**: 6 weeks

---

## 1. TL;DR

Every WebGPU kernel dispatch carries a fixed overhead of approximately 95 μs regardless of the computation it runs ([arXiv:2604.02344](https://arxiv.org/abs/2604.02344)). A transformer forward pass with 20 elementwise ops pays 1.9 ms in dispatch tax before a single multiply-add executes. PRD-006 eliminates this tax through two targeted fusion passes over the IR introduced in PRD-005. Phase 1 detects chains of elementwise ops in the UOp graph — any linear sequence of unary/binary nodes (add, mul, relu, sigmoid, etc.) where each node's output is consumed by exactly one downstream node of the same elementwise family — and emits a single WGSL compute shader that performs all N ops in one workgroup dispatch. Phase 2 targets the softmax pattern specifically: exp + reduce_max + subtract + exp + reduce_sum + divide, matching the Flash Attention stability formulation ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135)), and emitting one kernel that fuses the three reducing passes into a single workgroup-local pass using WebGPU workgroup memory. Together these two passes are projected to cut dispatch count by 35–50% on transformer-shaped workloads, with softmax specifically expected to show a 3× throughput improvement matching the reduction in kernel launches. Fusion is on by default and disableable via `BG_DISABLE_FUSION=1` for debugging. This PRD does NOT ship attention fusion (Flash Attention full tiling) — that is PRD-012.

---

## 2. Background

### 2.1 The 95 μs dispatch floor

[arXiv:2604.02344](https://arxiv.org/abs/2604.02344) provides the most precise publicly available measurement of WebGPU dispatch overhead: 24–36 μs of API overhead on Vulkan and 32–71 μs on Metal, with total per-operation stack cost (including Python interpreter overhead via Pyodide and the JSPI bridge) reaching approximately **95 μs per dispatch**. The same paper measured a reduction in dispatch count from 876 to 564 ops on a representative workload and reported a **53% throughput improvement** — a superlinear gain because each saved dispatch reclaims both the API overhead and the GPU idle time while the CPU queues the next command.

For educational-scale models, this overhead is dominant. A 2-layer MLP (hidden=512) has roughly 8 distinct dispatch-worthy ops per training step. At 95 μs each, dispatch tax alone is 760 μs — more than the actual compute at those tensor sizes. A 6-layer transformer block has 30+ ops; dispatch tax exceeds 2.8 ms before any arithmetic runs.

Fusion is the only architectural response to this: reduce the number of dispatches, not the per-dispatch cost.

### 2.2 Elementwise fusion: the cheap win

An elementwise chain like `(x * 2.0 + bias).relu().sigmoid()` allocates four separate GPU buffers and fires four separate dispatches in the unfused path. Every intermediate value is written to and read from GPU global memory unnecessarily. In the fused path, one workgroup reads `x` and `bias`, applies all four operations per thread without any memory round-trip, and writes the final result. JAX's XLA HLO has shipped elementwise fusion since 2017; PyTorch's TorchInductor (the backend for `torch.compile`) uses the same approach. The wins are well-documented and the correctness risk is low — elementwise ops compose without numerical dependencies between output elements.

### 2.3 Softmax fusion: the stability trick

Naive softmax computed as three separate passes — `m = max(x)`, `e = exp(x - m)`, `y = e / sum(e)` — is correct but expensive: three kernel dispatches, three full reads of the input row, two intermediate buffers. Flash Attention's core insight ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135)) is that these three passes collapse to one if the workgroup coordinates its local max and sum using workgroup memory. A single kernel reads each element once, maintains a running max and running sum in workgroup-shared registers, and rescales on the fly. The technique is numerically equivalent to the three-pass version and is the industry standard for softmax in any fused kernel context.

### 2.4 Why this ships after PRD-005

Fusion is a graph rewrite over the IR. Without the IR, there is no graph to rewrite — each op is a standalone eager dispatch with no graph structure. PRD-005 introduces the UOp graph, the tracer that builds it, and the IR realization pipeline. PRD-006 inserts a fusion pass between the tracer and the codegen step. This ordering is non-negotiable.

---

## 3. User Stories

- A student runs a GPT-style transformer block forward pass (hidden=768, heads=12, seq=128, batch=4). Currently (PRD-002 without fusion): 22 separate kernel dispatches, ~2.1 ms dispatch overhead. After this PRD: 11 dispatches (elementwise chains inside FFN fused, softmax fused), ~1.05 ms dispatch overhead, with eliminated memory round-trips yielding a measured throughput gain above and beyond the dispatch count reduction.
- A course author building an attention visualization lab needs softmax to run interactively at seq=512. The fused softmax kernel processes 512 elements in one workgroup pass; the lab feels responsive at interactive batch sizes.
- A developer debugging a training divergence sets `BG_DISABLE_FUSION=1` in the environment and gets back per-op dispatch behavior, where each intermediate value is a concrete GPU buffer they can inspect via `bg.jit.debug_fused_kernels()`.
- A library contributor implementing a new elementwise activation function (say, `silu`) adds it to the elementwise family registry and the fusion pass picks it up automatically without any additional changes.

---

## 4. Goals

1. **Elementwise chain fusion**: any linear UOp chain of unary/binary elementwise nodes where each node has exactly one consumer, fused into one WGSL kernel per chain.
2. **Softmax fusion**: the six-op softmax subgraph (reduce_max → subtract → exp → reduce_sum → divide) fused into one Flash Attention-style WGSL kernel with workgroup-local max and sum.
3. **Layernorm fusion**: as a direct corollary of softmax fusion, `LayerNorm` (reduce_mean → subtract → square → reduce_mean → rsqrt → multiply → add scale → add bias) fused into one kernel. Ships in week 5 alongside softmax.
4. **Feature flag**: `BG_DISABLE_FUSION=1` disables all fusion passes and reverts to per-op dispatch.
5. **Correctness oracle**: every fused kernel result verified against the unfused IR path within 1e-4 on the conformance suite.
6. **Introspection helper**: `bg.jit.debug_fused_kernels()` returns a list of fusion decisions made during the most recent realization.

## 5. Non-Goals

- **Attention block fusion (Flash Attention full tiling)**: fusing QKV matmul + softmax + V matmul into one megakernel with online softmax across tiles. This is PRD-012 and requires the full tiling algorithm; it is architecturally distinct from the per-row softmax fusion here.
- **Reduce fusion across non-contiguous axes**: axis=0 reductions, batch reductions, or reductions that require cross-workgroup synchronization beyond what a single dispatch supports.
- **Cross-op buffer sharing or memory planning**: deciding how to reuse intermediate buffers across fused groups. Deferred to PRD-012.
- **Backward pass fusion**: fusing the gradient computation kernels. The symbolic backward (PRD-007) must exist first; backward fusion is a future pass.
- **Mixed-precision fusion**: fusing fp16 and fp32 ops in the same kernel. Deferred to PRD-010.
- **Pattern-matching Flash Attention's full tiled algorithm**: that involves a fundamentally different kernel structure with tiled Q/K/V iteration, not just row-wise softmax. See PRD-012.

---

## 6. Architecture

### 6.1 Where the fusion pass lives

The fusion pass is a pure graph transformation that runs on the UOp graph after the tracer (PRD-005) has built the forward graph and before the WGSL codegen emits kernels. The pipeline is:

```
User Python code
    │
    ▼
Tracer (PRD-005)
    │  builds UOp graph (LOAD, STORE, ADD, MUL, EXP, REDUCE, ...)
    ▼
Fusion pass  ◄── PRD-006 inserts here
    │  rewrites UOp subgraphs → FusedUOp nodes
    ▼
WGSL codegen
    │  FusedUOp → one compute shader
    │  UnfusedUOp → existing per-op shaders
    ▼
Dispatcher → GPU dispatch or NumPy fallback
```

The fusion pass is a stateless function with signature:

```python
def fuse(graph: UOpGraph, config: FusionConfig) -> UOpGraph
```

It takes the IR graph, returns a new IR graph with fused nodes substituted, and does not mutate the input. `FusionConfig` holds the feature flags including whether fusion is enabled at all.

### 6.2 Elementwise fusion algorithm

The fusion algorithm is a single-pass greedy scan. Fixed-point iteration is not needed because the graph is a DAG and elementwise chains form linear paths that a single topological-order scan fully resolves.

**Pseudocode:**

```
ELEMENTWISE_OPS = {ADD, MUL, DIV, SUB, EXP, LOG, NEG, SQRT, RELU, SIGMOID,
                   TANH, GELU, SILU, ABS, CLAMP, POW}

function fuse_elementwise(graph: UOpGraph) -> UOpGraph:
    visited = set()
    groups = []

    for node in topological_order(graph):
        if node in visited:
            continue
        if node.op not in ELEMENTWISE_OPS:
            continue

        # Start a new chain from this node
        chain = [node]
        visited.add(node)
        cur = node

        while True:
            consumers = graph.consumers(cur)
            # Chain continues only if exactly one consumer AND it's elementwise
            if len(consumers) != 1:
                break
            next_node = consumers[0]
            if next_node.op not in ELEMENTWISE_OPS:
                break
            # External inputs are allowed (e.g. bias add); we include them as
            # kernel inputs, but do not extend the chain through them.
            chain.append(next_node)
            visited.add(next_node)
            cur = next_node

        if len(chain) > 1:
            groups.append(FusionGroup(chain, kind=ELEMENTWISE))

    return substitute_groups(graph, groups)
```

The key correctness invariant: `next_node` may have multiple inputs (e.g., an ADD with `x` from the chain and `bias` from a separate Load), but it must have exactly one *output consumer* to extend the chain. External inputs become additional kernel buffer bindings in the emitted WGSL.

**IR rewrite example — 3 elementwise ops → 1 fused UOp:**

Before fusion:

```
%0 = LOAD(buf_x)           # shape=[N], dtype=f32
%1 = LOAD(buf_w)           # shape=[N], dtype=f32
%2 = MUL(%0, %1)           # elementwise scale
%3 = LOAD(buf_b)           # shape=[N], dtype=f32
%4 = ADD(%2, %3)           # elementwise bias
%5 = RELU(%4)              # activation
%6 = STORE(%5, buf_out)
```

After fusion (single FusedUOp replaces %2, %4, %5):

```
%0 = LOAD(buf_x)
%1 = LOAD(buf_w)
%3 = LOAD(buf_b)
%F = FUSED_ELEMENTWISE(
         inputs=[%0, %1, %3],
         ops=[MUL(%0,%1), ADD(result_0,%3), RELU(result_1)],
         output_shape=[N]
     )
%6 = STORE(%F, buf_out)
```

The `FUSED_ELEMENTWISE` node carries the complete op sequence internally. The codegen traverses this sequence to emit the WGSL body.

### 6.3 WGSL kernel template for fused elementwise

For the example above (MUL → ADD → RELU over N elements), the codegen emits:

```wgsl
@group(0) @binding(0) var<storage, read>       buf_x : array<f32>;
@group(0) @binding(1) var<storage, read>       buf_w : array<f32>;
@group(0) @binding(2) var<storage, read>       buf_b : array<f32>;
@group(0) @binding(3) var<storage, read_write> buf_out : array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&buf_x)) { return; }

    // --- fused body: generated per chain ---
    var v0 : f32 = buf_x[idx] * buf_w[idx];   // MUL
    var v1 : f32 = v0 + buf_b[idx];            // ADD
    var v2 : f32 = max(v1, 0.0);               // RELU
    // --- end fused body ---

    buf_out[idx] = v2;
}
```

The codegen emits one binding per unique external input, one binding for the output, and one `var v{n}` per op node in the chain. No intermediate buffers are allocated. The workgroup size is 256 threads; the dispatch covers `ceil(N / 256)` workgroups.

For chains involving binary ops where both inputs come from within the chain, both `v{n}` references are used directly. For binary ops with one external input, the external input is added as a new binding. The codegen template is parameterized by the chain's op-sequence list and an input-binding map built during the graph substitution step.

### 6.4 Softmax fusion: IR pattern and WGSL kernel

**IR pattern matched:**

```
%x   = LOAD(buf_x)                     # shape=[B, S], dtype=f32
%m   = REDUCE(%x, op=MAX, axis=-1)     # shape=[B, 1]
%xs  = SUB(%x, %m)                     # broadcast subtract
%e   = EXP(%xs)
%s   = REDUCE(%e, op=SUM, axis=-1)     # shape=[B, 1]
%out = DIV(%e, %s)                     # normalize
```

This exact six-node subgraph is what the unfused softmax emits. The pattern matcher in the fusion pass checks: REDUCE(MAX) → SUB → EXP → REDUCE(SUM) → DIV, where all ops share the same reduction axis and the same input tensor `%x`.

**Emitted WGSL kernel (row-wise softmax, Flash Attention stability formulation):**

```wgsl
@group(0) @binding(0) var<storage, read>       x   : array<f32>;
@group(0) @binding(1) var<storage, read_write> out : array<f32>;

// Workgroup-local accumulators: one slot per thread in the workgroup
var<workgroup> wg_max : array<f32, 256>;
var<workgroup> wg_sum : array<f32, 256>;

@compute @workgroup_size(256)
fn softmax_fused(
    @builtin(global_invocation_id) gid  : vec3<u32>,
    @builtin(local_invocation_id)  lid  : vec3<u32>,
    @builtin(workgroup_id)         wid  : vec3<u32>
) {
    // One workgroup handles one row of the input.
    let row    = wid.x;
    let n_cols : u32 = uniforms.n_cols;
    let base   = row * n_cols;

    // --- Phase 1: thread-local max over this thread's column stripe ---
    var local_max : f32 = -1e38;
    var col = lid.x;
    loop {
        if (col >= n_cols) { break; }
        local_max = max(local_max, x[base + col]);
        col += 256u;
    }
    wg_max[lid.x] = local_max;
    workgroupBarrier();

    // --- Parallel reduction: max across workgroup ---
    var stride : u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) {
            wg_max[lid.x] = max(wg_max[lid.x], wg_max[lid.x + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let row_max = wg_max[0];

    // --- Phase 2: thread-local exp + sum ---
    var local_sum : f32 = 0.0;
    col = lid.x;
    loop {
        if (col >= n_cols) { break; }
        let e = exp(x[base + col] - row_max);
        out[base + col] = e;          // write exp values; overwritten below
        local_sum += e;
        col += 256u;
    }
    wg_sum[lid.x] = local_sum;
    workgroupBarrier();

    // --- Parallel reduction: sum across workgroup ---
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) {
            wg_sum[lid.x] = wg_sum[lid.x] + wg_sum[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let row_sum = wg_sum[0];

    // --- Phase 3: normalize ---
    col = lid.x;
    loop {
        if (col >= n_cols) { break; }
        out[base + col] = out[base + col] / row_sum;
        col += 256u;
    }
}
```

This kernel processes one row per workgroup. For a (B, S) input, the dispatch is `(B, 1, 1)` workgroups. Each thread handles `ceil(S / 256)` columns in the stripe loops. The workgroup reduction uses the standard log2-stride parallel reduction pattern over the 256 workgroup-local slots.

The Flash Attention stability property is provided by `x[i] - row_max` before the exp — the numerical values are shifted into the range (-∞, 0], preventing overflow regardless of input magnitude. This is exactly the online softmax stabilization described in [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) §2, adapted to the single-row case.

### 6.5 FusionConfig and feature flag

```python
# packages/browsergrad-jit/src/python/fusion_config.py

import os

class FusionConfig:
    elementwise: bool
    softmax:     bool
    layernorm:   bool

    def __init__(self):
        disabled = os.environ.get("BG_DISABLE_FUSION", "0") == "1"
        self.elementwise = not disabled
        self.softmax     = not disabled
        self.layernorm   = not disabled

FUSION_CONFIG = FusionConfig()
```

Setting `BG_DISABLE_FUSION=1` in the environment before the Pyodide worker boots disables all three fusion passes. Individual passes can also be disabled programmatically via `FUSION_CONFIG.elementwise = False` for targeted debugging.

---

## 7. API Surface

### 7.1 Feature flag

```
BG_DISABLE_FUSION=1
```

Set in the host environment (Node process env for tests, or passed via the runtime's worker initialization config object). When set, the fusion pass returns the input graph unchanged. No user code changes are needed.

### 7.2 Introspection helper

```python
import browsergrad_jit as bg

# After running a model forward pass:
report = bg.jit.debug_fused_kernels()
```

Returns a list of `FusionDecision` objects, one per fused group identified in the most recent realization:

```python
@dataclass
class FusionDecision:
    kind:        str          # "elementwise" | "softmax" | "layernorm"
    ops_fused:   list[str]    # e.g. ["MUL", "ADD", "RELU"]
    n_ops:       int          # ops in the group
    input_shapes: list[tuple] # shapes of all external inputs
    output_shape: tuple
    dispatch_saved: int       # (n_ops - 1) dispatches eliminated
```

This is stored on the `JitState` singleton after each realization and cleared on the next `forward()` call. It is not retained across realizations and adds zero overhead when not called.

---

## 8. Implementation Plan

### Week 1 — Fusion pass skeleton + elementwise detection

- Create `/packages/browsergrad-jit/src/python/fusion_pass.py` with `fuse(graph, config) -> UOpGraph`.
- Implement `fuse_elementwise()` using the greedy chain algorithm from §6.2. No codegen yet — chains are identified and logged but the graph substitution is a no-op.
- Create `FusionConfig` in `fusion_config.py`. Wire `BG_DISABLE_FUSION` env var. Add `FusionConfig` to the `JitState` created in PRD-005.
- Unit tests: six test cases in `fusion_pass_test.py` — single-op chain (no fusion), 2-op chain, 3-op chain with external input, branching (no fusion), chain ending at a REDUCE (no fusion), `BG_DISABLE_FUSION=1` (no fusion).

### Week 2 — Elementwise WGSL codegen + conformance

- Extend the WGSL codegen (`codegen.py` from PRD-005) to handle `FUSED_ELEMENTWISE` nodes. Implement `emit_elementwise_kernel(group: FusionGroup) -> str` generating the kernel template from §6.3.
- Add binding map generation: for each op in the chain, enumerate external inputs not produced within the chain; assign `@binding(n)` indices.
- PyTorch-conformance: run the five existing fixtures through the fused path. All must pass within 1e-4.
- Add a microbenchmark for `(x * w + b).relu().sigmoid()` at sizes [256, 1024, 4096, 16384].

### Week 3 — Elementwise hardening + `debug_fused_kernels`

- Implement `substitute_groups()` in the graph rewrite step. Replaces chain nodes with a single `FUSED_ELEMENTWISE` node with correct edge rewiring.
- Implement `bg.jit.debug_fused_kernels()` and the `FusionDecision` dataclass.
- Edge cases: chain starting at a LOAD; two elementwise ops sharing an external buffer; `requires_grad` tensors through a fused chain (PRD-007 treats `FUSED_ELEMENTWISE` as an opaque composite for symbolic backward).
- Full 234-integration-test suite: green.

### Week 4 — Softmax pattern matcher + kernel

- Implement `fuse_softmax(graph) -> UOpGraph` in `fusion_pass.py`. Pattern matcher walks for the six-node subgraph from §6.4.
- Emit the WGSL kernel from §6.4 from `emit_softmax_kernel(group: SoftmaxGroup) -> str`. Kernel parameterized by `n_cols` via a uniform buffer.
- Conformance: softmax on shapes [(1,64), (4,128), (16,512), (64,1024)] within 1e-4.
- Numerical stability test: input with mean=100, std=10 — verify fused kernel does not overflow.

### Week 5 — Layernorm fusion + integration

- Implement `fuse_layernorm(graph) -> UOpGraph`. LayerNorm IR: REDUCE(MEAN) → SUB → POW(2) → REDUCE(MEAN) → ADD(eps) → RSQRT → MUL(gamma) → ADD(beta).
- Emit `emit_layernorm_kernel()`. One workgroup per row, parallel reduction for mean then variance then normalize.
- Verify PyTorch-conformance on LayerNorm fixture within 1e-4.
- Wire all three fusion passes into `fuse()` and run in order: elementwise → softmax → layernorm.

### Week 6 — Benchmarks, acceptance validation, docs

- Run acceptance benchmark: transformer block (hidden=768, heads=12, seq=128, batch=4) — measure dispatch count before/after, wall-clock throughput; verify ≥2× improvement in both.
- Softmax microbenchmark (seq=512, batch=64): verify ≥3× throughput vs unfused.
- Full integration test suite (234 tests) with fusion enabled: all green.
- Full integration test suite with `BG_DISABLE_FUSION=1`: all green.
- Changelog entry, README section on fusion, env var documentation.

---

## 9. Acceptance Criteria

1. **Elementwise speedup**: microbenchmark on `(x * w + b).relu().sigmoid()` at N=4096 shows ≥2× wall-clock improvement vs unfused on M1-class hardware with WebGPU.
2. **Dispatch count reduction**: transformer block (hidden=768, seq=128) — dispatch count reduced by ≥30% vs unfused.
3. **Softmax throughput**: softmax forward on (batch=64, seq=512) shows ≥3× throughput improvement vs three-pass unfused.
4. **Numerical correctness (elementwise)**: all five PyTorch-conformance fixtures pass within 1e-4 via the fused path.
5. **Numerical correctness (softmax)**: fused softmax matches NumPy reference within 1e-4 across all tested shapes, including extreme-value inputs.
6. **Numerical correctness (layernorm)**: fused LayerNorm matches NumPy within 1e-4.
7. **Fallback correctness**: all 234 integration tests pass with `BG_DISABLE_FUSION=1`, identical results within 1e-6 to fused path.
8. **Feature flag works**: `BG_DISABLE_FUSION=1` reverts every dispatch to per-op mode; `bg.jit.debug_fused_kernels()` returns empty.
9. **Introspection works**: `bg.jit.debug_fused_kernels()` returns non-empty list with correct `ops_fused` and `dispatch_saved` values after a transformer forward pass.

---

## 10. Test Strategy

**Unit tests** (`fusion_pass_test.py`):
- Greedy chain detection: chains of length 1, 2, 3, 4 detected correctly.
- Branch breaks fusion: a node with two consumers must not be included beyond that node.
- External input binding: chain containing an ADD with one external input gets a binding for that input.
- Softmax pattern match: six-node subgraph matched; five-node variant not matched.
- Layernorm pattern match: matched correctly; partial layernorm falls through to elementwise.
- `BG_DISABLE_FUSION=1`: `fuse()` returns input graph unchanged.

**Conformance tests** (against PyTorch 2.12.0):
- Elementwise: 10 op-sequence variants at shapes [64, 512, 4096, 65536].
- Softmax: 8 (batch, seq) shape combinations; 2 extreme-value test cases.
- LayerNorm: 6 normalized_shape variants, 3 batch sizes.

**Benchmark fixtures** (`benchmarks/fusion-bench.ts`):
- Elementwise chain (N=4096): unfused vs fused, 100 runs, median wall-clock, ≥2× improvement.
- Softmax row (batch=64, seq=512): unfused vs fused, ≥3× improvement.
- Transformer block forward: dispatch count via `debug_fused_kernels`, ≥30% reduction.

**Regression tests**:
- Full 234-test integration suite runs with and without fusion. Results match within 1e-6 between modes.

---

## 11. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | WGSL workgroup reduction incorrect for non-power-of-2 row sizes | Medium | High | Test on seq [100, 300, 500, 768] explicitly. Pad to next power of 2 in kernel if needed. |
| R2 | Fused elementwise produces different results due to FP associativity reordering | Low | Medium | Test against PyTorch within 1e-4 (not 1e-7). Document the tolerance. |
| R3 | Pattern matcher false positives on subgraphs that look like softmax but aren't (e.g., log-softmax) | Medium | Medium | Add discriminant check: SUB's second input must be REDUCE(MAX) output, not a constant. Log-softmax = separate named pattern. |
| R4 | Fusion pass runs on backward graphs (PRD-007) and incorrectly fuses gradient ops | Medium | Medium | Gate the pass with `graph.is_forward` flag. Only fuse forward graphs in this PRD; backward fusion deferred. |
| R5 | Workgroup memory size exceeds device limits (max 16KB on older devices) | Low | Low | `wg_max` and `wg_sum` are each 256×4 bytes = 1KB. Total 2KB is well within the 16KB minimum. |
| R6 | JSPI bridge adds per-dispatch overhead making 2× elementwise unachievable for short chains | Medium | Low | If 2× missed for chains of length 2, raise threshold to ≥3 ops. Threshold is a constant in `fusion_pass.py`. |

---

## 12. Open Questions

1. **Should `FUSED_ELEMENTWISE` appear in symbolic backward's differentiation rules?** PRD-007 needs to handle this node type. Options: (a) decompose to individual ops before differentiating; (b) treat fused as a composite and generate per-op VJP rules over its internal chain. Option (a) is simpler and correct; (b) enables backward fusion. Decide before PRD-007 implementation.

2. **Uniform buffer strategy for `n_cols` in softmax kernel?** Passing `n_cols` via uniform requires one additional binding per dispatch. Override constant at pipeline-compile time improves codegen specialization but increases pipeline cache pressure. Depends on PRD-008's pipeline key strategy.

3. **Fusion scope: should chains be bounded by buffer lifetime?** Current greedy algorithm doesn't bound chain length. For very long chains (N > 20), generated WGSL may exceed GPU shader complexity limits on some browsers. Evaluate chain-length cap (max 16 ops) after week 2 empirical results.

---

## 13. References

1. **Flash Attention**: Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) (2022). Softmax stability trick (online max + sum rescaling) used in §6.4 is described in §2.1.
2. **WebGPU dispatch overhead measurement**: [arXiv:2604.02344](https://arxiv.org/abs/2604.02344) — 95 μs total stack cost per dispatch; 876→564 dispatch reduction yielding 53% throughput gain.
3. **JAX elementwise fusion**: XLA HLO has shipped elementwise fusion as default since 2017. [https://openxla.org/xla](https://openxla.org/xla)
4. **TorchInductor elementwise fusion**: PyTorch's `torch.compile` backend uses pointwise fusion as first optimization pass. [TorchInductor design doc](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747).
5. **WebGPU workgroup memory spec**: WebGPU specification §10.4 defines workgroup address space, workgroupBarrier(), and 16 KB minimum workgroup memory limit. [https://www.w3.org/TR/webgpu/](https://www.w3.org/TR/webgpu/)
6. **tinygrad UOps IR**: tinygrad's `ops.py` is the design reference. [https://github.com/tinygrad/tinygrad/blob/master/tinygrad/ops.py](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/ops.py)
7. **VISION.md §4 Layer 3**: browsergrad vision document's description of the IR + kernel fusion layer.
8. **PRD.md §7 P1.2 + P1.3**: original feature specs for elementwise and softmax fusion.
