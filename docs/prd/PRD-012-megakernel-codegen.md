# PRD-012 — Megakernel Codegen: Producer-Consumer + Cross-Block Fusion

| Field | Value |
|---|---|
| **Status** | Draft v1 |
| **Author** | unlocalhosted maintainers |
| **Date** | 2026-05-26 |
| **PRD ID** | PRD-012 |
| **Phase** | P2 (Months 10–14 of the 14-month roadmap in PRD.md §6 — heaviest single PRD in the program) |
| **Package** | `@unlocalhosted/browsergrad-jit`, `@unlocalhosted/browsergrad-kernels` |
| **Depends on** | PRD-005 (IR), PRD-006 (per-row softmax / layernorm fusion + WGSL codegen), PRD-007 (unified forward+backward IR), PRD-008 (OPFS pipeline cache) |
| **Enables** | Sub-3s training step (PRD.md §9 target); fast.ai Part 2 Ch 13–25 interactive; nanoGPT end-to-end at 25% of native PyTorch |
| **Companion docs** | [VISION.md](../../VISION.md) §4 Layer 3 / §5 ResNet18 walkthrough; [PRD.md](../../PRD.md) §7 P2.2 |

---

## TL;DR

PRD-006 cut dispatch count by ~35–50% on transformer-shaped workloads through two narrow patterns: linear elementwise chains and per-row softmax / layernorm. The remaining bottleneck is not dispatch count — it is **DRAM bandwidth**. A standard attention block touching a (B=4, H=12, S=512, D=64) Q/K/V tensor moves ~150 MB through global memory per forward pass on the unfused-megakernel path, against a WebGPU FP32 peak bandwidth ceiling on M2-class hardware of ~80 GB/s ([nuss-and-bolts.com WGSL matmul](https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel), reporting ~17% peak achieved). PRD-012 attacks bandwidth directly: it generalises fusion from "consecutive cheap ops" to **producer-consumer fusion across matmul boundaries with workgroup-shared tiles**, **cross-block fusion** that swallows attention + residual + layernorm + FFN into a single dispatch, **Flash-Attention-v2-style tiled attention** with online softmax in workgroup memory ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135), [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)), and **forward+backward joint fusion** that stashes activations in workgroup memory rather than DRAM when the working set fits. The pipeline emits megakernels gated by a cost model that respects WebGPU's `maxComputeWorkgroupStorageSize` (default 16384 bytes per [GPUSupportedLimits](https://gpuweb.github.io/gpuweb/#gpusupportedlimits)) and falls back to PRD-006 kernels when limits are exceeded. Tile sizes are autotuned by sweeping `(BM, BN, BK)` over a small grid and cached per-shape in OPFS via PRD-008. The target is a transformer training step within 25% of native PyTorch on M2-class hardware — the 1/4-of-native ceiling claim from VISION.md §3.6 — without regressing the 1e-3 numerical tolerance established by PRD-007. Implementation budget: 10 weeks.

---

## Background

### From "fewer dispatches" to "fewer bytes"

PRD-006's softmax fusion saves three dispatches and one full read of the input row, yielding a measured 3× speedup. That speedup confirms a structural fact about WebGPU workloads on consumer hardware: dispatch overhead matters at the small-tensor end, but **once tensors exceed roughly 1 MB the cost is bandwidth, not launch**. A (B=4, H=12, S=512, D=64) attention tensor is 6.3 MB; touching it three times costs ~19 MB of traffic at 80 GB/s peak = 0.24 ms minimum — and the unfused implementation touches it eleven times, blowing past 2 ms.

The remedy is the same remedy used on every modern accelerator: **keep intermediate tensors in on-chip memory** (CUDA shared memory; WGSL `var<workgroup>` arrays; Metal threadgroup memory). For an attention block, this means the QK^T product never materialises to DRAM — it lives in workgroup memory long enough to be softmaxed and immediately consumed by the V matmul. The Flash Attention paper ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135)) is the canonical statement of this idea; Flash Attention v2 ([arXiv:2307.08691](https://arxiv.org/abs/2307.08691)) refines the work partitioning to push closer to peak.

### What "megakernel" actually means

A megakernel is not "one big kernel that does everything." It is **one WGSL `@compute` entry-point whose body executes a tiled iteration of multiple logical ops, with no STORE to global memory between them**. The forward pass of a transformer block decomposes into 30+ UOps (per PRD-005's IR), but only ~4 of those UOps need to write to DRAM:

1. The block's final hidden state (consumed by the next block).
2. Activations needed by backward that won't fit in workgroup memory.
3. The KV cache during autoregressive generation (a separate path).
4. Loss / metric outputs.

Every other intermediate — Q, K, V, scores, attention weights, attention output, post-projection residual, normed activations, FFN-up output, GeLU output — can in principle be **register- or workgroup-resident** if a single thread block owns enough of the computation. PRD-012 is the engineering work of identifying when that's true and emitting WGSL that exploits it.

### Why this ships after PRD-005/006/007/008

- **PRD-005** gave us the IR. Without UOps, there is no graph to rewrite.
- **PRD-006** gave us a WGSL codegen that handles workgroup memory and barriers in a contained setting (per-row softmax). PRD-012 generalises that machinery to multi-stage tiles.
- **PRD-007** gave us a backward IR rooted in the same UOp graph. Without symbolic backward, forward+backward joint fusion is impossible — closures are not graph-rewritable.
- **PRD-008** gave us OPFS pipeline caching. PRD-012's autotuner produces a combinatorial number of WGSL variants per shape; we need the cache to keep cold-start manageable.

### Reference precedents

- **Triton** (OpenAI) is the template for "Python-friendly tile-level fusion" ([Triton design blog](https://openai.com/research/triton)). Its core insight — that you write tile-level WGSL/CUDA, not thread-level — informs our codegen style.
- **tinygrad's scheduler** ([tinygrad/codegen/kernel.py](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/codegen/kernel.py)) is the closest open-source analogue of what we need to build: a `Kernel` object that owns a `LazyOp` tree, decides tile sizes, emits backend-specific source. We borrow its `Kernel.shape_dims` + `Kernel.upcast_args` design directly.
- **TVM AutoScheduler / Ansor** ([arXiv:2006.06762](https://arxiv.org/abs/2006.06762)) is the reference cost-model design. Ansor uses XGBoost over hand-crafted features; we use a simpler hand-tuned heuristic for v0 (§Architecture 3.2), but reserve a seam for a learned model later.
- **Flash Attention v1/v2** ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135); [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)) is the algorithmic template for tiled attention with online softmax.
- **PyTorch's `torch.compile` + Inductor** does producer-consumer fusion on CUDA via Triton ([TorchInductor design](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)). Our target is "Inductor for WGSL."

---

## User Stories

**U1 — Transformer block, single dispatch.** A student trains a 6-layer GPT (hidden=512, heads=8, seq=256, batch=8) on a craftingattention lab. After this PRD, each block's forward executes in **one** WGSL dispatch covering QKV projection, attention, output projection, residual, layernorm, FFN, residual, layernorm. Wall-clock per training step on an M1 MacBook Air drops from ~80 ms (post-PRD-006) to ~22 ms — within the 25% of native PyTorch target.

**U2 — Memory-aware Flash Attention.** A user runs masked self-attention at sequence length 1024 on a device with 16 KB workgroup memory. The tiled attention kernel uses (BR=64, BC=64) Q/K tiles, fits within the limit, and runs without materialising the (1024×1024) scores matrix anywhere in DRAM. Peak GPU memory drops from O(seq²) to O(seq).

**U3 — Forward+backward joint fusion.** A training step on a small MLP executes forward and backward in a single megakernel — activations that backward needs are stashed in workgroup memory, never written to DRAM. The user sees a 1.5× speedup over PRD-007's separated forward/backward dispatch.

**U4 — Graceful fallback.** A student opens the same lab on an older laptop reporting `maxComputeWorkgroupStorageSize = 16384`. The cost model rejects the megakernel for one of the blocks (FFN tile too large), the dispatcher emits PRD-006 single-op kernels for that block only, and training still runs — slower than U1 but correct.

**U5 — Autotuned tile sizes, cached.** First visit to a new model shape triggers a 2-second autotune sweep over `(BM, BN, BK) ∈ {16,32,64}³`. Best tile sizes per kernel are written to OPFS via PRD-008. Second visit reads the cached choice; no sweep.

---

## Goals and Non-Goals

### Goals

1. **Producer-consumer fusion across matmul boundaries**: any `matmul → bias-add → activation → matmul` chain with compatible tile shapes fuses into one kernel with workgroup-shared intermediate tiles.
2. **Cross-block fusion**: emit one megakernel for the canonical transformer block subgraph (`layernorm → QKV-proj → attention → output-proj → residual → layernorm → FFN-up → GeLU → FFN-down → residual`), guarded by a cost model that respects workgroup memory limits.
3. **Flash-Attention-v2-style tiled attention**: a fused `scaled_dot_product_attention(Q, K, V, mask)` megakernel implementing online softmax in workgroup memory; O(seq) DRAM usage; correct under causal and arbitrary attention masks.
4. **Forward+backward joint fusion** for training: when a backward consumer of a forward activation fits in the same kernel's workgroup memory window, the activation is *not* written to DRAM; it is recomputed or kept in registers/workgroup memory.
5. **Cost model**: a hand-tuned heuristic combining (a) bytes-of-DRAM-traffic-saved, (b) workgroup memory required, (c) register pressure approximated as `var_count × dtype_size`, (d) `maxComputeWorkgroupStorageSize` constraint. Decides whether to fuse each candidate group.
6. **Tile-size autotuning**: sweep `(BM, BN, BK) ∈ {16, 32, 64} × {16, 32, 64} × {8, 16, 32}` per shape (27 candidates); pick best by measured wall-clock; cache the winner in OPFS keyed by `(shape, dtype, adapter_info)`.
7. **Codegen safety**: emit a static assertion on the workgroup memory size at codegen time; emit a runtime check on dispatch parameters; **fall back to PRD-006 kernels** if any safety check fails.
8. **Numerical tolerance**: forward+backward matches PRD-007's symbolic-backward reference within **1e-3** (relaxed from 1e-4 because tiled accumulation reorders FP32 additions).
9. **Performance target**: end-to-end transformer training step within 25% of native PyTorch on M2-class hardware on a (B=4, H=12, hidden=768, seq=128) GPT-style block.
10. **Determinism**: autotuner results cached; given the same hardware and same shape, the same kernel runs every time. Bitwise reproducibility across runs on the same machine.

### Non-Goals

1. **WebGPU subgroup ops** ([Chrome subgroup blog](https://developer.chrome.com/blog/new-in-webgpu-128)) for warp-level primitives. They land in Chrome 144+ but are not yet portable; treated as a future optimization (PRD-015).
2. **fp16 storage / mixed precision in megakernels**. fp16 lands in PRD-010 (real autocast). PRD-012's megakernels are fp32 throughout. Mixed-precision megakernels are a follow-up after PRD-010.
3. **Tensor-core / WMMA acceleration**. WebGPU does not expose tensor cores. Out of scope until either WGSL adds WMMA intrinsics or WebNN backend (PRD-011) is mature.
4. **Inter-block fusion across module boundaries**. We fuse *within* a transformer block, not across two consecutive transformer blocks. The block's output must materialise to DRAM at the boundary (it crosses a residual-stream batch sync that exceeds workgroup memory).
5. **Learned cost model**. v0 ships a hand-tuned heuristic. Replacing it with an XGBoost-style learned model (Ansor-style) is PRD-019.
6. **Dynamic-shape megakernels**. v0 specializes by shape signature; recompiles when shapes change. JAX-style "shape polymorphism" is out of scope.
7. **Megakernel codegen for convolutional blocks**. The cost model is shaped for transformer-style ops; conv-style megakernels (im2col + matmul + bias + activation + norm + pool fusion) are a separate PRD (PRD-013) once we have benchmark data on whether they matter for the surveyed curricula.
8. **CPU/WASM fallback megakernels**. PRD-006's single-op WASM path remains the floor. Megakernels are WGSL-only.

---

## Architecture

The pipeline extends PRD-006's `fuse(graph) → graph` pass into a multi-stage compiler. Each stage produces a graph or a kernel plan; later stages depend strictly on earlier ones.

```
UOp graph (forward + backward, from PRD-005/006/007)
   │
   ▼
1. Fusion grouper          ── PRD-012 stage A
   │  bottom-up DAG walk assigns each node to a fusion group
   ▼
2. Cost model                ── PRD-012 stage B
   │  per group: compute bytes-saved, workgroup memory, register estimate
   │  reject groups that exceed maxComputeWorkgroupStorageSize
   ▼
3. Tile-size selector        ── PRD-012 stage C
   │  for matmul-containing groups: pick (BM, BN, BK) from OPFS cache or autotune
   ▼
4. WGSL megakernel codegen   ── PRD-012 stage D
   │  emit one @compute entry-point per accepted group
   ▼
5. Pipeline cache + dispatch ── PRD-008 + PRD-006 reused
```

### 1. Fusion Grouper (Stage A)

**File**: `packages/browsergrad-jit/src/python/fusion/grouper.py`

The grouper walks the IR DAG bottom-up (reverse topological order). Each node is assigned a `group_id`. A node joins its consumer's group iff:

1. All its consumers are in the same group, or it has exactly one consumer.
2. Joining does not introduce a cycle (the DAG case is handled naturally; the constraint matters when multiple groups would merge).
3. The combined group's predicted workgroup memory (see Stage B) stays within `maxComputeWorkgroupStorageSize`.
4. The op is in the **fusible set**: every op already in PRD-006's elementwise/reduce families, plus `MATMUL`, `SOFTMAX_FUSED` (from PRD-006), `LAYERNORM_FUSED`, `GELU`, `SILU`, and the new `ATTENTION_TILED`.

This is the classic "Kennedy-McKinley loop fusion" formulation adapted to a DAG. Pseudocode:

```python
def group_dag(graph: UOpGraph, limits: GpuLimits) -> dict[UOp, int]:
    group_id = {}
    next_id = 0
    for node in reverse_topo(graph):
        # Try to inherit the group of the unique-or-compatible consumer.
        consumer_groups = {group_id[c] for c in graph.consumers(node)}
        if len(consumer_groups) == 1 and node.op in FUSIBLE_OPS:
            candidate = consumer_groups.pop()
            if cost_model.fits(group_members(candidate) + [node], limits):
                group_id[node] = candidate
                continue
        # Otherwise start a new group rooted here.
        group_id[node] = next_id
        next_id += 1
    return group_id
```

**Pattern templates.** On top of the generic grouper, we run **pattern-specific recognisers** that override grouping for known shapes:

- `AttentionPattern`: matches `(matmul(Q, K.T) → mul(scale) → add(mask) → softmax → matmul(V))` and tags the group as `ATTENTION_TILED`. Triggers the Flash-Attention codegen in Stage D.
- `TransformerBlockPattern`: matches the canonical sequence `(layernorm → linear → attention → linear → add(residual) → layernorm → linear → gelu → linear → add(residual))` and tags as `TRANSFORMER_BLOCK`. Triggers the cross-block codegen.
- `MLPLayerPattern`: matches `(linear → activation → linear)` and tags as `MLP_BLOCK`. Triggers producer-consumer fusion with shared workgroup tile.

The pattern recognisers run *before* the generic grouper. A node claimed by a pattern is locked to its template group; the generic grouper sees those locked groups as opaque and routes the rest.

### 2. Cost Model (Stage B)

**File**: `packages/browsergrad-jit/src/python/fusion/cost_model.py`

For each candidate group `G` the cost model computes:

| Metric | Formula |
|---|---|
| `bytes_saved` | Sum over edges internal to `G` of (tensor_size_bytes); these reads/writes vanish when fused. |
| `workgroup_bytes` | Sum over tiles required by ops in `G`: a matmul with tile `(BM, BK)` × `(BK, BN)` needs `(BM*BK + BK*BN + BM*BN) * 4` bytes; an attention tile needs `(BR*D + BC*D + BR*BC) * 4`. |
| `register_estimate` | `sum(var_count_per_op) × dtype_size`. Var counts are baked into the codegen templates per op. Mirrors [tinygrad's heuristic](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/codegen/kernel.py). |
| `compute_intensity` | `total_flops / total_bytes_read_after_fusion`. Used to detect bandwidth-bound vs compute-bound groups. |

**Accept rule** (heuristic constants tuned in week 9; placeholders shown):

```python
def accept_group(g: FusionGroup, limits: GpuLimits) -> bool:
    if g.workgroup_bytes > limits.max_workgroup_storage:    # hard limit
        return False
    if g.register_estimate > 256 * 4:                       # ~256 fp32 per thread soft cap
        return False
    if g.bytes_saved < 4096:                                # don't bother for tiny groups
        return False
    if g.size == 1:                                         # single-op group
        return False                                        # already handled by PRD-006
    return True
```

`limits.max_workgroup_storage` is read from `device.limits.maxComputeWorkgroupStorageSize` at runtime; per [GPUSupportedLimits](https://gpuweb.github.io/gpuweb/#gpusupportedlimits) the default minimum is **16384 bytes**. We assume 16 KB unless the adapter reports higher.

WGSL does not directly expose registers, so `register_estimate` is a proxy: the count of `var<private>` and function-scope `let`/`var` declarations the codegen will produce, times 4 bytes. We use the same heuristic tinygrad uses for its WebGPU backend, which empirically correlates with actual occupancy degradation on M-series GPUs.

### 3. Tile-Size Selector (Stage C)

**File**: `packages/browsergrad-jit/src/python/fusion/tile_search.py`

For groups containing matmul tiles, the tile selector chooses `(BM, BN, BK)`. The space is:

```python
BM_CANDIDATES = (16, 32, 64)
BN_CANDIDATES = (16, 32, 64)
BK_CANDIDATES = (8, 16, 32)
# Cartesian: 27 candidates per shape.
```

**Cache lookup first.** The cache key is `sha256(group_signature + adapter_info)` where `group_signature` is the tuple of shapes and the group's pattern tag. If hit, use the cached choice. (PRD-008 owns the OPFS layer; we just call its `getCached` / `store` API.)

**Cold path**: for each candidate, generate the kernel, compile, and time 5 dispatches on representative input. Pick the median-wall-clock winner. Total cost: ~27 × 5 × (compile + dispatch) ≈ 1.5–2.5 seconds the first time a new shape is seen. Per PRD-008 §3, OPFS persists this for second visits.

**Pruning.** Reject candidates whose workgroup memory exceeds the limit before timing. Typically only ~12–18 of the 27 candidates survive on a 16 KB device.

For attention groups, the tile dimensions are `(BR, BC)` (rows/columns of the Q × K^T tile). The space is `BR ∈ {32, 64, 128}, BC ∈ {32, 64, 128}` — 9 candidates, same protocol.

### 4. WGSL Megakernel Codegen (Stage D)

**File**: `packages/browsergrad-jit/src/python/fusion/codegen_mega.py`

The codegen emits **one WGSL `@compute` entry-point per group**. The body is a tiled iteration over output coordinates; intermediate tensors live in `var<workgroup>` arrays.

#### 4a. Producer-consumer fused MLP (matmul → bias → activation → matmul)

For `y = act(x @ W1 + b1) @ W2 + b2` where `x: (M, K1), W1: (K1, K2), W2: (K2, K3)`:

```wgsl
// Tile sizes: (BM, BK2, BK3) chosen by tile selector.
@group(0) @binding(0) var<storage, read>       x  : array<f32>;
@group(0) @binding(1) var<storage, read>       W1 : array<f32>;
@group(0) @binding(2) var<storage, read>       b1 : array<f32>;
@group(0) @binding(3) var<storage, read>       W2 : array<f32>;
@group(0) @binding(4) var<storage, read>       b2 : array<f32>;
@group(0) @binding(5) var<storage, read_write> y  : array<f32>;

// Workgroup-shared tile of the intermediate activation. This is the
// tensor that, in the unfused path, would round-trip through DRAM.
var<workgroup> hidden_tile : array<f32, BM_VAL * BK2_VAL>;

@compute @workgroup_size(BK2_VAL, BM_VAL / 1, 1)
fn fused_mlp(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(local_invocation_id)  lid : vec3<u32>,
    @builtin(workgroup_id)         wid : vec3<u32>,
) {
    // --- Stage 1: x @ W1 + b1, output kept in workgroup memory ---
    let m = wid.y * BM_VAL + lid.y;
    let k2 = lid.x;
    var acc : f32 = 0.0;
    for (var kk : u32 = 0u; kk < K1_VAL; kk = kk + 1u) {
        acc = acc + x[m * K1_VAL + kk] * W1[kk * K2_VAL + k2];
    }
    acc = acc + b1[k2];
    hidden_tile[lid.y * BK2_VAL + lid.x] = ACTIVATION(acc);  // GELU/RELU inlined
    workgroupBarrier();

    // --- Stage 2: hidden_tile @ W2 + b2, write to DRAM ---
    // Each thread now computes one column of y[m, :K3] using the tile.
    // ...standard tiled matmul, reading hidden_tile not DRAM...
}
```

The critical move: `hidden_tile` is never written to DRAM. The unfused path would write `(M, K2)` floats to global memory and immediately read them back; the fused path keeps them in `var<workgroup>`. For `M=BM=32, K2=64`, that is 8 KB of workgroup memory — within the 16 KB budget.

Verbose codegen is template-driven; constants like `BM_VAL` are spliced in as numeric literals at emission time, not WGSL pipeline-override constants (those are an optimization for PRD-015).

#### 4b. Flash-Attention-v2-style tiled attention

Algorithm follows [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) §3 (Flash Attention v2's tile traversal order):

- **Outer loop over Q tiles** (rows `BR`). Each workgroup owns one Q tile.
- **Inner loop over K/V tiles** (cols `BC`). The K and V tiles are streamed through workgroup memory.
- **Online softmax**: maintain running row-max `m_i` and running row-sum `l_i` in workgroup memory; rescale the partial output `O_i` when a new max is encountered ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135) §3.1).

Workgroup memory layout for `D=64, BR=64, BC=64`:

| Buffer | Bytes | Purpose |
|---|---|---|
| `Q_tile` | `BR*D*4` = 16 384 | Loaded once per workgroup, reused across inner loop |
| `K_tile` | `BC*D*4` = 16 384 | Streamed |
| `V_tile` | `BC*D*4` = 16 384 | Streamed |
| `S_tile` | `BR*BC*4` = 16 384 | Scores tile |
| `m_i` | `BR*4` = 256 | Running max per row |
| `l_i` | `BR*4` = 256 | Running sum per row |

Total: ~64 KB — **exceeds the 16 KB default workgroup limit**. We therefore default to `(BR=32, BC=32)` for devices reporting only the WebGPU minimum, giving us 4 × 4 KB + scores 4 KB + (m, l) = ~21 KB still too high. The accept rule rejects this config on minimum devices; the cost model then either (a) double-buffers — only `K_tile` *or* `V_tile` resident at a time — or (b) falls back to PRD-006 per-row softmax. Both paths are exercised in tests.

On devices reporting higher `maxComputeWorkgroupStorageSize` (Apple M-series typically report 32 KB; modern desktop GPUs 48 KB+), we use `(BR=64, BC=64, D=64)` with full residency.

The emitted WGSL is structurally:

```wgsl
@compute @workgroup_size(BR_VAL, 1, 1)
fn flash_attention(/* bindings */) {
    // Load Q_tile from DRAM into workgroup memory.
    // Initialise m_i = -inf, l_i = 0, O_i = 0.
    for (var j_block : u32 = 0u; j_block < N_KV_BLOCKS; j_block = j_block + 1u) {
        // Load K_tile, V_tile from DRAM.
        // Compute S_tile = Q_tile @ K_tile^T / sqrt(D).
        // Apply mask (causal / arbitrary).
        // m_new = max(m_i, rowmax(S_tile))
        // P_tile = exp(S_tile - m_new)
        // l_new = exp(m_i - m_new) * l_i + rowsum(P_tile)
        // O_i = (l_i / l_new) * exp(m_i - m_new) * O_i + (1 / l_new) * P_tile @ V_tile
        // m_i, l_i = m_new, l_new
        workgroupBarrier();
    }
    // Write O_i to DRAM.
}
```

The math is the v2 update rule: rescaling happens with `1 / l_new` once per inner iteration, not per element, saving ~30% of FLOPs vs v1.

#### 4c. Cross-block megakernel (transformer block in one dispatch)

This is the highest-impact codegen target and the trickiest. The cross-block megakernel covers:

```
x_in
  ├─ layernorm₁ ─ qkv_proj ─ flash_attention ─ out_proj ─┐
  └─────────────────────────────────────────────── + ────┤  (residual 1)
                                                         ↓
                                                         h_mid
                                                          │
                                                          ├─ layernorm₂ ─ ffn_up ─ gelu ─ ffn_down ─┐
                                                          └─────────────────────────────────────────┤  (residual 2)
                                                                                                    ↓
                                                                                                    x_out
```

Constraints that make this hard:
- The residual connections require holding `x_in` and `h_mid` somewhere across the block. Workgroup memory is too small for the full hidden state at any realistic seq length.
- The attention sub-block has its own tile structure that does not match the FFN's natural tile structure.

Resolution: **the cross-block kernel processes one batch element at a time per workgroup**, with `x_in` and `h_mid` materialised in **DRAM** (not eliminated) but with workgroup-shared `Q_tile`, `S_tile`, `gelu_tile`, etc. The "megakernel" eliminates the intermediate Q, K, V, scores, attention output, FFN-hidden tensors — *not* the block input/output. Per-block bytes saved drop from 12 tensor materializations to 2.

When the block does not fit in workgroup memory (cost model rejects), we degrade gracefully to two megakernels: `(layernorm + attention + residual)` and `(layernorm + FFN + residual)`, each fused but block-internal.

### 5. Forward+Backward Joint Fusion

**File**: `packages/browsergrad-jit/src/python/fusion/joint_fb.py`

Given PRD-007's unified forward+backward IR, the grouper extends its candidate-group construction: a forward UOp and its symmetric backward UOp(s) may join the same group iff:

1. The activation needed by backward fits in workgroup memory.
2. The forward output is consumed only by the immediate backward (no other forward op downstream needs it).
3. The combined kernel's `workgroup_bytes` and `register_estimate` still pass the accept rule.

The classic case is `relu`: forward `y = max(x, 0)`, backward `dx = dy * (x > 0)`. Joint fusion holds `(x > 0)` as a 1-bit mask in workgroup memory (1 byte per element via WGSL `bool` packing) instead of writing the full forward output to DRAM. Saves 4× bytes on the activation.

For tiled matmul-bias-activation chains, joint fusion stashes the pre-activation tile in workgroup memory; the backward kernel's first stage reads it from there. Saves the entire activation tensor's worth of DRAM traffic per training step.

When backward joint fusion does not fit, the standard PRD-007 backward kernel runs separately. Joint fusion is purely additive.

### Layer integration

The fusion pipeline replaces PRD-006's single `fuse()` call:

```python
def fuse_v2(graph: UOpGraph, device_limits: GpuLimits, cache: OpfsPipelineCache) -> KernelPlan:
    pattern_groups = run_pattern_recognisers(graph)         # locked groups
    groups         = group_dag(graph, device_limits, locked=pattern_groups)
    groups         = cost_model.filter(groups, device_limits)
    for g in groups:
        if g.has_matmul or g.is_attention:
            g.tile = tile_search.pick(g, cache)             # OPFS-cached
    return codegen_mega.emit(groups)
```

PRD-006's elementwise/softmax/layernorm passes still run but only on groups the v2 grouper assigned the trivial path; they handle the residual one-op cases that don't justify a megakernel.

---

## API Surface

The user-facing API is **internal**. Megakernels are emitted automatically when shape signatures match. No code change is required.

The one surface change: `torch.compile()` becomes a real (no-op) function so PyTorch code that wraps a module compiles cleanly:

```python
import browsergrad_jit as torch

@torch.compile  # accepted; tracing is already automatic, so this is a hint
def step(x, y):
    return F.cross_entropy(model(x), y)
```

`torch.compile(fn)` returns `fn` unchanged but tags it with `_bg_compile_hint=True`. Future optimisers may use the hint to pre-warm autotune; v0 ignores the hint.

Debug surfaces:

```python
import browsergrad_jit as bg

# After a forward pass:
report = bg.jit.debug_fused_kernels()
# Each entry now has additional fields:
#   tile_dims:   (BM, BN, BK) or None for non-tiled
#   workgroup_bytes: int
#   register_estimate: int
#   fallback_reason: Optional[str]  # e.g. "workgroup_storage_exceeded"

# Force-disable megakernels (keeps PRD-006 fusion):
bg.jit.set_fusion_level("basic")    # basic | producer_consumer | mega

# Inspect the autotune cache:
bg.jit.autotune_cache_stats()  # {"entries": 42, "hits_this_session": 17, ...}
```

Environment variables:

| Var | Effect |
|---|---|
| `BG_DISABLE_MEGAKERNEL=1` | Reverts to PRD-006 fusion only. |
| `BG_AUTOTUNE_TIMEOUT_MS=2000` | Bound the cold autotune sweep. Default 2000 ms. |
| `BG_FUSION_LOG=1` | Print every grouper decision to stderr (debugging). |

---

## Implementation Plan

The PRD spans **10 weeks**. Each week ends with a green conformance suite — landing partial work behind feature flags is mandatory.

### Week 1 — Fusion grouper + cost model skeleton

- Implement `fusion/grouper.py` reverse-topo DAG walk; produces `FusionGroup` objects but does not yet emit code (group results are logged only).
- Implement `fusion/cost_model.py` with the four metrics from §Architecture 2. Hand-code the var counts for every PRD-005 opcode.
- Wire `device.limits.maxComputeWorkgroupStorageSize` into runtime context. Add `GpuLimits` dataclass to `browsergrad-runtime`.
- Unit tests: 10 synthetic DAGs assert correct grouping outcomes.

### Week 2 — Producer-consumer fused MLP codegen

- Implement `codegen_mega.py` emit path for `MLP_BLOCK` pattern (linear → activation → linear).
- Emit WGSL template from §Architecture 4a. Hard-code tile sizes `(BM=32, BN=32, BK=16)` for week 2; autotune comes week 5.
- Conformance: 2-layer MLP forward matches PRD-006 path within 1e-4.
- Microbenchmark: `(x @ W1 + b1).relu() @ W2 + b2` at (B=64, K1=512, K2=512, K3=512) — measure ≥1.7× wall-clock improvement over PRD-006.

### Week 3 — Pattern recognisers + cross-block grouper

- Implement `AttentionPattern`, `MLPLayerPattern`, `TransformerBlockPattern` recognisers in `fusion/patterns.py`.
- Wire pattern outputs into the grouper as locked groups.
- Add `bg.jit.debug_fused_kernels()` extension fields (`tile_dims`, `workgroup_bytes`, etc.).
- Integration test: a 2-layer transformer block's IR is recognised; `bg.jit.debug_fused_kernels()` shows one `TRANSFORMER_BLOCK` group.

### Week 4 — Flash-Attention-v2 kernel (algorithm)

- Implement `codegen_mega.emit_attention()` for `ATTENTION_TILED` groups. Follow [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) §3 exactly.
- Hard-code `(BR=32, BC=32, D=64)` for minimum-limit devices; allow `(BR=64, BC=64)` when limits permit.
- Conformance: fused attention output matches PRD-006 (per-row softmax) reference within 1e-3 across 12 shape combinations including masked variants.
- Numerical-stability test: input with mean 100, std 10 — verify no overflow.

### Week 5 — Tile-size autotuner + OPFS cache integration

- Implement `fusion/tile_search.py` with the 27-candidate sweep for matmul tiles, 9-candidate for attention tiles.
- Wire OPFS lookup via PRD-008's `getCachedPipeline` API extended to also cache `(group_signature → best_tile_dims)`.
- Add `BG_AUTOTUNE_TIMEOUT_MS` ceiling; if sweep exceeds budget, fall back to default tile sizes.
- Integration test: load a new model shape; assert autotune writes to OPFS; reload same shape; assert cache hit.

### Week 6 — Cross-block megakernel codegen

- Implement `codegen_mega.emit_transformer_block()` from §Architecture 4c.
- Materialise `x_in` and `h_mid` to DRAM at block boundaries (not eliminated); keep all inter-op intermediates workgroup-local.
- Conformance: a (B=4, H=12, hidden=768, seq=128) GPT-style block's forward matches PRD-007 reference within 1e-3.
- Wall-clock benchmark: target ≥4× speedup vs PRD-006 fusion alone on the same block.

### Week 7 — Forward+backward joint fusion

- Implement `fusion/joint_fb.py`: extend grouper to admit backward UOps; new accept rule for joint groups.
- Implement codegen for joint groups in `codegen_mega.py`: backward kernel reads activation tiles from workgroup memory written by forward; for elementwise activations, stash mask bits.
- Conformance: gradient match vs PRD-007 within 1e-3 on full training step of 4-layer MLP.
- Benchmark: ≥1.4× speedup on full forward+backward of a 2-layer MLP at (B=128, hidden=512).

### Week 8 — Cost-model tuning + fallback paths

- Sweep cost-model constants against a benchmark suite of 20 representative kernels; tune `register_estimate` cap and `bytes_saved` threshold.
- Implement explicit fallback emission: when accept rule rejects a group, the dispatcher must produce PRD-006 single-op kernels for that group's nodes. Verify no graph-completeness gaps.
- Add `fallback_reason` field to fusion decisions. Integration test on a contrived limit-exceeding shape: assert fallback fires and result is correct.

### Week 9 — Determinism + autotune hardening

- Replace any nondeterministic float reductions in the autotuner timing with median-of-5.
- Add hardware-fingerprint gating: cached tile choices invalidate when `(adapterInfo, browser-major-version)` changes.
- Add stress test: run autotune in a worker, kill mid-sweep; assert partial cache is consistent (atomic OPFS writes via PRD-008's `*.tmp + rename` pattern).
- Implement `BG_DISABLE_MEGAKERNEL=1` switch end-to-end; assert PRD-006 path stays green.

### Week 10 — End-to-end target validation + ship

- Full 234-integration-test suite + 5 PyTorch conformance fixtures green with megakernels enabled.
- End-to-end nanoGPT training step at (B=4, hidden=768, seq=128) on M1 MacBook Air: measure vs native PyTorch on the same machine. Acceptance: within 25%.
- Browser matrix: Chrome desktop, Chrome Android, Safari 26, Firefox 145. Document any platform-specific fallbacks.
- Write `docs/megakernel.md` covering pattern catalog, cost-model rationale, autotune protocol, fallback semantics.
- Publish `@unlocalhosted/browsergrad-jit@0.4.0` and `@unlocalhosted/browsergrad-kernels@0.4.0`.

---

## Acceptance Criteria

| # | Criterion | Measurement |
|---|---|---|
| AC1 | Producer-consumer fused MLP (linear→relu→linear) ≥1.7× faster than PRD-006 fusion | `mlp_megakernel.bench.ts` |
| AC2 | Transformer block forward at (B=4, H=12, hidden=768, seq=128) ≥4× faster than PRD-006 | `transformer_block.bench.ts` |
| AC3 | Flash-Attention-v2 megakernel matches per-row softmax reference within 1e-3 across 12 shapes including masked | `attention_conformance.test.ts` |
| AC4 | Attention peak DRAM usage scales as O(seq), not O(seq²), verified to seq=2048 | `attention_memory.bench.ts` |
| AC5 | Forward+backward joint fusion ≥1.4× speedup on 2-layer MLP training step | `joint_fb.bench.ts` |
| AC6 | Cost model rejects groups exceeding `maxComputeWorkgroupStorageSize`; fallback to PRD-006 produces correct results | `fallback_paths.test.ts` |
| AC7 | Autotune cache hit on second visit; cold sweep bounded by `BG_AUTOTUNE_TIMEOUT_MS` | `autotune_cache.test.ts` |
| AC8 | Determinism: same hardware + same shape → same selected kernel across 5 runs | `determinism.test.ts` |
| AC9 | All 234 integration tests pass with megakernels enabled within 1e-3 tolerance | full suite green |
| AC10 | All 234 integration tests pass with `BG_DISABLE_MEGAKERNEL=1` identical to PRD-006 result within 1e-6 | regression mode |
| AC11 | End-to-end GPT-style training step on M1: ≤1.25× wall-clock of native PyTorch on same machine | `vs_native_pytorch.bench.ts` |
| AC12 | `debug_fused_kernels()` reports tile dims + workgroup bytes + register estimate per group | `introspection.test.ts` |

---

## Test Strategy

### Unit (Vitest, no Pyodide)

- `grouper_unit.test.ts` — 25 synthetic DAGs with hand-verified group assignments.
- `cost_model_unit.test.ts` — for every PRD-005 opcode, assert correct `workgroup_bytes`, `register_estimate`, `bytes_saved`.
- `pattern_recognisers.test.ts` — positive and negative match cases for every pattern.
- `tile_search_unit.test.ts` — given a synthetic timing oracle, the selector picks the minimum.

### Integration (Pyodide-in-Node + WebGPU)

- `mlp_megakernel.test.ts` — fused MLP forward and backward match PRD-007 within 1e-3.
- `attention_conformance.test.ts` — masked and unmasked attention across 12 shapes; numerical-stability extreme input.
- `transformer_block.test.ts` — full block forward+backward.
- `joint_fb.test.ts` — gradient match across 5 representative subgraphs.
- `fallback_paths.test.ts` — contrived too-large shape exercises the fallback emitter.
- `autotune_cache.test.ts` — write/read OPFS cache; verify pinning, atomic writes.
- `determinism.test.ts` — same hardware + same shape produces identical kernel hash across 5 runs.
- `regression_basic.test.ts` — with `BG_DISABLE_MEGAKERNEL=1`, results identical to PRD-006 path within 1e-6.

### Benchmarks (CI-tracked, not blocking)

- `mlp_megakernel.bench.ts`, `transformer_block.bench.ts`, `attention_memory.bench.ts`, `joint_fb.bench.ts`, `vs_native_pytorch.bench.ts`. Numbers logged to GitHub Actions summary; regressions ≥10% open an alert issue.

### Browser matrix (Playwright)

- `browser/cross_browser_megakernel.spec.ts` — runs the transformer block on Chrome desktop, Chrome Android, Safari 26, Firefox 145. Per-browser tolerance documented.

---

## Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | 16 KB workgroup memory floor on minimum-limit devices forces fallback for most attention shapes — observed speedup short of target | High | High | Default `(BR=32, BC=32)` with double-buffering; profile early; accept partial speedup with documented degradation tier |
| R2 | Tiled FP32 accumulation reorders additions, breaking 1e-4 conformance from PRD-007 | High | Medium | Relax tolerance to 1e-3 for megakernel-fused groups; document in `tolerances.md`; verify training convergence not affected over 1000 steps |
| R3 | Autotuner cold sweep blocks UI thread on first visit (~2 s) | High | Medium | Run autotune in dedicated worker; show progress UI; cap with `BG_AUTOTUNE_TIMEOUT_MS`; fall back to default tiles on timeout |
| R4 | Cross-browser WGSL behavior differs on workgroup barrier semantics (Safari ≠ Chrome on Apple Silicon) | Medium | Medium | Per-browser conformance test; emit conservative `workgroupBarrier()` after every shared-memory write |
| R5 | Cost model heuristic mis-rejects profitable groups or accepts unprofitable ones, regressing wall-clock | High | Medium | Week-8 dedicated to constant tuning; debug log every decision; ship `BG_FUSION_LOG=1` for users to report bad decisions |
| R6 | Register pressure proxy (var count × dtype size) under-estimates real occupancy loss on Apple GPUs | Medium | Medium | Calibrate proxy against measured occupancy on M1/M2/M3 in week 9; document the calibration; recheck quarterly |
| R7 | Cross-block megakernel's WGSL exceeds shader complexity limits on some browsers (max instructions, max bindings) | Medium | High | Hard cap on group size (max 20 ops, max 16 bindings); split larger groups; document limits |
| R8 | Joint forward+backward fusion breaks `create_graph=True` double backward from PRD-007 | Medium | Medium | Disable joint fusion for groups with `requires_grad` chains beyond first order; covered by PRD-007 conformance tests |
| R9 | OPFS autotune cache corruption causes wrong tile selection on second visit, leading to silent perf regression | Low | High | Validate cached entry's `group_signature` matches current at load time; on mismatch, evict and re-tune; SHA-256 keys avoid collisions |
| R10 | Driver update on user's machine invalidates browser-internal pipeline cache; our OPFS still hits but compile is slow | Medium | Low | Include `browser_major_version` in cache key; one slow page after driver update, then warm |
| R11 | Pattern recogniser false positive on a layernorm-like custom op that isn't actually layernorm | Medium | Medium | Strict discriminant on full subgraph match (all 8 nodes for LN); contrived test cases for near-misses |
| R12 | 25% of native PyTorch target unattainable on minimum-limit hardware (sub-16 KB workgroup memory) | High | Low | Document tier: "M1/M2-class: within 25%; older mobile GPUs: within 2×"; matches PRD.md §3.6 expectations |

---

## Open Questions

1. **WGSL pipeline-override constants for tile sizes.** WebGPU supports `override` constants ([WebGPU §10.2.1](https://www.w3.org/TR/webgpu/#dom-gpuprogrammablestage-constants)) which would let one shader module serve all 27 tile-size variants. v0 uses textual splicing because override constants do not specialize loop trip counts at compile time, which limits the optimizer. Re-evaluate after Chrome 145 (override-constant loop-unroll improvements landed).

2. **Should the cost model be a learned XGBoost regressor (Ansor-style) from day 1?** Resolution: no — bootstrap with hand heuristic to gather training data, learn in PRD-019 once we have ≥1000 (shape, kernel, time) tuples in OPFS.

3. **Megakernel for the optimizer step (Adam over all parameters)?** Adam is a dense elementwise computation that fuses trivially in principle, but parameter buffers are distributed (one per module). Decision: out of scope for v0; revisit if profiling shows optimizer step dominates after PRD-012.

4. **Recompile vs cache eviction on minor browser updates.** Chrome ships pipeline cache invalidations on every Canary; should our OPFS cache key include the patch version? Resolution: include only major; tolerate patch-version recompile pain (sub-second per kernel) since the autotune cache (which is the slow part) is patch-stable.

5. **Pattern recogniser priority when patterns overlap.** A subgraph might match both `MLPLayerPattern` and (with its neighbour) `TransformerBlockPattern`. Resolution: try larger patterns first; on rejection by cost model, decay to smaller patterns.

6. **Should `torch.compile(model, mode="max-autotune")` expand the autotune budget beyond default?** Resolution: yes — the hint maps to `BG_AUTOTUNE_TIMEOUT_MS=10000` for that compile call. v0 ignores; revisit in PRD-019.

---

## References

1. **Flash Attention v1** — Dao, T. et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) (2022). Online-softmax algorithm.

2. **Flash Attention v2** — Dao, T., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," [arXiv:2307.08691](https://arxiv.org/abs/2307.08691) (2023). Improved tile traversal order used in §Architecture 4b.

3. **Triton** — Tillet, P. et al., "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations," [Triton design](https://openai.com/research/triton); [OpenAI Triton GitHub](https://github.com/openai/triton). Tile-level fusion design template.

4. **tinygrad scheduler** — [tinygrad/codegen/kernel.py](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/codegen/kernel.py). Direct source for the `Kernel.shape_dims` / `upcast_args` patterns and the var-count register-pressure heuristic.

5. **TVM Ansor** — Zheng, L. et al., "Ansor: Generating High-Performance Tensor Programs for Deep Learning," [arXiv:2006.06762](https://arxiv.org/abs/2006.06762). Cost-model-driven tile-size search; reserved seam for PRD-019.

6. **WebGPU spec — GPUSupportedLimits** — [W3C WebGPU §3.6.2](https://gpuweb.github.io/gpuweb/#gpusupportedlimits). Defines `maxComputeWorkgroupStorageSize` with default 16384 bytes.

7. **WebGPU spec — Workgroup memory** — [W3C WebGPU §10.4 (workgroup address space)](https://www.w3.org/TR/webgpu/). Barriers and shared array semantics.

8. **WGSL matmul performance on M2** — [nuss-and-bolts.com WGSL matmul optimization](https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel). Quantifies the ~17% peak FP32 ceiling that Flash-Attention-style fusion targets to exceed (>50% peak).

9. **Chrome WebGPU dispatch + pipeline caching** — [Chrome WebGPU recipes](https://developer.chrome.com/docs/web-platform/webgpu); [arXiv:2604.02344](https://arxiv.org/abs/2604.02344) for the ~95 μs dispatch-floor measurement that motivates PRD-006; bandwidth-bound regime (this PRD) takes over above ~1 MB tensor sizes.

10. **PyTorch TorchInductor** — [TorchInductor design doc](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747). Reference for producer-consumer fusion at the IR level.

11. **VISION.md §4 Layer 3, §5 (ResNet18 walkthrough), §3.6 (1/4 of native ceiling)** — the architectural and performance targets this PRD must hit.

12. **PRD-005 (IR)**, **PRD-006 (basic fusion)**, **PRD-007 (symbolic backward)**, **PRD-008 (OPFS cache)** — direct dependencies; this PRD extends each.

13. **safetensors format** — [HuggingFace safetensors](https://github.com/huggingface/safetensors). Indirectly relevant: model weights loaded via PRD-008 are pinned during autotune to prevent cache eviction mid-sweep.

14. **WebGPU shader-f16 extension** — [W3C WebGPU §6.2 shader-f16](https://www.w3.org/TR/webgpu/#shader-f16). Not used in PRD-012 (fp32 only); referenced for the PRD-010 mixed-precision follow-up.
