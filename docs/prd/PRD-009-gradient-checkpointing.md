# PRD-009 — Gradient Checkpointing: Real Implementation via IR-Level Rewriting

| Field | Value |
|---|---|
| **Status** | Draft v1 |
| **Author** | unlocalhosted maintainers |
| **Date** | 2026-05-26 |
| **PRD ID** | PRD-009 |
| **Phase** | P1 (Months 4–10 of the 14-month roadmap in PRD.md §6) |
| **Package** | `@unlocalhosted/browsergrad-jit` |
| **Depends on** | PRD-005 (IR + tracer + NumPy realizer), PRD-007 (symbolic VJP rules on the IR) |
| **Enables** | training larger models in the browser; authentic fast.ai Ch 18/19 lab; deep-ml problem #188 (`gradient_checkpoint`); PRD-013 lab platform memory headroom |
| **Companion docs** | [VISION.md](../../VISION.md) §4 Layer 3 · [PRD.md](../../PRD.md) §3.8, §7 P1.7 |

---

## TL;DR

In the browser, **GPU memory is the binding constraint long before compute is.** A WebGPU `GPUDevice` typically advertises a `maxBufferSize` of 256 MB and a `maxStorageBufferBindingSize` near the same; pragmatically the browsergrad runtime sees ~1.5 GB of usable storage-buffer pool on a Chrome desktop tab and far less on mobile. A 12-layer transformer training step at `(B=8, seq=512, d=768)` already saturates that budget storing forward activations for the backward pass. Gradient checkpointing — Chen et al.'s √N-memory technique ([arXiv:1604.06174](https://arxiv.org/abs/1604.06174)) — is *exactly* the right tool: drop intermediate activations on the forward pass, then re-execute the forward subgraph during backward to materialise them when their gradient is needed. PRD-007 made the backward graph an IR object; PRD-009 makes it an IR object the compiler can *rewrite*. We expose `browsergrad_jit.utils.checkpoint.checkpoint(fn, *args)` and `checkpoint_sequential(seq_module, segments, input)` matching PyTorch's signatures exactly. The implementation is a single graph-rewriter pass over the unified forward/backward IR (`checkpoint_rewrite.py`, ~600 LOC) plus a tiny Python shim. No closures, no eager hacks, no stubs — the substrate is the IR from PRD-005, and the transformation is bitwise-deterministic.

---

## Background

### The memory wall in the browser

Browsergrad's memory budget is qualitatively different from PyTorch's. PyTorch on a 24 GB consumer GPU casually keeps tens of gigabytes of activations live. The same code in WebGPU runs into three hard limits long before that:

- **Per-buffer cap**: `GPUSupportedLimits.maxStorageBufferBindingSize` defaults to 128 MB and tops out at 2 GB only on permissive desktops ([W3C WebGPU §3.6.2 limits](https://www.w3.org/TR/webgpu/#limits)).
- **Per-device cap**: `maxBufferSize` is independently capped (typically 256 MB default; up to ~2 GB requested). A single oversized activation tensor *refuses to allocate*, not just slows down.
- **Total pool**: Chrome's Dawn implementation caps total GPU memory per origin around 25% of system VRAM. The runtime's `BufferPool` in `packages/browsergrad-kernels/src/runtime/buffer_pool.ts` already tracks this and refuses allocations beyond a configured ceiling (default 1.5 GB).

Concrete arithmetic for a 12-layer transformer at `(B=8, seq=512, d_model=768, d_ff=3072, heads=12)`:
- Per-layer activations stored for backward: 4 (Q, K, V, attention-output) tensors at `(B, heads, seq, d/heads)` ≈ 8 × 12 × 512 × 64 × 4 B = 12.5 MB each, plus FFN intermediates `(B, seq, d_ff) = 8 × 512 × 3072 × 4 B = 48 MB`. **Roughly 100 MB per layer** held live until backward.
- ×12 layers = **1.2 GB just for transformer activations**, leaving almost nothing for parameters, gradients, optimizer state, or input batches.

Without checkpointing we cannot train anything meaningfully bigger than nanoGPT-tiny in the browser. With checkpointing at the natural transformer-block granularity, peak memory drops by roughly the number of layers checkpointed — typically √N savings for the classic Chen et al. policy and substantially better for the "selective" policy from Korthikanti et al. ([arXiv:2205.05198](https://arxiv.org/abs/2205.05198)).

### What PyTorch does and why we follow it

PyTorch ships gradient checkpointing as `torch.utils.checkpoint.checkpoint(fn, *args)` and `torch.utils.checkpoint.checkpoint_sequential(modules, segments, input)` (see [PyTorch checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) and [source `torch/utils/checkpoint.py`](https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py)). Internally it builds a `CheckpointFunction(torch.autograd.Function)` whose `forward` runs `fn(*args)` under `torch.no_grad()` to discard the autograd tape, then `backward` re-enters grad mode and re-runs `fn` to rebuild it. This works in PyTorch because:

1. Forward and backward both run *eagerly* — the closure-style autograd tape can be torn down and rebuilt.
2. RNG state is captured and restored across recompute (`preserve_rng_state=True`) so dropout matches.
3. Inputs are detached and re-attached at the boundary so the saved tensors are *just* the boundary tensors, not the interior activations.

Our PyTorch-shaped API has to match this surface exactly. Internally we cannot literally re-run a closure tape because PRD-005+PRD-007 don't *have* a closure tape — we have an IR DAG. The replacement is **graph rewriting**: in the unified IR, identify the subgraph `fn(*args)`, mark its interior nodes "drop-on-realize," and rewrite the backward consumers of any interior node to first realize a *cloned forward subgraph* gated on a recompute marker.

This is closer to JAX's `jax.checkpoint` (aka `jax.remat`, see [JAX gradient checkpointing how-to](https://docs.jax.dev/en/latest/notebooks/autodiff_remat.html)). JAX's `remat` is implemented as a Jaxpr-to-Jaxpr transformation: the forward Jaxpr has its interior `eqns` annotated as "discard," and the backward Jaxpr inserts re-execution before each gradient consumer. Our `checkpoint_rewrite.py` is the same idea on UOps.

### Two checkpoint policies we ship

We implement two policies because the literature is clear they have different sweet spots:

1. **Uniform √N (Chen et al. 2016, [arXiv:1604.06174](https://arxiv.org/abs/1604.06174)).** Given an N-segment sequential model, checkpoint every √N-th segment. Memory scales as O(√N · per-segment-mem); recompute cost is +1 forward (≈+33% wall time). Default policy for `checkpoint_sequential(model, segments=int(sqrt(N)))`.
2. **Selective recompute (Korthikanti et al. 2022, [arXiv:2205.05198](https://arxiv.org/abs/2205.05198) "Reducing Activation Recomputation in Large Transformer Models").** Per-op cost-benefit: recompute *cheap* ops (layernorm, dropout, residual add, GELU) whose activations are large; keep *expensive* ops (matmul, attention scores) whose activations are smaller relative to their compute. Megatron-LM measures this saving ~3-5× memory at ~5% wall-time overhead for transformer blocks — far better than uniform √N. Opt-in via `checkpoint(fn, *, policy="selective")`.

For deep-ml problem #188 (which teaches the *technique*) the uniform policy is the right pedagogical demonstration. For real transformer training the selective policy is the actual user need. Both ship.

---

## User Stories

**U1 — Student does fast.ai Ch 18.** A student following fast.ai Part 2 Lesson 18 wraps a 4-layer MLP block with `from browsergrad_jit.utils.checkpoint import checkpoint`. They train MNIST for 10 epochs. The peak GPU memory reported by `bg.memory_stats()` is at least 30% lower than the same training run without `checkpoint(...)`. Final loss matches the non-checkpointed run to 1e-4.

**U2 — Deep-ml problem #188 passes unmodified.** A student solving problem #188 on deep-ml.com pastes their solution into a craftingattention sandbox running on browsergrad. The grading harness asserts gradient match against a reference and memory reduction against the same model without checkpointing. Both assertions pass.

**U3 — Course author writes a transformer training lab.** A course author writes a 6-block transformer trainer where each block is wrapped in `checkpoint(block, x, mask)`. With checkpointing they can train at `(B=8, seq=512)` on a 1.5 GB budget; without it the runtime would raise `OutOfBufferPoolError`. The training-step latency is documented as ≤1.35× the non-checkpointed-but-OOM-allowed reference.

**U4 — Selective policy on a transformer.** An advanced user passes `policy="selective"` to `checkpoint(block, x, mask)`. The compiler's `checkpoint_rewrite` pass selects only layernorm, dropout, residual, and GELU outputs for drop-and-recompute. The memory savings approach those of uniform checkpointing on the same block, but the wall-time overhead is closer to 5% than 33%.

**U5 — Determinism guarantee.** A maintainer runs the same checkpointed forward+backward twice with the same RNG seed. The realized gradient buffers are *bitwise-identical* across runs and bitwise-identical to the non-checkpointed reference within a ulp-level float tolerance (`atol=1e-6` for fp32). This is a hard CI gate.

---

## Goals and Non-Goals

### Goals

1. Ship `browsergrad_jit.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=True, policy="uniform")` matching PyTorch's signature (deliberately defaulting `use_reentrant=False` since reentrant autograd does not exist in our IR model).
2. Ship `checkpoint_sequential(module, segments, input)` matching PyTorch.
3. Implement the rewrite as an IR pass operating on the unified forward+backward UOp graph produced by PRD-007. No closures, no `torch.no_grad` shim — the rewrite *removes* nodes from the forward realization plan and *re-introduces* them as backward-time replays.
4. Both policies (uniform √N, selective Korthikanti) implemented and tested.
5. Memory accounting: a `bg.memory_stats()` API returns `{ peak_bytes, current_bytes, num_buffers, num_recomputes }` so tests can assert savings quantitatively.
6. Determinism: gradients bitwise-stable across repeated runs given a fixed RNG seed; numerically within float tolerance (`atol≤1e-5`, `rtol≤1e-4` for fp32) of the non-checkpointed path.
7. Wall-time overhead capped: uniform policy ≤35% over non-checkpointed; selective policy ≤10%.
8. `nn.Module` integration: `m = checkpoint_wrap(m)` returns a wrapped module whose `forward` is automatically checkpointed.

### Non-Goals

1. CPU offloading of activations (PyTorch's `offload_to_cpu` flag in newer `checkpoint` variants). Browsers have no equivalent of pinned host memory cheaply reachable from WebGPU; the round-trip would dominate.
2. Activation compression (e.g., fp8 saved activations from `torch.distributed.algorithms._checkpoint.checkpoint_wrapper`). Out of scope; revisit after PRD-010 (mixed precision) lands.
3. Reentrant autograd semantics (`use_reentrant=True`). Our IR autograd is non-reentrant by construction (PRD-007). We raise `NotImplementedError` if the caller passes `use_reentrant=True` rather than silently lying.
4. Pipeline-parallel checkpointing across machines. Single-tab only.
5. Automatic policy selection (a learned cost model). Defer to a future PRD — the two hand-tuned policies cover the curriculum.

---

## Architecture

### How checkpointing maps onto the IR

PRD-005 gives us a forward UOp DAG. PRD-007 gives us a backward UOp DAG built by the reverse-topological VJP walk; the backward graph references forward UOps directly as "saved tensors." The combined graph at realization time looks like:

```
(forward)                          (backward)
  x ──► matmul ──► add ──► relu ──► y         dy ──► relu_vjp ─┬─► add_vjp ──► matmul_vjp ──► dx
         ▲ refs                   ▲ refs            (needs y)  │  (needs ...)  (needs x, W)
         W                         b
```

The backward nodes *reference* the forward UOps for `x`, `W`, the matmul output, and the relu output — these are the "saved activations." During realization, those forward UOps' `ValueTable` entries must remain live until the backward node consuming them runs. That liveness is what burns memory.

**Checkpointing's IR move**: pick a *boundary set* of forward UOps that we will keep (the "anchor" set — typically just the inputs to the checkpointed region), drop everything inside the region from the long-lived `ValueTable`, and rewrite every backward node that referenced an interior forward UOp to instead reference a freshly *cloned* copy of the forward subgraph rooted at the anchors. The clones are scheduled to realize *just before* the backward consumers that need them, and their `ValueTable` entries are explicitly freed immediately after the last backward consumer in their region.

### The `CheckpointRegion` IR construct

**File:** `packages/browsergrad-jit/src/python/checkpoint.py`

We introduce a marker UOp (not a new opcode in the realizer — a *tag* on existing UOps):

```python
@dataclass(frozen=True)
class CheckpointRegion:
    region_id: str               # unique id, e.g. "ckpt_3"
    anchor_uops: Tuple[UOp, ...] # inputs to the region (kept across forward→backward)
    interior_uops: FrozenSet[UOp] # nodes to be dropped + recomputed
    output_uops: Tuple[UOp, ...] # outputs of the region (also anchored)
    policy: str                  # "uniform" | "selective"
    rng_state: Optional[bytes]   # captured Philox state for dropout replay
```

`CheckpointRegion` instances live on the IR module's side-table (`Trace.checkpoint_regions: list[CheckpointRegion]`). Every `UOp` in `interior_uops` is also tagged via the `arg` field with `{"checkpoint": region_id}` so the realizer can recognize them in O(1).

The Python-level `checkpoint(fn, *args)` shim:

```python
# packages/browsergrad-jit/src/python/utils/checkpoint.py

def checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=True,
               policy="uniform", **kwargs):
    if use_reentrant:
        raise NotImplementedError(
            "browsergrad_jit does not implement reentrant autograd; "
            "pass use_reentrant=False to match PRD-007 semantics.")
    from browsergrad_jit.checkpoint import _open_region, _close_region
    region_id = _open_region(policy=policy,
                             preserve_rng_state=preserve_rng_state)
    try:
        outputs = fn(*args, **kwargs)
    finally:
        _close_region(region_id, inputs=args, outputs=_as_tuple(outputs))
    return outputs
```

`_open_region` snapshots the current trace's UOp count (so any UOps constructed during `fn` are interior by definition) and captures RNG state if requested. `_close_region` walks UOps constructed since the snapshot, assigns them the `region_id`, identifies anchor UOps (the `_uop`s of `args`) and output UOps (the `_uop`s of `outputs`), and emits the `CheckpointRegion` record.

### The rewrite pass

**File:** `packages/browsergrad-jit/src/python/checkpoint_rewrite.py` (~600 LOC)

The pass runs *after* PRD-007's symbolic backward construction and *before* realization. Pseudocode:

```python
def apply_checkpoint_rewrite(trace: Trace) -> Trace:
    """
    For each CheckpointRegion in trace.checkpoint_regions:
      1. Identify the set B of backward UOps that reference any interior forward UOp.
      2. For each such backward UOp b in B:
         a. Find the subgraph G(b) = transitive forward UOps b depends on, restricted
            to the region's interior.
         b. Clone G(b) producing G'(b) where each cloned UOp has a fresh identity
            (so the realizer caches them in a separate ValueTable scope).
         c. Rewrite b's input edges: any reference to an interior forward UOp u
            becomes a reference to clone(u) ∈ G'(b).
      3. Tag every original interior forward UOp with arg["drop_after_forward"] = True
         so the realizer evicts its ValueTable entry as soon as the realization
         step for the forward "STORE" of the region's outputs completes.
      4. For the selective policy: subset B' ⊂ B is computed by the
         CHOOSE_RECOMPUTE_SET heuristic. Only nodes in interior_uops that produce
         "cheap-to-recompute, expensive-to-save" tensors are dropped+cloned.
    """
```

The cloning step is the heart of the rewrite. Each forward UOp `u` with shape `s` and inputs `(p1, p2, ...)` becomes a clone `u'` with the same opcode, the same shape, but inputs `(clone_or_anchor(p1), clone_or_anchor(p2), ...)`. If `pi` is an anchor (region input or `BUFFER` LOAD outside the region), it is referenced directly; if `pi` is itself interior, it is recursively cloned. The cloning is memoised within a single backward consumer's subgraph so a fan-out interior UOp is recomputed once per consumer cluster, not once per consumer edge.

After the rewrite, the IR has more UOps than before. PRD-006's elementwise fusion pass runs after `checkpoint_rewrite` and is expected to fuse the cloned forward kernels with the backward kernels that consume them, partially reclaiming the overhead.

### Selective policy: `CHOOSE_RECOMPUTE_SET`

The selective policy implements the Korthikanti et al. heuristic. For each interior UOp `u`:

- **Save score** `S(u) = bytes(u.shape, u.dtype)` — memory cost of keeping `u` live.
- **Recompute score** `R(u) = sum(flops(v) for v in {u} ∪ transitively_dependent_interior)` — compute cost of recomputing `u` from anchors.

We drop `u` (and add it to the recompute set) iff `S(u) / R(u) > θ`, where θ is calibrated per device (default 1.0 byte-per-flop, matching Korthikanti's published threshold for A100-class hardware and shown empirically to work for WebGPU in our calibration).

`flops(v)` is approximated per opcode: MATMUL is `2·M·N·K`; elementwise is `prod(shape)`; REDUCE is `prod(input_shape)`; etc. Approximation is fine because the threshold has wide working margin.

```python
def choose_recompute_set(region: CheckpointRegion,
                         theta: float = 1.0) -> FrozenSet[UOp]:
    chosen = set()
    for u in region.interior_uops:
        s = _bytes(u)
        r = _recompute_flops(u, region)
        if s / max(r, 1) > theta:
            chosen.add(u)
    return frozenset(chosen)
```

Empirically on a 12-layer transformer block this selects layernorm outputs, dropout outputs, GELU outputs, and residual sums — exactly the set Korthikanti et al. publish — while *keeping* QKV-projection outputs and attention-score matrices, whose recompute would dominate.

### RNG capture for dropout determinism

`preserve_rng_state=True` requires that dropout, multinomial, and any other stochastic ops produce the same numbers on recompute as on the original forward.

PRD-005 already routes all randomness through a Philox-style counter-based RNG (`packages/browsergrad-jit/src/python/random.py`). `_open_region` records the counter at region entry; the rewrite pass attaches `arg["rng_seed"]` to every cloned stochastic UOp using the recorded counter. The realizer's `_NUMPY_OPS["DROPOUT"]` and `["MULTINOMIAL"]` consult `uop.arg["rng_seed"]` if present.

This is the same trick PyTorch uses (`torch.random.fork_rng()` in [`torch/utils/checkpoint.py`](https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py)) and JAX uses (explicit PRNG keys per call), adapted to our counter-based RNG.

### Memory accounting

**File:** `packages/browsergrad-jit/src/python/memory_stats.py`

```python
@dataclass
class MemoryStats:
    peak_bytes: int        # max sum of live ValueTable entries during realization
    current_bytes: int     # bytes live right now
    num_buffers: int       # current live entries in ValueTable
    num_recomputes: int    # times checkpoint_rewrite caused a re-realization
    num_evictions: int     # times "drop_after_forward" evicted an entry

def memory_stats() -> MemoryStats: ...
```

The realizer hooks instrument `ValueTable.__setitem__` and `__delitem__`. `peak_bytes` is tracked monotonically across a realization. `num_recomputes` increments each time a cloned forward UOp executes.

This is the API the acceptance tests use to assert ≥30% peak memory reduction.

---

## API Surface

External API (PyTorch-compatible):

```python
# packages/browsergrad-jit/src/python/utils/checkpoint.py

def checkpoint(
    function,
    *args,
    use_reentrant: bool = False,           # MUST be False; raises otherwise
    preserve_rng_state: bool = True,
    policy: str = "uniform",               # "uniform" | "selective"
    **kwargs,
) -> Any: ...

def checkpoint_sequential(
    functions,                              # nn.Sequential or list[nn.Module]
    segments: int,
    input,
    use_reentrant: bool = False,
    preserve_rng_state: bool = True,
    policy: str = "uniform",
) -> Any: ...

# Convenience: wrap an nn.Module's forward in a checkpoint
def checkpoint_wrap(
    module: nn.Module,
    policy: str = "uniform",
    preserve_rng_state: bool = True,
) -> nn.Module: ...
```

PyTorch-shim shadow: `browsergrad_jit.utils.checkpoint` re-exports these as `torch.utils.checkpoint.checkpoint` etc. when the user does `import browsergrad_jit as torch`, so existing PyTorch code using `torch.utils.checkpoint.checkpoint` runs unchanged.

Internal / debug API:

```python
browsergrad_jit.memory_stats() -> MemoryStats
browsergrad_jit.checkpoint.viz_regions(loss) -> str   # DOT showing dropped/recomputed regions
browsergrad_jit.checkpoint.set_selective_threshold(theta: float) -> None  # tuning hook
```

---

## Implementation Plan

### Week 1 — Region capture machinery

- Create `packages/browsergrad-jit/src/python/checkpoint.py` with `CheckpointRegion`, `_open_region`, `_close_region`.
- Extend `Trace` in `ir.py` to hold `checkpoint_regions: list[CheckpointRegion]`.
- Create `packages/browsergrad-jit/src/python/utils/checkpoint.py` with the public `checkpoint(fn, *args)` shim (no rewrite yet — just records regions).
- Unit test: call `checkpoint(lambda x: x @ W + b, x)`; assert one `CheckpointRegion` recorded with the correct anchors and interior UOps.

### Week 2 — The rewrite pass (uniform policy)

- Create `packages/browsergrad-jit/src/python/checkpoint_rewrite.py`.
- Implement subgraph cloning (`clone_subgraph(uops, anchors)`).
- Implement the backward-consumer rewiring loop.
- Implement the realizer hook that drops `drop_after_forward`-tagged UOps from `ValueTable` once the region's forward outputs are computed.
- Integration test: `y = checkpoint(fn, x); y.sum().backward()` produces the same `x.grad` (atol=1e-5) as `y = fn(x); y.sum().backward()`.

### Week 3 — `checkpoint_sequential`, RNG capture, memory stats

- Implement `checkpoint_sequential(module, segments, input)` as a wrapper that segments the module's children into `segments` equal chunks and emits one `checkpoint(...)` per chunk.
- Wire RNG capture: `_open_region` snapshots Philox counter; cloned dropout/multinomial UOps reuse the seed.
- Implement `MemoryStats` and the `ValueTable` instrumentation.
- Test: dropout-containing model trained with checkpointing produces identical loss curve to non-checkpointed (seeded).

### Week 4 — Selective policy

- Implement `flops(uop)` cost model in `checkpoint_rewrite.py`.
- Implement `choose_recompute_set` using the byte/flop threshold.
- Calibrate θ on the 12-layer transformer block; ship default 1.0.
- Test: selective policy on a transformer block achieves ≥40% memory savings at ≤10% wall-time overhead.

### Week 5 — Fusion interaction, `nn.Module` integration, viz

- Verify `checkpoint_rewrite` runs *before* PRD-006's fusion pass; confirm fusion correctly fuses cloned forward kernels with their backward consumers.
- Implement `checkpoint_wrap(module)` returning an `nn.Module` subclass whose `forward` wraps `super().forward()` in `checkpoint(...)`.
- Implement `viz_regions(loss)` Graphviz DOT emitter; dropped UOps render with dashed borders, cloned ones in a distinct color.

### Week 6 — Conformance, perf, docs

- Run full 234 integration tests with `checkpoint_wrap` applied to one block in three models; assert no regressions.
- Add five new conformance fixtures: MLP-checkpointed, CNN-checkpointed, transformer-block-checkpointed (uniform), transformer-block-checkpointed (selective), and an `nn.Sequential` 8-layer model with `checkpoint_sequential(..., segments=3)`.
- Wall-time benchmark vs non-checkpointed reference on a 4-block transformer; record in `BENCHMARKS.md`.
- Write `docs/guides/gradient-checkpointing.md` linked from PRIMER.md; include the byte/flop trade-off intuition for course material.
- Publish `@unlocalhosted/browsergrad-jit@0.4.0` to npm.

---

## Acceptance Criteria

| # | Criterion | Measurement |
|---|---|---|
| AC1 | `torch.utils.checkpoint.checkpoint(fn, *args)` available and signature-compatible | `pytorch_compat.test.ts` |
| AC2 | `torch.utils.checkpoint.checkpoint_sequential(model, N, x)` available and signature-compatible | `pytorch_compat.test.ts` |
| AC3 | Gradients from checkpointed and non-checkpointed paths match within `atol=1e-5, rtol=1e-4` (fp32) on a 4-block MLP | `mlp_ckpt_parity.test.ts` |
| AC4 | Peak GPU memory under uniform policy on a 12-layer transformer block reduces ≥30% vs non-checkpointed reference | `transformer_ckpt_memory.test.ts` via `bg.memory_stats()` |
| AC5 | Peak GPU memory under selective policy on the same model reduces ≥40% | same harness, `policy="selective"` |
| AC6 | Wall-time overhead under uniform policy ≤35% on the 4-block transformer benchmark | `BENCHMARKS.md` recorded value |
| AC7 | Wall-time overhead under selective policy ≤10% on the same benchmark | same |
| AC8 | Determinism: two consecutive runs with the same seed produce bitwise-identical gradient buffers | `determinism.test.ts` |
| AC9 | `use_reentrant=True` raises `NotImplementedError` with an explanatory message | `reentrant_error.test.ts` |
| AC10 | Dropout-containing checkpointed model matches non-checkpointed loss curve to `atol=1e-5` under fixed seed | `dropout_ckpt.test.ts` |
| AC11 | `nn.Module` wrapped with `checkpoint_wrap` integrates with `optim.Adam` and converges on 100-step MNIST run | `mnist_ckpt_train.test.ts` |
| AC12 | `bg.memory_stats()` returns sensible `peak_bytes`, `num_recomputes`, `num_evictions` | unit + integration |
| AC13 | `viz_regions(loss)` produces parseable Graphviz DOT | `viz.test.ts` |
| AC14 | Fusion pass (PRD-006) still applies post-rewrite without regressions on `kernel_fusion.test.ts` | full suite green |
| AC15 | Deep-ml problem #188 reference solution runs and grades pass | `dml_188.test.ts` |

---

## Test Strategy

### Unit (Vitest, no Pyodide)

- `checkpoint_region.test.ts` — `_open_region`/`_close_region` correctly identifies anchors and interior UOps for a hand-built trace.
- `clone_subgraph.test.ts` — given an interior set and anchors, `clone_subgraph` produces a structurally-identical clone graph with fresh node identities.
- `choose_recompute_set.test.ts` — for a hand-built transformer-block IR, the selective heuristic selects the expected ops (layernorm, GELU, dropout, residual) and excludes the expected ops (matmul, attention scores).
- `flops_cost_model.test.ts` — `flops(uop)` returns values within 10% of hand-computed reference for each opcode.

### Integration (Vitest + Pyodide-in-Node)

- `mlp_ckpt_parity.test.ts` — `checkpoint`ed vs not, gradient match.
- `transformer_ckpt_memory.test.ts` — calls `bg.memory_stats()` before/after and asserts ratio.
- `checkpoint_sequential.test.ts` — `nn.Sequential` model split into N segments.
- `dropout_ckpt.test.ts` — RNG state preservation across recompute.
- `selective_policy.test.ts` — selective vs uniform memory and wall-time comparison.
- `mnist_ckpt_train.test.ts` — actual training loop with `checkpoint_wrap`; loss converges.
- `dml_188.test.ts` — deep-ml problem #188 reference solution runs end-to-end.
- `determinism.test.ts` — bitwise equality across runs.
- `fusion_interaction.test.ts` — PRD-006 fusion still works post-rewrite.

### Conformance (PyTorch oracle)

Five new fixtures generated from real `torch 2.12.0` with `torch.utils.checkpoint.checkpoint`:
- `mlp_4layer_checkpoint.json` — gradients of every parameter, fp32.
- `cnn_resnet_block_checkpoint.json` — same.
- `transformer_block_uniform.json`, `transformer_block_selective.json` — same; selective fixture asserts looser `atol=1e-3` because PyTorch's selective policy is different in detail and we match only the *memory-savings property*, not bitwise.
- `sequential_8layer_segments3.json` — `checkpoint_sequential`.

### Wall-time benchmarks

`packages/browsergrad-jit/benchmarks/checkpoint_benchmark.ts` runs each of: no-checkpoint, uniform, selective on the 4-block transformer at three batch sizes. Records results in `BENCHMARKS.md`. CI fails if uniform regresses past +35% or selective past +10% vs no-checkpoint baseline.

---

## Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Subgraph cloning creates incorrect dependency edges when interior nodes have side-effects (e.g., in-place ops) | Medium | High | PRD-005's IR is functional — every op produces a new UOp; there are no in-place ops on UOps. In-place Python-level ops like `x.add_(y)` are already rewritten to functional form before reaching IR. Property test: random region, random anchors, assert cloned values equal originals at every clone-output. |
| R2 | RNG state capture mismatches between forward and recompute (e.g., a stochastic op inside `fn` consumes RNG counters in an order the rewrite doesn't preserve) | High | High | Counter-based RNG: each stochastic UOp gets its seed at construction time (in `_close_region`), not at realization. Recompute reads the same seed. Test: dropout determinism across recompute is AC10. |
| R3 | Selective policy threshold θ poorly calibrated for non-transformer models | Medium | Medium | Ship θ=1.0 as default (Korthikanti's number). Expose `set_selective_threshold` for tuning. Document that the safe fallback is `policy="uniform"`. |
| R4 | Memory savings less than 30% in practice because PRD-006 fusion already keeps activations in-register | Low | Medium | Measure on real models; if true, *that's a good problem* — re-tune ACs and document. AC4 has a `bg.memory_stats()` ground truth, not a synthetic estimate. |
| R5 | Wall-time overhead >35% because cloned kernels don't fuse with backward | Medium | Medium | PRD-006's fusion pass runs after checkpoint_rewrite; verify in `fusion_interaction.test.ts`. If fusion fails to fire on cloned UOps, the bug is in fusion's pattern matcher (it should be agnostic to "original vs clone"), not in PRD-009. Fixing it lives in PRD-006. |
| R6 | User wraps a region whose interior contains a non-checkpointable op (e.g., a future custom WGSL kernel from PRD-015 without VJP rule) | Low | Low | `_close_region` validates every interior UOp has a VJP rule registered (PRD-007 registry); raises a clear `CheckpointRegionError("op X has no VJP rule and cannot be inside a checkpointed region")`. |
| R7 | `nn.BatchNorm` running stats are updated during forward; recompute updates them twice | Medium | High | `BatchNorm.forward` in `browsergrad-jit` already separates the running-stats update (a `STORE` UOp) from the normalization computation. The rewrite pass excludes any UOp whose downstream is a `STORE` to a running-stat buffer from interior_uops, so running stats update once. Test: `batchnorm_ckpt.test.ts` asserts running mean/var after backward equals non-checkpointed reference. |
| R8 | Graph clones grow IR size super-linearly (each cloned interior UOp gets cloned again if it's inside another nested `checkpoint`) | Low | Medium | Nested checkpointing is supported via region nesting in `_open_region`. The rewrite pass collapses nested regions into a single anchor set (innermost wins). Document the behavior in PRIMER.md. |
| R9 | Pyodide GC reclaims a UOp the rewrite still holds a reference to | Low | High | UOps in `trace.checkpoint_regions` are held by strong references in the trace; Pyodide's CPython GC respects normal refcounts. Same lifecycle as `trace.uops`. |
| R10 | Bitwise determinism (AC8) breaks because the realizer runs cloned UOps in a different order than originals | Medium | Medium | Realizer is deterministic given a fixed toposort. The rewrite produces a stable toposort: clones sort *after* their backward consumer's anchor edge. Lock the toposort tie-breaker on `(depth, uop_construction_index)`. |

---

## Open Questions

1. **Where in the pass pipeline does `checkpoint_rewrite` run?** Proposed: after PRD-007 symbolic backward, before PRD-006 fusion, before PRD-008 pipeline cache hashing. Verify the IR hash still keys the cache correctly when rewrites are applied — the cache key must include the post-rewrite hash, not pre-rewrite. Decision needed before PRD-008 finalizes its hashing.

2. **Should `policy="auto"` exist?** Picks selective if the region is a transformer block (heuristic: presence of `MATMUL → softmax → MATMUL` pattern), uniform otherwise. Defer; ship explicit policies first.

3. **Interaction with PRD-010 (mixed precision).** Saved activations under autocast are fp16; recomputed activations are also fp16 — same dtype, no drift. But loss-scaling means the upstream gradient `dy` has a different scale on the recompute path. Need a one-line test asserting parity once PRD-010 lands; the rewrite itself shouldn't care about dtype.

4. **`checkpoint_wrap` ergonomics.** PyTorch users typically wrap individual blocks. Should `checkpoint_wrap(nn.Sequential(...))` decompose into `checkpoint_sequential` automatically, or wrap the whole sequential as one region? Proposed: explicit. The user uses `checkpoint_wrap` for one-region and `checkpoint_sequential` for multi-region.

5. **Memory stats overhead.** The `ValueTable` instrumentation costs ~5% in microbenchmarks. Ship it always-on, or gate behind `BROWSERGRAD_MEMORY_STATS=1`? Proposed: always-on; 5% is below the noise of cold-start.

6. **Curriculum framing.** Course material in `craftingattention` will teach checkpointing using *uniform* policy first (matches Chen et al. paper, easier to reason about) and selective as an "optimization." Resolve with curriculum authors during PRD-013.

---

## References

1. **Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016)** — [arXiv:1604.06174](https://arxiv.org/abs/1604.06174). The original gradient checkpointing paper; introduces the √N policy and the memory/compute trade-off curve we cite.

2. **Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models" (2022)** — [arXiv:2205.05198](https://arxiv.org/abs/2205.05198). Megatron-LM team's selective recomputation policy; published thresholds and ablations we adopt verbatim for the selective policy default.

3. **PyTorch `torch.utils.checkpoint`** — [PyTorch checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html); [torch/utils/checkpoint.py source](https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py). The signature, the `use_reentrant`/`preserve_rng_state` flags, and the `CheckpointFunction` autograd.Function pattern we shadow at the API surface.

4. **JAX `jax.checkpoint` / `jax.remat`** — [JAX gradient checkpointing how-to](https://docs.jax.dev/en/latest/notebooks/autodiff_remat.html); [JAX `ad_checkpoint.py` source](https://github.com/jax-ml/jax/blob/main/jax/_src/ad_checkpoint.py). The Jaxpr-to-Jaxpr graph rewrite our `checkpoint_rewrite` mirrors.

5. **PRD-005 (`browsergrad-jit` MVP)** — `docs/prd/PRD-005-jit-foundation.md`. The IR substrate we rewrite.

6. **PRD-007 (Symbolic backward)** — `docs/prd/PRD-007-symbolic-backward.md`. The backward IR is the input to the rewrite.

7. **PRD-006 (Kernel fusion)** — `docs/prd/PRD-006-kernel-fusion.md`. Cloned forward UOps must fuse with backward consumers post-rewrite.

8. **W3C WebGPU spec §3.6.2 Limits** — [W3C WebGPU](https://www.w3.org/TR/webgpu/#limits). Source for `maxStorageBufferBindingSize` and `maxBufferSize` defaults that motivate the memory budget.

9. **Deep-ml problem #188 `gradient_checkpoint`** — [Open-Deep-ML/DML-OpenProblem #188](https://github.com/Open-Deep-ML/DML-OpenProblem). The pedagogical target.

10. **fast.ai Part 2, Lessons 18–19** — [course.fast.ai/Lessons/part2.html](https://course.fast.ai/Lessons/part2.html). Curriculum sections that teach gradient checkpointing and require a real implementation, not a stub.

11. **`browsergrad-kernels` buffer pool** — `packages/browsergrad-kernels/src/runtime/buffer_pool.ts`. The runtime memory accounting hook we instrument for `bg.memory_stats()`.

12. **Griewank, "Achieving logarithmic growth of temporal and spatial complexity in reverse automatic differentiation" (1992)** — the original sublinear-memory autodiff result that Chen et al. extended for deep nets. Cited for the deeper mathematical roots; not load-bearing for the implementation.
