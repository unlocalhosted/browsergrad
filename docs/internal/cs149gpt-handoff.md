# CS149GPT Handoff

Source: <https://github.com/stanford-cs149/cs149gpt>

Use this doc when turning Stanford CS149GPT into BrowserGrad and
CraftingAttention lab slices. Keep it as a benchmark profile record, not
BrowserGrad package identity.

## Upstream Shape

- The assignment centers on attention optimization for a small GPT-style
  workload.
- Native upstream assumptions include C++ extension builds, ISPC/OpenMP-style
  parallelism, PyTorch integration, memory-footprint reasoning, and performance
  comparison against less optimized attention implementations.
- The browser first slice should preserve the math and memory-behavior
  invariants before trying to clone native compiler/runtime behavior.

## Browser-Safe First Slice

- Start from `docs/internal/cs149gpt.profile.json`.
- Register `_bg_attention_math` as profile-local oracle glue backed by
  `@unlocalhosted/browsergrad-kernels`.
- Use generic attention helpers:
  - `reference.attention()` for naive scaled dot-product attention.
  - `referenceBlockedAttention()` for blocked attention with explicit
    intermediate-score memory estimates.
  - `referenceFusedAttention()` for fused attention output equivalence and
    zero materialized score-buffer accounting.
  - `estimateAttentionMemory()` for standalone memory-footprint rubrics.
  - `referenceFlashAttention()` for flash-attention-style output and
    log-sum-exp fixtures when the lab wants that comparison.
- Keep `attention_performance_smoke` informational until native external or
  calibrated browser timing fixtures exist.

## Platform Proof

- CraftingAttention loads the real `cs149gpt.profile.json`.
- It preflights the Python route with `pyodide`, `attention-oracle`,
  `browser-cpp-simulator`, `performance-rubric`, and simulated
  `simd-simulator` capabilities.
- It registers a Pyodide JS module named `_bg_attention_math`, then runs a real
  Pyodide lab test that imports that module from Python and checks:
  - naive attention correctness.
  - blocked attention matches naive output and reduces score-tile memory.
  - fused attention matches naive output and reports no materialized score
    buffer.
  - flash-attention output and log-sum-exp fixture values.
  - explicit memory estimates for score/probability buffers.

## Non-Portable Upstream Assumptions

- Native C++/PyTorch extension compilation inside the browser worker.
- ISPC/OpenMP compiler behavior.
- Exact CPU wall-clock speedups, cache counters, and native thread scheduling.

## Later Slices

- Add external native runner support for the real C++ extension path.
- Add browser Worker/SIMD timing calibration once stable enough to avoid fake
  performance claims.
- Add trace artifacts for score-buffer materialization, block size, and
  memory-bandwidth pressure.
