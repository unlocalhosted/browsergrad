# PRD-002 — First GPU Acceleration: Wire `browsergrad-kernels` → `browsergrad-grad`

**Status**: Draft
**Author**: vijayksingh
**Created**: 2026-05-26
**Target milestone**: P0 (Months 1–3)
**Roadmap ref**: [PRD.md](../../PRD.md) §7 P0.2
**Companion PRDs**: PRD-001 (coverage closeout), PRD-003 (cold-start UX)
**Implementation estimate**: 3 weeks

---

## 1. TL;DR

Today every operation in `browsergrad-grad` runs on NumPy via WASM. The `browsergrad-kernels` package ships six WGSL shaders (matmul, softmax, layernorm, activations, attention) but has no runtime connection to the grad library — the two packages share zero code at runtime. This PRD wires that connection for the four ops that dominate training-step latency: `matmul` (via `_matmul` in tensor.py line 544), `softmax` (via `F.softmax` in functional.py line 55), `layer_norm` (via `nn.LayerNorm.forward` delegation), and `scaled_dot_product_attention` (via `F.scaled_dot_product_attention` in functional.py line 498). The mechanism is a new JS bridge module `_bg_dispatch` registered in the Pyodide worker at boot; Python calls into it via Pyodide's `from js import` FFI. When WebGPU is available and input shapes exceed per-op size thresholds, the GPU path is taken; otherwise NumPy runs unchanged. PyTorch-conformance within 1e-4 is a hard requirement on every WGSL result.

---

## 2. Background

### 2.1 Why this matters now

`browsergrad-grad` v0.5.0 has solid correctness coverage — 234 integration tests, 5 PyTorch-conformance fixtures verified against `torch 2.12.0` within 1e-4 — but every training step runs on a single CPU core via NumPy-in-WASM. The `browsergrad-kernels` package has existed since the project's inception as a future fast path that was never connected. P0 is the right window to close this gap: curriculum labs are the imminent use case, and early adopters benchmark training step latency before anything else.

### 2.2 Performance evidence

**WebGPU matmul ceiling**: A hand-optimized WGSL matmul kernel on Apple M2 Pro exceeds 1 TFLOP/s against the chip's ~6 TFLOP/s theoretical peak ([nuss-and-bolts.com](https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel)). PRD.md §3.6 frames this as approximately 17% of FP32 peak, with native cuBLAS reaching ~75% on comparable hardware — meaning "WebGPU ≈ 1/4 of native CUDA throughput." The naive triple-loop WGSL kernel in `matmul.ts` sits below that ceiling, but it is GPU-parallelized and expected to beat NumPy-in-WASM for matrices above roughly 64×64.

**Dispatch overhead**: [arXiv:2604.02344](https://arxiv.org/abs/2604.02344) measured per-dispatch WebGPU API overhead at **24–36 μs on Vulkan** and **32–71 μs on Metal**. Total per-operation stack cost including Python interpreter overhead is approximately **95 μs per dispatch**. This is the floor: below the shape threshold, 95 μs of fixed dispatch cost exceeds the GPU compute savings. For matrices of 64×64 and larger (4,096 output elements), GPU compute time exceeds dispatch overhead and the GPU wins. Reducing dispatch count from 876 to 564 ops via kernel fusion yielded a 53% throughput improvement in that paper — quantifying why per-op overhead matters.

**Pyodide–JS interop**: Pyodide's `from js import` FFI enables Python to call JavaScript. JavaScript Promise Integration (JSPI) reached W3C Stage 4 on April 8, 2025, and shipped in Chrome 137 (May 27, 2025), enabling async JS Promises to be awaited synchronously from Python. Per-call FFI overhead is microseconds — negligible compared to the 95 μs dispatch floor.

**Expected speedup**: PRD.md §7 P0.2 states "Expected speedup: 3–5× on matmul-heavy paths." Calibrated against per-op dispatch at educational batch sizes with forward-pass-only GPU dispatch and NumPy backward.

---

## 3. User Stories

- A student runs a 2-layer MLP training loop on CIFAR-style data (hidden=512, batch=64). Currently: ~500 ms per training step. After this PRD: matmul ops dispatch to the GPU; the training step drops to under 200 ms on M1-class hardware.
- A course author builds a craftingattention lesson with a 6-layer transformer. They need the forward pass to feel interactive (under 3 seconds). The GPU attention kernel fires for 2D attention paths; the lesson runs without modifying any Python code.
- A developer testing in CI (Node, no `navigator.gpu`) sees `model(x)` continue to work and all 234 integration tests pass. No code change needed; the dispatch is transparent.

---

## 4. Goals and Non-Goals

**Goals:**
1. Four ops dispatch through WGSL kernels when WebGPU is available and shapes exceed thresholds: matmul (all call sites of `_matmul`), softmax (`F.softmax`, `dim=-1`), layer_norm (`nn.LayerNorm.forward`), scaled_dot_product_attention (`F.scaled_dot_product_attention`, 2D single-head only).
2. PyTorch-conformance: every WGSL result matches NumPy within 1e-4 on the existing fixture suite.
3. Benchmarked training step on a 2-layer MLP (CIFAR-style, 500 steps) is ≥3× faster than v0.5.0 when WebGPU is available.
4. NumPy fallback: 100% of 234 existing integration tests pass unchanged in Node.
5. User API completely unchanged. No new import, no device annotation required.

**Non-goals:**
- Batched matmul beyond 2D. The WGSL `matmul.ts` kernel is 2D-only. Higher-rank `@` falls back to NumPy; deferred to PRD-005.
- On-device tensor lifetime. Every dispatch is host→device + compute + device→host per call.
- GPU backward pass. Gradients computed by existing NumPy `_ctx` closures unchanged.
- Headless WebGPU in CI. The Node test harness has no `navigator.gpu`.
- `relu`, `gelu`, and other element-wise activations through WGSL. Dispatch overhead dominates at educational sizes.
- Batched multi-head attention. `attention.ts` is 2D single-head; batched paths fall back to NumPy until PRD-012.

---

## 5. Architecture

### 5.1 The dispatch mechanism

Python in Pyodide has no direct WebGPU access. The dispatcher lives in JavaScript. The chosen approach: register a JS module object `_bg_dispatch` in the Pyodide worker via `py.registerJsModule("_bg_dispatch", ...)`, then import it from Python via `from js import _bg_dispatch`. Python calls async methods on this object; Pyodide's JSPI bridge makes the JS Promise awaitable from Python synchronously.

The `_bg_dispatch` object is created by `createGpuBridge(device: KernelDevice | null)` in `gpu-bridge.ts`. When `device` is null (WebGPU unavailable), all methods return `null`. Python's `dispatcher.py` treats `null` as "use NumPy."

### 5.2 Data flow diagram

```
Python call site (tensor.py or functional.py)
  │  e.g. _matmul(a, b) — decides: GPU or NumPy?
  ▼
dispatcher.py  (NEW)
  │  checks: _BG_DISPATCH is not None
  │          AND input is 2D (for matmul/attention)
  │          AND numel(output) >= GPU_CUTOFF[op]
  │  if yes: serialize np.ndarray → Python list
  ▼
from js import _bg_dispatch
  │  await _bg_dispatch.matmul(aFlat, aShape, bFlat, bShape)
  │  (JSPI bridge: async JS Promise → synchronous from Python)
  ▼
gpu-bridge.ts  (NEW, runs in Worker JS context)
  │  reconstructs Tensor { shape, data: Float32Array }
  │  calls kernels.matmul(device, A, B) → Tensor C
  ▼
browsergrad-kernels/src/kernels/matmul.ts  (EXISTING)
  │  allocates GPU buffers (via runner.ts)
  │  dispatches WGSL shader (workgroup 8×8)
  │  reads back via MAP_READ, destroys all buffers
  ▼
gpu-bridge.ts
  │  returns { data: number[], shape: number[] }
  ▼
dispatcher.py
  │  np.array(result["data"], dtype=np.float32).reshape(result["shape"])
  │  returns numpy array to caller
  ▼
tensor.py (_matmul)
  │  Tensor(gpu_result) — identical Tensor construction as NumPy path
  │  backward closure captures a_data / b_data for NumPy grad (unchanged)
  ▼
Tensor returned to user code (autograd graph intact)
```

### 5.3 New files

**`packages/browsergrad-grad/src/python/dispatcher.py`**

At load time: `from js import _bg_dispatch` inside try/except (failure sets `_BG_DISPATCH = None`). Exports `_DISPATCH_AVAILABLE: bool`, `GPU_CUTOFF: dict`, and four dispatch functions. Each checks shape thresholds and device availability, serializes via `.flatten().tolist()`, calls the JS method via `await`, and returns a numpy array or `None`.

**`packages/browsergrad-runtime/src/worker/gpu-bridge.ts`**

Exports `createGpuBridge(device: KernelDevice | null): GpuBridgeApi`. When `device` is null: stub returning `Promise<null>` for all methods. Otherwise: four async methods calling `kernels.matmul`, `kernels.softmax`, `kernels.layernorm`, `kernels.attention`. Each reconstructs a `Tensor`, dispatches, returns `{ data: Array.from(result.data), shape: result.shape as number[] }`.

### 5.4 Files to modify

- **`packages/browsergrad-grad/src/python/tensor.py`** (line 544, `_matmul`): if both inputs are 2D, try `dispatcher.dispatch_matmul`. On non-None result, build `out = Tensor(gpu_result)`. Backward closure unchanged.
- **`packages/browsergrad-grad/src/python/functional.py`** (line 55, `softmax`): probe `dim == -1` + dispatch available + `x.data.size >= GPU_CUTOFF["softmax"]`.
- **`packages/browsergrad-grad/src/python/functional.py`** (line 498, `scaled_dot_product_attention`): 2D guard + no mask + not is_causal + size threshold.
- **`packages/browsergrad-grad/src/python/nn_chunks/norm.py`** (`LayerNorm.forward`): probe `normalized_shape` product threshold.
- **`packages/browsergrad-runtime/src/worker/index.ts`** (near line 89): `const kernelDevice = await createDevice({ powerPreference: "high-performance" }).catch(() => null)`, then `py.registerJsModule("_bg_dispatch", createGpuBridge(kernelDevice))`. Must precede `py.runPythonAsync(PY_PREAMBLE)`. Null device must not abort boot.
- **`packages/browsergrad-runtime/package.json`**: add `"@unlocalhosted/browsergrad-kernels": "workspace:*"` as dependency.

### 5.5 Backward pass: always NumPy

The GPU path is forward-only in this PRD. Gradients are computed by existing NumPy `_ctx` closures. The 3× end-to-end target accounts for this forward-GPU / backward-NumPy split.

### 5.6 Shape cutoff heuristics

Starting values exposed as `GPU_CUTOFF` in `dispatcher.py` (tunable post-benchmark):

| Op | Threshold | Gate condition |
|---|---|---|
| matmul | M×N ≥ 4096 (≥64×64 output) | Both A and B must be 2D |
| softmax | numel ≥ 512 | dim must be -1 |
| layernorm | normalized_shape product ≥ 512 | Any shape |
| attention | S×D ≥ 512 | 2D Q/K/V only |

### 5.7 Test environment without WebGPU

Node has no `navigator.gpu`. `createDevice` is wrapped in try/catch; failure sets `kernelDevice = null`. `createGpuBridge(null)` returns null-stubs. `dispatcher.py` treats null as NumPy fallback. The 234 integration tests continue to pass unchanged.

---

## 6. API Surface

The user-facing API is completely unchanged:

```python
import browsergrad_grad as grad
import browsergrad_grad.functional as F
import browsergrad_grad.nn as nn

model = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))
x = grad.Tensor(batch, requires_grad=True)
logits = model(x)           # matmul routes through GPU when available
loss = F.cross_entropy_loss(logits, targets)
loss.backward()             # gradients on NumPy (unchanged)
```

New internal symbols:
- `dispatcher.py`: `_BG_DISPATCH`, `GPU_CUTOFF`, `dispatch_matmul`, `dispatch_softmax`, `dispatch_layernorm`, `dispatch_attention_2d`
- `gpu-bridge.ts`: `GpuBridgeApi`, `createGpuBridge(device)`
- Worker init: `kernelDevice: KernelDevice | null`, registered as `_bg_dispatch`

---

## 7. Implementation Plan

**Week 1 — Bridge and plumbing**

- Create `gpu-bridge.ts` with all four methods and null-device guard. Unit test: `createGpuBridge(null)` returns all-null stubs.
- Wire `createDevice` + `createGpuBridge` into `worker/index.ts`. Try/catch must not abort Pyodide boot on failure.
- Add `@unlocalhosted/browsergrad-kernels` as workspace dependency in `browsergrad-runtime/package.json`.
- Create `dispatcher.py` with `dispatch_matmul` and `dispatch_softmax` first. Implement `from js import _bg_dispatch` guard and `GPU_CUTOFF`.
- Manual browser smoke test: DevTools GPU timeline shows a WebGPU dispatch when running a 512×512 Python matmul.

**Week 2 — All four ops + conformance**

- Wire all four ops in `tensor.py`, `functional.py`, and `nn_chunks/norm.py`.
- Extend `cross-package-conformance.test.ts` with four NumPy round-trip tests (Node, no WebGPU) confirming serialization/deserialization correctness.
- Create stub `tests-integration/gpu-conformance.browser.test.ts` for Playwright wiring in week 3.

**Week 3 — Benchmark, calibrate, acceptance**

- Add `scripts/benchmark-gpu.ts`: browser-runnable wall-time table for matmul shapes [32², 64², 128², 256², 512², 1024²] on both paths. Calibrate `GPU_CUTOFF` from empirical crossover.
- Run 2-layer MLP training benchmark (batch=64, in=784, hidden=512, out=10, 500 steps). Confirm ≥3× vs v0.5.0.
- Wire Playwright for `gpu-conformance.browser.test.ts`; verify 1e-4 conformance on all four ops in Chrome with WebGPU.
- Run full 234-test Node suite: green.
- Run PyTorch-conformance fixture suite: green within 1e-4.

---

## 8. Acceptance Criteria

1. **WGSL conformance**: each of matmul, softmax, layernorm, attention produces results within 1e-4 of the NumPy oracle when dispatched through the WGSL backend in Chrome. Verified by `gpu-conformance.browser.test.ts` via Playwright.
2. **Performance**: ≥3× faster training step (2-layer MLP, batch=64, 500 steps, CIFAR-style) on M1-class Mac in Chrome vs v0.5.0.
3. **Fallback correctness**: 100% of 234 existing integration tests pass in Node unchanged.
4. **No API change**: every existing integration test's Python code runs without modification.
5. **Error safety**: `createDevice` failure is caught; Pyodide boots normally; console warning emitted; no Python exception propagates.

---

## 9. Test Strategy

**Node integration tests (existing, must stay green)**: All 234 tests in `packages/browsergrad-grad/tests-integration/`.

**Cross-package conformance extension** (`cross-package-conformance.test.ts`): Four new tests comparing `dispatcher.dispatch_*` outputs (Node, NumPy path) against `reference.*` from `browsergrad-kernels` within 1e-4.

**Browser GPU conformance** (`gpu-conformance.browser.test.ts`, new): Playwright-launched Chrome with WebGPU. For each op, dispatch through WGSL and compare to JS reference within 1e-4.

**Benchmark script** (`scripts/benchmark-gpu.ts`, new): Browser-runnable wall-time tables. Run manually; attach output to PR as evidence for the ≥3× criterion.

---

## 10. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Host↔device copy cost per call erases GPU gain for matrices near the threshold | High | Medium | Benchmark-driven cutoff calibration; expose `GPU_CUTOFF` as tunable |
| R2 | Batched matmul (higher-rank `@`) used by transformer models falls back to NumPy in v0 | High | Medium | Documented non-goal; addressed in PRD-005 |
| R3 | WGSL conformance untestable in CI (no Node WebGPU); gap persists until Playwright CI wired | High | High | Browser conformance tests via Playwright close this; manual verify in interim |
| R4 | JSPI async bridge is Chrome 137+ only; Firefox/Safari use older coroutine model | Medium | Medium | Test on Chrome 137+ first; older async path used as fallback |
| R5 | 3× speedup target not met because naive WGSL matmul is memory-bandwidth-bound at educational sizes | Medium | Medium | Revise target based on benchmark; tiled matmul variant noted in `matmul.ts` comments is additive |
| R6 | `_bg_dispatch` registration ordering relative to `PY_PREAMBLE` introduces timing constraint | Low | Low | Enforce registration before `runPythonAsync(PY_PREAMBLE)` in code review |

---

## 11. Open Questions

1. **Package boundary**: Should GPU dispatch fold into `@unlocalhosted/browsergrad-grad` (transparent) or ship as separate `@unlocalhosted/browsergrad-grad-gpu`? **Decided: fold in.** The user API is unchanged; a separate package creates friction for no benefit. Only `browsergrad-runtime` gains a new dependency on `browsergrad-kernels`.

2. **Published benchmark artifact**: Should `benchmark-gpu.ts` results be hosted at a public URL? Deferred until post-PR when numbers are confirmed favorable.

3. **Attention op scope**: `F.scaled_dot_product_attention` is called with batched 4D inputs from `nn.MultiHeadAttention.forward`. Those calls fall back to NumPy in v0. Punted to PRD-012 (megakernel).

4. **Dawn for CI headless WebGPU**: The gap between "WGSL conformance requires a browser" and "CI runs in Node" remains. Headless WebGPU via Dawn + wgpu-native is feasible but adds significant CI infrastructure. Tracked as future improvement outside this PRD.

---

## 12. References

- [PRD.md §7 P0.2](../../PRD.md) — canonical feature spec
- [arXiv:2604.02344](https://arxiv.org/abs/2604.02344) — WebGPU dispatch overhead, 95 μs total stack cost, 53% throughput gain from fusion
- [nuss-and-bolts.com: Optimizing a WebGPU Matmul Kernel](https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel) — 1 TFLOP+ on M2 Pro; ~17% of FP32 peak
- [Pyodide JSPI blog](https://blog.pyodide.org/posts/jspi/) — JSPI finished April 2025, Chrome 137 May 2025
- [WebGPU Baseline (web.dev)](https://web.dev/blog/webgpu-supported-major-browsers) — Baseline January 2026
- `packages/browsergrad-kernels/src/kernels/matmul.ts` — WGSL matmul, 2D-only, workgroup 8×8
- `packages/browsergrad-kernels/src/kernels/attention.ts` — WGSL attention, 2D single-head
- `packages/browsergrad-kernels/src/runner.ts` — GPU dispatch helper, per-call buffer lifecycle
- `packages/browsergrad-kernels/src/device.ts` — `createDevice`, pipeline cache
- `packages/browsergrad-grad/src/python/tensor.py:544` — `_matmul`
- `packages/browsergrad-grad/src/python/functional.py:55,498` — `softmax`, `scaled_dot_product_attention`
- `packages/browsergrad-runtime/src/worker/index.ts:89` — `bootPyodide`, `py.registerJsModule`
- `packages/browsergrad-grad/tests-integration/cross-package-conformance.test.ts` — existing cross-package tests
- `packages/browsergrad-grad/tests-integration/pytorch_conformance.test.ts` — PyTorch fixture conformance

---

*If implementation diverges from what is specified here, update this PRD and log the decision in ARCHITECTURE.md.*
