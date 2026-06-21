# Changelog

All notable changes to `@unlocalhosted/browsergrad-kernels`.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `defineCuda1DProgram()`, `simulateCuda1DProgram()`, and
  `emitCuda1DProgramWgsl()` provide a tiny CUDA-shaped 1D program IR that can
  run through a deterministic simulator and lower to WGSL.
- `simulateCuda1DGrid()`, `referenceSaxpy()`, and
  `referenceExclusiveScan()` provide CUDA-shaped browser-safe oracles for GPU
  Puzzles and CS149 A3-style rubrics.
- `referenceFlashAttention()` returns browser-safe FlashAttention output plus
  log-sum-exp tensors for CS336 A2-style forward/LSE rubric checks.
- `referenceFlashAttentionBackward()` recomputes Q/K/V gradients for
  CS336 A2-style backward checks without PyTorch autograd, Triton, or CUDA.
- `createKernelRubric()` records pass/fail assertions for JS/WebGPU lab
  rubrics, including `assertCloseTensor()` with shape checks, compact previews,
  first failing index, non-finite value detection, and max absolute error.
- `kernelRubricFailureToAssertionDetails()` formats kernel failure details into
  `expected` / `actual` strings for BrowserGrad-style assertion callbacks.
- `createBrowsergradKernelRubric(target)` adapts kernel rubric assertions to a
  BrowserGrad JS rubric context or any compatible assertion target.

## [0.1.1] — 2026-06-02

Dogfood pass on the published 0.1.0 tarball surfaced three issues. All fixed.

### Fixed

- **GPU `kernels.softmax` returned all zeros** on Chromium's
  SwiftShader + Metal-driver path (real-WebGPU dogfood in headed
  Chromium on macOS). Root cause, isolated via incremental probe
  kernels: the literal `-3.4028235e38` used as the max-init sentinel
  parsed to **0** on that driver path, so the comparison
  `v > maxVal` always evaluated false, no max was found, and the
  whole row's subsequent exp-divide-by-sum collapsed (sum overflowed
  or wrote zeros). Fix: initialize `maxVal` with `X[base]` and iterate
  from `i = 1`. Sidesteps the literal-parse issue and is also a touch
  more numerically robust. Pass 3 was additionally tightened to a
  pure-write (re-computing `exp(x - max)` rather than reading Y back
  from pass 2) — defense in depth against read-after-write storage
  semantics on the same driver path.
  Caught by `@unlocalhosted/browsergrad-dogfood`; was not covered by
  the package's own `tests-browser/webgpu_real.test.ts` (which
  exercises matmul/tiled-GEMM/fused-elementwise/FA-v2 but not
  softmax). Cascades into `kernels.attention` correctness (attention
  composes softmax).
- **`WebGpuRealizerBridge.materialize` type contract**: declared
  `Uint8Array`, runtime always returned `Promise<Uint8Array>`. TS
  consumers following the `.d.ts` got `undefined.buffer` at runtime;
  the package's own browser test silently cast via
  `as unknown as Promise<Uint8Array>`. The interface now honestly
  declares `Promise<Uint8Array>` and the implementation is `async`.
  Pyodide JSPI consumers (Python) still see a synchronous return — the
  Promise is unwrapped at the boundary as documented.
- **`kernels.attention` / `reference.attention` error message** when
  Q seq ≠ K=V seq is now explicit about the v0 self-attention-only
  limitation and points at the PRD-012c follow-on. Behavior unchanged;
  documentation gap closed.

## [0.1.0] — 2026-05-25

Initial release. Six WGSL kernels, each with a pure-JS reference.

### Added

- `createDevice(options?)` — wraps a `GPUDevice` with a pipeline cache.
- `tensor(shape, data)` — small constructor helper for `Tensor` literals.
- `kernels.matmul(device, A, B)` — naive 2D matmul, f32.
- `kernels.softmax(device, x)` — stable softmax along the last axis.
- `kernels.relu(device, x)` — elementwise.
- `kernels.gelu(device, x)` — tanh-approximation (GPT-2/BERT variant).
- `kernels.layernorm(device, x, { gamma?, beta?, eps? })` — along last axis.
- `kernels.attention(device, Q, K, V)` — scaled dot-product. Single head, `[S, D]` shapes.
- `reference.*` — pure-JS counterparts for every kernel.
  Exposed at the top-level entry and at the `./reference` subpath import.
- `KernelDevice.getStats()` / `clearCache()` for debugging.
- `KernelError` for input shape and device errors.
- Full TypeScript declarations + source maps.
- 26 vitest tests covering the public surface, argument validation, and
  reference-impl numerical correctness against hand-checked values.

### Deferred

Planned but not in 0.1.0; additive when they land:

- **Browser conformance tests** — run the WGSL kernels in a real WebGPU
  context and compare to the JS reference (`1e-4` tolerance). Requires
  `@vitest/browser` + Playwright; planned for next.
- **Pre-allocated buffer mode** (`kernel.runOnGpu`) — for hot training loops
  that want to keep tensors on the GPU between calls. v0 always copies
  back to JS each call.
- **Tiled / optimized variants** of matmul, fused attention. Same surface;
  faster paths chosen via a future `mode` option.
- **Batched + multi-head attention.** v0 is single-head, unbatched.
- **f16 support.** All v0 kernels are f32.
- **Mask support for attention.** v0 has no masking.

### Known limitations

- v0 allocates fresh GPU buffers every call. Useful for "compute once" work;
  not optimal for hot loops. The grad library (or any consumer) should
  pre-allocate via the planned `runOnGpu` path once it lands.
- Attention dispatches four separate kernels (transpose → matmul → scale →
  softmax → matmul). A fused implementation will be much faster but is
  not required for correctness or pedagogy.

[0.1.0]: https://github.com/unlocalhosted/browsergrad/releases/tag/kernels%40v0.1.0
