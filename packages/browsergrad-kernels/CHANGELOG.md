# Changelog

All notable changes to `@unlocalhosted/browsergrad-kernels`.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
