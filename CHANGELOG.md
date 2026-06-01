# Changelog

This is the workspace-level changelog. Per-package changelogs are at:

- [`packages/browsergrad-runtime/CHANGELOG.md`](./packages/browsergrad-runtime/CHANGELOG.md)
- [`packages/browsergrad-kernels/CHANGELOG.md`](./packages/browsergrad-kernels/CHANGELOG.md)
- [`packages/browsergrad-jit/CHANGELOG.md`](./packages/browsergrad-jit/CHANGELOG.md)
- [`packages/browsergrad-grad/CHANGELOG.md`](./packages/browsergrad-grad/CHANGELOG.md)

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Each package follows independent [SemVer](https://semver.org/).

## [Unreleased]

## 2026-05 — browsergrad-jit 0.8.0

- 28-opcode UOp IR + lazy `TensorProxy`
- Elementwise + softmax fusion with introspection
- Symbolic backward (13 VJP rules) + closure-backward safety net
- Mixed precision: `autocast` + cast-insertion IR pass + `GradScaler`
- Gradient checkpointing via IR rewrite
- Trace cache + browser-friendly safetensors (returns bytes)
- Functional transforms: `grad`, `vjp`, `vmap` (17 rules), `functional_call`
- WebGPU realizer bridge (forward-only); tiled GEMM; runtime fused-elementwise WGSL codegen
- Custom WGSL kernels via `@bg.custom_kernel`
- ONNX inference export (hand-rolled proto3 encoder; 14 ops mapped)
- `bg.experimental.webnn.matmul` spike behind a flag
- `bg.jit.cost_model.*` for tier selection
- `bg.kernels.transformer_block` megakernel constructor

## 2026-05 — browsergrad-runtime 0.1.1

- Hand-written `LabManifest` schema + parser + semver gate (`isSemverCompatible`, `assertCompatibleRuntime`, `LabRuntimeMismatch`)

## 2026-05 — browsergrad-kernels 0.1.0

- Real-WebGPU CI via Playwright + Chromium (`pnpm test:browser`)
- `createWebGpuRealizerBridge(device)` — production bridge consumed by jit
- `runDirect` / `materializeFloat32` / `uploadFloat32` — `GPUBuffer`-in/out dispatch
- `matmulTiledDirect` — 16×16 tiled GEMM
- `fusedElementwiseDirect` + `generateFusedWgsl` — runtime WGSL codegen
- `flashAttentionDirect` — Flash Attention v2 forward (known kernel-numerical issue on real Metal; tracked)

## Earlier history

See per-package changelogs.
