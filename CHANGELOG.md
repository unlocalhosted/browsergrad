# Changelog

This is the workspace-level changelog. Per-package changelogs are at:

- [`packages/browsergrad-runtime/CHANGELOG.md`](./packages/browsergrad-runtime/CHANGELOG.md)
- [`packages/browsergrad-kernels/CHANGELOG.md`](./packages/browsergrad-kernels/CHANGELOG.md)
- [`packages/browsergrad-semantic-core/CHANGELOG.md`](./packages/browsergrad-semantic-core/CHANGELOG.md)
- [`packages/browsergrad-compiler/CHANGELOG.md`](./packages/browsergrad-compiler/CHANGELOG.md)
- [`packages/browsergrad-jit/CHANGELOG.md`](./packages/browsergrad-jit/CHANGELOG.md)
- [`packages/browsergrad-grad/CHANGELOG.md`](./packages/browsergrad-grad/CHANGELOG.md)

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Each package follows independent [SemVer](https://semver.org/).

## [Unreleased]

- Closes Grad's remaining fake placement seams: `torch.tensor(device=...)`
  accepts CPU only, and `nn.Module.to(...)` preserves CPU identity while
  rejecting unavailable devices or unimplemented parameter dtype conversion.
- Makes Grad tensor device placement honest: CPU requests preserve identity,
  while CUDA/MPS/XPU/Meta and other unavailable eager devices fail before
  execution. Invalid or ambiguous `Tensor.to()` requests no longer become
  silent CPU no-ops.
- Preserves Grad autograd through float16/float32/float64 `Tensor.to()` casts,
  including source-dtype VJPs and non-contiguous layout order; casts involving
  bool or integer storage remain explicit detached boundaries.
- Makes Grad `Tensor.detach()` a truthful zero-copy metadata operation: it
  returns a distinct tensor sharing the source storage, dtype, strides, and
  layout while severing autograd history.
- Makes Grad `Tensor.contiguous()` truthful: already C-contiguous tensors keep
  identity, while non-contiguous tensors produce an owning C-order,
  dtype-preserving copy with an identity-gradient edge.
- Removes Grad's silent `bf16`/`bfloat16` to float32 substitution.
  `torch.bfloat16` is now a distinct unsupported token, and tensor,
  conversion, and parameter-construction paths reject it before allocation.
  `nn.Linear` and `nn.Embedding` now enforce their exact float32 parameter
  contract instead of ignoring their `dtype` argument.
- Migrates the legacy `bg.kernels.flash_attention` compatibility constructor
  to typed `ATTENTION_FORWARD` IR and introduces the accurately named
  `bg.kernels.attention_forward` entrypoint. The bounded rank-4 float32 profile
  owns stable CPU semantics and explicit autograd/transform/export/plan
  refusals; direct WebGPU remains the legacy row-wise online-softmax route and
  makes no block-tiled or FlashAttention-v2 claim.
- Removes the constructor-only `bg.kernels.transformer_block` placeholder. It
  produced an opaque node with no executable backend or autograd contract;
  PRD-012c remains an unimplemented typed graph/codegen design.
- Begins Gate 6 framework convergence by migrating JIT `Tensor.expand` from an
  opaque callback to typed `BROADCAST_TO` across CPU, autograd, transforms,
  export, planning, and resident WebGPU bridge paths. Grad eager expand now
  shares its validated shape/dtype fixture, preserves non-f32 dtypes, and
  retains explicitly owning materialization.
- Adds the first versioned executable JIT framework-operation registry and
  generates detached public support reporting from its validator-bound typed
  records rather than method presence or hand-written tables.
- Removes the non-executable `bg.experimental.webnn.matmul` constructor-only
  spike. WebNN presence detection remains, while future WebNN execution must
  consume typed graph IR through an explicit backend contract.
- Migrates JIT `Tensor.abs` and `Tensor.sign` together from opaque callbacks to
  typed unary semantics across CPU, closure/symbolic autograd, functional grad,
  vmap, and ONNX. Their tensor-plan/WebGPU profile remains an explicit refusal
  until a portable lowering exists.
- Migrates JIT `Tensor.sin` and `Tensor.cos` together to typed floating-only
  unary semantics, removing integer declared/realized dtype drift and enabling
  mutually typed closure/symbolic gradients, functional grad, vmap, and ONNX
  while retaining explicit tensor-plan/WebGPU refusal.
- Migrates JIT `Tensor.clamp` to typed finite-bound semantics with hostile
  scalar-coercion rejection, floating dtype preservation, inclusive typed VJP,
  functional grad, vmap, and ONNX `Clip`; device planning remains an explicit
  refusal.
- Migrates JIT `Tensor.flip` to typed single-axis reversal with strict scalar
  admission, owning CPU materialization, involutive typed VJP, batch-safe
  vmap, ONNX `Slice`, and explicit negative-stride device refusal.
- Migrates JIT `Tensor.repeat` to typed bounded tile semantics with owning
  dtype-preserving CPU realization, reduction VJP, batch-safe vmap, ONNX
  `Tile`, and explicit canonical-layout/device refusal. Grad repeat now shares
  the same conformance fixture and no longer silently casts every result to
  float32.
- Migrates JIT `Tensor.repeat_interleave` to typed selected-axis replication
  with strict scalar admission, block-sum VJP, batch-safe vmap, exact ONNX
  `Unsqueeze`/`Tile`/`Reshape`, and explicit layout/device refusal. Grad now
  shares its conformance fixture and preserves output and gradient dtype.
- Migrates JIT `Tensor.prod` to typed static product reduction with canonical
  axes, owning dtype-preserving scalar/tensor results, zero-aware VJP,
  batch-safe vmap, ONNX `ReduceProd`, and explicit device refusal. Grad shares
  the same conformance fixture and zero-aware dtype-preserving derivative.
- Migrates JIT `Tensor.gather` to typed bounds-checked `INDEX` semantics with
  int64 indices, deterministic duplicate-index scatter-add VJP, paired vmap,
  ONNX `GatherElements`, and explicit device refusal. Grad shares the strict
  conformance fixture, preserves source/output/gradient dtype, and keeps int64
  index tensors intact through slice, reshape, transpose, and permute views.
- Migrates JIT `Tensor.var` to typed static reduction semantics with canonical
  axes and correction, owning float16/32/64 CPU results, centered closure and
  symbolic gradients, batch-safe vmap, exact float32 ONNX decomposition, and
  explicit device refusal. Grad shares its dtype, correction, and refusal
  fixture instead of silently casting variance results and gradients to f32.
- Migrates JIT `Tensor.masked_fill` from an opaque callback to typed `WHERE`
  with strict bool-mask broadcasting, exact source-dtype scalar fill,
  mask-complement VJP, vmap, ONNX `Where`, and explicit device refusal. Grad
  shares the same fixture and preserves source/output/gradient dtype.
- Migrates JIT `Tensor.tril` from an opaque callback to typed triangular
  selection with rank/dtype/diagonal validation, owning CPU results,
  idempotent closure and symbolic VJP, batch-safe vmap, ONNX `Trilu`, and
  explicit device refusal. Grad shares the same values, dtype, saturation,
  gradient, and refusal fixture.
- Migrates JIT `Tensor.triu` from the final opaque triangular callback to typed
  upper selection with the shared strict triangular contract, owning CPU
  results, idempotent closure and symbolic VJP, batch-safe vmap, exact ONNX
  `Trilu upper=1`, and explicit device refusal. Grad and JIT now consume one
  two-variant triangular conformance harness.

## 2026-07 — semantic-core, kernels, compiler, and JIT semantic migration

- Adds the bounded canonical semantic wire/layout/kernel foundation and one
  constructor-owned materializing view-copy contract shared across frontends.
- Adds kernels-owned WGSL view-copy lowering with strict CPU/actual-WebGPU
  evidence and exact-commit publish authorization.
- Adds compiler verified-layout bindings, semantic/cache identity, guarded
  read-only lowering through the existing CPU and WGSL backends, and a required
  actual-device conformance lane.
- Adds typed JIT permutation requests beside the frozen tensor plan and routes
  them through constructor-owned kernels artifacts and resident canonical WGSL
  without legacy offset reconstruction.

## 2026-07 — browsergrad-kernels/compiler 0.1.2

- Republishes `@unlocalhosted/browsergrad-kernels` from rebuilt `dist/` so the
  WGSL program, float16, CUDA concept, rubric, and CUDA program APIs are present
  on npm.
- Republishes `@unlocalhosted/browsergrad-compiler` through `pnpm publish` so
  its kernels dependency is rewritten from `workspace:*` to the published
  kernels version.

## 2026-07 — browsergrad-compiler 0.1.1

- CUDA-lite/WebGPU compiler bugbash release: source and dist real-world verifier
  gates are green at `677/0/0`, compile/codegen audit has `0` hard failures,
  pinned corpus WebGPU fixture outputs are `117/117`, normal smoke is
  `568/0/0`, and skips remain `0`.
- Adds fixture/status/perf guardrails for faster compiler iteration and broader
  real-browser coverage of pointer/vector storage, texture/surface lanes,
  active-lane barriers, byte helper atomics, and descriptor-specialized texture
  lowering.

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
