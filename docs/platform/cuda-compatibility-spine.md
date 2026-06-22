# CUDA Compatibility Spine

BrowserGrad compiler is a real-GPU compatibility compiler, not a CPU simulator.
CUDA-shaped source lowers into semantic IR, then each primitive gets one of four
lowering decisions:

- `native`: direct WGSL/WebGPU lowering runs on real GPU.
- `gpu-polyfill`: one or more WGSL/WebGPU passes implement CUDA semantics on real GPU.
- `cpu-reference`: CPU reference exists for correctness/traces, but GPU lowering is not available.
- `unsupported`: no honest implementation yet; emit diagnostic.

Every CUDA gap should map to a semantic family:

- `frontend`: C/CUDA syntax, macros, helpers, types.
- `memory`: global/shared/constant/local memory spaces.
- `atomic`: CUDA atomics and memory ordering.
- `texture`: texture/surface objects and reads/writes.
- `subgroup`: warp intrinsics and cooperative groups.
- `library`: cuRAND, cuFFT, and reusable device-library islands.
- `runtime`: streams, events, dynamic parallelism, launch orchestration.
- `feature`: browser/WebGPU capability gates such as `shader-f16` or subgroups.
- `safety`: invalid or unsafe kernels rejected before dispatch.

Public APIs:

- `getCudaFeatureRegistry()` returns known feature records.
- `describeCudaDiagnostic(diagnostic)` maps compiler diagnostics to semantic features.
- `createCudaLoweringPlan(diagnostics)` summarizes whether a kernel can run on GPU,
  requires GPU polyfill, has CPU reference coverage, or is unsupported.
- `createCudaRuntimePlan(compiled)` reports runtime operations that need host
  orchestration before single-dispatch WebGPU can run: device launches,
  device sync, peer copies, and grid sync.
- `createCudaWebGpuExecutionPlan(compiled, input, launch, { compileKernel })`
  returns the exact executable WebGPU plan kind and sequence steps:
  `single-dispatch`, `grid-sync-phases`, `host-dynamic-launch`, or
  `host-peer-copy`. Platform preflight should use this interface instead of
  duplicating runner heuristics.
- `prepareCompiledKernelWebGpu(device, compiled, input, launch)` prepares the
  same executable WebGPU plan once and reruns it over resident buffers. Use it
  for hot loops with fixed launch shape, scalar params, and bindings.
- `normalizeCudaWebGpuReadbackNames(compiled, names)` maps logical compiler
  readback names, such as `DevicePool* dp` -> `dp`, to internal WGSL storage
  bindings. Platform code should not depend on backing buffer names.

Rule: do not add assignment-specific fixes. Add semantic primitives, reference
truth, WGSL lowering, browser tests, and corpus audit evidence.

Current corpus gate:

```sh
pnpm --filter @unlocalhosted/browsergrad-compiler audit:cuda-120
```

- `AdepojuJeremy/CUDA-120-DAYS--CHALLENGE` audit: `225/243` real code-kernel
  definitions compile as single-dispatch WGSL/WebGPU. Another `10/243` are
  real-GPU runnable through WebGPU orchestration lifts (`grid-sync-phases`),
  for `235/243` total WebGPU coverage. `7/243` remain true reference-only,
  and `1/243` remains a hard gap after filtering docs/pseudocode.
- `referenceFallbackOk` is `17/243`: kernels whose semantics are understood by
  CPU reference. `referenceOnlyOk` is stricter and excludes kernels now runnable
  on real WebGPU through orchestration.
- Recent semantic lifts: `DevicePool*` bump allocation, raw pointer pool allocation
  with integer offset counters, casted pool pointer reads/writes, WebGPU atomic
  offset updates, DevicePool aliasing across host-lifted child launches,
  positive pointer-offset child launches via generated base-offset uniforms, and
  conservative host-lifted peer copies through a typed WebGPU copy dispatch.
- Hot-loop dispatch can keep both caller buffers and compiler-generated
  execution sequences resident: `residentBuffers` avoids upload/readback churn,
  and prepared compiler/WebGPU runners avoid rebuilding pipelines and bind
  groups between iterations. Prepared single-dispatch and grid-sync phase plans
  can also update scalar params without reprepare. No-readback prepared runs can
  opt into `awaitCompletion: true` when timing gates or watchdogs need real GPU
  completion instead of JS command submission. Host-orchestrated dynamic launch /
  peer-copy plans keep scalar params fixed for now.
- Device-side launches now parse into IR and can run in CPU reference when
  `referenceDynamicParallelism` is enabled. WebGPU can host-lift conservative
  child launches into a multi-dispatch sequence when the parent launch has one
  workgroup, the child block size is statically known, pointer args alias the
  same named storage buffers, and child scalar launch args are host-evaluable.
  DevicePool pointer params alias their pool data/offset bindings, and positive
  pointer-offset args lower to base-offset uniforms. Child kernels may also
  contain single-invocation guarded peer copies when that copy itself is
  host-liftable. Per-thread launch queues, recursive launches, negative pointer
  offsets, and device-derived launch args remain reference-only.
- CUDA runtime calls such as `cudaDeviceSynchronize` and `cudaMemcpyPeerAsync`
  classify as runtime orchestration gaps. Standalone `cudaDeviceSynchronize()`
  is a WebGPU-safe no-op because dispatch completion is host-managed. Peer
  copies can run in CPU reference with `referenceCudaRuntime`; WebGPU can
  host-lift single-invocation guarded typed buffer copies when source,
  destination, offsets, and byte count are host-evaluable. Mixed types, pools,
  device-derived counts, or side effects after copy remain reference-only.
- Cooperative `grid.sync()` can run in CPU reference with `referenceGridSync`.
  Safe top-level uniform `grid.sync()` also runs on real WebGPU as multiple
  dispatch phases over shared GPU buffers. Pure launch-derived locals are
  replayed in later phases, and shared memory can be reused after sync when the
  phase rewrites it before any read. Non-uniform sync, non-replayable private
  locals crossing phases, or shared-memory read-before-rewrite remain
  reference-only.
- Remaining failures group cleanly: dynamic parallelism/runtime launches,
  cooperative groups/grid sync, and one incomplete pseudocode symbol.
- Use `--limit N` to cap printed failures and `--details`/`--json` to emit
  `{ summary, failures }` with `webGpuLiftBlocker` values. Use threshold flags
  such as `--expect-webgpu-min`, `--expect-reference-only-max`, and
  `--expect-hard-fail-max` so feature triage is grounded in executable
  regression gates instead of prose-only notes.

Performance gate:

- Compiler/runtime perf is tracked by
  `pnpm --filter @unlocalhosted/browsergrad-compiler bench -- --markdown /tmp/bg-cuda-lite-bench.md`.
  It reports JSON plus optional markdown for compile hot paths, CPU reference
  execution, host-lifted dynamic planning, and peer-copy planning. Treat the
  numbers as regression evidence, not universal pass/fail thresholds.
- Browser/WebGPU hot-loop perf should compare one-shot execution against
  `prepareCompiledKernelWebGpu()` over resident buffers. Keep this as measured
  evidence, not prose claims; no-readback prepared measurements should use
  `awaitCompletion: true`. Run:
  `pnpm --filter @unlocalhosted/browsergrad-compiler bench:browser -- --markdown /tmp/bg-cuda-lite-webgpu-bench.md`.
