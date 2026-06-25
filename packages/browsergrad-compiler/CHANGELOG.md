# @unlocalhosted/browsergrad-compiler Changelog

## 0.1.0

- Initial CUDA-lite parser, analyzer, Kernel IR, reference interpreter, WGSL
  emitter, and WebGPU runner.
- Added real WebGPU orchestration for safe `grid.sync()` phase splitting,
  standalone `cudaDeviceSynchronize()`, and conservative host-lifted dynamic
  child launches.
- Added DevicePool aliasing and positive pointer-offset support for host-lifted
  dynamic launches, plus conservative host-lifted `cudaMemcpyPeerAsync` typed
  buffer copies.
- Added composed host orchestration for child dispatches whose child kernel
  performs a host-liftable runtime copy.
- Added `createCudaRuntimePlan()`, `createCudaGridSyncPhasePlan()`,
  `createCudaHostDynamicLaunchPlan()`, `createCudaPeerCopyPlan()`, and
  `createCudaWebGpuExecutionPlan()` for platform/rubric preflight.
- Refactored WebGPU execution through an explicit plan interface so native
  dispatch, grid-sync phases, dynamic child launches, and runtime-copy lifts share
  one runner path.
- Added `residentBuffers` pass-through for compiler WebGPU execution so
  platform callers can keep storage buffers on GPU and opt out of readback.
- Added `prepareCompiledKernelWebGpu()` for hot-loop compiler dispatch over
  resident buffers without rebuilding pipelines and bind groups each iteration.
- Added logical compiler readback-name normalization so `DevicePool* dp`
  callers can request `"dp"` instead of internal WGSL storage names.
- Prepared compiler runners can update scalar params for single-dispatch and
  grid-sync phase plans without rebuilding bind groups.
- Prepared compiler runners pass through `awaitCompletion: true` for no-readback
  hot-loop timing and watchdog gates.
- Added `pnpm bench` benchmark harness for compiler, CPU reference, and
  orchestration planner timing.
- Added corpus-audit threshold flags and `audit:cuda-120` so CUDA corpus
  coverage baselines fail on regression.
- Corpus audit now reports `compileCodegenOk` / `compileCodegenGaps` and
  treats `webGpuRunnableOk` as a legacy alias, making compile-plan coverage
  distinct from fixture-backed real WebGPU execution.
- Corpus audit now emits preferred `planCompiledOk` / `planCompileGaps` fields
  and `legacyAliases` so platform consumers can avoid runnable/executed naming
  unless browser execution actually happened.
- Source normalization now avoids replacing macro parameters inside member
  properties and lowers simple two-short/one-u32 CUDA unions into bitfield
  views, raising the cuda-samples gate to `305/357` with `51` hard gaps.
- Templated POD records, direct `SharedMemory<T>` values, and 3-argument
  cuRAND init overloads now normalize into the CUDA-lite subset, raising the
  cuda-samples gate to `308/357` with `48` hard gaps.
- Browser WebGPU benchmarks now fail on validation errors and accept optional
  prepared-dispatch ratio thresholds for machine-local perf gates.
- Fixed thread-local arrays now run through CPU reference and WGSL/WebGPU
  lowering.
- Prepared compiler scalar updates now support fixed-topology host-dynamic and
  host-copy plans through per-step uniform updates, with deterministic
  rejection when scalar changes alter plan topology.
- Prepared scalar-update topology checks now use compact WGSL/binding
  signatures instead of JSON stringifying full programs.
- Host-lifted runtime-copy planning now supports resident GPU buffers and rejects
  copies that exceed source or destination capacity before dispatch.
- Unsupported WebGPU execution plans now expose stable `blockers[]` entries
  with `{ kind, code, message }` for platform preflight and audit reporting.
- CUDA corpus audit now skips placeholder identifiers such as `someCount`, so
  pseudocode no longer counts as a hard compiler failure.
- CUDA corpus audit now skips explicit pseudocode solution blocks; CUDA-120
  real-code baseline is `235/240` WebGPU runnable and `0/240` hard failures.
- Added `e2e:webgpu`, a real-browser reference-vs-WebGPU proof for examples,
  grid-sync phases, host runtime copy, host dynamic launch, and prepared resident
  dispatch.
- `e2e:webgpu:corpus` now requires output-verified real WebGPU fixtures from
  CUDA-120, NVIDIA `cuda-samples`, `llm.c`, and LeetCUDA when corpus fixtures
  are required.
- Added `verify:real-world-cuda`, a combined gate that runs pinned corpus
  compile/codegen audit plus exact-kernel browser/WebGPU corpus fixture e2e.
- Real-world WebGPU verification now runs both TS source aliases and built
  package `dist/` exports by default, with `e2e:webgpu:dist` available for a
  focused published-surface smoke.
- Added shared launch-shape diagnostics so platform preflight, CPU reference,
  and WebGPU runners reject invalid grid/block dimensions consistently.
- `createCudaWebGpuExecutionPlan()` now returns `launch` blockers for invalid
  launch shapes before building dispatch plans.
- Host dynamic launch planning now expands parent invocations with CUDA builtin
  coordinates, supports recursive host-dynamic flattening with a depth cap, and
  raises the CUDA-120 WebGPU audit baseline to `239/240`.
- Host dynamic launch planning can pass single-invocation `DevicePool`
  allocation pointers into child pointer params through pool-data aliases and
  base-offset uniforms.
- Host dynamic launch planning can now lift expanded `DevicePool` allocations
  when child launches are order-stable except for pointer base offsets.
- Launched `__device__` functions can now be promoted to child kernels for
  host-lifted dynamic launches, raising the CUDA-120 WebGPU audit baseline to
  `240/240`.
- Host-dynamic WebGPU plans now elide pure parent replay and seed host-planned
  `DevicePool` offsets once through generic storage metadata, avoiding double
  allocation without adding a no-op anchor dispatch.
- Added `cudaLiteFeatureOptionsFromKernelFeatures()` and
  `compileCudaLiteOptionsFromKernelFeatures()` so platforms can feed
  `detectKernelFeatures()` results directly into CUDA-lite compile gates.
- Accepted common C/CUDA integer aliases (`signed`, `unsigned`, `short`,
  `long`, `long long`, `size_t`, fixed-width 32/64-bit aliases, and
  `uintptr_t`) as CUDA-lite `i32`/`u32` spellings for learner kernels.
- CUDA-lite half reference execution now uses BrowserGrad's float16 backing
  array helper, so Node versions without native `Float16Array` still run f16
  unit tests and scalar packing deterministically.
- Added `summarizeCudaWebGpuExecutionPlan()` so platforms can distinguish direct
  WebGPU, host-orchestrated WebGPU, and unsupported plans without misusing the
  conservative lowering plan.
- Compiler corpus, e2e, and benchmark package scripts now run through a locked
  tool wrapper so concurrent invocations cannot import a partially rebuilt
  `dist/` tree.
- Added `compileCudaLiteKernelForWebGpu()` and `cudaLiteWebGpuCompileOptions()`
  for platform code that wants host-orchestrated WebGPU plans without manually
  toggling reference/runtime warning flags.
- CUDA corpus audit output now separates strict direct lowering from total
  WebGPU runnable coverage with `directLoweringOk`, `strictCompileGaps`, and
  `webGpuRunnableOk`.
- Added native WGSL/reference lowering for common CUDA float math builtins:
  `fabsf`, `floorf`, `ceilf`, `roundf`, `truncf`, `sinf`, `cosf`, `tanf`,
  `powf`, `fminf`, and `fmaxf`.
- `__device__` helper functions now accept storage and one-dimensional shared
  pointer params for `float`/`int`/`uint`/`half`, lowering pointer args to a
  compact `{ memory id, base index }` ABI in WGSL and matching address values
  in the CPU reference interpreter.
- Added `createCudaLiteCompilerCache()` and deterministic compile cache keys
  for bounded LRU reuse of parsed/analyzed/lowered/WGSL compiler outputs in
  platform hot paths.
- Prepared WebGPU host-orchestrated runs now reuse a bounded child-kernel
  compile cache during scalar-update replanning, avoiding repeated child
  parse/analyze/lower work in hot loops.
- `runCompiledKernelWebGpu()` and `prepareCompiledKernelWebGpu()` now expose
  host-dynamic expansion/depth caps and validate cap values before planning.
- Host-copy orchestration now supports `cudaMemcpy` and `cudaMemcpyAsync`
  device-to-device copies alongside `cudaMemcpyPeerAsync`.
- CUDA stream/event lifecycle, record, and synchronize calls now parse and run
  as host-managed ordering no-ops for async-copy examples.
