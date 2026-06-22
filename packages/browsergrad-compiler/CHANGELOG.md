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
  performs a host-liftable peer copy.
- Added `createCudaRuntimePlan()`, `createCudaGridSyncPhasePlan()`,
  `createCudaHostDynamicLaunchPlan()`, `createCudaPeerCopyPlan()`, and
  `createCudaWebGpuExecutionPlan()` for platform/rubric preflight.
- Refactored WebGPU execution through an explicit plan interface so native
  dispatch, grid-sync phases, dynamic child launches, and peer-copy lifts share
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
- Browser WebGPU benchmarks now fail on validation errors and accept optional
  prepared-dispatch ratio thresholds for machine-local perf gates.
- Fixed thread-local arrays now run through CPU reference and WGSL/WebGPU
  lowering.
- Prepared compiler scalar updates now support fixed-topology host-dynamic and
  host-peer-copy plans through per-step uniform updates, with deterministic
  rejection when scalar changes alter plan topology.
- Prepared scalar-update topology checks now use compact WGSL/binding
  signatures instead of JSON stringifying full programs.
- Host-lifted peer-copy planning now supports resident GPU buffers and rejects
  copies that exceed source or destination capacity before dispatch.
- Unsupported WebGPU execution plans now expose stable `blockers[]` entries
  with `{ kind, code, message }` for platform preflight and audit reporting.
- CUDA corpus audit now skips placeholder identifiers such as `someCount`, so
  pseudocode no longer counts as a hard compiler failure.
- CUDA corpus audit now skips explicit pseudocode solution blocks; CUDA-120
  real-code baseline is `235/240` WebGPU runnable and `0/240` hard failures.
- Added `e2e:webgpu`, a real-browser reference-vs-WebGPU proof for examples,
  grid-sync phases, host peer copy, host dynamic launch, and prepared resident
  dispatch.
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
  `DevicePool` offsets once, avoiding double allocation while preserving typed
  pool readback.
