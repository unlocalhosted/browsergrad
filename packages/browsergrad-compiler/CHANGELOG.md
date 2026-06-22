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
- Added `pnpm bench` benchmark harness for compiler, CPU reference, and
  orchestration planner timing.
