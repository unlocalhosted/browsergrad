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
- Added `createCudaRuntimePlan()`, `createCudaGridSyncPhasePlan()`,
  `createCudaHostDynamicLaunchPlan()`, and `createCudaPeerCopyPlan()` for
  platform/rubric preflight.
