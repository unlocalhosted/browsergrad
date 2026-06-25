# @unlocalhosted/browsergrad-compiler

CUDA-lite compiler for browser-native GPU labs.

The default path is small and inspectable:

```text
CUDA-lite source -> BrowserGrad Kernel IR -> WGSL -> WebGPU
                     \-> lockstep CPU reference
```

This package is independent of Pyodide. It uses
`@unlocalhosted/browsergrad-kernels` for WGSL dispatch.

Compatibility vocabulary:

- **native lowering**: CUDA primitive lowers directly to WGSL/WebGPU.
- **GPU polyfill lowering**: CUDA primitive lowers into one or more real WebGPU passes.
- **CPU reference**: correctness and teaching trace, not primary runtime.
- **unsupported diagnostic**: no honest lowering exists yet.

Low-level extension boundaries live in
[`docs/platform/cuda-lite-compiler-architecture.md`](../../docs/platform/cuda-lite-compiler-architecture.md).

## Quick Start

```ts
import { createDevice, detectKernelFeatures } from "@unlocalhosted/browsergrad-kernels";
import {
  compileCudaLiteOptionsFromKernelFeatures,
  createCudaLiteCompilerCache,
  compileCudaLiteKernelForWebGpu,
  compileCudaLiteKernel,
  createCudaWebGpuExecutionPlan,
  prepareCompiledKernelWebGpu,
  runCompiledKernelReference,
  runCompiledKernelWebGpu,
  summarizeCudaWebGpuExecutionPlan,
} from "@unlocalhosted/browsergrad-compiler";

const source = `
__global__ void saxpy(const float* x, float* y, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}`;

const device = await createDevice();
const features = await detectKernelFeatures(device);
const compilerCache = createCudaLiteCompilerCache({
  maxEntries: 128,
  compileOptions: compileCudaLiteOptionsFromKernelFeatures(features),
});
const compiled = compilerCache.compile(
  source,
  {
    workgroupSize: [8, 1, 1],
  },
);
console.log(compiled.loweringPlan.canRunOnGpu);
const input = {
  buffers: {
    x: new Float32Array([1, 2, 3, 4]),
    y: new Float32Array([10, 20, 30, 40]),
  },
  scalars: { a: 2, n: 4 },
};
const launch = { gridDim: [1, 1, 1], blockDim: [8, 1, 1] } as const;

const reference = runCompiledKernelReference(compiled, input, launch);
const plan = createCudaWebGpuExecutionPlan(compiled, input, launch);
const status = summarizeCudaWebGpuExecutionPlan(plan);
console.log(status.canRunOnWebGpu, status.mode, status.kind);
const gpu = await runCompiledKernelWebGpu(device, compiled, input, launch);
```

Examples live in `examples/`: SAXPY, guarded map, and shared-memory tiled
matmul. The emitted WGSL is intentionally inspectable so labs can show source,
bindings, workgroup size, shared memory, and barriers directly.
Use `compileCudaLiteOptionsFromKernelFeatures()` to pass browser/device facts
from `detectKernelFeatures()` into the compiler. This keeps `shader-f16`,
subgroup, and compatibility-mode gates aligned with the runtime instead of
duplicating string flags in platform code.
Use `summarizeCudaWebGpuExecutionPlan()` for platform readiness UI. A kernel can
have `compiled.loweringPlan.canRunOnGpu === false` because it contains runtime
gaps, while a host-orchestrated WebGPU plan can still run it through real GPU
passes.
Use `compileCudaLiteKernel()` when you need strict direct-lowering diagnostics.
Use `compileCudaLiteKernelForWebGpu()` when the platform intends to run
host-orchestrated WebGPU plans such as grid-sync phases, host runtime copy, or
host-lifted dynamic launches.
Fixed thread-local arrays lower to WGSL function arrays and CPU-reference typed
arrays, so small per-thread scratch patterns do not need shared memory.
Use `createCudaLiteCompilerCache()` for platform/rubric hot paths that compile
the same source repeatedly. It uses deterministic option keys plus bounded LRU
eviction; set `maxEntries: 0` to preserve the same call shape while disabling
caching.
Common CUDA float math helpers lower natively in both WGSL and CPU reference:
`sqrt`, `sqrtf`, `expf`, `logf`, `fabsf`, `floorf`, `ceilf`, `roundf`,
`truncf`, `sinf`, `cosf`, `tanf`, `tanhf`, `coshf`, `powf`, `fminf`,
`fmaxf`, `fma`, `fmaf`, `rsqrtf`, `__expf`, `__logf`, `__saturatef`, and
`__fdividef`. Integer CUDA helpers include `__clz`, `__mul24`, and `__umul24`;
`__usad4` lowers as byte-lane sum-of-absolute-differences plus add.
`assert(expr)` is accepted as a no-op runtime check in browser kernels.
Scalar half helpers lower behind `shader-f16`: `__half2float`, `__float2half`,
`hexp`, `__hadd`, `__hsub`, `__hmul`, `__hdiv`, `__hneg`, `__hfma`,
`__hmin`, `__hmax`, `__heq`, `__hne`, `__hgt`, `__hge`, `__hlt`, and
`__hle`.
CUDA/C named constants such as `INFINITY`, `NAN`, `FLT_MAX`, `M_PI`,
`cudaMemcpyDeviceToDevice`, and stream/event flag values lower through the same
analyzer, CPU-reference, and WGSL path.
CUDA cache-hint memory builtins `__ldcs` and `__stcs` lower as ordinary storage
pointer loads/stores; BrowserGrad preserves semantics and intentionally ignores
the cache placement hint on WebGPU.
CUDA 2D float texture references and `cudaTextureObject_t` kernel params lower
to named WebGPU `texture_2d<f32>` bindings. `tex2D<float>(tex, x, y)` uses the
same CPU-reference and WGSL path; true bindless handles, layered/3D textures,
samplers, and non-float texture formats remain explicit compatibility gaps.
CUDA vector storage types `float2/3/4`, `int2/3/4`, and `uint2/3/4` lower
through a scalar storage ABI with `make_*` constructors and lane member access.
This keeps caller buffers as ordinary typed arrays while still emitting vector
values inside WGSL and the CPU reference interpreter.
CUDA/C++ pointer casts such as `reinterpret_cast<float4 *>(&x[i])` lower as
typed storage views over that same scalar ABI. Macro spellings like `FLOAT4(x)`
work when they expand to the standard cast/index idiom; BrowserGrad models the
memory view, not the macro name.
Simple C++ intake accepts scalar/vector `typedef` and `using` aliases,
`constexpr` integer expressions in array dimensions and template integer
arguments, `static` kernel qualifiers, late `__launch_bounds__` placement, and
`static_assert` statements. This is bounded CUDA/C++ normalization, not full
C++ template compatibility.
WMMA fragments are accepted as scalarized cooperative-matrix primitives:
`wmma::fragment`, `wmma::fill_fragment`, `wmma::load_matrix_sync`,
`wmma::mma_sync`, and `wmma::store_matrix_sync` lower through CPU reference and
WGSL for small educational matrix tiles. The fragment surface includes
`wmma::precision::tf32`, `.num_elements`, and `.x[index]` lane access. This
preserves learner-visible semantics and browser execution; it does not claim
Tensor Core performance or lane-accurate NVIDIA fragment layout.
Bounded integer template defaults such as `template <const int N = 256>` are
accepted on kernels and device helpers when the default is an integer constant
expression. Functional scalar casts such as `float(i)` lower to CUDA-lite casts.
Named CUDA constants include `warpSize` and `NULL`.
Cooperative-groups syntax supports both member calls and namespace calls:
`block.sync()`, `tile.shfl_down(value, offset)`, `cg::sync(block)`, and
tile-scoped `cg::reduce(tile, value, cg::plus<T>{})` / `cg::greater<T>{}`.
Common C/CUDA integer spellings are accepted for learner kernels: `signed`,
`unsigned`, `short`, `long`, `long long`, `long long int`, `size_t`, `int32_t`, `uint32_t`,
`int64_t`, `uint64_t`, and `uintptr_t`. Current WebGPU lowering maps them onto
WGSL `i32`/`u32`; true 64-bit integer semantics remain a future polyfill or
backend capability.
CUDA `double` remains rejected by default because WebGPU/WGSL has no native f64.
For educational kernels where f32 precision is acceptable, pass
`{ f64Mode: "f32" }`; the compiler emits `f64-lowered-to-f32`, uses f32
storage/WGSL/reference ABI, and keeps the precision loss visible to rubrics.
Dynamic extern shared memory supports common CUDA qualifier spellings:
`extern __shared__ T name[]`, `extern T __shared__ name[]`, `volatile __shared__`
fixed arrays, and trailing fixed dimensions such as
`extern __shared__ T name[][N]` where launch metadata supplies the leading
extent.

For hot WebGPU paths, pass caller-owned buffers through `residentBuffers` and
set `readback: []`. This keeps data on GPU across compiler-dispatched kernels;
use `readWgslStorageBuffer()` from `@unlocalhosted/browsergrad-kernels` at the
actual materialization boundary.
Readback names are logical compiler names: use `"dp"` for a `DevicePool* dp`
input, not its internal WGSL backing buffer name.

If launch shape and bindings stay fixed across iterations, use
`prepareCompiledKernelWebGpu()` once. It prebuilds the WebGPU sequence, pipelines,
and bind groups, then reruns over resident buffers without per-iteration setup:

```ts
const prepared = await prepareCompiledKernelWebGpu(device, compiled, {
  buffers: {},
  residentBuffers: { x, y },
  scalars: { a: 2, n: 4 },
  readback: [],
}, launch);

await prepared.run();
await prepared.run({ readback: [] });
await prepared.run({ scalars: { a: 4 }, readback: [], awaitCompletion: true });
prepared.destroy();
```

Prepared host-orchestrated plans keep a bounded child-kernel compile cache by
default. Tune with `childCompileCacheMaxEntries`, or pass `0` when a caller
wants no cache for deterministic instrumentation.
Use `maxHostExpandedParentInvocations` and `maxHostDynamicLaunchDepth` on
`runCompiledKernelWebGpu()` / `prepareCompiledKernelWebGpu()` to bound
host-lifted dynamic launch expansion in learner-facing hot paths.
Prepared scalar updates are supported when the WebGPU plan topology remains
fixed. That includes host-orchestrated dynamic launch / runtime-copy plans whose
step count, dispatch counts, storage aliases, and WGSL programs do not change.
Topology-changing scalar updates fail with a deterministic compiler diagnostic.
Use `awaitCompletion: true` when a no-readback hot loop must measure GPU
completion instead of JS command submission.

## CUDA Memory Pools

The compiler lowers simple bump allocators to real WebGPU atomics:

- `DevicePool* pool` with `streamOrderedAllocate(pool, size)` / `deviceAllocate(pool, size)`.
- external device pools referenced as `deviceAllocate(&g_pool, size)` and supplied
  through `memoryPools.g_pool`.
- raw pool form `deviceAllocate(poolBase, offset, poolSize, size)` where `poolBase`
  is a pointer parameter and `offset` is an integer pointer counter.

`DevicePool*` inputs use `memoryPools`:

```ts
const input = {
  buffers: { out: new Float32Array(2) },
  memoryPools: {
    pool: { data: new Uint32Array(2), offset: new Uint32Array([0]) },
  },
  scalars: { n: 2 },
};
```

Pool pointers are byte offsets with `0` as null. Casted accesses such as
`((float*)ptr)[0]` lower to raw pool words for `DevicePool*`, and to the typed
base buffer for raw pointer pools. This is a teaching-grade CUDA allocator
primitive, not a course-specific shim.

## Dynamic Parallelism

Device-side launches parse into Kernel IR. By default they remain compile-time
runtime gaps. For CPU teaching traces, opt in:

```ts
compileCudaLiteKernel(source, {
  kernelName: "parent",
  referenceDynamicParallelism: true,
});
```

`runCompiledKernelReference` executes child kernels against shared buffers and
memory pools.

`runCompiledKernelWebGpu` can host-lift conservative child launches into a real
WebGPU dispatch sequence when parent invocations, launch branches, child block
sizes, pointer args, and scalar args are host-evaluable. Parent invocations are
expanded with CUDA builtin coordinates up to a deterministic cap, recursive
dynamic launches are flattened up to a depth cap, and inactive host-evaluable
launch branches fall back to single dispatch. Named `DevicePool*` arguments
alias their backing pool data/offset bindings, `DevicePool` allocation pointers
can be passed to child pointer params when offsets are known, and positive
pointer offsets such as `out + 1` lower to generated base-offset uniforms so
WebGPU bindings stay whole-buffer and alignment-safe. Expanded parent
allocations are liftable when child launches are order-stable except for pointer
base offsets. Pure parents with host-planned pool allocations are not replayed;
their pool offsets are seeded once before child dispatch so allocation state does
not advance twice. Inspect this before dispatching:

```ts
const plan = createCudaHostDynamicLaunchPlan(compiled, input, launch);
console.log(plan.supported, plan.reason);
```

Order-sensitive parent-side pool allocations, device-derived launch args,
unknown branch guards before a launch, negative pointer offsets, and parent side
effects after launch stay reference-only so GPU output is never silently wrong.

## CUDA Runtime Reference Calls

Some runtime orchestration calls have CPU reference truth before GPU lowering.
Runtime copies are byte-accurate over modeled buffers/pools when explicitly
enabled:

```ts
compileCudaLiteKernel(source, {
  referenceCudaRuntime: true,
});
```

`cudaMemcpy`, `cudaMemcpyAsync`, and `cudaMemcpyPeerAsync` copy bytes inside the
reference interpreter. CUDA stream/event lifecycle, record, and synchronize
calls are modeled as host-managed ordering points; they do not create browser
CUDA streams, but they let common async-copy examples compile and run honestly.
`runCompiledKernelWebGpu` can host-lift conservative device-to-device copies
into a typed WebGPU copy dispatch when the call is
single-invocation guarded, source/destination are named `Float32Array`,
`Int32Array`, `Uint32Array`, or matching resident GPU buffers, offsets are
non-negative and host-evaluable, byte count is element-aligned, and the copy
fits both buffers. The same copy lift can compose after a host-lifted child
dispatch. Mixed buffer types, pools, host/device transfer kinds,
device-derived counts, and side effects after copy remain reference-only.

## Cooperative Grid Sync

`cg::grid_group::sync()` has CPU reference truth and a WebGPU phase-splitting
path for safe top-level uniform barriers:

```ts
compileCudaLiteKernel(source, {
  referenceGridSync: true,
});
```

The CPU reference scheduler runs all blocks in lockstep at grid barriers, so
block-0 reduction patterns see writes from every block. WebGPU runs safe
top-level uniform `grid.sync()` as multiple dispatch phases over shared GPU
buffers. Pure launch-derived locals are replayed in later phases, and shared
memory reuse is allowed when the phase rewrites it before reading. Non-uniform
sync, non-replayable locals, and shared-memory read-before-rewrite stay
reference-only.

Use `createCudaLoweringPlan(diagnostics)` and `describeCudaDiagnostic()` to group
compatibility gaps by semantic family instead of raw parser messages.
Use `createCudaRuntimePlan(compiled)` to see which kernels need host
orchestration (`device-launch`, `device-sync`, `runtime-copy`, `grid-sync`) before
trying WebGPU single-dispatch execution.
Use `createCudaLaunchValidationDiagnostics(launch, compiled.ir.workgroupSize)`
or `validateCudaKernelLaunch()` to preflight launch shape before selecting CPU
reference or WebGPU execution. Reference and WebGPU runners use the same
validator, and `createCudaWebGpuExecutionPlan()` reports the same failures as
`launch` blockers. Bad grid/block dimensions fail with the same diagnostic
codes across planning and execution.
Use `createCudaWebGpuExecutionPlan(compiled, input, launch, { compileKernel })`
to inspect the exact executable WebGPU plan before running: `single-dispatch`,
`grid-sync-phases`, `host-dynamic-launch`, or `host-copy`. The runner uses
the same plan interface internally, so platform preflight and execution share
one source of truth. Unsupported plans include both a human `reason` and
machine-readable `blockers[]` entries with `{ kind, code, message }`; route
platform UI/rubric behavior from blocker codes instead of parsing reason text.
For runtime-gap kernels, pass a `compiled` object from
`compileCudaLiteKernelForWebGpu()` and use that helper as `compileKernel` so
dynamic children receive the same host-orchestration semantics.
Use `normalizeCudaWebGpuReadbackNames(compiled, names)` if platform code needs
to inspect the internal WGSL storage readbacks that correspond to logical
compiler names.

## Browser Testing

```bash
pnpm test:browser
pnpm test:browser:open
pnpm verify:compiler
pnpm e2e:webgpu
pnpm e2e:webgpu:dist
pnpm e2e:webgpu:corpus -- --require-webgpu
pnpm verify:real-world-cuda -- --skip-fetch --require-webgpu
```

`test:browser:open` keeps Chromium open for inspection; quit with `q`. If the
Vitest browser watch rerun path reports an orchestrator-session error, restart
the command instead of trusting that rerun.

`verify:compiler` is the local release gate: build, typecheck, lint, unit tests,
source-normalizer test, and corpus-audit self-test. Real WebGPU/corpus hardware
proof stays in `verify:real-world-cuda`.

`e2e:webgpu` launches a real browser and runs example kernels plus runtime
orchestration probes against both `runCompiledKernelReference()` and real
WebGPU. It covers SAXPY, guarded map, tiled matmul, grid-sync phases, host
runtime copy, host dynamic launch, and prepared resident dispatch.
`e2e:webgpu:corpus` additionally requires fixture-backed corpus kernels loaded
from pinned local corpora under `/tmp` and executes them through real WebGPU
with readback comparisons. Required fixture names currently cover CUDA-120
`vectorAddKernel`, NVIDIA `cuda-samples` `vectorAdd`, `llm.c` `add_bias`,
`llm.c` `set_vector`, and LeetCUDA `elementwise_add_f32_kernel`.
Fixture source is emitted through the same corpus-audit normalization path used
for full-corpus compile/codegen counts, so helper/context handling does not
silently diverge between audit and browser execution gates.
Pass `-- --bundle dist` or use `e2e:webgpu:dist` to run the same browser proof
against built package exports instead of TS source aliases.
Compiler e2e, corpus, and benchmark package scripts use
`scripts/run-cuda-lite-tool.mjs`, which locks build + tool execution so parallel
invocations cannot import a partially rebuilt `dist/` tree. The wrapper builds
`browsergrad-kernels` before `browsergrad-compiler` so dist-bundle browser gates
exercise fresh package output.
`verify:real-world-cuda` is the combined truth gate: it runs the pinned
real-world compile/codegen audit and then the exact-kernel browser/WebGPU corpus
fixture e2e against both source aliases and built dist exports by default. Use
`--bundle src`, `--bundle dist`, or `--bundle both` to choose the browser bundle.

## Corpus Audit

```bash
pnpm --filter @unlocalhosted/browsergrad-compiler audit:corpus -- /path/to/cuda-corpus --expect-webgpu-min 10
pnpm --filter @unlocalhosted/browsergrad-compiler audit:cuda-120
pnpm --filter @unlocalhosted/browsergrad-compiler audit:real-world-cuda
```

`audit:corpus` extracts CUDA-shaped kernels from Markdown/CUDA/C++ files and
reports compile/lowering coverage: strict single-dispatch WGSL compile,
host-lifted WebGPU plan compile, CPU reference fallbacks, and hard gaps. It does
not execute every external corpus kernel because most corpora do not carry
portable launch fixtures and expected outputs. Failure details include WebGPU
blocker kind/code/message. Threshold flags make corpus compile coverage
regressions fail fast instead of living only in docs.
`directLoweringOk` means strict one-pass WGSL compile. `compileCodegenOk` means
direct WGSL compile or host-orchestrated WebGPU plan compile, such as grid-sync
phases and dynamic launch lifts. `webGpuRunnableOk` remains as a legacy alias
for `compileCodegenOk`; new consumers should not treat it as output-verified
execution. Prefer `executionTierCounts`: `compileCodegenOnlyOk`,
`fixtureBackedExecutedOk`, `browserWebGpuExecutedOk`, and `outputVerifiedOk`.
Real execution proof lives in fixture-backed tests such as
`pnpm --filter @unlocalhosted/browsergrad-compiler test:browser` and
`scripts/e2e-cuda-lite-webgpu.mjs`.

## Performance Harness

```bash
pnpm --filter @unlocalhosted/browsergrad-compiler bench -- --markdown /tmp/bg-cuda-lite-bench.md
pnpm --filter @unlocalhosted/browsergrad-compiler bench:browser -- --bundle dist --markdown /tmp/bg-cuda-lite-webgpu-bench.md
```

The harness emits stable JSON timing for compile, CPU reference, dynamic-launch
planning, and runtime-copy planning paths. Use
`--expect-median-max benchmark=ms[,benchmark=ms]` and
`--expect-p95-max benchmark=ms[,benchmark=ms]` only on pinned machines; no fixed
threshold is portable across laptops, browsers, and CI runners. Without
threshold flags, use median/p95 deltas across commits to catch regressions
before promoting platform perf rubrics.
The browser harness launches Chromium through Playwright and compares one-shot
resident-buffer dispatch against `prepareCompiledKernelWebGpu()` hot-loop
dispatch. Pass `--require-webgpu` when CI should fail instead of reporting a
skipped WebGPU bench. When WebGPU is available, benchmark validation failures
exit nonzero. Use `--expect-prepared-ratio-max N` and
`--expect-prepared-scalar-ratio-max N` for machine-local perf regression gates
that compare prepared median time against one-shot median time.
