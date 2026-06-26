# @unlocalhosted/browsergrad-compiler Changelog

## 0.1.0

- Source normalization now folds CuTe `Int<N>{}` launch-context values and
  lowers direct CuTe TN GEMM tensor/tile object graphs into scalar WebGPU matmul.
  It also expands dependent `typename Traits::Arguments` carrier params,
  resolves `constexpr static` trait shapes such as
  `get<0>(decltype(make_shape(...)){})`, and closes the LeetCUDA
  `hgemm_mma_stages_block_swizzle_tn_cute_kernel` and
  `ws_hgemm_naive_cute_kernel` gaps. The LeetCUDA compile/codegen gate is now
  `288/293` with `5` hard gaps.
- Source normalization now lowers vector-valued POD record returns, record-shaped `memcpy` from scalar arrays, return-switch local lambdas, and vector cooperative-group shuffles. This closes the cuda-samples `shfl_intimage_rows` gap and raises the cuda-samples compile/codegen gate to `341/357` with `13` hard gaps.
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
  `planCompiledOk` / `planCompileGaps`, making compile-plan coverage distinct
  from fixture-backed real WebGPU execution.
- Browser corpus e2e now runs `51` exact kernels from pinned CUDA-120,
  NVIDIA `cuda-samples`, `llm.c`, and LeetCUDA sources through real WebGPU in
  both source and dist bundles, with CPU-reference readback comparisons.
- Browser corpus e2e now enforces those real WebGPU fixtures as a no-regression
  floor: `51` total passing fixtures, with per-corpus minimums of CUDA-120 `2`,
  NVIDIA `cuda-samples` `8`, `llm.c` `12`, and LeetCUDA `29`.
- Browser corpus e2e now adds real WebGPU LeetCUDA direct transpose fixtures and
  `llm.c` encoder/cross-entropy-backward transformer fixtures, including a
  regression for local storage-pointer alias dereferences that need WGSL pointer
  helper emission.
- Corpus fixture specs can now pin explicit expected readbacks; the browser gate
  compares both CPU reference and real WebGPU output against those arrays and
  enforces a minimum pinned-output fixture count, currently `22`.
- Corpus audit summaries now expose `compileFeatureProfile` so downstream
  platform code does not confuse feature-full compile/codegen coverage with
  device-portable browser execution.
- Root release verification and CI's Chromium job now run the combined
  real-world CUDA compile/codegen plus real WebGPU fixture gate with WebGPU
  required, npm publish workflows run the same gate before compiler-capable
  packages publish, and `verify:real-world-cuda` now requires WebGPU by default
  unless `--allow-missing-webgpu` is passed for local capability discovery.
- Cooperative groups now lower `cg::binary_partition(tile, predicate)` through
  subgroup predicate masks and lockstep CPU-reference collectives, raising the
  cuda-samples compile/codegen gate to `340/357` with `14` hard gaps.
- C++ template/class object declarations now fail with
  `unsupported-cpp-object-model` instead of generic parse errors, keeping
  dynamic `new`/member-call/object-lifetime gaps explicit without wrong-code
  erasure.
- Source normalization and WGSL emission now lower generic CUDA device
  function-pointer table/param dispatch, device-function local type inference,
  and explicit signed/unsigned index casts. This closes cuda-samples
  `SobelShared` / `SobelTex`, raises the cuda-samples compile/codegen gate to
  `339/357` with `15` hard gaps, and adds a real WebGPU SobelTex corpus
  fixture.
- Source normalization now lowers bounded rank-2 callable tensor views and
  `cuda::shared_memory_mdspan` aliases to pointer/extent/stride primitives,
  raising the cuda-samples compile/codegen gate to `336/357` with `18` hard
  gaps and adding real WebGPU fixtures for mdspan row scaling and shared-tile
  transpose.
- Source normalization now admits scalarized POD-record pointer device helpers
  and expands `&recordPtr[i]` helper arguments into field pointers, closing the
  MonteCarlo `sumReduce` gap and raising the cuda-samples compile/codegen gate
  to `337/357` with `17` hard gaps.
- Added real WebGPU LeetCUDA `float4` fixture coverage for GELU and
  hardshrink vector-pack kernels, plus CPU-reference pointer-view hardening for
  multi-thread vector reinterpret loads.
- CPU reference now models scalar warp-reduction collectives across active
  subgroup lanes. Browser WebGPU subgroup fixtures stay out of the hard gate
  until the current browser environment allows WGSL `subgroups` or a workgroup
  fallback lands.
- Added CUDA graph conditional setter validation/lowering as a host-managed
  scheduler side effect, raising the cuda-samples compile/codegen gate to
  `333/357` with `21` hard gaps.
- Browser corpus fixtures now support per-case numeric tolerances for
  transcendental kernels such as LeetCUDA RoPE while still comparing CPU
  reference and real WebGPU readback.
- CPU reference semantics now preserve C-style integer locals, integer
  division, and remainder behavior so tensor indexing kernels such as
  `llm.c` permute and cross-entropy match WGSL execution.
- Corpus audit now omits legacy `webGpuRunnableOk`, `webGpuTotalOk`, and
  `webGpuCompiledOk` counters from the top-level JSON summary. Deprecated CLI
  flags still map to `planCompiledOk` for compatibility, and
  `deprecatedCompilePlanAliases` documents that mapping.
- Source/context recovery now retries header kernels when fixed local array
  dimensions need reverse translation-unit defines, and side-effect
  canonicalization lowers safe `array[idx++]` standalone statements. This raises
  the cuda-samples gate to `312/357` with `44` hard gaps.
- Source normalization now lowers mutable local scalar references to storage
  elements, raising the cuda-samples gate to `313/357` with `43` hard gaps.
- CUDA-lite now folds `sizeof(type)` for C layout aliases, tokenizes C character
  literals, accepts conditional local pointer initializers, and permits explicit
  pointer casts over arrays, raising the cuda-samples gate to `315/357` with
  `41` hard gaps.
- CUDA-lite now decays fixed C array device-helper params, prunes reachable
  helper `#if/#else` branches before portability checks, accepts initialized
  scalar CUDA vector constants, and lowers `rintf`, raising the cuda-samples
  gate to `316/357` with `40` hard gaps.
- CUDA corpus normalization now evaluates `__CUDACC__` guarded helpers, prunes
  inactive preprocessor branches before helper collection, and specializes
  defaulted kernel template types into referenced device helpers. CUDA warp
  reduction aliases now accept masked forms and lower the value operand, raising
  the cuda-samples gate to `318/357` with `38` hard gaps.
- CUDA surface/cooperative-group intake now lowers storage-backed `surf3Dwrite`,
  removes cooperative-groups `block_tile_memory` scratch carriers, prunes
  inactive typedef branches before alias collection, and carries double-field
  POD records plus explicit f64-to-f32 double atomic compatibility, raising the
  cuda-samples gate to `320/357` with `36` hard gaps.
- Source normalization now avoids replacing macro parameters inside member
  properties and lowers simple two-short/one-u32 CUDA unions into bitfield
  views, raising the cuda-samples gate to `305/357` with `51` hard gaps.
- Templated POD records, direct `SharedMemory<T>` values, and 3-argument
  cuRAND init overloads now normalize into the CUDA-lite subset, raising the
  cuda-samples gate to `308/357` with `48` hard gaps.
- CuTe rank-2 transpose motifs, row-broadcast GEMV tensor views, and malformed
  macro-assignment recovery now normalize into direct CUDA-lite loops, raising
  the LeetCUDA gate to `286/293` with `7` hard gaps.
- CUDA-lite now treats `&const_storage[i]` as a read address until a real write
  boundary, supports conditional local read pointers over storage buffers, and
  emits dynamic shared-memory pointer handles for derived addresses, raising the
  cuda-samples gate to `324/357` with `31` hard gaps.
- CUDA-lite now lowers CUDA helper `lerp(floatN, floatN, t)` as vector math and
  decays fixed local arrays into function-local helper pointer params, raising
  the cuda-samples gate to `326/357` with `29` hard gaps.
- CUDA-lite now supports fixed local pointer arrays that carry shared/local
  scratch addresses through helper calls and dereferences, raising the
  cuda-samples gate to `327/357` with `28` hard gaps.
- Source normalization now converts escaped newline fragments outside strings
  back into real CUDA lines, and stdout no-op analysis accepts pointer/local
  array debug arguments, raising the cuda-samples gate to `328/357` with `26`
  hard gaps.
- Source normalization now scalarizes by-value POD records that carry pointer
  fields, raising the cuda-samples gate to `329/357` with `25` hard gaps.
- Source normalizer synthetic names now avoid WGSL-reserved double-underscore
  prefixes, and the browser fixture suite now executes all four lifted LeetCUDA
  CuTe transpose variants through real WebGPU.
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
  real-code baseline is `235/240` compile/codegen-ok and `0/240` hard failures.
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
  raises the CUDA-120 compile/codegen audit baseline to `239/240`.
- Host dynamic launch planning can pass single-invocation `DevicePool`
  allocation pointers into child pointer params through pool-data aliases and
  base-offset uniforms.
- Host dynamic launch planning can now lift expanded `DevicePool` allocations
  when child launches are order-stable except for pointer base offsets.
- Launched `__device__` functions can now be promoted to child kernels for
  host-lifted dynamic launches, raising the CUDA-120 compile/codegen audit
  baseline to `240/240`.
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
  compile/codegen coverage with `directLoweringOk`, `strictCompileGaps`, and
  `planCompiledOk`.
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
