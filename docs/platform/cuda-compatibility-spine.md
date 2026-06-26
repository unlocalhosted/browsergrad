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
- `createCudaLaunchValidationDiagnostics(launch, workgroupSize)` reports
  launch-shape errors before execution. `validateCudaKernelLaunch()` throws the
  same diagnostics. `createCudaWebGpuExecutionPlan()` returns `launch` blockers
  for the same failures, and reference/WebGPU runners share this validator.
- `createCudaWebGpuExecutionPlan(compiled, input, launch, { compileKernel })`
  returns the exact executable WebGPU plan kind and sequence steps:
  `single-dispatch`, `grid-sync-phases`, `host-dynamic-launch`, or
  `host-copy`. Unsupported plans keep a human `reason` and expose
  machine-readable `blockers[]` with `{ kind, code, message }`, so platform
  preflight can route failures without parsing prose. Platform preflight should
  use this interface instead of duplicating runner heuristics.
- `prepareCompiledKernelWebGpu(device, compiled, input, launch)` prepares the
  same executable WebGPU plan once and reruns it over resident buffers. Use it
  for hot loops with fixed launch shape and bindings; scalar params can change
  when the execution-plan topology stays identical.
- `normalizeCudaWebGpuReadbackNames(compiled, names)` maps logical compiler
  readback names, such as `DevicePool* dp` -> `dp`, to internal WGSL storage
  bindings. Platform code should not depend on backing buffer names.

Rule: do not add assignment-specific fixes. Add semantic primitives, reference
truth, WGSL lowering, browser tests, and corpus audit evidence.
For file-level boundaries and extension order, see
[CUDA-lite Compiler Architecture](./cuda-lite-compiler-architecture.md).

Current corpus gate:

```sh
pnpm --filter @unlocalhosted/browsergrad-compiler audit:cuda-120
pnpm --filter @unlocalhosted/browsergrad-compiler audit:real-world-cuda
pnpm --filter @unlocalhosted/browsergrad-compiler verify:real-world-cuda -- --skip-fetch
```

- `AdepojuJeremy/CUDA-120-DAYS--CHALLENGE` audit: `225/240` real code-kernel
  definitions compile as strict direct-lowering WGSL/WebGPU
  (`directLoweringOk`). Another `15/240` compile into host-orchestrated WebGPU
  plans (`grid-sync-phases` and `host-dynamic-launch`), for `240/240`
  `compileCodegenOk`. `0/240` remain
  reference-only and `0/240` remain hard gaps after filtering docs/pseudocode.
- Corpus audit is compile/lowering evidence, not fixture execution for every
  external kernel. `planCompiledOk` is preferred for compile/codegen evidence.
  Top-level legacy `webGpuRunnableOk` / `webGpuTotalOk` counters are omitted
  from audit JSON because they sounded like execution proof. Fixture-backed
  execution belongs in browser tests and `scripts/e2e-cuda-lite-webgpu.mjs`.
- `referenceFallbackOk` is `15/240`: kernels whose semantics are understood by
  CPU reference or host/WebGPU orchestration. `referenceOnlyOk` is stricter and
  excludes kernels with host-orchestrated WebGPU plan coverage; current baseline
  is `0/240`.
- Real-world no-regression gate:
  `NVIDIA/cuda-samples@b7c5481` must stay at `357` kernel definitions, `>=341`
  compile/codegen-ok, and `<=13` hard gaps;
  `karpathy/llm.c@f1e2ace` must stay at `148` kernel definitions, `>=148`
  compile/codegen-ok, and `0` hard gaps;
  `xlite-dev/LeetCUDA@c5dde9a` must stay at `293` kernel definitions, `>=289`
  compile/codegen-ok, and `<=4` hard gaps. The aggregate gate also verifies
  CUDA-120 at its pinned commit.
- Corpus audits now emit `executionTierCounts` plus
  `compileFeatureProfile` and `deprecatedCompilePlanAliases` so platform code can distinguish
  compile/codegen coverage from fixture-backed browser execution and
  output-verified readback without parsing prose. Current compile audits assume
  `shader-f16`, `subgroups`, and `f64Mode: "f32"` compatibility lowering;
  fixture-backed browser e2e is the device-portability proof.
- `verify:real-world-cuda` is the combined hardware-backed gate: it runs the
  pinned full-corpus compile/codegen audit, then runs exact external corpus
  fixtures through real Chromium/WebGPU with output comparison. Missing WebGPU
  fails by default; use `--allow-missing-webgpu` only for read-only capability
  discovery on machines without browser GPU support. Pinned corpus checkouts
  must match the expected commit and have clean git status.
- Recent semantic lifts: `DevicePool*` bump allocation, raw pointer pool allocation
  with integer offset counters, casted pool pointer reads/writes, WebGPU atomic
  offset updates, DevicePool aliasing across host-lifted child launches,
  CUDA cache-hint memory builtins lowered as plain storage pointer memory ops,
  generic unary pointer dereference lvalues, alias-preserving vector member
  writes, `tex2DLod` / `tex1Dfetch` texture aliases, guarded `surf2Dread`,
  scalarized CUDA vector storage views for `float2/3/4`, `int2/3/4`, and
  `uint2/3/4`, simple C++ alias and constexpr integer intake, local quoted
  header context for corpus audits, CUDA `static` kernel qualifiers, modeled
  device-pointer `atomicAdd` helper dispatch through storage/shared buffer ids,
  same-name template value fallback propagation, dynamic extern shared-memory
  context from translation units and device helpers,
  and late `__launch_bounds__` placement, plus cooperative-groups namespace
  call forms such as `cg::sync(block)` and `cg::reduce(tile, value, op)`,
  generic `thread_group` helper-parameter lowering with block/tile metadata,
  adjacent C string literal intake, scalar `std::size_t`/`auto` declarations,
  scalar brace constructors such as `__half{expr}`, and driver API
  `CUtexObject` / `CUsurfObject` texture-surface aliases, texture-handle
  device-helper params, and generic tile-reduce helper params,
  C++ `reinterpret_cast<T*>` / `static_cast<T*>` pointer casts for typed
  scalarized storage views such as `FLOAT4(x)` and local pointer aliases,
  bounded integer template defaults on kernels/device helpers, multi-dimensional
  shared-memory address lowering for `__cvta_generic_to_shared` and helper
  pointer params, vector reinterpret memory-view helpers through the device
  pointer ABI, semantic `blockReduce<warpReduce*>` lowering, safe named
  constants such as `warpSize` and `NULL`,
  fast CUDA math/bit intrinsics such as `__saturatef`, `__fdividef`, `__clz`,
  `__mul24`, and `__umul24`,
  CUDA 2D float texture-object params / `tex2D<float>` lowering, typed
  texture-vector reads such as `tex2D<float4>` / `tex2D<uchar4>`, and
  multi-channel WebGPU texture uploads, plus atlas-backed `tex1D`,
  `tex2DLayered`, `tex3D`, and `texCubemap` point-sampling over WebGPU
  `texture_2d`,
  templated `surf2Dread<T>` return-form loads, scalarized `surf1Dwrite` /
  `surf2DLayeredwrite` surface writes, vector min/max overloads, CUDA vector
  assignment chains, and POD-style vector field aliases,
  positive pointer-offset child launches via generated base-offset uniforms,
  expanded order-stable DevicePool allocation launches, launched `__device__`
  child functions, and conservative host-lifted peer copies through a typed
  WebGPU copy dispatch. C-style assignment-chain statements lower into ordered
  WGSL writes, and WGSL output alpha-renames CUDA source symbols such as
  `array`, `var`, and builtin-shadowing locals without changing public input
  names. Translation-unit `__shared__` scratch arrays are injected into kernels
  that reference them, and simple preprocessor branches inside selected kernel
  bodies are pruned with C `#if`/`#ifdef` semantics without changing corpus
  kernel census or included-header context.
  Homogeneous POD structs with two to four scalar fields lower to CUDA vector
  aliases before parse, safe numeric object macros fold into code without
  breaking parameter/local shadows, local const/template integer expressions
  feed later fixed array dimensions, and scalar bitwise compound assignments
  (`&=`, `|=`, `^=`) lower through parser, reference, and WGSL.
  Scalarized POD record values/storage params, constant-record reachability,
  macro-sized record arrays, DirectX-style float-vector fields, and C-style
  array typedef vector aliases now desugar into existing scalar/vector IR.
  CUDA inverse trig aliases and CUDA vector `length(v)` helpers lower to WGSL
  math over scalar/vector primitives. Conditional helper pointer args now use
  C-style pointer truthiness in analyzer, reference, and WGSL lowering.
  `Packed128<half|bf16>` pointer views over byte-backed shared memory now
  scalarize into lane-addressed shared arrays without leaking source alias names.
  Function-pointer-like template symbols propagate through host wrappers into
  launched kernels so device calls bind concrete helpers.
  C++ std math aliases such as `std::isinf` and
  `std::numeric_limits<float>::infinity()` lower to generic CUDA-lite
  intrinsics/constants.
  Dynamic extern shared-memory inference includes bf16 storage, late
  `extern T __shared__ name[]` qualifier order, and trailing fixed dimensions
  such as `extern __shared__ T name[][N]`; scalar 128-bit cache-load
  assignments expand into lane-wise stores for 2/4-byte scalar pointer packs.
  Custom CUDA-vector `cg::reduce` calls lower through scalar subgroup
  shuffle-XOR loops backed by user device merge helpers.
  Device helper calls can now pass local scalar out-params through WGSL
  function pointers, CUDA opaque RNG handles parse as u32-backed handles,
  function-pointer typedefs normalize as opaque handles, scalar template
  count/index params fall back to numeric integer types, C `frexp` writes local
  exponent outputs in reference and WGSL, and `typename vecN<T>::Type` carrier
  aliases normalize into CUDA vector types. Mutable integer C++ reference
  params used by CUDA atomics lower conservatively into existing pointer helper
  ABI, pointer-form `atomicExch` / `atomicCAS` dispatch through storage/shared
  helper ids, and CPU reference no longer treats shared scalar assignment as
  pointer rebinding. Corpus intake now treats CUDA `short`/`uint16_t` pointer
  bases as integer-compatible helper params and admits scalar device helpers
  that use parser-supported `static_cast<T>` expressions.
  Fixed thread-local arrays lower to per-thread WGSL function arrays and CPU
  reference typed arrays. Source normalization now also supplies conservative
  block-size defaults for unresolved launch-bound template value params and
  inlines simple device helper wrappers that forward to CUDA atomics or pointer
  stores, so learner kernels can keep small correctness wrappers without
  requiring a broader device-pointer ABI. Inline PTX now has a shared classifier
  and multi-output operand ABI for `ldmatrix` and `mma.sync.m16n8k16`; v0 lowers
  these as deterministic register-carrier semantics in CPU reference and WGSL,
  while full lane/layout-accurate tensor-core simulation remains a later
  semantic lift. CUDA reciprocal intrinsic `__frcp_rn` lowers through the
  shared intrinsic table, closing the post-PTX LeetCUDA flash-attention math
  gap without repo-specific matching. Shared-memory pipeline template params
  such as stage count, padding, and warp-swizzle flags get conservative
  defaults only inside shared/pipeline contexts, closing one more staged-matmul
  frontend gap without path-specific logic.
- Hot-loop dispatch can keep both caller buffers and compiler-generated
  execution sequences resident: `residentBuffers` avoids upload/readback churn,
  and prepared compiler/WebGPU runners avoid rebuilding pipelines and bind
  groups between iterations. Prepared single-dispatch and grid-sync phase plans
  can update scalar params without reprepare. Host-orchestrated dynamic launch /
  runtime-copy plans can also update scalar params when step count, dispatch counts,
  aliases, and WGSL programs remain unchanged; topology-changing updates fail
  before dispatch. No-readback prepared runs can opt into `awaitCompletion: true`
  when timing gates or watchdogs need real GPU completion instead of JS command
  submission.
- Device-side launches now parse into IR and can run in CPU reference when
  `referenceDynamicParallelism` is enabled, or when platforms compile through
  `compileCudaLiteKernelForWebGpu()`. WebGPU can host-lift conservative child
  launches into a multi-dispatch sequence when parent invocations, launch
  branches, child block sizes, pointer args, and scalar args are host-evaluable.
  Parent invocations expand with CUDA builtin coordinates up to a cap, recursive
  launches flatten up to a depth cap, and inactive host-evaluable launch
  branches use single dispatch. DevicePool pointer params alias their pool
  data/offset bindings, single-invocation DevicePool allocation pointers can be
  passed to child pointer params, and positive pointer-offset args lower to
  base-offset uniforms. Pure parents with host-planned pool allocations are
  elided and their pool offsets seeded once before child dispatch; parents with
  other side effects plus pool allocation are rejected instead of replayed.
  Concurrent parent-side pool allocations, unknown branch guards before launch,
  negative pointer offsets, device-derived launch args, and parent side effects
  after launch remain reference-only.
- CUDA runtime calls such as `cudaDeviceSynchronize` and `cudaMemcpyPeerAsync`
  classify as runtime orchestration gaps. Standalone `cudaDeviceSynchronize()`
  is a WebGPU-safe no-op because dispatch completion is host-managed. Peer
  copies can run in CPU reference with `referenceCudaRuntime`, or through
  `compileCudaLiteKernelForWebGpu()` for WebGPU planning. WebGPU can host-lift
  single-invocation guarded typed buffer copies when source,
  destination, offsets, and byte count are host-evaluable. Typed-array and
  resident-buffer copies are capacity-checked before dispatch. Mixed types,
  pools, device-derived counts, or side effects after copy remain reference-only.
- Cooperative `grid.sync()` can run in CPU reference with `referenceGridSync`,
  or through `compileCudaLiteKernelForWebGpu()` for WebGPU planning. Safe
  top-level uniform `grid.sync()` also runs on real WebGPU as multiple dispatch
  phases over shared GPU buffers. Pure launch-derived locals are
  replayed in later phases, and shared memory can be reused after sync when the
  phase rewrites it before any read. Non-uniform sync, non-replayable private
  locals crossing phases, or shared-memory read-before-rewrite remain
  reference-only.
- Remaining failures group cleanly: dynamic parallelism/runtime launches and
  cooperative groups/grid sync.
- Use `--limit N` to cap printed failures and `--details`/`--json` to emit
  `{ summary, failures }` with `webGpuPlanLiftBlockerKind`,
  `webGpuPlanLiftBlockerCode`, and `webGpuPlanLiftBlocker` values. Use
  threshold flags such as `--expect-plan-compiled-min`,
  `--expect-reference-only-max`, and `--expect-hard-fail-max` so feature triage
  is grounded in compile/codegen regression gates instead of prose-only notes.

Current real-browser e2e gate:

```sh
pnpm --filter @unlocalhosted/browsergrad-compiler e2e:webgpu
pnpm --filter @unlocalhosted/browsergrad-compiler e2e:webgpu:dist
pnpm --filter @unlocalhosted/browsergrad-compiler e2e:webgpu:corpus -- --require-webgpu
```

- Runs compiler examples and runtime-orchestration probes through CPU reference
  and real WebGPU in Chromium, then compares outputs. Covered probes: SAXPY,
  guarded map, tiled matmul, grid-sync phases, host runtime copy, host dynamic
  launch, and prepared resident dispatch.
- `e2e:webgpu:corpus` also requires pinned local corpora under `/tmp`, extracts
  exact external kernels through the same source-normalized compilation-unit
  path as the corpus audit, dispatches them in Chromium/WebGPU, and compares
  GPU readbacks to CPU reference. Fixture launch/input/output specs live in
  `scripts/cuda-lite-corpus-registry.mjs`. Current no-regression floor is `51`
  output-verified real WebGPU corpus fixtures: CUDA-120 `2`, NVIDIA
  `cuda-samples` `8`, `llm.c` `12`, and LeetCUDA `29`. Coverage includes
  vector add/scale, cuda-samples Bezier/mdspan/SobelTex, `llm.c` forward and
  backward transformer kernels, LeetCUDA scalar/vector activations, SGEMM,
  histogram, RoPE, direct transpose variants, and lifted CuTe transpose motifs.
  Fixture specs may pin explicit expected readbacks; when present, the gate
  checks both CPU reference and real WebGPU against that expected output instead
  of allowing the two implementations to agree on wrong-code. Current hard
  floor requires at least `22` pinned-output corpus fixtures.
- `e2e:webgpu:dist` runs the browser proof through built package exports. The
  combined `verify:real-world-cuda` gate runs both `src` and `dist` browser
  bundles unless a narrower `--bundle` is supplied.
- Root `verify:release` and CI's Chromium job run
  `verify:real-world-cuda`, so full-corpus
  compile/codegen coverage and exact-kernel real GPU fixtures are both hard
  regression gates. Npm publish workflows also run the compiler gate before
  publishing compiler-capability packages.

Performance gate:

- Compiler/runtime perf is tracked by
  `pnpm --filter @unlocalhosted/browsergrad-compiler bench -- --markdown /tmp/bg-cuda-lite-bench.md`.
  It reports JSON plus optional markdown for compile hot paths, CPU reference
  execution, host-lifted dynamic planning, and runtime-copy planning. Pinned
  machines can add `--expect-median-max` or `--expect-p95-max` with
  comma-separated `benchmark=ms` entries. Treat uncalibrated numbers as
  regression evidence, not universal pass/fail thresholds.
- Browser/WebGPU hot-loop perf should compare one-shot execution against
  `prepareCompiledKernelWebGpu()` over resident buffers. Keep this as measured
  evidence, not prose claims; no-readback prepared measurements should use
  `awaitCompletion: true`. Run:
  `pnpm --filter @unlocalhosted/browsergrad-compiler bench:browser -- --bundle dist --markdown /tmp/bg-cuda-lite-webgpu-bench.md`.
