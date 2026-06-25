# CUDA-lite Compiler Architecture

This doc is the low-level design map for extending
`@unlocalhosted/browsergrad-compiler` without turning it into a pile of
assignment patches.

## Spine

Pipeline:

```text
CUDA-lite source
  -> lexer/parser
  -> semantic analysis
  -> Kernel IR
  -> CPU reference interpreter
  -> WGSL emitter
  -> WebGPU execution plan
  -> kernels package dispatch
```

The compiler owns CUDA-shaped semantics. `@unlocalhosted/browsergrad-kernels`
owns WebGPU device resources, WGSL program validation, prepared sequences,
resident buffers, and readback mechanics.

## Source Of Truth

- `src/types.ts`: AST, diagnostics, Kernel IR, launch/input/result types.
- `src/parser.ts` and `src/lexer.ts`: syntax only. No semantic rewrites.
- `src/analyzer.ts`: symbols, types, feature gates, safety checks, and lowering
  eligibility.
- `src/reference.ts`: CPU truth and traces from Kernel IR semantics.
- `src/wgsl.ts`: WGSL emission only. It should not rediscover CUDA semantics.
- `src/runtime_plan.ts`: CUDA runtime operations discovered from IR.
- `src/webgpu_orchestration.ts`: exact WebGPU executable plan selection.
- `src/launch.ts`: shared launch-shape diagnostics for reference/WebGPU parity.
- `src/runner.ts`: public compile/run/prepare APIs.
- `src/compatibility.ts`: diagnostic to semantic-family mapping.

If a feature needs support in multiple files, add it in this order:

1. Parse it without losing source spans.
2. Analyze it and emit deterministic diagnostics for unsupported cases.
3. Represent it in Kernel IR or runtime-plan data.
4. Execute it in CPU reference when semantics are understood.
5. Lower it to WGSL or classify why it is reference-only.
6. Add corpus-audit coverage and focused unit/browser tests.

## Invariants

- Parser accepts syntax, analyzer decides meaning.
- Diagnostics always carry stable codes and source spans.
- Reference and WGSL lowering must read the same Kernel IR facts.
- GPU support is a lowering decision, not a parser side effect.
- Runtime orchestration goes through `createCudaWebGpuExecutionPlan()`.
- Platform code must not duplicate compiler planning heuristics.
- Unsupported WebGPU execution plans expose stable blocker codes; prose reasons
  are for humans, not platform branching.
- Readback names exposed to callers are logical compiler names. Internal WGSL
  storage aliases stay internal.
- No assignment-specific names or course-specific branches in compiler logic.
- Every unsupported CUDA feature maps to a semantic family in
  `compatibility.ts`.
- Any new GPU lift needs either a real browser test or a documented reason why
  only planner/unit coverage is possible.

## Memory Model

Modeled memory spaces:

- Global pointer params: storage buffers in WebGPU, typed arrays in reference.
- Storage pointer params and one-dimensional `__shared__` arrays passed into
  `__device__` helpers lower to a compact WGSL `{ memory id, base index }` ABI
  plus generated typed read/write helpers. Reference execution uses the same
  address model, so helper calls preserve pointer offsets instead of copying
  arrays.
- Scalar params: packed uniform buffer in WebGPU, JS numbers in reference.
- Fixed thread-local arrays: function-scope WGSL arrays, per-thread typed arrays
  in reference.
- Fixed `__shared__` arrays: workgroup memory in WGSL, per-block arrays in
  reference.
- Dynamic shared memory: launch metadata decides size; unsupported sizes get a
  deterministic diagnostic.
- `DevicePool*`: pool data plus offset buffers. Pool aliases must be explicit
  in orchestration plans.

Rules:

- Const pointer writes are rejected in analysis.
- `const` helper pointer writes are rejected before WGSL emission; writable
  helper pointer cases only emit switch branches for non-const storage buffers.
- Pointer arithmetic must resolve to modeled base plus non-negative element
  offset before WebGPU lowering.
- Device-derived offsets can run in reference if semantics are known, but stay
  reference-only until an explicit GPU plan models them.
- Atomic pointer params must use atomic-compatible storage lowering.

## Synchronization Model

Supported layers:

- `__syncthreads()` maps to `workgroupBarrier()` and lockstep per-block
  reference execution.
- Safe top-level uniform `grid.sync()` maps to multi-dispatch
  `grid-sync-phases`.
- Device launches and runtime copies are runtime operations. They may be
  host-lifted only when `webgpu_orchestration.ts` can build a complete,
  deterministic sequence.
- Stream/event create, destroy, record, and synchronize calls are modeled as
  host-managed ordering no-ops. They preserve source compatibility for async
  copy lessons without pretending WebGPU exposes CUDA stream semantics.
- Host-dynamic launch planning can expand parent invocations with CUDA builtin
  coordinates and flatten recursive child launches, but bounded caps protect the
  browser from runaway launch trees.
  High-level runner APIs expose those same caps as
  `maxHostExpandedParentInvocations` and `maxHostDynamicLaunchDepth`; platform
  code should set them per lab profile instead of relying on global defaults.
- Host-lifted pool allocation pointers are allowed when they map to known
  `DevicePool` byte offsets. Expanded parent allocations are liftable when
  child launches are order-stable except for pointer base offsets. When the
  host planner owns pool offset advancement, pure parent kernels are skipped and
  pool offsets are seeded once before child dispatch; mixed parent side effects
  plus host-planned pool allocation are unsupported.

Rules:

- Divergent barriers are rejected unless analysis can prove uniform control.
- Shared memory cannot cross WebGPU dispatch phases unless the phase plan proves
  rewrite-before-read.
- Private locals crossing grid phases must be replayable from launch-derived or
  uniform data.
- Host orchestration must preserve CUDA ordering by sequence order, not by
  hidden async assumptions.

## Execution Plans

`createCudaWebGpuExecutionPlan()` returns one of:

- `single-dispatch`: direct WGSL compute.
- `grid-sync-phases`: multiple WGSL dispatches over shared GPU buffers.
- `host-dynamic-launch`: host-lifted child dispatch sequence.
- `host-copy`: host-lifted typed runtime-copy dispatch.
- unsupported plan with diagnostics.

Use `summarizeCudaWebGpuExecutionPlan()` for platform UI/readiness rows. Do not
infer WebGPU execution readiness from `compiled.loweringPlan.canRunOnGpu` alone:
that lowering plan is intentionally conservative and marks CUDA runtime gaps as
unsupported even when `createCudaWebGpuExecutionPlan()` can host-orchestrate real
WebGPU passes for the same kernel.
Compile runtime-gap kernels with `compileCudaLiteKernelForWebGpu()` before
planning. It keeps strict `compileCudaLiteKernel()` available for direct
lowering checks, while turning host-orchestratable runtime diagnostics into
warnings for WebGPU planning.

Prepared execution uses the same plan. `prepareCompiledKernelWebGpu()` must not
create a second planning path.
Launch-shape validation also lives at this boundary: invalid grid/block
dimensions return unsupported `launch` blockers before any dispatch plan is
built.

Hot-loop rules:

- Cache compile results with `createCudaLiteCompilerCache()` at assignment,
  rubric, or editor-session scope. Keep cache bounded; compiled outputs are
  source/option immutable by contract.
- Use resident buffers to avoid upload/readback churn.
- Use prepared WGSL sequences to avoid pipeline/bind-group rebuilds.
- Use scalar uniform updates only when launch shape and binding topology remain
  fixed.
- For host-orchestrated prepared plans, replan scalar updates and compare
  topology before writing per-step uniforms. Never mutate prepared dispatch
  counts or aliases implicitly.
- Prepared host-orchestrated replanning uses a bounded child-kernel compile
  cache; keep it enabled for hot loops unless deterministic compile-count
  instrumentation matters more than throughput.
- Use `awaitCompletion: true` for no-readback timing and watchdog gates.

## Compatibility Growth

Feature families:

- `frontend`: C/CUDA syntax, macros, helpers, types.
- `memory`: local, constant, texture, and pointer alias spaces.
- `atomic`: CUDA atomics and memory ordering.
- `texture`: texture/surface reads and writes.
- `subgroup`: warp intrinsics and cooperative groups.
- `library`: cuRAND, cuFFT, reductions, scan/sort islands.
- `runtime`: streams, events, launches, sync, copies.
- `feature`: WebGPU gates like `shader-f16` and subgroups.
- `safety`: divergent barriers, invalid writes, unbounded behavior.

New support should add a semantic primitive, not a string match against one
lesson. If a corpus gap is broad, add a plan or compatibility family first, then
lower concrete cases.

## Evidence Gates

Required before claiming a CUDA feature works:

- Unit test for parse/analyze/diagnostics.
- Reference test when semantics are known.
- WGSL snapshot or emitted-source assertion for lowering.
- Browser WebGPU test for real dispatch when browser support exists.
- Corpus audit run for broad compatibility claims.

Current compile/codegen corpus gate:

```sh
pnpm --filter @unlocalhosted/browsergrad-compiler audit:cuda-120
```

Current WebGPU smoke/perf gate:

```sh
pnpm --filter @unlocalhosted/browsergrad-compiler e2e:webgpu
pnpm --filter @unlocalhosted/browsergrad-compiler e2e:webgpu:dist
pnpm --filter @unlocalhosted/browsergrad-compiler bench:browser -- --require-webgpu --expect-prepared-ratio-max 10
```

`verify:real-world-cuda` runs both source-alias and dist-export browser bundles
by default, so library consumers are covered in addition to local TS source.
Use stricter ratio values only on pinned machines. Browser/GPU timing is not
portable enough for global absolute thresholds.

## Extension Checklist

- Does public API stay generic and package-neutral?
- Did syntax, analysis, reference, WGSL, and orchestration stay separated?
- Are unsupported paths loud, deterministic, and source-spanned?
- Does platform code have one official API to ask "can this run on GPU?"
- Does real WebGPU execute the feature, or does docs say reference-only?
- Does corpus coverage improve or stay protected by thresholds?
- Are hot-loop paths resident/prepared by default where possible?
- Are names learner-facing but not assignment-facing?
