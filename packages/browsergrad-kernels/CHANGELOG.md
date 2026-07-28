# Changelog

All notable changes to `@unlocalhosted/browsergrad-kernels`.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Host-graph program version 1.9 adds bounded top-level
  `dynamic-dispatch` with a positive request-time u32 logical-prefix element
  count. The portable WebGPU initial profile uses exactly one invocation per
  workgroup so the admitted count is also the exact x-workgroup launch count;
  it prewarms the artifact maximum once, reuses the same device-bound
  pipeline, changes only launch geometry, and records the actual count in
  specialization and terminal evidence. CPU and real WebGPU match for one-
  and two-element launches.
- Host-graph program version 1.8 adds bounded
  `runtime-u32-count-sequential` repetition. Preparation prewarms the
  artifact-maximum schedule; execution admits the exact request-time u32 count
  before copying inputs or accessing the device, rejects values above the
  maximum, and selects exact authorized slots for zero through the maximum
  iterations. Terminal traces report actual work and completion while
  preparation retains maximum budgets.
- Generic WGSL pipeline-set authority now accepts a strictly increasing exact
  step-index selection in addition to contiguous ranges. This authorizes
  omission of artifact-prewarmed runtime-repeat slots without permitting
  reordering, duplicated slots, cross-device use, or program substitution.
- Host-graph program version 1.7 adds one bounded
  `resource-u32-branch-sequential` conditional. The predicate is an ordered
  rank-local temporary u32 written by prior graph work. The CPU oracle reads it
  after its producer; WebGPU retains private buffers across one explicit
  four-byte readback/resubmission boundary, resumes an already-prewarmed exact
  branch slot, and preserves one timeout, cancellation, device-ownership,
  cleanup, numerical-status, and terminal-publication contract across both
  submissions.
- Generic WGSL pipeline-set authority now admits exact contiguous step ranges
  at validated slot offsets. This lets a bounded feedback executor split one
  authorized sequence without granting program reordering, arbitrary
  omission, cross-device use, or an unadmitted branch.
- Add `browsergrad.wgsl.pipeline-set@1` and
  `browsergrad.host-graph.webgpu-pipeline@1` as separate opaque,
  device-bound pipeline authorities. Preparation prewarms every unique program
  admitted at each exact step slot, including bounded conditional
  alternatives; copied, destroyed, cross-device, reordered, or program-
  mismatched use fails closed before request resource allocation.
- Host-graph pipeline identity binds the semantic graph, WebGPU backend
  version, WGSL modules, schedule, negotiated device features, relevant
  limits, and explicit numerical policies into the low-level cache namespace.
  Preparation is bounded by a caller-lowerable maximum under the fixed
  128-pipeline portable ceiling. The convenience graph runner captures caller
  inputs/controls first and then delegates through the same ephemeral authority
  path, while hot callers can prepare once, reuse, and explicitly destroy the
  authority.
- Authority-bound `browsergrad.host-graph.webgpu@1` preparation and execution.
  Verified static DAG dispatches expand through the canonical view-copy
  lowerer per rank; f32/i32/u32 all-reduce expands into bounded pairwise
  rank-order reduction followed by raw-word replication, without a second
  source-shaped orchestration path.
- Host-graph program version 1.1 whole-allocation copy nodes lower to one
  bit-preserving raw-word copy per rank. Required actual-device evidence adds
  an exact u8 case while the CPU reference also preserves odd-sized
  allocations; portable WebGPU explicitly refuses non-word-sized copies.
- Host-graph program version 1.2 requires one terminal
  `host-readback-after-graph-success` materialization node per output.
  Materialization selects fail-stop readback without adding a GPU dispatch;
  required actual-device u8 evidence now crosses that explicit node.
- Host-graph program version 1.3 adds unique dependency-ordered completion
  events. Successful CPU/WebGPU traces report completed event IDs without
  adding element work, GPU commands, timestamps, queue fences, or external
  wait authority.
- Host-graph program version 1.4 adds bounded fixed-count sequential
  repetition over linear dispatch, all-reduce, and copy bodies. The WebGPU
  backend lowers each body definition once through the existing canonical
  lowerers, reuses frozen step templates for the exact verified count, enforces
  preparation and expanded-step limits throughout expansion, and reports
  completion only after whole-graph success.
- Host-graph program version 1.5 adds bounded
  `input-u32-branch-sequential` conditionals. The WebGPU backend pre-lowers
  both branches, requires equal execution shapes, selects from the captured
  external-input predicate before device work, includes branch identity in
  specialization evidence, and reports completion only after whole-graph
  success. It grants no GPU-derived or mid-graph branch authority.
- Host-graph program version 1.6 adds bounded
  `runtime-u32-branch-sequential` conditionals through the same verified
  branches. Execution requires exactly the graph's runtime control set,
  captures every u32 value with the inputs before device access, selects zero
  as else and nonzero as then, and rejects missing, duplicate, unknown, or
  out-of-range bindings without granting GPU-derived, mid-graph, dynamic
  launch, or runtime loop-count authority.
- Graph execution snapshots complete direct/unshared inputs, initializes
  private temporary/output storage deterministically, bounds expanded steps
  and aggregate host/GPU storage, verifies device limits, and publishes fresh
  outputs only after complete submission, numerical-status validation, and
  readback. Cancellation, timeout, device loss, validation, shader, pipeline,
  and out-of-memory paths cannot commit partial caller-visible outputs.
- Required headed-Chromium evidence on Apple Metal 3 bit-matches the semantic
  CPU graph oracle for rank-ordered f32 sum, signed-zero f32 min, wrapping i32
  sum, exact u32 max, exact u8 allocation copy, and three fixed repetitions of
  f32 all-reduce plus both branches of captured-input, runtime-control, and
  produced-resource conditionals, and separately rejects non-finite f32
  collectives.
- A separate required host-graph performance lane compares version-1.4
  fixed-count repetition with a bit-exact, equal-work version-1.2 unrolled
  graph over two ranks and 65,536 f32 elements. It uses an untimed CPU and
  WebGPU correctness preflight, prewarms both exact pipeline authorities,
  then takes eight warmups and twelve alternating paired authority-bound
  execution samples with full readback and queue drains. Raw samples, pipeline
  identities, and the named browser/device configuration are retained; the
  record makes no superiority or regression claim.
- Semantic attention preparation consumes exact verified rank-4 f32 logical
  meaning plus an independently authorized online K/V-tile schedule. Generated
  WGSL cooperatively stages K/V rows, keeps one Q/output row private per lane,
  applies causal and tail masks before online-state updates, and places two
  all-lane uniform barriers around every staged tile.
- Preparation binds full logical, schedule, backend, and WGSL identities and
  bounds WGSL bytes, workgroup invocations/storage, private elements, key-tile
  count, dispatch workgroups, and aggregate host/GPU transient storage. It
  reports only `block-tiled-kv-online-softmax-forward` and
  `portable-relegalized`; no device execution, numerical preservation,
  performance, FlashAttention-v2, frontend, or resident-buffer claim is made.
- Semantic attention host execution admits only exact fixed unshared finite-f32
  Q/K/V bindings, snapshots all inputs before yielding or touching the device,
  rejects overlap/accessors/subclasses/resizable or detached storage, checks
  device limits, bounds validation and execution waits, and drains scoped GPU
  diagnostics before publishing a complete finite destination.
- The required browser lane executes irregular `(B=1,H=2,Sq=9,Sk=11,D=4,Dv=6)`
  causal and non-causal attention under 8x8 and 8x16 schedules on Apple Metal
  3. Every complete output passes the semantic-core absolute-or-relative CPU
  comparator and same-mask cross-schedule comparison. Execution traces still
  require declared-policy comparison and do not self-assert preservation.
- A separate required performance lane records the named
  `block-tiled-kv-online-softmax-forward` implementation against the frozen
  `row-wise-online-softmax-baseline` on `(B=1,H=2,Sq=256,Sk=256,D=Dv=32)`.
  It uses 16 warmups, 20 alternating paired samples, complete output readback,
  and a named browser/device/configuration. Raw samples and lifecycle
  differences are retained; the record makes no superiority or regression
  claim and correctness remains a separate required lane.
- Production semantic GEMM preparation and WebGPU execution consume the exact
  verified layout, logical GEMM, physical schedule, and concrete-input
  certificate authorities. Cooperative scalar workgroup staging uses uniform
  barriers, zero-filled boundary loads, masked complete-root stores, and
  deterministic logical/schedule/backend/WGSL trace identities. Execution
  traces report `portable-webgpu-core` and `portable-relegalized`; they do not
  imply a preserved CUDA/CuTe invocation schedule or native MMA facility.
- The portable numerical profile is deliberately limited to semantic-core's
  exact nonnegative-integer f32 certificate. Host execution uploads only fresh
  authority-retained snapshots and compares complete destination bytes; raw
  resident `GPUBuffer` inputs fail closed until trusted upload provenance exists.
- Focused advisory and required-device browser lanes execute irregular
  17-by-23-by-19 GEMM through 8-by-8-by-8 and 16-by-16-by-16 schedules and
  require both actual WebGPU outputs to bit-match the source-ordered CPU result.

### Changed

- Generic WGSL sequence preparation now creates every storage, texture,
  uniform, shader/pipeline request, and bind group for every step before its
  first await. Production error scopes therefore own the complete sequence
  issue phase rather than only the first step.
- Resident semantic dispatch and materialization now issue all synchronous GPU
  work under production validation, out-of-memory, and internal error scopes;
  all LIFO pops start before the first await and race operation completion with
  device loss. Diagnostic failures destroy roots, clear pipeline/output pools,
  settle profiles, and cannot mint bridge handles or re-pool poisoned buffers.
- Direct callers use an async scoped resident API. Tensor-plan execution alone
  receives a private, non-exported synchronous issue capability so future
  consumers cannot bypass diagnostic ownership accidentally.
- Kernels publication runs build/typecheck/lint/tests before its final clean-
  commit gate and requires kernels view-copy, semantic-host-graph,
  semantic-GEMM, semantic-attention correctness/performance, and JIT
  semantic-permutation evidence markers for the exact source revision.

## [0.2.0] - 2026-07-15

### Added

- Kernels-owned WGSL lowering and execution for verified
  `browsergrad.kernel@1` `view-copy@1.0` artifacts. It consumes the shared
  backend-neutral specialization proof and never widens `TensorGpuPlan`.
- Signed-i32 interval-checked lowering for canonical index/predicate
  expressions, whole-root `array<u32>` bindings for exact f32/NaN bits,
  structured guarded padding loads, and two-level semantic/device hashes.
- Focused advisory and required-device browser lanes. The required lane fails
  on adapter/device absence and emits one validated terminal evidence record
  over nine ordered static/dynamic/zero-extent cases. It distinguishes adapter
  versus negotiated features and records artifact, input, schedule, limit,
  backend, environment, and bit-exact comparison facts.
- The required lane passes all nine cases in headed Chromium on Apple Metal 3;
  headless adapter absence remains a separate failed environment record, not a
  false pass or a contradiction of the actual-device result.
- Authority-bound immutable prepared plans, full-digest pipeline names,
  bounded host/GPU working sets, one-in-flight device ownership, timeout/abort
  stale-result suppression, scoped LIFO error drainage, distinct shader and
  pipeline diagnostics, and cache invalidation on device loss.
- Resident prepared view-copy dispatch through the exact canonical WGSL with
  no upload/readback or offset reconstruction. It validates whole-root source
  bytes and permits only dense zero-offset destinations that overwrite their
  complete allocation.
- Strict preparation and execution for JIT semantic tensor-plan request side
  tables. Routing identity is authority-bound but excluded from layout,
  kernel, specialization, and WGSL hashes; semantic-route `PERMUTE` never calls
  the legacy shape/axes kernel.
- Prepared JIT semantic requests expose immutable backend profile/version and
  planned logical/workgroup topology. Live semantic bridge handles expose the
  exact authority-bound preparation plus settled dispatch profiles, separating
  planned topology from workgroups actually submitted without exposing offsets.
- Exact packed dependency on `@unlocalhosted/browsergrad-semantic-core@0.2.0`
  and a public `./semantic_view_copy` subpath. Publishing is blocked unless the
  required-device evidence marker names the exact current commit.
- `runThreadGrid()`, `defineKernel1DProgram()`,
  `runKernel1DProgramReference()`, `emitKernel1DProgramWgsl()`, and
  `runKernel1DProgramWebGpu()` expose the generic BrowserGrad kernel-authoring
  surface. CUDA-shaped names remain compatibility aliases for labs that teach
  CUDA/HIP vocabulary.
- `defineCuda1DProgram()`, `simulateCuda1DProgram()`, and
  `emitCuda1DProgramWgsl()` provide a tiny CUDA-shaped 1D program IR that can
  run through a deterministic simulator and lower to WGSL. It now supports
  scalar params and `outputRead` expressions for A3-style SAXPY.
- `runCuda1DProgramWebGpu()` dispatches emitted `Cuda1DProgram` WGSL through a
  real browser `GPUDevice`; browser tests skip clearly when no adapter exists.
- `simulateCuda1DGrid()`, `referenceSaxpy()`, `referenceExclusiveScan()`,
  `referenceFindRepeats()`, and `referenceOrderedCircleRender()` provide
  CUDA-shaped browser-safe references for GPU Puzzles and CS149 A3-style
  rubrics.
- `referenceFlashAttention()` returns browser-safe FlashAttention output plus
  log-sum-exp tensors for CS336 A2-style forward/LSE rubric checks.
- `referenceFlashAttentionBackward()` recomputes Q/K/V gradients for
  CS336 A2-style backward checks without PyTorch autograd, Triton, or CUDA.
- `createKernelRubric()` records pass/fail assertions for JS/WebGPU lab
  rubrics, including `assertCloseTensor()` with shape checks, compact previews,
  first failing index, non-finite value detection, and max absolute error.
- `kernelRubricFailureToAssertionDetails()` formats kernel failure details into
  `expected` / `actual` strings for BrowserGrad-style assertion callbacks.
- `createBrowsergradKernelRubric(target)` adapts kernel rubric assertions to a
  BrowserGrad JS rubric context or any compatible assertion target.
- `createWgslStorageBuffer()`, `writeWgslStorageBuffer()`, `readWgslStorageBuffer()`, and
  `residentBuffers` let generic WGSL callers keep `GPUBuffer`s alive across
  dispatches and opt out of readback with `readback: []`.
- `prepareWgslKernelProgramSequence()` prebuilds pipelines and bind groups for
  reusable hot-loop WGSL sequences over resident buffers.
- Prepared WGSL sequences can update uniform buffers at `run()` time without
  rebuilding bind groups; step-specific overrides use `stepUniforms`.
- Prepared WGSL sequence runs accept `awaitCompletion: true` so no-readback
  dispatches can explicitly wait for GPU completion without hand-rolled queue
  fences.
- WGSL sequences can bind the same storage buffer through multiple step-local
  value types, enabling byte-identical storage views for compiler-generated
  pool/data aliases.
- `storageMetadata` lets generic WGSL callers declare canonical storage
  readback types and byte-compatible aliases, including readback-only state
  buffers that are not bound by any sequence step.
- `createWgslFloat16Array()` and float16 conversion helpers provide a small
  JS `Float16Array` backing path when the host runtime has WebGPU `shader-f16`
  support but Node/browser JS lacks the typed array constructor. Global
  `Float16Array` installation is explicit via `installWgslFloat16ArrayPolyfill()`.

### Fixed

- Mixed-view WGSL sequence inputs now validate against all byte-compatible
  storage value types for a physical buffer, not only the first step-local
  binding.

## [0.1.2] — 2026-07-06

### Fixed

- Publishes the current rebuilt `dist/` surface for the WGSL program,
  float16, CUDA concept, rubric, and CUDA program APIs that were present in the
  repo but missing from the stale npm `0.1.1` tarball.
- Restores the published export surface required by
  `@unlocalhosted/browsergrad-compiler`, including the WGSL program sequence
  and float16 helpers used by compiler WebGPU execution.

## [0.1.1] — 2026-06-02

Dogfood pass on the published 0.1.0 tarball surfaced three issues. All fixed.

### Fixed

- **GPU `kernels.softmax` returned all zeros** on Chromium's
  SwiftShader + Metal-driver path (real-WebGPU dogfood in headed
  Chromium on macOS). Root cause, isolated via incremental probe
  kernels: the literal `-3.4028235e38` used as the max-init sentinel
  parsed to **0** on that driver path, so the comparison
  `v > maxVal` always evaluated false, no max was found, and the
  whole row's subsequent exp-divide-by-sum collapsed (sum overflowed
  or wrote zeros). Fix: initialize `maxVal` with `X[base]` and iterate
  from `i = 1`. Sidesteps the literal-parse issue and is also a touch
  more numerically robust. Pass 3 was additionally tightened to a
  pure-write (re-computing `exp(x - max)` rather than reading Y back
  from pass 2) — defense in depth against read-after-write storage
  semantics on the same driver path.
  Caught by `@unlocalhosted/browsergrad-dogfood`; was not covered by
  the package's own `tests-browser/webgpu_real.test.ts` (which
  exercises matmul/tiled-GEMM/fused-elementwise/FA-v2 but not
  softmax). Cascades into `kernels.attention` correctness (attention
  composes softmax).
- **`WebGpuRealizerBridge.materialize` type contract**: declared
  `Uint8Array`, runtime always returned `Promise<Uint8Array>`. TS
  consumers following the `.d.ts` got `undefined.buffer` at runtime;
  the package's own browser test silently cast via
  `as unknown as Promise<Uint8Array>`. The interface now honestly
  declares `Promise<Uint8Array>` and the implementation is `async`.
  Pyodide JSPI consumers (Python) still see a synchronous return — the
  Promise is unwrapped at the boundary as documented.
- **`kernels.attention` / `reference.attention` error message** when
  Q seq ≠ K=V seq is now explicit about the v0 self-attention-only
  limitation and points at the PRD-012c follow-on. Behavior unchanged;
  documentation gap closed.

## [0.1.0] — 2026-05-25

Initial release. Six WGSL kernels, each with a pure-JS reference.

### Added

- `createDevice(options?)` — wraps a `GPUDevice` with a pipeline cache.
- `tensor(shape, data)` — small constructor helper for `Tensor` literals.
- `kernels.matmul(device, A, B)` — naive 2D matmul, f32.
- `kernels.softmax(device, x)` — stable softmax along the last axis.
- `kernels.relu(device, x)` — elementwise.
- `kernels.gelu(device, x)` — tanh-approximation (GPT-2/BERT variant).
- `kernels.layernorm(device, x, { gamma?, beta?, eps? })` — along last axis.
- `kernels.attention(device, Q, K, V)` — scaled dot-product. Single head, `[S, D]` shapes.
- `reference.*` — pure-JS counterparts for every kernel.
  Exposed at the top-level entry and at the `./reference` subpath import.
- `KernelDevice.getStats()` / `clearCache()` for debugging.
- `KernelError` for input shape and device errors.
- Full TypeScript declarations + source maps.
- 26 vitest tests covering the public surface, argument validation, and
  reference-impl numerical correctness against hand-checked values.

### Deferred

Planned but not in 0.1.0; additive when they land:

- **Browser conformance tests** — run the WGSL kernels in a real WebGPU
  context and compare to the JS reference (`1e-4` tolerance). Requires
  `@vitest/browser` + Playwright; planned for next.
- **Pre-allocated buffer mode** (`kernel.runOnGpu`) — for hot training loops
  that want to keep tensors on the GPU between calls. v0 always copies
  back to JS each call.
- **Tiled / optimized variants** of matmul, fused attention. Same surface;
  faster paths chosen via a future `mode` option.
- **Batched + multi-head attention.** v0 is single-head, unbatched.
- **f16 support.** All v0 kernels are f32.
- **Mask support for attention.** v0 has no masking.

### Known limitations

- v0 allocates fresh GPU buffers every call. Useful for "compute once" work;
  not optimal for hot loops. The grad library (or any consumer) should
  pre-allocate via the planned `runOnGpu` path once it lands.
- Attention dispatches four separate kernels (transpose → matmul → scale →
  softmax → matmul). A fused implementation will be much faster but is
  not required for correctness or pedagogy.

[0.1.0]: https://github.com/unlocalhosted/browsergrad/releases/tag/kernels%40v0.1.0
