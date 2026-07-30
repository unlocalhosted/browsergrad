# @unlocalhosted/browsergrad-kernels

[![npm](https://img.shields.io/npm/v/@unlocalhosted/browsergrad-kernels.svg)](https://www.npmjs.com/package/@unlocalhosted/browsergrad-kernels)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

WGSL compute-shader catalog for browser ML and systems workloads. Each kernel
ships with a pure-JS reference implementation that acts as a conformance
oracle; reference execution is distinct from device execution. The package also
ships the production `WebGpuRealizerBridge` that
[`browsergrad-jit`](../browsergrad-jit/) consumes for its WebGPU realizer tier.

No tensor-framework dependency. The package depends only on BrowserGrad's
dependency-free semantic-core protocols for verified layout, kernel, schedule,
and graph artifacts. Drop it in for WGSL primitives; layer in JIT for the
PyTorch-shaped surface.

## What's shipped

### Kernels (with JS reference)

| Kernel | Variant | Status |
|---|---|---|
| `matmul` | Naive triple-loop, host-tensor input/output | ✅ |
| `matmulTiled` / `matmulTiledDirect` | 16×16 workgroup-tiled GEMM. **Production path.** | ✅ |
| `softmax` | Stable, along last axis | ✅ |
| `relu`, `gelu` | Elementwise activations | ✅ |
| `layernorm` | Along last axis, optional gamma/beta | ✅ |
| `attention` | Composed 3-kernel SDPA | ✅ |
| `referenceFlashAttention` / `referenceFlashAttentionBackward` | Pure-JS FlashAttention oracle with output, log-sum-exp, and Q/K/V gradients | ✅ |
| `defineKernel1DProgram` / `runKernel1DProgramReference` / `emitKernel1DProgramWgsl` / `runKernel1DProgramWebGpu` | BrowserGrad-owned 1D kernel IR with reference executor, WGSL lowering, and browser WebGPU dispatch | ✅ |
| `runThreadGrid`, `referenceSaxpy`, `referenceExclusiveScan`, `referenceFindRepeats`, `referenceOrderedCircleRender` | Thread-grid teaching references for GPU Puzzles and CS149 A3 browser rubrics | ✅ |
| `defineCuda1DProgram` / `simulateCuda1DProgram` / `emitCuda1DProgramWgsl` / `runCuda1DProgramWebGpu` / `simulateCuda1DGrid` | CUDA-shaped compatibility aliases for labs and rubrics that teach CUDA vocabulary | ✅ |
| `prepareSemanticViewCopyWgsl` / `runSemanticViewCopyWebGpu` | Verified `view-copy@1.0` lowering over canonical layout/index artifacts, with exact raw 8-bit/16-bit/32-bit/64-bit storage and structured guarded padding | ✅ 75-case strict CPU/WebGPU parity on Apple Metal 3 across all supported storage dtypes at ranks 1–8, with negative source strides for every storage width, guarded negative predicates, broadcast, offsets, packed tails, and float padding; release evidence remains commit-scoped |
| `prepareSemanticHostGraphWebGpu` / `prepareSemanticHostGraphWebGpuPipeline` / `runSemanticHostGraphWebGpuPipeline` | Authority-bound `browsergrad.host-graph@1` execution with a separately prepared exact device-bound pipeline authority, per-rank private storage, canonical view-copy dispatches, whole-allocation raw copies, dependency-ordered completion events, bounded fixed-count, request-time u32-count, and one produced-resource u32-count repetition, bounded positive request-time or produced-resource arbitrary one-dimensional prefix dispatch including exact shared-selection fanout and one exact two-stage producer-chain profile, rank-2-through-rank-8 request-time and produced-resource rectangular prefix dispatch including one exact shared-rectangle two-dispatch fanout, captured-input, runtime-control, and one produced-resource u32 conditional, terminal materialization, and ordered f32/i32/u32 all-reduce | ✅ required 62-case real-WebGPU complete-output bit parity with the CPU graph oracle for finite f32 sum/signed-zero min, wrapping i32 sum, exact u32 max, event-marked/materialized u8 allocation copy, fixed plus zero/two-iteration request-time and produced-resource f32 repetition, 1/2-, aligned 64/128-, and unaligned 65/127-element workgroup-64 request-time and produced-resource linear launch, shared-count two-dispatch fanout, one/two-element two-stage produced launch, small/full request-time and produced-resource rank-2-through-rank-8 rectangular launch, shared rank-8 rectangular two-dispatch fanout, and both branches of all three conditional sources through prewarmed pipeline authority, plus a separate observational fixed-repeat/unrolled authority-reuse record; broader repeated/device-side feedback, transport, and native companions remain separate |
| `prepareSemanticGemmWgsl` / `runSemanticGemmWebGpu` | Verified logical GEMM plus independent schedule lowering with cooperative workgroup staging, uniform barriers, and masked boundary tiles | ✅ bit-exact only for semantic-core certified exact f32 inputs; required irregular two-schedule WebGPU evidence |
| `prepareSemanticAttentionWgsl` / `runSemanticAttentionWebGpu` | Verified attention plus independent online K/V-tile schedule lowering/execution with cooperative staging, uniform barriers, and causal/tail masks before state updates | ✅ required causal/non-causal two-schedule CPU/WebGPU comparison plus separate observational host-API performance record on Apple Metal 3 |
| `rowWiseOnlineAttentionDirect` | Fused row-wise online-softmax attention baseline with strict real-WebGPU parity vs composed reference; not block-tiled FlashAttention. | ✅ |
| `flashAttentionDirect` | Deprecated compatibility alias for `rowWiseOnlineAttentionDirect`. | ⚠️ |
| `fusedElementwiseDirect` | Runtime WGSL codegen for arbitrary elementwise chains | ✅ |
| `runTensorGpuPlan` / `runTensorGpuPlanResident` | Generic tensor-plan executor for primitive f32 BUFFER/LOAD/MATMUL/elementwise/shape/reduce/Conv1d/Conv2d/ConvTranspose2d/Conv3d/LayerNorm-forward-backward/SGD/Adam/AdamW-update steps with GPUBuffer residency; materialized or resident root modes | ✅ |
| `WebGpuRealizerBridge.conv1d*` / `conv2d*` | Resident Conv1d/Conv2d forward plus input/weight/bias backward kernels for jit explicit realization. | ✅ |

### Realizer-tier surface (consumed by jit)

- `createWebGpuRealizerBridge(device)` — production bridge satisfying the `WebGpuBridge` Protocol declared in jit. Opaque integer handles; bridge owns `GPUBuffer` lifetimes; pipeline cache via `runDirect`.
- `runDirect(device, desc, opts)` — `GPUBuffer`-in / `GPUBuffer`-out dispatch. The realizer-tier path; no host round-trip per op.
  By default, owned output buffers come from a per-device reusable output pool
  (`createDevice({ outputBufferPoolSize })`), visible through
  `device.getStats().outputBufferPool*`. Pass `opts.outputBuffer` when the
  caller owns output storage.
- `runTensorGpuPlan(device, plan, inputs)` — generic tensor-IR plan executor.
  It consumes scheduled primitive steps, accepts the snake_case plan payload
  emitted by `browsergrad-jit`, keeps intermediates resident, and materializes
  only the declared root. Current coverage: f32 BUFFER/LOAD/2-D MATMUL,
  scalar elementwise ops, FUSED_ELEMENTWISE runtime WGSL codegen,
  FUSED_SOFTMAX last-axis direct softmax, RESHAPE, PERMUTE, BROADCAST_TO, and
  REDUCE(sum/mean) rank <= 4, plus
  Conv1d/Conv2d/ConvTranspose2d/Conv3d/LayerNorm forward/backward and
  functional SGD/Adam/AdamW updates. SDPA-shaped graphs run as primitive
  tensor-plan steps (`MATMUL` -> scale -> `FUSED_SOFTMAX` -> `MATMUL`), not
  a framework-specific bridge method. The executor uses plan liveness to
  return dead owned direct-dispatch outputs to the reusable output pool before
  the root boundary, destroys uploaded host inputs when they die, and reports
  `earlyReleasedBuffers` / `earlyReleasedBytes` for tests and profiling. This
  is the future framework-runtime direction; per-op bridge methods are
  legacy/interim coverage.
- `runTensorGpuPlanResident(device, plan, inputs)` — same executor, but returns
  an owned resident root `GPUBuffer` instead of reading it back. Inputs may be
  host data or bridge-owned resident handles.
- `createWebGpuRealizerBridge(device).run_tensor_plan(plan, inputs, dtype)` —
  Pyodide bridge entrypoint for the same graph-level executor. Prefer extending
  this path for core framework ops instead of adding new per-op bridge methods.
- `createWebGpuRealizerBridge(device).run_tensor_plan_resident(plan, inputs, dtype)` —
  bridge entrypoint for resident tensor-plan roots; follow-on plans can pass
  the returned handle as an input without CPU readback.
- `materializeFloat32(device, buffer, byteLength)` — read a `GPUBuffer` back to a `Float32Array` (the single readback at the realize boundary).
- `uploadFloat32(device, data)` — upload a typed array into a fresh `GPUBuffer`.
- `createWgslStorageBuffer()` / `writeWgslStorageBuffer()` /
  `readWgslStorageBuffer()` — caller-owned resident storage buffers for generic
  WGSL programs. Use `residentBuffers` with `runWgslKernelProgramSequence()` to
  avoid per-call upload and skip readback with `readback: []`.
- `prepareWgslKernelProgramSequence()` — prebuilds pipelines and bind groups
  once, then reruns the same WGSL sequence over resident buffers for hot loops.
- `prepareWgslKernelPipelineSet()` — prewarms an opaque device-bound set for
  exact step slots and explicitly declared alternatives. Passing that set to
  `prepareWgslKernelProgramSequence()` authorizes a complete sequence or an
  exact contiguous slot range and rejects copies, destroyed authorities,
  cross-device use, reordering, invalid offsets, and unadmitted programs before
  allocation.
- `getWgslPipelineCacheStats()` / `clearWgslPipelineCache()` — inspect or
  invalidate only the generic WGSL program cache for one device. This is kept
  separate from `device.getStats()` so compiler-prepared sequences have
  attributable cache telemetry.

## Install

```bash
npm install @unlocalhosted/browsergrad-kernels
```

## Public import surface

Most consumers should use the top-level package:

```ts
import {
  createDevice,
  createWgslFloat16Array,
  defineWgslKernelProgram,
  prepareWgslKernelProgramSequence,
  runThreadGrid,
  createKernelRubric,
} from "@unlocalhosted/browsergrad-kernels";
```

These subpaths are stable for bundlers and agents that want smaller,
domain-specific imports:

```ts
import { reference } from "@unlocalhosted/browsergrad-kernels/reference";
import { defineWgslKernelProgram } from "@unlocalhosted/browsergrad-kernels/wgsl_program";
import { createWgslFloat16Array } from "@unlocalhosted/browsergrad-kernels/float16";
import { runThreadGrid } from "@unlocalhosted/browsergrad-kernels/cuda_concepts";
import { defineCuda1DProgram } from "@unlocalhosted/browsergrad-kernels/cuda_program";
import { createKernelRubric } from "@unlocalhosted/browsergrad-kernels/rubric";
import {
  prepareSemanticViewCopyWgsl,
  runSemanticViewCopyWebGpu,
} from "@unlocalhosted/browsergrad-kernels/semantic_view_copy";
import {
  prepareSemanticHostGraphWebGpu,
  runSemanticHostGraphWebGpu,
} from "@unlocalhosted/browsergrad-kernels/semantic_host_graph";
```

Do not import private files under `src/` or `dist/` from consumer code.

`PreparedSemanticViewCopyWgsl` values are immutable and accepted only by the
module instance that prepared them. View-copy execution bounds its planned
owned working set, permits one in-flight run per `GPUDevice`, distinguishes
shader/pipeline/validation/memory/device-loss failures, and retains the device
slot after timeout or cancellation until submitted work has cleaned up. Pass an
`AbortSignal` or a bounded `timeoutMs` to `runSemanticViewCopyWebGpu` when the
caller owns a shorter lifecycle.
Release CI verifies the packed tarball exports these entry points before npm
publish.

### Verified semantic view copy

The semantic view-copy surface accepts only opaque verified layout/kernel
artifacts from `@unlocalhosted/browsergrad-semantic-core`. Shared preparation
resolves bindings, proves every guarded access and dense destination write,
and derives one semantic specialization hash. Kernels then lowers the same
canonical index/predicate expressions into either
`browsergrad.webgpu.view-copy.word32@2` or the distinct packed8, packed16, and
word64 profiles. The word32 profile uses
signed-i32 address expressions while preserving same-dtype f32, i32, or u32
storage bit-for-bit. Positive-affine word32 profiles cover ranks 1 through 8.
The distinct
`browsergrad.view-copy.signed-affine-rank2-rank3-word32@1` profile admits
negative source-coordinate scales at ranks 2 and 3; the separate
`browsergrad.view-copy.signed-affine-rank4-rank5-word32@1` profile covers
ranks 4 and 5, and `browsergrad.view-copy.signed-affine-rank1-word32@1`
covers rank 1. Separate rank-6, rank-7, and rank-8 identities complete signed source
coverage through rank 8. All retain a dense positive-affine destination and
the same exact address proof.

The separately named positive and signed-affine packed8 and packed16 profiles
preserve same-dtype bool/i8/u8 and i16/u16/f16/bf16 bits at ranks 1 through 8. One
invocation owns all four destination bytes or both destination halfwords,
preventing concurrent read-modify-write races in raw `array<u32>` storage.
Partial final words preserve every unrelated destination byte. The semantic
profiles require a dense destination, reject-on-invalid-source, and
either a positive-affine source or the separately named signed-affine source
profile. A nonnegative view byte offset must rebase every proved source access
into the root allocation. The current WebGPU backends additionally require
static launch, a word-aligned destination, and word-sized root allocations.
They perform no boolean canonicalization, arithmetic, or conversion, need no
`shader-f16` feature, and make no widened-arithmetic claim.

The separately named positive and signed-affine word64 profiles preserve
same-dtype f64/i64/u64 bits at ranks 1 through 8. Each invocation copies one logical
element as two adjacent raw u32 words, so destination writes are disjoint. The
separate signed-affine source profile uses the same nonnegative rebased byte
address proof. The current WebGPU backend requires static launch, a
positive-affine dense destination, reject-on-invalid-source, aligned views, and
word-sized roots. It uses no native 64-bit WGSL type and grants no f64 or
64-bit integer arithmetic or conversion claim.

Root allocations are bound at offset zero as `array<u32>`, so ordinary values,
signed zero, infinities, NaN payloads, and integer bit patterns copy exactly.
Float padding initializes exact fill bits and performs the source load only
inside a structured `if`; integer profiles reject invalid source coordinates.
The lowerer never uses eager `select`, implicit robust-buffer zeroing, address
clamping, or ignored writes as semantics. Device execution validates storage,
dispatch, workgroup, and binding limits before submission and reports separate
pipeline, validation, memory, device-loss, and execution diagnostics.

Required Chromium 148 on Apple Metal 3 passes 75 complete-destination cases
through backend 3.7.0. Correctness artifact
`d7c57f3ff63ba63f0b987a21eb4cabdc9c30d5539d9babb78b21d074df1f1e6b`;
case set
`281e050c4037981e238edee404a21c6ee6cb15e88de925468abd1e1d89c23d58`;
device profile
`9589abc8fafb412d83194febaf210f7f89da7a580bf20d3272e1eef9dcda2f66`.

The advisory focused lane records a real skip when no adapter exists:

```bash
pnpm --filter @unlocalhosted/browsergrad-kernels test:browser:view-copy
```

The evidence gate is deliberately strict and fails on missing adapter/device:

```bash
pnpm --filter @unlocalhosted/browsergrad-kernels test:browser:view-copy:required
```

Passing Node tests or packed-tarball checks is not WebGPU conformance evidence.

### Verified semantic host graphs

The host-graph adapter consumes only exact verifier-issued graph authority plus
the exact opaque kernel/layout artifacts referenced by that graph. Preparation
expands verified topological nodes into one canonical view-copy dispatch per
rank, one raw-word copy per rank for version-1.1 whole-allocation copy nodes,
and pairwise rank-ordered reduction/replication steps. Version-1.2 terminal
materialization nodes select output readbacks after whole-graph success without
adding a GPU dispatch. Version-1.3 completion events preserve dependency order
and appear in the successful execution trace without adding a GPU command.
They do not claim a timestamp, queue fence, or external wait. Raw copy
preserves all dtype bits and requires allocation byte length divisible by four
in the portable WebGPU profile. It bounds expanded steps, generated WGSL,
preparation time, and the complete private host/GPU working set before device
access.
Version-1.4 fixed-count repeat bodies are lowered once through these same
canonical dispatch/copy/collective lowerers, then their frozen steps are reused
for bounded static expansion. Preparation checks
cancellation and time throughout expansion and enforces the step ceiling after
every body node. A successful trace reports the repeat node, exact iteration
count, and body-node IDs; no completion is published on failure. This is
bounded static expansion, not runtime branching or a dynamic loop claim.
Version-1.5 conditionals pre-lower both bounded linear branches through the
same canonical lowerers. The backend requires equal lowered execution shapes,
accounts exactly one selected branch while conservatively binding the union of
branch resources/modules, and selects only from the complete captured
external-input u32 predicate before device work. Branch identity is included
in the backend specialization hash and successful trace. This is bounded
host-known selection, not shader, GPU-derived, or mid-graph control flow.
Version-1.6 conditionals use the same pre-lowered branch path but name a
required runtime u32 control. Execution accepts exactly the graph's unique
control set, rejects missing, duplicate, unknown, or greater-than-u32 values,
and admits that set before copying inputs. It captures controls with all inputs
before its first device access or await. Zero selects else and nonzero selects
then; branch identity remains part of specialization and terminal evidence.
This adds bounded request-time host control, not GPU/backend-derived
predicates, mid-graph feedback, dynamic launches, or runtime loop counts.
Version-1.7 adds one `resource-u32-branch-sequential` conditional over an exact
rank-local temporary `u32` with an ordered graph writer. Both branches are
still pre-lowered and prewarmed. Execution uploads private graph storage once,
runs the exact prefix over resident buffers, reads back only the four-byte
predicate, selects zero as else or nonzero as then, and resumes the authorized
branch plus suffix over the same resident buffers. The single timeout,
cancellation, device-in-flight, cleanup, numerical-status, and terminal-output
contract spans both submissions; no intermediate graph resource or partial
output is published. This is one explicit bounded GPU-produced feedback point,
not shader branching, nested control, a runtime loop count, or dynamic launch.
Version-1.8 adds `runtime-u32-count-sequential` repetition. Preparation lowers
and prewarms the complete artifact-bounded maximum schedule once. Execution
captures the required request-time `u32-count` before device access, rejects a
value above the artifact maximum before copying inputs, and selects exactly
the authorized step slots for zero through the maximum iterations without
pipeline compilation or slot reordering. Successful traces report actual
steps, collective/copy counts, and the completed repeat count while prepared
budgets retain the maximum. This is bounded host-known work selection, not a
GPU/backend-derived loop or dynamic launch.
Version-1.9 adds top-level `dynamic-dispatch` with an artifact-capped positive
request-time logical-prefix element count. The portable profile lowers one
canonical view-copy program with a runtime-prefix uniform guard before
coordinate or address evaluation. Preparation compiles the maximum once;
execution reuses the same pipeline authority and changes only the validated
logical dispatch count and four-byte uniform. The sequence runner submits
`ceil(elementCount / workgroupSize)` x workgroups; every tail invocation
returns before resource access, so arbitrary positive prefixes remain exact.
Zero, above-maximum, and nested use fail before device execution.
Version-1.10 adds one `resource-u32-count-sequential` repeat. Preparation
prewarms its complete artifact-maximum schedule. Execution submits the ordered
producer prefix over private resident graph buffers, reads back only the exact
four-byte rank-local count, rejects a value above the maximum, and submits
zero through that maximum iterations plus the suffix through the already
authorized exact slots. The same timeout, cancellation, device owner,
device-loss, cleanup, numerical-status, and terminal-publication contract spans
both stages; the count and intermediate resources never become graph outputs.
Version-1.11 adds one `resource-u32-prefix-elements` dynamic dispatch under the
same graph-wide feedback bound. Preparation prewarms the artifact-maximum
runtime-guarded launch once. Execution submits the ordered producer prefix over
resident private buffers, reads back only the four-byte count, rejects zero
and above-maximum values, substitutes that count as both logical launch
geometry and the prefix guard in the already-authorized rank slots, and
submits the guarded dispatch plus suffix. The same timeout, cancellation,
device ownership, device-loss, cleanup, numerical-status, and
terminal-publication contract spans both stages.
Version-1.12 adds `runtime-u32-rectangular-prefix` for rank-2 and rank-3
view-copy domains. Preparation prewarms one maximum-shape program and pipeline
slot. Execution captures one positive bounded u32 extent per semantic axis,
changes only true 2D/3D launch geometry plus one 16-byte uniform, and maps
WebGPU x/y/z to the canonical trailing-to-leading tensor axes. Every out-of-
rectangle or physical tail invocation returns before coordinate, address, or
resource evaluation. The CPU oracle executes the same selected rectangle and
completion records retain the logical extents plus their product.
Version-1.13 adds one `resource-u32-rectangular-prefix` dispatch under the
same graph-wide feedback bound. Preparation retains the same maximum-prewarmed
rectangular program and exact rank slots. Execution submits all ordered extent
producers in one prefix stage, reads back exactly one four-byte rank-local u32
per axis, rejects zero or above-maximum values, and substitutes only the
validated 2D/3D geometry plus the existing 16-byte uniform before submitting
the suffix. Intermediate extents remain private and the lifecycle remains one
timeout, cancellation, device-owner, loss, cleanup, and terminal-publication
contract.
Version-1.14 extends request-time rectangular dispatch to rank 4. The canonical
16-byte uniform already carries four extents; WebGPU keeps x/y on the trailing
axes and flattens the two leading selected axes into z, then reconstructs their
logical coordinates with guarded quotient/remainder operations before any
semantic address evaluation. The maximum program and pipeline slot remain
stable across selected extents, while x/y/z device admission independently
checks the flattened leading-axis product.
Version-1.15 extends produced-resource rectangular dispatch to rank 4 without
adding another feedback boundary. Execution reads all four ordered private u32
extent resources after one resident-buffer prefix stage, validates them as one
bounded selection, and reuses the version-1.14 flattened-z launch, 16-byte
uniform, maximum program, and exact pipeline slots for the suffix.
Version-1.16 extends request-time rectangular dispatch to rank 5. The separate
rank-5 view-copy profile uses one 32-byte, eight-u32 uniform whose first five
words carry extents. WebGPU keeps axes 4 and 3 on x/y, flattens axes 0 through
2 into z, guards their selected product, and reconstructs all three leading
coordinates with mixed-radix quotient/remainder operations before semantic
evaluation. The maximum program and exact pipeline slot remain stable across
selected rectangles. Version-1.17 extends produced-resource rectangular
dispatch to rank 5 without another feedback boundary. Execution reads all five
ordered private u32 extent resources after one resident-buffer prefix stage,
validates them as one bounded selection, and reuses the version-1.16
flattened-z mapping, 32-byte uniform, maximum program, and exact pipeline slots
for the suffix.
Each dynamic rank step charges an aligned 16-byte GPU uniform allocation for
ranks 1 through 4 or 32 bytes for rank 5, and either a four-byte linear value,
16-byte rank-2-through-rank-4 rectangular value, or 32-byte rank-5 rectangular
value to the graph's transient working-set plan. Pipeline
admission derives actual maximum total, storage, and uniform binding counts
from every reachable program, admits x/y/z workgroup counts independently, and
rejects the device before resource creation when any relevant limit is
insufficient.

`prepareSemanticHostGraphWebGpuPipeline()` binds the exact graph to one
`GPUDevice` and compiles every unique pipeline admitted at every exact step
slot, including both sides of each conditional. Its immutable
`browsergrad.host-graph.webgpu-pipeline@1` result hashes the graph, backend,
WGSL modules, schedule, negotiated features, relevant limits, and numerical
policies and enforces a caller-lowerable maximum under the fixed 128-pipeline
portable ceiling. `runSemanticHostGraphWebGpuPipeline()` reuses that authority while
creating private request buffers; copied, destroyed, cross-device, or
program-mismatched authorities fail closed. Runtime repetition uses a
validated strictly increasing exact-slot selection, so skipped maximum slots
do not grant reordering or substitution. Call
`destroySemanticHostGraphWebGpuPipeline()` when the hot-run lifetime ends.
`runSemanticHostGraphWebGpu()` remains the convenience API and delegates
through an ephemeral instance of the same authority path after synchronously
capturing caller inputs and controls.

Execution snapshots every rank-local input before its first await, creates only
private zero-initialized temporary/output storage, checks device allocation,
binding, workgroup, and dispatch limits, and permits one graph run per device.
F32 collective operands/results must be finite; min/max preserves CPU
signed-zero selection. Integer sums wrap at 32 bits and integer min/max is
exact. Cancellation or timeout suppresses stale results while submitted work
finishes cleanup; device loss clears both generic and kernel-device caches.
Outputs are fresh byte copies published only after all dispatches, collective
status checks, and readbacks succeed.

The advisory and strict actual-device lanes are:

```bash
pnpm --filter @unlocalhosted/browsergrad-kernels test:browser:semantic-host-graph
pnpm --filter @unlocalhosted/browsergrad-kernels test:browser:semantic-host-graph:required
pnpm --filter @unlocalhosted/browsergrad-kernels test:browser:semantic-host-graph:performance
pnpm --filter @unlocalhosted/browsergrad-kernels test:browser:semantic-host-graph:performance:required
```

The separately retained performance record compares version-1.4
`fixed-count-sequential` control with a version-1.2 statically unrolled graph
that has bit-exact CPU/WebGPU outputs, equal element-operation counts, and the
same eight expanded WebGPU steps. The f32 two-rank, 65,536-element workload
prewarms both exact device-bound pipeline authorities, then uses eight warmups
and twelve alternating paired samples around authority-bound execution,
including readback and queue drain. The current Apple Metal 3 observation
records 2.10 ms candidate and 2.10 ms unrolled medians under backend 1.20.0; it
asserts no superiority or regression threshold.

This profile is a bounded DAG plus fixed-count and bounded request-time
u32-count sequential repetition, one bounded produced-resource u32-count
repeat, and bounded request-time or produced-resource
arbitrary positive one-dimensional prefix dynamic dispatch,
rank-2-through-rank-8 request-time or produced-resource rectangular prefix
dynamic dispatch, plus exact two-dispatch fanout for one shared linear or
rectangular produced selection,
captured-input/runtime-control conditionals, and one produced-resource
conditional or repeat count, with at most two compatible produced-resource
dispatch consumers in one aggregate feedback stage or one exact linear
producer chain across two reported feedback stages. It does
not claim a third feedback stage, repeated conditional/repeat/rectangular
control, device-side feedback, or rank-9-and-higher dynamic domains,
nested/device-side branching, event
timestamps/external waits, transport/topology, a worker mesh, or native
collectives. Its performance record is observational and does not establish a
general performance advantage.

## Quick start

### One-shot kernel (host round-trip)

```ts
import { createDevice, kernels, tensor, matmulTiled } from "@unlocalhosted/browsergrad-kernels";

const device = await createDevice();
const A = tensor([2, 3], new Float32Array([1, 2, 3, 4, 5, 6]));
const B = tensor([3, 2], new Float32Array([7, 8, 9, 10, 11, 12]));

const C = await matmulTiled(device, A, B);   // tiled GEMM — production path
console.log(C.shape, C.data);                 // [2, 2], Float32Array(4)
```

### Pure-JS reference (no WebGPU required)

```ts
import { reference } from "@unlocalhosted/browsergrad-kernels/reference";
const C = reference.matmul(A, B);  // identical surface; CPU only
```

For CS336 A2-style FlashAttention rubrics, import
`referenceFlashAttention()` and `referenceFlashAttentionBackward()` from the
top-level package. The forward oracle returns `{ output, logSumExp }`, matching
the upstream test's saved-LSE contract; the backward oracle recomputes softmax
probabilities and returns Q/K/V gradients without requiring PyTorch autograd,
Triton, or CUDA.

For GPU Puzzles and CS149 A3-style kernel-concept rubrics, import
`runThreadGrid()`, `referenceSaxpy()`, and `referenceExclusiveScan()`.
`runThreadGrid()` runs a browser-safe 1D thread/block callback, records
per-thread reads/writes, and reports out-of-bounds access instead of hiding
missing guards. It is a correctness and pedagogy oracle, not a native CUDA
performance runner. `simulateCuda1DGrid()` remains as a compatibility alias for
rubrics that intentionally use CUDA vocabulary.

For a durable author-once path, define a tiny BrowserGrad Kernel1D program and
run it through both adapters:

```ts
import {
  createDevice,
  defineKernel1DProgram,
  emitKernel1DProgramWgsl,
  runKernel1DProgramReference,
  runKernel1DProgramWebGpu,
} from "@unlocalhosted/browsergrad-kernels";

const program = defineKernel1DProgram({
  name: "saxpy_guarded",
  inputLength: 4,
  outputLength: 4,
  parameters: { a: 2 },
  launch: { blocks: 1, threadsPerBlock: 8 },
  body: [{
    op: "if",
    condition: { op: "lt", left: { op: "threadId" }, right: { op: "outputLength" } },
    body: [{
      op: "write",
      index: { op: "threadId" },
      value: {
        op: "add",
        left: {
          op: "mul",
          left: { op: "param", name: "a" },
          right: { op: "read", index: { op: "threadId" } },
        },
        right: { op: "outputRead", index: { op: "threadId" } },
      },
    }],
  }],
});

const simulated = runKernel1DProgramReference(program, {
  initialInput: [1, 2, 3, 4],
  initialOutput: [10, 20, 30, 40],
});
const wgsl = emitKernel1DProgramWgsl(program);
const device = await createDevice();
const gpu = await runKernel1DProgramWebGpu(device, program, {
  initialInput: [1, 2, 3, 4],
  initialOutput: [10, 20, 30, 40],
});
```

This is the first HipScript-inspired kernel-authoring spine: BrowserGrad owns
the small IR and reference executor, then CUDA/HIP-like syntax can grow as a
frontend. The shipped path already has explicit grid/thread semantics, scalar
params, input/output buffer reads, deterministic traces, WGSL source generation,
and real browser WebGPU dispatch without shipping a browser LLVM toolchain.

For hot WGSL paths, keep storage buffers resident:

```ts
import {
  createDevice,
  createWgslStorageBuffer,
  defineWgslKernelProgram,
  readWgslStorageBuffer,
  runWgslKernelProgram,
  writeWgslStorageBuffer,
} from "@unlocalhosted/browsergrad-kernels";

const device = await createDevice();
const program = defineWgslKernelProgram({
  name: "inc",
  workgroupSize: [4, 1, 1],
  bindings: [{ kind: "storage", name: "x", valueType: "f32" }],
  wgsl: `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(4, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < arrayLength(&x)) { x[gid.x] = x[gid.x] + 1.0; }
}`,
});
const x = createWgslStorageBuffer(device, {
  valueType: "f32",
  data: new Float32Array([1, 2, 3, 4]),
});
writeWgslStorageBuffer(device, x, new Float32Array([5, 6]), Float32Array.BYTES_PER_ELEMENT);

await runWgslKernelProgram(
  device,
  program,
  { buffers: {}, residentBuffers: { x }, readback: [] },
  { dispatchCount: [4, 1, 1] },
);

const out = await readWgslStorageBuffer(device, x);
```

For repeated dispatches, prepare the sequence once:

```ts
const prepared = await prepareWgslKernelProgramSequence(
  device,
  [{ program, launch: { dispatchCount: [4, 1, 1] } }],
  { buffers: {}, residentBuffers: { x }, readback: [] },
);
await prepared.run();
await prepared.run({ readback: [] });
await prepared.run({ readback: [], awaitCompletion: true });
await prepared.run({ uniforms: { params: new Float32Array([3]) } });
prepared.destroy();
```

Prepared uniform updates rewrite existing uniform buffers and reuse bind groups.
Use `stepUniforms` only when sequence steps need different values for the same
uniform binding name. Use `awaitCompletion: true` for no-readback timing gates
or platform watchdogs that need command completion, not only command submission.

Use `storageMetadata` when one physical storage buffer is viewed through
multiple WGSL value types, or when state needs readback even though no step binds
it:

```ts
await runWgslKernelProgramSequence(
  device,
  [{ program, launch, storageAliases: { floats: "raw" } }],
  {
    buffers: { raw: new Uint32Array(1024), state: new Uint32Array([0]) },
    storageMetadata: {
      raw: { valueType: "u32", compatibleValueTypes: ["f32"] },
      state: "u32",
    },
    readback: ["raw", "state"],
  },
);
```

### Kernel rubric assertions

```ts
import {
  createBrowsergradKernelRubric,
  kernels,
  reference,
} from "@unlocalhosted/browsergrad-kernels";

const rubric = createBrowsergradKernelRubric(ctx);

const actual = await kernels.matmul(device, A, B);
const expected = reference.matmul(A, B);
rubric.assertCloseTensor("matmul_tiny", actual, expected, { atol: 1e-4 });
```

`createKernelRubric()` is CPU-only and works without WebGPU. It records
pass/fail assertions, checks tensor shapes, compares values with absolute and
relative tolerance, and emits compact previews plus first failing index/max
error for learner-facing JS rubrics. Non-finite actual or expected values fail
the comparison instead of slipping through tolerance math.
`kernelRubricFailureToAssertionDetails()` formats structured rubric details into
`expected` / `actual` strings for BrowserGrad-style assertion callbacks.
`createBrowsergradKernelRubric(ctx)` is the convenience adapter for
`runAssignmentJavascriptRubric()` contexts and any compatible assertion target.

### Realizer-tier (chained ops, GPU residency)

```ts
import {
  createDevice,
  matmulTiledDirect,
  materializeFloat32,
  uploadFloat32,
} from "@unlocalhosted/browsergrad-kernels";

const device = await createDevice();

const x = uploadFloat32(device, xData);
const w1 = uploadFloat32(device, w1Data);
const w2 = uploadFloat32(device, w2Data);

// (x @ w1) stays on the GPU; only the final readback crosses host.
const mid = matmulTiledDirect(device, x, w1, M, K, N);
const out = matmulTiledDirect(device, mid.buffer, w2, M, N, N);
const result = await materializeFloat32(device, out.buffer, out.byteLength);

mid.buffer.destroy();
out.buffer.destroy();
```

### Hand the bridge to browsergrad-jit

```ts
import { createDevice, createWebGpuRealizerBridge } from "@unlocalhosted/browsergrad-kernels";

const device = await createDevice();
const bridge = createWebGpuRealizerBridge(device);

// Expose the bridge to Pyodide
pyodide.registerJsModule("_bg_webgpu_bridge", bridge);
```

```python
# In Python (Pyodide)
import browsergrad_jit as bg
from js import _bg_webgpu_bridge
bg.register_webgpu_bridge(_bg_webgpu_bridge)

out = bg.realize_webgpu(model(x))   # all matmuls + fused chains run on the GPU
```

### Runtime WGSL codegen

```ts
import { generateFusedWgsl, fusedElementwiseDirect } from "@unlocalhosted/browsergrad-kernels";

// Produces a self-contained WGSL compute shader for the chain.
// Hash of the ops list = pipeline cache key.
const wgsl = generateFusedWgsl(
  [
    ["ADD", -1, -2],   // step0 = in0 + in1
    ["EXP", 0, 0],     // step1 = exp(step0)
    ["DIV", 1, -1],    // step2 = step1 / in0
  ],
  2,                    // num inputs
);
```

## Browser testing

```bash
pnpm test:browser
```

Launches Chromium via Playwright with WebGPU enabled. Runs against a real `GPUDevice`. On macOS the browser is headed (Metal driver only exposed when visible); on Linux CI set `BG_BROWSER_HEADLESS=1`.
Use `pnpm test:browser:open` when you want the browser window to stay open for
inspection; quit with `q`.

7 scenarios: adapter info, naive vs tiled matmul, residency contract (3 uploads + 1 readback chained matmul), fused-elementwise codegen output matches NumPy semantics, fused row-wise online-attention baseline (known-issue advisory), end-to-end `WebGpuRealizerBridge.matmul`.

Real-WebGPU CI is the only reliable way to catch shader-level bugs — NumPy mocks pass everything green even when the WGSL is wrong. A numerical issue in the direct online-attention baseline, tracked in the changelog, was caught this way.

## API stability

| Surface | Stability |
|---|---|
| `kernels.*`, `matmul`, `matmulTiled`, `softmax`, `relu`, `gelu`, `layernorm`, `attention` | Semver-stable across `0.x` |
| `runDirect`, `matmulTiledDirect`, `fusedElementwiseDirect`, `rowWiseOnlineAttentionDirect` | Semver-stable |
| `flashAttentionDirect` | Deprecated compatibility alias; retained through the documented removal window |
| `materializeFloat32`, `uploadFloat32` | Semver-stable |
| `createWebGpuRealizerBridge`, `WebGpuRealizerBridge` interface | Semver-stable; new methods added additively |
| `KernelError` | Semver-stable |
| WGSL source strings | **Internal.** Tuned freely. |
| Pipeline cache keys | **Internal.** Same WGSL → same key, but the encoding may change. |

## License

[MIT](LICENSE).
