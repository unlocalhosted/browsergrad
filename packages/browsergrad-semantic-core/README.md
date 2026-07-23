# `@unlocalhosted/browsergrad-semantic-core`

Versioned `0.x` package for BrowserGrad's canonical semantic wire format,
value/layout model, kernel/schedule meaning, and requirement protocol. The API
remains intentionally narrow and unstable while the schemas prove themselves
across two frontends and two backends.

Only explicit subpaths exist:

- `@unlocalhosted/browsergrad-semantic-core/schema`
- `@unlocalhosted/browsergrad-semantic-core/layout`
- `@unlocalhosted/browsergrad-semantic-core/kernel`
- `@unlocalhosted/browsergrad-semantic-core/schedule`
- `@unlocalhosted/browsergrad-semantic-core/requirement`

There is no root barrel. The package must remain browser-safe and cannot import
compiler frontends, framework packages, runtimes, or device APIs. Public
consumers depend only on explicit subpaths; new subpaths require a concrete
cross-package consumer and architecture evidence.

`/requirement` contains immutable versioned assignment requirement definitions
and environment-scoped resolution records. An available resolution must bind
one provider ID, one closed browser/simulated/external mode, and a deterministic
evidence-ID set. An unavailable resolution carries no invented provider or
evidence. The protocol records readiness facts only; it owns no semantic
lowering or backend policy.

`/kernel` currently contains one concrete `browsergrad.kernel@1` operation: a
verified, materializing view copy over a verified `browsergrad.layout@1`
artifact. The initial portable execution profile is intentionally narrow:
same-dtype f32, rank 2 or 3, global-memory views, explicit source-read and
destination-write effects, disjoint alias sets, and either reject or exact-bit
fill behavior for invalid source coordinates. Generic operation verification
is separate from this lowering profile.

`/kernel` also defines one frontend-neutral logical GEMM tile with exact dense
f32 operand/view roles, boundary policy, increasing-K accumulation order, and
an explicit prohibition on contraction and reassociation. `/schedule` binds an
exact verified logical GEMM semantic hash to physical tiles, cooperative
workgroup staging, full-workgroup boundary participation, uniform
acquire-release barriers, scalar vectors, and masked memory effects. It owns no
logical dtype, numerical, frontend, or backend meaning. Device backends must
either prove the logical numerical contract for a declared input domain or
refuse it; selecting a schedule cannot weaken the contract.

The initial `/kernel` numerical proof is the closed
`browsergrad.kernel.gemm-exact-f32-input@1` certificate. It synchronously
snapshots direct, fixed, unshared host allocation bytes and admits only finite
nonnegative integer f32 inputs whose products and complete output sums remain
at most 2^24. On that narrow domain every relevant f32 operation is exact, so
backend contraction or reassociation cannot change the strict logical result.
Certificates retain private snapshots and return fresh copies for immediate
upload; they do not authorize arbitrary or previously resident GPU buffers and
make no general-f32 preservation claim.

`/kernel` also owns the closed, frontend-neutral
`browsergrad.kernel.attention-forward@1` meaning. It binds verified rank-4
Q/K/V/output views, non-causal or upper-left causal masking, the exact f32
inverse-square-root query-depth scale, stable softmax phases, finite-score and
finite-online-state preconditions, a named `1e-4` absolute-or-relative
comparison policy, pairwise-disjoint effects, and an explicit forward-only VJP
refusal. The dense constructor admits positive portable-u32 dimensions and
limits query/value depths to 256. Workgroups, tiles, staging, barriers, WGSL,
backend facilities, and performance claims are absent by construction; those
require separate schedule and execution artifacts. No CPU or WebGPU execution
claim is made by this semantic slice.

The initial separate `/schedule` attention artifact binds that exact logical
hash to physical query-row and key-row tiles, increasing key traversal,
cooperative single-buffered K/V workgroup staging, a tile-wise online-softmax
recurrence, full-workgroup barrier participation, and fail-closed boundary and
logical-mask placement. Invalid or causally masked keys are excluded before
the tile maximum or any online-state update; zero-filled staging alone never
makes them scores. The schedule remains backend-neutral and scalar. It does
not claim target legality, numerical preservation, execution, performance, or
a named fused-attention implementation tier.

Schedule specialization composes one authority-bound logical attention proof
with one exact schedule and derives bounded workgroup size, K/V staging bytes,
per-invocation private-state elements, key-tile count, and three-dimensional
dispatch geometry. Distinct 8x8 and 8x16 schedules reuse the same logical proof
but retain distinct schedule-specialization hashes. Device feature/limit
admission and numerical preservation remain backend responsibilities.

The schedule-independent CPU oracle prepares the logical artifact directly. It
proves the initial dense row-major address profile under explicit element,
scalar-operation, evaluation-step, time, and cancellation limits; snapshots
fixed unshared Q/K/V bytes before yielding; rejects non-finite inputs, scores,
exponentials, denominators, or outputs; and delays every destination write
until the complete result is valid. Its canonical f32 evaluation rounds each
product, sum, scale, exponential result, probability, and weighted-value step.
The accompanying comparator implements the declared `1e-4` absolute-or-
relative policy and rejects non-finite outputs. This is CPU reference evidence,
not WebGPU execution or preservation evidence.

Frontends construct the operation through one `/kernel` sink rather than
assembling allocation, alias, index-map, view, and operation IDs themselves.
`createVerifiedViewCopyArtifacts` accepts layout construction algebra plus
explicit allocation geometry, snapshots it as canonical JSON, normalizes both
layouts, fixes source/destination role order, verifies both artifacts, and
returns their semantic hashes and canonical role IDs. The
`createVerifiedDensePermutationViewCopyArtifacts` wrapper accepts only source
shape, axes, and dtype; output shape, row-major strides, storage size, effects,
and disjoint materialization are derived. Transport producer/artifact metadata
is passed separately and cannot affect semantic hashes.

`/layout` also supports standalone layout expressions for frontend facts that
define index algebra but no tensor storage. `prepareLayoutExpression` produces
an authority-bound `browsergrad.layout@1` value with one verified index map and
zero allocations/views; `traceLayoutExpressionCoordinate` reports element
locations and logical/predicate bounds. It deliberately makes no dtype,
allocation, byte-address, alias, effect, CPU, or GPU claim. Storage-bearing
frontends must add those facts through the view/kernel contracts instead of
inferring them from layout size or codomain extent.

The backend-neutral specialization step resolves bindings and hashes once,
compiles the canonical index evaluators, proves guarded source access and a
dense injective destination, and derives a binding-sensitive specialization
hash shared by CPU and device backends. Independent element,
aggregate-evaluation-step, and optional prepared-scratch budgets bound work;
wall-time and abort checks yield through the browser scheduler with a timer
fallback. CPU execution checks native typed-array slots,
exact allocation lengths, declared alignment, overlap, and shared-memory
exclusion; it never turns an invalid address into clamping or implicit
zero-fill.

This is the semantic/reference contract, not a blanket GPU-support claim.
Kernels-owned WGSL plus the compiler and JIT adapters have strict actual-device
evidence for their declared Gate 2 profiles at the exact revisions recorded in
the implementation ledger. Version `0.3.0` is the selected identity for the
new logical-GEMM and schedule API and has not been published. Release CI must
repeat the exact-source evidence gates.

Current status and evidence live in
[`docs/internal/package-requirements-implementation-ledger.md`](../../docs/internal/package-requirements-implementation-ledger.md).

## Cross-language reference

`python/browsergrad_semantic_core.py` is the dependency-free Python reference
for the current closed `browsergrad.layout@1` wire contract. It independently
decodes, validates, normalizes, canonicalizes, hashes, and traces the golden
fixtures under `fixtures/layout-v1/`; it is a parity and review oracle, not a
second runtime implementation or a stable Python package API.

The Vitest parity suite runs both implementations over positive fixtures and a
differential rejection corpus. Any schema change must update both references,
their fixtures, and the implementation ledger in the same coherent change.
