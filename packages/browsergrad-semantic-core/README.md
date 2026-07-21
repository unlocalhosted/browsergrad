# `@unlocalhosted/browsergrad-semantic-core`

Versioned `0.x` package for BrowserGrad's canonical semantic wire format and
value/layout model. The API remains intentionally narrow and unstable while
the schemas prove themselves across two frontends and two backends.

Only explicit subpaths exist:

- `@unlocalhosted/browsergrad-semantic-core/schema`
- `@unlocalhosted/browsergrad-semantic-core/layout`
- `@unlocalhosted/browsergrad-semantic-core/kernel`
- `@unlocalhosted/browsergrad-semantic-core/schedule`

There is no root barrel. The package must remain browser-safe and cannot import
compiler frontends, framework packages, runtimes, or device APIs. Public
consumers depend only on explicit subpaths; new subpaths require a concrete
cross-package consumer and architecture evidence.

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
