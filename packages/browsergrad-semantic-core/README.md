# `@unlocalhosted/browsergrad-semantic-core`

Versioned `0.x` package for BrowserGrad's canonical semantic wire format,
value/layout model, kernel/schedule meaning, and capability/requirement
protocols. The API remains intentionally narrow and unstable while the schemas
prove themselves across two frontends and two backends.

Only explicit subpaths exist:

- `@unlocalhosted/browsergrad-semantic-core/schema`
- `@unlocalhosted/browsergrad-semantic-core/layout`
- `@unlocalhosted/browsergrad-semantic-core/kernel`
- `@unlocalhosted/browsergrad-semantic-core/schedule`
- `@unlocalhosted/browsergrad-semantic-core/graph`
- `@unlocalhosted/browsergrad-semantic-core/capability`
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

`/capability` separates immutable semantic/backend definitions from
program-or-artifact-scoped lowering decisions. A positive decision must name a
backend-owned execution tier and a capability-owned preservation level;
conditional decisions must retain their exact feature, limit, or runtime-guard
requirements. Refusal/unknown records require a reason and cannot claim
preservation. Static definitions contain no support state or evidence outcome.

`/kernel` currently contains one concrete `browsergrad.kernel@1` operation: a
verified, materializing view copy over a verified `browsergrad.layout@1`
artifact. The portable execution profiles admit same-dtype f32, i32, or u32
rank 1 through rank 5 global-memory views with explicit source-read and
destination-write effects and disjoint alias sets. Float profiles admit reject
or exact-bit fill behavior for invalid source coordinates; integer profiles
require rejection. Generic operation verification is separate from these
lowering profiles. Rank 5 uses the distinct
`browsergrad.view-copy.positive-affine-rank5-word32@1` profile so the existing
rank-1-through-rank-4 identities retain their exact meaning. The separate
`browsergrad.view-copy.signed-affine-rank2-rank3-word32@1` profile admits
negative coordinate scales only in rank-2/rank-3 source maps and predicates;
the distinct
`browsergrad.view-copy.signed-affine-rank4-rank5-word32@1` profile provides
the same source capability at ranks 4 and 5, and
`browsergrad.view-copy.signed-affine-rank1-word32@1` covers rank 1. The
destination remains positive-affine, dense, and exactly proved.

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

`/graph` defines the closed `browsergrad.host-graph@1` profile for bounded,
backend-neutral multi-dispatch DAGs and explicit all-reduce meaning. Program
version 1.1 additively admits whole-allocation byte copies between distinct
same-dtype, same-size per-rank resources while retaining version-1.0 reads.
Version 1.2 requires every declared output to have exactly one terminal
`host-readback-after-graph-success` materialization node. That node is an
ordered read effect: it must follow the output's final writer, cannot expose an
input or temporary resource, cannot be duplicated, and cannot have dependents.
Version 1.3 adds unique `completion-after-dependencies` events. Events carry
no resource effect, timing value, queue identity, or external wait authority;
they are dependency-ordered completion markers reported only with a successful
whole-graph result.
Version 1.4 adds fixed-count sequential repetition. A repeat body is a
nonempty, bounded linear sequence of dispatch, all-reduce, and copy nodes,
with a positive iteration count fixed in the verified artifact. Nested
repetition, events, materialization, and runtime-derived counts
are rejected. Verification applies the same resource effects and hazard rules
to the aggregate body, bounds both iterations and expanded nodes, and requires
all top-level and body node IDs to be globally unique.
Version 1.5 adds `input-u32-branch-sequential` conditionals. The predicate is
exactly one rank-local, four-byte, aligned external-input `u32`; zero selects
the else body and any nonzero value selects the then body. Both branches are
nonempty bounded linear dispatch/all-reduce/copy sequences with equal node-kind
structure. Nested control, events, materialization, backend-derived predicates,
and runtime-generated work remain rejected. Verification conservatively merges
all possible reads/writes, treats a write as guaranteed only when both branches
write the resource, preserves exact one-branch work counts, and keeps
top-level plus both branch bodies in one global node-ID namespace.
Version 1.6 adds `runtime-u32-branch-sequential` conditionals. The predicate
names one required host execution control rather than a graph resource; zero
selects else and any other admitted `u32` selects then. The verifier retains
the same bounded equal-shape branches, conservative effects, work accounting,
and global node-ID rules, limits the graph to 64 unique controls, and grants no
GPU/backend-derived predicate, mid-graph feedback, dynamic launch, or
runtime-derived repetition authority.
Version 1.7 adds one `resource-u32-branch-sequential` conditional. Its
predicate is an exact rank-local, four-byte, aligned temporary `u32` that must
have an ordered graph writer before the conditional. Zero selects else and
nonzero selects then. The same bounded branch-shape, conservative-effect,
exact-work, and global-ID rules apply. The one-node profile makes a single
mid-graph feedback boundary explicit without admitting nested control,
runtime-derived loop counts, dynamic launches, or opaque callbacks.
Version 1.8 adds `runtime-u32-count-sequential` repetition. The verified
artifact names one required request-time `u32-count` control and a positive
maximum iteration count no greater than the existing repeat ceiling. Values
from zero through that maximum execute the bounded linear body exactly that
many times; a larger value fails during request admission before input bytes
are copied. Verification charges the maximum expanded work and conservatively
treats runtime-repeat writes as non-guaranteed because zero iterations perform
no write. This is bounded host-known work selection, not a GPU/backend-derived
loop or dynamic launch.
Version 1.9 adds top-level `dynamic-dispatch` with
`runtime-u32-prefix-elements` mode. The node binds one required request-time
`u32-prefix-element-count` control and a positive artifact maximum no larger
than the verified view-copy semantic domain. Values from one through the
maximum execute exactly that logical linear prefix; zero and larger values
fail before input copying. Dynamic dispatch cannot appear in repeat or
conditional bodies, and its maximum work enters preparation budgets.
Version 1.10 adds one `resource-u32-count-sequential` repeat. Its
`iterationSource` names an exact rank-local, four-byte, aligned, zero-filled
temporary `u32` with an ordered graph writer. Zero through a positive artifact
maximum execute the bounded linear body exactly that many times. The body
cannot access the captured count resource, zero-possible body writes are not
guaranteed, and a graph admits only one produced-resource conditional or
repeat feedback node.
Version 1.11 adds `resource-u32-prefix-elements` dynamic dispatch. Its
`launchSource` names the same exact ordered temporary-u32 class, one through a
positive artifact maximum execute the exact logical prefix, and zero or larger
values fail closed. The dispatch cannot bind its launch source as data, maximum
work enters preparation budgets, and the graph-wide at-most-one produced-
resource feedback bound now spans conditional, repeat, and dynamic dispatch.
Version 1.12 adds `runtime-u32-rectangular-prefix` dynamic dispatch for rank-2
and rank-3 semantic view-copy domains. One required request-time
`u32-prefix-extent` control binds each axis, each positive artifact maximum
must fit the corresponding verified logical shape extent, and the product of
the maxima enters preparation work budgets. Zero, above-maximum, missing,
duplicate-axis, duplicate-control, rank-mismatched, and pre-version forms fail
before input copying.
Version 1.13 adds `resource-u32-rectangular-prefix` under the existing
at-most-one feedback-node bound. Its `launchSources` bind one distinct,
ordered, rank-local four-byte temporary u32 per axis. Each source writer must
dominate the dispatch; the dispatch cannot also bind a source as data. Zero,
above-maximum, duplicate-axis/resource, unordered, malformed, and pre-version
forms fail closed.
Version 1.14 extends only request-time rectangular dispatch to rank 4. The
four positive bounded controls and maxima retain the same canonical axis,
maximum-product admission, and fail-stop contracts; versions 1.12 and 1.13
continue to reject rank 4 exactly. Version 1.15 extends produced-resource
rectangular dispatch to rank 4 through four distinct ordered extent sources
under the same one-feedback-node bound. Older versions retain their exact rank
limits. Version 1.16 extends only request-time rectangular dispatch to rank 5
under the distinct rank-5 view-copy profile. Five positive bounded controls,
their canonical axes, and the maximum product preserve the existing admission
and fail-stop contracts. Version 1.17 extends produced-resource rectangular
dispatch to rank 5 through five distinct ordered extent sources under the same
one-feedback-node bound. Older versions retain their exact rank limits.
Each resource carries per-rank multiplicity, exact dtype, allocation byte
length, alignment, and input/temporary/output role. Input resources require
external bytes; temporary and output resources are deterministically
zero-filled before the first node. Dispatches reference opaque verified kernel
and layout artifacts, retain exact dimension bindings, and derive their
read/write effects and resource geometry from those artifacts; callers cannot
declare effects independently. Verification rejects cycles, dangling
references, read-before-write, input mutation, unordered hazards, incompatible
dtype or allocation bindings, invalid initialization or collective
participants/numerical policy, and graphs above the fixed node, edge, rank,
artifact, or 1 GiB aggregate-resource ceilings; the byte ceiling accounts for
rank multiplicity. Copy effects enter the same dependency/hazard analysis and
cannot mutate inputs, self-copy, reinterpret dtype, or partially copy an
allocation. The failure model is fail-stop with no partial output commit.

The authority-bound `browsergrad.host-graph.cpu-reference@1` executor consumes
the exact verified graph plus its opaque kernel/layout artifacts. It snapshots
all rank-local inputs, runs dispatches, exact byte copies, and f32/i32/u32
all-reduces against private storage, and exposes outputs only after the
complete graph succeeds. Version-1.2 results are selected only by verified
materialization nodes; the node adds no element work and preserves the
fail-stop publication point. Version-1.3 events likewise add no element work;
completed event IDs are returned only after every later node also succeeds.
Version-1.4 repeats execute the exact body sequentially for every fixed
iteration, check cancellation and time at iteration and body-node boundaries,
charge expanded element work before admission, and report completion only with
the successful whole-graph result.
Version-1.5 conditionals select only from the executor's private snapshot of
the declared input predicate. The CPU profile requires equal branch
element-operation counts, admits that exact one-branch cost before execution,
checks cancellation/time at every selected body node, and reports the selected
branch and body IDs only with successful whole-graph publication.
Version-1.6 conditionals require exactly one binding for every declared runtime
control and reject missing, duplicate, unknown, or out-of-range values. The
executor admits the complete control set before copying inputs, captures those
values with the inputs before its first await, and then uses the same verified
branch plan and fail-stop completion contract.
Version-1.7 conditionals read the predicate from private rank-local storage
only after all ordered producer work has executed. The CPU oracle therefore
proves the same graph-derived branch meaning without treating a request-time
value as the predicate or exposing intermediate storage.
Version-1.8 repeats admit the exact control set before copying inputs, reserve
memory and element-operation capacity for the artifact maximum, execute only
the captured count with per-iteration cancellation/time checks, report the
actual completed count, and return the actual element-operation total.
Version-1.9 dynamic dispatch admits its positive prefix count with the same
exact control snapshot, reserves the artifact maximum, executes only that
logical prefix, and reports both the completed element count and actual
element-operation total after whole-graph success.
Version-1.10 repeats read the count from private rank-local storage only after
its ordered producer has executed, reject values above the artifact maximum,
reserve maximum work, execute zero through that maximum with per-iteration
cancellation/time checks, and publish actual completion/work only with
whole-graph success.
Version-1.11 dynamic dispatch likewise reads its positive prefix count only
after the ordered producer, rejects zero and above-maximum values without
output publication, executes the exact canonical prefix for every rank, and
reports actual completion/work only with whole-graph success.
Version-1.12 dynamic dispatch captures every axis extent with the request,
executes exactly the selected dense rectangle against the full semantic
row-major coordinate domain, and reports both logical extents and their
product only after whole-graph success. Rank-2 and rank-3 CPU cases cover
partial and full rectangles plus hostile extent admission without input
observation.
Version-1.13 reads every rectangular extent from private rank-local storage
only after all ordered producers execute, applies the same per-axis bounds and
maximum-product admission, and publishes the same logical-extent/product
completion only after whole-graph success.
Version-1.14 executes the exact selected rank-4 dense rectangle through the
same coordinate-domain CPU path and reports all four extents plus their
product. Version-1.15 reads four produced extents through the same ordered
private-resource path and executes that identical rank-4 rectangle.
Version-1.16 executes the exact selected rank-5 dense rectangle through the
same generic coordinate-domain CPU path and reports all five extents plus
their product. Version-1.17 reads five produced extents through the same
ordered private-resource path and executes that identical rank-5 rectangle.
It enforces aggregate working-memory, element-operation, preparation-time, and
execution-time ceilings plus native cancellation. F32 collectives reduce
finite values in ascending participant-rank order, rounding after every sum;
non-finite operands or results fail before output commit. Integer sums wrap at
32 bits, while integer min/max is exact.

The semantic graph itself grants no execution authority, and its CPU reference
does not imply device execution. Compiler owns the first concrete producer: it
lowers one or more opaque prepared view-copy bindings into a verified linear
host graph with derived intermediate resources. Kernels separately owns the
authority-bound `browsergrad.host-graph.webgpu@1`
DAG/repeat/dynamic-dispatch/bounded-input/runtime-control/resource-feedback adapter and
its required actual-device evidence. The version-1.7 backend profile uses one
explicit bounded host readback/resubmission point for its GPU-produced
predicate; versions 1.10 and 1.11 reuse that authority for one bounded
GPU-produced loop count or dynamic-launch prefix, and version 1.13 reuses it
for one rank-2/rank-3 produced rectangle. Version 1.15 extends that same
single feedback lifecycle to one rank-4 produced rectangle, and version 1.17
extends it to one rank-5 produced rectangle. Neither adapter grants
transport, topology, retries, event
timestamps or external waits, repeated/device-side feedback,
rank-6-and-higher dynamic domains,
worker-mesh, native-companion,
performance, or release authority.

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
