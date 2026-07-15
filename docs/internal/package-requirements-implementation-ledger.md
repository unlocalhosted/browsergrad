# Package Requirements Implementation Ledger

- **Normative source:**
  [`docs/platform/package-requirements-lld.md`](../platform/package-requirements-lld.md)
- **Ledger status:** active
- **Last updated:** 2026-07-15
- **Current implementation slice:** Gate 2 verified view-copy contract and CPU reference

This is the durable implementation and recovery record for the semantic-systems
architecture. Update it after every implementation slice, material decision,
verification run, failure, or scope change. It records what happened; it does
not weaken or replace the normative requirements in the LLD.

## Status Vocabulary

| Status | Meaning |
|---|---|
| `not-started` | No implementation work or current evidence. |
| `audit` | Existing behavior and ownership are being inventoried. |
| `in-progress` | The gate has an active implementation slice. |
| `partial` | Useful capability landed, but the gate exit criteria are not met. |
| `blocked` | A named dependency or unresolved decision prevents progress. |
| `verified` | Gate exit criteria have current, recorded evidence. |
| `superseded` | Replaced by a recorded decision and migration path. |

`verified` is the only status that means a gate is complete. Passing one narrow
test does not make a gate verified unless every exit criterion is covered.

## Gate Ledger

| Gate | Status | Current result | Missing before `verified` | Evidence |
|---|---|---|---|---|
| Gate 0 — freeze and inventory | `verified` | Workspace direction and all six legacy adapters are machine-frozen. Stable diagnostic/capability/backend/requirement vocabularies are separated. Compiler pointer/scalar, runtime requirements, Grad behavior, and the exact JIT 36-constructor-call/39-operation matrix have pinned inventories and executable contracts. | None. Any baseline change requires the accepted-ADR exception path. | Architecture check; 945 compiler tests; 125 runtime tests; blocking Grad contract; 5-case JIT Gate 0 contract and full JIT integration. |
| Gate 1 — value/layout core and wire foundation | `verified` | Semantic-core `0.1.0` implements the bounded wire/value/layout contract, closed `browsergrad.layout@1` verification, authority-bound opaque artifacts, content-scoped IDs, deterministic normalization, and coordinate/address/alias traces. An independent Python reference matches TypeScript normalization, full-envelope canonicalization, semantic hashes, and traces for pinned static and symbolic fixtures. | None. The separate packed/release-tested `0.x` transition required by D-004 is also complete locally; registry publication remains a release operation, not a Gate 1 criterion. | Semantic-core typecheck/build/lint; 8 files and 68 tests; two pinned cross-language fixtures; 14 verifier rejection mutations; dynamic trace rejection and dominating-predicate parity; architecture check; packed-tarball gate. |
| Gate 2 — multi-frontend, multi-backend view slice | `in-progress` | The ownership audit is complete. The selected seam is one verified L2 materializing view-copy operation over a verified layout artifact, with explicit read/write effects, exact padding-fill policy, fail-closed overlap, and a layout side table beside the frozen legacy plan. Rank-2 transpose is the first compiler/JIT tracer bullet. | Implement and verify the shared operation and CPU evaluator; add kernels-owned WGSL lowering with strict device evidence; lower compiler read-only flat-logical-index bindings and typed JIT permutation; extend unchanged operation shape to slice, broadcast, padding, dynamic rank-2/3 fixtures; derive/refuse the legacy plan without widening it. | Three read-only ownership audits; semantic-core package gate; existing JIT typed permutation/VJP/plan and kernels real-device tests traced. No implementation or WebGPU conformance claim yet. |
| Gate 3 — real C++/CuTe frontend slice | `not-started` | No implementation in this workstream. | All Gate 3 exit criteria. | None. |
| Gate 4 — tiled GEMM and schedule separation | `not-started` | No implementation in this workstream. | All Gate 4 exit criteria. | None. |
| Gate 5 — tiled attention flagship | `not-started` | No implementation in this workstream. | All Gate 5 exit criteria. | None. |
| Gate 6 — framework convergence | `not-started` | No implementation in this workstream. | All Gate 6 exit criteria. | None. |
| Gate 7 — host graphs and optional systems expansion | `not-started` | No implementation in this workstream. | All Gate 7 exit criteria. | None. |

## Active Slice

### Objective

Gate 0 and Gate 1 are verified. Gate 2 begins with the verified materializing
view-copy contract and CPU evaluator, then one rank-2 transpose tracer bullet
from compiler and JIT through the same actual WebGPU lowering. Broader view
families extend the operation rather than add source-shaped execution paths.

### Work in flight

| Work item | Status | Notes |
|---|---|---|
| Frozen-adapter inventory | `verified` | Baselines are machine-enforced for compiler pointer/scalar memory, `cute_static_layout`, the 44-op shape/f32 `TensorGpuPlan`, 36 JIT constructors/39 real opaque labels and their decisions, runtime requirement mapping, and 17 Grad dtype/view/materialization behaviors. |
| Semantic-core seam audit | `partial` | Existing compiler/JIT/kernels types must adapt into the core; none can be moved wholesale. Exact initial package split is selected. |
| Test-topology analysis | `verified` | TypeScript + Vitest; no specialized catalog match, so native focused Vitest route is recorded locally. |
| Gate 0 architecture check | `verified` | Cross-package boundaries, generated-source parity, all six required freezes, exact runtime mapping/status unions, reviewed vocabulary, profile-usage parity, pinned inventories/harnesses, normalized definition fingerprints, and representative mutations are implemented and wired into delivery gates. |
| Gate 1 schema/value core | `verified` | `/schema` and `/layout` only; all Gate 1 requirements and the explicit cross-language exit are covered. The Python code is a synchronized reference oracle, not a second runtime or stable public API. |
| Semantic-core package adoption gate | `verified` | `0.1.0` is public-package shaped, dependency-free, subpath-only, and locally packed with schema/layout imports, declarations, Python oracle, fixtures, license, and changelog verified. It has not been published. |
| Gate 2 view-family selection | `verified` | Selected typed JIT `PERMUTE` plus a compiler read-only flat-logical-index storage binding, starting with rank-2 transpose. Both must emit the same artifact hash; Gate 2 remains incomplete until the full required view matrix and strict WebGPU proof pass. |
| L2 materializing view-copy contract | `in-progress` | Same-dtype f32 rank-2/3, source-read/destination-write effects, exact reject/fill invalid-source policy, and overlap forbidden initially. CPU and WGSL must consume the same verified maps or proved specializations. |

### Audit findings recorded so far

- Public compiler and kernels packages ship unbundled ESM and their release
  test rejects leaked `workspace:` dependencies. An unpublished private
  semantic-core package would therefore make installed public consumers
  unresolved once they import it at runtime.
- The LLD now permits `private` only during standalone Gate 1 incubation and
  requires a packed/release-tested `0.x` package before public-package runtime
  adoption. Semantic-core `0.1.0` now satisfies that local package gate. The
  alternative bundling path is not the current repository strategy, and npm
  publication has not occurred.
- Existing architecture checks cover compiler-local cycles, removed AST
  backends, representation purity, and line budgets, but not cross-package
  direction, frozen adapters, stable capability IDs, or core `CUSTOM` growth.
- `cute_static_layout` is rank-one parser sugar; `TensorGpuPlan` is a 44-op
  shape-only/f32 backend compatibility plan; JIT has 36 executable
  `OP_CUSTOM` construction sites producing 39 real labels. The original
  41-label scan falsely included `gt` and `sum` from non-CUSTOM dictionaries.
- Compiler pointer/scalar behavior is spread across source-shaped symbol,
  alias, memory-reference, expression, and operation records. Its canonical IR
  does not retain every source local or emit `pointer-rebind` for every source
  pointer update; behavior fixtures must assert emitted store/copy/memory facts
  and outputs rather than imagined source-shaped nodes.
- Runtime assignment labels mix semantic features, runtime facilities, device
  features, oracles, simulators, external services, and policy. They are legacy
  routing requirements, not all `SemanticCapabilityDefinition` values or
  lowering decisions. Current repository-owned usage is 53 labels: 51 in
  profiles plus `shader-f16` and `subgroups` emitted by the browser mapping.
- Grad's current `bf16`/`bfloat16` path is f32 substitution. View/materialize
  behavior depends on both NumPy layout and dtype: reshape/transpose/permute
  may alias f32 but convert non-f32 inputs to new f32 storage; indexing also
  forces f32; expand and detach copy and force non-f32 to f32;
  `contiguous()` returns `self` even for a non-contiguous transpose. The final
  `to()` detaches cross-dtype conversions and treats an unrecognized string as
  a device-like no-op. These are compatibility debt, not bf16/view support.
- Compiler semantic types remain CUDA/source-shaped, kernels plan types mix
  scheduling and execution, and JIT plan types own framework scheduling. They
  are adapter inputs, not semantic-core source material.
- JIT's GPU planner currently admits `SLICE` and `PAD` as primitive plan ops,
  while kernels' frozen 44-op `TensorGpuPlan` parser/executor accepts neither.
  NumPy handlers and the WGSL `PERMUTE`/`BROADCAST_TO` kernels also reconstruct
  offsets independently. Gate 2 must not widen the frozen plan to paper over
  this mismatch; it needs one verified view-materialization contract and a
  shared index-expression lowering/evaluation path.
- The typed JIT `PERMUTE` family is the strongest first frontend slice: public
  construction, NumPy reference, inverse-permutation VJP, vmap, ONNX, tensor
  plan, and real-WebGPU endpoints already exist. They are disconnected today,
  and NumPy, mock-plan execution, and WGSL reconstruct permutation semantics
  independently. Negative axes are accepted by NumPy but rejected by kernels;
  the JIT boundary must normalize them exactly once and reject duplicates or
  out-of-range axes before UOp construction.
- The compiler's useful seam is after ordinary CUDA-lite semantic lowering and
  before semantic-IR verification. A prepared, read-only global-storage
  binding can rewrite a direct flat logical read into one canonical physical
  index expression consumed unchanged by CPU and WGSL. Mutable pointer
  rebasing, writes through the bound view, helpers, atomics, reinterpret casts,
  dynamic dimensions, and possibly-false predicates remain fail-closed in the
  first adapter slice; frozen pointer/scalar schemas gain no fields.
- Current compiler out-of-range CPU reads return zero and writes are ignored;
  current WGSL conditionals lower to eager `select`. Neither behavior can
  implement padded views. Padding requires the L2 fill policy plus a structured
  guarded load that never evaluates the invalid memory arm.
- Existing browser tests may return early without an adapter, while kernels CI
  is non-blocking. Gate 2 evidence needs a required-device mode that treats
  adapter absence and skipped execution as environment failure, not success.
- The frozen tensor plan cannot express offsets, aliases, predicates, dynamic
  bindings, general rank, non-f32 dtype, or zero-extent truth. It remains a
  compatibility schedule/liveness projection only. Verified layout/kernel
  artifacts live beside it; no semantics may be recovered from plan shapes or
  axes.
- The LLD's initial view model conflated geometry and effects and used constant
  offsets despite dynamic shape support. It now uses symbolic byte lengths and
  offsets, puts access mode in L2 effects, and makes normalized `IndexMap` the
  sole executable coordinate truth.
- A second design audit found L1 allocation geometry mixed with L4 ownership
  and binding, content-derived IDs that would collapse distinct identical
  entities, an impossible freeze-then-brand order, and unbounded BigInt growth.
  The LLD now separates allocation spec/binding, scopes entity IDs, brands
  wrappers before freezing, and budgets integer width plus arithmetic work.

### Current constraints

- Preserve unrelated user changes in the dirty worktree.
- Do not claim a capability tier above its evidence.
- Do not add empty public abstractions merely to mirror the target package map.
- Keep the semantic core browser-safe and independent of frontend, runtime,
  Python, WebGPU, and native backend internals.
- New tests must follow the repository's discovered framework and focused gate.

## Decision Log

| ID | Date | Status | Decision | Reason / consequence |
|---|---|---|---|---|
| D-001 | 2026-07-15 | accepted | Keep this implementation ledger separate from the normative LLD and link them both ways. | Mutable state can change without silently changing architectural requirements. |
| D-002 | 2026-07-15 | accepted | Gate completion requires all recorded exit criteria and evidence; partial implementation remains `partial`. | Prevents test counts or one successful demo from overstating completion. |
| D-003 | 2026-07-15 | accepted | Begin with Gate 0 enforcement and concrete Gate 1 `/schema` and `/layout` capabilities; do not pre-create empty kernel, schedule, host, or capability APIs. | Audits confirmed existing models cannot move wholesale and speculative entrypoints would freeze the wrong seams. |
| D-004 | 2026-07-15 | accepted | Treat semantic-core as private only during standalone incubation; publish and release-test it as `0.x` before any public unbundled package ships a runtime dependency on it. | Prevents workspace-only success followed by broken npm installations while preserving an explicitly unstable internal API. |
| D-005 | 2026-07-15 | accepted | Normalize `LayoutExpr` into one executable `IndexMap`; keep dimension symbols, logical coordinates, and predicates as separate expression domains. | Prevents two serialized coordinate truths and accidental symbol-kind ambiguity. |
| D-006 | 2026-07-15 | accepted | Semantic hashes use SHA-256 over canonical UTF-8 JSON, exclude provenance/transport metadata, and compose named components through a domain-separated canonical record. | Removes circular IDs, undocumented cache salt, and cross-language float/string ambiguity. |
| D-007 | 2026-07-15 | accepted | Keep allocation geometry and alias identity in L1 `AllocationSpec`; keep ownership and runtime binding in L4 `AllocationBinding`. | Prevents value/layout artifacts from owning host lifecycle state or duplicating binding truth. |
| D-008 | 2026-07-15 | accepted | Content-address only pure value nodes; derive identity-bearing entity IDs from deterministic artifact scope and canonical entity position. | Distinct identical allocations and views retain separate alias/lifecycle identity. |
| D-009 | 2026-07-15 | accepted | Enforce integer-bit and arithmetic-operation budgets; deep-freeze normalized artifacts before placing them in an immutable opaque verified wrapper. | Closes BigInt denial-of-service and impossible post-freeze mutation paths. |
| D-010 | 2026-07-15 | accepted | V1 uses a closed byte-addressable scalar dtype registry and an affine/strided index algebra; swizzles and sub-byte storage require named extensions. | Prevents strings and backend spellings from becoming dtype/layout semantics before their bit rules exist. |
| D-011 | 2026-07-15 | accepted | Minor additions are restricted to open bags with lossless unknown-field preservation; closed semantic records change only by major version or required extension. | Keeps old-reader canonical hashes stable instead of silently dropping new payload meaning. |
| D-012 | 2026-07-15 | accepted | Keep the dependency-free Python implementation as an independent conformance oracle for the closed layout wire schema, synchronized atomically with TypeScript, fixtures, pinned goldens, and this ledger. | Provides cross-language drift detection without creating a second runtime or implying a stable Python package API. |
| D-013 | 2026-07-15 | accepted | Separate semantic capabilities/lowering/evidence from assignment requirements/resolutions and universal diagnostics; register legacy assignment strings as routing requirements rather than promoting them to semantic capabilities. | Prevents device facts, simulators, oracles, policies, and external services from receiving fictitious preservation levels or lowering decisions. Runtime may eventually consume only the narrow diagnostic, capability, and requirement protocol subpaths. |
| D-014 | 2026-07-15 | accepted | Model Grad compatibility as a frozen, versioned observation inventory with inventory-scoped execution context and per-behavior dtype, alias, contiguity, materialization, autograd, condition, failure, evidence, reference-contract, and target-conformance facts. | Prevents verified debt from being mislabeled conformant, prevents f32-only observations from becoming universal view claims, and makes Pyodide/NumPy upgrades explicit baseline reviews. |
| D-015 | 2026-07-15 | accepted | Model JIT opaque compatibility as 36 exact constructor-call records and 39 operation records using five closed decision policies; preserve `name` versus `op`, conditional reachability/effects, declared versus realized dtype behavior, replay, autograd/transform/export/residency, default versus inspection-only versus executable tensor-plan decisions, and constructor-only status. | Counts and one-way allowlists hid two ghost labels, grouped distinct calls, allowed same-count relabels, treated declared dtype as realized dtype, overstated tensor-plan/WebNN/transformer execution, and missed silent gradient disconnection plus conditional stateful callback replay. |
| D-016 | 2026-07-15 | accepted | Promote semantic-core `0.1.0` from private incubation to a dependency-free public-package shape before adding compiler, kernels, or JIT runtime imports; keep only explicit `/schema` and `/layout` exports and prove the packed tarball locally. | BrowserGrad packages ship unbundled ESM. A workspace-only private dependency would pass locally and break external installs. `0.x`, narrow exports, publish ordering, and tarball gates communicate instability without creating an unavailable dependency. This decision does not claim npm publication. |
| D-017 | 2026-07-15 | accepted | Gate 2 introduces one verified L2 materializing view-copy operation over verified L1 layouts. It owns explicit effects, exact reject/fill behavior for invalid source coordinates, and initially forbids overlap. Verified artifacts live in a side table; the frozen tensor plan is derived or refused and gains no semantic fields. The first tracer is rank-2 f32 transpose from typed JIT `PERMUTE` and a read-only compiler flat-logical-index binding through shared CPU and strict real-WebGPU execution. | All three ownership audits found independent offset reconstruction and disconnected proof paths. This seam makes padding and materialization explicit, prevents another source-shaped backend, preserves frozen schemas, and leaves signed/negative-stride semantics rejected until backend integer equivalence is proved. |

Provisional decisions MUST be accepted, replaced, or rejected before the
affected implementation slice is marked complete.

## Change Log

### 2026-07-15 — Ledger created

- Added this durable gate/status/decision/evidence record at the user's request.
- Added `.wibey-test/` to `.gitignore` for local test-orchestration state.
- Started three read-only parallel audits: Gate 0, semantic-core seams, and test
  topology.
- Corrected the LLD's private-package direction after the seam audit found it
  incompatible with the repository's public unbundled package model.
- No architecture gate or semantic-core implementation is claimed complete.

### 2026-07-15 — Architecture freeze enforcement

- Added a versioned semantic-freeze manifest and a workspace architecture
  checker covering dependency direction, deep/cross-package imports,
  generated-source parity, required baseline decisions, and the three initial
  legacy adapter surfaces.
- Wired the check into root tests, compiler verification, release verification,
  and CI/publish workflows.
- Added mutation tests for union and interface widening, exports, dynamic and
  deep imports, inheritance, readonly removal, comments, custom-op aliases and
  positional labels, and missing required freezes.
- Kept Gate 0 `partial`: the current fingerprints freeze high-risk structure,
  not every semantic behavior or public capability claim.

### 2026-07-15 — Gate 1 wire/value foundation

- Added the private `@unlocalhosted/browsergrad-semantic-core` 0.1 package with
  only `/schema`, `/layout`, and `/package.json` exports.
- Implemented bounded duplicate-aware JSON parsing and programmatic-value
  validation, canonical JSON and domain-separated SHA-256 hashing, exact wire
  integers and float bit patterns, envelope normalization, exact budgeted
  dimension arithmetic and constraints, the closed builtin dtype registry,
  complete numerical policy records, and initial allocation/view/layout types.
- Added 43 focused tests. Kept Gate 1 `partial`: layout normalization,
  verification, traces, and cross-language fixtures remain.
- Closed review findings before commit: untrusted bytes are limited before
  fatal UTF-8 decoding; semantic hashes require an unforgeable internal
  verified-wrapper instance; constraint sets share one node/operation budget;
  resolved invalid divisors cannot hide behind unresolved symbols; bindings
  require declared domains; array accessors are never executed during JSON
  validation.
- Recorded architecture direction in commit `e16791d0` (`docs: define semantic
  systems architecture`).
- Recorded architecture enforcement in commit `6985611d` and the truthful
  row-wise attention naming correction in commit `aa632f98`.
- Recorded the bounded wire/value foundation in commit `d292595c`.

### 2026-07-15 — Gate 1 layout normalization

- Defined composition through an explicit source-coordinate map, removing the
  ambiguous scalar-layout composition interpretation.
- Normalized strided, permutation, slice, broadcast, padding, and composition
  builders into one element-location expression plus one in-bounds predicate.
- Added exact shared-budget index and predicate evaluation; logical out-of-
  bounds coordinates retain their computed address and return a false predicate
  instead of being clamped.
- Added stable layout diagnostic constants and seven focused normalization
  scenarios. Gate 1 remains `partial` until the normalized layout artifact
  verifier, address/alias traces, and cross-language proof land.
- Recorded this slice in commit `5d28bcef`.

### 2026-07-15 — Gate 1 verified layout artifacts and traces

- Added the closed `browsergrad.layout@1` payload verifier for symbols,
  constraints, allocations, index maps, and views; unknown fields, duplicate
  IDs, dangling references, rank drift, dtype drift, range errors, and
  alignment errors fail with stable diagnostics.
- Removed raw local IDs from semantic identity. The verifier hashes an ID-free
  provisional projection, then derives allocation, alias-set, index-map, and
  view IDs from the full scope digest plus canonical ordinal.
- Added byte and programmatic verification entrypoints. Only the schema-
  specific verifier can construct the opaque layout wrapper consumed by hashes
  and trace evaluators.
- Added coordinate/address traces that separately report logical, predicate,
  and allocation bounds and retain negative/OOB computed addresses. Added alias
  traces that distinguish same allocation, same alias set, and resolved
  same-root byte overlap without inferring L2 write safety.
- Added nine verification/trace scenarios; Gate 1 remains `partial` pending
  TypeScript/Python parity and property-generated proof.

### 2026-07-15 — Gate 1 cross-language closure

- Bound opaque layout wrappers to a private schema authority so a generic or
  foreign verifier cannot forge an executable layout artifact.
- Tightened static and runtime range/alignment checks, made alias overlap
  unknown for invalid accesses, and deep-froze all nested trace evidence.
- Added a dependency-free Python reference for the complete current closed
  `browsergrad.layout@1` schema. It independently decodes, validates,
  normalizes, canonicalizes, hashes, and traces artifacts under the same
  structural, resource, divisor, range, alignment, and predicate rules.
- Added pinned static rank-2/element and symbolic constrained
  rank-3/byte-addressed golden fixtures. Full canonical artifacts, semantic
  hashes, normalized payloads, trace outcomes, UTF-16 astral-key ordering, and
  Python-to-TypeScript normalized re-encoding are checked.
- Added 14 verifier-only rejection mutations plus dynamic trace rejection and
  dominating-predicate acceptance parity. The verifier corpus uses empty trace
  cases so a later trace failure cannot mask an incorrectly accepted artifact.
- Added 96 deterministic generated rank-1 through rank-4 samples covering
  contiguous, signed/non-contiguous, permutation, positive/negative slice,
  explicit composition, singleton/leading broadcast, padding, boundary, and
  OOB invariants. All 36 coordinate pairs in the canonical rank-2 fixture also
  prove valid-access byte-overlap alias invariants.
- Bounded divisor lower-bound proof before BigInt multiplication in both
  languages and added a differential integer-growth bomb. This closes the last
  resource path found by the principal-level exit review.
- Gate 1 is `verified`: its serialized fixtures now have one verified meaning
  and pinned deterministic hashes across TypeScript and Python.

### 2026-07-15 — Compiler pointer/scalar memory freeze

- Promoted `compiler.pointer-scalar-memory.v0` from documentary metadata to a
  required architecture freeze accepted by ADR-0001.
- Froze the exact 12-value address-space union; complete symbol, pointer alias,
  pointer selection, and memory-reference interfaces; the pointer-bearing
  index expression; nine memory operation shapes; and six root public exports.
- Added five versioned behavior cases for storage rebasing, byte-root vector
  reinterpretation, shared multidimensional byte addresses, cross-root
  selection, and pointer-array rebinding. Each case records only normalized IR
  facts, semantic CPU-reference output, and WGSL eligibility; source offsets,
  generated IDs, and WGSL text are intentionally excluded.
- Added mutation coverage for address-space widening, interface-field addition,
  readonly loss, operation widening, public-export removal, and missing fixture
  IDs. At this checkpoint Gate 0 remained `partial` pending Grad,
  runtime/vocabulary, diagnostic ID, and exact opaque-operation inventories.

### 2026-07-15 — Platform vocabulary and runtime requirement freeze

- Corrected the normative model: semantic capabilities, program-specific
  lowering decisions, execution evidence, assignment requirements, requirement
  resolutions, and diagnostics are separate records with separate IDs.
- Added a reviewed vocabulary registry with the closed nine diagnostic stages,
  the implemented layout/index-map capability ID, three concrete backend
  IDs, eight requirement kinds, and all 53 repository-owned legacy assignment
  requirement IDs. Static capabilities reject runtime/evidence outcome fields.
- Added a generated usage inventory for the 51 requirement IDs referenced by
  the ten checked-in profiles. Architecture checking regenerates the same model
  in memory and rejects a stale file or unregistered profile/mapping ID.
- Promoted `runtime.generic-backend-labels.v0` to a required freeze covering the
  exact eight browser-input fields/mappings, WebGPU-parent conditions, route
  modes, readiness states, runner targets, and three behavior fixtures. The
  compatibility API still accepts externally supplied strings; that does not
  register them for repository-owned profiles.
- Corrected platform guidance that inferred compiler/subset support solely from
  `features.webgpu`. Device facts now produce only device requirements, a
  loaded compiler is an environment facility, and source-subset/executability
  results remain attached to the concrete artifact.
- Exit review closed three P1s: source-subset/executability resolutions are now
  explicitly artifact-scoped, fixtures have their own requirement kind, and
  every vocabulary record is closed with validated capability links. It also
  closed the fixture-drift gap with a pinned SHA-256 and all 256 boolean browser
  mapping combinations; legacy meanings now define exact set-membership truth.

### 2026-07-15 — Grad dtype/view/materialization freeze

- Added a closed 17-record Grad compatibility inventory and 30-case Pyodide
  fixture. Observation status is separate from target conformance; the
  inventory pins Pyodide/NumPy versions, storage byte widths, bf16-to-f32
  substitution, execution context, exact conditions, failure policy, aliasing,
  contiguity, materialization, autograd, source definitions, and evidence.
- Proved f32, f16, int64, bool, and NumPy-delegated dtype paths. Non-f32
  counterexamples prevent reshape, transpose, permute, indexing, expand, or
  detach from being described as dtype-preserving views. Rounding-sensitive
  bf16 evidence proves f32 value semantics rather than only a dtype label.
- Added bidirectional mutation, parent-identity, and backward-gradient proof
  for reshape/view, transpose, permute, expand, basic slice, and duplicate
  fancy indexing. `_build_ctx` is now part of the source freeze.
- Pinned the inventory, fixture, and executable harness by SHA-256 and made the
  focused Grad contract a blocking CI step even while the broader stochastic
  Grad integration suite remains a non-blocking stabilization lane.
- Normalized Python definition fingerprints now include one-line decorators,
  signatures, executable statements, and behavior strings while excluding
  comments and leading docstrings. Mutation tests cover executable changes,
  decorators, graph-builder drift, dtype-token drift, schema widening,
  fixture removal, and removal of the required Grad freeze.
- Removed the dead shadowed `Tensor.to` definition and corrected the false
  `contiguous()` docstring, then regenerated the embedded Python source.

### 2026-07-15 — Exact JIT opaque-operation freeze and Gate 0 closure

- Replaced the count/allowlist-only JIT baseline with a closed inventory of 36
  constructor-call records, 39 executable labels, five decision policies, and exact
  normalized fingerprints for constructors plus shared realization, backward,
  transform, export, and plan decisions.
- Removed `gt` and `sum`: both were token-scanner false positives from ordinary
  CMP/REDUCE argument dictionaries, not `CUSTOM` labels.
- Distinguished CPU `name` callbacks from backend-dispatched `op` nodes and
  froze the four reviewed callers of the dynamic elementwise-loss helper.
- Recorded six silent autograd disconnections, conditional interpolate
  backward, legacy WebGPU-only flash/user routes, constructor-only WebNN and
  transformer nodes, dropout RNG replay, BatchNorm state replay, and einsum's
  construction-time host shape evaluation.
- Split sort/top-k value and index constructors into four exact call records;
  separated default tensor-plan refusal, inspection-only `allow_custom`
  admission, and executable-plan refusal; and separated declared dtype from
  version-pinned realized NumPy dtype observations.
- Added a five-case Pyodide contract that directly executes all 39 labels,
  proves closure gradients for all 29 callback-with-closure labels, covers both
  sides of Dropout/BatchNorm reachability and effects, and pins shared
  refusals, disconnected gradients, legacy WebGPU routing, and
  constructor-only failures. Corrected source comments that falsely promised
  NoBackwardError, WebNN fallback, or transformer fallback.
- Gate 0 is `verified`; Gate 1 was already `verified`. Gate 2 is the next
  implementation gate.

### 2026-07-15 — Semantic-core `0.x` package adoption gate

- Removed the private-workspace marker from semantic-core `0.1.0` without
  widening its two explicit exports or adding a root barrel or dependency.
- Added package-local license and changelog, public publish metadata, and the
  same build/typecheck/lint/test prepublish gate used by shipping packages.
- Extended the release-package harness to pack and extract semantic-core, then
  verify tarball metadata, zero dependencies, declarations, runtime imports,
  Python parity oracle, both fixture families, README, license, and changelog.
- This is local package readiness only. No registry lookup, publish, push, or
  deployment occurred.

### 2026-07-15 — Gate 2 ownership audit and contract selection

- Completed three read-only traces across compiler, JIT, semantic-core,
  kernels, CPU reference, plan bridging, and current browser evidence.
- Selected one verified L2 materializing view-copy operation rather than
  widening `TensorGpuPlan` or teaching separate source-shaped backends about
  offsets. It owns explicit effects, exact reject/fill behavior, and an initial
  no-overlap rule.
- Selected typed JIT `PERMUTE` plus a compiler read-only flat-logical-index
  storage binding for the first rank-2 transpose tracer. Both must converge on
  one semantic hash before backend execution.
- Recorded the padding hazard: compiler CPU zero-fill, ignored writes, eager
  WGSL `select`, and target robust-buffer behavior are not valid guarded-load
  semantics.
- Recorded strict real-device evidence requirements and kept Gate 2
  `in-progress`; no implementation, GPU conformance, registry publish, push,
  or deployment is claimed by this audit.

## Verification Log

| Date | Scope | Command | Result | Follow-up |
|---|---|---|---|---|
| 2026-07-15 | Architecture guard | `pnpm architecture:check` | Passed. | Re-run before each architecture-affecting commit. |
| 2026-07-15 | Compiler guard integration | `pnpm --filter @unlocalhosted/browsergrad-compiler test` | Passed: 27 files, 931 tests. | Re-run after final guard diff. |
| 2026-07-15 | Compiler types | `pnpm --filter @unlocalhosted/browsergrad-compiler typecheck` | Passed. | None. |
| 2026-07-15 | Kernels attention truth correction | `pnpm --filter @unlocalhosted/browsergrad-kernels typecheck` and `pnpm --filter @unlocalhosted/browsergrad-kernels test` | Passed: typecheck and 10 files, 71 tests. | Re-run before commit. |
| 2026-07-15 | Semantic core | `pnpm --filter @unlocalhosted/browsergrad-semantic-core typecheck` | Passed. | None. |
| 2026-07-15 | Semantic core foundation | `pnpm --filter @unlocalhosted/browsergrad-semantic-core test` | Passed: 4 files, 43 tests. | Superseded by the current 50-test run. |
| 2026-07-15 | Semantic core | `pnpm --filter @unlocalhosted/browsergrad-semantic-core build` | Passed. | None. |
| 2026-07-15 | Semantic core | `pnpm --filter @unlocalhosted/browsergrad-semantic-core lint` | Passed with no warnings after replacing the control-character regex with explicit code-point comparisons. | None. |
| 2026-07-15 | Workspace integration | `pnpm -r run build` | Passed for 7 of 8 workspace projects, including semantic core. | None. |
| 2026-07-15 | Workspace integration | `pnpm -r run typecheck` | Semantic core and all packages before runtime passed; runtime failed on three pre-existing optional-value checks in `assignment-javascript-profile-e2e.test.ts`. | Keep as unrelated workspace evidence; semantic-core focused typecheck is green. |
| 2026-07-15 | Layout normalization | `pnpm --filter @unlocalhosted/browsergrad-semantic-core typecheck && pnpm --filter @unlocalhosted/browsergrad-semantic-core test && pnpm --filter @unlocalhosted/browsergrad-semantic-core build && pnpm --filter @unlocalhosted/browsergrad-semantic-core lint` | Passed: 5 files, 50 tests; build/typecheck/lint clean. | None. |
| 2026-07-15 | Verified layout artifacts and traces | `pnpm --filter @unlocalhosted/browsergrad-semantic-core typecheck && pnpm --filter @unlocalhosted/browsergrad-semantic-core test && pnpm --filter @unlocalhosted/browsergrad-semantic-core build && pnpm --filter @unlocalhosted/browsergrad-semantic-core lint` | Passed: 6 files, 59 tests; build/typecheck/lint clean. | Add cross-language parity. |
| 2026-07-15 | Gate 1 cross-language exit | `pnpm architecture:check && pnpm --filter @unlocalhosted/browsergrad-semantic-core typecheck && pnpm --filter @unlocalhosted/browsergrad-semantic-core test && pnpm --filter @unlocalhosted/browsergrad-semantic-core build && pnpm --filter @unlocalhosted/browsergrad-semantic-core lint` | Passed: architecture check; 8 files, 68 tests; typecheck/build/lint clean. Python source also parsed through `ast.parse`. | Gate 1 verified; resume Gate 0 inventory before Gate 2. |
| 2026-07-15 | Compiler pointer/scalar freeze | `node --check scripts/semantic-architecture-check.mjs && pnpm architecture:check && pnpm --filter @unlocalhosted/browsergrad-compiler typecheck && pnpm --filter @unlocalhosted/browsergrad-compiler test` | Passed: script syntax, architecture check, compiler typecheck, 28 files and 938 tests. | None. |
| 2026-07-15 | Vocabulary and runtime requirement freeze | `pnpm architecture:generate-requirements && node --check scripts/semantic-architecture-check.mjs && node --check scripts/generate-assignment-requirement-usage.mjs && pnpm architecture:check && pnpm --filter @unlocalhosted/browsergrad-compiler typecheck && pnpm --filter @unlocalhosted/browsergrad-compiler test && pnpm --filter @unlocalhosted/browsergrad-runtime test` | Passed: deterministic 51-ID usage generation, script syntax, architecture check, compiler typecheck, 28 files/940 compiler tests, and 11 files/125 runtime tests. | Runtime repo-wide typecheck remains blocked by the recorded pre-existing optional-WGSL errors. |
| 2026-07-15 | Grad dtype/view/materialization freeze | `node --check scripts/semantic-architecture-check.mjs && pnpm architecture:check && pnpm --filter @unlocalhosted/browsergrad-compiler typecheck && pnpm --filter @unlocalhosted/browsergrad-compiler test && pnpm --filter @unlocalhosted/browsergrad-grad typecheck && pnpm --filter @unlocalhosted/browsergrad-grad test && pnpm --filter @unlocalhosted/browsergrad-grad exec vitest run --config vitest.integration.config.ts tests-integration/gate0_dtype_view_contract.test.ts && pnpm --filter @unlocalhosted/browsergrad-grad test:integration` | Passed: script syntax, architecture check, compiler typecheck and 28 files/942 tests, Grad typecheck and 2 files/30 unit tests, blocking contract 1 file/1 test, and full Grad integration 33 files/322 tests. | Exact JIT opaque-operation inventory remains for Gate 0. |
| 2026-07-15 | Exact JIT opaque-operation freeze | `node --check scripts/semantic-architecture-check.mjs && pnpm architecture:check && pnpm --filter @unlocalhosted/browsergrad-jit typecheck && pnpm --filter @unlocalhosted/browsergrad-jit test && pnpm --filter @unlocalhosted/browsergrad-jit exec vitest run --config vitest.integration.config.ts tests-integration/gate0_opaque_operation_contract.test.ts && pnpm --filter @unlocalhosted/browsergrad-jit test:integration && pnpm --filter @unlocalhosted/browsergrad-compiler typecheck && pnpm --filter @unlocalhosted/browsergrad-compiler test` | Passed: script syntax, architecture check, JIT typecheck and 8 unit tests, focused 5-case Pyodide contract, full JIT integration 23 files/228 tests, compiler typecheck and 28 files/945 tests. | None. |
| 2026-07-15 | Gate 0 cross-package closure | `pnpm architecture:check && pnpm --filter @unlocalhosted/browsergrad-runtime test && pnpm --filter @unlocalhosted/browsergrad-grad typecheck && pnpm --filter @unlocalhosted/browsergrad-grad exec vitest run --config vitest.integration.config.ts tests-integration/gate0_dtype_view_contract.test.ts` | Passed: architecture check, 11 files/125 runtime tests, Grad typecheck, and blocking Grad contract 1 file/1 test. | Gate 0 verified. |
| 2026-07-15 | Semantic-core package adoption gate | `pnpm --filter @unlocalhosted/browsergrad-semantic-core typecheck && pnpm --filter @unlocalhosted/browsergrad-semantic-core test && pnpm --filter @unlocalhosted/browsergrad-semantic-core build && pnpm --filter @unlocalhosted/browsergrad-semantic-core lint && pnpm test:release-packages && pnpm architecture:check` | Passed: semantic-core typecheck/build/lint, 8 files/68 tests, packed tarball metadata/content/runtime-import checks, existing kernels/compiler release-package checks, and architecture check. | Package is ready to become a workspace runtime dependency in a later coherent Gate 2 slice; npm publication is still pending the normal release workflow. |

## Failure and Recovery Log

Record failures that may matter after context loss. Include the exact failing
command, concise error, suspected cause, resolution or next experiment, and
whether any files may be left partially changed.

- `pnpm install` failed with `ERR_PNPM_ABORTED_REMOVE_MODULES_DIR_NO_TTY` because
  pnpm would not recreate `node_modules` without a TTY. No source files were
  partially changed.
- `CI=true pnpm install` then failed with `ERR_PNPM_OUTDATED_LOCKFILE` after the
  new workspace importer was added. This was the expected frozen-lockfile gate.
- `CI=true pnpm install --no-frozen-lockfile` succeeded and updated
  `pnpm-lock.yaml`; all eight workspace projects were installed.
- A semantic-core typecheck caught an `exactOptionalPropertyTypes` violation
  when forwarding an absent envelope limit as `{ limits: undefined }`. The
  call now omits the property when absent; typecheck, tests, build, and lint all
  pass afterward.
- `pnpm -r run typecheck` reached runtime after every other package passed, then
  failed at lines 675, 682, and 694 of
  `packages/browsergrad-runtime/tests/assignment-javascript-profile-e2e.test.ts`
  because `compiled.wgsl` / `compiled.wgslProgram` may be undefined. Semantic
  core is not consumed there; no runtime files were changed in this slice.
- The vocabulary slice re-ran runtime typecheck after dependency builds and hit
  the same three pre-existing optional-WGSL errors. The new fixture test and 53
  related runtime routing/profile tests pass; no runtime source file was
  changed.
- The first layout-normalizer typecheck found two `noUncheckedIndexedAccess`
  errors when indexing already rank-validated stride/step arrays. Explicit
  post-validation casts document the invariant; all focused gates pass.
- The first artifact-verifier typecheck found an unused wire type import and an
  intentionally opaque JSON-model cast that needed an explicit `unknown`
  boundary. The first trace typecheck also caught three absent optional limits
  forwarded as `undefined`; all were corrected before tests and build passed.
- Exit review found that generic verified wrappers shared one runtime registry
  with layout artifacts. Schema-specific authority identity is now required,
  and foreign-authority forgery has an adversarial test.
- The first Python parity oracle structurally validated the positive fixture
  but did not mirror TypeScript's semantic range, divisor, alignment, and
  dynamic trace checks. The reference now implements the complete current
  closed layout-v1 verification path and a differential rejection corpus.
- Exit review found lower-bound positivity analysis could construct integers
  beyond the configured bit budget before the normal evaluator ran. Both
  TypeScript and Python now estimate multiplication width, budget lower-bound
  operations, and fail proof before oversized multiplication; an adversarial
  repeated-product divisor is rejected by both.
- The first rejection harness used non-empty trace cases, which could hide a
  Python verifier acceptance behind a later trace failure. Verifier mutations
  now always run with an empty cases file. The first Unicode ordering probe was
  also a literal spelling rather than an astral key; it now compares actual
  U+1F600 against U+E000.
- Final cross-language review found a wire `dimension` could name the internal
  `__bg_coordinate_*` evaluator namespace and be accepted only by TypeScript.
  Wire dimension references now reject all reserved `__bg_` names, with a
  differential regression case.
- The first compiler pointer behavior fixture assumed source pointer rebases
  would always survive as `pointer-rebind` operations and local pointer names
  would remain module symbols. The focused test rejected four of five cases.
  Inspection showed the canonical IR truth: the tested rebases are reflected
  in store address expressions, packed reinterpretation lowers to `copy`, and
  some locals are consumed during lowering. The fixture now freezes emitted
  operation/memory facts plus outputs instead of source-shaped expectations.
- The first Grad Gate 0 fixture run failed with `ModuleNotFoundError: No module
  named 'torch'` because the test imported the compatibility namespace before
  calling `grad.install_torch_alias()`. The harness now installs the alias
  explicitly; the focused contract passes.
- The first full compiler gate after adding Grad mutation tests failed because
  `scripts/semantic-architecture-check.d.mts` did not declare the two new
  checker exports. The declarations now cover the source checker, inventory
  validator, and digest extractor; compiler typecheck and all 942 tests pass.
- The first full Grad integration run for this slice passed 321 of 322 tests
  but failed the stochastic classifier's improvement-margin assertion:
  `0.0234375` was not greater than `0.05`; its final-accuracy assertion had
  already passed. The exact test passed immediately in isolation. No Grad
  numerical behavior was changed in this slice; keep the full suite result
  distinct from the blocking deterministic Gate 0 contract. A subsequent full
  rerun passed all 33 files and 322 tests.
- The first exact-label mutation test exposed that the scanner treated keyword
  arguments named `arg={...}` as reusable local dictionaries, reintroducing
  non-CUSTOM `gt` and `sum` labels. Dictionary-label propagation now accepts
  actual assignments only; the exact `name`/`op` test and repository baseline
  both pass.
- The JIT freeze exit review rejected the first draft on five P1 gaps: 34
  grouped site records for 36 constructor calls, one collapsed tensor-plan
  decision, declared dtype presented as realized dtype, incomplete
  Dropout/BatchNorm branch reachability, and `gate0-contract` evidence claimed
  by operations the test did not execute. Resolution split the four sort/top-k
  calls, added three plan decisions, added version-pinned realized dtype
  observations and conditional-effect fields, and made the fifth fixture case
  directly execute all 29 closure callbacks. The other ten labels remain
  directly covered by the disconnected and accelerator/constructor cases.
- The first focused JIT command omitted `--config
  vitest.integration.config.ts`; Vitest searched only `tests/**/*.test.ts` and
  reported no test files. Re-running the same path with the integration config
  executed the intended contract. No source or fixture change was needed.
- The first exhaustive callback run used scalar `prod()`. Its NumPy callback
  returned `np.float32`, while `_h_custom` accepts only `np.ndarray`, so the
  test failed at the existing realization boundary. The inventory already
  records ndarray-only validation; the coverage case now uses a dimension
  reduction returning an array. Scalar callback behavior was not relabeled as
  supported.
- The first dtype fixture assumed native 64-bit NumPy integer accumulation.
  Pyodide 0.26.4 with NumPy 1.26.4/WASM realizes bool/int32 cumsum and int32
  prod as int32, while int32 sin/cos/var realize float64. The fixture and
  inventory now pin both dependency versions and record observed declared and
  realized dtypes instead of importing host-platform assumptions.

## Quick Resume Checklist

1. Read this ledger, then the relevant gate and exit criteria in the normative
   LLD.
2. Check `git status --short`; assume unrelated dirty files belong to the user.
3. Review the latest decision, verification, and failure entries before reading
   source broadly.
4. Resume only the current slice or explicitly record a scope change.
5. After work: update gate status, changed files, commands/results, failures,
   decisions, and the next smallest checkpoint.

## Next Checkpoint

Implement and commit the concrete semantic-core `/kernel` view-copy schema,
verifier, shared CPU materializer, exact padding-fill behavior, and negative
tests. Then add the kernels-owned WGSL lowering and strict device harness before
frontend adoption. The first compiler and JIT adapters must produce the same
rank-2 transpose artifact/hash and may only derive or refuse the frozen legacy
tensor plan; they cannot widen it or reconstruct offsets independently.
