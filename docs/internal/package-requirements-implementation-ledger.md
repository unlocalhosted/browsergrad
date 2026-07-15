# Package Requirements Implementation Ledger

- **Normative source:**
  [`docs/platform/package-requirements-lld.md`](../platform/package-requirements-lld.md)
- **Ledger status:** active
- **Last updated:** 2026-07-16
- **Current implementation slice:** Gate 2 compiler L2 structured padding and required-device evidence

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
| Gate 2 — multi-frontend, multi-backend view slice | `partial` | Semantic-core `0.2.0` owns verified view-copy meaning, shared specialization, and the sole canonical frontend construction sink. Kernels `0.2.0` passes the full nine-case CPU/actual-WebGPU bit-exact matrix on Apple Metal 3. The compiler's frozen L1 read binding still passes its six-case CPU/actual-WebGPU non-padded matrix. A separate L2 authority now binds the exact verified view-copy operation, lowers signed predicates and addresses into structured raw-u32 guard/read/fill/store IR, and passes CPU/WGSL rank-2/rank-3 padding, exact fill-bit, nonzero-offset/canary, always-false, reject-policy, authority, and arithmetic tests. JIT `0.9.0` emits a separate closed typed permutation request; kernels strictly constructs and prepares its artifacts; materializing/resident JIT bridge routes use canonical WGSL without reading semantic args from the frozen plan. | Run the compiler L2 matrix on required actual WebGPU, source-bind and retain the terminal record, then re-run strict lanes on the exact release commit. Registry publication remains separate. | Semantic-core 99 tests; compiler L1 six-case actual-WebGPU record plus L2 focused CPU/WGSL tests; kernels semantic preparation/residency and required nine-case lane; JIT production-emission/bridge tests plus exact-commit retained evidence at `09d868d0`; isolated and integrated packed JIT runtime/declaration consumers; deterministic dependency-first release guard; architecture guard. |
| Gate 3 — real C++/CuTe frontend slice | `not-started` | No implementation in this workstream. | All Gate 3 exit criteria. | None. |
| Gate 4 — tiled GEMM and schedule separation | `not-started` | No implementation in this workstream. | All Gate 4 exit criteria. | None. |
| Gate 5 — tiled attention flagship | `not-started` | No implementation in this workstream. | All Gate 5 exit criteria. | None. |
| Gate 6 — framework convergence | `not-started` | No implementation in this workstream. | All Gate 6 exit criteria. | None. |
| Gate 7 — host graphs and optional systems expansion | `not-started` | No implementation in this workstream. | All Gate 7 exit criteria. | None. |

## Active Slice

### Objective

Gate 0 and Gate 1 are verified. Gate 2 has one shared materializing view-copy
contract, CPU evaluator, canonical WGSL lowering, and compiler/JIT tracer
bullets through actual WebGPU. Compiler structured padding now exists through
that operation's existing guard/fill semantics; required actual-device and
retained-release evidence remain. Broader view families must extend the
operation rather than add source-shaped execution paths.

### Work in flight

| Work item | Status | Notes | Remaining | Evidence |
|---|---|---|---|---|
| Frozen-adapter inventory | `verified` | Baselines are machine-enforced for compiler pointer/scalar memory, `cute_static_layout`, the 44-op shape/f32 `TensorGpuPlan`, 36 JIT constructors/39 real opaque labels and their decisions, runtime requirement mapping, and 17 Grad dtype/view/materialization behaviors. |
| Semantic-core seam audit | `verified` | Existing compiler/JIT/kernels types adapt into the core through explicit public subpaths; none moved wholesale. The package split and dependency direction are architecture-guarded. |
| Test-topology analysis | `verified` | TypeScript + Vitest; no specialized catalog match, so native focused Vitest route is recorded locally. |
| Gate 0 architecture check | `verified` | Cross-package boundaries, generated-source parity, all six required freezes, exact runtime mapping/status unions, reviewed vocabulary, profile-usage parity, pinned inventories/harnesses, normalized definition fingerprints, and representative mutations are implemented and wired into delivery gates. |
| Gate 1 schema/value core | `verified` | `/schema` and `/layout` only; all Gate 1 requirements and the explicit cross-language exit are covered. The Python code is a synchronized reference oracle, not a second runtime or stable public API. |
| Semantic-core package adoption gate | `verified` | `0.2.0` is public-package shaped, dependency-free, subpath-only, locally packed, and now consumed through kernels' packed exact dependency. A fresh temporary consumer installs both tarballs, resolves bare public subpaths, typechecks, and prepares matching CPU/WGSL specializations. It has not been published. |
| Gate 2 view-family selection | `verified` | Selected typed JIT `PERMUTE` plus a compiler read-only flat-logical-index storage binding, starting with rank-2 transpose. Both must emit the same artifact hash; Gate 2 remains incomplete until the full required view matrix and strict WebGPU proof pass. |
| L2 materializing view-copy contract and CPU reference | `verified` | `view-copy@1.0` owns effects, exact reject/fill bits, and forbid-overlap semantics. Generic L2 verification stays backend-neutral; the shared initial profile legalizes positive-affine f32 rank-2/3 global views. Prepared CPU execution compiles maps once, proves guarded reads and dense destination writes, caches source offsets, derives binding-sensitive specialization hashes, and enforces element/step/scratch/wall-time budgets, cooperative browser yielding and abort, plus native buffer-slot, length, alignment, overlap, and shared-memory checks. |
| Kernels-owned WGSL view-copy lowering | `verified` | The lowerer consumes authority-bound immutable backend-neutral specializations, preserves whole-root f32 bits through u32 storage, interval-proves signed i32 arithmetic, lowers canonical source/destination maps, emits structured guarded fill loads, validates device and transient-working-set limits, and derives semantic plus device-specific hashes. One-in-flight ownership, timeout/abort stale-result suppression, exact scope drainage, distinct error stages, and device-loss invalidation have deterministic fake-device coverage. The required headed lane emitted one validated `passed` terminal record for all nine bit-exact CPU/WebGPU cases on Apple Metal 3; headless absence remains a truthful failed environment record. |
| Shared required-WebGPU evidence test contract | `verified` | A neutral, unpublished test-support module now owns adapter/device acquisition, required-versus-advisory outcome rules, generic terminal-envelope validation, and exactly-once emission. Kernels retains its suite-specific ordered case and observation validation and passes the same advisory no-adapter path through the shared contract. Compiler and later JIT lanes consume this helper rather than fork release-evidence semantics. |
| Compiler verified-layout binding preparation | `verified` | Compiler now depends on semantic-core through public `/layout` and `/schema` protocols and prepares explicit read-only, row-major-flat parameter bindings. Prepared objects are authority-bound and deeply immutable, retain the semantic layout hash plus a deterministic binding-projection hash, resolve dynamic dimensions once, reject non-global views and duplicate/malformed bindings, and provide a collision-resistant layout-bound compile-cache key without changing frozen semantic IR. Lowering into memory references is the next slice. |
| Compiler read-only layout lowering | `verified` | A separate layout-bound compile entrypoint rewrites direct guarded reads after ordinary runtime lowering and before semantic-IR verification. It unflattens one stable non-escaping `uint` logical index, substitutes the verified positive-affine map, and sends the same physical expression through CPU reference and WGSL legalization. Initial support is specialized nonempty rank-2/3 global `f32`; only index-map predicates proved true over the complete logical domain are erased, while conditional predicates, writes, aliases, pointer offsets/rebasing, signed/mutated/escaped indices, non-affine maps, unaligned byte maps, rank drift, and u32 overflow fail closed. The frozen compiled wrapper is instance-authorized, execution validates the complete verified root allocation, and full semantic/binding hashes enter pipeline identity. Six supported cases pass complete source/output bit comparison on actual Apple Metal 3; padding remains explicitly unsupported. |
| Compiler L2 structured view-copy lowering | `in-progress` | A sibling opaque preparation/compile boundary consumes the exact verified `view-copy@1.0` operation and shared specialization; L1 remains unchanged. The first source profile admits one guarded direct flat source-to-destination copy, lowers canonical rank-2/3 maps with exact BigInt i32/u32 interval proofs, emits a real inner branch, keeps the source address/load only in its true arm, stores exact fill words in its false arm, and binds whole roots as `u32`. Compiled/runtime authority is instance-bound; source/destination roots must be exact, distinct, native, and non-shared. Layout, kernel, specialization, and routing hashes enter cache and program identities. | Add required actual-WebGPU rank-2/rank-3 cases, observed dispatch evidence, source-bound retained-log verification, packed-consumer coverage, then run full release gates. Current L2 profile intentionally rejects byte-unit/non-affine maps, helpers, aliases, pointer rebasing, extra effects, and arithmetic outside signed-i32 addressability. | Focused CPU/WGSL tests cover exact NaN fill bits, canonical CPU differential, 4x5 and 4x4x4 padding, nonzero root offsets/canaries, always-false zero-read behavior, reject policy, semantic-widening rejection, forged authority, shared/overlapping/wrong roots, and intermediate overflow. Existing core/control/L1 tests remain green. |
| Compiler required-device layout conformance | `verified` | The required/advisory lane prepares the exact compiler proof once per case, differentially checks semantic-core physical indices against CPU traces, validates one single-dispatch prepared topology, executes on real WebGPU, reads back complete source and destination allocations, drains queue/late errors, races device loss/timeouts, and emits one validated terminal record. Headless absence is failed/not-run by mode. Compiler publish workflows retain an exact-SHA log and prepublish rejects a missing, stale, dirty, or foreign marker. | Re-run required mode on the exact future release commit. The current record's workgroup count is derived from the prepared plan, not captured at `dispatchWorkgroups`; rename it planned or add an immutable actual-dispatch trace in the padded-L2 evidence upgrade. The generic compiler runner also does not yet claim the kernels view-copy runner's complete scoped error taxonomy. |
| Canonical view-copy artifact construction | `verified` | One generic sink snapshots closed canonical JSON, sorts set-like symbols/constraints, normalizes resource-bounded layout algebra, fixes source/destination allocation/map/view roles, forces disjoint materialization, verifies both opaque artifacts, and returns canonical IDs plus hashes. Its dense-permutation wrapper derives output shape, balanced row-major stride/storage expressions, effects, and reject policy from only canonical input shape, axes, and dtype. Producer/artifact metadata is non-semantic. | No initial-constructor work remains; future frontends must use this sink and keep routing identity out of semantic input. | 11 semantic-core files/99 tests; pinned shared rank-2/rank-3 artifact hashes and exact CPU bits; zero extent, >i64 byte lengths, u64 overflow, rank/expansion budgets, transport-hash independence, set-order independence, mutation/authority, hostile-input rejection, packed export/runtime/declaration proof. |
| Typed JIT permutation request emission | `verified` | One `GpuExecutionSubmission` builds the frozen plan once, then emits a separate `browsergrad.jit.tensor-plan-semantic-requests@1.0` envelope for every post-fusion `PERMUTE`. Requests contain only `inputShape`, normalized `axes`, canonical `f32`, and plan-local `valueId`; malformed dtype/rank/extent/axis/output-shape/arg cases fail before bridge execution. Public `permute` and `transpose` normalize negative axes and reject missing, duplicate, non-integer, or out-of-range axes before UOp construction. | No initial-profile work remains; broader view families require new typed request variants, not plan reinterpretation. | Shared rank-2/rank-3 emission fixture plus focused JIT integration; typecheck/lint/diff-check clean; generated Python source synchronized. |
| Resident semantic view-copy dispatch | `verified` | Kernels executes a module-authorized prepared view-copy directly from a resident whole-root `GPUBuffer` through the exact prepared WGSL. The route performs no host upload, readback, or index reconstruction; validates declared and physical source bytes, storage usage, device/dispatch limits, and permits only a nonempty dense zero-offset destination that overwrites its complete root. Direct callers use an async production-scoped API; tensor-plan execution alone receives a private, non-exported synchronous issue capability under its owning scopes. Semantic preparation preserves liveness/release accounting, and failed dispatch/materialization destroys roots and clears pools rather than re-pooling invalid buffers. | No initial-profile work remains; partial destinations still require an explicit initialized-destination contract. | Deterministic fakes prove LIFO validation/OOM/internal scope initiation, delayed loss, synchronous issue/pop/completion failures, no pre-scope handle mint, no upload/readback/legacy dispatch, poisoned-pool rejection, and clean retry. The JIT required-device lane proves resident roots followed by exactly one explicit complete-root materialization. |
| JIT semantic request consumption and execution | `verified` | Kernels accepts only the closed JSON request envelope, requires exact one-for-one ordered `PERMUTE` coverage, constructs artifacts solely through semantic-core, excludes `valueId` from every semantic hash, and authority-binds prepared WGSL. JIT sends one fused submission and chooses separate semantic materializing/resident methods; missing methods fail before legacy dispatch. The semantic schedule projection erases PERMUTE args, so execution cannot recover axes from the frozen plan. Live semantic handles retain the exact authority-bound preparation and its per-dispatch profile promises until release. | Re-run required evidence on the exact future release commit. | Shared fixture tests; mock legacy refusal/no-readback selection; per-handle preparation/profile lifecycle tests; advisory/required environment behavior; validated two-case Apple Metal 3 terminal pass with one actual submitted workgroup profile and one explicit readback per resident root. |
| JIT required-device semantic-permute conformance | `verified` | Before Chromium starts, the JIT lane runs production `_tensor_plan_submission` over each shared fixture and captures its exact plan plus canonical request JSON. The browser executes those captured submissions through `run_tensor_plan_resident_semantic`, compares the bridge's authority-bound execution trace with the prepared manifest, requires one settled actual dispatch profile with an exact legal timing/confidence pair, verifies one resident complete root, performs exactly one explicit materialization, compares every u32 bit, drains queue/late errors, races loss/timeouts, and emits one validated terminal record. Planned and submitted topology are separate. The artifact hash is recomputed from the complete ordered plan/request/input/expected/backend manifest; a separate domain-separated terminal-manifest hash binds the full outcome, environment, source revision, stage, diagnostics, errors, observations, and timestamp without a hash cycle. | Release CI must repeat and retain the terminal log for the exact clean release commit after the `0.9.0` metadata alignment. | Exact clean-source retained-log verification passed on Apple Metal 3 for source revision `09d868d077e02b8f8727d9b65923d19969650761`, artifact `5de13ee553a25bb45fdfe198267c64c0ac505903c8f44946c58a34ae7b5121a3`, case set `962280999544fc424e23205ec9bd40e7ec5186d32d72027c0fd0239321d556cc`, device profile `0ef434a09b4cc9919ba30c92e791dc2a2138ef42a6f02cc282d1ce1d07d22b63`, and terminal manifest `1d2055f8b0602d18ad957344f04787f81842c5fe76674f477bd27bdf52381250`. Exact-SHA/clean-source prepublish and JIT/kernels workflow gates remain implemented. |
| Compiler Gate 2 release-version alignment | `verified` | Compiler is `0.2.0`, matching the public layout-binding API boundary and its semantic-core/kernels `0.2.0` dependency chain. Package and workspace changelogs describe the change. One generic closure preflight now verifies every transitive public workspace dependency before any selected-package device evidence. | Publish remains an explicit release operation after exact-commit device evidence; no package was published here. | Packed-tarball version/dependency checks, fresh-consumer runtime/declaration proof, generic closure/order tests, workflow syntax, and release-workflow prerequisite assertions. |
| JIT Gate 2 release-version alignment | `verified` | JIT is `0.9.0`, matching its new public semantic-request/bridge boundary. Kernels `^0.2.0` and Pyodide `^0.26.4` are standard optional peers; semantic-core remains kernels-owned. `pnpm pack` removes every workspace protocol. Both pnpm and npm JIT-only offline consumers import/typecheck root, source, and Node-adapter surfaces without either optional peer; an integrated packed consumer resolves kernels and semantic-core. Publish-all uses a deterministic cycle-checked dependency graph over runtime, optional, and peer edges and rejects private workspace targets under any range. | Publish and exact-release-commit device evidence remain explicit release operations; no package was published here. | JIT build, both typechecks, 4 files/24 unit tests, 23 files/234 integration tests, lint; publish-order tests; packed pnpm/npm runtime and declaration consumers; frozen lockfile install. |
| Grad `0.5.2` release metadata alignment | `verified` | Grad now uses standard optional Pyodide peer metadata, preserves kernels as its required runtime dependency, and leaves semantic-core ownership with kernels. Its version moved from immutable published `0.5.1` to `0.5.2`; no runtime/API behavior changed. | Registry publication remains an explicit release operation. | Full Grad build/codegen/typecheck/lint, 30 unit and 322 integration tests, frozen install, packed exact dependency/peer assertions, and fresh npm Grad/kernels/semantic-core consumer. |
| Immutable npm release pipeline | `verified` | Validation and every target `prepublishOnly` run without npm credentials/OIDC after all target baselines are captured. Staging rejects tracked changes and every untracked source except its declared output directory. pnpm then packs once with lifecycle scripts disabled; raw bounded gzip/tar preflight rejects extension metadata, links, specials, decompression bombs, nonportable collisions, and file/ancestor ambiguity before semantic parsing. Protected publication copies only the exact closure into a private single-link artifact directory, strips auth/password/user/cert/key/token and OIDC authority plus user/global npm configs from every verification child, and rechecks identity/SRI immediately before scripts-disabled publication. npm `>=11.12.0` must cryptographically verify the exact returned attestation bundle; only that bundle's single SLSA statement can establish subject/workflow/ref/repository/commit identity. Manual batch mode stages/audits all seven current versions and publishes only missing versions; selected-tag resume requires exact current identity, while batch/prior dependencies require an approved workflow/ref and an attested commit reachable from `origin/main`. Exact dependencies are rechecked immediately before each registry mutation. Runtime `0.1.2` and primitives `0.1.1` replace locally drifted immutable versions. | None for the pipeline implementation. Registry publication and exact-device evidence remain explicit release operations, not local implementation claims. | Commit `155161b7`; clean detached-worktree frozen install/build; exact semantic-core `0.2.0` stage with 124 files, tree `47fe9853...`, SRI `sha512-Eegpt07B...HjCNgoA==`, and source revision `155161b7b36f691ca9935045af48488fffbab265`; 19 adversarial tar tests; 35 Node release-security tests; all-seven packed/fresh-consumer suite; architecture/YAML/static gates; seven-version live dry-run. |
| Operational release instruction convergence | `partial` | `RELEASING.md` and `docs/platform/release-readiness.md` require the protected staged-only path. | `DEVELOPMENT.md`, `RTK.md`, and `docs/platform/agent-consumption-guide.md` still instruct direct `npm publish` or `pnpm publish`. Those files contain pre-existing overlapping edits and were intentionally not modified or staged in this slice; reconcile them without losing their owner’s work before calling repository-wide release guidance consistent. | Exact stale lines recorded at `DEVELOPMENT.md:113`, `RTK.md:122-123`, and `docs/platform/agent-consumption-guide.md:72-73`. |

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
- Principal/adversarial review of the first CPU draft found destination
  self-overlap, SharedArrayBuffer alias/concurrency, spoofable typed-array
  properties/methods, dynamic offset/alignment gaps, unbounded per-element AST
  reconstruction, profile rules mixed into generic verification, unversioned
  operations, ambiguous multi-operation ordering, and dynamic cache-key drift.
  The landed contract closes each issue: dense destination proof, shared-memory
  rejection, native internal-slot validation and direct byte copies, resolved
  geometry/alignment checks, compiled evaluators with deterministic work and
  scratch budgets, generic/profile separation, `view-copy@1.0`, one operation
  per kernel-v1 artifact, and binding-sensitive specialization hashes.
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
| D-018 | 2026-07-15 | accepted | `browsergrad.kernel@1` contains exactly one independently versioned `view-copy@1.0`; host graphs own sequencing. Generic verification checks semantic validity, while the shared positive-affine portable profile and backend profiles own dtype/rank/space/integer limits. Prepared specializations include resolved bindings/guards in their hash and are bounded independently by elements, evaluation steps, and scratch bytes. The initial destination must be proved dense and injective; shared runtime memory is rejected without synchronization semantics. | Prevents backend limits from becoming wire meaning, ordinal arrays from becoming an accidental program, dynamic cache collisions, write races, buffer spoofing/concurrency, and browser hangs from valid but multiplicatively expensive artifacts. |
| D-019 | 2026-07-15 | accepted | Resolve, prove, and hash a view-copy once through backend-neutral `prepareViewCopySpecialization`. CPU and device backends consume the same prepared accessors, portable profile, coordinate proof, and specialization hash; only interpreter backends request the optional source-offset cache. | Having kernels call a CPU-branded API would invert ownership and allocate unnecessary per-element scratch, while duplicating binding, guard, and destination proofs would create a second meaning for the same artifact. |
| D-020 | 2026-07-15 | accepted | The first WGSL profile lowers canonical arithmetic as interval-proved signed i32, converts to word indices only after the source guard, binds whole root allocations as u32 words, and layers a device-specific backend hash over the shared semantic-specialization hash. Required evidence acquires through `navigator.gpu` and fails on missing adapter/device; adapter identity and input hashes are evidence only. | Padding maps can have negative intercepts outside their true predicate, so global u32 lowering wraps incorrectly. f32 loads/stores can canonicalize NaN payloads. Whole-root u32 bindings preserve exact bits and keep view offsets semantic. Separating cache facts from evidence avoids cache fragmentation by adapter labels or test inputs. |
| D-021 | 2026-07-15 | accepted | WebGPU view-copy plans are module-authorized and deeply immutable; the runtime admits one operation per `GPUDevice`, budgets owned host/GPU working bytes, suppresses timed-out/aborted results until cleanup settles, scopes diagnostics to creation/submission rather than readback, and invalidates all participating wrapper caches through device-loss watchers. A conformance run emits one schema-validated terminal evidence record only after the full ordered case set and late-error drain. | Prevents WGSL/hash TOCTOU, caller-forged plans, aggregate queued-memory growth, interleaved error scopes, stale results, cache reuse after loss, partial pass lines, version drift, and failure records that cannot reproduce the current artifact/input/stage. The long-term cross-operation coordinator still belongs in `KernelDeviceImpl`; Gate 2 remains partial until actual-device and frontend evidence pass. |
| D-022 | 2026-07-15 | accepted | Adapt compiler layout bindings after ordinary runtime lowering and before semantic-IR verification. The first profile accepts only direct read-only global `f32` access from one unmodified, non-escaping `uint` logical index dominated by its exact logical-domain guard; it interval-proves positive-affine rank-2/3 maps and root containment, then places one physical index expression in the existing memory reference consumed by both CPU and WGSL. The compiled wrapper is frozen and privately bound to its prepared proof, runtime admission validates the complete verified root allocation, and full semantic plus binding hashes enter pipeline identity; frozen semantic schemas gain no fields. | This preserves one indexing truth and one verification pipeline while preventing legacy CPU zero-fill, WebGPU robust-buffer behavior, indirect mutation, forged proof metadata, aliases, pointer rebasing, integer wrap, undersized buffers, or pipeline-cache collisions from silently substituting semantics. Padding and richer signed maps require their own structured lowering rather than widening this contract. |
| D-023 | 2026-07-15 | accepted | Keep WebGPU evidence infrastructure test-only until the production capability/evidence package seam is implemented, but share one neutral repository module for real adapter/device acquisition, required/advisory outcomes, generic `browsergrad.execution-evidence@1` terminal validation, and exactly-once emission. Each suite owns its ordered cases, observations, comparison policy, and stage diagnostics. | A second package was about to copy the kernels evidence policy. Sharing the stable generic test contract prevents divergent false-green behavior without prematurely publishing or freezing a semantic-core `/capability` API around a test harness. |
| D-024 | 2026-07-15 | accepted | Compiler layout conformance uses prepared single-dispatch execution, complete finite-f32 source and output bit comparison, semantic-core/CPU physical-index differential evidence, full semantic/binding/compile/WGSL hashes, explicit queue and late-error drainage, timeout/device-loss races, and no optional WebGPU features. Required release workflows retain the terminal log under the exact commit, and compiler prepublish has its own SHA/cleanliness marker. | One-shot execution hid topology/cleanup facts, output-only comparison could miss source mutation or tail writes, and the kernels release marker cannot authorize a compiler capability. Compiler padding and arbitrary NaN-payload preservation remain outside this profile rather than being inferred from finite f32 success. |
| D-025 | 2026-07-15 | accepted | All frontend-created view copies pass through one semantic-core constructor that owns role ordering, entity IDs, disjoint aliases, effects, overlap, normalization, verification, and hashes. A dense permutation wrapper accepts only canonical input shape, axes, and dtype; transport metadata is a separate non-semantic option. Rank/dtype backend limits remain legalization policy. | Prevents compiler/JIT from independently reconstructing output shapes, strides, allocation sizes, aliases, or IDs; keeps zero extents exact; makes cross-frontend hash equality an architectural consequence rather than a fixture convention; and leaves the constructor reusable beyond the initial rank-2/3 f32 profile. |
| D-026 | 2026-07-15 | accepted | Bump compiler from published `0.1.2` to `0.2.0` before any release containing the verified-layout preparation/lowering APIs and direct semantic-core dependency. A compiler tag release must verify the exact workspace semantic-core and kernels versions already exist on npm before it can publish. | Reusing `0.1.2` would make published bytes mutable and hide a new public/API dependency boundary. The minor bump follows the repository's pre-1.0 compatibility policy and preserves topological release order: semantic-core, kernels, compiler. |
| D-027 | 2026-07-15 | accepted | Keep the frozen tensor plan and new semantic requests as separate members of one JIT execution submission. Emit every typed `PERMUTE` request from the already-fused plan traversal with canonical string extents, normalized axes, `f32`, and plan-local `valueId`; reject unsupported requests before bridge execution. `valueId` may correlate runtime entries but cannot enter semantic-core constructor input or hashes. | Running fusion twice could produce divergent identities and waste work; extending the frozen plan would turn shape/axes compatibility fields into new semantics; delaying validation to kernels would leave malformed public IR in circulation. The separate closed/versioned envelope gives kernels enough canonical constructor input without duplicating output shape, strides, allocations, effects, aliases, or IDs. |
| D-028 | 2026-07-15 | accepted | Execute prepared semantic view copies over resident whole-root buffers through the prepared canonical WGSL. Initial resident output allocation is legal only when the verified destination is nonempty, dense, zero-offset, and overwrites the complete root; broader partial destinations require an explicitly initialized destination-binding contract. | Reusing the legacy permutation kernel would reconstruct offsets. Using the host conformance runner would add upload/readback. Allocating an uninitialized partial destination would expose unspecified bytes. The narrow resident dispatch preserves semantic authority and device residency without inventing destination initialization semantics. |
| D-029 | 2026-07-15 | accepted | Treat semantic-route `TensorGpuPlan` as scheduling/liveness projection only: erase `PERMUTE.arg` in the submission, strictly correlate each side-table request to its plan value/source/output projection, construct and hash solely from the request, and require a separately named semantic bridge method. Existing `gpu_plan_summary()` and legacy plan execution remain frozen compatibility paths. | VJP metadata made legacy args non-serializable and proved that allowing plan axes beside the request would preserve two truths. Erasure prevents accidental fallback or later reconstruction; separate bridge names make an old kernels package fail closed instead of silently running `permuteDirect`. The change narrows the adapter and advances its retirement gate without widening its schema. |
| D-030 | 2026-07-15 | accepted | Semantic-core owns the versioned dense-permutation cross-frontend fixture as an exact packed JSON subpath. It includes only semantic constructor input, derived output shape, complete source/expected u32 words, and pinned layout/kernel hashes; JIT-local `valueId` is explicitly excluded. A strict shared decoder, whole-file architecture hash, ordered case manifest, package export, and fresh-consumer gate make every consumer fail on drift or partial coverage. | A JIT-owned fixture would make one frontend the semantic authority; copied TS constants would permit divergent truths; ordinary float values would miss signed-zero, subnormal, infinity, and NaN-payload corruption. The shared fixture keeps routing separate from meaning and makes rank-2/rank-3 equality executable across all current consumers. |
| D-031 | 2026-07-15 | accepted | JIT owns its strict actual-device conformance lane but consumes the semantic-core fixture and production kernels bridge. The lane first captures the exact plan and canonical request wire from production `_tensor_plan_submission`, then invokes only the resident semantic bridge entrypoint, proves one resident complete root before exactly one explicit readback, compares complete raw-u32 output, records semantic/request/plan/input/output/topology/device hashes, and fails required mode on device absence. JIT and kernels releases must run and retain the lane by exact commit SHA; JIT prepublish rejects missing, stale, dirty, or foreign evidence. | A kernels-only proxy or hand-built same-shaped plan would not establish the JIT producer contract; rebuilding axes from the plan would preserve two truths; output-only float comparison would miss NaN payloads and residency regressions. One chained Pyodide-emission/browser-execution run proves the real seam. |
| D-032 | 2026-07-15 | accepted | Resident semantic execution returns the exact authority-bound preparation that supplied its private prepared-WGSL map. The production bridge attaches that preparation and the exact dispatch-profile promises to the live output handle, exposes an async read-only trace, deep-freezes settled topology, and drops trace reachability with handle release. Prepared metadata names logical invocations and planned workgroups; only `DirectDispatchProfile` is called submitted topology. JIT terminal `artifactHash` binds the complete ordered prepared case manifest, including exact emitted plan/wire, input/expected bits, semantic/backend hashes, and planned topology; validation recomputes every component and aggregate. | Preflight preparation alone could differ from the independently executed preparation. Calling planned ceil-divisions submitted topology could hide over-dispatch. A partial aggregate hash could survive request/plan/input changes. Per-handle authority and recomputation close all three gaps without a racy global last-trace or a second execution entrypoint. |
| D-033 | 2026-07-15 | accepted | Keep the deterministic prepared-artifact hash chain separate from a domain-separated `terminalManifestHash` over the canonical complete terminal record with only that digest excluded. Validate exact producer versions, closed environment semantics, and a closed pass/failed/not-run state matrix before emission. | Artifact identity must remain reusable across runs, while retained evidence must also bind outcome, environment, device provenance, completed-prefix/current-case state, stage, diagnostics, uncaptured errors, failure details, and timestamp. Separating the hashes avoids recursion and prevents either role from weakening the other. |
| D-034 | 2026-07-15 | accepted | The JIT `prepublishOnly` lifecycle runs mutating build/codegen before the exact-commit cleanliness/evidence guard, and release tests pin that ordering. | Checking the marker first allowed generated Python to drift after evidence validation and before packing. Running the guard last makes any post-evidence codegen change dirty and blocks publication. |
| D-035 | 2026-07-15 | accepted | Every production semantic resident dispatch and materialization issue phase pushes `internal`, `out-of-memory`, then `validation`; synchronously issues all GPU calls; initiates every pop in reverse order before the first await; and races pop/operation settlement against device loss. Public direct dispatch is async and scoped; the tensor-plan synchronous issuer is private and unexported. Any diagnostic failure destroys the produced root/readback, clears pipeline/output pools, settles profiles, and cannot mint a handle or return a poisoned buffer to a pool. | `uncapturederror` and a later queue tick are telemetry, not authoritative error ownership. Without scopes at the exact synchronous issue site, late validation could false-pass; without private ownership, future callers could bypass the contract; without failure-specific pool handling, dead or invalid roots could be reused. |
| D-036 | 2026-07-15 | accepted | Release authorization requires one bounded regular retained log containing exactly one compact reporter-prefixed terminal record whose whole-record digest, exact producer versions, full source revision, legal successful state, and canonical cases validate. A shared source-scope list must be clean for the verifier and final JIT/kernels publish guards. Evidence, verifier, always-upload, and publish steps have pinned order/conditions; JIT, kernels, and compiler publish guards run last after every mutating verifier, and kernels also requires the exact JIT semantic evidence marker. | A digest proves integrity, not commit authority; HEAD alone can label dirty bytes as committed; post-gate build/codegen can mutate publish inputs; unbounded or non-regular logs can exhaust/block the verifier; and mismatched workflow conditions can let kernels publish without the JIT lane. The combined contract closes those bypasses while retaining failure logs for diagnosis. |
| D-037 | 2026-07-15 | accepted | Release the JIT semantic-request boundary as `0.9.0`. Model kernels `workspace:^` and Pyodide `^0.26.4` as standard optional peers through `peerDependenciesMeta`; do not add a direct semantic-core dependency or peer. Pack only through pnpm. Derive publish-all order from every public workspace runtime, optional, and peer edge regardless of source or packed range; reject private/missing workspace targets and cycles. Before downstream evidence, require exact published kernels/semantic-core versions, the kernels-to-semantic-core edge, and byte/mode/path-equivalent unpacked local and immutable registry artifacts. Publication is restricted to a protected environment and a SHA reachable from `main`; an existing equivalent npm version and GitHub release are resumable rather than fatal. | JIT core/source installation is backend-neutral, while kernels owns semantic preparation and its semantic-core version. Nonstandard optional-peer metadata was not enforced by package managers; npm pack preserves workspace protocols; dependency-only sorting could publish JIT before kernels; version existence alone could certify locally drifted dependency bytes. Standard metadata, isolated/integrated packed consumers, deterministic graphing, registry-tree equality, and source/ref constraints prevent workspace-only success or broken/resumed releases without forcing optional backends onto core-only consumers. |
| D-038 | 2026-07-15 | accepted | A production npm release is one immutable pnpm-packed tarball created after all mutating validation in a credential-free job, identified by exact spec, source SHA, SHA-512 SRI, and a bounded canonical file snapshot. The protected job may run no workspace install or package lifecycle script; it publishes only that tarball with npm scripts disabled. Success and resume both require exact registry integrity/tree equality, registry signature verification, one SLSA v1 provenance statement, and exact subject/repository/workflow/ref/commit identity. | Publishing a live package directory could rerun lifecycle code and produce bytes never tested. Job-wide tokens/OIDC exposed write authority to dependency installs and browser tests. Tree equality alone could resume a package published without valid provenance. Two-phase staging, least authority, strict archive inspection, and post-publication cryptographic/semantic verification close those gaps. |
| D-039 | 2026-07-15 | accepted | Release Grad's packaging-only correction as `0.5.2`: use standard optional Pyodide peer metadata, retain kernels as the required runtime dependency, and do not add semantic-core directly. Generic release closure must therefore enforce semantic-core → kernels → Grad before a Grad tag can proceed. | Local Grad `0.5.1` bytes had drifted from the immutable registry version and its nonstandard peer field was not enforceable. A new patch version preserves npm immutability and package ownership without claiming runtime behavior changed. |
| D-040 | 2026-07-15 | accepted | A staged release manifest must name targets in deterministic dependency order, bind selected-package mode to exactly one matching target, and contain exactly the computed dependency closure. Downloaded artifacts are capped, single-link regular files copied through open descriptors into a private `0700` directory, hashed while copied, identity-checked, and reverified immediately before publish. Every validation/registry-read child receives isolated tokenless user/global npm configs with token, auth, password, username, OTP, cert, key, GitHub, and OIDC authority removed; only the exact `npm publish <tarball>` child inherits publication authority. A fresh batch after partial publication must stage/audit every current public target, apply approved-workflow/ref plus protected-main-reachability provenance to existing versions, and publish only missing versions. Selected-tag resume alone requires exact current workflow/tag/commit identity. | Manifest labels are not authority, downloaded artifacts can be swapped between validation and `npm publish`, project/global config and non-token npm credentials can bypass token-only stripping, and filtering an already-published package out of a fresh batch could bypass resume proof. Exact plan equality, private immutable copies, isolated npm config, complete credential removal, and explicit batch-versus-tag provenance states close these gaps without falsely requiring every prior batch target to have been produced by the current release commit. |
| D-041 | 2026-07-15 | accepted | Require npm `>=11.12.0` for provenance verification and decode semantic identity only from the exact `attestationBundles` entry returned by `npm audit signatures --include-attestations` for the installed root package. Pin `11.12.1` in protected jobs, install it with scripts disabled and OIDC request variables blank, and compare all three semver components. | Validating a separately fetched DSSE document beside a successful signature audit does not prove that npm cryptographically verified that same statement. npm versions before `11.12.0` do not expose the verified bundle required to bind those facts. |
| D-042 | 2026-07-15 | accepted | Release the accumulated additive runtime assignment/platform surface as `0.1.2` and reissue unchanged primitives behavior as packaging-only `0.1.1`. Never treat local bytes that differ from immutable runtime `0.1.1` or primitives `0.1.0` as resumable. | The all-target batch audit exposed real registry drift hidden by the former missing-only filter. Reusing published versions would either fail immutable equivalence or encourage a bypass; explicit new versions preserve npm immutability and make the staged artifact contract reproducible. |
| D-043 | 2026-07-16 | accepted | Keep compiler padding behind a sibling L2 view-copy authority rather than widening the frozen L1 read binding. Consume one exact verified operation and shared specialization; accept one pure guarded materializing-copy source shape; lower signed-i32 predicate/location arithmetic into existing structured branch/store IR over raw-u32 roots; evaluate source address/load only in the true arm; require exact distinct non-shared allocations; bind layout, kernel, specialization, and routing hashes into compile identity. | L1 has no effect/fill authority and its unsigned always-in-bounds contract cannot represent padding. A separate authority preserves frozen schemas and L1 compatibility, prevents caller-provided fill or robust-buffer substitution, preserves NaN payloads, and leaves future richer operations at the semantic operation seam instead of source-pattern handlers. |

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

### 2026-07-15 — Gate 2 view-copy contract and CPU reference

- Added the dependency-free `/kernel` subpath and `browsergrad.kernel@1` with
  one independently versioned `view-copy@1.0` operation tied to the exact
  verified layout hash and canonical view IDs.
- Kept generic semantic verification separate from the shared initial portable
  profile. The profile currently admits positive-affine f32 rank-2/3 global
  views and rejects negative strides plus division/modulo until target integer
  equivalence is proved.
- Refactored coordinate traces through prepared accessors, compiled index-map
  evaluators once per specialization, and kept dominating-predicate behavior
  differential with the Python oracle.
- Added a prepared CPU reference with guarded source preflight, exact f32 fill
  bits, dense destination proof, cached source offsets, dynamic specialization
  hashes, separate element/evaluation-step/scratch/wall-time budgets, and
  cooperative scheduler yielding with abort checks.
- Hardened runtime bindings against short, overlapping, misaligned, shared,
  subclassed, proxied, or property-spoofed typed arrays. Copy execution uses
  native indexed byte slots rather than overridable user methods.
- Covered transpose, rank-3 permutation, positive strided slice, broadcast,
  padded reads, dynamic and zero shapes, byte-unit maps, and nonzero source and
  destination offsets. Negative/profile, geometry, access, alias, resource,
  version, hash, and buffer mutations fail closed.
- Bumped the locally packed package to `0.2.0` and made the release-package
  harness execute a real transpose from the extracted tarball. No registry
  publish, push, deployment, WebGPU conformance, or two-frontend support is
  claimed.

### 2026-07-15 — Backend-neutral view-copy specialization seam

- Extracted verified operation selection, binding normalization, portable
  profile legalization, coordinate/access proof, dense-destination proof,
  resource limits, and specialization hashing from the CPU wrapper.
- CPU execution now consumes the same backend-neutral preparation API intended
  for kernels-owned WGSL. Only the CPU reference requests the optional exact
  source-offset cache; device lowering does not inherit CPU scratch storage.
- Preserved all 81 semantic-core tests and added direct parity coverage for the
  backend-neutral and CPU specialization hashes.

### 2026-07-15 — Kernels-owned WGSL view-copy lowering

- Added kernels' first runtime dependency on semantic-core through its narrow
  `/schema`, `/layout`, and `/kernel` protocols; the architecture check rejects
  other semantic-core subpaths and any compiler/framework dependency.
- Lowered canonical source and destination index maps with per-node signed-i32
  interval proofs. Source word conversion and loading stay inside the true
  predicate branch, so negative padding intercepts do not wrap through u32.
- Bound whole root allocations as u32 words to preserve exact f32 and NaN bits,
  including exact padding fill. The generated WGSL contains no eager `select`,
  address clamping, implicit zero-fill, or ignored-write fallback.
- Added device/owned-working-set legalization, exact runtime word-buffer
  validation, authority-bound deeply immutable plans, full-digest pipeline
  names, two-level semantic/device hashes, one-in-flight ownership,
  zero-element no-submit, timeout/abort stale-result suppression, exact LIFO
  scope drainage, and typed shader/pipeline/validation/memory/device-loss/
  execution diagnostics.
- Added the nine-case actual-device matrix: rank-2 transpose, rank-3
  permutation, positive strided slice, read-only broadcast, byte maps with
  nonzero offsets, exact-NaN padding in ranks 2 and 3, dynamic rank-2
  specialization, and zero-extent no-submit. One validated terminal record
  binds the full ordered artifact/input/case set and records logical
  invocations separately from submitted workgroups/pipelines, adapter versus
  negotiated features, relevant limits, producer versions from package
  metadata, stage/current case on failure, environment, and bit-exact
  whole-destination comparison policy.
- Bumped kernels to `0.2.0`. Packed metadata rewrites the semantic-core
  workspace dependency to exact `0.2.0`; a temporary consumer installs both
  tarballs, resolves the bare public subpaths, runs the lowering, and typechecks
  against packed declarations. No registry publication is claimed.
- Official bulk/tag workflows run the strict lane before kernels publication,
  retain its complete commit-addressed log, and pass an exact-HEAD evidence
  marker. Kernels' `prepublishOnly` hook rejects missing/stale markers, so the
  documented direct/manual path cannot accidentally bypass device evidence.
- Headless Chromium exposed no adapter. Advisory mode recorded one actual
  not-run; required mode emitted failure evidence and exited nonzero. This
  verifies strict absence handling. A subsequent headed Chromium run acquired
  Apple Metal 3 and passed all nine cases with one validated terminal record;
  artifact hash `446889ab9e081a277508552c19b376cfcc44a499cdcec6c14cffd6c05342e64c`,
  case-set hash `84320356003ad4f26995d3dcdb3ea5331b51ea80cdb3b83c110483adad7fd337`,
  and device-profile hash
  `9589abc8fafb412d83194febaf210f7f89da7a580bf20d3272e1eef9dcda2f66`.

### 2026-07-15 — Compiler verified-layout binding preparation

- Added a separate asynchronous preparation boundary for verified layout
  artifacts rather than placing opaque artifacts inside `CompileCudaLiteOptions`.
- Bound each source parameter to an explicit read-only, row-major-flat logical
  ABI; resolved dimensions and canonical entity IDs once; rejected duplicate
  parameters, malformed requests, non-global memory, and bounded-resource
  overflow before compiler lowering.
- Kept authority in a private side table and exposed only deeply immutable
  facts, the semantic layout hash, and a deterministic binding-projection hash.
  The frozen compiler pointer/scalar schemas remain unchanged.
- Added a distinct layout-bound compile-cache key carrying both hashes, so
  different views, semantic layouts, dynamic specializations, and compile
  options cannot alias. Packed and fresh-installed compiler consumers exercise
  the public API and strict declarations through exact semantic-core metadata.

### 2026-07-15 — Compiler read-only verified-view lowering

- Added a separate layout-bound compile path after CUDA-lite runtime lowering
  and before semantic verification; the ordinary compiler path and all frozen
  semantic record shapes remain unchanged.
- Lowered one guarded row-major-flat logical `uint` into verified rank-2/3
  coordinates and then the canonical positive-affine physical map. CPU
  reference, verified/typechecked IR, WGSL legalization, and later WebGPU
  emission all consume that same rewritten memory expression.
- Proved logical and physical u32 ranges, root-allocation containment, byte-map
  divisibility, and exact guard dominance. Rejected writes, aliases, helpers,
  pointer rebasing/offsets, mutable or signed indices, conditional predicates,
  non-affine maps, unsupported ranks/dtypes, and unsafe arithmetic.
- Added differential trace/reference cases for transpose, positive slice,
  broadcast, byte-unit maps, rank-3 permutation, and dynamic specialization;
  packed and fresh-installed consumers compile and execute the transpose.
- Closed adversarial execution gaps before commit: all operation-level writes
  and address escapes invalidate logical-index proofs; typed and resident input
  buffers must cover the complete verified root allocation; compiled wrappers
  use private instance authority; and the complete semantic/binding hashes are
  part of the WGSL program and pipeline-cache identity. Public lowering errors
  are source-spanned stable compiler diagnostics registered in compatibility.
- Kept Gate 2 `partial`: this commit has CPU and generated-WGSL structural
  evidence, but the compiler-specific required-device run and JIT frontend
  convergence remain.

### 2026-07-15 — Shared required-WebGPU evidence test contract

- Extracted real adapter/device acquisition, required-versus-advisory outcome
  rules, generic terminal-envelope validation, and exactly-once emission into
  a neutral unpublished `test-support` module.
- Kept suite-specific case matrices, observation schemas, comparison checks,
  stages, and diagnostic mapping in their owning package; no public semantic-
  core capability/evidence export was created prematurely.
- Migrated the kernels nine-case lane to the shared contract without changing
  its artifact, case-set, or no-adapter hashes. Added mutation coverage for a
  required `not-run`, a passed record without device profile, passed
  diagnostics, and a second terminal emission.

### 2026-07-15 — Compiler required-device layout conformance

- Added an ordered six-case compiler matrix for rank-2 transpose with nonzero
  offset, positive strided slice, read-only broadcast, byte map with nonzero
  offset, rank-3 permutation, and dynamic rank-2 specialization. Padding stays
  rejected by the compiler profile and is not claimed by this lane.
- Prepared each compiled program once, required one single-dispatch/one-step
  topology, differentially matched semantic-core physical indices to CPU
  traces, and compared complete cloned source and output allocations bit-exact
  over finite f32 patterns after real WebGPU execution.
- Bound terminal evidence to layout, binding projection, compile identity,
  WGSL module, physical-index, input, launch, case-set, producer-version,
  adapter-feature, negotiated-feature/limit, and device-profile hashes. Queue
  drainage, a late-error task, uncaptured errors, timeouts, device loss, and
  prepared-resource destruction are explicit.
- Added advisory and required commands. Headless no-adapter behavior emits one
  validated `not-run` or `failed` record respectively; headed Chromium passed
  all six cases on Apple Metal 3 with artifact hash
  `55b7c44c8fb209d11e886a3aa559962852ee4b7220bfbb8cac3a0429d74c3b63`,
  case-set hash
  `a6411afd6b14973bba169cd422e27f472621d8189fc6a73b1f214cfe8783bf86`,
  and device-profile hash
  `0ef434a09b4cc9919ba30c92e791dc2a2138ef42a6f02cc282d1ce1d07d22b63`.
- Wired the required lane and retained log into bulk and compiler-tag release
  workflows. A compiler-specific prepublish guard requires the exact evidenced
  SHA and a clean compiler/kernels/semantic-core/test-support/lockfile scope.
  The local headed result proves this implementation, not a future release
  commit; release CI must rerun it.

### 2026-07-15 — Canonical view-copy construction

- Added the only frontend-facing construction sink for verified view-copy
  layout/kernel pairs. The sink accepts no allocation, alias, map, view, or
  operation IDs; it fixes role order, snapshots hostile in-memory input,
  normalizes layouts, verifies both authority-bound artifacts, and returns
  their canonical role IDs and semantic hashes.
- Added a dense-permutation wrapper whose semantic inputs are only canonical
  source extents, axes, and dtype. It derives output shape, balanced row-major
  stride and byte-length expressions, global allocations, distinct aliases,
  exact effects, forbid-overlap, and reject-invalid-source policy. Zero extents
  allocate zero semantic bytes; no legacy `max(dim, 1)` rule is imported.
- Canonicalized set-like symbol/constraint order and proved producer/artifact
  metadata cannot affect either semantic hash or canonical entity IDs. Backend
  rank/dtype restrictions stay outside construction and remain profile policy.
- Fixed layout substitution to clone repeated coordinate expressions. The first
  constructor run found that permutation normalization returned a frozen object
  graph with shared coordinate nodes, which wire verification correctly
  rejected as non-canonical JSON. Normalized results now remain both immutable
  and canonical trees. Substitution consumes node/depth budgets while expanding
  so a repeated-coordinate compose chain cannot allocate exponentially before
  the resource limit fires.
- Pinned the canonical rank-2 transpose layout and kernel hashes. The compiler
  consumes that exact constructor layout after proving its explicit logical-
  bounds predicate true over the complete view domain; conditional predicates
  remain unsupported rather than being silently discarded.
- Replaced release-test hand assembly with the public constructor and covered
  the extracted tarball plus a fresh bare-import runtime/declaration consumer.
  Semantic-core stays at unreleased `0.2.0`; wire schema and operation versions
  are unchanged.

### 2026-07-15 — Typed JIT permutation request emission

- Added a frozen execution submission that calls the existing post-fusion plan
  builder once and emits semantic requests beside, never inside, the legacy
  plan. The frozen plan-builder body and schema remain unchanged.
- Added a closed `browsergrad.jit.tensor-plan-semantic-requests@1.0` envelope.
  Each permutation request carries only canonical input extents, axes, `f32`,
  and routing-only `valueId`; it carries no output shape, offsets, strides,
  allocation facts, artifact IDs, effects, or alias policy.
- Initial JIT eligibility is intentionally static positive rank-2/3 `f32`.
  Invalid axes, dtype, rank, zero extent, output shape, or open legacy arg fail
  before bridge execution. Public permutation construction now normalizes
  negative axes once and rejects malformed permutations before UOp creation.

### 2026-07-15 — Resident semantic view-copy dispatch

- Added a resident kernels entrypoint that consumes only module-authorized
  prepared artifacts and dispatches their exact WGSL through the existing
  pooled direct runner. It adds no uniform params or alternate index path.
- Source validation covers declared bytes, physical `GPUBuffer.size`, storage
  usage, device loss, allocation limits, and dispatch limits. Destination
  admission requires a nonempty dense zero-offset full-root overwrite.
- Focused fakes prove no upload, readback, buffer copy, forged-plan dispatch,
  partial-destination allocation, or limit bypass occurs.

### 2026-07-15 — JIT semantic request consumption and execution

- Added a strict kernels parser/preparer for the closed JIT request envelope.
  It requires exact ordered request coverage, compares only transport and
  scheduling projections, then calls the semantic-core dense-permutation
  constructor. Public prepared metadata exposes exact layout, kernel,
  specialization, and WGSL hashes while private authority retains execution
  objects by routing value.
- Added materializing and resident semantic tensor-plan routes. Only prepared
  canonical WGSL executes `PERMUTE`; legacy `permuteDirect` remains available
  to the frozen compatibility API but is unreachable from the JIT semantic
  bridge. Resident output shape comes from prepared semantics, not plan axes.
- JIT now builds fusion once, serializes requests as bounded closed JSON, and
  calls separate semantic bridge methods only when requests exist. A bridge
  lacking them fails before legacy execution; plans without semantic requests
  retain the existing bridge route.
- First full bridge run exposed typed VJP `PERMUTE` args carrying internal
  `vjp_of` metadata and attention tests correctly moving from legacy to
  semantic counters. The producer now admits only known internal VJP metadata
  while erasing the complete PERMUTE arg from the submitted schedule. Kernels
  rejects any semantic-route plan that still carries legacy arg meaning.

### 2026-07-15 — Shared dense-permutation fixture contract

- Added one semantic-core-owned, versioned rank-2/rank-3 fixture containing
  closed constructor requests, derived output shapes, complete raw-u32 inputs
  and outputs, and pinned layout/kernel hashes. `valueId` is absent because it
  is routing identity, not semantic identity.
- Rank-2 data deliberately covers signed zero, a NaN payload, infinity, a
  subnormal, and finite values; rank-3 covers every address in the unchanged
  operation shape. Semantic-core CPU execution matches every output bit.
- Compiler proves the same physical-word mapping and both hashes; JIT proves
  its emitted projection and erased plan arg; kernels reconstructs the exact
  artifacts solely from the side-table request.
- Added strict closed decoding, ordered coverage, whole-file SHA architecture
  enforcement, an exact packed JSON export, and fresh-consumer release proof.

### 2026-07-15 — JIT required-device semantic-permute conformance

- Added a JIT-owned browser lane over the shared rank-2/rank-3 fixtures. Its
  Vitest config boots Pyodide, calls production `_tensor_plan_submission`, and
  injects the exact emitted plan/request wire. Browser execution uses that
  capture through the production resident kernels bridge; no synthetic plan or
  reconstructed request remains.
- Resident semantic execution now returns its exact authority-bound
  preparation. The bridge stores it with the exact dispatch-profile promises
  on the live handle, exposes a deep-frozen async trace, and removes trace
  reachability on release. Prepared topology is explicitly planned; submitted
  workgroups come only from the actual dispatch profile.
- The terminal record retains a complete ordered prepared manifest. Validation
  independently recomputes every plan/wire/input/expected/backend component,
  aggregate case-set hash, final artifact hash, environment/device hashes, and
  observation-to-manifest links before exactly-once emission.
- A pure finalizer snapshots the complete terminal record and adds a separate
  domain-separated whole-record hash. Mutation tests reseal semantically
  invalid records to prove exact producer provenance, environment/outcome
  coherence, ordered current-case state, diagnostics/errors, queue drains, and
  uncaptured GPU errors are validated rather than merely covered by a digest.
- Advisory headless mode records not-run; required headless mode emits one
  failed record and exits nonzero; headed required mode passed both fixtures
  on Apple Metal 3 with one profiled submitted dispatch, one pipeline/invocation,
  one resident root, and one readback per case.
- Added an exact-SHA JIT prepublish gate after build/codegen so generated drift
  invalidates the marker. Publish-all retains the evidence; JIT and kernels
  tag releases both rerun and retain it.
- Added production issue-site error scopes for resident dispatch and explicit
  materialization. Direct callers use the safe async API; tensor-plan execution
  alone receives a private synchronous capability. Failures destroy roots,
  clear pools, settle profiles, and deterministic tests prove recovery.
- Added one retained-log verifier with a bounded regular-file reader, exact
  reporter parsing, whole-record digest/version/source validation, canonical
  clean-source scope, and pinned workflow condition/order checks. JIT, kernels,
  and compiler evidence gates now run last; kernels also requires the exact JIT
  semantic evidence marker.

### 2026-07-15 — JIT `0.9.0` release alignment

- Bumped JIT to `0.9.0` for the semantic-request/production-bridge boundary.
  Replaced nonstandard optional-peer metadata with standard optional kernels
  and Pyodide peers; kept semantic-core exclusively behind kernels.
- Added isolated packed proof that root, source, and Node-adapter imports plus
  declarations work offline without either optional peer. Added an integrated
  packed JIT/kernels/semantic-core consumer that prepares the same canonical
  semantic operation. Packed metadata must contain no workspace protocol.
- Extracted a side-effect-free deterministic publish sorter over public
  runtime, optional, and peer dependencies. It rejects duplicate names,
  malformed maps, missing workspace targets, and cycles; focused tests pin all
  three workspace protocol variants and the repository's real ordering.
- JIT tag releases now verify exact semantic-core and kernels registry
  versions, including the published kernels-to-semantic-core dependency, before
  Playwright/device evidence. They also unpack locally pnpm-packed and registry
  tarballs and require the same path, mode, and bytes, so an immutable
  same-version dependency cannot drift behind local evidence.
- Release and manual-publish workflows require a protected `npm-production`
  environment and a SHA reachable from `main`. Existing versions are skipped
  only after registry-tree equality, GitHub release creation is idempotent, and
  provenance remains enabled. Release docs replace removed legacy token advice
  with trusted publishing plus a short-lived granular fallback.
- Release docs disallow package-local npm publish and document dependency order
  plus evidence-gated manual dispatch.
- Historical exact-commit proof for `09d868d0` is recorded above. `0.9.0`
  release metadata changes require a new exact-clean-commit device record after
  this slice commits; registry publication remains separate.

### 2026-07-15 — Immutable npm artifact and Grad release hardening

- Bumped Grad to `0.5.2` because its local `0.5.1` package tree no longer
  matched the immutable published version. Replaced nonstandard Pyodide peer
  metadata, preserved kernels ownership, and added packed plus fresh-npm
  consumer proof without changing runtime code.
- Added the repository metadata semantic-core needs for trusted publishing.
  Every public package now has one exact monorepo repository/directory identity
  and a complete `prepublishOnly` gate; artifact-mutating pack/publish lifecycle
  hooks are rejected.
- Replaced live-directory publication with a two-job artifact protocol.
  Validation has no token or OIDC permission, runs all semantic/device gates,
  packs once, and uploads tarballs plus an exact source/SRI manifest. Protected
  publication installs no workspace dependency, invokes no lifecycle code, and
  has read-only repository access. A third contents-only job creates the GitHub
  Release after npm publication without receiving npm/OIDC authority.
- Replaced package-specific release prerequisites with one transitive closure
  over runtime, optional, and peer edges. Both workflows share
  `browsergrad-npm-production` concurrency with `queue: max`, so pending release
  attempts serialize rather than overwrite each other.
- Added bounded tarball inspection without extraction: raw gzip/tar preflight
  rejects extension metadata before allocation, enforces compressed and full
  decompressed-stream limits, and admits only regular files/directories. The
  semantic pass enforces canonical `package/` paths, portable Unicode/case
  uniqueness, no file/ancestor ambiguity, exact package identity, and canonical
  path/mode/size/SHA-512 snapshots.
- Staged publication now rejects oversized or hardlinked manifests/artifacts,
  copies exact closure artifacts through open descriptors into a private
  directory, hashes while copying, and revalidates SRI plus package identity
  immediately before `npm publish`. Selected-package labels cannot disagree
  with targets, and artifacts cannot fall outside or omit the computed closure.
- Validation, tar inspection, registry reads, fresh npm installs, and signature
  audit children receive a fixed tokenless npm config with token/OIDC variables
  removed. The publisher admits ambient authority only for the single exact
  `npm publish <staged-tarball>` child.
- Added post-publication proof that exact staged SRI equals registry integrity,
  the registry tree is identical, npm signature audit succeeds, and exactly one
  SLSA v1 statement binds the npm subject digest to the expected GitHub-hosted
  repository, workflow, ref, commit, and invocation. Resume uses the same proof;
  dependencies from a prior release require an allowlisted workflow and an
  attested commit reachable from protected main before any mutation.
- `pnpm test:release-packages` now includes 18 hostile archive cases, 12
  provenance boundary/mutation cases, 4 staged-manifest boundary cases, exact
  coverage of all seven public packages, fresh npm runtime/primitives/Grad/JIT
  consumers, workflow least-authority/order/concurrency assertions, and generic
  release graph assertions. Exact clean-worktree staging remains the final check
  after the release implementation commit.

### 2026-07-15 — Immutable release exit-audit closure

- Bound provenance identity to npm's exact cryptographically verified
  attestation bundle and raised the protected npm floor to `11.12.0` with
  pinned `11.12.1` installation. Mocked tests now distinguish trusted npm audit
  output from raw registry payloads and reject root, location, registry, URL,
  bundle, workflow/ref, and statement mutations.
- Replaced missing-only batch target filtering with an explicit publication
  state machine. Manual dispatch stages/audits all seven current targets,
  publishes only missing versions, uses prior approved provenance for existing
  batch targets, and reserves exact current identity for selected-tag resume or
  newly published artifacts.
- Captured every target's packed baseline before the first mutating lifecycle
  command, descriptor-bound the staged-manifest read, and rechecks exact
  workspace dependencies immediately before each `npm publish` mutation.
  Cleanliness now rejects untracked package inputs while permitting only the
  declared staging-output directory.
- Isolated every read-only npm child from project/user/global npm credential
  config and removed token, auth, password, username, OTP, certificate, key,
  GitHub, and OIDC authority. The fallback token remains visible only to the
  exact publish child.
- The new all-target dry-run exposed immutable drift that the missing-only plan
  had hidden. Runtime's large additive assignment/platform surface moves to
  `0.1.2`; primitives' unchanged behavior moves to packaging-only `0.1.1`.
  Live registry planning now reports all seven current versions as missing;
  nothing was published.
- Focused and packed release gates now pass 19 hostile-archive tests and 35
  Node release-security tests, all seven packed/fresh-consumer checks, frozen
  pnpm install, architecture enforcement, workflow YAML parsing, and whitespace
  validation. Exact clean-commit staging remains the only open exit criterion.

### 2026-07-16 — Compiler L2 structured padding

- Added an opaque compiler view-copy binding that requires both verified layout
  and kernel artifacts, prepares exactly one operation through semantic-core,
  and hashes semantic, specialization, operation, and routing identities.
- Added a compiler-owned affine index-map adapter with exact BigInt subtree
  intervals. Signed comparisons are localized through explicit same-type casts;
  the established general WGSL inference surface remains unchanged.
- Added strict materializing-copy lowering after runtime lowering and before IR
  verification. Existing branch/store IR now carries raw-u32 source/destination
  words; invalid padding writes the exact artifact fill bits without constructing
  or loading a source address outside the true branch.
- Runtime admission rejects forged compiled wrappers, wrong element carriers,
  short/oversized/subview roots, SharedArrayBuffer roots, typed-plus-resident
  duplication, and source/destination aliasing.
- Focused CPU/WGSL coverage proves rank-2/rank-3 exact padding, canonical CPU
  differential parity, nonzero offsets and canaries, always-false zero-read
  behavior, reject-policy refusal, cache/program identity, source-shape refusal,
  and cancellation-hidden integer overflow. Required actual-WebGPU and retained
  release evidence remain before Gate 2 can be verified.
- Post-commit adversarial review strengthened both-end allocation canaries and
  exact destination-write indices, canonical always-false CPU parity plus
  source immutability/read/fill counts, padded-reject cause identity, dense
  reject success, exact overflow code/path, and signed multiplication overflow.

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
| 2026-07-15 | Gate 2 view-copy and CPU reference | `pnpm --filter @unlocalhosted/browsergrad-semantic-core typecheck && pnpm --filter @unlocalhosted/browsergrad-semantic-core test && pnpm --filter @unlocalhosted/browsergrad-semantic-core build && pnpm --filter @unlocalhosted/browsergrad-semantic-core lint && pnpm test:release-packages && pnpm architecture:check && git diff --check` | Passed: semantic-core typecheck/build/lint, 9 files/81 tests, packed `/schema`/`layout`/`kernel` imports and real extracted-tarball transpose, architecture check, and whitespace check. | Gate 2 remains partial; implement kernels-owned WGSL and strict actual-device proof next. |
| 2026-07-15 | Backend-neutral view-copy specialization | `pnpm --filter @unlocalhosted/browsergrad-semantic-core typecheck && pnpm --filter @unlocalhosted/browsergrad-semantic-core test && pnpm --filter @unlocalhosted/browsergrad-semantic-core build && pnpm --filter @unlocalhosted/browsergrad-semantic-core lint && pnpm test:release-packages && pnpm architecture:check && git diff --check` | Passed: semantic-core typecheck/build/lint, 9 files/81 tests, CPU/shared specialization-hash parity, packed execution, architecture check, and whitespace check. | Kernels WGSL may now depend on the shared proof without calling a CPU API or allocating CPU-only offset scratch. |
| 2026-07-15 | Kernels WGSL lowering and package adoption | `pnpm --filter @unlocalhosted/browsergrad-kernels typecheck && pnpm --filter @unlocalhosted/browsergrad-kernels typecheck:browser:view-copy && pnpm --filter @unlocalhosted/browsergrad-kernels test && pnpm --filter @unlocalhosted/browsergrad-kernels lint && pnpm --filter @unlocalhosted/browsergrad-kernels build && pnpm --filter @unlocalhosted/browsergrad-compiler exec vitest run tests/semantic_architecture_check.test.ts && pnpm architecture:check && pnpm test:release-packages` | Passed: both typechecks, kernels 13 files/86 tests including lifecycle/error/device-loss fakes and the exact-commit publish guard, build, 20 architecture-guard tests, architecture check, packed exact dependency/lowering, and fresh two-tarball consumer bare-import execution plus declaration typecheck. Lint exited zero with one pre-existing `realizer.ts` warning outside this slice. | Actual GPUDevice proof remains required. |
| 2026-07-15 | Advisory view-copy browser lane | `BG_BROWSER_HEADLESS=1 pnpm --filter @unlocalhosted/browsergrad-kernels test:browser:view-copy` | Passed command with one test recorded as skipped/not-run because `requestAdapter` returned no adapter. | Advisory result is not conformance evidence. |
| 2026-07-15 | Headless required view-copy browser lane | `BG_BROWSER_HEADLESS=1 pnpm --filter @unlocalhosted/browsergrad-kernels test:browser:view-copy:required` | Failed as designed: emitted one validated `browsergrad.execution-evidence@1` terminal record with `outcome=failed`, `required=true`, the complete nine-case artifact/input manifest, and diagnostic `BG-WEBGPU-EVIDENCE-DEVICE-UNAVAILABLE`; Vitest exited 1 because `requestAdapter` returned no adapter. | Preserve as environment evidence; absence is not a product failure and cannot be a release pass. |
| 2026-07-15 | Headed required actual-device view-copy lane | `pnpm --filter @unlocalhosted/browsergrad-kernels test:browser:view-copy:required` | Passed: headed Chromium acquired Apple Metal 3 with negotiated WebGPU core, all nine ordered static/dynamic/zero-extent cases matched the CPU reference bit-exact over complete destination allocations, queue/late-error drainage was clean, and one validated terminal record reported `outcome=passed`. | Re-run in release CI and retain the complete terminal log for the exact publish commit. |
| 2026-07-15 | Compiler verified-layout binding preparation | `pnpm --filter @unlocalhosted/browsergrad-compiler typecheck && pnpm --filter @unlocalhosted/browsergrad-compiler test && pnpm --filter @unlocalhosted/browsergrad-compiler lint && pnpm --filter @unlocalhosted/browsergrad-compiler build && pnpm architecture:check && pnpm test:release-packages` | Passed: strict typecheck, 29 files/951 tests including 6 focused preparation/cache-identity tests, lint/build, architecture guard, exact packed dependency assertions, extracted-tarball execution, and fresh three-tarball bare-import runtime plus declaration typecheck. | Lower prepared bindings into read-only compiler memory references next. |
| 2026-07-15 | Compiler read-only verified-view lowering | `pnpm --filter @unlocalhosted/browsergrad-compiler typecheck && pnpm --filter @unlocalhosted/browsergrad-compiler test && pnpm --filter @unlocalhosted/browsergrad-compiler lint && pnpm --filter @unlocalhosted/browsergrad-compiler build && pnpm architecture:check && pnpm test:release-packages && git diff --check` | Passed: strict typecheck, 29 files/963 tests including 18 focused binding/lowering/authority/runtime-admission tests, lint/build, architecture guard, extracted-tarball execution, fresh three-tarball bare-import runtime/declaration proof, and whitespace check. | Add required actual-WebGPU compiler conformance using these same fixtures and buffers. |
| 2026-07-15 | Shared required-WebGPU evidence contract | `pnpm --filter @unlocalhosted/browsergrad-kernels typecheck && pnpm --filter @unlocalhosted/browsergrad-kernels typecheck:browser:view-copy && pnpm --filter @unlocalhosted/browsergrad-kernels test && BG_BROWSER_HEADLESS=1 pnpm --filter @unlocalhosted/browsergrad-kernels test:browser:view-copy && pnpm architecture:check && git diff --check` | Passed: both typechecks, 14 files/89 unit tests including 3 generic terminal-contract mutations, architecture guard, whitespace check, and advisory browser lane. Headless Chromium emitted the same one validated `not-run` terminal record because no adapter was available; artifact and case-set hashes were unchanged. | Reuse this contract in compiler required-device evidence; the shared module remains test-only. |
| 2026-07-15 | Compiler layout browser typecheck/unit/release guard | `pnpm --filter @unlocalhosted/browsergrad-compiler typecheck && pnpm --filter @unlocalhosted/browsergrad-compiler test && pnpm --filter @unlocalhosted/browsergrad-compiler lint && pnpm --filter @unlocalhosted/browsergrad-compiler build && pnpm architecture:check && pnpm test:release-packages && git diff --check` | Passed: strict browser-inclusive typecheck, 30 files/965 tests including exact-SHA/dirty/GitHub-SHA guard mutations, lint/build, architecture, packed/fresh consumers, and whitespace. | None for this slice. |
| 2026-07-15 | Advisory compiler layout browser lane | `BG_BROWSER_HEADLESS=1 pnpm --filter @unlocalhosted/browsergrad-compiler test:browser:layout-bindings` | Passed command with one skipped/not-run test and one validated terminal record because headless Chromium returned no adapter. Prepared artifact and case-set hashes match required/headed runs. | Advisory result is not conformance evidence. |
| 2026-07-15 | Headless required compiler layout browser lane | `BG_BROWSER_HEADLESS=1 pnpm --filter @unlocalhosted/browsergrad-compiler test:browser:layout-bindings:required` | Failed as designed: one validated required terminal record reported `outcome=failed` and `BG-WEBGPU-EVIDENCE-DEVICE-UNAVAILABLE`; Vitest exited 1. | Preserve as truthful environment evidence; absence cannot authorize release. |
| 2026-07-15 | Headed required actual-device compiler layout lane | `pnpm --filter @unlocalhosted/browsergrad-compiler test:browser:layout-bindings:required` | Passed: headed Chromium acquired Apple Metal 3/WebGPU core; all six ordered cases matched semantic-core physical traces, CPU reference, and complete source/output finite-f32 bits; each prepared plan was one single-dispatch pipeline/workgroup; queue, late-error, device-loss, and uncaptured-error checks were clean; one validated terminal pass was emitted. | Required release CI must repeat on the exact publish commit and retain its full log. |
| 2026-07-15 | Canonical view-copy construction | `pnpm --filter @unlocalhosted/browsergrad-semantic-core typecheck && pnpm --filter @unlocalhosted/browsergrad-semantic-core test && pnpm --filter @unlocalhosted/browsergrad-semantic-core lint && pnpm --filter @unlocalhosted/browsergrad-semantic-core build` | Passed: strict typecheck, 10 files/95 tests, lint, and build. Tests include pinned artifact hashes, structured hostile-input failures, rank/node/depth budgets, and CPU execution. | None. |
| 2026-07-15 | Constructor/compiler packed integration | `pnpm --filter @unlocalhosted/browsergrad-compiler typecheck && pnpm --filter @unlocalhosted/browsergrad-compiler test && pnpm --filter @unlocalhosted/browsergrad-compiler lint && pnpm --filter @unlocalhosted/browsergrad-compiler build && pnpm architecture:check && pnpm test:release-packages && git diff --check` | Passed: strict typecheck, 30 files/966 tests, lint/build, architecture guard, packed constructor exports/execution, and fresh three-tarball bare-import runtime/declaration proof. | JIT adapter is next; exact actual-device lanes still rerun at release commit. |
| 2026-07-15 | Compiler `0.2.0` release alignment | `node -e 'JSON.parse(...)' && ruby -e 'require "yaml"; YAML.load_file(...)' && pnpm architecture:check && pnpm test:release-packages && git diff --check` | Passed: package JSON and release-workflow YAML parse, architecture guard, compiler `0.2.0` tarball with exact semantic-core/kernels `0.2.0` ranges, workflow dependency-prerequisite assertions, and fresh-consumer runtime/declaration proof. | No publish was performed; exact-commit actual-device evidence is still mandatory before tags. |
| 2026-07-15 | Typed JIT permutation request emission | `pnpm --filter @unlocalhosted/browsergrad-jit exec vitest run --config vitest.integration.config.ts tests-integration/gpu_plan.test.ts tests-integration/ir_construction.test.ts && pnpm --filter @unlocalhosted/browsergrad-jit typecheck && pnpm --filter @unlocalhosted/browsergrad-jit lint && git diff --check` | Passed: 2 focused integration files/31 tests, strict typecheck, lint, generated-source parity from codegen, and whitespace check. | Kernels parsing, constructor/hash proof, execution routing, and strict real-device evidence remain. |
| 2026-07-15 | Resident semantic view-copy dispatch | `pnpm --filter @unlocalhosted/browsergrad-kernels exec vitest run tests/semantic_view_copy_resident.test.ts && pnpm --filter @unlocalhosted/browsergrad-kernels typecheck && pnpm --filter @unlocalhosted/browsergrad-kernels test && pnpm --filter @unlocalhosted/browsergrad-kernels build && pnpm --filter @unlocalhosted/browsergrad-kernels lint && git diff --check` | Passed: focused 7 tests, full 15 files/96 tests, strict typecheck, build, lint, and whitespace check. No readback or upload path is invoked by the resident tests. | Integrate with strict JIT side-table preparation and tensor-plan liveness next. |
| 2026-07-15 | JIT semantic request preparation and bridge routing | `pnpm --filter @unlocalhosted/browsergrad-kernels typecheck && pnpm --filter @unlocalhosted/browsergrad-kernels exec vitest run tests/tensor_plan_semantics.test.ts tests/semantic_view_copy_resident.test.ts && pnpm --filter @unlocalhosted/browsergrad-jit typecheck && pnpm --filter @unlocalhosted/browsergrad-jit exec vitest run --config vitest.integration.config.ts tests-integration/gpu_plan.test.ts tests-integration/webgpu_realizer.test.ts && pnpm architecture:check` | Passed after the recorded VJP metadata correction: kernels 2 files/20 tests, JIT 2 files/40 tests, both strict typechecks, and architecture guard. Tests pin the canonical rank-2 layout/kernel hashes, prove `valueId` hash independence, strict hostile-envelope rejection, semantic-only PERMUTE routing, resident deferral, legacy-bridge refusal, attention, and symbolic matmul VJP. | Full package suites, packed consumers, shared cross-frontend fixture, and required actual-device JIT evidence remain. |
| 2026-07-15 | Full kernels/JIT semantic-route regression | `pnpm --filter @unlocalhosted/browsergrad-kernels test && pnpm --filter @unlocalhosted/browsergrad-kernels build && pnpm --filter @unlocalhosted/browsergrad-kernels lint && pnpm --filter @unlocalhosted/browsergrad-jit test && pnpm --filter @unlocalhosted/browsergrad-jit test:integration && pnpm --filter @unlocalhosted/browsergrad-jit build && pnpm --filter @unlocalhosted/browsergrad-jit lint && pnpm architecture:check && git diff --check` | Passed: kernels 16 files/109 tests plus build/lint; JIT 8 unit tests and 23 integration files/233 tests plus generated-source build/lint; architecture guard and whitespace check. Kernels lint retains one pre-existing `realizer.ts` no-useless-spread warning outside the semantic route. | Packed consumers, shared cross-frontend fixture, and required actual-device JIT evidence remain. |
| 2026-07-15 | Shared dense-permutation fixture | `pnpm --filter @unlocalhosted/browsergrad-semantic-core typecheck && pnpm --filter @unlocalhosted/browsergrad-semantic-core test && pnpm --filter @unlocalhosted/browsergrad-kernels typecheck && pnpm --filter @unlocalhosted/browsergrad-kernels test && pnpm --filter @unlocalhosted/browsergrad-compiler typecheck && pnpm --filter @unlocalhosted/browsergrad-compiler test && pnpm --filter @unlocalhosted/browsergrad-jit typecheck && pnpm --filter @unlocalhosted/browsergrad-jit exec vitest run --config vitest.integration.config.ts tests-integration/gpu_plan.test.ts && pnpm architecture:check && pnpm test:release-packages` | Passed: semantic-core 11 files/99 tests, kernels 16 files/111 tests, compiler 30 files/969 tests including architecture guard mutations, focused JIT 1 file/5 tests, strict typechecks, architecture path/content/order guard, packed fixture export, and fresh-consumer import. The first JIT assertion observed Pyodide converting Python `None` to JavaScript `undefined`; the evidence now carries an explicit `permuteArgErased` boolean. | Add the JIT-owned strict actual-device semantic bridge lane using these exact cases. |
| 2026-07-15 | JIT semantic-permute lane unit/type/release wiring | `pnpm --filter @unlocalhosted/browsergrad-jit typecheck && pnpm --filter @unlocalhosted/browsergrad-jit typecheck:browser:semantic-permute && pnpm --filter @unlocalhosted/browsergrad-jit test && pnpm --filter @unlocalhosted/browsergrad-kernels typecheck && pnpm --filter @unlocalhosted/browsergrad-kernels exec vitest run tests/tensor_plan_semantics.test.ts && ruby -e 'require "yaml"; ...' && pnpm test:release-packages && pnpm architecture:check && git diff --check` | Passed: both JIT typechecks, 2 files/10 JIT unit tests including exact-SHA gate mutations, 14 focused kernels tests, workflow YAML parse, packed/release assertions, architecture guard, and whitespace check. | Re-run full package suites before commit and required device lane on the exact release commit. |
| 2026-07-15 | Advisory JIT semantic-permute browser lane | `BG_BROWSER_HEADLESS=1 pnpm --filter @unlocalhosted/browsergrad-jit test:browser:semantic-permute` | Passed command with one skipped/not-run test and one validated terminal record because headless Chromium returned no adapter. Prepared artifact/case-set hashes were retained. | Advisory evidence is not conformance evidence. |
| 2026-07-15 | Headless required JIT semantic-permute browser lane | `BG_BROWSER_HEADLESS=1 pnpm --filter @unlocalhosted/browsergrad-jit test:browser:semantic-permute:required` | Failed as designed: one validated required terminal record reported `outcome=failed` and `BG-JIT-SEMANTIC-PERMUTE-DEVICE-UNAVAILABLE`; Vitest exited 1. | Preserve as truthful environment evidence; absence cannot authorize release. |
| 2026-07-15 | Pre-hardening headed JIT semantic-permute lane | `pnpm --filter @unlocalhosted/browsergrad-jit test:browser:semantic-permute:required` | Execution passed on Apple Metal 3, but adversarial review later proved the lane hand-built its plan/request, labeled planned workgroups submitted, and left request/plan/input facts outside the terminal artifact hash. | Superseded as conformance evidence by the hardened JIT-emitted run below; retained here to prevent repeating the false-positive design. |
| 2026-07-15 | Full JIT semantic-permute regression | `pnpm --filter @unlocalhosted/browsergrad-kernels typecheck && pnpm --filter @unlocalhosted/browsergrad-kernels test && pnpm --filter @unlocalhosted/browsergrad-kernels build && pnpm --filter @unlocalhosted/browsergrad-kernels lint && pnpm --filter @unlocalhosted/browsergrad-jit typecheck && pnpm --filter @unlocalhosted/browsergrad-jit typecheck:browser:semantic-permute && pnpm --filter @unlocalhosted/browsergrad-jit test && pnpm --filter @unlocalhosted/browsergrad-jit test:integration && pnpm --filter @unlocalhosted/browsergrad-jit build && pnpm --filter @unlocalhosted/browsergrad-jit lint` | Passed: kernels 16 files/111 tests plus build/typecheck/lint; JIT 2 files/10 unit tests and 23 files/234 integration tests plus both typechecks, generated-source build, and lint. Kernels lint retains one pre-existing `realizer.ts` warning outside this slice. | JIT release version/packed-consumer alignment is next. |
| 2026-07-15 | Hardened JIT producer/trace focused regression | `pnpm --filter @unlocalhosted/browsergrad-kernels exec vitest run tests/tensor_plan_semantics.test.ts tests/semantic_view_copy_resident.test.ts && pnpm --filter @unlocalhosted/browsergrad-jit typecheck && pnpm --filter @unlocalhosted/browsergrad-jit exec vitest run --config vitest.integration.config.ts tests-integration/gpu_plan.test.ts` | Passed: 2 kernels files/23 tests and exact production-capture JIT fixture 1 file/5 tests. Per-handle authority, actual dispatch profiles, release cleanup, exact emitted wire, and fixture coverage are exercised. | Run full package/release gates before commit. |
| 2026-07-15 | Hardened JIT-emitted actual-device lane | `pnpm --filter @unlocalhosted/browsergrad-jit typecheck:browser:semantic-permute && pnpm --filter @unlocalhosted/browsergrad-jit test:browser:semantic-permute:required` | Passed: Pyodide emitted deterministic `BUFFER,LOAD,PERMUTE` plans and exact request JSON; headed Chromium acquired Apple Metal 3; both complete raw-u32 outputs matched; live traces matched authority-bound manifests; actual dispatch profiles reported `[1,1,1]` workgroups at `[64,1,1]`; terminal artifact `5de13ee553a25bb45fdfe198267c64c0ac505903c8f44946c58a34ae7b5121a3`, case set `962280999544fc424e23205ec9bd40e7ec5186d32d72027c0fd0239321d556cc`, and device profile `0ef434a09b4cc9919ba30c92e791dc2a2138ef42a6f02cc282d1ce1d07d22b63` recomputed before pass. | Release CI must repeat and retain evidence for the exact publish commit. |
| 2026-07-15 | Hardened JIT semantic-permute full regression | `pnpm --filter @unlocalhosted/browsergrad-kernels typecheck && pnpm --filter @unlocalhosted/browsergrad-kernels test && pnpm --filter @unlocalhosted/browsergrad-kernels build && pnpm --filter @unlocalhosted/browsergrad-kernels lint && pnpm --filter @unlocalhosted/browsergrad-jit typecheck && pnpm --filter @unlocalhosted/browsergrad-jit typecheck:browser:semantic-permute && pnpm --filter @unlocalhosted/browsergrad-jit test && pnpm --filter @unlocalhosted/browsergrad-jit test:integration && pnpm --filter @unlocalhosted/browsergrad-jit build && pnpm --filter @unlocalhosted/browsergrad-jit lint` | Passed: kernels 16 files/112 tests plus typecheck/build/lint; JIT 2 files/10 unit tests and 23 files/234 Pyodide integration tests plus both typechecks, generated-source build, and lint. Kernels lint retains one pre-existing `realizer.ts` no-useless-spread warning. | None before staged audit. |
| 2026-07-15 | Hardened JIT release/architecture gates | `ruby -e 'require "yaml"; ...' && pnpm test:release-packages && pnpm architecture:check && git diff --check` | Passed: workflow YAML parse, packed/fresh-consumer release gate, production-capture/trace wiring assertions, semantic architecture guard, and whitespace check. | Audit staged ownership and commit this slice. |
| 2026-07-15 | Terminal evidence mutation and type gates | `pnpm --filter @unlocalhosted/browsergrad-jit typecheck && pnpm --filter @unlocalhosted/browsergrad-jit typecheck:browser:semantic-permute && pnpm --filter @unlocalhosted/browsergrad-jit exec vitest run tests/semantic_permute_evidence.test.ts tests/semantic_permute_publish_gate.test.ts` | Passed: both strict typechecks and 2 files/9 tests. Seven evidence tests cover digest exclusion/tampering, exact producer versions, manifest/observation drift, advisory/pre-device states, environment contradictions, queue-drain state, and uncaptured errors. | None. |
| 2026-07-15 | Whole-record JIT actual-device lane | `pnpm --filter @unlocalhosted/browsergrad-jit test:browser:semantic-permute:required` | Latest implementation run passed on Apple Metal 3: deterministic artifact/case/device hashes matched, both raw-u32 cases passed with actual queue-completion profiles, and terminal manifest `7dd44831892ca92595522f6d1428c0dbb46853ab19f9d561f3a4918c99b827ac` bound the complete successful record. | Relevant source was intentionally uncommitted during implementation, so the retained verifier rejects this log as release authorization. Release CI must generate and retain a new terminal record on the exact clean publish commit. |
| 2026-07-15 | Final JIT terminal-evidence regression | `pnpm --filter @unlocalhosted/browsergrad-jit test && pnpm --filter @unlocalhosted/browsergrad-jit test:integration && pnpm --filter @unlocalhosted/browsergrad-jit build && pnpm --filter @unlocalhosted/browsergrad-jit lint && ruby -e 'require "yaml"; ...' && pnpm test:release-packages && pnpm architecture:check && git diff --check` | Passed: JIT 3 files/17 unit tests, 23 files/234 Pyodide integration tests, deterministic codegen/build/lint, workflow syntax, release static/packed gates, architecture guard, and whitespace check. | Commit the coherent JIT-emitted evidence slice. |
| 2026-07-15 | Production GPU scope and recovery hardening | `pnpm --filter @unlocalhosted/browsergrad-kernels exec vitest run tests/webgpu_error_scope.test.ts tests/semantic_view_copy_resident.test.ts tests/tensor_plan_semantics.test.ts tests/publish_gate.test.ts && pnpm --filter @unlocalhosted/browsergrad-kernels typecheck` | Passed: 4 files/40 tests plus strict typecheck. Coverage includes immediate LIFO pop initiation, validation/OOM/internal/pop/completion/sync-throw classification, delayed device loss, materialization failures, no pre-scope handle mint, cache/pool invalidation, poisoned-root destruction, clean retry, private issuer surface, and exact dual evidence markers. | None. |
| 2026-07-15 | Latest actual-device implementation run and dirty-source refusal | `pnpm --filter @unlocalhosted/browsergrad-jit test:browser:semantic-permute:required` then retained-log verifier | Browser execution passed both raw-u32 cases on Apple Metal 3 with terminal manifest `7dd44831892ca92595522f6d1428c0dbb46853ab19f9d561f3a4918c99b827ac`. The verifier then rejected release authorization with `relevant source differs from expected git HEAD`, as required for the uncommitted implementation tree. | After commit, rerun required lane and verifier without modifying the evidenced source. |
| 2026-07-15 | Final scoped Gate 2 package regression | Kernels full typecheck/browser-typecheck/test/build/lint; JIT full typecheck/browser-typecheck/unit/integration/build/lint; compiler `verify:compiler` | Passed: kernels 17 files/127 tests; JIT 4 files/24 unit tests and 23 files/234 Pyodide integration tests; compiler 30 files/969 tests plus architecture and all synthetic/fixture/scope/status/CLI/tool-lock/corpus gates. Builds and lints passed; kernels retains one pre-existing `realizer.ts` no-useless-spread warning. | None before release/architecture gates. |
| 2026-07-15 | Final retained-evidence/release/architecture gates | workflow YAML parse; `pnpm test:release-packages`; `pnpm architecture:check`; `git diff --check`; staged diff check | Passed: workflow syntax, unique/order/condition/pipefail/no-bypass assertions, gate-last assertions for JIT/kernels/compiler, packed/fresh-consumer checks, semantic architecture guard, and whitespace checks. | Stage only owned files, audit, and commit this coherent slice. |
| 2026-07-15 | Exact-commit JIT retained evidence after `09d868d0` | `pnpm --filter @unlocalhosted/browsergrad-jit test:browser:semantic-permute:required 2>&1 \| tee /tmp/browsergrad-jit-semantic-permute-09d868d0.log` then `node packages/browsergrad-jit/scripts/verify_semantic_permute_evidence_log.mjs /tmp/browsergrad-jit-semantic-permute-09d868d0.log "$(git rev-parse HEAD)"` | Passed on Apple Metal 3. Verifier accepted exact source revision `09d868d077e02b8f8727d9b65923d19969650761` with artifact `5de13e...`, case set `962280...`, device `0ef434...`, and terminal manifest `1d2055...`. | Historical proof remains valid for that commit; rerun after `0.9.0` release-alignment commit. |
| 2026-07-15 | JIT `0.9.0` release alignment | `pnpm install --frozen-lockfile`; JIT build/typecheck/browser-typecheck/unit/integration/lint; `pnpm test:release-packages`; registry-equivalence `--allow-missing`; `pnpm architecture:check`; workflow YAML parse; `git diff --check` | Passed: frozen lockfile; generated-source build; both typechecks; 4 files/24 unit tests; 23 files/234 Pyodide integration tests; lint; six publish-order tests plus packed-tree mutation test; packed JIT-only and integrated runtime/declaration consumers; workflow registry/source/order/resume assertions; architecture and YAML checks. Live npm query truthfully reported semantic-core `0.2.0` and kernels `0.2.0` are not published, so no equivalence or publication claim is made. | Commit owned files, rerun exact-clean-commit required-device lane, then begin compiler structured padding. |
| 2026-07-15 | Immutable release hardening focused gate | `python3 scripts/snapshot-package-tar.test.py`; provenance/order/tree Node tests; syntax checks; `node scripts/publish-missing-npm.mjs --dry-run` | Passed: 12 bounded hostile-archive cases, 17 focused Node tests, all script syntax, and deterministic five-package missing-version plan in semantic-core → kernels → compiler/Grad/JIT order. No publication occurred. | Run exact staging from a clean committed worktree. |
| 2026-07-15 | Immutable release hardening integration gate | `pnpm test:release-packages`; `pnpm install --frozen-lockfile`; `pnpm architecture:check`; Ruby workflow YAML parse; `git diff --check` | Passed: complete packed consumers including fresh npm Grad and JIT installs, lifecycle/repository/workflow guards, 10 provenance tests, 12 tar tests, frozen pnpm `10.34.5` install, semantic architecture, workflow syntax, and whitespace. | Commit release infrastructure, then stage semantic-core from a separate clean worktree before marking the pipeline verified. |
| 2026-07-15 | Adversarial immutable-release exit gate | `pnpm install --frozen-lockfile`; `pnpm test:release-packages`; `pnpm architecture:check`; `node scripts/publish-missing-npm.mjs --dry-run`; Ruby workflow YAML parse; script syntax/compile checks; `git diff --check` | Passed: frozen pnpm `10.34.5`; 18 raw/semantic hostile-tar tests; 25 Node tests including 4 staged-manifest, 12 provenance, 6 dependency-order, 2 credential-boundary, and 1 tree-equivalence cases; all seven public tarballs and fresh runtime/primitives/Grad/JIT consumers; architecture and workflow syntax. Registry plan contains exactly the five missing versions in semantic-core → kernels → compiler/Grad/JIT order. No publication occurred. | Commit only owned release infrastructure, then stage one exact package from a separate clean worktree and record its manifest/SRI before marking this row verified. |
| 2026-07-15 | Immutable-release exit-audit closure | `pnpm install --frozen-lockfile && pnpm test:release-packages && pnpm architecture:check`; workflow YAML parse; `node scripts/publish-missing-npm.mjs --dry-run`; `git diff --check` | Passed: frozen pnpm `10.34.5`; 19 hostile tar tests; 35 Node release-security tests; all-seven packed/fresh-consumer suite; architecture and workflow syntax. Live dry-run plans all seven current versions in dependency-first order as missing after the required runtime `0.1.2` and primitives `0.1.1` bumps. No publication occurred. | Commit owned files, then stage an exact package from a separate clean worktree and record manifest/SRI evidence. |
| 2026-07-15 | Exact clean-commit staged-release proof | Detached worktree at `155161b7`; `pnpm install --frozen-lockfile`; `pnpm -r run build`; `node scripts/publish-missing-npm.mjs --stage-dir npm-release-artifacts --package @unlocalhosted/browsergrad-semantic-core`; packed identity inspection | Passed: manifest source revision `155161b7b36f691ca9935045af48488fffbab265`; exact target/closure semantic-core `0.2.0`; 124 files; tree `47fe98534b683c853d7d872189fdf2ba598394ab03b6a7cf64f88049fc7b2de1`; SRI `sha512-Eegpt07BC+R7/pJZeMpRzMK5P6pIHD43m2z/WNWUUplHXFKXRbW4ObMouWm4KKByTfvIOgYrG2elIf2HjCNgoA==`; worktree status contained only the two declared staging outputs. No publication occurred. | Release pipeline row is verified. Reconcile stale overlapping release instructions, then implement compiler structured padding. |
| 2026-07-16 | Compiler L2 focused behavior | `pnpm --filter @unlocalhosted/browsergrad-compiler exec vitest run tests/compiler/semantic_view_copy_bindings.test.ts tests/compiler/semantic_index_map_lowering.test.ts` | Passed: 2 files, 13 tests covering authority, raw-u32 structured padding, exact fill, offsets/canaries, reject policy, always-false zero reads, hostile runtime roots, and arithmetic intervals. | Add actual-WebGPU proof. |
| 2026-07-16 | Compiler L2 regression scope | `pnpm --filter @unlocalhosted/browsergrad-compiler exec vitest run tests/compiler/semantic_view_copy_bindings.test.ts tests/compiler/semantic_layout_bindings.test.ts tests/compiler/core.test.ts tests/compiler/control.test.ts` | Passed after localizing signed predicates: 4 files, 235 tests. | Re-run after final adversarial edits and as part of full compiler gate. |
| 2026-07-16 | Compiler L2 types and lint | `pnpm --filter @unlocalhosted/browsergrad-compiler typecheck && pnpm --filter @unlocalhosted/browsergrad-compiler lint` | Passed before final SharedArrayBuffer/return adversarial edits. | Re-run before commit; this row does not authorize the final diff. |
| 2026-07-16 | Compiler L2 full package gate | `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler` | Passed on the final unit-slice diff: repository architecture; kernels/compiler builds; compiler typecheck/lint; 32 files and 982 tests; compiler dependency/cycle/representation architecture; synthetic input, source normalizer, WebGPU fixtures, test scope, bugbash status, real-world CLI, tool lock, and corpus-audit harnesses. | Commit the CPU/WGSL slice; actual-WebGPU L2 evidence remains separate. |
| 2026-07-16 | Compiler L2 post-commit adversarial hardening | Focused view-copy/index-map Vitest plus compiler typecheck and whitespace check | Passed: 2 files, 14 tests; strict typecheck; clean diff whitespace. | Commit test hardening separately, then build required-device lane. |

## Failure and Recovery Log

Record failures that may matter after context loss. Include the exact failing
command, concise error, suspected cause, resolution or next experiment, and
whether any files may be left partially changed.

- Initial JIT packed-consumer typecheck failed because an empty observation
  array inferred implicit `any[]`; replaced it with a structurally typed scalar
  observation record shared by the runtime and strict declaration check.
- First integrated packed-consumer runtime/typecheck used string literals where
  `WireI64` brands were required and read `operationId` one level above its
  prepared specialization owner. It now imports `parseWireI64` from the packed
  schema subpath and checks `prepared.semantic.operation.operationId`; both
  runtime and strict declaration checks pass.
- Node-based workflow parsing failed because root has no direct `yaml` module.
  Repository validation uses Ruby's installed YAML parser instead; workflow
  static semantics remain covered by the release harness.
- Registry-equivalence hardening first passed unsupported
  `pnpm pack --ignore-scripts`; pnpm rejected the option. The verified form is
  `--config.ignore-scripts=true`, which preserves pnpm workspace-range rewriting
  without allowing package lifecycle scripts to mutate evidence inputs.
- Adversarial staged review invalidated the first JIT device proof: the browser
  test synthesized plan/request data instead of invoking JIT, called a planned
  ceil-division submitted topology, prepared evidence separately from the
  object actually dispatched, and kept request/plan/input hashes outside the
  terminal artifact identity. The lane now captures production Pyodide
  emission, obtains preparation/profile evidence from the live bridge handle,
  separates planned/submitted fields, and recomputes a complete manifest hash.
- Final staged review found the deterministic artifact hash did not bind the
  terminal outcome/environment/state, producer versions could contradict the
  hashes, failure/not-run states were open, queue-drain used a stale stage, and
  the validator had no mutation corpus. The extracted finalizer now preserves
  deterministic artifact identity while hashing the whole terminal record and
  enforcing a closed state/environment model under adversarial resealing.
- The first JIT `prepublishOnly` ordering checked exact-commit evidence before
  mutating codegen. A clean commit with stale generated Python could therefore
  pass the marker and then publish regenerated, unevidenced bytes. Build/codegen
  now runs first and the release test pins the gate last, so drift blocks publish.
- The padding-slice audit found the compiler browser record still names a
  plan-derived ceil-division `submittedWorkgroupCount`; it does not observe the
  `dispatchWorkgroups` call. Existing output/device proof remains valid, but
  topology wording is overstated. The padded-L2 evidence upgrade must rename it
  planned or capture an immutable actual dispatch trace before claiming
  submitted topology.
- The first signed-padding WGSL fix changed global comparison operand inference
  and caused 18 existing core/control expectation failures. L2 predicate
  operands now receive explicit same-type casts in the index-map adapter, using
  the established compiler rule while leaving global WGSL inference unchanged;
  the 235-test focused regression scope then passed.
- The first L2 regression run also found the new runtime-buffer diagnostic was
  absent from the compatibility registry. All L2 preparation, source, range,
  runtime, and authority diagnostics now have explicit registry entries.
- The first always-false test used a zero-byte source allocation, outside the
  compiler's positive whole-root address profile. It now uses a one-word canary
  root with a zero-extent logical source and proves no source read occurs; no
  backend behavior was widened to make an invalid test input pass.
- GPU evidence review found `uncapturederror` plus queue/task drainage could
  miss or throttle late WebGPU errors, and public/direct resident paths could
  bypass diagnostic ownership. Production issue-site scopes, a private sync
  capability, safe async public wrapper, device-loss race, and failure cleanup
  now make scope settlement authoritative before handles/results escape.
- The first scoped nonresident reuse returned a failed materialization root to
  the output pool in `finally`. The failure path now destroys the root, clears
  cache/pools, settles profiles, and rethrows; only successful materialization
  re-pools. A failed-map then clean-retry fake freezes this invariant.
- Retained-evidence review found the first verifier read files before applying
  its cap, accepted dirty source labeled only by HEAD, and did not pin workflow
  condition parity. It now rejects non-regular/oversized files before bounded
  reading, shares a canonical clean source scope with final publish guards, and
  statically requires identical JIT/kernels conditions, pipefail, no bypass,
  exact order, and always-upload behavior.
- Dispatch-profile review found unavailable or cross-paired timing could enter
  `completedCases`; terminal validation would then reject both the pass and the
  attempted failure record. Execution now admits only
  `timestamp-query/exact` or `queue-completion/coarse` before case completion,
  with matching mutation coverage.
- The first production-capture advisory run failed before device acquisition:
  the strict profile expected `BUFFER,PERMUTE`, while actual JIT correctly emits
  `BUFFER,LOAD,PERMUTE`. Routing validation now admits only that exact three-step
  projection, binds the bridge input to `BUFFER`, correlates `PERMUTE` to
  `LOAD`, and still requires one executable dispatch. Advisory and headed
  required reruns pass.

- The first JIT browser preparation failed before device acquisition with
  `BG-SCHEMA-NONCANONICAL-VALUE` because the evidence plan reused the same
  frozen shape array in its step and buffer projections. Numerically equal
  references are still an object graph, not canonical JSON. The plan builder
  now snapshots each occurrence into an independent array; advisory and
  required preparation produce stable plan/artifact hashes.

- The first constructor test run failed eight cases with
  `BG-SCHEMA-NONCANONICAL-VALUE` because permutation substitution reused one
  coordinate object in location and predicate branches. The semantics were
  numerically correct but the result was an object graph, not canonical JSON.
  Substitution now emits a fresh expression tree per occurrence and a direct
  canonicalization regression guards the invariant; all 95 tests pass.
- The first packed constructor consumer failed with
  `BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-PREDICATE`: normalized layouts
  correctly make logical bounds explicit, while the compiler adapter accepted
  only literal `true`. The adapter now interval-proves a predicate is true over
  the complete resolved logical domain before erasing it; the existing
  conditional-predicate rejection still passes, as do all 966 compiler tests
  and the packed fresh-consumer gate.
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
- The first view-copy CPU draft passed positive cases but principal/adversarial
  review found destination self-overlap, SharedArrayBuffer aliasing, spoofable
  typed-array properties and methods, missing dynamic offset/binding alignment
  checks, per-element AST rebuilding, multiplicative work and memory exposure,
  generic/profile coupling, missing operation versioning, ambiguous operation
  ordering, and dynamic specialization-key collision. The final slice moves
  backend limits into a shared profile, compiles evaluators, preflights and
  caches bounded offsets, requires a dense destination, validates native slots,
  versions the single operation, and hashes resolved specialization facts.
- Extending the default kernels TypeScript project to all historical browser
  tests exposed unrelated pre-existing unused-variable and tuple-narrowing
  errors. The slice now has a focused `tsconfig.browser-view-copy.json` that
  typechecks all production sources plus the new strict evidence harness; the
  existing browser suite was not silently relabeled clean.
- The first fresh-consumer install attempted two tarballs with `pnpm add
  --offline`, but pnpm still resolved kernels' packed exact semantic-core
  dependency through registry metadata and failed with
  `ERR_PNPM_NO_OFFLINE_META`. The harness now declares both tarballs as file
  dependencies and uses a temporary override for semantic-core. Packed
  metadata is separately asserted exact and free of `workspace:` ranges.
- Adding the compiler tarball to that consumer first failed with
  `ERR_PNPM_NO_MATCHING_VERSION` because compiler's exact packed kernels
  dependency correctly named unpublished local `0.2.0`, while the temporary
  consumer overrode only semantic-core. The harness now overrides both exact
  transitive package names to their packed tarballs; packed dependency metadata
  remains separately asserted and unchanged.
- Headless Chromium 148 returned no WebGPU adapter despite the existing
  SwiftShader flags. Advisory mode recorded a not-run. Required mode emitted
  auditable failure evidence and exited nonzero, so absence cannot become a
  false green. Headed Chromium then exposed Apple Metal 3 and the same required
  lane passed all nine cases; headless and headed outcomes remain separate
  environment records rather than overwriting one another.
- Adversarial review of the first WGSL runner found mutable prepared programs,
  partial per-case pass lines, missing canonical evidence fields, no aggregate
  working-set/in-flight bound, shader/pipeline diagnostic collapse, error
  scopes spanning readback, cache reuse after device loss, i32-min literal
  overflow, and buffer-slot TOCTOU risk. The runner now uses branded/deeply
  frozen plans, full digests, one terminal schema-validated record, explicit
  owned-memory and in-flight bounds, synchronous upload before the first await,
  two scoped creation/submission phases with all LIFO pops initiated before
  readback, stable staged diagnostics, device-loss watchers, and deterministic
  fake-device regressions for the lifecycle/error branches.
- The first packed-consumer declaration check tried a nonexistent root
  `node_modules/typescript/bin/tsc`, then `pnpm exec tsc` where TypeScript was
  not a root dependency; both failed before checking consumer types. It now
  invokes kernels' declared TypeScript binary. The next run correctly found
  the temporary consumer missing `type: module`; adding it made bare-import
  runtime execution and strict NodeNext typecheck pass.
- The first tarball assertion expected pnpm to retain `prepublishOnly` in the
  packed `package.json`; pnpm intentionally strips publish lifecycle scripts.
  The gate remains enforced and release-tested in the workspace package before
  packing; its internal verifier is intentionally excluded from the published
  artifact. The corrected release-package run passes.
- The first abort-cleanup regression waited for six error-scope pops, but scopes
  intentionally drain before readback and therefore did not prove active-slot
  cleanup. The fake now exposes readback completion; the test waits for owned
  cleanup before proving the next run is admitted. The final package run,
  including the exact-commit publish guard, passes 13 files/86 tests.
- The first compiler differential run exposed that the legacy CPU reference's
  generic `uint` division expression produced a fractional JavaScript number,
  while WGSL integer division truncates. Generated logical unflattening now
  wraps division in the existing explicit `uint` cast, restoring CPU/WGSL
  parity locally without changing the frozen reference semantics for ordinary
  compiler programs.
- Adversarial compiler review found that assignment/update-only mutation
  tracking missed helper address escapes, surface output targets, inline-asm
  outputs, and other operation-level writes; runtime accepted undersized root
  buffers; public compiled proof properties were structurally forgeable; and
  the GPU pipeline name omitted layout identity. The final lowering rejects
  all such index definitions/escapes, validates typed/resident root extents
  through private compiled authority, freezes the public wrapper, and puts both
  full hashes in program identity. Focused regressions cover helper and inline-
  asm mutation, forged wrappers, wrong/short typed arrays, and short residents.
- The first full compiler rerun after runtime admission failed its emitted-
  diagnostic registry check because the new runtime-buffer code was not yet
  registered. All compile-facing layout and runtime authority/buffer codes are
  now explicit compatibility records; the subsequent 29-file/963-test run
  passed.
- The first compiler publish-gate typecheck lacked a declaration for its `.mjs`
  validator, then exposed an `exactOptionalPropertyTypes` mismatch for explicit
  `undefined` test inputs. A colocated `.d.mts` now declares the exact optional
  inputs; strict typecheck and both marker tests pass.
- The first successful headed compiler evidence record copied the plan's
  logical dispatch count into a field named submitted workgroups. The runner
  actually ceil-divides logical counts by workgroup size. The observation now
  records both facts separately (`[8,1,1]` logical and `[1,1,1]` submitted),
  and a second headed required run passed with the corrected terminal record.
- Required headless compiler evidence produced Vitest screenshots and
  attachments by design. Compiler-local generated evidence artifacts are now
  ignored just like kernels screenshots; no failure artifact is staged as
  source.
- Release-path audit found that the first resumable publisher interleaved
  equivalence checks with mutations, published live directories, extracted
  untrusted registry tarballs, exposed the fallback token job-wide, omitted
  Grad from hard-coded dependency checks, and accepted tree-equivalent versions
  without provenance. Publication now precomputes a generic closure, stages all
  exact artifacts before mutation, uses a bounded non-extracting snapshotter,
  separates authority into a protected job, and applies identical integrity,
  signature, provenance, and identity checks to new and resumed versions.
- The first provenance integration call used the asynchronous verifier inside
  a synchronous retry helper, which would have logged success before npm audit
  completed. The publisher now uses top-level await plus a dedicated bounded
  async retry path; focused tests exercise cleanup and proof failures.
- An initial local experiment combined CommonJS `require` with top-level await
  and Node rejected the ambiguous module format. The corrected inspection used
  explicit ESM imports; no source file was affected.
- `npm audit signatures` against a package-lock-only install reported no
  installed supported-registry dependencies. Provenance verification now does
  a fresh real install with scripts and optional peers omitted, then audits the
  installed exact spec in an isolated cache/project.
- Exit audit found that a separately fetched DSSE statement could be validated
  while npm cryptographically audited a different installed bundle. The
  verifier now requires npm `>=11.12.0`, requests attestation-inclusive audit
  output, and decodes only the exact verified root bundle.
- The first all-target dry-run correctly failed on immutable primitives
  `0.1.0`, then runtime `0.1.1` drift. Primitives differed because the new pnpm
  artifact contract omits the packed `prepublishOnly` field; runtime also ships
  substantial new assignment/platform files. Both are new versions (`0.1.1`
  and `0.1.2`) rather than equivalence exceptions.
- The first package-identity extension referenced `relative_path` before it was
  assigned in the tar stream loop. Direct Python compilation/tests caught it
  before integration; the canonical path is now derived before payload capture,
  with invalid/missing package-manifest and symlink-input regressions.

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

Reconcile the three stale direct-publish instructions without losing their
owners' overlapping dirty edits. Then implement compiler padded rank-2/rank-3
reads only through the existing structured L2 guard/fill contract. Re-run
retained strict lanes for the exact release commit; current local evidence does
not prove a future publish.
