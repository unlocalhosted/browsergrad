# Package Requirements Implementation Ledger

- **Normative source:**
  [`docs/platform/package-requirements-lld.md`](../platform/package-requirements-lld.md)
- **Ledger status:** active
- **Last updated:** 2026-07-17
- **Current implementation slice:** Gate 3 browser-WASM asset conformance and Worker/VFS protocol

**Runtime boundary:** the portable product runs the Clang/extractor as Wasm in
a browser Worker, then executes verified semantics through CPU or WebGPU. It
does not run user C++ as Wasm. Docker is build-time/optional parity machinery
only and is not on the Gate 3 browser-local critical path.

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
| Gate 2 — multi-frontend, multi-backend view slice | `verified` | Semantic-core `0.2.0` owns verified view-copy meaning, shared specialization, and the sole canonical frontend construction sink. Kernels `0.2.0` passes the full nine-case CPU/actual-WebGPU bit-exact matrix. Compiler L1 preserves its six-case non-padded contract; sibling L2 binds the exact verified operation and proves rank-2/rank-3 guarded padding through CPU, structured IR, WGSL, and actual WebGPU. JIT `0.9.0` emits a closed typed permutation request and executes the same canonical operation through materializing/resident production routes without recovering semantic args from the frozen plan. All strict lanes passed from one clean detached worktree at exact source revision `aa605421410e9d4190d8939c24b1057731111231`. | None for the initial Gate 2 profile. Release CI must repeat the exact-source lanes before publication; registry publication is a separate release operation. | Exact clean-source Apple Metal 3 records: compiler L2 3 cases, compiler L1 6 cases, kernels 9 cases, and JIT 2 cases; compiler `verify:compiler` passed 35 files/1004 tests; release package gate passed 19 hostile-archive and 35 Node security tests plus packed/fresh consumers; combined publish guards accepted only exact L1/L2/JIT/kernels markers. |
| Gate 3 — real C++/CuTe frontend slice | `in-progress` | Browser-local Clang-WASM is the primary portable-product producer. The canonical runtime ABI, asset/build binding, and bounded raw-Wasm inspector are closed design authorities. A Worker-owned lazy VFS session now composes source snapshots and verified packs through the exact six-import ABI with dedicated path, handle, call, live-open, index-node, and logical-index-byte ceilings. Pure Worker result validation is caller-frame consistency only and cannot claim execution, termination, timing, or lowering authority. Profile 2.4 names logical reservations accurately and keeps actual Wasm residency separate. Current empty first-build projections still prevent production Wasm conformance. Existing semantic/lowering seams remain unchanged; Docker/native stays optional build/CI/reference infrastructure outside the portable graph. | Build the real extractor and reviewed header packs; independently repin complete first-build Wasm projections; implement the package-owned Worker module plus host controller/evidence/authorization; emit a canonical artifact from unmodified source in-browser; then prove semantic convergence, dynamic tensor copy, and required real WebGPU execution. | Current Worker/VFS/ABI/profile boundary: 9 files/167 tests plus strict typecheck and focused lint; full compiler 57 files/1,286 tests. Zero real Clang-Wasm builds, Worker spawns, or browser-local C++ executions. |
| Gate 4 — tiled GEMM and schedule separation | `not-started` | No implementation in this workstream. | All Gate 4 exit criteria. | None. |
| Gate 5 — tiled attention flagship | `not-started` | No implementation in this workstream. | All Gate 5 exit criteria. | None. |
| Gate 6 — framework convergence | `not-started` | No implementation in this workstream. | All Gate 6 exit criteria. | None. |
| Gate 7 — host graphs and optional systems expansion | `not-started` | No implementation in this workstream. | All Gate 7 exit criteria. | None. |

## Active Slice

### Objective

Gates 0 through 2 are verified. Gate 2 has one shared materializing view-copy
contract, CPU evaluator, canonical WGSL lowering, and compiler/JIT tracer
bullets through actual WebGPU, all re-proved at one exact clean revision. Gate
3 now audits and implements the browser-local C++/CuTe frontend boundary. Its
primary producer is a pinned CUDA-capable Clang frontend running as WASM in a
dedicated browser worker. Its first tracer must reuse the verified artifact,
layout, and backend seams rather than add source-shaped execution or
spelling-specific lowering paths. Docker/native AOT remains an optional
CI/reference parity lane and is not a browser requirement.

### Work in flight

Rows below that retain `AOT`, `OCI`, or `Docker` in their names record completed
protocol and synthetic-adapter work for the optional native parity lane. Their
`verified` status applies only to the named contract; it does not count as
browser-local producer evidence or make Docker a product dependency.

| Work item | Status | Notes | Remaining | Evidence |
|---|---|---|---|---|
| Frozen-adapter inventory | `verified` | Baselines are machine-enforced for compiler pointer/scalar memory, `cute_static_layout`, the 44-op shape/f32 `TensorGpuPlan`, 36 JIT constructors/39 real opaque labels and their decisions, runtime requirement mapping, and 17 Grad dtype/view/materialization behaviors. |
| Semantic-core seam audit | `verified` | Existing compiler/JIT/kernels types adapt into the core through explicit public subpaths; none moved wholesale. The package split and dependency direction are architecture-guarded. |
| Test-topology analysis | `verified` | TypeScript + Vitest; no specialized catalog match, so native focused Vitest route is recorded locally. |
| Gate 0 architecture check | `verified` | Cross-package boundaries, generated-source parity, all six required freezes, exact runtime mapping/status unions, reviewed vocabulary, profile-usage parity, pinned inventories/harnesses, normalized definition fingerprints, and representative mutations are implemented and wired into delivery gates. |
| Gate 1 schema/value core | `verified` | `/schema` and `/layout` only; all Gate 1 requirements and the explicit cross-language exit are covered. The Python code is a synchronized reference oracle, not a second runtime or stable public API. |
| Semantic-core package adoption gate | `verified` | `0.2.0` is public-package shaped, dependency-free, subpath-only, locally packed, and now consumed through kernels' packed exact dependency. A fresh temporary consumer installs both tarballs, resolves bare public subpaths, typechecks, and prepares matching CPU/WGSL specializations. It has not been published. |
| Gate 2 view-family selection | `verified` | Selected typed JIT `PERMUTE` plus compiler L1 read binding and sibling L2 guarded materializing view copy. The full required matrix and strict exact-source WebGPU proof now pass; broader view families must add typed operation variants rather than reinterpret frozen plans. |
| Gate 3 legacy CuTe motif freeze | `verified` | Existing transpose, GEMV, GEMM, affine-tile, flash-attention, and WGMMA/TMA source-spelling normalizers are explicitly frozen compatibility debt. Exact exception-file membership and source hashes are architecture-guarded; new motifs, replacement bodies, files, or call sites require an accepted architecture decision. | Delete these adapters after pinned resolved frontend artifacts cover their retained fixtures through shared semantics. | Architecture guard and two mutation tests. |
| Gate 3 browser-local Clang-WASM producer | `partial` | Closed browser-local profile 2.4, producer-neutral request, artifact v3, request binding, compilation contract 1.0, common lowering, explicit CUDA-pass authorities, exact ABI asset binding, raw-Wasm inspection, Worker invocation/result frames, and Worker-owned VFS session now share one closed contract. Caller frames remain non-authoritative. Current empty first-build projections make positive production conformance intentionally impossible. | Implement actual package-owned Worker/controller evidence and browser authorization; build and review the first real Wasm module and licensed packs; execute one unmodified browser-local fixture. | Focused Worker/VFS/ABI/profile 9 files/167 tests; full compiler 57 files/1,286 tests; zero Worker spawns or browser-local C++ executions. |
| Gate 3 Clang-WASM build and distribution | `partial` | The deterministic input lock now includes the exact package runtime-ABI resource as a typed deterministic distribution output and no longer reports that already-landed resource as a release blocker. Asset manifest 1.1 binds the identical ID/hash/length. A standards-aware raw inspector exists, but the first-build interface projections, extractor source, licenses, output bytes, and reproducibility evidence remain absent. | Build twice from the locked inputs; review and repin complete imports/exports/tables/globals/custom sections; resolve extractor/CUDA/sysroot file-level licensing; package the real extractor, Worker, and VFS packs; authenticate build provenance. | Build-lock/asset/ABI/inspector focused tests pass. No build process, output binary, license closure, reproducibility proof, or release authority exists. |
| Gate 3 Worker-owned VFS session | `partial` | One exact unshared Wasm memory serves canonical source snapshots and verified pack ranges through the six runtime-ABI imports. Preparation rejects cross-authority inputs, final mounted paths beyond the ABI ceiling, file/directory collisions, and profile-pinned index-node/index-byte overflow. Handles, calls, logical live-open reservations, deterministic directory order, failure atomicity, cancellation, and destructive disposal are bounded and tested. Logical reservations are not mislabeled as resident Wasm bytes or JavaScript heap evidence. | Integrate the session inside the real Worker module; connect actual allocator/page instrumentation; compare terminal observations through the host controller. | VFS pack/session focused coverage is part of the 9-file/147-test boundary; 65,536-node/32-MiB browser profile limits under 262,144-node/128-MiB ABI maxima. |
| Gate 3 Worker invocation/result protocol | `partial` | Exact profile/asset/VFS/request/ABI/raw-Wasm identities bind one single-use invocation. Module/control/artifact sizes are checked before copies; terminal state compacts heavy inputs. Result decoding enforces canonical bytes, exact call arithmetic, handle/open consistency, input/artifact/diagnostic projection, and logical live-open ceilings. Its only public result authority is explicitly `caller-frame-consistency-only`; discard proves no termination. | Build the host-owned controller that constructs the exact verified Blob Worker, owns its event source and nonce, measures wall time, accepts one terminal event, and terminates/replaces the Worker on every non-success path. Only that controller may mint execution/lowering authority. | Worker protocol 9 focused tests plus cross-boundary typecheck/lint. Positive lifecycle preparation is intentionally blocked until first-build Wasm projections are independently repinned; no test-only authority backdoor exists. |

### Optional native/AOT parity ledger

These rows are not portable-product work or Gate 3 critical-path evidence.
Their `verified` labels apply only to synthetic optional-lane contracts.

| Work item | Status | Notes | Remaining | Evidence |
|---|---|---|---|---|
| Gate 3 optional native/AOT parity producer | `partial` | Ahead-of-time profile, producer-neutral frontend request, AOT-only run metadata, structural artifact, host-derived request binding, canonical receipt resource, receipt-authenticated provenance, offline source/frame authority, local Docker observation, and completed lifecycle authority form one opaque chain. No parallel AOT job/source-intent authority remains. This lane is retained only for optional CI/reference parity, corpus qualification, and release precompilation. The Docker lifecycle is synthetic; no live native producer exists. | Optional: build a real native producer only when it directly supports cross-producer parity or CI qualification. Its absence does not block the browser-local Gate 3 exit. | 12 focused files/121 tests plus 96 Docker shell/lifecycle tests; pinned synthetic identities only; no live image, extractor, receipt, signature, or parity result. |
| Gate 3 AOT metadata over common request | `verified` | Run-metadata v1 composes AOT-only profile and a signed-but-unverified declared Git source reference around the exact producer-neutral request without duplicating source descriptors, bytes, entry selection, expected output, or conformance assertions. One consistently named `runMetadataId` binds profile, request, and reference statement. Source snapshots can originate only from request authority. Repository/revision is not called source provenance because no acquisition authority binds it to those bytes. | Real optional runner execution, source-acquisition proof, and external provenance remain unproved. | Metadata/request-binding/runner/receipt/provenance tests cover exact instance composition, detached-reference mismatch, hostile options, source copying, cross-wiring, and removal of legacy AOT job authority. |
| Gate 3 execution-environment authority | `verified` | Version 2.0 snapshots and strict-decodes one canonical resource before authority minting. Its content ID and raw digest remain distinct. The resource is instance-bound to one prepared profile and closes declared Linux/kernel/cgroup/LSM/Docker/containerd/runc/seccomp identities, ordered image layers/diff IDs, compiler/extractor/supervisor binaries, dynamic libraries, exact dependency/header ownership, include roots, and external-attestor policy. Inline rootfs, binary, dynamic-library, and header inventories have independently recomputed canonical hashes. The resource explicitly says environment-only and detached; it is not run evidence. | Real externally authenticated environment bytes and live enforcement evidence remain required. A declared seccomp hash is not proof Docker loaded that filter. | 11 execution-environment tests; canonical/hostile-byte/version/budget/cancellation/authority/profile/closure-hash matrices; full transitive plan/receipt/OCI coverage. |
| Gate 3 logical AOT execution plan | `verified` | Checked policy `1.4` defines one exact local Docker endpoint and client/engine/API/store tuple, authorized-manifest creation with pull forbidden, fixed native supervisor/hostname/argv, Docker-injected versus supervisor-cleared environment, non-root identity, no network/IPC, private remaining namespaces, read-only image rootfs plus explicit Docker-managed mounts, no added capabilities/new privileges/host-device passthrough/socket, bounded tmpfs/frame/stderr, profile-derived limits, complete lifecycle/decode budgets, fail-stop cleanup, and a dedicated single-run host trust boundary. It stages distinct request and run-metadata controls. Its seccomp request binds the exact 13,470-byte Moby snapshot and requires separate external run evidence for effective enforcement. Plan-v3 hashing re-verifies policy, run metadata, and exact prepared environment authority. | CPU/peak resource observations and actual seccomp/runc/kernel/cgroup/LSM enforcement still require live authenticated evidence; Docker create/inspect and environment declaration cannot prove them. | Checked JSON/TypeScript equality; policy `8a410d3292a165813894a37fca339d583c6781f2704070314347a5247bd0def9`; pinned upstream seccomp lock; nested immutability; 96 Docker tests; strict typecheck/lint. |
| Gate 3 offline-runner I/O authority | `verified` | Caller supplies only prepared run metadata plus exact prepared execution-environment authority; source snapshots are copied exclusively from the bound common request. The lifecycle stages canonical profile, request, run-metadata, and environment bytes plus exact source snapshots with exclusive no-follow files, read-only bind trees, readback hashes, and a closed path inventory. Output accepts one bounded framed canonical artifact+receipt only after container absence and staging removal. The host derives request-artifact selection binding from verified bytes before receipt verification. | No contract gap. Real producer output remains unproved until live execution. | Runner-plan tests plus 96 lifecycle tests; source/request identity, 0444 files, 0555 bind trees, frame limits, strict decode, abort, and copy isolation coverage. |
| Gate 3 OCI manifest/config metadata authority | `verified` | One cacheable opaque authority snapshots and strict-decodes exact raw OCI manifest/config bytes, derives their SHA-256 descriptors, requires a closed OCI leaf manifest and self-contained distributable layer descriptors, binds exact config bytes, requires `linux/amd64`, exact layer/diff-ID correspondence, bounded history/annotations, and empty image execution config. Plan authorization additionally compares every ordered layer media type/digest/size and diff ID against the exact prepared execution environment. Separate authorities observe the selected local image and verify the created container still names that closure. Layer blobs and actual execution remain outside synthetic evidence. | Exercise the exact contract against the pinned live image/environment; current adapter tests prove verifier behavior only. | OCI contract/tamper tests plus 96 Docker observation/lifecycle/strict-decode/adversarial tests. No live-daemon claim. |
| Gate 3 local Docker runtime/image observation contract | `verified` | Fixed `/usr/bin/docker` probes run in strict order `version -> info -> image-inspect` against one Unix socket with a closed environment, private empty mode-0700 config/HOME, exact Go templates, and independent ceilings. Exact Docker 29.6.1 client/engine/API/containerd store precedes image inspection. The selected manifest/config/rootfs/platform is cross-bound to plan-authorized OCI metadata. The same private session is consumed once by the lifecycle; neither public observation nor test authority can mint production completed-run authority. | Live invocation and binary/execution-environment provenance remain required. | 79 observation/process tests plus 17 lifecycle tests; timeout/abort/reaping, hostile shapes, config/HOME/seccomp mutation, issuer separation, and structural-copy rejection. |
| Gate 3 AOT runner receipt contract | `verified` | Receipt v3.0 is a closed content-addressed report binding exact run-metadata/request/request-binding/profile/environment authority, deterministic invocation, plan-v3 identity, opened input closure, exact output, and complete extraction/process accounting. It does not duplicate source files or resolved selection; those belong to request and host-derived binding authorities. Actual observed inputs, process measurements, emitted artifact counts, and configured upper bounds remain separate typed categories. `userProducedNativeExecution: forbidden` stays explicit because pinned compiler/supervisor are native. Structural and strict-byte authorities remain separate. | Real optional runner must prove actual container state implements the logical plan and emit authenticated observations. | Focused v3 receipt/provenance tests plus policy coverage; exact opaque chain, environment instance, plan/config, actual-versus-ceiling categories, legacy-field rejection, and mutation matrices. |
| Gate 3 detached provenance authorization | `verified` | Profile pins extractor, runner, resolved `linux/amd64` OCI manifest, sandbox policy, allowed builders, and exact P-256 SPKI trust store. Verification authenticates one canonical DSSE envelope, then requires one strict-decoded receipt resource. Artifact/profile/source/toolchain/sandbox/run/output facts derive from run metadata, receipt, and request binding. Common lowering authorization exposes no AOT-only run-metadata field; producer-specific lineage remains opaque. Caller artifact/profile/expected hashes cannot mint authority; rejected frontend outcomes cannot lower. The predicate is explicitly not SLSA or Sigstore. | Real producer and Sigstore-backed external evidence remain separate and are not claimed. | Focused provenance/authorization tests plus mutation matrices; exact metadata/request/binding/receipt lineage; authorized-only lowering; strict typecheck/lint. |

### Shared Gate 3 semantic seams

| Work item | Status | Notes | Remaining | Evidence |
|---|---|---|---|---|
| Gate 3 canonical artifact resource | `verified` | Artifact v3 has one authority-bound canonical byte representation with independent raw SHA-256 and wire-u64 length. Source declarations require exact identifier/declarator `identitySpanId`, closing anchor selection without relying on broad declaration ranges. Exact input ownership distinguishes source, compiler resource, and pinned dependency files; forced includes are profile-bound; initializer expressions and location ownership remain explicit. Optional AOT receipt and in-toto contracts bind the raw resource; browser worker evidence remains missing. | Real browser-produced bytes and browser authorization remain unproved. | Artifact suite 28 tests plus request-binding/AOT mutation coverage; pinned v3 semantic hash `bbb05a48...`, byte SHA-256 `02e00b21...`, and byte length `15081`. |
| Gate 3 allocation-free layout semantics | `verified` | Semantic-core snapshots, normalizes, verifies, hashes, and authority-binds one standalone layout expression as a `browsergrad.layout@1` artifact containing exactly one index map and zero allocations/views. The compiler now lowers the exact authorized CuTe layout fact through this API, preserving flat and nested static hierarchies and signed element locations. Coordinate traces expose element location plus logical/predicate bounds only; no dtype, byte range, alias, effect, tensor, or backend meaning is implied. The pinned `(3,2):(1,3)` fixture maps to `[0,3,1,4,2,5]` with semantic hash `4e1fa226...`; dynamic bindings, ordering, resource limits, hostile input, transport-neutral identity, and structural-copy rejection are covered. | Dynamic CuTe expressions remain a later compiler profile. CPU copy and WebGPU proof wait for real tensor/storage facts. | Semantic-core typecheck/lint and 12 files/107 tests; compiler authorized-layout tests and full compiler gate. |
| Gate 3 authorized static layout lowering | `verified` | The internal compiler boundary accepts one instance-bound authorized artifact plus one closed explicit entry ID. It requires exactly one accepted selected layout entry, binds it to exactly one affine fact/result declaration, preserves unrelated typed facts without interpreting them, evaluates all static integer algebra through shared bounded semantics, lowers nested modes with colexicographic coordinate composition, and stores source/macro/attestation origin in a compiler side table. It uses CuTe `cosize(layout) = layout(size(layout) - 1) + 1`, including signed strides; it does not manufacture storage. | Keep internal until a hermetic producer supplies real artifacts. Add a dynamic profile only with explicit binding authority; add tensor/copy lowering only when dtype, engine, alias, destination, and effects are genuine source facts. | 9 focused lowering tests; 14 artifact and 11 provenance tests; compiler 39 files/1053 tests; semantic/compiler architecture checks; all synthetic/fixture/corpus gates. |
| L2 materializing view-copy contract and CPU reference | `verified` | `view-copy@1.0` owns effects, exact reject/fill bits, and forbid-overlap semantics. Generic L2 verification stays backend-neutral; the shared initial profile legalizes positive-affine f32 rank-2/3 global views. Prepared CPU execution compiles maps once, proves guarded reads and dense destination writes, caches source offsets, derives binding-sensitive specialization hashes, and enforces element/step/scratch/wall-time budgets, cooperative browser yielding and abort, plus native buffer-slot, length, alignment, overlap, and shared-memory checks. |
| Kernels-owned WGSL view-copy lowering | `verified` | The lowerer consumes authority-bound immutable backend-neutral specializations, preserves whole-root f32 bits through u32 storage, interval-proves signed i32 arithmetic, lowers canonical source/destination maps, emits structured guarded fill loads, validates device and transient-working-set limits, and derives semantic plus device-specific hashes. One-in-flight ownership, timeout/abort stale-result suppression, exact scope drainage, distinct error stages, and device-loss invalidation have deterministic fake-device coverage. The required headed lane emitted one validated `passed` terminal record for all nine bit-exact CPU/WebGPU cases on Apple Metal 3; headless absence remains a truthful failed environment record. |
| Shared required-WebGPU evidence test contract | `verified` | A neutral, unpublished test-support module now owns adapter/device acquisition, required-versus-advisory outcome rules, generic terminal-envelope validation, and exactly-once emission. Kernels retains its suite-specific ordered case and observation validation and passes the same advisory no-adapter path through the shared contract. Compiler and later JIT lanes consume this helper rather than fork release-evidence semantics. |
| Compiler verified-layout binding preparation | `verified` | Compiler now depends on semantic-core through public `/layout` and `/schema` protocols and prepares explicit read-only, row-major-flat parameter bindings. Prepared objects are authority-bound and deeply immutable, retain the semantic layout hash plus a deterministic binding-projection hash, resolve dynamic dimensions once, reject non-global views and duplicate/malformed bindings, and provide a collision-resistant layout-bound compile-cache key without changing frozen semantic IR. Lowering into memory references is the next slice. |
| Compiler read-only layout lowering | `verified` | A separate layout-bound compile entrypoint rewrites direct guarded reads after ordinary runtime lowering and before semantic-IR verification. It unflattens one stable non-escaping `uint` logical index, substitutes the verified positive-affine map, and sends the same physical expression through CPU reference and WGSL legalization. Initial support is specialized nonempty rank-2/3 global `f32`; only index-map predicates proved true over the complete logical domain are erased, while conditional predicates, writes, aliases, pointer offsets/rebasing, signed/mutated/escaped indices, non-affine maps, unaligned byte maps, rank drift, and u32 overflow fail closed. The frozen compiled wrapper is instance-authorized, execution validates the complete verified root allocation, and full semantic/binding hashes enter pipeline identity. Six supported cases pass complete source/output bit comparison on actual Apple Metal 3; padding remains explicitly unsupported. |
| Compiler L2 structured view-copy lowering | `verified` | A sibling opaque preparation/compile boundary consumes the exact verified `view-copy@1.0` operation and shared specialization; L1 remains unchanged. The initial source profile admits one guarded direct flat source-to-destination copy, lowers canonical rank-2/3 maps with exact BigInt i32/u32 interval proofs, emits a real inner branch, keeps source address/load only in its true arm, stores exact fill words in its false arm, and binds whole roots as `u32`. Compiled/runtime authority is instance-bound; exact distinct native non-shared roots and all semantic/routing identities are enforced. | No initial-profile work remains. Byte-unit/non-affine maps, helpers, aliases, pointer rebasing, extra effects, and arithmetic outside signed-i32 addressability remain explicit future profile work. | Focused CPU/WGSL/adversarial tests plus three-case exact-source actual-WebGPU evidence at `aa605421`; retained verification independently rederived the expected manifest from the checked-out fixture and built public APIs. |
| Compiler required-device layout conformance | `verified` | The L1 and L2 required/advisory lanes prepare exact proofs once per case, compare semantic/CPU/complete raw-u32 roots, validate prepared topology, execute on real WebGPU, drain queue and late errors, race device loss/timeouts, and emit closed terminal records. Plan-derived topology is explicitly labeled planned. Compiler publish workflows retain exact-SHA logs; the final guard requires both L1 and L2 markers and rejects missing, stale, dirty, or foreign evidence. | Standard release rerun only. The compiler runner does not claim the kernels runner's broader cross-operation error taxonomy. | At exact clean `aa605421`: L1 suite `@2` passed 6 cases and L2 passed 3 padded rank-2/rank-3 cases on Apple Metal 3; combined compiler publish guard accepted both exact markers. |
| Canonical view-copy artifact construction | `verified` | One generic sink snapshots closed canonical JSON, sorts set-like symbols/constraints, normalizes resource-bounded layout algebra, fixes source/destination allocation/map/view roles, forces disjoint materialization, verifies both opaque artifacts, and returns canonical IDs plus hashes. Its dense-permutation wrapper derives output shape, balanced row-major stride/storage expressions, effects, and reject policy from only canonical input shape, axes, and dtype. Producer/artifact metadata is non-semantic. | No initial-constructor work remains; future frontends must use this sink and keep routing identity out of semantic input. | 11 semantic-core files/99 tests; pinned shared rank-2/rank-3 artifact hashes and exact CPU bits; zero extent, >i64 byte lengths, u64 overflow, rank/expansion budgets, transport-hash independence, set-order independence, mutation/authority, hostile-input rejection, packed export/runtime/declaration proof. |
| Typed JIT permutation request emission | `verified` | One `GpuExecutionSubmission` builds the frozen plan once, then emits a separate `browsergrad.jit.tensor-plan-semantic-requests@1.0` envelope for every post-fusion `PERMUTE`. Requests contain only `inputShape`, normalized `axes`, canonical `f32`, and plan-local `valueId`; malformed dtype/rank/extent/axis/output-shape/arg cases fail before bridge execution. Public `permute` and `transpose` normalize negative axes and reject missing, duplicate, non-integer, or out-of-range axes before UOp construction. | No initial-profile work remains; broader view families require new typed request variants, not plan reinterpretation. | Shared rank-2/rank-3 emission fixture plus focused JIT integration; typecheck/lint/diff-check clean; generated Python source synchronized. |
| Resident semantic view-copy dispatch | `verified` | Kernels executes a module-authorized prepared view-copy directly from a resident whole-root `GPUBuffer` through the exact prepared WGSL. The route performs no host upload, readback, or index reconstruction; validates declared and physical source bytes, storage usage, device/dispatch limits, and permits only a nonempty dense zero-offset destination that overwrites its complete root. Direct callers use an async production-scoped API; tensor-plan execution alone receives a private, non-exported synchronous issue capability under its owning scopes. Semantic preparation preserves liveness/release accounting, and failed dispatch/materialization destroys roots and clears pools rather than re-pooling invalid buffers. | No initial-profile work remains; partial destinations still require an explicit initialized-destination contract. | Deterministic fakes prove LIFO validation/OOM/internal scope initiation, delayed loss, synchronous issue/pop/completion failures, no pre-scope handle mint, no upload/readback/legacy dispatch, poisoned-pool rejection, and clean retry. The JIT required-device lane proves resident roots followed by exactly one explicit complete-root materialization. |
| JIT semantic request consumption and execution | `verified` | Kernels accepts only the closed JSON request envelope, requires exact one-for-one ordered `PERMUTE` coverage, constructs artifacts solely through semantic-core, excludes `valueId` from every semantic hash, and authority-binds prepared WGSL. JIT sends one fused submission and chooses separate semantic materializing/resident methods; missing methods fail before legacy dispatch. The semantic schedule projection erases PERMUTE args, so execution cannot recover axes from the frozen plan. Live semantic handles retain the exact authority-bound preparation and its per-dispatch profile promises until release. | No initial-profile implementation remains; release workflow repeats exact-source evidence. | Shared fixture tests; mock legacy refusal/no-readback selection; per-handle preparation/profile lifecycle tests; exact clean-source two-case Apple Metal 3 evidence with actual submitted topology and one explicit readback per resident root. |
| JIT required-device semantic-permute conformance | `verified` | Before Chromium starts, the JIT lane runs production `_tensor_plan_submission` over each shared fixture and captures its exact plan plus canonical request JSON. The browser executes those captured submissions through `run_tensor_plan_resident_semantic`, compares the authority-bound execution trace with the prepared manifest, requires one settled actual dispatch profile, verifies one resident complete root, performs exactly one explicit materialization, compares every u32 bit, drains queue/late errors, races loss/timeouts, and emits one validated terminal record. Planned and submitted topology are separate; deterministic artifact and terminal hashes bind all semantic, environment, and outcome facts. | Standard release rerun only. | Exact clean-source Apple Metal 3 evidence at `aa605421410e9d4190d8939c24b1057731111231`: artifact `a38abc4d1c3c004abd42a0bed3748052b5c5e2a82a20f1328171edc9d87e1fbf`, case set `4f910a47fade14133377f4161474207295b803cca4da2aaaff91921dbda1e374`, prepared backend `2076022d9c95cb04282395d0b1e2c46f354e4b4c6b5979d7b65c07a8e80ae563`, device profile `0ef434a09b4cc9919ba30c92e791dc2a2138ef42a6f02cc282d1ce1d07d22b63`, and terminal manifest `3838a75341295d79900036e6c51ab64795cca87b11f4781d425983b4c3b866b7`. |
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
| D-044 | 2026-07-16 | accepted | Compiler L2 conformance is a separate closed three-case suite over complete raw-u32 roots. Its retained verifier independently derives the expected case, artifact, specialization, backend, device, and terminal manifests from the checked-out fixture and built public APIs. A terminal digest proves record integrity only; exact clean source plus the trusted workflow and final combined L1/L2 publish guard provide execution authority. Plan-derived topology is labeled planned unless an immutable runtime trace proves submission. | A self-sealed record could otherwise validate invented cases, hashes, or source claims, and planned pipeline/workgroup counts could be mislabeled as observed execution. Separating suite meaning, record integrity, source binding, and workflow authority prevents false conformance while preserving the established L1 contract. |
| D-045 | 2026-07-16 | accepted | Freeze every existing CuTe/WGMMA source-spelling normalizer as one legacy compatibility adapter. Gate 3 adds a sibling ahead-of-time C++/CuTe frontend artifact produced by a real pinned compiler/header profile; it lowers resolved facts into semantic-core and never widens the CUDA-lite parser, normalizers, or compiler-shaped semantic IR. | Current regex motif replacement can preserve corpus outputs while bypassing C++ lookup, templates, overloads, CuTe evaluation, spans, and typed target intrinsics. Freezing it prevents migration work from adding more false compatibility and keeps deletion measurable. |
| D-046 | 2026-07-16 | accepted | Split Gate 3 authority into four records: a reusable pinned frontend profile, a structurally verified one-TU resolved artifact, a detached attestation-authorized wrapper, and tiered lowering/execution evidence. Artifact semantic identity excludes transport producer metadata and never establishes trust. Version 1 keeps closed resolved C++ facts, source provenance, dynamic layout integer expressions, typed target intrinsics, and structured function bodies; it has no raw-AST or opaque-operation escape hatch. | Coupling source sets to toolchain profiles would multiply profiles; treating `producer`, hashes, or a `trusted` bit as provenance would be unsafe; omitting dynamic expressions/control flow/intrinsics from v1 would force immediate schema replacement. Separate authority layers preserve deterministic reuse, make CUTE-007 outcomes explicit, and allow only an attested artifact to enter semantic lowering. |
| D-047 | 2026-07-16 | accepted | Represent detached Gate 3 provenance as canonical JSON in an in-toto Statement v1 carried by standard DSSE PAE, with a versioned BrowserGrad predicate rather than a false SLSA claim. Version 1 authenticates one fixed-width P-256 P1363 signature against a profile-pinned SPKI trust store before evaluating claims; binds the full artifact projection, exact tagged Git revision, profile/toolchain, resolved `linux/amd64` container manifest, runner, sandbox/limits, deterministic invocation manifest, output manifest, successful exit, and VFS roots; then mints instance-bound attestation and authorization wrappers. Trust-root rotation creates a new profile/artifact authorization lineage, while shared semantic lowering hashes remain provenance-neutral. | A custom non-in-toto envelope would require wholesale replacement for later Sigstore support; a caller-supplied unpinned key would self-authorize; SHA-1-shaped strings, mutable image tags/indexes, unauthenticated policy evaluation, and source-root prefix matching would leave replay, substitution, interoperability, and VFS gaps. DSSE/in-toto preserves a standard envelope and permits a later Sigstore verifier to mint the same opaque authority without changing lowering consumers. |
| D-048 | 2026-07-16 | accepted | Represent a standalone layout as an authority-bound `browsergrad.layout@1` value with one canonical index map and no allocations or views. Its trace reports element location and logical/predicate bounds only. CuTe layout-only parity compares this value and its full-domain traces; Gate 2 layout/kernel hash equality is deferred until a later source fact genuinely supplies the same dtype, storage, views, effects, destination, and materializing operation. | CuTe `cosize` is a codomain property, not allocation bytes. Manufacturing f32/global storage, aliases, a destination, or copy effects would prove BrowserGrad's invented program. Requiring full Gate 2 hashes would also force source-spelling reconstruction because equivalent direct-strided and permutation-derived maps are not yet canonicalized to identical expression trees. |
| D-049 | 2026-07-16 | accepted | Lower the first CuTe layout only from an instance-authorized artifact and one explicit selected entry. Recheck one-to-one entry/result/fact ownership, preserve nested mode hierarchy through explicit colexicographic composition, and evaluate source integer trees plus derived size/cosize through shared bounded dimension algebra using the caller's limits. Keep the semantic layout hash provenance-neutral; retain artifact/profile/attestation/span/macro lineage in a separate origin hash and side table. Define CuTe `cosize` as `layout(size(layout) - 1) + 1`, not affine address span. The initial profile rejects runtime integers and emits no storage, dtype, effects, CPU, or GPU meaning. | This prevents authorization bypass, ambiguous fact capture, hierarchy flattening, arithmetic denial of service, cache fragmentation by provenance, and invented tensor semantics. Static layout tracing is independently useful, but widening it to runtime or tensor execution without explicit binding/storage authority would recreate the debt this gate is meant to remove. |
| D-050 | 2026-07-16 | accepted | Freeze a pre-run `browsergrad.compiler.cpp-cute.aot-job@1.0` request before adding the producer. It content-addresses the exact prepared profile; canonical source repository/revision; stable source file IDs, paths, bytes, and hashes; one main-file Clang declaration-token anchor; expected stable output entry; and expected artifact schema plus source/header/input closure. The schema contains no raw flags, command, environment, host path, include override, output destination, or trust claim. The runner must derive all execution from the prepared profile and compare actual opened bytes and verified output back to the request. | Deriving an invocation only after artifact emission lets output choose its own input/entry story. A caller-supplied command or path would bypass the pinned compiler/VFS policy. A source-text name selector is ambiguous and encourages spelling handlers; a byte/token anchor is only a root request that the real frontend must resolve through AST ownership. Separating request authority from runner receipt and attestation keeps intent, observation, and trust independently auditable. |
| D-051 | 2026-07-16 | accepted | Treat the canonical normalized artifact envelope bytes as a distinct resource from semantic artifact identity. Recompute and expose raw SHA-256 plus byte length, reject noncanonical byte inputs, use the raw digest as the in-toto subject, and bind raw/transport/semantic identities in the signed predicate and output manifest. Retain both raw and semantic identity in authorization and compiler-origin records. | An in-toto subject identifies an emitted resource, not an equivalence class of meanings. Attesting only the semantic hash would allow distinct producer metadata or serialized bytes to share a subject, while using only raw bytes would make semantic caches provenance-dependent. The split prevents substitution without polluting shared semantic hashes. |
| D-052 | 2026-07-16 | accepted | Model the runner receipt as a closed content-addressed report over prepared intent, observed inputs/invocation/resources, and exact output, with separate opaque authority for structural verification and strict canonical-byte origin. Derive invocation identity only from the prepared job/profile. Keep process success distinct from accepted/rejected frontend outcome, and cover every profile extraction/process ceiling while independently recomputing artifact-visible counts. Do not expose a production helper that can synthesize observations from intent. | An artifact-derived invocation is post-hoc; an object-only wrapper can fabricate byte observation; accepting only resolved artifacts loses negative provenance; and four process metrics leave most profile ceilings unaudited. The split allows later detached provenance to authenticate one exact receipt without letting the schema itself claim Docker isolation or producer trust. |
| D-053 | 2026-07-16 | accepted | Make the strict-decoded receipt resource the sole input authority for detached attestation verification. Derive artifact, profile, job, source, toolchain, sandbox, invocation, and output from its opaque record; sign receipt ID plus exact raw digest/length; and remove caller artifact/profile inputs from semantic authorization. Preserve rejected artifacts as attestable producer results but refuse lowering authority. | Comparing signed declarations only to a profile authenticates policy intent, not a runner observation. Accepting caller artifact/profile objects after receipt verification leaves a substitution seam. Receipt-owned authority closes both gaps while retaining negative diagnostics and keeping semantic hashes provenance-neutral. |
| D-054 | 2026-07-16 | accepted | Before adding the Docker shell, bind a canonical built-in sandbox policy, deterministic output-independent logical execution-plan hash, OCI image-config digest, and closed execution-environment manifest through profile, receipt, detached provenance, authorization, and compiler-origin lineage. Describe the ban as user-produced native execution, not all native execution. | A policy digest without deterministic plan derivation lets a trusted runner report safe declarations while executing different argv/mount/runtime settings. A platform-manifest digest alone does not let the offline runner prove which local config executes. Builder identity without runtime/kernel/cgroup/security-module identity leaves resource and isolation claims ambiguous. The old native-execution wording also contradicted use of a native compiler/supervisor. |
| D-055 | 2026-07-16 | accepted | Give the offline runner only prepared-job authority and exact source blobs; synchronously snapshot those blobs before any asynchronous policy/hash work, verify length/digest and the declaration-token anchor, and retain snapshots behind opaque authority. Accept producer output only as one fixed magic/version frame containing independently bounded canonical artifact and receipt bytes; strict-decode artifact first and receipt second. Raw-byte accessors return non-authoritative copies. | Reaccepting profile, commands, environment, paths, or output destinations would split execution ownership. Hashing caller buffers after an await creates a TOCTOU seam. Streaming multiple JSON documents or exposing mutable authoritative arrays makes completeness, framing, memory limits, and byte origin ambiguous. One closed frame lets the process shell cap capture before parsing and preserves the existing strict artifact/receipt authority chain. |
| D-056 | 2026-07-16 | accepted | Make the entire built-in sandbox policy deeply immutable and require plan hashing to re-verify its canonical identity. Pin every artifact/receipt decode limit in that policy. Require no IPC namespace and an empty image-config, override, and effective environment. At the byte boundary, use captured data descriptors and typed-array intrinsics, reject unknown/duplicate IDs and wrong lengths before copying, and prove the internal element kind is exactly `Uint8Array`. | A shallow-frozen policy can change under a stable claimed digest. Ambient decoder defaults let one plan hash acquire different behavior after dependency upgrades. Docker private IPC creates an undeclared writable `/dev/shm`; an empty override list does not suppress image `Config.Env`. Ordinary getters, typed-array species, oversized pre-verification copies, and prototype-disguised word arrays reopen TOCTOU, memory-amplification, and byte-length/element-semantics gaps. |
| D-057 | 2026-07-16 | accepted | Separate image proof into three opaque authorities: cacheable exact OCI manifest/config metadata, authorization of that metadata for one prepared plan, and a later temporal live-Docker observation minted only by the fixed Node process state machine. Metadata verification derives raw manifest/config descriptors, accepts only one closed OCI `linux/amd64` leaf, requires self-contained distributable layer descriptors plus exact config diff-ID correspondence, rejects all image execution config, and pins per-resource plus aggregate layer budgets. It does not claim unseen layer bytes. Caller-supplied Docker JSON can never mint live observation authority. | Docker image inspection is a daemon projection, not the raw OCI config, and caller bytes can fabricate it. Combining static content with temporal local state creates stale authority and no endpoint/argv/exit/order proof. Conversely, calling manifest/config metadata a complete image would falsely imply layer blobs were supplied and rehashed. The three-stage split preserves cacheability, prevents self-observation, and gives the later container state machine one exact plan-bound input without overclaiming acquisition or execution. |
| D-058 | 2026-07-16 | accepted | Pin the local observation contract to Docker CLI/Engine 29.6.1, request API 1.49, client-default/engine-maximum API 1.55, engine-minimum API 1.40, the exact local Unix socket, `linux/amd64`, and the containerd image store. Probe in fixed order `version -> info -> image-inspect`; accept the platform-selected manifest ID only after the runtime/store proof; and bind its raw config descriptor, repo digest, diff IDs, platform, and semantically empty image config to the authorized OCI metadata. Every child settles on `close`; timeout, abort, and overflow terminate the Linux process group and wait for closure, while an unreaped child after grace fail-stops the parent. Observation authority is minted only after private run-root removal, and production/test issuers are disjoint. This contract does not establish Docker binary provenance, live-daemon availability, container execution, or producer trust. | Generic Docker API descriptions do not determine image-ID semantics across storage backends and versions. Exact runtime/store preflight plus pinned implementation semantics prevent a config digest from being misread as the selected manifest digest. Waiting for `close` prevents cleanup or authority minting while stdio or descendants remain live; fail-stop is safer than returning after failed termination. Disjoint issuers prevent fake-process evidence from crossing the production authority seam. |
| D-059 | 2026-07-16 | accepted | Make the execution environment an exact canonical byte authority bound to one prepared profile instance, not a free digest string. Keep declared configuration separate from per-run enforcement evidence. Bind its raw/content identities into plan v2, staged supervisor input, receipt 1.1, provenance, and ordered OCI layer/diff-ID authorization. Recompute every inline inventory hash rather than trusting self-declared closure digests. | A profile string alone cannot prove which bytes named the kernel/runtime/toolchain/image closure, and daemon inspect cannot establish kernel enforcement. Passing an opaque prepared authority closes substitution across pre-run seams while explicit `environment-only`/`detached` semantics prevent configuration from masquerading as run evidence. Recomputed inline hashes prevent internally contradictory manifests. |
| D-060 | 2026-07-16 | accepted | Vendor one exact upstream Moby seccomp profile with repository/revision/path/hash/length lock; bind its raw identity into sandbox policy 1.3 and execution environment; open the checked source with `O_NOFOLLOW`, require one regular single-link exact-length file, verify descriptor identity before/after read, compact validated JSON, and stage one private mode-0444 snapshot under the run root. Docker creation receives only that private path. Created-container inspection must report the exact compact profile after `no-new-privileges`; missing, changed, or reordered security options force cleanup. Effective seccomp remains separate externally authenticated run evidence. | Relying on Docker's mutable runtime default leaves policy meaning dependent on host daemon version/config. Passing a shared host path permits replacement between verification and Docker CLI read. Conversely, treating create arguments or `HostConfig.SecurityOpt` as proof of kernel filter installation would overclaim enforcement. Private checked staging closes request substitution while preserving the declared/requested/effective evidence boundary. |
| D-061 | 2026-07-16 | accepted | Make browser-local Clang-WASM the primary Gate 3 producer and portable-product requirement. Keep the three deployment modes and producer-neutral artifact protocol. Reclassify native/Docker AOT as an optional CI/reference parity producer that can qualify corpora or prebuild artifacts but can never be required by browser runtime, ordinary browser compilation, or the portable Gate 3 exit. This supersedes only D-045's AOT-first deployment selection; its legacy-adapter freeze and resolved-artifact seam remain in force. D-050 through D-060 continue to define the optional native lane and do not constitute browser-local evidence. | The product promise is source compilation and portable execution in the browser. Treating the native container as the primary next checkpoint would turn an optional producer into an end-user/environment requirement and allow extensive sandbox evidence to hide the absence of a browser compiler. One artifact protocol plus explicit parity avoids a second semantic implementation while preserving useful CI/reference tooling. |
| D-062 | 2026-07-16 | accepted | Replace the unpublished CuTe frontend profile, artifact, AOT request, execution-environment, and receipt wire contracts with closed version-2 schemas before attaching either real producer. Model input roots and files with exact source/compiler-resource/dependency ownership; bind ordered forced includes to the prepared profile; preserve declaration initializer expressions; permit locationless diagnostics only for invocation/profile/compiler subjects; and distinguish exact observations/emitted counts from configured ceilings. Keep an entry request output-independent: it identifies declaration kind plus source anchor, while a receipt records the actual resolved artifact entry. Do not retain v1 readers or aliases because no real producer, persisted artifact, release, or compatibility consumer exists. | Carrying ambiguous v1 ownership and self-asserted forced includes into browser asset work would make artifacts producer-dependent and permit undeclared header influence. Prescribing a future semantic entry ID from the request reverses authority: producer output must resolve the source anchor and prove its own canonical artifact closure. Preproduction replacement is cheaper and safer than shipping a migration burden for bytes that never existed outside fixtures. |
| D-063 | 2026-07-16 | accepted | Make lowering consume one common producer-authorized artifact authority. Browser-local evidence binds an exact prepared browser profile, pinned asset set, worker request/result, verified artifact bytes, and resource/cancellation outcome. Optional native evidence binds the existing AOT job, receipt, and detached attestation chain. The common authority exposes producer-neutral artifact/profile/source identities plus a closed evidence-kind and evidence hash; producer-specific records remain behind their own opaque authority. Compiler origin records the evidence kind/hash instead of requiring an attestation ID. This supersedes D-046 only where it says every lowering input must be attestation-authorized and D-049 only where origin is hard-coded to attestation lineage. | Browser-local compilation cannot honestly reuse an OCI/DSSE receipt as if a local worker were a remote trusted builder. Letting lowering accept raw verified artifacts would remove execution-policy authority entirely. A small common authority preserves one semantic lowering path, keeps provenance out of semantic identity, and makes browser/native parity possible without inventing a second artifact or compiler stack. |
| D-064 | 2026-07-16 | accepted | Split deployment identity from compilation semantics. A prepared profile retains its full `profileHash`, while canonical frontend artifacts bind a versioned `compilationContractHash` covering language/options, requested target, semantic compiler build/resource identity, exact dependencies and VFS, compatibility, semantic-adapter manifest, and semantic extraction/output ceilings. Producer binary hashes, worker/container policy, provenance, and process ceilings remain deployment evidence and are excluded. Browser and optional AOT profiles may therefore emit the same semantic artifact while retaining distinct transport, resource, and trust evidence. | Putting the full deployment profile hash inside the artifact semantic payload made browser/native parity impossible: Docker image, worker, executable, provenance, and process-policy differences changed semantic identity even when resolved frontend facts were identical. Removing all profile binding would be equally unsafe. The explicit compilation contract preserves every source-analysis input while separating producer mechanics from semantic meaning. |
| D-065 | 2026-07-16 | accepted | Define the browser executable and asset trust boundary before implementing fetch or execution. The package owns the dedicated module-worker JavaScript and pins its build/hash in the browser profile; the remote asset manifest cannot supply executable JavaScript. One monolithic Clang/extractor WASM binary must equal both the profile compiler and extractor binary identity. The profile, not the manifest, owns hard resource ceilings and pins an asset-set hash plus build-provenance lock. The manifest admits only unique normalized same-origin root-relative URLs and exact content-addressed WASM, semantic-adapter, compiler-resource, and dependency-header assets with source-ABI, mount, content-set, ownership, and total-byte closure. Manifest authority explicitly proves no fetch, unpack, mount, or execution. | A manifest that chooses its own ceilings or loader code is self-authorizing. Separating Clang and extractor identities without a second exact executable asset leaves an unbound compiler binary. Treating manifest verification as asset availability would hide cache substitution, archive traversal/collision, decompression, and worker-runtime failures. This boundary keeps the later fetch/unpack/worker authorities independently testable and prevents policy from being weakened by downloaded metadata. |
| D-066 | 2026-07-16 | accepted | Replace generic tar/gzip header assets before any real compiler asset ships. Use closed identity-encoded `application/vnd.browsergrad.vfs-pack.v1` bytes: fixed `BGVFSPK1` header/version and lengths, exact index/content-set hashes, sorted unique portable-ASCII relative paths, per-file length/hash index entries, and file bytes concatenated in index order. Admit regular files only; directories are implicit. Do not encode offsets, links, sparse files, devices, modes, ownership, timestamps, xattrs, Unicode aliases, or archive extensions. Identity `byteLength` and `unpackedByteLength` cover the complete pack equally; a separate `fileContentByteLength` owns mounted content. Structural inspection cannot authorize use: only exact bytes bound to one prepared manifest asset/include-root/mount authority may proceed. A future compressed transport requires a new pinned streaming decoder and output ceiling without widening the pack model. | The compiler VFS needs deterministic files, not archive interoperability. Tar would import PAX/GNU names, link traversal, sparse data, duplicate-path policy, metadata, encoding, and decompression behavior into a browser security boundary. Removing offsets and entry types makes overlap, gaps, and unsafe file kinds unrepresentable instead of merely checked. Separating structural validity from manifest-instance authorization prevents callers from self-signing arbitrary pack bytes. This closes a predictable post-asset hardening/refactor sprint. |
| D-067 | 2026-07-16 | accepted | Build one BrowserGrad-owned custom Clang `FrontendAction`/LibTooling AST extractor and cross-compile it with pinned Emscripten for a single-threaded dedicated Worker. Do not ship or invoke the full Clang driver, LLD, CodeGen, PTX assembler, CUDA runtime, or user-produced WASM. Keep the Emscripten build sysroot separate from the parsed-program virtual sysroot and drive explicit pinned Clang-CUDA host/device semantic passes. A pinned OCI builder is allowed only to produce reproducible release assets. Initial feasibility pins are LLVM `llvmorg-22.1.8` at `ca7933e...` and emsdk `6.0.3` at `db04e882...`; they remain spike candidates until corpus, browser, license, reproducibility, size, and memory gates replace fixture identities. | LLVM has no supported browser Clang distribution, while LibTooling and in-memory VFS are the correct in-process semantic-extraction seams. Emscripten owns browser Worker/module integration more directly than a browser WASI shim. A minimal extractor reduces asset size and syscall/process surface while preserving actual Clang preprocessing, lookup, Sema, templates, constexpr, and diagnostics. Candidate version pins prevent `latest` drift without prematurely claiming a shippable toolchain. |
| D-068 | 2026-07-16 | superseded | Before worker implementation, extract the unpublished AOT-named source job into a producer-neutral frontend request. Bind compilation-contract hash, exact source descriptors/snapshots, main path, source anchor, and expected artifact schema/version. Keep deployment profile/assets/worker/container, repository/revision, and pre-known output/header/input-closure hashes out. Declared source references and conformance expectations are detached. The profile pins the available VFS/header universe; evidence records the request-specific files actually opened. Add a distinct compiler-runtime ABI and separately coherent transfer/cache/VFS/WASM-memory/stack/output budgets. Host-verify exact self-contained worker bytes and length before Blob-module construction; no worker self-attestation or unverified URL fallback. D-074 supersedes its AOT composition and source-reference terminology. | Mandatory Git provenance blocks ordinary editor buffers. Comparing a request's actually opened headers to one profile-wide constant rejects valid sources with different include closures. Source target ABI cannot describe the WASM compiler runtime, and current asset ceilings can exceed the worker heap without a declared host-backed versus in-WASM VFS model. Module workers have no SRI option, so a stored hash is not enforcement until the host verifies the exact bytes before spawn. Correcting these unpublished seams now avoids producer-specific requests, false closure checks, impossible memory profiles, and bundler-dependent worker identity. |
| D-069 | 2026-07-16 | accepted | Remove the profile-wide expected-opened-header hash; dependency and include-root manifests define the complete available universe while artifacts/evidence own request-specific successful reads. Define compiler-runtime ABI v1 independently from source ABI: wasm32, a closed sorted required-feature set, exact import/export hash, unshared worker-owned memory, initial/maximum pages, stack and compiler-working reservations, host-backed lazy VFS ceilings, and host-verified WASM handoff with worker fetch forbidden. Package-owned Worker bytes have exact hash/length, one self-contained ES-module graph, host-verified Blob construction, no network authority, terminate-on-cancel lifecycle, and host-verified asset transfer. Browser `maxMemoryBytes` caps WASM linear memory; stack + compiler working + opened VFS copies + output fit inside it, it fits maximum pages, and stack fits initial pages. Host-retained pack bytes remain independently bounded and observed. | A profile-wide opened closure rejects valid sources; using source target width as compiler runtime identity is false; allowing worker URL fetch or imported module graphs bypasses host byte verification; and independent-looking ceilings that cannot coexist produce deterministic browser OOMs. Closing these unpublished fields before request/worker implementation keeps producer-neutral semantic identity separate from deployment mechanics and makes later acquisition, execution, and evidence authorities enforceable without another schema replacement. |
| D-070 | 2026-07-16 | accepted | Make frontend request v1 the common source-analysis intent. Its wire identity contains compilation-contract hash, exact sorted source descriptors, main virtual path, one source anchor, expected artifact schema/version, and only semantic/input/output ceilings shared with compilation identity. Exact unshared source bytes are supplied out of band, length-checked before copy, copied before await, hashed, and retained behind opaque authority. Support both current artifact entry families: layout-variable and view-copy-function. Deployment profile, worker/container/assets, raw compiler argv, host paths, environment, Git repository/revision, wall/CPU/memory/process ceilings, expected artifact/output/closure hashes, and actual opened inputs are absent. Detached provenance/conformance records cannot alter request identity. Retained AOT execution metadata must compose around this request rather than define a second source intent. | One request identity must survive browser/native producer differences and ordinary unsaved editor buffers. Including process ceilings or Git identity would fragment parity; including pre-known outputs or opened headers would make compilation self-fulfilling; copying before actual-length checks permits memory amplification; and hard-coding layout-only selection would force a schema replacement for Gate 3 dynamic view-copy work. Shared semantic-limit keys and two entry families keep request evolution at the artifact seam while opaque snapshot authority closes caller mutation. |
| D-071 | 2026-07-16 | accepted | Run two independent Clang CUDA semantic actions in BrowserGrad policy order: explicit device-only extraction first, then host-only validation only after device success. The device pass exclusively owns the canonical resolved declarations/types/templates/bodies/facts/entries and device ABI. The host pass contributes validation diagnostics plus host ABI keyed only by a common source-entity registry; it cannot reference device graph IDs or add semantic facts. Source-entity IDs are recomputed from entity kind, canonical source identity, and resolved file-content/range origin; declared domains must equal exact ABI ownership, and `shared` is derived from that domain set. Device ABI source identity must match canonical device types/declarations and exact function return/parameter types. Each executed pass records its exact target/auxiliary triple, device architecture, invocation mode, opened files, include edges, canonical observed-input hash, diagnostics, and explicit shared source-surface projection. Compiler-forced includes must appear in every executed pass; target-conditional source includes may differ. Accepted output requires both passes to succeed and their shared projections to converge. | Merging two target-resolved ASTs is undefined for `__CUDA_ARCH__`, target typedefs, overloads, ABI, and mangling. One shared opened-file closure is false under conditional preprocessing. Host-first is Clang Tooling's default sequencing, so device-first must be stated as BrowserGrad policy and implemented with explicit separate actions rather than misattributed to the driver. Syntax-shaped IDs and producer-declared `shared` flags would let accidental cross-pass drift collapse to two empty surfaces. Device-owned semantics plus content-derived common source identity gives one lowering graph without discarding cross-target correctness evidence. |
| D-072 | 2026-07-16 | accepted | Keep the canonical source-entity registry at payload scope rather than inside target ABI. Every successful semantic pass must report the exact sorted content-derived source IDs for every serialized device-entry root. Root IDs derive from the canonical device declaration kind, USR, and resolved source origin; accepted host/device passes require the same nonempty set. Registry domains equal the exact union of selected-root pass observations and ABI ownership, while ABI `shared` derives from that union. Shared-surface hashes include selected-root IDs before ABI projection. | ABI-only ownership allowed a producer to split one real shared root into device-only and fabricated host-only entities, mark both nonshared, and converge on empty ABI surfaces. The root projection closes that structural bypass without merging target-resolved ASTs. Authenticated worker/provenance evidence remains responsible for proving the producer truthfully reported Clang observations. |
| D-073 | 2026-07-16 | accepted | Give closed VFS-pack v1 one canonical writer. Validate bounded path length before UTF-8 allocation, require safe portable paths, snapshot each file immediately after intrinsic byte-view inspection and before inspecting the next caller record, sort by encoded path bytes, reject duplicate/file-as-parent paths, enforce independent index/content/pack ceilings, and compute every file/index/content-set digest from the retained snapshots. Writer output remains non-authoritative until the existing inspector and manifest-instance binder accept its exact bytes. | A reader without one writer invites format drift. Encoding an unbounded string first permits avoidable allocation, while deferring byte copies across hostile proxy or resizable-buffer reentrancy permits mixed-time or fabricated content. Immediate per-record snapshots give one deterministic invocation boundary without weakening the separate use-authority contract. |
| D-074 | 2026-07-16 | accepted | Delete the unpublished AOT job authority. Compose optional AOT as `frontend request -> run metadata -> verified artifact -> host-derived request binding -> receipt -> detached attestation -> common authorization`. Run metadata owns only AOT profile, request, and a declared Git source reference; source snapshots remain request-owned. The reference is signed metadata, not source-acquisition provenance. Receipt v3 names exact `runMetadataId`, `requestId`, and `requestBindingId`, but does not duplicate source descriptors, expected outputs, or resolved selection. Common lowering authorization exposes no AOT-only run field and advertises only evidence kinds with a real mint path. Profile 2.2 at this slice pinned provenance predicate `/v3`; D-079 advances the closed profile to 2.3. Policy 1.4 stages separate request and run-metadata controls; Docker labels those exact identities. Artifact v3 requires an exact declaration identity-token span for source-anchor binding. | Parallel source/job/output authorities let browser and AOT disagree while each appears internally valid. Producer-declared selection lets a receipt self-authorize its source relation. Calling caller-declared repository/revision provenance overclaims a relation not bound to exact request bytes. AOT-only or nonexistent evidence kinds on common authorization make browser work require fake metadata or misleading capability. One opaque chain keeps producer mechanics distinct while preserving one request, artifact, semantic lowerer, and future parity comparison. |
| D-075 | 2026-07-17 | accepted | Host asset acquisition accepts only the prepared manifest's exact same-origin, redirect-free identity resources; verifies exact response URL/status/optional content length, streamed byte ceiling, final length, and SHA-256; rehashes every cache hit; snapshots cache methods and bytes; and mints separate opaque acquisition, cache-admission, and VFS-installation authorities. Pack installation rebinds each exact manifest asset, rejects all cross-pack file/directory collisions after mounting, and counts both source-set and independently verified pack copies against the host-retained ceiling. Do not expose a generic installed-file copy API. Lazy file reads belong to one future worker-execution session that meters the aggregate WASM-resident opened-file reservation and owns transfer/disposal/cancellation. | A per-file size check would let repeated opens exceed the single linear-memory reservation. A generic copy would also split ownership from the Worker lifecycle and invite prototype/retention bugs. Keeping installation metadata-only closes the host trust chain without pretending to enforce memory that only the Worker can own. |
| D-076 | 2026-07-17 | accepted | Separate the canonical Clang-WASM build-input lock from final build provenance. The lock may select exact upstream archives, builder image, toolchain components, recipe, output plan, and reviewed notices, but its opaque authority always reports `releaseReady: false`, carries derived blockers, and cannot mint or satisfy the browser profile's build-provenance identity. Output hashes, runtime ABI conformance, observed Wasm interface, license-file closure, reproducibility, and producer trust require independent authorities and detached evidence. | A self-hashing recipe proves selected bytes and policy only. Treating it as build proof would let an unexecuted plan authorize fabricated outputs and hide missing licensing, ABI, and two-build evidence. |
| D-077 | 2026-07-17 | accepted | Materialize the verified build-lock recipe through one deterministic, side-effect-free planner. The planner substitutes only verified absolute tool/root bindings and reproducibility prefix maps, emits native TableGen before the Emscripten cross-build, constructs a closed environment, and never acquires inputs, starts processes, invokes Docker, touches runtime state, or authorizes artifacts. The decoded lock recipe remains the sole build-command authority; direct integration tests must materialize the real builtin lock. | Keeping command construction separate from execution makes recipe drift testable and prevents ambient environment or convenience shell code from becoming an undocumented second build definition. Direct real-lock coverage closes the gap left by mutable mirrored test fixtures. |
| D-078 | 2026-07-17 | accepted | Make runtime ABI v1 one canonical strict-decoded design authority with exact C/VFS signatures, wasm32 memory, framing, statuses, lifecycle, result lifetime, and destructive cancellation. Keep observed-WASM and release authority detached. Pin an MVP instruction baseline, exact allowed extensions, and forbid every unlisted extension. Generated Emscripten imports, support exports, table/global projection, and allowed custom-section/`target_features` bytes remain empty, hash-pinned, and release-blocked until a first real build is independently inspected and the manifest is reviewed and repinned. An observed module can never extend its own allowlists. VFS imports require checked non-wrapping ranges, complete validation, stable input snapshots, aligned disjoint outputs, no memory growth, and no invalid-range mutation. Opened-VFS memory is one aggregate session budget, and coexistence includes the live input frame. | Link flags and browser reflection cannot prove exact raw-WASM signatures, memory limits, structural features, or host-memory safety. Letting observation define authorization would allow a compromised or drifted binary to self-approve new host capabilities. Per-file memory language would permit repeated opens to exceed the heap reservation, while omitting the live input frame made the declared coexistence proof four MiB short. A design manifest plus separate bounded observation authority closes those boundaries while preserving truthful pre-build status. |
| D-079 | 2026-07-17 | accepted | Advance the closed frontend profile to 2.3 to bind the exact runtime-ABI resource and derive its fixed runtime projection. Give semantic compilation identity its own explicit `browsergrad.compiler.cpp-cute.compilation-contract@1.0` schema/hash domain, excluding the full profile wire version. Working-memory and aggregate opened-WASM ceilings may narrow ABI maxima; host-retained pack bytes remain an independent profile/asset budget. | Coupling semantic identity to the enclosing profile version invalidated browser/AOT parity and caches for deployment-only revisions. A separate contract version preserves cache meaning and makes future profile migrations local. Deriving runtime invariants from one canonical ABI prevents a second manually synchronized interface truth. |
| D-080 | 2026-07-17 | accepted | Bind the exact canonical runtime-ABI bytes into asset-manifest 1.1, the verified acquired asset set, and the deterministic build-output closure before Worker work. Keep standards validity, exact Wasm byte identity, observed structural projection, and ABI conformance as separate facts. Raw inspection accepts spec-valid padded LEB and generic fatal UTF-8 names, preserves exact target-feature wire order while requiring unique names, validates the complete module, classifies proposal-specific opcodes, and compares only against independently prepared ABI authority. Large or untrusted inspection belongs in a disposable verifier Worker because synchronous engine validation and hashing are not preemptible. The unresolved first-build allowlists remain empty, so this implementation cannot mint production conformance until independent review and repin. | Fetching an ABI document without strict decoding permits same-hash/profile drift at the Worker boundary. Treating valid padded encodings as malformed confuses canonical JSON policy with the Wasm format. Parser-wide ASCII, assumed target-feature sorting, delimiter-built inventory keys, incomplete extension classification, or one-way export checks can reject valid modules or falsely authorize a different interface. Running a 256-MiB validation path on the main thread would turn a bounded parser into a responsiveness failure. |
| D-081 | 2026-07-17 | accepted | Treat pure Worker control/artifact verification as caller-frame consistency only. It terminalizes and compacts the prepared invocation before parsing, checks byte ceilings before copies, enforces exact identity/counter/artifact/input/resource relationships, and may return only an opaque authority with execution, termination, timing, and lowering claims all false. Discard is local lifecycle closure, not Worker termination proof. A separate host-owned controller must create the exact verified Worker, own its event source and nonce, measure host time, accept one terminal event, and terminate/replace the Worker before minting browser execution evidence. | A fresh nonce prevents replay but does not authenticate the sender. Allowing caller bytes, booleans, or wall-time claims to mint evidence would make the Worker path self-attesting and let fabricated results enter common lowering. Keeping the pure verifier useful but non-authoritative preserves testability without pre-approving the missing controller. |
| D-082 | 2026-07-17 | accepted | Advance the closed browser profile to 2.4 and name the VFS budgets by what they prove. A live file handle consumes a logical full-file reservation; range reads copy into caller-owned Wasm destinations, so neither the reservation nor a successful read is reported as resident Wasm memory. Runtime ABI maxima are 402,653,184 live-open logical bytes, 262,144 indexed nodes, and 134,217,728 logical index bytes; the current browser profile narrows index limits to 65,536 nodes and 33,554,432 bytes. Logical index bytes count one 32-byte ABI metadata record plus canonical absolute-path and immediate-basename UTF-8 bytes per node. Final mounted paths must fit the ABI ceiling. Internal nodes use owned frozen discriminants. Release values require measured final-pack inventory and browser peak-memory qualification. This supersedes the Wasm-resident-copy wording in D-069, D-075, and D-078 without changing their authority/lifecycle decisions. | Reusing session-call or asset-byte limits for index metadata couples unrelated resources and guarantees a later schema cleanup. Calling full-file reservations resident Wasm bytes is observably false for lazy range reads. An unbounded eager Map/tree allows valid packs to amplify Worker heap/CPU before compilation, while relative-path-only validation can create mounted paths the ABI cannot address. Dedicated, narrowable limits and exact terminology close those gaps before the real Worker exists. |

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

### 2026-07-16 — Gate 2 exact-device closure

- Added a compiler-owned three-case L2 required-device suite for rank-2 and
  rank-3 guarded padding. It checks semantic-core CPU, compiler CPU, and actual
  WebGPU over complete raw-u32 roots, including NaN fill bits and canaries.
- Hardened retained evidence so expected manifests are independently rederived
  from the checked-out fixture and built public compiler/semantic-core APIs.
  Closed schemas and mutation tests reject forged self-sealed pass records.
- Added exact source-scope checks, exact-SHA retained logs, release-workflow
  ordering/conditions, packed compiler L2 consumers, and a final compiler
  publish guard that requires both L1 and L2 evidence markers.
- Corrected compiler L1 evidence terminology: preparation-derived pipeline and
  workgroup facts are `planned`, not `submitted`; the suite is now version 2.
- In one clean detached worktree at `aa605421410e9d4190d8939c24b1057731111231`,
  compiler L2, compiler L1, kernels, and JIT strict Apple Metal 3 lanes all
  passed and their retained verifiers/guards accepted the exact source. Gate 2
  is therefore `verified`. Registry publication was not performed.

### 2026-07-16 — Gate 3 legacy-normalizer freeze

- Audited parser, normalizer, semantic, backend, corpus, and release paths.
  Current CuTe corpus success is source-pattern replacement, not real frontend
  compatibility under CUTE-002.
- Froze the three CuTe/WGMMA exception modules plus their production pipeline
  entrypoint by exact source hash. Architecture checks also reject missing or
  newly split exception modules.
- Selected an AOT resolved frontend artifact as the first Gate 3 deployment
  mode. Browser-safe verification/lowering stays in compiler; native extraction
  remains outside browser packages and must use pinned compiler/header inputs.
  D-061 later superseded this AOT-first deployment selection while retaining
  the artifact seam and optional native parity lane.

### 2026-07-16 — Gate 3 closed AOT profile authority

- Added a closed `browsergrad.compiler.cpp-cute.frontend-profile@1.0` verifier
  with an opaque prepared authority and a pinned canonical profile hash.
- Separated extractor identity from CUDA-capable compiler identity. The profile
  pins the extractor and compiler binaries/builds, container digest, compiler
  resource directory, CUDA/CUTLASS header manifests, exact CUTLASS commit,
  host/device ABI, ordered virtual include roots, source-feature scope, typed
  unsupported-intrinsic families, and producer resource ceilings.
- Rejected raw compiler flags. Version 1 accepts only ordered closed compiler
  option variants and refuses plugins, response files, host paths, duplicate or
  conflicting singleton options, unknown fields, accessors, and raised limits.
- Kept source-set identity out of the reusable toolchain profile. Exact source
  closure and detached attestation authority belong to artifact preparation;
  the profile's provenance record is policy only and cannot establish trust.

### 2026-07-16 — Gate 3 resolved frontend artifact verifier

- Added a closed, exact-version, one-translation-unit frontend artifact with
  bounded decode, opaque verification authority, normalized set ordering,
  semantic artifact identity, and separate transport identity.
- Added exact virtual input closure, include graph, byte ranges, macro origins,
  resolved types/constants/declarations/templates/overloads, source ABI,
  structured function bodies, layout/tensor facts, typed target intrinsics,
  diagnostics, and accepted/rejected frontend outcomes. Unknown variants and
  optional metadata fail closed; raw AST and backend-result escape hatches do
  not exist.
- Recompute source/header/closure hashes; verify graph ownership, reachability,
  range bounds, parent cycles, type containment, ABI bounds, static CuTe
  `size`/`cosize`, and typed unsupported-intrinsic retention. Dynamic layout
  values use bounded integer-expression trees rather than a static-only field.
- Kept transport `producer` outside semantic identity. Structural artifact
  verification still grants no provenance or lowering authority; detached
  attestation is the next authority layer.
- Extended compiler architecture reporting so every C++/CuTe frontend module
  is classified as frontend and forbidden from importing frozen CUDA-lite
  parser, semantic IR, static-layout, or layout/view-copy binding paths.

### 2026-07-16 — Gate 3 detached provenance authorization

- Extended the AOT profile with explicit runner identity, a canonical OCI
  repository, the resolved `linux/amd64` platform-manifest digest, sandbox
  policy, preprocessed-token/constexpr ceilings, allowed builders, and an exact
  P-256 SPKI trust-store hash. Container identity now belongs to deployment,
  not the compiler binary record.
- Added a canonical DSSE envelope using standard PAE over an in-toto Statement
  v1. Its predicate is BrowserGrad-specific and explicitly does not claim SLSA,
  Sigstore, GitHub-attestation, or real producer evidence.
- Signature verification now precedes policy evaluation. The authenticated
  statement binds complete artifact and transport identity, profile/toolchain,
  tagged Git SHA-1/SHA-256 source revision, sandbox/limits, resolved container,
  runner, deterministic invocation manifest, output manifest, zero exit, and
  successful outcome.
- Added profile-pinned trust roots, canonical SPKI/base64 and fixed 64-byte
  P1363 checks, exact builder/key binding, source/include VFS confinement with
  segment-safe prefix checks, caller-pinned profile/source expectations, and
  opaque instance-bound attestation/authorized wrappers. Rejected artifacts and
  structural copies cannot receive lowering authority.
- Consolidated profile/artifact test builders into one shared fixture seam so
  authorized tests derive compatible profile, input-closure, and artifact
  identities without duplicating hashes or schema construction.
- This slice implements the provenance primitive only. No real producer has
  emitted the artifact, no Sigstore bundle is verified, and no semantic lowerer
  consumes the authorized wrapper yet.

### 2026-07-16 — Gate 3 allocation-free layout semantics

- Added public standalone layout preparation and coordinate tracing to
  semantic-core. Preparation canonicalizes set-like symbols/constraints,
  normalizes layout algebra, verifies a one-index-map artifact, hashes its pure
  semantic projection, and mints instance-bound immutable authority.
- The verified artifact has zero allocations and zero views. Traces report
  element locations and logical/predicate bounds; they expose no dtype, byte
  range, alias, memory space, tensor, effect, CPU, or WebGPU claim.
- Pinned the first CuTe layout mapping `(3,2):(1,3)` to the complete logical
  sequence `[0,3,1,4,2,5]` plus negative/high boundary probes. Added dynamic
  binding/constraint, transport-neutral identity, canonical ordering, hostile
  request, resource-limit, mutation, and structural-authority tests.
- Recorded that full Gate 2 artifact/kernel hash equality belongs only to the
  later tensor/view-copy rung. The immediate next slice is an authorized
  compiler lowerer into this storage-free API.

### 2026-07-16 — Gate 3 authorized static layout lowering

- Added an internal compiler lowerer with no raw or merely verified artifact
  overload. It accepts one instance-authorized artifact and one closed explicit
  entry ID, then rechecks exact accepted-entry, root declaration, affine fact,
  and unique result ownership before lowering.
- Added bounded conversion of CuTe static integer trees into semantic-core
  dimension algebra. Caller limits now flow through artifact verification and
  derived size/cosize checks, preventing unbounded BigInt intermediates and
  per-layer limit drift.
- Preserved nested CuTe top-level modes through explicit colexicographic
  coordinate composition. Flat, nested, signed-stride, equivalent compound
  expression, ambiguous ownership, dynamic rejection, authority substitution,
  hostile request/options, cancellation, and resource-limit cases are covered.
- Corrected static `cosize` verification to CuTe's codomain definition,
  `layout(size(layout) - 1) + 1`; it is not an allocation address span.
- Kept shared semantic hashes independent of provenance. A compiler-owned side
  table binds the lowered value to exact artifact, profile, attestation,
  selected entry/fact/declaration, source spans, and macro closure without
  inventing storage, dtype, effects, or backend execution.

### 2026-07-16 — Gate 3 pre-run AOT producer request

- Added a closed content-addressed producer request as a separate authority
  from profile, artifact, receipt, attestation, and lowering. It pins exact
  profile and Git source identity, sorted source VFS blobs, one main source,
  one declaration-token anchor and expected stable entry, and expected artifact
  schema plus source/header/input closure.
- Kept execution and authority out of the request. Raw compiler flags,
  commands, environment variables, host paths, include-root overrides, output
  paths, credentials, and trust bits are impossible by schema; a future runner
  must derive its fixed invocation only from the prepared profile.
- Added profile source-file/byte ceilings, semantic decode budgets, exact closed
  versioning, canonical job/request hashes, instance-bound authority,
  cancellation, normalized HTTPS/Git identity, segment-safe VFS roots, unique
  stable file IDs, exactly one main source, and bounded in-file token anchors.
- Recorded two producer-evidence blockers before any real run: in-toto subject
  must bind exact canonical artifact bytes, and signed provenance must bind an
  observed sandbox receipt rather than policy declarations alone.

### 2026-07-16 — Gate 3 canonical artifact resource identity

- Canonicalized the normalized verified envelope once and exposed its exact
  SHA-256 and wire-u64 byte length separately from semantic and transport
  hashes. Authority-bound serialization returns that same resource.
- Made byte decoding strict: it snapshots the caller input and rejects any
  representation that is valid JSON but not the canonical normalized artifact.
- Changed the in-toto subject from semantic hash to raw canonical-byte digest.
  The signed predicate and output manifest now bind raw digest, byte length,
  transport identity, semantic identity, and all existing source/profile facts.
- Retained raw and semantic identities through opaque attestation,
  authorization, and compiler-origin records. Mutation tests prove neither can
  substitute for the other. No runner observation or producer claim was added.

### 2026-07-16 — Gate 3 AOT runner receipt contract

- Added a deterministic invocation manifest derived solely from the prepared
  job and profile. Post-run artifact facts cannot select their own invocation.
- Added a closed content-addressed receipt over reported opened job files,
  source/header closure, runner, resolved image/platform, extractor/compiler,
  dependency manifest, sandbox contract, selection result, and exact output.
- Kept process success separate from frontend acceptance. Exit-zero receipts
  cover both a resolved entry and a structurally valid rejected artifact with
  its exact blocking diagnostics; semantic authorization still rejects the
  latter.
- Covered every profile extraction/process limit. Source/header bytes and files,
  macro/template/declaration/type/constant/fact/diagnostic counts, and output
  bytes are independently recomputed where the artifact exposes them; native
  preprocessing/AST/constexpr/depth and process counters remain reported facts
  for a future authenticated producer.
- Split object verification from byte-origin authority for both artifact and
  receipt. Strict decoders snapshot caller bytes, reject noncanonical encodings,
  and mint the only wrappers provenance may later accept.
- Deliberately omitted a production receipt-construction helper. Only the real
  runner may create observations; the browser-safe package verifies them.

### 2026-07-16 — Gate 3 receipt-authenticated provenance

- Changed detached attestation verification to require one strict-decoded
  receipt resource. Structural receipt objects and copied wrappers fail before
  signature policy can mint authority.
- Bound receipt ID, exact receipt-byte digest/length, job ID, deterministic
  pre-run invocation, and exact output manifest in the signed run predicate.
- Derived signed source, toolchain, container, sandbox, and runner expectations
  from the verified receipt record rather than independently repeating profile
  declarations. The receipt verifier already binds those facts to the profile.
- Removed caller-supplied artifact/profile objects from authorization. Accepted
  lowering uses the receipt-owned decoded artifact; rejected artifacts retain
  authenticated diagnostics but cannot receive lowering authority.
- Added receipt/job lineage to the compiler origin side table while keeping the
  shared layout semantic hash independent of provenance.

### 2026-07-16 — Gate 3 logical execution-plan identity

- Added one checked-in canonical sandbox policy and deterministic logical plan
  hash derived from exact prepared job/profile/toolchain/VFS/limit inputs.
- Added OCI image-config and execution-environment manifest identities to the
  profile, runner receipt, detached statement, authorization, and origin chain.
- Corrected `nativeExecution: forbidden` to
  `userProducedNativeExecution: forbidden`; the pinned native supervisor and
  compiler are part of the producer, while user-produced binaries remain
  prohibited.
- Kept host paths, container IDs, timestamps, and other per-run operational
  values out of reusable plan identity. Actual container-state verification is
  the next imperative-shell boundary.

### 2026-07-16 — Gate 3 opaque offline-runner I/O authority

- Added an opaque output-independent run plan that synchronously snapshots only
  exact source `{ fileId, bytes }` values and verifies job-owned file count,
  identities, lengths, hashes, and declaration-token anchor before staging.
- Kept profile, command, environment, host paths, mount destinations, and output
  controls out of the caller surface. The prepared job remains their sole
  logical owner.
- Added one fixed magic/versioned output frame with big-endian u64 artifact and
  receipt lengths, independent and aggregate byte ceilings, exact EOF, and
  strict canonical artifact-then-receipt decoding.
- Kept authoritative snapshots and decoded bytes private. Staging and diagnostic
  accessors return disposable copies whose mutation cannot alter authority.
- Deep-froze the complete policy and pinned every decoder limit into its plan
  identity. Removed IPC and made empty image-config, override, and effective
  environment an explicit future container-inspection requirement.
- Hardened hostile-byte handling around captured descriptors and typed-array
  intrinsics: unknown/duplicate IDs and wrong lengths fail before copies;
  proxies, shared buffers, hostile species, and prototype-disguised word arrays
  cannot acquire byte authority.
- No Docker process, native extractor, real receipt, image, cgroup observation,
  or producer trust is claimed by this pure boundary.

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
| 2026-07-16 | Compiler L2 evidence and release regression | Compiler L2 evidence unit tests; compiler typecheck; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `pnpm test:release-packages`; workflow YAML parse; `git diff --check` | Passed: 3 evidence files/21 tests; full compiler architecture/build/typecheck/lint and 35 files/1004 tests; 19 hostile-archive and 35 Node release-security tests; packed/fresh consumers; workflow syntax and whitespace. | None. |
| 2026-07-16 | Exact-source compiler L2 required lane | Detached clean worktree at `aa605421`; L2 required browser lane retained to `/tmp/compiler-view-copy-bindings-webgpu-evidence-aa605421.log`, then independent verifier | Passed 3 cases on Apple Metal 3. Artifact `40fc9b7d73631e899a620b5bde7242794f358e2d6d0047de897b4a267b6d27d1`; case set `89ceeeae2c0a6d2f37999c48aedbffe3a0eaf4b6332307545cdb6188a4f96e0a`; prepared backend `6aa212edacaadab7ee7ff33605a08cc2293a0ac1d6859047d781719a3b523879`; terminal manifest `a38f794c3e87d69b3d8421ba73550b74fd8f7aae839e29a704b77a094ea62544`. | None. |
| 2026-07-16 | Exact-source Gate 2 strict-lane matrix | Same detached clean worktree at `aa605421`; compiler L1 required lane and combined compiler guard; kernels required view-copy lane; JIT required semantic-permute lane and verifier | Passed on Apple Metal 3: compiler L1 6 cases, kernels 9 cases, JIT 2 cases. Compiler guard accepted exact L1/L2 markers. Kernels artifact `446889ab...`; JIT artifact `a38abc4d...`, case set `4f910a47...`, terminal manifest `3838a753...`. All declared evidence source paths remained clean. | Gate 2 verified; begin Gate 3 audit. |
| 2026-07-16 | Gate 3 legacy-normalizer freeze | `node --check scripts/semantic-architecture-check.mjs`; focused semantic architecture Vitest; `pnpm architecture:check`; `git diff --check` | Passed: script syntax, 1 file/23 tests including source mutation and file-set drift, full semantic architecture guard, and whitespace. | Implement closed AOT frontend artifact/profile verifier. |
| 2026-07-16 | Gate 3 AOT profile authority | focused C++/CuTe profile Vitest; compiler typecheck; compiler lint; `git diff --check` | Passed: 1 file/13 tests covering canonical hash/authority, closed versions/fields, ordering, VFS traversal, toolchain identities, resource caps, option allowlist/conflicts, cancellation, and hostile accessors; strict typecheck and lint clean. | Implement closed frontend artifact verifier; keep artifact integrity separate from attested authorization. |
| 2026-07-16 | Gate 3 resolved artifact verifier | focused profile + artifact Vitest; compiler typecheck/lint; compiler and semantic architecture checks; `git diff --check` | Passed: 2 files/27 tests. Artifact tests cover bounded byte decode, opaque authority, canonical ordering and pinned hashes, producer/semantic identity separation, closed fields, input closure hashes, VFS/include/span/macro/reference/layout/outcome failures, typed unsupported intrinsics, lowered-only limits, and cancellation. Both architecture checks report zero legacy/backend/representation/C++-frontend leaks. | Add detached provenance verification and authorized artifact preparation before any semantic lowering. |
| 2026-07-16 | Gate 3 detached provenance authorization | focused profile/artifact/provenance Vitest; compiler typecheck/lint; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `git diff --check` | Passed: 3 focused files/38 tests; full compiler gate 38 files/1044 tests; semantic/compiler architecture, kernels/compiler builds, strict typecheck/lint, synthetic input, frozen normalizer, WebGPU fixtures, test scope, bugbash status, real-world CLI, tool lock, and corpus audit. Provenance coverage includes pinned deterministic identities, DSSE PAE bytes, valid P-256 authorization, trust-root/key/base64/P1363 failures, authenticated subject/toolchain/sandbox/runner mutations, canonical source/revision rules, caller pins, opaque substitution, VFS prefix escape, rejected artifacts, cancellation, and hostile accessors. | Commit provenance primitive. Next make the layout tracer consume only the authorized wrapper; hermetic producer and real external attestation remain unproved. |
| 2026-07-16 | Gate 3 allocation-free layout semantics | Semantic-core typecheck/lint/test/build; `pnpm architecture:check`; `pnpm test:release-packages`; `git diff --check` | Passed: strict typecheck/lint/build and 12 files/107 tests. The 8 new cases prove zero-storage topology, full-domain/static boundary traces, dynamic binding/domain failures, canonical set ordering, transport-neutral hashes, opaque authority, hostile input snapshots, and resource limits. Architecture passed; release tests passed 19 hostile-archive and 35 Node security tests plus all packed/fresh consumers; whitespace clean. | Implement the authorized compiler lowerer. |
| 2026-07-16 | Gate 3 authorized static layout lowering | focused artifact/provenance/layout Vitest; compiler typecheck/lint; semantic and compiler architecture checks; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `git diff --check` | Passed: 3 focused files/34 tests; full compiler gate 39 files/1053 tests; kernels/compiler builds, strict typecheck/lint, zero dependency cycles or architecture leaks, synthetic input, frozen normalizer, WebGPU fixtures, test scope, bugbash status, real-world CLI, tool lock, and corpus audit. Coverage includes exact and hierarchical mappings, signed CuTe cosize, bounded static algebra, caller-limit propagation, authorization/ownership ambiguity, no-storage topology, dynamic rejection, cancellation, hostile input, and instance authority. | Commit the internal static layout tracer. Next pin and implement the hermetic producer; no real-source or backend claim yet. |
| 2026-07-16 | Gate 3 pre-run AOT producer request | focused profile/job/artifact/provenance/layout Vitest; compiler typecheck/lint; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `git diff --check` | Passed: 5 focused files/53 tests; full compiler gate 40 files/1059 tests plus semantic/compiler architecture, kernels/compiler builds, strict typecheck/lint, synthetic input, frozen normalizer, WebGPU fixtures, scope/status/CLI/tool-lock/corpus gates. Six job tests pin request/job IDs and cover profile/header/hash drift, source VFS/main/anchor ownership, operational escape hatches, hostile accessors, authority copies, version, cancellation, and resource ceilings. | Commit request authority. Next bind exact canonical artifact bytes and an observed runner receipt before implementing the offline container shell. |
| 2026-07-16 | Gate 3 canonical artifact resource identity | focused profile/job/artifact/provenance/layout Vitest; compiler typecheck/lint; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `git diff --check` | Passed: 5 focused files/53 tests; full compiler gate 40 files/1059 tests plus semantic/compiler architecture, kernels/compiler builds, strict typecheck/lint, synthetic input, frozen normalizer, WebGPU fixtures, scope/status/CLI/tool-lock/corpus gates. Exact canonical byte SHA-256/length are pinned; noncanonical input and signed raw-digest, length, transport, semantic, output-manifest, and authority substitutions fail closed. | Commit raw resource identity. Next add an observed runner receipt and require it in provenance. |
| 2026-07-16 | Gate 3 AOT runner receipt contract | focused profile/job/receipt/artifact/provenance/layout Vitest; compiler typecheck/lint; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `git diff --check` | Passed: 6 focused files/60 tests; full compiler gate 41 files/1066 tests plus semantic/compiler architecture, kernels/compiler builds, strict typecheck/lint, synthetic input, frozen normalizer, WebGPU fixtures, scope/status/CLI/tool-lock/corpus gates. Receipt ID `a86dec16...`, raw SHA-256 `1226f130...`, byte length `3837`, and pre-run invocation `edaa38ee...` are pinned. Coverage includes strict byte-origin artifact/receipt authority, accepted/rejected separation, every extraction/process ceiling, independently recomputed artifact counts, snapshot TOCTOU resistance, and closed identity/policy/input/output mutations. | Commit receipt primitive. Next require its strict-decoded resource in detached provenance. |
| 2026-07-16 | Gate 3 receipt-authenticated provenance | focused job/receipt/artifact/provenance/layout Vitest; compiler typecheck/lint; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `git diff --check` | Passed: 5 focused files/47 tests; full compiler gate 41 files/1066 tests plus semantic/compiler architecture, kernels/compiler builds, strict typecheck/lint, synthetic input, frozen normalizer, WebGPU fixtures, scope/status/CLI/tool-lock/corpus gates. Detached provenance requires the strict-decoded receipt resource; signed receipt/job/invocation/output identities are mutation-tested; authorization derives the exact receipt-owned artifact/profile; rejected outcomes cannot lower; compiler origins retain receipt lineage without changing semantic hashes. | Commit the receipt-bound provenance slice. Next implement the offline fail-closed runner boundary; no real producer or external attestation claim yet. |
| 2026-07-16 | Gate 3 logical execution-plan identity | 7 focused policy/profile/job/artifact/receipt/provenance/layout Vitest files; compiler typecheck/lint; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `git diff --check` | Passed: 7 focused files/61 tests; full compiler gate 42 files/1067 tests plus semantic/compiler architecture, kernels/compiler builds, strict typecheck/lint, synthetic input, frozen normalizer, WebGPU fixtures, scope/status/CLI/tool-lock/corpus gates. Policy SHA `314957ce...` and plan SHA `18d13939...` are pinned. OCI config, execution environment, plan, corrected native-execution meaning, receipt/provenance mutation, checked-in policy equality, and origin lineage are covered. | Commit policy/schema hardening. Next implement opaque source snapshots and single bounded output frame before Docker process effects. |
| 2026-07-16 | Gate 3 opaque offline-runner I/O authority | 8 focused policy/profile/job/artifact/receipt/provenance/layout/runner Vitest files; compiler typecheck/lint; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `git diff --check` | Passed: focused 8 files/71 tests; full compiler gate 43 files/1077 tests plus semantic/compiler architecture, kernels/compiler builds, strict typecheck/lint, synthetic input, frozen normalizer, WebGPU fixtures, scope/status/CLI/tool-lock/corpus gates. Policy `e63d85ff...` and plan `bd107924...` are repinned. Coverage includes deep policy/decoder-budget identity, synchronous caller-buffer snapshots, descriptor-only blob closure, pre-copy membership/length checks, exact Uint8 semantics, digest/anchor recheck, shared/proxy/accessor/species rejection, accepted/rejected outcome preservation, fixed frame boundaries/limits, noncanonical artifact/receipt rejection, abort, and isolation of returned copies. | Commit the pure runner I/O boundary. Next implement and fake-process-test the fixed Docker shell without claiming a real producer. |
| 2026-07-16 | Gate 3 OCI manifest/config metadata authority | 9 focused policy/profile/job/artifact/receipt/provenance/layout/runner/OCI Vitest files; compiler typecheck/lint; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `git diff --check` | Passed: focused 9 files/81 tests; full compiler gate 44 files/1087 tests plus semantic/compiler architecture, kernels/compiler builds, strict typecheck/lint, synthetic input, frozen normalizer, WebGPU fixtures, scope/status/CLI/tool-lock/corpus gates. Policy `c8bf6bdc...` and plan `636f1c9f...` are pinned. Coverage includes cacheable metadata versus plan authorization, exact descriptor/rootfs closure, empty image config, no foreign layers, per-resource and aggregate budgets, strict JSON, impossible timestamps, hostile byte/options/cancellation ordering, deep immutability, and structural authority rejection. | Commit the pure metadata/authorization boundary. Next implement shell-owned live Docker observation and the fixed process state machine; no real local image or producer is claimed. |
| 2026-07-15 | Immutable release hardening integration gate | `pnpm test:release-packages`; `pnpm install --frozen-lockfile`; `pnpm architecture:check`; Ruby workflow YAML parse; `git diff --check` | Passed: complete packed consumers including fresh npm Grad and JIT installs, lifecycle/repository/workflow guards, 10 provenance tests, 12 tar tests, frozen pnpm `10.34.5` install, semantic architecture, workflow syntax, and whitespace. | Commit release infrastructure, then stage semantic-core from a separate clean worktree before marking the pipeline verified. |
| 2026-07-15 | Adversarial immutable-release exit gate | `pnpm install --frozen-lockfile`; `pnpm test:release-packages`; `pnpm architecture:check`; `node scripts/publish-missing-npm.mjs --dry-run`; Ruby workflow YAML parse; script syntax/compile checks; `git diff --check` | Passed: frozen pnpm `10.34.5`; 18 raw/semantic hostile-tar tests; 25 Node tests including 4 staged-manifest, 12 provenance, 6 dependency-order, 2 credential-boundary, and 1 tree-equivalence cases; all seven public tarballs and fresh runtime/primitives/Grad/JIT consumers; architecture and workflow syntax. Registry plan contains exactly the five missing versions in semantic-core → kernels → compiler/Grad/JIT order. No publication occurred. | Commit only owned release infrastructure, then stage one exact package from a separate clean worktree and record its manifest/SRI before marking this row verified. |
| 2026-07-15 | Immutable-release exit-audit closure | `pnpm install --frozen-lockfile && pnpm test:release-packages && pnpm architecture:check`; workflow YAML parse; `node scripts/publish-missing-npm.mjs --dry-run`; `git diff --check` | Passed: frozen pnpm `10.34.5`; 19 hostile tar tests; 35 Node release-security tests; all-seven packed/fresh-consumer suite; architecture and workflow syntax. Live dry-run plans all seven current versions in dependency-first order as missing after the required runtime `0.1.2` and primitives `0.1.1` bumps. No publication occurred. | Commit owned files, then stage an exact package from a separate clean worktree and record manifest/SRI evidence. |
| 2026-07-15 | Exact clean-commit staged-release proof | Detached worktree at `155161b7`; `pnpm install --frozen-lockfile`; `pnpm -r run build`; `node scripts/publish-missing-npm.mjs --stage-dir npm-release-artifacts --package @unlocalhosted/browsergrad-semantic-core`; packed identity inspection | Passed: manifest source revision `155161b7b36f691ca9935045af48488fffbab265`; exact target/closure semantic-core `0.2.0`; 124 files; tree `47fe98534b683c853d7d872189fdf2ba598394ab03b6a7cf64f88049fc7b2de1`; SRI `sha512-Eegpt07BC+R7/pJZeMpRzMK5P6pIHD43m2z/WNWUUplHXFKXRbW4ObMouWm4KKByTfvIOgYrG2elIf2HjCNgoA==`; worktree status contained only the two declared staging outputs. No publication occurred. | Release pipeline row is verified. Reconcile stale overlapping release instructions, then implement compiler structured padding. |
| 2026-07-16 | Compiler L2 focused behavior | `pnpm --filter @unlocalhosted/browsergrad-compiler exec vitest run tests/compiler/semantic_view_copy_bindings.test.ts tests/compiler/semantic_index_map_lowering.test.ts` | Passed: 2 files, 13 tests covering authority, raw-u32 structured padding, exact fill, offsets/canaries, reject policy, always-false zero reads, hostile runtime roots, and arithmetic intervals. | Add actual-WebGPU proof. |
| 2026-07-16 | Compiler L2 regression scope | `pnpm --filter @unlocalhosted/browsergrad-compiler exec vitest run tests/compiler/semantic_view_copy_bindings.test.ts tests/compiler/semantic_layout_bindings.test.ts tests/compiler/core.test.ts tests/compiler/control.test.ts` | Passed after localizing signed predicates: 4 files, 235 tests. | Re-run after final adversarial edits and as part of full compiler gate. |
| 2026-07-16 | Compiler L2 types and lint | `pnpm --filter @unlocalhosted/browsergrad-compiler typecheck && pnpm --filter @unlocalhosted/browsergrad-compiler lint` | Passed before final SharedArrayBuffer/return adversarial edits. | Re-run before commit; this row does not authorize the final diff. |
| 2026-07-16 | Compiler L2 full package gate | `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler` | Passed on the final unit-slice diff: repository architecture; kernels/compiler builds; compiler typecheck/lint; 32 files and 982 tests; compiler dependency/cycle/representation architecture; synthetic input, source normalizer, WebGPU fixtures, test scope, bugbash status, real-world CLI, tool lock, and corpus-audit harnesses. | Commit the CPU/WGSL slice; actual-WebGPU L2 evidence remains separate. |
| 2026-07-16 | Compiler L2 post-commit adversarial hardening | Focused view-copy/index-map Vitest plus compiler typecheck and whitespace check | Passed: 2 files, 14 tests; strict typecheck; clean diff whitespace. | Commit test hardening separately, then build required-device lane. |
| 2026-07-16 | Gate 3 local Docker runtime/image observation contract | Focused Gate 3 plus architecture Vitest; dedicated Docker-shell typecheck/lint/Vitest; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `pnpm test:release-packages`; `git diff --check` | Passed: focused 10 files/105 tests; Docker shell 1 file/76 tests; full compiler 44 files/1088 tests with build, strict typecheck/lint, semantic/compiler architecture, synthetic input, frozen normalizer, WebGPU fixtures, scope/status/CLI/tool-lock/corpus gates; release 19 hostile-tar and 35 Node security tests plus packed/fresh consumers; whitespace clean. | Commit the observation contract. Next implement the created-container lifecycle; no live daemon/image or producer claim. |
| 2026-07-16 | Gate 3 fail-closed Docker container lifecycle | Dedicated Node typecheck/oxlint/Vitest; focused policy/runner/OCI Vitest; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `pnpm test:release-packages`; checked-policy deep equality/JSON parse; `git diff --check` | Passed: private Docker suite 2 files/92 tests; focused 3 files/21 tests; full compiler 44 files/1088 tests with semantic architecture, kernels/compiler builds, strict typecheck/lint, compiler architecture, synthetic-input/normalizer/WebGPU-fixture/scope/status/CLI/tool-lock/corpus gates; release 19 hostile-tar and 35 Node security tests plus all packed/fresh consumers; policy JSON equals built policy; whitespace clean. | Commit the synthetic lifecycle. Next build and exercise the real pinned extractor/supervisor image and external attestation path; no live-execution claim. |
| 2026-07-16 | Gate 3 pinned seccomp request | Exact vendored-resource lock/hash/length test; focused environment/job/OCI/policy/receipt/plan/profile/provenance Vitest; dedicated Docker shell/lifecycle typecheck, oxlint, and Vitest; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `pnpm test:release-packages`; `git diff --check` | Passed: exact Moby source 13,470 bytes and SHA-256 `536529b6...`; focused 8 files/70 tests; Docker 2 files/96 tests; full compiler 45 files/1100 tests with both architecture checks, kernels/compiler builds, strict typecheck/lint, synthetic-input/normalizer/WebGPU-fixture/scope/status/CLI/tool-lock/corpus gates; release 19 hostile-tar and 35 Node security tests plus packed/fresh consumers. Policy 1.3 SHA `14052e2d...`; plan SHA `684616ab...`. No live-enforcement claim. | Commit the checked seccomp request. Next correct producer-facing schema gaps before native extractor/supervisor implementation. |
| 2026-07-16 | Gate 3 producer schema v2 hardening | Compiler typecheck/lint; explicit 10-file schema-v2 CuTe Vitest; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `pnpm test:release-packages`; `git diff --check` | Passed: typecheck/lint; 10 files/108 schema-v2 tests; full current-worktree compiler gate 46 files/1126 tests plus private Docker 2 files/96 tests, both architecture checks, kernels/compiler builds, synthetic input, frozen normalizer, WebGPU fixtures, scope/status/CLI/tool-lock/corpus gates; release 19 hostile-tar and 35 Node security tests plus packed/fresh consumers. Full gate also saw 11 uncommitted browser-asset next-slice tests; they are not counted as schema-v2 evidence. | Commit only schema-v2 and ledger files. Browser deployment profile, common producer authorization, worker, real assets, and browser execution remain next. |
| 2026-07-16 | Gate 3 browser deployment and asset authority | Focused profile/asset Vitest; compiler typecheck/lint; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `pnpm test:release-packages`; `git diff --check` | Passed: focused 2 files/36 tests; full compiler 46 files/1134 tests plus private optional-Docker 2 files/96 tests, both architecture checks, kernels/compiler builds, strict typecheck/lint, synthetic-input/frozen-normalizer/WebGPU-fixture/scope/status/CLI/tool-lock/corpus gates; release 19 hostile-tar and 35 Node security tests plus packed/fresh consumers; whitespace clean. Coverage includes AOT/browser mode narrowing, producer-neutral compilation identity, monolithic compiler/extractor WASM binding, package-owned worker policy, profile-owned ceilings, exact canonical asset bytes, profile/ABI/asset-set/build/mount/content/total closure, same-origin URLs, hostile objects/bytes, cancellation, and common producer authorization. All asset/profile hashes are fixtures; no fetch, unpack, worker, or browser execution occurred. | Commit the contract slice. Next implement safe asset acquisition/unpack authority and the closed browser worker request/result evidence that alone may mint browser lowering authorization. |
| 2026-07-16 | Gate 3 closed VFS pack and manifest binding | Focused profile/asset/VFS Vitest; compiler typecheck/lint; independent read-only protocol/code review; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `pnpm test:release-packages`; `git diff --check` | Passed: focused 3 files/44 tests; full compiler 47 files/1142 tests plus private optional-Docker 2 files/96 tests, both architecture checks, kernels/compiler builds, strict typecheck/lint, synthetic-input/frozen-normalizer/WebGPU-fixture/scope/status/CLI/tool-lock/corpus gates; release 19 hostile-archive and 35 Node security tests plus packed/fresh consumers; whitespace clean. Coverage includes fixed binary framing, portable-ASCII paths, all-prefix file/directory collisions, exact region/hash closure, input snapshots, hostile/shared/accessor inputs, realistic independent limits, identity versus file-content accounting, structural-copy rejection, and real pack bytes bound to one exact prepared manifest asset/include-root/mount. No fetch, cache, installation, worker, or browser execution occurred. | Commit the closed-pack slice. Next replace AOT/test-oracle source assumptions with the producer-neutral request and distinct browser compiler-runtime ABI before implementing worker evidence. |
| 2026-07-16 | Gate 3 common request and browser compiler-runtime ABI | Focused request/profile/runtime/asset Vitest; compiler typecheck/lint; parallel adversarial-test and build/distribution audits; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `pnpm test:release-packages`; `git diff --check` | Passed: focused 4 files/53 tests; full compiler 49 files/1159 tests plus private optional-Docker 2 files/96 tests, both architecture checks, kernels/compiler builds, strict typecheck/lint, synthetic-input/frozen-normalizer/WebGPU-fixture/scope/status/CLI/tool-lock/corpus gates; release 19 hostile-archive and 35 Node security tests plus packed/fresh consumers; whitespace clean. Request tests cover producer-neutral identity, copied pre-bounded snapshots, both entry families, semantic-only narrowed limits, detached records, source/anchor/hash closure, accessors/proxies/shared bytes, opaque authority, cancellation, and decode budgets. Runtime tests cover host-verified no-fetch handoff, exact Worker bytes/Blob/no-network policy, closed WASM features, ownership/sharing, page/stack/working/VFS/output coexistence, asset ceilings, and hostile mutations. No asset fetch, mount, worker spawn, Clang-WASM build, or browser-local producer execution occurred. | Commit this boundary. Next close CUDA host/device pass domains and build-provenance/license authority, then compose optional AOT metadata around the common request. |
| 2026-07-16 | Gate 3 CUDA device/host semantic-pass contract | Focused artifact/profile/AOT-job/provenance/lowering/receipt Vitest; compiler typecheck/lint; independent P0/P1 review; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; serial `pnpm test:release-packages`; `git diff --check` | Passed: focused 6 files/89 tests; full compiler 53 files/1189 tests plus private optional-Docker 2 files/96 tests, both architecture checks, kernels/compiler builds, strict typecheck/lint, synthetic-input/frozen-normalizer/WebGPU-fixture/scope/status/CLI/tool-lock/corpus gates; release 19 hostile-archive and 35 Node security tests plus packed/fresh consumers; final review found no remaining P0/P1 in this slice. Coverage includes exact separate device-only/host-only profiles, device-canonical graph ownership, payload-level content-derived source identity, exact nonempty selected-root convergence, split-identity rejection, source-keyed host ABI, per-pass target and observed VFS closure, target-conditional closure divergence, mandatory forced includes, exact diagnostic/note/fact ownership, unresolved-edge ownership, shared-surface convergence, device-first fail-stop state matrix, bounded pass references, and producer/profile/asset identity propagation. All identities remain synthetic; this is a contract/verifier result, not a Clang-WASM execution. | Commit this slice, then review producer-neutral AOT metadata and the exact build recipe. |
| 2026-07-16 | Gate 3 canonical VFS-pack writer | `corepack pnpm --filter @unlocalhosted/browsergrad-compiler typecheck`; focused VFS reader/writer Vitest; focused source oxlint; independent adversarial review; `git diff --check` | Passed before commit: strict typecheck, 2 files/15 tests, lint, and whitespace. Coverage includes deterministic caller-order independence, canonical round trip, immediate per-file snapshot against same-length mutation and resizable-buffer shrink during hostile later-record inspection, path pre-allocation ceiling, duplicate/file-parent rejection, sparse/accessor/proxy/shared input rejection, independent file/content/index/pack ceilings, and cancellation. No real toolchain/header pack was produced. | Land writer as a separate coherent chunk. Fetch/cache/mount authority and real release packs remain pending. |
| 2026-07-17 | Gate 3 producer-neutral AOT composition | Focused 12-file artifact/request/run-metadata/request-binding/runner/receipt/provenance/lowering Vitest; strict compiler typecheck; independent authority and terminology reviews; `corepack pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; serial `corepack pnpm test:release-packages`; legacy-authority/stale-term searches; `git diff --check` | Passed: focused 12 files/121 tests; compatibility regression plus run-metadata suite 2 files/183 tests; final full current-worktree compiler gate 52 files/1167 tests; optional Docker shell/lifecycle 2 files/96 tests; both architecture checks; kernels/compiler builds; strict typecheck/lint; synthetic-input/frozen-normalizer/WebGPU-fixture/scope/status/CLI/tool-lock/corpus gates; release 19 hostile-archive and 35 Node security tests plus packed/fresh consumers. Final review found no P0; four P1 naming/documentation/version-axis gaps were fixed. Full gate also saw one uncommitted browser-asset-installation next-slice file; it is not counted as AOT evidence. | Commit the producer-neutral AOT chain. Then review and land verified browser asset installation separately. No live Docker, native extractor, Clang-WASM, worker, or browser C++ execution is claimed. |
| 2026-07-17 | Gate 3 browser asset acquisition and VFS installation | `corepack pnpm --filter @unlocalhosted/browsergrad-compiler typecheck`; focused asset/manifest/VFS Vitest; focused oxlint; two independent P0/P1 reviews; `corepack pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `corepack pnpm test:release-packages`; `git diff --check` | Passed: focused 3 files/30 tests; full compiler 52 files/1172 tests; optional Docker 2 files/96 tests; both architecture checks; kernels/compiler builds; strict typecheck/lint; synthetic-input/frozen-normalizer/WebGPU-fixture/scope/status/CLI/tool-lock/corpus gates; release 19 hostile-archive and 35 Node security tests plus packed/fresh consumers; whitespace clean. Coverage includes exact same-origin redirect-free streamed fetch, length/hash verification, cache copy/rehash/admission, late-response/body cleanup, hostile platform/adaptor properties, cancellation precedence, exact manifest-pack rebinding, collision-free installation, opaque authorities, and both retained pack copies. Final review found no P0/P1. No real asset, Worker, Clang-WASM, Docker, or browser execution occurred. | Commit acquisition/install authority. Next close exact build/provenance/license recipe. Aggregate opened-file accounting remains Worker-session-owned. |
| 2026-07-17 | Gate 3 Clang-WASM input lock and deterministic planner | Compiler typecheck; focused build-lock Vitest/oxlint; `corepack pnpm --filter @unlocalhosted/browsergrad-compiler test:browser-clang-wasm-build-plan`; `corepack pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `corepack pnpm test:release-packages`; `git diff --check` | Passed after fixes: strict compiler typecheck; build-lock 1 file/21 tests; planner typecheck/oxlint and 1 file/25 tests using the real opaque decoded lock; full compiler 53 files/1193 tests; optional Docker 2 files/96 tests; both architecture checks; kernels/compiler builds; synthetic-input/frozen-normalizer/WebGPU-fixture/scope/status/CLI/tool-lock/corpus gates; release 19 hostile-archive and 35 Node security tests plus packed/fresh consumers; whitespace clean. Initial review found three P1s—structural lock forgery, link-map output leakage, and unsafe path interpolation—which are fixed; final independent review found no remaining P0/P1. The lock stays release-blocked and no Docker process, Clang-WASM build, asset, Worker, or browser execution occurred. | Commit coherent input-only slice. Then define canonical runtime ABI and Worker session contracts. |
| 2026-07-17 | Gate 3 canonical runtime ABI and profile binding | Runtime ABI/profile/asset focused Vitest; compiler typecheck/oxlint; independent P0/P1 review; `corepack pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `corepack pnpm test:release-packages`; stale profile/hash searches; `git diff --check` | Passed after review fixes: ABI 1 file/34 tests; profile plus ABI 3 files/70 tests; asset manifest repin 1 file/13 tests; full compiler 54 files/1,232 tests; optional Docker 2 files/96 tests; planner 1 file/25 tests; both architecture checks, kernels/compiler builds, strict typecheck/lint, synthetic-input/frozen-normalizer/WebGPU-fixture/scope/status/CLI/tool-lock/corpus gates; release 19 hostile-archive and 35 Node security tests plus packed/fresh consumers. Review found one P0 extension-closure gap and three P1 memory-contract gaps; all are fixed. No stale 2.2 profile, old interface field, ambiguous per-file opened budget, or replaced pins remain. No build, Docker process, Worker, or C++ execution occurred. | Commit the design/profile boundary. Next bind the ABI bytes into asset/build closure and implement raw-WASM conformance before Worker execution. |
| 2026-07-17 | Gate 3 ABI asset/build binding and raw-WASM inspection | Strict compiler typecheck; focused ABI/assets/acquisition/build-lock/inspector Vitest; focused oxlint; `test:browser-clang-wasm-build-plan`; official WebAssembly binary-format and tool-conventions review; independent P0/P1 adversarial review; `git diff --check` | Passed: 5 files/111 tests, typecheck, lint, deterministic planner 1 file/25 tests, and whitespace. Review found no P0 and corrected padded-LEB validity, UTF-8 names, target-feature vocabulary/order, module validation, extension/opcode classification, memarg/table-copy decoding, collision-free function inventories, and exact non-function export projection. The current empty first-build ABI projections prevent positive production conformance. No build, Docker process, Worker, or C++ execution occurred. | Commit the asset/conformance boundary. Run the inspector inside a disposable verifier Worker in the actual host adapter; build and independently repin the first real module before any production conformance or release claim. |
| 2026-07-17 | Gate 3 Worker/VFS protocol and profile 2.4 | Focused ABI/profile/assets/acquisition/build-lock/VFS/inspector/Worker Vitest; compiler typecheck/lint; two independent P0/P1 reviews; `corepack pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `corepack pnpm test:release-packages`; `git diff --check` | Passed after review fixes: focused 9 files/167 tests; full compiler 57 files/1,286 tests; optional Docker 2 files/96 tests; Clang-Wasm planner 1 file/25 tests; both architecture checks, kernels/compiler builds, strict typecheck/lint, and all synthetic/corpus guards; release 19 hostile-archive and 35 Node security tests plus packed/fresh consumers. Review removed caller self-attestation, false termination/timing and Wasm-residency claims, counter drift/impossible arithmetic, after-copy limits, heavy terminal retention, prototype-sensitive node classification, overlong mounted paths, unbounded VFS index expansion, and final logical-live-open coupling to Wasm linear memory. Exact ABI/profile/build-lock/asset identities were repinned. Final independent review found no P0/P1. Positive prepared Worker lifecycle remains deliberately unavailable while first-build Wasm projections are empty. No Worker, Clang-Wasm, C++, Docker, or WebGPU execution occurred. | Commit. Next implement the actual Worker module/controller; pure frame validation is not browser evidence. |

## Failure and Recovery Log

Record failures that may matter after context loss. Include the exact failing
command, concise error, suspected cause, resolution or next experiment, and
whether any files may be left partially changed.

- The first post-profile-2.4 focused run
  (`vitest` over ABI/profile/assets/acquisition/build-lock/VFS/inspector/Worker)
  passed 145 of 147 tests. Both failures were stale exact expectations after
  the intentional repin: browser asset `manifestSha256` expected
  `377626e9...` but derived `24389174...`, and the overlong mounted-path test
  expected an aggregate path while the implementation correctly reported
  `$.installation.files[0].virtualPath`. No functional assertion failed. Both
  expectations were corrected. A later authority-ownership repin produced a
  second stale asset identity during an overlapping test run; the settled
  9-file suite now passes all 167 tests. No partial generated file or external
  state was involved.
- Final VFS review found that the newly renamed logical live-open quota was
  still located under `wasm.memory` and added to linear-memory coexistence
  arithmetic. That remained semantically false for lazy range reads. The quota
  now belongs to `body.vfs`, carries the exact accounting declaration
  `logical-full-file-per-live-handle-reservation-not-wasm-residency`, and is
  absent from ABI/profile Wasm-memory sums. All identities were repinned;
  final review found no remaining P0/P1.

- Raw-WASM review initially rejected spec-valid padded LEB encodings, assumed
  sorted `target_features`, restricted every Wasm name to ASCII, and missed
  proposal-specific memarg/table/bulk-memory forms. Independent review also
  found delimiter-colliding function inventory keys and one-way non-function
  export checks. All are fixed with standards-valid decoding, intrinsic module
  validation, structured keys, enriched exact projections, and 27 focused
  adversarial tests. Synchronous engine validation remains non-preemptible;
  the actual host must run large/untrusted inspection in a disposable verifier
  Worker. No partial inspector finding remains in the current code.
- Profile 2.3 and the final reviewed ABI policy each changed the browser profile
  hash, so deterministic browser-asset fixture assertions twice exposed stale
  manifest ID/raw-SHA pins. The asset-set hash and byte length remained stable.
  Each time, repinned only derived identities, reran the focused 13-test asset
  suite, then reran the complete compiler gate successfully. No runtime code or
  asset semantics changed.
- Raw-WASM design review found a P0 circular-authority gap: observed Emscripten
  imports/support exports could have defined their own allowed surface, and
  table/global/custom-section policy was absent. Runtime ABI v1 now keeps those
  allowlists empty, independently hash-pinned, and release-blocked; observation
  cannot extend them. The first real build still requires bounded raw parsing,
  independent review, and an explicit manifest repin before release.
- Final independent review then found one P0 and three P1 contract gaps: no
  reject-all rule for unlisted Wasm extensions, coexistence arithmetic omitted
  the live input frame, VFS pointer/range/alias safety was implicit, and the
  opened-VFS budget read as per-file rather than aggregate. All four are fixed
  in the ABI/profile and hostile tests. Final focused, compiler, and release
  gates pass; no partial files or unresolved finding remain in this slice.

- Extractor-design review originally found five P0 gaps after the input-lock
  slice. This slice resolves the canonical runtime-ABI design contract. Four
  implementation/evidence gaps remain: real unannotated CuTe layout `VarDecl`
  roots conflict with artifact-v3 CUDA-attribute admission; Worker invocation/
  result/evidence authorities do not exist; host-backed lazy VFS still needs its
  Worker-owned transfer/session implementation across JavaScript-retained packs
  and WASM-opened copies; and declared preprocessing, template, constexpr, AST,
  and working-memory ceilings lack pinned Clang hooks or must be narrowed. P1
  follow-ups separate wall time from unavailable browser
  CPU/process metrics, reconcile profile/request maxima with decoder caps,
  separate logical opened bytes from allocator-resident bytes, and validate
  primary/auxiliary target data before parsing. These are next-slice work, not
  claims hidden by the green input-lock tests.

- First post-review full compiler rerun reached the final corpus audit, then
  transiently reported `ERR_MODULE_NOT_FOUND` for compiler `dist/index.js`
  while shared-workspace agents were still rebuilding the same package. The
  focused corpus audit passed immediately; after all writing agents completed,
  a fully serial `verify:compiler` passed 53 files/1193 tests and every trailing
  gate, followed by the release-package gate. Treat this as a shared-worktree
  verification-order hazard, not a product failure: do not run final build/
  clean gates concurrently with agent-owned package builds.

- Initial browser asset review found seven P1 boundary bugs: cancellation
  awaited hostile stream cleanup; terminal response bodies were not always
  cancelled; cache admission could mint after abort; pack-verifier
  cancellation became `PACK_INVALID`; hostile response/cache property access
  leaked raw errors or changed across reads; shadowed signal methods could run;
  and retained-pack accounting counted one of two resident copies. Fixes use
  captured platform intrinsics, nonblocking cleanup, post-await abort checks,
  exact cancellation mapping, one-time adapter snapshots, and explicit source,
  verified, and total pack-byte counters.
- Re-review found late fetch fulfillment could retain a response body and
  hostile/nonthrowing response brand traps could misclassify cancellation.
  Added late-fulfillment disposal plus abort precedence and body cleanup after
  every trap-capable brand boundary. A proposed per-file installed-VFS copy API
  was removed: it used overridable `Uint8Array.prototype.slice` and could not
  enforce the profile's aggregate WASM opened-file reservation. D-075 assigns
  file copies, transfer/disposal, aggregate metering, and cancellation to the
  future Worker execution session. Final independent review and all gates pass.

- Producer-neutral AOT migration changed profile, environment, artifact, and
  transitive browser-asset identities. Early focused/full runs found only stale
  deterministic pins: AOT environment/profile hashes, browser compilation-
  contract hash, then browser asset manifest ID/hash. Each observed value was
  re-derived through production constructors, reviewed against the intended
  schema change, repinned, and rerun. Final compiler gate passes 52 files/1167
  tests plus 96 Docker tests; no external artifact or runtime state was changed.
- Independent final AOT review found four P1 consistency gaps after the first
  green gate: predicate URI remained `/v2`; caller-declared repository/revision
  was still named source provenance; common authorization advertised a browser
  evidence kind without a mint path; and later terminology review found the
  normative LLD, public compatibility registry, changelog, and exported
  predicate type/version axes still described deleted or v1 job/provenance
  contracts. Fixed predicate/build `/v3`, renamed declared-source fields,
  removed the nonexistent evidence kind, replaced public AOT-job diagnostics
  with current run-metadata codes, corrected LLD/changelog text, and separated
  trust-store v1 from BrowserGrad predicate v3 type names. Final P0 review,
  compiler gate, release-package gate, and stale-term searches are green.

- Initial VFS-writer review found two P1 bugs: paths were UTF-8 encoded before
  their length ceiling, and byte views were inspected in one pass but copied
  only after every caller record had been touched. A first correction rechecked
  only byte length before deferred copy; parent review found same-length
  reentrant mutation remained possible. Final writer now bounds string length
  before encoding and copies each file immediately after intrinsic inspection.
  Hostile same-length mutation and resizable-buffer shrink tests prove later
  record inspection cannot change earlier snapshots. Focused tests/typecheck/
  lint are green. Final review also found cumulative index/content/pack and
  path-collision checks occurred after per-file copy; these now use prospective
  totals and incremental path sets before each snapshot allocation. Hostile
  duplicate/collision records prove byte inspection is never reached. No
  external state or real pack was produced.

- Independent semantic-pass review found the initial draft silently merged
  host/device target-resolved AST state, assigned the accepted layout to the
  host pass, forced one observed VFS closure onto both passes, allowed invalid
  status combinations, and described `sm_80` as a target rather than an
  architecture. Replaced it with a device-owned canonical graph, source-keyed
  host ABI, separate target/auxiliary triples and invocation modes, per-pass
  closures, exact ownership, and a device-first fail-stop matrix. Focused
  verification is green; no partial external state.
- The first downstream semantic-pass run failed six deterministic fixture
  pins plus one rejected-provenance fixture that did not mark the host pass
  failed. Re-derived every changed profile/job/asset/artifact/receipt identity
  through production constructors, made rejected host validation own its
  blocking compiler diagnostic, and reran the 6-file/92-test suite green.
  No runtime or external artifact was produced.
- The first full compiler gate found three remaining optional-AOT transitive
  pins/state fixtures: execution-plan and invocation IDs changed, and the AOT
  receipt rejection case still left host validation successful. Re-derived
  the two IDs, assigned the blocking compiler diagnostic to a failed host
  pass, and reran the full 53-file/1188-test compiler gate green. A concurrently
  started release-pack check raced the compiler build's `dist` cleanup and
  failed to import `dist/index.js`; the required serial rerun passed completely.
- Final independent review found three P1 holes after the initial green gate:
  device function returns were not cross-checked against the resolved function
  type, host diagnostics could name device-only graph IDs, and syntax-shaped
  source IDs plus producer-declared `shared` flags could hide correspondence.
  Added return-type equality, host diagnostic subject restrictions, and a
  canonical source-entity registry whose IDs derive from resolved content/range
  origin and whose domains exactly equal ABI ownership. Added hostile tests for
  forged identity, hidden sharedness, wrong return ABI, and host graph leakage.
- Follow-up review then reproduced one remaining split-identity bypass: ABI-only
  domain evidence let a producer replace one shared selected root with separate
  device-only and fabricated host-only identities and converge on empty shared
  surfaces. Moved the registry to payload scope, added exact per-pass selected-
  root IDs derived from canonical device declarations, derived domains from
  pass observations plus ABI ownership, included roots in surface hashes, and
  added the exact hostile mutation. Final independent P0/P1 recheck found no
  remaining issue in this slice. Full compiler and release gates passed.

- First full compiler run for request/runtime ABI failed only the pinned
  optional-AOT execution-plan assertion: expected `9f649e1e...`, observed
  `96970fb6...`. Removing the false profile-wide opened-header field changed
  the full AOT profile and transitive plan identity as intended. Re-pinned that
  deterministic test value, reran all 49 compiler files/1159 tests and the
  complete compiler/release gates successfully. No partial external state.
- Initial vendoring added one trailing LF, producing 13,471 bytes and SHA-256
  `de1f5327...` instead of the upstream locked 13,470-byte `536529b6...`
  resource. Removed only the added byte, then rechecked exact length/hash before
  minting the provenance lock. Policy drift intentionally changed transitive
  profile/job/invocation/receipt/artifact identities; production constructors
  regenerated every pin before focused, compiler, and release gates passed.
- The first lifecycle run could not remove its private root after a successful
  container simulation: source/control directories were intentionally mode
  0555, so recursive unlink lacked parent write permission. Lifecycle cleanup
  now reopens every known staging directory with `O_DIRECTORY|O_NOFOLLOW`,
  verifies the FD is a directory, restores owner-only mode 0700 only after
  container absence, and then lets the session remove/prove the root absent.
- Exact Moby 29.6.1 source review corrected a false inspect expectation:
  long-form `--mount ...,readonly` records top-level `Mounts[].Mode` as `""`,
  while read-only authority is `HostConfig.Mounts[].ReadOnly=true` plus
  `Mounts[].RW=false`. The verifier and fake projection now match that contract.
- The first full compiler gate after policy 1.1 failed four pinned profile/job/
  invocation/receipt/artifact identity assertions. The policy was already
  stable and checked JSON-equal; all transitive fixture identities were
  regenerated through production constructors, patched explicitly, and the
  complete 44-file/1088-test compiler gate then passed.
- Final audit found that valid profiles may set `maxMemoryBytes` below four;
  plain `floor(memory/4)` then emitted tmpfs `size=0`, which is not a bounded
  one-byte minimum. Policy, request construction, projection verification, and
  a minimum-memory regression now use
  `max(1,min(floor(maxMemoryBytes/4),536870912))`; transitive identities were
  repinned again only after that correction stabilized.

- The first focused rerun used an unquoted package-relative glob from the
  workspace root; zsh rejected it before pnpm changed directories. The explicit
  ten-file command then passed 105 tests. No files were affected.
- Initial Docker-shell TypeScript checks exposed missing Node-only declarations
  and stale transitive fixture hashes after the policy gained runtime probes.
  Added colocated `.d.mts` declarations, a Node-only ES2022 project, and
  repinned the full profile/job/invocation/receipt/provenance chain only after
  final policy/plan identities stabilized.
- A first generic Docker-API reading treated `ImageInspect.Id` as the config
  digest. Exact Docker 29.6.1 containerd-store source shows a platform-selected
  inspect uses the selected manifest descriptor. The shell now proves the exact
  client, engine, API compatibility, containerd store, and platform before
  accepting manifest-ID semantics; other runtime/store tuples fail closed.
- The first bounded-child termination draft called `unref()` and rejected after
  kill grace, allowing caller cleanup while the child or descendants might
  remain live. Settlement now waits for `close`; Linux termination targets the
  process group with child fallback; failure to reap after grace fail-stops via
  `process.abort()`.
- Runtime and image-store assumptions were initially policy comments rather
  than observations. Dedicated fixed `docker version` and `docker info` probes
  now precede image inspection and strict-decode exact closed projections.
- An intermediate policy hash was computed before all runtime fields and
  templates were finalized. The checked JSON, TypeScript policy, fixtures, and
  transitive identities are now pinned to policy `b8c08d81...` and plan
  `3dd711a8...`; focused equality and lineage tests catch future stale pins.
- First OCI draft accepted raw manifest/config bytes and caller-supplied full
  `docker image inspect` JSON in one constructor. Security review classified
  this P0: fabricated JSON could mint a falsely named local-image authority and
  static cached state could outlive daemon presence. Removed the combined
  constructor. Current chain is metadata verification -> plan authorization;
  only the pending Node process state machine may mint live observation.
- OCI over-limit fixture initially failed in strict decode at the document path
  because `maxStringBytes` is aggregate and was set to one annotation-value
  cap. Raised only the manifest aggregate decoder budget to 262,144 bytes while
  retaining smaller per-annotation semantic caps; the semantic 256-layer limit
  now produces the stable resource error.
- `Date.parse("2026-02-30T00:00:00Z")` normalized the impossible date instead
  of rejecting it. Replaced host parsing with explicit Gregorian month/day,
  time, and offset validation; both config and history timestamp paths share it.
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
- The first standalone-layout typecheck found an `Array.map` callback signature
  mismatch and insufficient narrowing before `Object.keys`. Both were local
  compile-time defects; explicit callbacks/narrowing fixed them before tests,
  and no partially verified artifact or caller contract changed.
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
- The first compiler L2 retained verifier trusted the record's own case and
  hash declarations, so a minimal forged pass could self-seal. It now rebuilds
  the exact expected manifest from checked-out fixtures and built public APIs,
  compares every closed field, and rejects the forged record in mutation tests.
- Compiler evidence originally used `submittedPipelineCount` for a count known
  only from preparation. The schema and suite were version-bumped and now call
  preparation-derived topology `planned`; only runtime dispatch traces may use
  submitted terminology.
- Root-level evidence tests import semantic-core public APIs directly. The first
  clean install exposed that the root test harness lacked a declared dependency;
  adding the exact workspace dev dependency and frozen-lock update made clean
  detached-worktree verification reproducible.
- Gate 3 audit found the structural verifier computed `cosize` as affine
  address span, which is not CuTe's definition and rejects valid signed-stride
  layouts. It now checks `layout(size(layout) - 1) + 1`; the signed fixture maps
  `[0,3,-1,2,-2,1]` with `cosize == 2` and creates no storage claim.
- Final layout-lowerer review found derived size/cosize recomputation used
  unbounded BigInt arithmetic and artifact verification did not propagate the
  caller's integer limit into static expression checks. Both paths now use the
  shared bounded dimension evaluator with exact caller limits; focused
  `maxIntegerBits` and node-budget tests fail closed before authority is minted.
- Producer feasibility audit found no tracked native extractor, Dockerfile,
  OCI/toolchain lock, sandbox policy, real CuTe fixture/header closure,
  production trust store, or signing workflow. Local Docker has no running
  daemon and the fixture hashes/workflow identity are intentionally synthetic.
  The request/runner contracts may be implemented and tested offline, but the
  real producer remains `in-progress` until exact external assets are acquired,
  built, and evidenced; synthetic fixtures cannot satisfy Gate 3.
- Provenance audit found the in-toto subject bound semantic identity rather than
  SHA-256 of exact emitted canonical bytes, and signed declared sandbox facts
  without an observed runner receipt. Canonical-byte authority and the
  receipt-authenticated provenance contract are now fixed and mutation-tested.
  Real producer evidence remains blocked until an external runner observes the
  same snapshotted inputs and outputs and an external attestor signs that exact
  strict-decoded receipt; synthetic receipts do not satisfy Gate 3.
- Receipt review caught two authority/meaning gaps before commit: object-level
  artifact verification could synthesize canonical bytes without proving byte
  origin, and receipt success required semantic acceptance. The implementation
  now has strict-decoder-only artifact/receipt resource wrappers and a closed
  resolved/rejected selection union. The same review expanded four process
  metrics into every profile ceiling and added independent artifact counters.
- Offline-runner security review found the receipt could authenticate policy
  declarations without binding deterministic effective execution, the profile
  omitted OCI config/runtime-environment identities, and the blanket native
  execution ban contradicted a native compiler. The logical plan/config/runtime
  chain and corrected user-produced-native-execution meaning now fail closed;
  actual Docker/container/cgroup observations remain unproved.
- Runner-I/O review found the first opaque-plan draft exposed its private source
  snapshots and decoded result arrays through unwrap records, so callers could
  mutate bytes after verification. Unwrap records now expose metadata and prior
  authorities only; explicit accessors return fresh non-authoritative copies,
  with mutation tests. The review also added a direct declaration-anchor recheck
  and guards SharedArrayBuffer use without assuming the constructor exists.
- Adversarial re-review found the policy was only shallow-frozen, decoder limits
  were ambient, a private IPC namespace implied undeclared writable shared
  memory, and `environment: []` did not prohibit image-config environment. It
  also found descriptor validation reopened ordinary getters, oversized blobs
  were copied before length checks, and prototype-swapped word arrays could pass
  a prototype-only byte check. Deep policy/plan identity, pinned decode budgets,
  no IPC, zero effective environment, descriptor-only snapshots, pre-copy checks,
  and `%TypedArray%` element-kind verification now cover these gaps.

### 2026-07-16 — Gate 3 OCI manifest/config metadata authority

- Added cacheable opaque verification for exact raw OCI manifest/config bytes,
  then a separate opaque authorization that binds the metadata to one prepared
  execution plan. Identical verified metadata can be reused across plans
  without conflating job/profile lineage.
- Closed the accepted image subset to one self-contained OCI leaf manifest,
  `linux/amd64`, distributable layers, exact config descriptor and rootfs
  diff-ID correspondence, bounded annotations/history, and absent/null/empty
  image execution config. Layer blob bytes are not supplied or claimed.
- Pinned manifest/config decode budgets, layer count/size and aggregate bytes,
  exact local Docker endpoint, control/source paths, and tmpfs mode into the
  canonical policy. Current policy SHA is `c8bf6bdc...`; current logical plan
  SHA is `636f1c9f...`.
- Security review rejected the initial combined constructor because
  caller-provided inspect JSON could fabricate local-image state. Live Docker
  observation is now a distinct pending shell-owned authority.
- Added explicit Gregorian RFC 3339 validation after review found `Date.parse`
  normalized impossible dates. Pre-aborted verification now returns before
  touching or copying up to nine MiB of evidence.
- Repinned profile/job/invocation/receipt/artifact lineage because policy
  identity is intentionally transitive. No real Docker, layer acquisition,
  container enforcement, extractor, receipt, or producer trust is claimed.

### 2026-07-16 — Gate 3 local Docker runtime/image observation contract

- Added a private Node-only shell and bounded-child primitive. Production code
  can mint a temporal observation only from three shell-owned probes in fixed
  order: Docker version, daemon/store info, then platform-selected image
  inspection. Caller-provided process output cannot mint production authority.
- Pinned the exact Docker client/engine/API range, containerd image store, Unix
  socket, platform, command templates, environment, timeouts, byte ceilings,
  and strict one-line JSON projections into the checked policy. Current policy
  SHA is `b8c08d81712865a1e7ad1b2e1abc4ea043bc9692291ffc71a8e6e5d8c3ea2816`;
  current logical plan SHA is
  `3dd711a8ed71e1f068fb0d5e2c37930e378afe2986f96467d334ee31633c5ede`.
- Bound the selected manifest ID, descriptor/config lineage, repo digest,
  platform, diff IDs, and semantically empty image configuration back to the
  prepared plan's authorized exact OCI metadata.
- Made child settlement close-based and cleanup fail closed. Timeout, abort,
  and output overflow terminate the Linux process group; an unreaped child
  after the fixed grace period aborts the parent rather than returning while
  effects may remain. Observation authority is minted only after the private
  mode-0700 run root is proven removed.
- Kept the Node shell out of browser builds and packed compiler artifacts.
  Dedicated TypeScript/lint/Vitest configuration uses no DOM ambient types;
  architecture and release guards reject Node/Docker leakage into the public
  compiler surface.
- Added 76 adversarial fake-adapter and real-child-process tests. They prove the
  contract, cleanup, and issuer separation only. No live daemon/image,
  container execution, Docker binary provenance, native extractor, real
  receipt, or producer trust is claimed.

### 2026-07-16 — Gate 3 fail-closed Docker container lifecycle

- Extended the private observation session instead of creating a second Docker
  authority path. One session now stages inputs and performs fixed create,
  created-inspect, attached-start, terminal-inspect, remove, and absence-proof
  transitions before its run root is removed and the result frame is decoded.
- Derived every Docker-affecting value from policy plus the authorized plan:
  digest-qualified image, random 128-bit name/session label, exact job/plan
  labels, numeric user, hostname, entrypoint/argv, namespaces, capability and
  privilege settings, limits, tmpfs, two read-only rprivate binds, runtime,
  restart, health, and logging policy. Pulling remains forbidden.
- Bound create stdout and the private CID file to one full container ID. An
  ambiguous create recovers only an exact random name with all session/job/plan
  labels. Missing, malformed, or forged recovery evidence preserves staging
  and fails stop; cleanup never removes by name.
- Verified the daemon-recorded created and terminal projections, including the
  selected manifest descriptor/platform, original image reference, effective
  path/args/config, CID path, host security/resource settings, registered
  mounts, labels, restart count, state, and timestamps. This intentionally does
  not claim kernel mount, seccomp, cgroup, or runc enforcement.
- Corrected five review findings before commit: Docker injects fixed `PATH` and
  `HOSTNAME` outside `Config.Env`; a long-form read-only bind reports top-level
  inspect `Mode` as empty; the old 64 MiB child cap could not hold the 64 MiB
  artifact plus 8 MiB receipt and frame header; recovery exceptions could
  delete staging while an orphan remained possible; and mode-0555 staging trees
  had to be reopened through no-follow directory FDs before safe recursive
  removal.
- Added a monotonic aggregate lifecycle deadline, closed Docker config and HOME
  mutation checks, exact no-follow staging/readback verification, cleanup that
  ignores caller cancellation, failure aggregation, and explicit fail-stop host
  assumptions for a dedicated one-job runner UID with core dumps disabled.
- Added 13 lifecycle tests and three HOME-mutation cases. The combined private
  Docker suite passes 92 tests. It remains synthetic adapter evidence; no live
  daemon/image/extractor/supervisor/receipt/signature was exercised.

### 2026-07-16 — Gate 3 execution-environment authority

- Added strict canonical `execution-environment@1.0` bytes with separate raw
  and content identities, exact prepared-profile instance binding, bounded
  decoding, immutable opaque authority, and disposable staging copies.
- Closed declared platform/runtime/image/toolchain/attestor configuration.
  Recomputed hashes for inline rootfs, binary, dynamic-library, and
  header/include-root inventories; kept external OCI-layout and build-
  attestation hashes distinct.
- Bumped sandbox policy to `1.2`, execution-plan hashing to domain v2, and
  receipts to `1.1`. Exact environment authority now flows through offline
  preparation, staged `execution-environment.json`, receipt verification,
  provenance, and result-frame decoding.
- OCI authorization now compares ordered layer media type, digest, size, and
  diff ID against the prepared environment before local image observation.
- Repinned policy/plan/profile/job/invocation/artifact/receipt lineage. Focused
  execution-environment/plan/receipt/provenance/OCI suite passes 64 tests;
  private Docker suite passes 92 tests; full compiler gate passes 45 files and
  1100 tests; release package gate passes 19 archive, 35 Node security, and
  packed/fresh-consumer checks.
- This proves configuration authority only. No live kernel/runc/seccomp/cgroup
  enforcement, native producer, real image, real receipt, or external
  signature was exercised.

### 2026-07-16 — Gate 3 WASM-first producer pivot

- Clarified the portable-product requirement: supported C++/CuTe source is
  compiled by a pinned CUDA-capable Clang-WASM frontend in a dedicated browser
  worker, then verified and lowered through the existing producer-neutral
  artifact and semantic-core contracts.
- Reclassified Docker/native AOT as an optional CI/reference parity producer.
  It may qualify corpora, compare producer output, or prebuild artifacts for an
  explicitly selected deployment profile; it is never a browser runtime,
  ordinary browser-compilation, or portable Gate 3 dependency.
- Preserved the normative three-mode protocol and retained completed AOT,
  receipt, provenance, OCI, lifecycle, and seccomp contracts as optional-lane
  work. Their evidence remains valid only for those named contracts.
- Corrected current status: there is no tracked Clang-WASM build, compiler
  worker, packaged CUDA/CuTe header closure, browser-produced artifact, or live
  native producer. No real C++/CuTe source-compatibility claim is currently
  supported by either lane.
- Defined the browser-local acceptance path: pinned compiler and asset
  provenance; bounded worker/VFS/resource/cancellation behavior; unmodified
  source to canonical artifact; source-map/diagnostic/entry closure; semantic
  convergence with Gate 2; and required actual-WebGPU execution. Optional
  native parity is evaluated separately and cannot substitute for this path.

### 2026-07-16 — Gate 3 producer schema v2 hardening

- Replaced unpublished profile, artifact, AOT job, execution-environment, and
  receipt fixture contracts with closed version-2 schemas. No v1 reader or
  migration path remains because no live producer or persisted artifact exists.
- Added exact ownership for source, compiler-resource, and dependency roots and
  files. Compiler-forced include edges must now match the prepared profile by
  option ordinal, root, and virtual path; dependency/header ownership and root
  containment are rechecked across artifact, receipt, and provenance seams.
- Preserved declaration initializer expressions and constrained locationless
  diagnostics to invocation, profile, or compiler subjects so a producer cannot
  erase source locations for source-owned failures.
- Separated exact opened-input observations, process measurements, emitted
  artifact counts, and enforced upper bounds. Receipt selection now binds one
  output-independent source request to the actual resolved artifact entry.
- Focused compiler typecheck and lint pass. Schema-v2 CuTe tests pass: 10 files
  and 108 tests. Full compiler gate passes 46 files and 1126 tests in the
  current worktree; 11 browser-asset tests are still uncommitted next-slice
  work. Release-package gate passes 19 hostile-tar and 35 Node security tests
  plus packed/fresh consumers.

### 2026-07-16 — Gate 3 execution-path clarification

- Added one explicit product flow to the normative LLD: C++/CUDA/CuTe source
  enters Clang-WASM in a dedicated browser worker; verified facts enter shared
  semantics; portable execution occurs through CPU reference or WGSL/WebGPU.
  User C++ is not linked or executed as a WASM program.
- Moved detailed Docker/AOT work into a visibly optional parity subsection so
  synthetic contract coverage cannot read as portable-product progress.
- Corrected the producer-neutrality claim: only artifact/resource verification
  and semantic/static-layout seams are common today. Browser profile, request,
  worker evidence, and common authorization still need implementation.
- Accepted D-063: browser-local worker evidence and optional native
  receipt/attestation evidence must mint one common producer-authorized
  artifact consumed by the same lowerer.

### 2026-07-16 — Gate 3 browser deployment and asset authority

- Added a closed AOT/browser profile union while preserving the exact AOT
  profile identity. Browser-local profiles pin the package-owned dedicated
  module worker, single-thread/terminate-on-cancel policy, source-network ban,
  content cache, WASM memory ceiling, asset ceilings, build-provenance lock,
  and asset-set identity.
- Split full deployment `profileHash` from producer-neutral
  `compilationContractHash`. Canonical artifacts and extraction records now
  bind only the latter; AOT receipts and attestations retain the full selected
  deployment profile. Browser and native producers can therefore compare
  semantic artifact identity without pretending their execution evidence is
  identical.
- Moved lowering authority into one producer-neutral opaque wrapper with a
  closed evidence kind/hash. The optional AOT attestation now mints through an
  AOT-specific adapter; browser-local minting remains blocked until a verified
  worker-result authority exists. Lowered origin no longer embeds Docker,
  receipt, or DSSE-specific fields.
- Added a strict browser asset-manifest authority over exact canonical bytes.
  It binds the browser profile, source ABI, one monolithic Clang/extractor WASM
  executable, semantic adapter, compiler-resource pack, every dependency
  header pack, exact mounts/content sets, build-provenance references, asset
  set, and independently recomputed byte totals.
- Kept executable loader JavaScript package-owned and outside downloaded asset
  metadata. Asset URLs are unique normalized same-origin root-relative paths;
  profile ceilings dominate manifest values. The authority records no fetched,
  unpacked, mounted, cached, or executed claim.
- Current hashes remain synthetic fixtures. No Clang-WASM distribution,
  license/build recipe, real header pack, verified asset acquisition/mount,
  worker request, browser producer, or browser execution has been implemented.

### 2026-07-16 — Gate 3 closed VFS and Clang-WASM build direction

- Replaced tar/gzip-shaped compiler-resource and dependency-header assets with
  identity-encoded `application/vnd.browsergrad.vfs-pack.v1`.
- Added the exact v1 binary verifier: fixed `BGVFSPK1` header/version, bounded
  canonical index and data lengths, strict sorted portable-ASCII relative
  paths, regular files only, implicit contiguous data, and exact pack/index/
  file/content-set hashes. There are no offsets, links, sparse files, Unicode
  aliases, archive metadata, compression, or general extraction behavior to
  secure later.
- Split structural inspection from use authority. Only exact pack bytes bound
  to one prepared asset-manifest instance, asset ID, include root, mount root,
  byte lengths, and content set mint verified pack authority. The real-byte
  integration fixture proves that binding and rejects a different valid pack.
- Preserved identity-encoding semantics: transfer and unpacked lengths cover
  the same complete pack bytes. Added separate per-pack and aggregate mounted
  file-content lengths and profile-owned ceilings.
- Recorded the researched implementation direction: one BrowserGrad-owned
  LibTooling/AST extractor cross-built from pinned LLVM/Clang with pinned
  Emscripten. A pinned Docker/OCI builder may make release assets reproducible;
  no Docker daemon or native toolchain enters browser runtime.
- Corrected the planned worker boundary before implementation: common local
  source requests cannot require Git provenance or expected output hashes;
  available profile headers differ from the request-specific opened closure;
  compiler runtime ABI differs from source target ABI; worker bytes need
  host-verified hash and length before Blob-module construction.
- Accepted D-066 through D-068. Real build pins remain feasibility candidates,
  not release/toolchain evidence. No asset was fetched, cached, mounted, or
  executed and no worker was spawned.

### 2026-07-16 — Gate 3 browser compiler-runtime ABI

- Removed the false profile-wide expected opened-header hash. Dependency and
  include-root manifests continue to pin available content; each artifact and
  later worker evidence owns its source-specific successful-read closure.
- Added compiler-runtime ABI v1 separately from source ABI: closed wasm32
  features, exact import/export identity, host-verified module-or-byte handoff,
  worker-side fetch prohibition, unshared worker-owned memory, page/stack/
  compiler-working reservations, and host-backed lazy VFS ceilings.
- Replaced URL-shaped Worker assumptions with one exact package-owned,
  self-contained ES module. Host verifies bytes and length, creates a Blob
  module Worker, then transfers verified assets. Worker network is forbidden.
- Made browser memory ceilings coexist: stack, compiler working allocation,
  opened VFS copies, and output fit the WASM linear-memory limit; that limit
  fits maximum pages; stack fits initial pages. Host-retained pack bytes remain
  a separate bounded/observed category.
- Added six focused adversarial runtime-ABI tests. Broader profile, asset, VFS,
  artifact, AOT receipt/provenance, and lowering contracts remain green after
  expected preproduction identity re-pins. Accepted D-069.
- Parallel build audit found two pre-build P0 contract gaps: current profile
  can request only a CUDA host semantic pass, and artifacts cannot distinguish
  host/device domains; the browser build-provenance hash has no verified bytes,
  license inventory, or reproducibility authority behind it. These remain
  explicit missing work, not hidden behind synthetic fixture hashes.

### 2026-07-16 — Gate 3 producer-neutral frontend request

- Added closed frontend-request v1 over compilation-contract identity, exact
  sorted source descriptors, main path, source anchor, current artifact
  schema/version, and shared semantic/input/output ceilings.
- Exact source bytes remain outside JSON. Preparation inspects unshared plain
  byte views, rejects declared/actual length differences before allocation,
  bounds cumulative copies, snapshots before await, hashes each file/anchor,
  and retains bytes behind opaque authority with copy-out access only.
- Deployment profile, worker/container/assets, Git identity, argv/environment,
  host/output paths, execution ceilings, expected outputs/closures, and actual
  opened inputs are absent. Detached source-provenance and conformance hashes
  cannot change request identity.
- Request v1 supports both artifact families needed by Gate 3: layout-variable
  and view-copy-function roots. Semantic limit keys are shared directly with
  the compilation contract to prevent schema drift.
- Focused request/runtime/profile/asset tests pass: 4 files and 53 tests.
  Strict typecheck, lint, and diff checks pass. Accepted D-070. Retained AOT
  metadata still needs composition around this authority before one-request
  cross-producer parity is complete.

### 2026-07-16 — Gate 3 CUDA semantic-pass authority

- Replaced the ambiguous host/device merge with two exact semantic actions.
  BrowserGrad runs explicit device-only extraction first; host-only validation
  runs only after device success. This ordering is product policy, not a claim
  about Clang driver defaults.
- Made `cuda-device-sema` the sole canonical semantic graph owner. Every fact
  and selected entry must resolve through that graph; host validation cannot
  contribute facts or reference device declaration/type IDs.
- Split source ABI from target graph identity. Device ABI binds the canonical
  graph and NVPTX conventions; host ABI binds stable source entities and host
  conventions. Only explicitly shared source projections must converge.
- Added a common canonical source-entity registry. Entity IDs are recomputed
  from entity kind, canonical source identity, and resolved spelling/expansion
  file content plus byte ranges. Registry lives at payload scope. Every
  successful pass reports exact selected-root IDs derived from serialized
  device entry declarations; accepted host/device sets must be identical and
  nonempty. Registry domains equal selected-root observations plus ABI owners;
  shared-surface hashes include roots, and ABI `shared` is derived. Device type/function
  identities and function return/parameter types are cross-checked against the
  canonical graph.
- Added per-pass target/auxiliary triples, device architecture, invocation
  mode, opened files, include edges, input-closure hash, surface hash, fact
  ownership, and diagnostic ownership. Conditional source closures may differ;
  compiler-forced includes remain mandatory in every executed pass.
- Closed diagnostic-note, unresolved-include, accepted/rejected, and fail-stop
  state matrices. Host diagnostics cannot name device declaration/type/
  expression/fact IDs. Parser ceilings now bound pass file/edge/fact/diagnostic
  IDs.
- Focused 6-file suite passes 89 tests; full compiler gate passes 53 files/1189
  tests plus optional-Docker 2 files/96 tests; release package gate passes.
  Final independent scoped P0/P1 review reports no remaining issue.
  No Clang-WASM build, worker, asset fetch/mount, or browser C++ execution has
  occurred.

### 2026-07-16 — Gate 3 canonical VFS-pack writer

- Added one release/build-time writer for exact `BGVFSPK1` bytes already owned
  by the closed reader and manifest binder. Caller order cannot change bytes;
  writer-generated packs round-trip byte-for-byte through structural proof.
- Bounded raw path length before UTF-8 encoding, then retained the exact byte
  ceiling and portable-path decoder. File/index/content/whole-pack ceilings are
  independent and checked before final pack allocation.
- Snapshotted every file immediately after safe intrinsic `Uint8Array`
  inspection, before any later caller record can trigger proxy or resizable-
  buffer reentrancy. Same-length mutation and buffer shrink cannot change an
  earlier retained file.
- Writer output carries no authority by itself. Existing exact-byte inspection
  and manifest-instance binding remain mandatory before mount or worker use.
- Focused reader/writer suite passes 2 files/15 tests with strict compiler
  typecheck and focused lint. No real compiler/header pack, fetch, cache, mount,
  worker, or browser-local C++ execution exists.

### 2026-07-17 — Gate 3 producer-neutral AOT composition

- Deleted the unpublished `cpp_cute_aot_job` source-intent authority and its
  test. Optional AOT now consumes the same prepared frontend request and exact
  source snapshots as browser-local compilation.
- Added AOT run-metadata v1 for exact AOT profile/request and caller-declared
  Git source-reference composition. Standardized its wire/public identity as
  `runMetadataId`; no `metadataId`/`runMetadataId` translation seam remains.
  Renamed every reference field so signed metadata cannot masquerade as
  source-acquisition provenance.
- Added a producer-neutral host-derived request binding over exact request and
  artifact authorities. It rehashes request-owned bytes, requires exact
  source-owned artifact inputs, resolves one declaration identity-token span,
  and keeps detached conformance outside request identity.
- Bumped frontend artifact to closed v3 because source declarations now
  require `identitySpanId`. Renamed artifact and receipt TypeScript types to
  their actual wire majors; no unpublished v2 aliases remain.
- Replaced AOT job/source-blob staging with canonical request, run-metadata,
  and request-owned source snapshots. Policy 1.4 and plan v3 bind those exact
  controls. OCI/Docker summaries and labels use `runMetadataId` plus
  `requestId`.
- Receipt v3 names run metadata, request, and host-derived request binding. It
  no longer duplicates source files, expected output, or selection. Detached
  attestation authenticates the declared source reference from run metadata;
  output/selection derives from the verified receipt/binding chain.
- Removed AOT-only run metadata from common lowering authorization and removed
  the unimplemented browser evidence kind. Browser worker evidence will add a
  real branch/mint path without fabricating native-run fields or advertising
  unsupported authority.
- Corrected the normative LLD and package changelog, replaced deleted public
  AOT-job diagnostic features with current run-metadata taxonomy, and separated
  trust-store v1 names from BrowserGrad predicate/build v3 names.
- Focused artifact/request/AOT/lowering suite passes 12 files/121 tests; final
  compiler gate passes 52 files/1167 tests; optional Docker suite passes 2
  files/96 tests; release-package gate passes. All identities and outputs
  remain synthetic; no Docker or browser producer executed.

### 2026-07-17 — Gate 3 browser asset acquisition and VFS installation

- Added exact same-origin, redirect-free, identity-only streamed acquisition
  over one prepared asset manifest. Response URL, status, optional content
  length, actual length, SHA-256, and aggregate manifest identity are verified
  before opaque authority.
- Added content-addressed cache admission and reload. Adapter methods are
  snapshotted once, admission receives isolated copies, every hit is copied and
  rehashed, and abort after any async boundary prevents authority minting.
- Added manifest-complete VFS installation. Every closed v1 pack is rebound to
  its exact manifest asset/include root, cross-pack file/directory collisions
  fail, and source-set plus verifier-owned pack copies both count against the
  retained-host ceiling.
- Hardened cancellation and hostile platform boundaries with captured
  AbortSignal/EventTarget/Headers/ReadableStream intrinsics, nonblocking
  terminal cleanup, late-fetch-response disposal, and cancellation precedence
  after trap-capable response checks.
- Removed premature generic installed-file copying. Worker execution will own
  aggregate WASM opened-file metering, transfer/disposal, and cancellation;
  installation authority remains metadata-only.
- Focused suite passes 3 files/30 tests. Full compiler passes 52 files/1172
  tests; optional Docker passes 2 files/96 tests; release-package gate passes.
  Independent final review found no P0/P1. Assets remain synthetic; no Worker,
  Clang-WASM, Docker, or browser C++ execution occurred.

### 2026-07-17 — Gate 3 pinned Clang-WASM build inputs and planner

- Added one strict canonical input-only lock for exact LLVM 22.1.8, CUTLASS
  3.7.0, emsdk 6.0.3, and `linux/amd64` OCI identities. It binds archive and
  notice hashes/lengths, a deterministic two-stage native-TableGen then
  Emscripten recipe, selected link closure, and the exact planned distribution.
- Kept release authority impossible. Seven blockers are derived from unresolved
  extractor, licensing, runtime-ABI, observed-Wasm-interface, and reproducible-
  build requirements; the lock always reports `releaseReady: false` and its
  assertion API always fails closed.
- Added one pure planner that consumes the verified lock recipe, substitutes
  only materialized absolute roots/tools and prefix maps, closes environment
  discovery, and emits deterministic configure/build steps. It performs no
  acquisition, process, network, Docker, runtime, filesystem, or artifact-
  authorization effect.
- Closed final-review gaps: planner now accepts only the opaque prepared lock
  and unwraps it inside the built authority module; copied structural lookalikes
  fail. Link-map evidence writes under the disjoint state root, never the exact
  distributed-output root. Every interpolated root/tool/PATH value uses one
  closed portable-safe absolute-path grammar.
- Wired planner typecheck/lint/Vitest into `verify:compiler`. Direct integration
  coverage materializes the actual decoded builtin lock, preventing mirrored
  fixture success from hiding recipe drift.
- Corrected the LLD's VFS direction: the browser extractor uses a custom
  `llvm::vfs::FileSystem` with no physical fallback and one Worker session owns
  lazy-copy accounting and lifecycle. No build or browser execution occurred.

### 2026-07-17 — Gate 3 canonical runtime ABI and profile binding

- Added one strict canonical 64-KiB-bounded runtime-ABI manifest and opaque
  design authority. It pins eight C exports, six synchronous host-backed VFS
  imports and record encodings, one unshared wasm32 memory, a 64-byte input
  frame, typed statuses, result lifetime, reset rules, and destructive Worker
  cancellation. It always reports observed-WASM and release authority false.
- Closed a self-authorization gap before the first build. Generated Emscripten
  imports, support exports, table/global projection, and allowed custom-section
  including `target_features` projection are empty and independently hash-
  pinned. Observation cannot add an allowlisted capability. Release stays
  forbidden until the first real module is raw-parsed, independently reviewed,
  and the design manifest is deliberately repinned.
- Closed final P0/P1 review gaps before commit: the Wasm policy now names an
  MVP baseline, admits exactly four declared extensions, and rejects every
  unlisted extension. VFS calls pin checked ranges, snapshot/alias/alignment
  rules, no memory growth, and invalid-range atomicity. Opened VFS memory is an
  aggregate session budget, and linear-memory coexistence now includes the
  live four-MiB input frame as well as stack, compiler, VFS, and result bytes.
- Advanced the closed frontend profile from 2.2 to 2.3. Browser profiles bind
  the exact ABI resource and derive address width, required features, fixed
  pages/stack, and maximum working/opened/result limits from that single source.
  Deployment profiles may narrow working/opened ceilings; host-retained packs
  remain independently bounded.
- Introduced semantic compilation-contract schema 1.0 with its own hash domain.
  Removing the enclosing profile version from this projection preserves the
  exact AOT/browser semantic hash across deployment-only profile revisions and
  prevents unnecessary cache invalidation.
- Clarified the normative execution boundary: Docker may reproducibly build or
  compare content-addressed assets, but the portable browser graph is source to
  Clang-WASM Worker to verified artifact to CPU/WGSL/WebGPU. User C++ is never
  linked or executed as generic WASM.
- Full compiler verification passed 54 files/1,232 tests plus both architecture
  checks, strict build/typecheck/lint, optional Docker 2 files/96 tests, build
  planner 1 file/25 tests, and all synthetic/corpus guards. Release-package
  verification passed 19 hostile-archive and 35 Node security tests plus packed
  and fresh consumers. No build, Docker process, Worker, or C++ execution ran.

### 2026-07-17 — Gate 3 ABI asset binding and raw-WASM inspection

- Advanced browser asset manifest to 1.1 with one exact identity-encoded
  runtime-ABI resource. The acquired bytes are strict-decoded and remain bound
  to the exact verified asset-set instance used by VFS installation.
- Bound the identical ABI ID, content hash, length, media type, and output path
  into the deterministic build closure. The input lock remains non-release
  authority and still requires build, interface, license, reproducibility, and
  provenance evidence.
- Added a bounded raw-WASM inspector with exact section framing/order,
  signatures, memory/table/global/tag/start/custom projection, complete engine
  validation, target-feature cross-checking, and closed extension decoding.
  Review reports cannot authorize execution; opaque conformance can mint only
  against the independently prepared ABI.
- Corrected the ABI's tool-conventions vocabulary to `sign-ext` and
  `multimemory`, added exact Worker-session handle/call ceilings, and preserved
  distinct BrowserGrad feature names internally.
- Closed standards/adversarial review findings for padded LEB, UTF-8 names,
  target-feature uniqueness/order, multi-memory memargs, `table.copy`,
  `bulk-memory-opt`, call-indirect-overlong, structured inventory keys, and
  exact non-function export projections.
- Focused boundary verification passes 5 files/111 tests with strict typecheck,
  lint, and whitespace checks. No Clang-WASM build, Worker, Docker process, or
  browser C++ execution occurred. Current empty first-build projections make
  production conformance deliberately impossible until review and repin.

### 2026-07-17 — Gate 3 Worker/VFS boundary hardening

- Commit `0159c7f1` closed the exact ABI asset/build binding and bounded raw-
  Wasm inspection slice. This follow-up closes the Worker/VFS design boundary;
  it still does not implement or execute the real Worker.
- Added a Worker-owned VFS session over one exact unshared `WebAssembly.Memory`.
  It composes prepared source snapshots with verified installed packs, serves
  bounded range reads, implements the six pinned VFS imports, orders directory
  entries by UTF-8 bytes, accounts live handles/logical reservations, and
  destructively disposes terminal state. It is explicitly not Worker-execution
  evidence.
- Downgraded pure Worker-result handling to caller-frame consistency only.
  Public decoding/validation/discard APIs cannot claim Worker execution,
  termination, host timing, or lowering authority. Actual authority still
  requires a host-owned Worker controller and event source.
- Independent review found and corrected counter-name drift, impossible
  counter combinations, post-terminal heavy retention, after-copy limit
  checks, false Wasm-residency naming, prototype-sensitive node detection,
  overlong mounted paths, unbounded expanded VFS indexing, and logical-open
  quota coupling to Wasm linear-memory arithmetic.
- Profile wire 2.4 and the runtime ABI are repinned with dedicated
  logical index limits: runtime maxima of 262,144 nodes/128 MiB and browser
  profile limits of 65,536 nodes/32 MiB. Full-file live-open bytes are a
  logical per-handle reservation, not measured Wasm residency. Release profile
  values must be justified by the final pack inventory and browser peak-memory
  qualification.
- Final identity anchors: ABI manifest `ae051550...`, resource `65b9de16...`
  (13,523 bytes), contract `f58247d7...`; build lock `ccff973f...`, resource
  `643ef802...` (18,667 bytes); browser profile `24b7a09d...`; asset manifest
  `ea50512c...`, asset set `b14f1b43...`, raw manifest `81b5a7e4...` (10,004
  bytes). Full values remain canonical constants and executable assertions.
- Final focused verification passes 9 files/167 tests. Full compiler
  verification passes 57 files/1,286 tests plus optional Docker 96 tests,
  Clang-Wasm planner 25 tests, all builds/architecture/synthetic/corpus gates,
  and strict typecheck/lint. Release-package verification passes 19 Python and
  35 Node security tests plus packed/fresh consumers. Final independent review
  found no P0/P1. No Worker was spawned and no C++, Clang-Wasm, Docker, or
  WebGPU execution occurred.

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

Build-input selection, deterministic command materialization, canonical ABI,
profile binding, exact ABI asset/build closure, and bounded raw-WASM inspection
are closed as design/verification boundaries. They are not real build,
execution, or release evidence. Keep optional AOT frozen and keep the build
lock release-blocked until extractor source, redistribution/license closure,
complete independently reviewed first-build interface allowlists, and two
distinct clean builds produce authenticated evidence.

The pure Worker protocol and Worker-owned VFS session are now closed bounded
components, but neither observes Worker execution. Next add the exact package-
owned self-contained Worker module and a host controller that verifies its
bytes, creates the Blob Worker, owns the fresh invocation nonce/event source,
integrates the VFS session, accepts one terminal result, measures host time,
and performs terminate-and-replace cleanup. Actual Wasm page/allocator
instrumentation stays separate from logical live-open reservations. Only this
controller may mint browser execution evidence or feed common lowering
authorization. The current caller-frame validator must remain
non-authoritative.

Then build the LibTooling extractor and first reviewed header packs from the
pinned inputs. Correct artifact-v3 root ownership for real unannotated CuTe
layout declarations before emitting C++: device-pass selection evidence owns
root membership while CUDA attributes remain source facts. Enforce declared
Clang work ceilings with real hooks or narrow the profile claims.

The first executable checkpoint is one unmodified pinned layout-only source
compiled in a supported browser with network and Docker absent. Its canonical
frontend artifact must pass the existing strict verifier, preserve exact
diagnostics/source spans/selected-entry identity, lower through the authorized
layout seam, and match an independently derived expected semantic result. A
retained native producer may later run the same fixture as an optional parity
oracle, but it cannot block or stand in for this browser-local proof. Dynamic
tensor copy and required real-WebGPU convergence follow that checkpoint.
