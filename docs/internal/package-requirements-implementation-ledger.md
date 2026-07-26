# Package Requirements Implementation Ledger

- **Normative source:**
  [`docs/platform/package-requirements-lld.md`](../platform/package-requirements-lld.md)
- **Ledger status:** active
- **Last updated:** 2026-07-26
- **Historical handover:**
  [`package-requirements-handover-2026-07-17.md`](./package-requirements-handover-2026-07-17.md)
- **Current implementation slice:** Gate 3 production browser-local C++/CuTe execution

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
| Gate 0 — freeze and inventory | `verified` | Workspace direction and all six legacy adapters are machine-frozen. Stable diagnostic/capability/backend/requirement vocabularies are separated. Compiler pointer/scalar, runtime requirements, Grad behavior, and the exact remaining JIT one-constructor/one-operation intentional user-kernel opaque boundary have pinned inventories and executable contracts. ADR-0002 and ADR-0004 through ADR-0051 classify the original JIT baseline and revised Grad/runtime contracts without widening opaque callers or silently substituting semantics. The JIT set is 36 typed retirements, one intentional opaque identity, and two removed unsupported surfaces; the same executable 36-record registry now generates detached Python and JavaScript/platform projections. Grad owns a closed twelve-dtype eager registry, separates direct and torch constructor semantics, implements zero-stride expand views, rejects fake bfloat16 before allocation, fixes contiguous/detach/cast semantics, keeps every eager placement entrypoint CPU-only and fail-closed, zero-copy wraps all twelve supported writable NumPy storage dtypes/layouts, exports only owning NumPy snapshots, and generates a detached 22-record platform projection from that frozen executable inventory. Runtime generates all 53 registered assignment definitions and the one capability/three backend static registry, requires provider-bound resolution before requirement availability, preserves relevant records through every public readiness carrier, requires program/artifact-scoped lowering decisions before constructing a support view, and composes framework sources without importing their packages. No Grad behavior remains marked compatibility debt. | None. Any baseline change requires the accepted-ADR exception path. | Architecture check and mutation suite; JIT and Grad Gate 0 contracts; focused typed-attention, dtype/view/cast/device, expand-view, NumPy integration, runtime requirement-resolution, program-capability, and framework-platform tests; full package and actual-device evidence remain separately scoped. |
| Gate 1 — value/layout core and wire foundation | `verified` | Semantic-core `0.1.0` implements the bounded wire/value/layout contract, closed `browsergrad.layout@1` verification, authority-bound opaque artifacts, content-scoped IDs, deterministic normalization, and coordinate/address/alias traces. An independent Python reference matches TypeScript normalization, full-envelope canonicalization, semantic hashes, and traces for pinned static and symbolic fixtures. | None. The separate packed/release-tested `0.x` transition required by D-004 is also complete locally; registry publication remains a release operation, not a Gate 1 criterion. | Semantic-core typecheck/build/lint; 8 files and 68 tests; two pinned cross-language fixtures; 14 verifier rejection mutations; dynamic trace rejection and dominating-predicate parity; architecture check; packed-tarball gate. |
| Gate 2 — multi-frontend, multi-backend view slice | `verified` | Semantic-core `0.2.0` owns verified view-copy meaning, shared specialization, and the sole canonical frontend construction sink. Kernels `0.2.0` passes the full nine-case CPU/actual-WebGPU bit-exact matrix. Compiler L1 preserves its six-case non-padded contract; sibling L2 binds the exact verified operation and proves rank-2/rank-3 guarded padding through CPU, structured IR, WGSL, and actual WebGPU. JIT `0.9.0` emits a closed typed permutation request and executes the same canonical operation through materializing/resident production routes without recovering semantic args from the frozen plan. All strict lanes passed from one clean detached worktree at exact source revision `aa605421410e9d4190d8939c24b1057731111231`. | None for the initial Gate 2 profile. Release CI must repeat the exact-source lanes before publication; registry publication is a separate release operation. | Exact clean-source Apple Metal 3 records: compiler L2 3 cases, compiler L1 6 cases, kernels 9 cases, and JIT 2 cases; compiler `verify:compiler` passed 35 files/1004 tests; release package gate passed 19 hostile-archive and 35 Node security tests plus packed/fresh consumers; combined publish guards accepted only exact L1/L2/JIT/kernels markers. |
| Gate 3 — real C++/CuTe frontend slice | `in-progress` | Browser-local Clang-Wasm remains the primary portable-product producer. The current extractor is package-pinned from two byte-identical cache-free builds, and the strict eight-case Chromium matrix now uses one package-owned exact compile profile rather than a synthetic fixture. The five-pack/5,788-file header universe, Runtime ABI 1.17, both zero-import Workers, unchanged C++17/CuTe f32/i32/u32 rank-1-through-rank-4 sources, accepted Artifact V3 outputs, and distinct semantic view-copy candidates are content-bound. CPU and actual WebGPU independently share the canonical word32 view-copy seam. A deterministic materializer writes the exact 24-subject tree, and a producer-gated finalizer adds only the detached DSSE envelope. Two live exact 25-output roots are byte-reproducible and the package-pinned observation independently rebinds every output to current authorities. One no-clobber external-evidence exchange emits exact policy-scoped producer or reviewer signing material, rebinds each returned envelope to its exact request and current inputs, runs the corresponding opaque transition in process, and serializes no reusable trust or approval authority. Docker stays outside the portable runtime graph. | Admit the exact observed candidates through externally rooted producer trust and shared CPU/required real-WebGPU execution. Install the production producer and distribution-approval policies, externally controlled keys, and externally issued exact-build and exact-header-distribution statements. Issue final release authority only after consuming the distinct current reproducibility, build, legal, trust, execution, and backend authorities. | Capability commits through `c00769c6`; current lock `bg.cpp.browser-build-input-lock.sha256.fa21cfe45dec6b4869662cd613a7a300848657518f375c04f7f2193f3a874ad4`; complete-distribution reproducibility `bg.cpp.browser-full-distribution-reproducibility.sha256.b9b6bab28354bc3ddb7466886f958413d253aab65c00bcd61ef13eeaa074c2bf`; package evidence SHA-256 `4637cf9624ad00dd833b72b832f2c9e25ee0a8ffcdf62d8b95b732791d36a65a`. Workflow `30069614333` produced identical 31,841,008-byte Wasm at `19edd5622461b2308e83f10fb90f9f029241a5ba706e4c1741b194cb52a82138`. The 26,213-byte strict resource is `4d8b956050834e550405b15f1c7d52b16927ee3e8d4bc7b7da4035d430edc80a` at source `e9062c20f2a070774743e4d839c275c05df47225`. Fast gate 96 files/789 tests in 14.14 seconds; complete compiler 103 files/1,646 tests in 11.52 seconds; required native 54 files/286 tests with nine skips in 52.38 seconds; architecture zero cycles/leaks. Production producer trust, production license/distribution approval, lowering authority, backend execution for the exact observed payload, and `releaseReady` remain false. |
| Gate 4 — tiled GEMM and schedule separation | `verified` | The initial closed portable profile separates canonical logical GEMM tile meaning from physical schedule artifacts and schedule specialization. It supports dense row-major 4-byte-aligned certified exact-input f32 only. Two derived schedules use scalar memory operations, one output per invocation, cooperative single-buffered workgroup staging, zero-filled boundary loads, suppressed boundary stores, and all-invocation uniform barriers. Compiler typed-artifact lowering converges on the same semantic constructor without claiming source-body equivalence. The kernels backend executes the exact certified snapshots after caller mutation and reports portable re-legalization plus bit-exact certified-input preservation. | None for the initial closed profile. General f32, additional dtypes/layouts/base offsets, resident-buffer provenance, vectorized/native MMA schedules, and source-produced schedule preservation require separately named future profiles and evidence. | Capability commits `2cb7cea3` through `fe5581d3`, with exact Worker repin `343523fe`; semantic-core 16 files/138 tests; kernels 18 files/147 tests plus 20 focused semantic-GEMM tests; compiler convergence tests and 96-file/1,602-test full suite; packed-release checks; required real-WebGPU irregular 17x23 by 23x19 execution under 8x8x8 and 16x16x16 schedules with complete byte equality. CI `29818182317` dedicated semantic-GEMM job passed. |
| Gate 5 — tiled attention flagship | `verified` | The initial closed f32 profile owns frontend-neutral attention meaning, an independent online K/V-tile schedule, a schedule-independent CPU oracle, authority-bound specialization, exact scalar WGSL, and bounded host/WebGPU execution. The proved algorithm stages K/V cooperatively, carries online softmax across increasing tiles, excludes causal/tail keys before state updates, and uses uniform barriers. Required correctness and performance records are separate. | None for the initial closed profile. Additional dtypes/layouts, resident-buffer provenance, frontend schedule convergence, vectorized/native facilities, or a FlashAttention-v2 claim require separately named profiles and evidence. | Semantic-core 19 files/166 tests. Kernels passes 19 files/154 tests, build, Node/browser typecheck, lint, architecture, hostile binding/domain/authority/resource/device tests, and packed-release consumption. Required headed Chromium on Apple Metal 3 executes causal/non-causal `(B=1,H=2,Sq=9,Sk=11,D=4,Dv=6)` through 8x8 and 8x16 schedules with complete declared-policy CPU and cross-schedule comparison. A separate `(B=1,H=2,Sq=256,Sk=256,D=Dv=32)` host-API record uses 16 warmups and 20 alternating paired samples against `row-wise-online-softmax-baseline`; three final local runs observed candidate medians of 3.9–4.9 ms and baseline medians of 6.9–10.5 ms, retaining that variability without a superiority claim. Final local performance artifact `31d0656703465037204cec096a517ef20e9f28331868220f3c54c3c137c9bbf3` passed on headed Chromium/Apple Metal 3. Commit `f4e25d4f` passed eight-family CI `29824737993`; exact-commit correctness and performance release markers are both required. |
| Gate 6 — framework convergence | `verified` | Thirty-six public operation records are retired from `CUSTOM` into typed IR with executable registry decisions. No advertised framework operation remains opaque; only the intentional user-authored WGSL extension stays opaque. JIT projects the same executable registry into detached JavaScript and framework-neutral platform records. Grad schema v2 has no remaining compatibility-debt behavior and generates a detached 22-record platform source from its frozen source- and fixture-bound eager inventory. Runtime generates all 53 requirement definitions, consumes complete provider-bound environments throughout readiness APIs, requires subject-bound lowering decisions for program support, and composes the generated Grad and JIT sources without importing either framework. Requirement, program, framework, and terminal-evidence facts remain distinct. Host-only typed operations explicitly refuse tensor-plan/WebGPU until their canonical portable lowerings exist. | None for the initial framework-convergence profile. New framework operations, dtype/layout contracts, or backend profiles require new executable contracts and regenerated support records; terminal execution evidence remains a separate authority. | ADR-0002 and ADR-0004 through ADR-0051; architecture freeze preserves the exact JIT partition, revised Grad contracts, byte-exact generated requirement, program-capability, and Grad platform registries, six direct resolution consumers, five resolution carriers, and architecture implementation/declaration parity. Node 25.9.0 passes Grad's 33 unit tests in 0.99 seconds and the seven-package build in 8.36 seconds; full JIT integration passes 55 files/336 tests and full Grad integration passes 59 files/362 tests. Semantic-core passes 21 files/174 tests, runtime passes 14 files/137 tests, the full compiler gate passes 96 files/1,602 tests plus parallel lanes, semantic architecture plus its 29-test mutation suite pass, and the release gate proves the packed 22-record Grad plus 36-record JIT runtime view. |
| Gate 7 — host graphs and optional systems expansion | `not-started` | No implementation in this workstream. | All Gate 7 exit criteria. | None. |

## Active Slice

### Objective

Gates 0 through 2 and Gate 4 are verified. Gate 2 has one shared materializing
view-copy contract, CPU evaluator, canonical WGSL lowering, and compiler/JIT tracer
bullets through actual WebGPU, all re-proved at one exact clean revision. Gate
3 now audits and implements the browser-local C++/CuTe frontend boundary. Its
primary producer is a pinned CUDA-capable Clang frontend running as WASM in a
dedicated browser worker. Its first tracer must reuse the verified artifact,
layout, and backend seams rather than add source-shaped execution or
spelling-specific lowering paths. Docker/native AOT remains an optional
CI/reference parity lane and is not a browser requirement.

Gate 4's initial profile is complete independently of Gate 3's unfinished
production producer. It admits only certified exact-input dense row-major f32,
keeps logical meaning and physical schedule in separate verified artifacts,
and reports portable schedule re-legalization rather than native-MMA or source
schedule preservation. Gate 5's initial closed f32 profile is now verified. It defines the
frontend-neutral attention-forward semantic contract while keeping the existing
row-wise online-softmax implementation explicitly named as a baseline. The
separate tiled-attention schedule contract is also implemented. The current
CPU reference, numerical conformance, and bounded schedule specialization are
implemented. Kernels WGSL preparation and authority-bound host/WebGPU execution
are implemented, with required causal/non-causal two-schedule correctness and a
separate observational performance record against the frozen row-wise baseline
on a named device/browser. Gate 6's initial framework-convergence profile is
verified. The current slice returns to Gate 3's real browser-local C++/CuTe
producer; broader dtype/layout profiles remain independently named follow-on
capabilities.

### Work in flight

Rows below that retain `AOT`, `OCI`, or `Docker` in their names record completed
protocol and synthetic-adapter work for the optional native parity lane. Their
`verified` status applies only to the named contract; it does not count as
browser-local producer evidence or make Docker a product dependency.

| Work item | Status | Notes | Remaining | Evidence |
|---|---|---|---|---|
| Frozen-adapter inventory | `verified` | Baselines are machine-enforced for compiler pointer/scalar memory, `cute_static_layout`, the 44-op shape/f32 `TensorGpuPlan`, and the original 36-constructor/39-operation JIT set partitioned into 36 typed retirements, one intentional user-kernel opaque identity, and two removed unsupported surfaces, plus runtime requirement mapping and 17 Grad dtype/view/materialization behaviors. Grad schema v2 classifies bfloat16 as an explicit refusal and contiguous as conditional identity/materialization. |
| Semantic-core seam audit | `verified` | Existing compiler/JIT/kernels types adapt into the core through explicit public subpaths; none moved wholesale. The package split and dependency direction are architecture-guarded. |
| Test-topology analysis | `verified` | TypeScript + Vitest; no specialized catalog match, so native focused Vitest route is recorded locally. |
| Gate 0 architecture check | `verified` | Cross-package boundaries, generated-source parity, all six required freezes, exact runtime mapping/status unions, reviewed vocabulary, profile-usage parity, pinned inventories/harnesses, normalized definition fingerprints, and representative mutations are implemented and wired into delivery gates. |
| Gate 1 schema/value core | `verified` | `/schema` and `/layout` only; all Gate 1 requirements and the explicit cross-language exit are covered. The Python code is a synchronized reference oracle, not a second runtime or stable public API. |
| Semantic-core package adoption gate | `verified` | `0.2.0` is public-package shaped, dependency-free, subpath-only, locally packed, and now consumed through kernels' packed exact dependency. A fresh temporary consumer installs both tarballs, resolves bare public subpaths, typechecks, and prepares matching CPU/WGSL specializations. It has not been published. |
| Gate 2 view-family selection | `verified` | Selected typed JIT `PERMUTE` plus compiler L1 read binding and sibling L2 guarded materializing view copy. The full required matrix and strict exact-source WebGPU proof now pass; broader view families must add typed operation variants rather than reinterpret frozen plans. |
| Gate 3 legacy CuTe motif freeze | `verified` | Existing transpose, GEMV, GEMM, affine-tile, flash-attention, and WGMMA/TMA source-spelling normalizers are explicitly frozen compatibility debt. Exact exception-file membership and source hashes are architecture-guarded; new motifs, replacement bodies, files, or call sites require an accepted architecture decision. | Delete these adapters after pinned resolved frontend artifacts cover their retained fixtures through shared semantics. | Architecture guard and two mutation tests. |
| Gate 3 browser-local Clang-WASM producer | `partial` | Closed browser-local profile 2.6, producer-neutral request, Artifact V3, compilation contract 1.2, CUDA-pass authorities, exact semantic-adapter binding, separate zero-import raw-Wasm verifier/compiler Workers, closed VFS, canonical transfer, one-shot entry, and typed terminal protocol compile source-derived view-copy graphs. ABI 1.17 retains the independent diagnostic cap and 32 MiB result ceiling. The package-pinned current extractor recognizes exact f32/i32/u32 rank-1-through-rank-4 view-copy facts without adding a source-spelling execution path. The real browser lane now constructs its exact profile through package source and imports no synthetic test-profile fixture. | Admit the exact promoted candidates through externally rooted producer trust, then cross the existing shared lowering, CPU, and required real-WebGPU boundaries. Distribution approval and release remain independent. | Green run `30069614333` produced byte-identical 31,841,008-byte Wasm. The exact package matrix covers eight real Worker compilations at resource SHA-256 `4d8b956050834e550405b15f1c7d52b16927ee3e8d4bc7b7da4035d430edc80a`. Compiler Worker SHA-256 `a3b8610fc116c7b4949379dbcdcdd55e06ce5f9f59f11bf692abd689b0f17916`; verifier Worker SHA-256 `06ffb66e4e808e9df030cc3fe2981fa3adddf13d03780680abb091cbcbd4b9eb`; both retain zero static/dynamic imports. |
| Gate 3 Clang-WASM build and distribution | `partial` | The lock pins the extractor closure, LLVM 22.1.8, Emscripten image, recipe, ABI, and exact 25-path distribution plan. Cached diagnostics remain untrusted and separate from clean/release authority. The exact reviewer rejects interface drift after preserving bounded evidence. The current extractor is package-pinned from two distinct-path cache-free builds; the exact 17-file header subset is independently rehashed across two roots and package-pinned. The deterministic materializer authenticates those headers plus current package Wasm, Worker, lock, profile, and policy inputs and exclusively writes the exact 24-subject tree without a signing interface. The finalizer independently rereads that tree and profile and adds only the detached envelope after the in-process producer transition binds the same subject. Two distinct live 25-output roots passed the complete verifier; their package observation rebinds all outputs to current package authorities and grants reproducibility only. The unified no-clobber exchange issues and verifies the exact distribution-review request while persisting only a non-reusable observation. | Install the package-controlled production producer and approval policies, externally controlled builder and reviewer keys, and externally issued exact statements. Retain those authorities separately from current full-distribution reproducibility and final release approval. | Commits `8eea3660`, `b1caf988`, `628c718a`, `23d8bb60`, `0840801a`, and `c00769c6`; current lock `bg.cpp.browser-build-input-lock.sha256.fa21cfe45dec6b4869662cd613a7a300848657518f375c04f7f2193f3a874ad4`; header reproducibility `bg.cpp.browser-header-distribution-reproducibility.sha256.4d4c054fd4c93dbdbdef9581eeac52b037af3425e6a1c7eff8acc585abce1e55`; full reproducibility `bg.cpp.browser-full-distribution-reproducibility.sha256.b9b6bab28354bc3ddb7466886f958413d253aab65c00bcd61ef13eeaa074c2bf`; full package resource SHA-256 `4637cf9624ad00dd833b72b832f2c9e25ee0a8ffcdf62d8b95b732791d36a65a`; 25 outputs/103,637,461 bytes per root. The local engineering signer grants no external producer or release authority; production external approval remains absent. |
| Gate 3 exact real-world CUDA corpus gate | `partial` | Corpus provisioning owns exact commit/origin/physical-tree admission under a cooperative lease. Pinned gitlinks are bound as non-audit metadata and excluded from source snapshots; stale leases recover only from one canonical dead-owner marker; confirmed checkouts avoid network/mutation; Git/Python and descriptor-relative/no-follow/atomic-no-replace facilities are selected and probed once through a fail-closed host-toolchain capability. Provisioning is a direct standard-gate dependency. Four independent compile/codegen audits execute concurrently, preserve the same limits and browser thresholds, and emit versioned per-step timing artifacts before the serial real-WebGPU phase. | Add bounded safe reclamation for owned failure residue, resolve or explicitly bound the final validation-to-unlink interval, reduce repeated corpus-byte passes, and shard the browser phase with aggregate case/threshold evidence. This gate qualifies the existing CUDA-lite/WebGPU compatibility ladder; it does not prove the Clang-Wasm producer. | Commits `b08429ae`, `f252171d`, `c253526c`, `aaf12006`, `1669d1a8`, and `6c671559`. Main CI `29768391553` passed source/dist gates in 145.48/153.92 seconds total: parallel audits bounded by 33.55/35.71 seconds and browser execution 111.91/118.20 seconds. Focused local provisioning completes in about 0.44 seconds. |
| Gate 3 browser build signature and producer trust | `partial` | Manifest v1.4 binds the exact predicate type, trust-store digest, and canonical builder allowlist into the profile-pinned asset-set identity. One strict canonical DSSE/in-toto verifier authenticates P-256 P1363 bytes and rebinds the statement to the exact profile/compilation contract, asset manifest, build-input lock/recipe, package Worker/factory, and cycle-free build subject. A separate bounded canonical host-only policy admits the exact predicate, trust store, builder IDs, key IDs, and policy version; only that opaque policy plus the exact opaque signature binding may mint `VerifiedCppCuteBrowserBuildProducer`. The host exchange accepts only canonical immutable profile, manifest, package lock, package Worker, policy, and trust-store files; emits one exclusive read-only format-only request; requires that same request while verifying the returned envelope; and persists only a non-reusable observation after both opaque transitions pass in one process. It accepts no private key. Exact asset bytes, full distributed-output reproducibility, legal/distribution approval, Worker execution, lowering, backend execution, and release remain separate and false. | Install a package-controlled production policy, externally controlled key, and externally issued statement for the exact current build subject; run the request and verification modes against those exact files. | Commits `332ddf7a`, `22522508`, `e722434c`, and `3cf11a47`. Six focused exchange tests cover canonical deterministic output, exact external signing, no-clobber persistence, path/inode/link/mutability/package drift, request/envelope/signature drift, hostile arguments, and cancellation. The 93-file/777-test fast gate, 101-file/1,640-test compiler suite, 53-file/280-test required-native gate with nine skips, full compiler verifier, and zero-cycle/zero-leak architecture checks pass. Only ephemeral P-256 fixture evidence exists. |
| Gate 3 Worker-owned VFS session | `partial` | One exact unshared Wasm memory serves canonical source snapshots and verified pack ranges through the six runtime-ABI imports. VFS preparation remains split into a memory-independent mount and one-time bind; canonical transfer preserves exact authorities and destructive ownership. All eight promoted strict cases mounted the five exact package-matched packs, installed 5,788 files, opened one source and 1,168 headers, and completed one accepted compile without ambient filesystem fallback. | Install production external approval for the exact pack identities; do not reinterpret the successful mount or synthetic approval fixture as redistribution authority. | Strict matrix SHA-256 `4d8b956050834e550405b15f1c7d52b16927ee3e8d4bc7b7da4035d430edc80a`; package-pinned header packs matched. Production header approval and externally rooted producer trust remain false. |
| Gate 3 Wasm runtime and frontend-work metrics | `partial` | C ABI 1.2 retains the exact allocator and generation-bound frontend-work records. Native callbacks count bounded preprocessing, AST, constexpr, template, and CUDA-pass work; overflow fails closed. Runtime ABI 1.17 keeps fixed source-independent Artifact V3 serialization/result failure stages, the 32 MiB result ceiling, and the separate 8 MiB normalized-diagnostic bound. | Keep execution observation separate from trust, lowering, backend, and release authority as the exact candidates cross those later boundaries. | The eight promoted strict cases compiled in 23.869–26.319 seconds and completed in 26.988–29.587 seconds, each with one accepted Artifact V3 terminal plus raw-Wasm and exact-interface conformance. |
| Gate 3 Worker invocation/result protocol | `partial` | Exact profile/asset/VFS/request/ABI/verifier-evidence identities bind one invocation. Host transfer remains single-reservation/single-materialization with no network authority. The controller verifies Worker bytes, validates one terminal frame, closes every effect, and rebinds the retained verifier/invocation/request/profile/request-binding/artifact chain before minting execution evidence. Chromium produced eight distinct accepted source-derived view-copy candidates from unchanged source using the package-pinned reproducible Wasm. | Admit the externally rooted producer independently, then replay those exact payloads through shared lowering, CPU, and required WebGPU. | Package matrix resource SHA-256 `4d8b956050834e550405b15f1c7d52b16927ee3e8d4bc7b7da4035d430edc80a`; eight unique execution, Artifact V3, and candidate identities; `producerTrusted=false`, lowering false, backend false, release false. |
| Gate 3 browser layout-lowering authorization | `partial` | One exact observed Worker layout candidate and one exact independently admitted build producer are re-unwrapped and cross-bound to the retained profile, asset manifest, asset set, package Worker, invocation, request, request binding, accepted Artifact V3, selected layout entry, and shared semantic hash. Only that composition enters the existing canonical `AuthorizedCppCuteFrontendArtifact` side table and `lowerAuthorizedCppCuteLayoutEntry`; structural copies fail. Non-authoritative layout preparation remains in a dependency-lower semantic module, so the canonical transition adds no import cycle or parallel lowering path. Public authority states local semantic lowering true while backend execution, distribution, and release remain false. | Combine the promoted exact browser candidates with a real external signer policy, then compare their lowered layouts with Gate 2 CPU and required real WebGPU. | Commit `c806214c` established the seam; current strict matrix SHA-256 `4d8b956050834e550405b15f1c7d52b16927ee3e8d4bc7b7da4035d430edc80a` supplies eight promoted candidates. Architecture checks retain zero cycles/leaks. |
| Gate 3 browser view-copy authorization | `partial` | One exact observed Worker `view-copy` candidate and one exact independently admitted build producer are re-unwrapped and cross-bound to the retained Worker/profile/request/artifact lineage, the selected entry, and its prepared semantic subject. Only that composition enters the existing canonical `AuthorizedCppCuteFrontendArtifact` side table used by the producer-neutral view-copy lowerer. Candidate and authorization identities deliberately exclude source/destination allocation sizes and byte offsets; the lowerer accepts those explicit storage facts only after authorization. Eight real promoted source-produced candidates now exist; synthetic authority fixtures continue to prove the later authorization/lowering/WebGPU transition without pretending external producer trust. | Combine the promoted reproducible lineage with an externally rooted producer, then execute the exact authorized candidates through CPU and required real WebGPU; keep browser execution and release authority separate. | Package matrix SHA-256 `4d8b956050834e550405b15f1c7d52b16927ee3e8d4bc7b7da4035d430edc80a`; the independent 13-case word32 CPU/WebGPU matrix passes. `producerTrusted=false`, lowering false, backend false, release false for the real observations. |
| Gate 3 deterministic header-pack selection and assembly | `partial` | The exact plan binds eight archives, nine header subtrees, and nine upstream license/copyright files. Admission streams all 334,136,433 archive bytes. The release-shaped pipeline preserves case-distinct Linux paths, materializes one exact configured Clang CUDA runtime-wrapper derivative, and writes five canonical packs containing 5,788 files/69,680,000 bytes. It admits the exact CUDA index, writes the deterministic complete `license-inventory.json`, copies the ten retained component notices, and constructs the aggregate notice. Every output is no-clobber and independently reread. All 17 outputs/71,114,743 bytes are independently rehashed and identical across two distinct roots. Stable output metadata carries the exact header-input projection instead of the extractor-sensitive full-lock ID; live authorities still require full current-lock parity. Policy remains non-authoritative: the prepared map still requires production external review, and full release-output reproducibility/release facts remain false. | Install the production approval policy and verify an externally issued decision over the exact current subject, then bind the approved pack/notice identities into the asset and provenance chain and mount only those exact packs inside the strict package Worker. | Reproducibility ID `bg.cpp.browser-header-distribution-reproducibility.sha256.4d4c054fd4c93dbdbdef9581eeac52b037af3425e6a1c7eff8acc585abce1e55`; package resource SHA-256 `7a39e78d7aa3f1f0ff68e4b7b095425c75aa5e3947b208cdc6c5c30b46524838`; two-root materialization completed in about 28 seconds. Production license approval and release authority remain false. |
| Gate 3 external header-distribution approval | `partial` | One bounded host-only policy admits an exact policy ID, trust-store hash, reviewer allowlist, and key allowlist. The verifier reauthenticates the current package-pinned 17-output distribution, derives the exact current review subject including `license-inventory.json`, verifies one canonical DSSE/in-toto P-256 P1363 decision, and mints an opaque authority that upgrades only the external legal-review and distribution claims for that policy and subject. The shared no-clobber exchange accepts exact immutable policy/trust/request/envelope files, emits no private-key interface, and serializes only a non-reusable verification observation. | Replace the synthetic fixture with a package-controlled production policy, externally controlled reviewer key, and externally issued statement over the exact current subject. Retain this authority separately from full-distribution reproducibility, producer trust, execution, backend, and release. | Commits `b1caf988` and `23d8bb60`; nine focused exchange cases, strict typecheck/lint, 93-file/780-test fast gate, 101-file/1,640-test complete compiler suite, 53-file/283-test required-native gate with nine skips, full compiler verification, and zero-cycle/zero-leak architecture checks pass. No production policy, external key, or external decision is installed. |

| Gate 3 native compile-session foundation | `verified` | One closed noncopyable session validates canonical profile/request regions, recomputes every bound identity, verifies exact VFS source bytes after identity admission, privately binds output ceilings, materializes closed argv, and runs fresh device-first/host-second policy/VFS/diagnostic state before canonical Artifact V3 production. Accepted and rejected outcomes remain explicit and bounded. This status applies only to the native producer slice, not Gate 3 or browser execution. | None for the native producer slice. The exact code must still compile to Wasm and execute in the Worker before it contributes browser-local evidence. | Commit `bf353f5f`; focused native strict/UBSan and TypeScript artifact tests green; no Emscripten/Worker execution evidence. |

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
| Gate 3 canonical artifact resource | `verified` | Artifact v3 has one authority-bound canonical byte representation with independent raw SHA-256 and wire-u64 length. Source declarations require exact identifier/declarator `identitySpanId`; exact input ownership distinguishes source, compiler resource, and pinned dependency files; forced includes are profile-bound. View-copy control graphs use bounded non-recursive traversal with target-intrinsic cycle detection, including cycles that do not return through the first operand. Optional AOT receipt and in-toto contracts bind the raw resource; the browser path now has exact observed-candidate and producer-authorized lowering transitions for both layout and view-copy entries. | Real browser-produced accepted bytes and a real externally rooted producer instance remain unproved. | Commits `912195fa`, `9e20a610`, `c806214c`, and `a8a861e2`; artifact, request-binding, candidate, authorization, and mutation coverage. |
| Gate 3 allocation-free layout semantics | `verified` | Semantic-core snapshots, normalizes, verifies, hashes, and authority-binds one standalone layout expression as a `browsergrad.layout@1` artifact containing exactly one index map and zero allocations/views. The compiler now lowers the exact authorized CuTe layout fact through this API, preserving flat and nested static hierarchies and signed element locations. Coordinate traces expose element location plus logical/predicate bounds only; no dtype, byte range, alias, effect, tensor, or backend meaning is implied. The pinned `(3,2):(1,3)` fixture maps to `[0,3,1,4,2,5]` with semantic hash `4e1fa226...`; dynamic bindings, ordering, resource limits, hostile input, transport-neutral identity, and structural-copy rejection are covered. | Dynamic CuTe expressions remain a later compiler profile. CPU copy and WebGPU proof wait for real tensor/storage facts. | Semantic-core typecheck/lint and 12 files/107 tests; compiler authorized-layout tests and full compiler gate. |
| Gate 3 authorized static layout lowering | `verified` | The internal compiler boundary accepts one instance-bound authorized artifact plus one closed explicit entry ID. It requires exactly one accepted selected layout entry, binds it to exactly one affine fact/result declaration, preserves unrelated typed facts without interpreting them, evaluates all static integer algebra through shared bounded semantics, lowers nested modes with colexicographic coordinate composition, and stores source/macro/producer origin in a compiler side table. It uses CuTe `cosize(layout) = layout(size(layout) - 1) + 1`, including signed strides; it does not manufacture storage. The browser composition now reaches this exact seam only from an observed Worker candidate cross-bound to independently admitted producer trust. | Instantiate the browser transition with a valid production compile and real external signer evidence. Add a dynamic layout profile only with explicit binding authority. | 9 focused lowering tests plus 5 browser-authorization tests; 78-file/692-test fast harness; 92-file/1,578-test compiler suite; semantic/compiler architecture checks. |
| Gate 3 authorized CuTe view-copy lowering | `partial` | One exact authorized frontend artifact plus a closed entry/storage request lowers through semantic-core's sole `createVerifiedViewCopyArtifacts` constructor. The initial profile requires f32 with exact 32-bit device ABI/alignment, two distinct non-null global parameter pointers with source-only const qualification, synchronous portable 32-bit copy with exact read/write effects, equal positive flat static rank-2 or rank-3 source/destination layouts, reject-invalid-source semantics, and forbid overlap. Allocation byte lengths and view offsets are explicit host facts, aligned and range-checked against the verified affine address spans; CuTe `cosize` never grants pointer capacity. Semantic-core owns IDs, alias sets, effects, and canonical hashes. The canonical CPU reference executes both pinned transposes bit-for-bit with nonzero offsets and untouched canaries, while rank mismatch and rank 4 fail closed. The exact observed Worker candidate plus independent producer reaches this same canonical seam without folding storage facts into candidate or authorization identity, and the two realm-neutral payloads execute through required actual WebGPU. | Replace the synthetic browser authority fixture and producer material with a valid production Worker compile and externally rooted producer. Dynamic layouts, rank 4, and broader tensor-engine profiles remain unsupported. | Commits `85631464`, `a8a861e2`, and `dca33be5`; pinned rank-2 hashes `5ade6e06...`/`64dc9d67...` and rank-3 hashes `c2b5e8a0...`/`e335ea9d...`; complete compiler suite 95 files/1,591 tests; required actual-WebGPU 2 cases; exact-source CI `29811673981` passed all eight jobs. The current browser authority remains synthetic with `productionBrowserCompileObserved=false`. |
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
- At the Gate 0 freeze, Grad's `bf16`/`bfloat16` path was f32 substitution.
  View/materialize
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
| D-083 | 2026-07-17 | accepted | Build the Clang frontend as an Emscripten ES-module factory plus Wasm sidecar, then deterministically bundle the exact generated factory into the package-owned Worker and instantiate only the host-verified Wasm bytes through its supplied `instantiateWasm` hook. A bare standalone/WASI `.wasm` is not an equivalent output because it omits the Emscripten support factory required by this integration. The user's C++ is parsed by Clang-Wasm into BrowserGrad semantic artifacts; it is not linked as a user Wasm application. Until the closed VFS bridge, device-first/host-second CUDA actions, and canonical artifact-v3 writer exist, the checked-in tracer must keep `artifact-ready` unreachable. | This preserves the browser-only product path without mistaking a maintainer Docker build environment for runtime architecture. It also prevents an ambient sidecar fetch, an unverified generated-JavaScript dependency, or a review-only AST trace from self-authorizing browser execution. |
| D-084 | 2026-07-17 | accepted | Keep production browser-Worker execution unconditionally capability-blocked before inspecting caller input, options, or ambient browser globals until a self-contained package-owned Worker module and transferred Emscripten factory are internally pinned. Caller-prepared profile/module bytes cannot establish package-code authority. The controller lifecycle may be exercised only through a disjoint test simulation type with execution false and separate brand/store/unwrap. Future production enablement must capture and brand-check native browser intrinsics, verify the internally owned module bytes before Blob construction, enforce the absolute prepared deadline independently of timer task ordering, own one nonce/event source, and complete timer/listener/terminate/revoke cleanup before issuing execution evidence. | A correct controller state machine is not evidence when its executable Worker is absent. Allowing caller-selected module bytes or mutable same-realm globals would make the event source self-authorizing; returning production-shaped fields from a fake platform would leak false evidence to field/type consumers even if a WeakMap unwrap rejected it. A hard capability gate preserves the completed lifecycle design without fabricating the missing product boundary. |
| D-085 | 2026-07-17 | accepted | Treat header-pack preparation as a complete-profile, exact-inventory, offline content-authority boundary—not as acquisition, build, license, output, reproducibility, or release authority. Bind every non-source include root to the exact prepared build lock/profile and expected content-set identity; require caller-supplied bytes to match the prepared path/length/hash inventory; emit only the existing canonical closed VFS format; and re-inspect the encoded bytes before returning an opaque non-release result. Use explicit closed canonical limits at both content-set and selection-manifest layers, translate all schema/hash failures into header-pack diagnostics, and preflight a conservative live byte-copy projection including the inspector's original pack, snapshot, and full SHA input copy. Keep exact notice-byte verification and an externally reviewed per-file license map as permanent blockers until separate evidence closes them. | Corpus-minimal header closures would make compilation behavior depend on the current fixture set and force later pack redesign. Treating build-lock notice policy as file-level license review would self-authorize redistribution. Generic semantic-core JSON defaults do not compose with the larger bounded inventories, and two-pack memory arithmetic misses the hash copy. One exact full-profile offline contract keeps acquisition, licensing, runtime mounting, and release evidence separate while preventing latent scale and memory failures. |
| D-086 | 2026-07-17 | accepted | Treat build-source preparation and Wasm-sidecar installation as two observation-only authorities. The executor must derive its plan from the opaque build lock, verify the exact recursive source closure by byte length, SHA-256, canonical source-set hash, and staged inode identity, and stage owned read-only snapshots before any future process execution. Sidecar installation copies only the generated `.wasm` bytes to the lock-derived destination through a no-clobber commit; it never distributes the generated factory module. Filesystem admission requires trusted POSIX ancestry, current-user-owned non-group/world-writable work roots, identity-bound cleanup, and an explicit same-UID single-writer boundary because Node exposes no `openat`/`linkat`/`unlinkat` authority. The result keeps WebAssembly validation, ABI conformance, build execution, output identity, reproducibility, provenance, and release facts false. | Source hashes enforced only by tests leave the actual build boundary open to substitution. A plain copy can clobber an existing asset, expose partial bytes, leak a replaced temporary inode, or delete unrelated paths during cleanup. Conversely, describing byte installation as build or release proof would let arbitrary external bytes self-authorize. Separate opaque observations plus exact filesystem and authority boundaries close the implementable pre-build seam without inventing evidence. |
| D-087 | 2026-07-17 | accepted | Prepare one memoized runtime-v1 input frame per pending invocation from cached, bounded canonical profile/request regions. Recheck single-use invocation liveness after asynchronous hashing and on every byte copy; terminalization revokes every outstanding frame. Before copying or hashing executable assets, derive one complete artifact-verification contract: every artifact-v3 collection ceiling is explicit, profile/request-derived ceilings must fit fixed verifier maxima, and one request-bounded semantic-core decode budget is threaded through decoding, canonicalization, semantic verification, source-entity IDs, input/semantic-pass/shared-surface hashes, artifact hashes, and diagnostic hashes. Pure positive tests prove only caller-frame consistency. | The former mapping named a nonexistent artifact field, allowed request ceilings that the verifier could not represent, and left hidden generic JSON limits in internal hashes. Repeat frame preparation retained duplicate multi-megabyte buffers, while terminalization during hashing could mint stale authority. One cached frame and one end-to-end verifier budget prevent delayed production failures and denial-of-service amplification before the real Worker is enabled. |
| D-088 | 2026-07-17 | accepted | Record VFS `openedFiles` only after a successful content read commits to stable Wasm memory. Opening, stat/directory lookup, failed reads, and zero-length reads of nonempty files do not enter the set; a zero-length read of an actually empty file does. Repeated or partial successful reads count the exact full-file identity and logical bytes once. This is a successful-read observation, not proof that Clang used the bytes semantically; browser evidence must still compare it with each artifact semantic pass's `openedFileIds` before claiming contribution to preprocessing or Sema. | Counting open calls as input evidence included probes and unused handles, while counting transferred ranges understated full-file logical reservations. Calling the VFS set semantic contribution would overclaim what the host import can observe. A post-commit unique successful-read set preserves exact accounting and leaves semantic ownership to the artifact/pass verifier. |
| D-089 | 2026-07-17 | accepted | Implement the C++ `llvm::vfs::FileSystem` adapter directly over the exact six runtime-ABI imports. Canonicalize and bound every absolute UTF-8 path, materialize directory entries lazily, reserve bounded handles, close every acquired handle exactly once, guard post-open construction with RAII, and permanently poison the adapter after any close failure. A source-level adapter is not runtime evidence until the pinned Emscripten build compiles it and the package Worker executes it. | An eager tree duplicates the complete VFS and creates avoidable Worker-heap amplification. Continuing after a close failure can leak logical handles and make limits bypassable. Post-open metadata or allocation failures otherwise leak the newly acquired handle. Lazy traversal plus fail-stop exact-close semantics keeps the imported bridge bounded and recoverably false rather than silently degraded. |
| D-090 | 2026-07-17 | accepted | Keep Wasm-local runtime metrics as a separate versioned observation from browser-Worker evidence. Record exact linear-memory pages/capacity and allocator-request counters with fixed record size and consistency equations; distinguish input-frame copy, frontend extraction, and result-frame copy; enforce fixed-stack and profile/ABI coexistence. Label the current source `record-exact-unverified-producer` and local `performance.now` values `local-performance-now`. Do not infer JS heap, resident memory, CPU, process, Worker termination, or browser provenance. | Linear-memory pages and module allocator counters are measurable, but they cannot establish browser-process resource use or Worker lifecycle. Overclaiming these as resident/heap metrics would become an API compatibility debt. The runtime ABI currently exposes no producer record, so explicit unverified status prevents a test helper from becoming false execution evidence. |
| D-091 | 2026-07-17 | accepted | Bind one package-owned Worker runtime reservation to the exact opaque invocation, canonical input frame, verified Wasm bytes, and active VFS session before any future instantiation. Snapshot/rehash bytes, reject reuse and hostile accessors, recheck liveness after asynchronous hashing, and terminalize both VFS and invocation on every stop with aggregate cleanup reporting. Keep the runtime capability-blocked until the package owns self-contained Worker bytes, a generated Emscripten factory, and independently reviewed first-build projections; expose no generic loader, fetch, factory, or caller-selected module seam. | A partially wired runtime can otherwise turn caller bytes or ambient loaders into executable authority. Cleanup that retains only the first failure hides leaked state, and reservation races can let one invocation execute twice. A closed blocked binding proves ownership and teardown without fabricating the missing product executable. |
| D-092 | 2026-07-17 | accepted | Advance the strict compiler runtime ABI to 1.1 for one additive ninth C export, `bg_cpp_cute_allocator_metrics_pointer`, over a fixed 72-byte module-global record. The pointer is nonzero, stable for one module instance, host-read-only, and sampled only between synchronous calls from one unshared memory epoch. Counters use caller-requested bytes before rounding, keep stack/static/JS/VFS-logical categories separate, fail before u64 wrap, and explicitly define zero-byte creation, `free(nullptr)`, `realloc(nullptr,n)`, successful/failed nonzero realloc, and `realloc(p,0)`. The current C++ extractor remains ABI 1.0; its locked command stays buildable and a release blocker names the missing export, producer, and behavioral conformance. | Silently changing strict ABI 1.0 or adding an undefined linker export would make the canonical build internally inconsistent. Leaving realloc and zero-size behavior implementation-defined allows two producers to satisfy the same current-byte equation while emitting incompatible cumulative/count evidence. A minor-versioned design authority plus explicit mismatch blocker closes the contract without fabricating an implementation. |
| D-093 | 2026-07-17 | accepted | Reconstruct VFS and invocation authority inside the dedicated Worker from exact canonical transferred bytes; do not transfer host-realm opaque WeakMap objects or a VFS session already bound to host-realm memory. Split VFS setup into a memory-independent verified mount and a one-time binding to `instance.exports.memory`. Imports fail closed before binding. Host acquisition/cache remains outside the Worker; transferred source/pack buffers, profile/request/runtime-ABI/manifest bytes, and Clang-Wasm bytes are reverified before local authority is minted. | A nonshared `WebAssembly.Memory`, JavaScript closures, and module-local opaque authority cannot serve as a cross-realm protocol. Bundling the factory before correcting this ownership seam would force a later Worker/VFS refactor or tempt structural clones to masquerade as authority. |
| D-094 | 2026-07-17 | accepted | Split the BrowserGrad C++ extractor before adding frame parsing, CUDA pass coordination, metrics hooks, and artifact serialization. Use separate runtime/lifecycle, imported-VFS, Clang-action/instrumentation, and artifact-v3 translation units; keep the top-level extractor as composition only. Add a metrics translation unit only when it owns the real ABI 1.1 producer. Isolate LLVM-version-sensitive CUDA/tooling APIs in the Clang-action unit and repin the complete source-set closure atomically. | The current source already exceeds a reviewable composition boundary. Adding several independent state machines and canonical serialization to one file would create hidden coupling between ABI lifetime, VFS failure, Clang pass ownership, allocator accounting, and artifact identity. An empty future-facing unit would add dead build surface without establishing ownership. |
| D-095 | 2026-07-17 | accepted | Treat the memory-independent VFS mount and one-time memory bind as same-realm capabilities only. Capture authority-map, reflection, memory, and cleanup intrinsics at module initialization; reject cloned projections and exact-memory subclasses; sever authority and settle terminal state before best-effort destructive cleanup. Do not call this Worker-local until the Worker reconstructs installation, request, runtime-ABI, and pack authorities from verified transferred bytes. | Splitting memory binding fixes module-instantiation order but does not make WeakMap authority transferable. Mutable intrinsics could otherwise forge stored authorities or interrupt cleanup before terminalization. Keeping the claim narrow prevents the intermediate seam from becoming a false cross-realm protocol. |
| D-096 | 2026-07-17 | accepted | Create the exact six VFS import functions once from the verified mount, before Wasm instantiation, and keep their references stable through one-time binding to exact exported memory. Before binding they return only `sessionClosed`; bind activates the same references; discard/close are terminal. | An Emscripten module needs its import object before the instance exposes memory. Creating imports only after memory binding forms an impossible instantiation cycle, while replacing functions after binding can leave the instance calling stale closures. Stable pre-bind fail-closed imports remove the cycle without weakening memory ownership. |
| D-097 | 2026-07-17 | accepted | Implement ABI 1.1 requested-byte allocator metrics with strong overrides only for one closed supported libc entrypoint set and route raw allocation through pinned `emscripten_builtin_*` bypasses. Store pointer-to-requested-size metadata in a separate raw open-addressed table, exclude its bytes from counters, reject counter wrap with sticky poison, and keep unobserved grouped/in-place allocator APIs explicitly forbidden rather than approximating their semantics. | Linker wrapping can miss internal libc aliases, recursive metadata allocation can corrupt both the producer and its measurements, and fake grouped or in-place behavior would silently substitute incompatible allocator semantics. The exact ingress/bypass closure plus poison-on-inconsistency is reviewable and can later be proven against the pinned object/final-Wasm call graph. |
| D-098 | 2026-07-17 | accepted | Separate allocator proof into source/contract closure, native behavioral-model evidence, pinned Emscripten object/final-Wasm call-graph evidence, and executed-Wasm ABI behavior. Native tests may validate accounting state transitions and corruption handling, but cannot clear the runtime metrics release blocker. | Compiling the model with a host allocator does not prove Emscripten weak/strong symbol resolution, builtin bypass reachability, complete allocation ingress, exported pointer shape, or actual Wasm memory behavior. Keeping the proof layers explicit prevents source tests from becoming false production conformance. |
| D-099 | 2026-07-17 | accepted | Cross the host-to-Worker boundary with exactly one destructively transferred canonical message. The message owns bounded canonical invocation, profile, request, and asset-manifest regions plus unique full-span asset and source buffers; it carries neither opaque host-realm authorities, Worker-module bytes, generated-factory authority, nor a network/loading seam. The receiving realm must reverify the complete closure and reconstruct every invocation, asset, request, runtime-ABI, raw-Wasm, input-frame, and VFS-mount authority locally. | WeakMap-backed authority and JavaScript closures cannot cross realms. Transferring caller-selected executable loaders or package Worker bytes would let data authorize code. One closed destructive transfer keeps ownership, memory amplification, replay, and executable-code authority reviewable without claiming that an actual Worker has consumed it. |
| D-100 | 2026-07-17 | accepted | The package runtime adopts one reconstructed realm input containing exact invocation, canonical frame, verified Wasm, pre-bind VFS mount, and stable imports—not a memory-bound session. Adoption is one-time, rechecks hashes and liveness after asynchronous work, and preserves the same import references until a future reviewed factory binds only `instance.exports.memory`. Preparation failure and explicit abandonment use `abandoned`; a start blocked by missing package-owned Worker/factory capability uses `worker-unavailable`. All paths terminalize invocation and mount with aggregate cleanup reporting. | Binding memory before Emscripten instantiation recreates the import cycle, while accepting a host-bound session recreates the cross-realm authority defect. Phase-accurate terminal reasons prevent lifecycle telemetry from claiming a missing Worker when the caller simply abandoned work, and deterministic aggregate cleanup prevents a secondary failure from hiding retained authority. |
| D-101 | 2026-07-17 | accepted | Use one package Worker entry and one controller terminal protocol for both future success and current infrastructure failure. The entry removes its sole message listener before asynchronous work, consumes only the canonical transfer, derives terminal identity only after reconstruction, and routes pre-identity rejection through the Worker `error` event rather than self-attesting an invocation. After trusted identity exists it emits exactly one discriminated outcome: standalone transferable control/artifact bytes on success, or bounded phase/code/path fields on failure with execution and lowering fixed false. Entry owns cleanup only while the reconstructed realm input is still prepared; runtime owns cleanup after adoption. The controller preserves authenticated failure fields in a dedicated typed error and never requires clients to parse an error string. | A failure-only callback, generic launch message, or flattened error string would force an API redesign when Wasm execution becomes available. Unconditional entry cleanup would double-discard runtime-owned authority, while no pre-adoption settlement would leak it. A success-capable one-shot protocol and explicit ownership handoff close those future refactor seams without enabling the still-unreviewed runtime. |
| D-102 | 2026-07-17 | accepted | Order canonical set-like wire strings with explicit lexicographic UTF-16 code-unit comparison, matching canonical JSON object-key order. Locale collation is forbidden in profile, request, artifact, and provenance validation or normalization. | `localeCompare` depends on ambient locale/ICU data and cannot be reproduced reliably by the C++/Wasm decoder. A shared comparator plus Unicode vectors makes order and identity portable without restricting valid normalized POSIX virtual paths to ASCII. |
| D-103 | 2026-07-17 | accepted | Runtime ABI 1.1 owns exact per-region canonical-JSON decode budgets and accounting. Native frame work must validate canonical bytes before allocation or VFS access, use bounded locale-free UTF-16 key order and safe-integer numbers, and recompute identities through BrowserGrad-owned SHA-256. The primitive parser has a compiled recursion ceiling of 256 while runtime v1 pins 128. Canonical validation and hashing are part of the exact extractor source/build lock and Linux CI gate, but they grant no typed profile/request, Clang, Wasm, or artifact authority alone. | Decoder limits otherwise become an unversioned implementation detail and valid producer bytes can trigger unbounded native work. Direct canonical-byte validation avoids a second DOM/reserialization truth. A reviewed local SHA implementation avoids LLVM/OpenSSL/environment dependencies. Explicit proof-layer limits prevent host-native tests from masquerading as executed-Wasm conformance. |
| D-104 | 2026-07-17 | accepted | Bind every C++/CuTe producer to one canonical semantic-adapter manifest for exact Clang 22.1.8. Profile 2.5 and compilation contract 1.1 select a reject-only temporal-macro policy, compiler-default warning baseline, and closed namespaced warning registry. Generate the native warning/temporal tables deterministically from that TypeScript authority; accept typed policy options only and emit owning argv elements, never raw shell or caller argv. Give every future CUDA semantic pass a fresh imported-VFS observer and preprocessor-policy state. These source-closed primitives grant no Clang invocation or artifact authority until a sealed full-argv builder and pass lifecycle wire them into the production action. | Compiler warning names, predefined temporal macros, and implicit driver defaults are version-sensitive semantics. Duplicated tables or a generic argv escape hatch would let browser and AOT producers drift, consult wall-clock/file-mtime state, or access ambient toolchain paths while claiming one compilation contract. One manifest plus generated native projection prevents policy drift without falsely treating helper compilation as executed-Clang evidence. |
| D-105 | 2026-07-19 | accepted | Treat the canonical per-file header distribution manifest as exact external-review input, never as self-approval. It must bind live extraction, inventory, persisted pack, package-notice, upstream evidence, and CUDA-index authorities in one process; write the build-lock-declared `license-inventory.json` through a private no-clobber path; and keep external review, distribution authorization, and release claims literal false. | A generated file map is necessary engineering evidence but cannot decide redistribution rights. Separating deterministic review input from a future external approval authority prevents the harness from converting its own hashes or license text into a legal conclusion. |
| D-106 | 2026-07-19 | accepted | Materialize declared distribution notices only from verifier-retained exact byte snapshots. Use one shared bounded private-output primitive that verifies the complete initial file and directory tree, rehashes every immutable existing file, writes each new file with no-follow/no-clobber semantics, syncs it, independently rereads it, verifies the complete final tree, and cleans only identities created by the failed operation. Aggregate metadata must be printable ASCII and bind the exact lock/component/hash/length records while embedding untouched notice bytes. The result remains non-approval authority with every legal and release claim false. | Reopening resource paths after verification would create a check/use gap, caller-provided bytes would let projections mint false notice output, and duplicated filesystem code had already become harness maintenance debt. One byte-retaining authority plus one generic fail-closed output primitive preserves exactness, reduces duplicated code, and cannot convert package policy into external legal approval. |
| D-107 | 2026-07-19 | accepted | Prove header-distribution reproducibility only from two distinct live pipeline authorities rooted in four canonical, pairwise non-overlapping output paths. Rehash all 17 immutable files in both private trees, verify the exact file/directory sets before and after hashing, recheck hashed inode identities at the terminal boundary, and compare path/hash/length records. The reproducibility identity excludes host paths. Keep the claim scoped to five packs, license inventory, and eleven notice outputs; full distribution, license approval, provenance, Worker execution, and release remain false. | Comparing serialized projections or trusting pipeline IDs alone would miss post-materialization drift and allow forged evidence. Calling the subset the complete distribution would incorrectly include neither the Wasm/factory/Worker nor all declared release outputs and detached evidence. Live authorities plus terminal rehashing prove the available deterministic subset without widening its meaning. |
| D-108 | 2026-07-19 | accepted | Package-pin a path-independent projection of the exact two-root header-distribution observation. Admit only the exact package bytes, bind the current build-input lock and resource hash, validate the canonical 17-output projection, and independently rederive the output-verification and reproducibility identities before issuing an opaque authority. Keep external license review, distribution approval, signed provenance, full-distribution reproducibility, Worker execution, and release literal false. | Live same-process authorities prove the filesystem operation but cannot be reconstructed by a later package consumer. A byte-pinned, path-free technical record preserves the exact result without serializing host roots or allowing caller-shaped evidence to mint authority, while the narrow claim prevents reproducibility from being mistaken for legal or release approval. |
| D-109 | 2026-07-19 | accepted | Run the complete JavaScript build-plan verification exactly once in an independent job concurrently with the expensive Clang-Wasm matrix. Keep exact package/runtime-closure materialization inside every build job, and require the final reproducibility verifier to depend on both the verification job and both clean builds. | The same 86-to-134-second JavaScript test phase previously ran serially before each compiler build even though it neither produced native prerequisites nor required native outputs. Moving only verification off the critical path preserves each exact runtime closure and makes either branch blocking while eliminating duplicated serial latency. |
| D-110 | 2026-07-19 | accepted | Scope workflow concurrency by exact ref and execution mode. Cancel only superseded `fast-validation` runs; never cancel clean validation or reproducibility, and allow independent modes to use separate runners concurrently. | One ref-wide non-cancelling group forced a seconds-to-minutes feedback request to queue behind a 25-to-45-minute evidence build. Mode isolation removes that queue dependency without sharing filesystem state, caches, or authority between runs; preserving non-cancellation for evidence modes avoids incomplete clean/repro records. |
| D-111 | 2026-07-19 | accepted | Attach a bounded backpressured bridge to a spawned normalizer's stdout before starting asynchronous parser setup. Await child close, stdout transport, strict consumer, and bounded stderr settlement; destroy the bridge and terminate the process group on consumer failure. Keep the strict two-zero-block tar terminator and owned-output cleanup unchanged. | Linux/Node 24 repeatedly lost the tail of a small fast-exiting fixture while a larger output that held the pipe open passed. Buffering only up to the bridge high-water mark closes that startup race without unbounded capture, while separate transfer and consumer settlement preserves fail-closed diagnostics and cleanup. |
| D-112 | 2026-07-21 | accepted | Close Gate 4 with one narrow portable scalar profile. Canonical logical GEMM semantics own operands, layouts, effects, tile meaning, accumulation order, and numerical policy but contain no schedule or backend vocabulary. A separately verified schedule owns physical tiles, workgroup mapping, cooperative staging, uniform barrier participation, scalar vectors, and boundary masks. Exact-input certification binds retained immutable snapshots to the logical and schedule-specialization hashes. WebGPU may report only portable re-legalization and bit-exact certified-input preservation; it cannot claim general f32, native MMA, resident-buffer provenance, preserved CUDA/CuTe schedule, or source compatibility. | A narrow fully evidenced profile closes the architectural separation without turning one shader or typed producer fixture into a general GEMM claim. Backend-neutral identity remains reusable, uniformity and masks remain fail-closed, and later dtype/layout/native schedules require new explicit profiles rather than silently widening the proved one. |
| D-113 | 2026-07-21 | accepted | Define Gate 5 logical attention first as one closed `browsergrad.kernel.attention-forward@1` artifact over verified rank-4 views. It owns exact scale derivation, causal/non-causal mask meaning, stable-softmax phases, finite-domain requirements, numerical/comparison policy, effects, and VJP refusal. It owns no query/key tiles, staging, barriers, backend mapping, frontend provenance, or FlashAttention claim. | Keeping logical attention independent from its physical online K/V-tile schedule prevents the existing row-wise baseline or one future WGSL program from becoming the semantic source of truth. Exact scale/profile revalidation and closed fields make forged positive scales and backend-shaped payloads fail before authority is minted. |
| D-114 | 2026-07-21 | accepted | Represent the first attention schedule as a separate closed `browsergrad.schedule.attention-online-kv-tile@1` artifact bound to one exact verified attention-forward hash. It owns physical query/key rows, increasing key traversal, cooperative single-buffered K/V staging, tile-wise running maximum/denominator/weighted-value rescaling, uniform all-lane barriers, scalar vectors, and boundary/logical-mask placement before online-state updates. It owns no logical dtype, scale, view, comparison, autodiff, backend, execution, preservation, performance, or named fused-kernel claim. | This keeps physical online-softmax realization replaceable without changing logical identity, prevents zero-filled tail staging from becoming an unintended valid score, and makes divergent barrier control or post-exponential masking unverifiable rather than relying on backend convention. |
| D-115 | 2026-07-21 | accepted | Make the initial attention CPU oracle schedule-independent and dense-row-major only. Prepare and hash logical addresses before execution; bound elements, scalar work, evaluator steps, preparation, execution, and cancellation; copy admitted fixed unshared Q/K/V bytes before yielding; reject non-finite inputs or derived values; compute the declared stable-softmax phases with stepwise f32 rounding; and commit destination bytes only after complete success. Use one shared native-buffer admission helper for GEMM and attention. | The CPU oracle must prove logical meaning rather than imitate the tiled schedule. Private snapshots close mutation races across cooperative yields, delayed commit prevents partial results on domain/time/abort failure, and shared binding admission prevents another divergent parser for accessors, subclasses, shared/resizable/detached storage, alignment, and overlap. |
| D-116 | 2026-07-21 | accepted | Compose attention schedule specialization only from one exact module-authorized logical specialization, its verified logical artifact, and a schedule artifact bound to that exact logical hash. Derive and resource-bound workgroup invocations, K/V staging bytes, per-invocation query/output private elements, key-tile count, and x/y/z dispatch geometry; hash the full resolved projection. Keep device limits, target legality, numerical preservation, and execution out of semantic-core. | Backends need one reusable geometry authority without repeating logical address proof or trusting caller arithmetic. Explicit budgets prevent oversized staging/private/dispatch plans from reaching shader generation, while withholding device/preservation claims keeps a semantically valid schedule from masquerading as portable WebGPU evidence. |
| D-117 | 2026-07-21 | accepted | Lower the exact attention logical/schedule composition into a separately named scalar WebGPU preparation profile. One invocation owns one query/output row, all lanes cooperatively stage K/V rows, causal and tail keys are excluded before the tile maximum or recurrence, and every lane crosses both barriers. Bind the emitted WGSL and bounded transient/resource projection into backend preparation identity. Report `block-tiled-kv-online-softmax-forward` and `portable-relegalized`, not FlashAttention-v2, source schedule preservation, device execution, numerical preservation, or performance. | A real block-tiled algorithm can be represented honestly before device evidence exists without reusing the frozen row-wise baseline or moving backend facts into semantic-core. Exact bit scale embedding, no early return, zero-filled staging, and pre-update masks make the generated control/data flow auditable; withholding stronger authority keeps code generation from becoming an execution claim. |
| D-118 | 2026-07-21 | accepted | Execute prepared semantic attention only from private exact fixed unshared finite-f32 Q/K/V snapshots retained before the first yield or device access. Admit device limits before pipeline work, allow only one operation in flight per device, run preparation/dispatch/readback under LIFO validation/OOM/internal scopes, race device loss and bounded cancellation/timeout, and reject non-finite or incomplete output. Required correctness evidence covers causal and non-causal irregular rank-4 input under two schedules and compares every output element with the semantic-core policy. Keep `numericalPreservation=requires-declared-policy-comparison` in the execution trace. | Mutable caller arrays, unscoped asynchronous WebGPU errors, or a trace that self-certifies numerical preservation would make evidence forgeable or stale. Private pre-yield snapshots close mutation races, resource/device admission prevents doomed allocation work, scoped cleanup prevents latent device errors from escaping, and an external complete-output comparator keeps execution fact separate from preservation fact. |
| D-119 | 2026-07-21 | accepted | Close the initial Gate 5 profile with a separate required performance observation, not a correctness test or release threshold. Measure the named production block-tiled host API and frozen row-wise baseline on the same non-causal f32 workload after 16 warmups with 20 alternating paired samples. Include input upload, cached-pipeline dispatch, complete readback, queue drain, and each API's real validation/resource lifecycle; retain raw samples, named browser/device/configuration, and those lifecycle differences. Make no superiority or regression claim. | The two public paths do not have identical validation and output-buffer ownership, so dispatch-only timing would require a new trusted resident authority and pretending the current boundaries are identical would be misleading. End-to-end observational records expose the actual production costs while separate untimed CPU preflight and the required correctness suite prevent performance data from becoming semantic evidence. |
| D-120 | 2026-07-22 | accepted | Retire public `Tensor.expand` from the frozen JIT `CUSTOM` inventory and emit the existing typed `BROADCAST_TO` primitive. Revalidate its closed shape/dtype/broadcast contract at construction and every CPU, transform, export, and tensor-plan boundary; add the symbolic unbroadcast VJP; retain owning CPU materialization and explicit materializing/resident routes. Narrow the machine freeze under ADR-0002 rather than repinning the old callback. | The public surface was needlessly disconnected from a typed primitive already implemented across CPU, vmap, ONNX, tensor-plan, and WebGPU. Reusing that semantic operation removes an opaque callback, prevents float-to-int shape truncation and post-construction arg drift, and advances Gate 6 without adding a second backend-shaped path. |
| D-121 | 2026-07-22 | accepted | Add one bounded versioned JIT framework-operation registry whose records are admitted only when bound one-to-one to executable typed validators. Generate `framework_operation_support()` from those admitted records, return detached data, distinguish backend-profile eligibility from availability/evidence, and machine-check the exact partition of the original opaque operation IDs. | Support claims cannot be inferred from method presence, opcode allowlists, prose, or compatibility inventories. One executable registry prevents CPU/autograd/transform/export/backend decisions from drifting across public tables while preserving honest absence and migration history. |
| D-122 | 2026-07-22 | accepted | Migrate `Tensor.abs` and `Tensor.sign` together to typed `ABS` and `SIGN` under one real-numeric unary contract. Preserve exact shape/dtype, reject bool, revalidate at every admitted boundary, define zero-at-origin abs VJP and zero sign VJP, support leading-axis vmap and direct ONNX export, and explicitly refuse tensor-plan/WebGPU until a portable lowering exists. | Abs symbolic differentiation depends on sign, so migrating the pair avoids an opaque derivative. One registry-bound contract closes CPU/closure/symbolic/transform/export meaning without converting host execution into a false device claim or widening the plan allowlist. |
| D-123 | 2026-07-22 | accepted | Migrate `Tensor.sin` and `Tensor.cos` together to typed `SIN` and `COS` under one floating-only unary contract. Preserve exact shape and float16/32/64 dtype, reject bool/integer inputs, revalidate at every admitted boundary, express both mutually dependent symbolic derivatives without `CUSTOM`, support leading-axis vmap and direct ONNX export, and explicitly refuse tensor-plan/WebGPU until a portable lowering exists. | The opaque callbacks declared integer outputs but NumPy realized float64, violating IR truth, and either derivative requires the other operation. Early rejection plus one registry-bound pair removes that drift and opaque derivative without making an unevidenced device claim. |
| D-124 | 2026-07-22 | accepted | Migrate `Tensor.clamp` to typed `CLAMP` with one closed pair of finite optional float bounds and a floating-only shape/dtype-preserving profile. Admit only exact built-in/NumPy scalar types without invoking arbitrary conversion hooks; revalidate at CPU/VJP/vmap/ONNX boundaries; define the inclusive-bound mask VJP; emit ONNX `Clip` optional inputs; and explicitly refuse tensor-plan/WebGPU until a portable lowering exists. | The opaque callback exposed host coercion, hid its bounds from IR, blocked transforms, and could declare integer dtype while NumPy promoted the result. One normalized registry-bound contract makes the piecewise meaning and refusals executable without broadening scalar authority or claiming device support. |
| D-125 | 2026-07-22 | accepted | Migrate `Tensor.flip` to typed `FLIP` with one exact built-in/NumPy integer axis, one negative-axis normalization, and no arbitrary conversion hooks or modulo wrapping. Revalidate exact arity, closed arguments, shape/dtype preservation, and normalized range at construction and CPU/VJP/vmap/ONNX boundaries; return an owning CPU copy; use involutive symbolic VJP; shift the logical axis under vmap; emit ONNX `Slice`; and explicitly refuse the negative-stride tensor-plan/WebGPU profile. | The callback silently accepted bool and wrapped every integer axis, hid reversal from transforms and export, and left its negative-stride meaning outside typed IR. A closed registry-bound operation preserves valid behavior while making malformed axes and unsupported portable-device semantics fail at their true boundaries. |
| D-126 | 2026-07-22 | accepted | Migrate JIT `Tensor.repeat` to typed `REPEAT` with exact bounded tile multipliers and left-rank padding. Revalidate its closed shape/dtype contract at CPU/VJP/vmap/ONNX boundaries; use an owning CPU tile and block-sum VJP; prepend a unit repeat under vmap; emit ONNX `Tile`; and refuse tensor-plan/WebGPU until canonical tile/index layout semantics exist. Make Grad consume the same conformance fixture and preserve input dtype. | The opaque callback invoked arbitrary integer coercion, admitted resource-hostile empty-output multipliers, hid its derivative, and left export/backend decisions implicit; Grad independently cast every result to float32. One bounded registry contract plus shared eager/lazy fixture removes those correctness and security gaps without adding a backend-shaped modulo path. |
| D-127 | 2026-07-22 | accepted | Migrate JIT `Tensor.repeat_interleave` to typed `REPEAT_INTERLEAVE` with one exact bounded repeat count and normalized selected axis. Revalidate its closed shape/dtype contract at CPU/VJP/vmap/ONNX boundaries; use an owning CPU result and selected-axis block-sum VJP; shift the axis under vmap; emit an exact ONNX `Unsqueeze`/`Tile`/`Reshape` decomposition; and refuse tensor-plan/WebGPU until canonical selected-axis replication layout semantics exist. Make Grad consume the same conformance fixture and preserve output and gradient dtype. | The opaque callback invoked arbitrary integer coercion, wrapped axes through modulo, admitted negative/resource-hostile counts, hid its derivative, and left export/backend decisions implicit; Grad independently cast results and gradients to float32. One bounded registry contract plus shared eager/lazy fixture removes those correctness and security gaps without adding a source-shaped backend handler. |
| D-128 | 2026-07-22 | accepted | Migrate JIT `Tensor.prod` to typed `PROD` with one canonical static axis tuple and exact keepdims flag. Revalidate its closed shape/dtype contract at CPU/VJP/vmap/ONNX boundaries; return owning dtype-preserving scalar/tensor CPU arrays; use a zero-aware product VJP; shift axes under vmap; emit ONNX `ReduceProd` for its exact dtype profile; and refuse tensor-plan/WebGPU until a portable product-reduction lowering exists. Make Grad consume the same conformance fixture and zero-aware derivative. | The opaque callback admitted arbitrary axis iteration/coercion, wrapped malformed axes, failed scalar CPU realization, hid its derivative, and returned an incorrect zero gradient for one-zero groups; Grad independently cast results and gradients to float32. One registry-bound product operation closes the reduction meaning and correctness gap without treating the existing generic reduction plan as unevidenced device support. |
| D-129 | 2026-07-22 | accepted | Migrate JIT `Tensor.gather` to the existing typed `INDEX` seam with exact normalized axis, same-rank int64 index, non-gather extent, runtime bounds, and source-dtype contracts. Return an owning CPU result; use deterministic `SCATTER_ADD` closure/symbolic VJP; map source and index together under vmap; emit ONNX `GatherElements`; and refuse tensor-plan/WebGPU until deterministic bounds-checked index/scatter lowering exists. Make Grad consume the same conformance fixture and preserve output/gradient dtype. | The callback coerced axes and indices, inherited NumPy negative-index wrapping, hid duplicate-index accumulation from transforms/export, and silently cast eager values and gradients to float32. Completing the pre-existing INDEX/SCATTER_ADD seam removes those correctness and security gaps without inventing a second indexing path or misreporting structural plan admission as device execution. |
| D-130 | 2026-07-22 | accepted | Migrate JIT `Tensor.var` to typed `VAR` with canonical static axes, exact signed 32-bit correction, exact keepdims, and floating dtype preservation. Return owning CPU scalar/tensor arrays; use the centered correction-aware closure/symbolic VJP; shift axes under vmap; emit an exact float32 ONNX variance decomposition; and refuse tensor-plan/WebGPU until a portable reduction exists. Make Grad consume the same correction, dtype, and refusal fixture. | The callback admitted coercive or wrapped axes, exposed only a legacy boolean switch, hid the derivative and reduction from transforms/export, drifted integer results to float64, and failed scalar realization; Grad independently cast every result and gradient to float32. One closed registry contract removes that correctness and security debt without misreporting generic reduction syntax as device support. |
| D-131 | 2026-07-22 | accepted | Migrate JIT `Tensor.masked_fill` to canonical `WHERE(mask, fill, source)` with an actual bool tensor mask that broadcasts into but cannot enlarge the source, and an exact scalar `CONST` normalized to source dtype. Return an owning CPU result; add mask-complement closure/symbolic VJP and generic `WHERE` VJP; batch a leading mapped source with captured or mapped masks; emit ONNX `Where`; and refuse tensor-plan/WebGPU until portable masked selection exists. Make Grad consume the same fixture and preserve output/gradient dtype. | The callback coerced arbitrary masks and host values, allowed output-shape enlargement, hid selection from functional transforms/export, and left device meaning implicit; Grad independently cast results and gradients to float32. Reusing and closing the existing `WHERE` seam removes those correctness/security gaps and fixes structural plan admission without inventing a source-shaped opcode. |
| D-132 | 2026-07-22 | accepted | Migrate JIT `Tensor.tril` to typed `TRIL` over the final two matrix axes. Require rank at least two, a supported dtype, and an exact built-in/NumPy integer diagonal; saturate that diagonal into the shape-derived all-zero/all-input semantic range before IR construction. Return an owning CPU result; use idempotent triangular closure/symbolic VJP; preserve matrix axes under leading-axis vmap; emit ONNX `Trilu` with `upper=0`; and refuse tensor-plan/WebGPU until portable triangular selection exists. Make Grad expose the same instance/top-level semantics and consume the shared fixture. | The callback invoked unchecked integer coercion, deferred rank failure, hid selection from functional transforms/export, and left device meaning implicit; Grad independently returned a disconnected forward-only tensor. One registry-bound triangular operation closes those correctness/security gaps while bounded diagonal normalization prevents arbitrarily large integers from reaching NumPy or ONNX and avoids a source-shaped backend escape hatch. |
| D-133 | 2026-07-22 | accepted | Migrate JIT `Tensor.cumsum` to typed `CUMSUM` with one exact normalized axis and a closed source/output dtype contract. Preserve floating defaults, promote integral and boolean defaults to int64, cast before accumulation for an explicit dtype, return owning exact-dtype CPU arrays, use the opposite-direction inclusive scan for closure/symbolic VJP, shift the axis under vmap, emit exact ONNX `Cast`/`CumSum`, and refuse `out=` plus tensor-plan/WebGPU until typed mutation and portable scan lowering exist. Make Grad consume the same fixture. | The callback deferred coercion, declared dtype metadata that could disagree with realized NumPy storage, hid the derivative and scan direction from transforms/export, and left mutation/backend decisions implicit. One registry-bound scan removes those correctness and security gaps without admitting host callbacks as device execution or inventing a second dtype path. |
| D-134 | 2026-07-22 | accepted | Migrate top-level JIT `cat` to typed variadic `CONCAT` over one exact normalized existing axis. Require a bounded plain tuple/list of exact same-session tensors, preserve the `(0,)` compatibility empty, close dimensioned-tensor dtype promotion and output allocation, return an owning promoted CPU array, split closure/symbolic cotangents through typed `NARROW`, broadcast captured inputs under vmap, emit exact ONNX `Cast`/`Concat`, and refuse `out=` plus tensor-plan/WebGPU until typed mutation and canonical variadic copy lowering exist. Make Grad consume the same fixture. | The callback hid promotion and splitting from transforms/export, accepted an unbounded variadic allocation, and left backend decisions implicit; Grad separately wrapped non-tensor values. One registry-bound concatenation and internal slicing primitive remove those correctness/security gaps without treating a host copy as device execution or adding a source-shaped lowering path. |
| D-135 | 2026-07-22 | accepted | Migrate top-level JIT `stack` to typed variadic `STACK` over one exact normalized inserted axis. Require a bounded plain tuple/list of exact same-session equal-shaped tensors, close dimensioned-tensor dtype promotion and output allocation, return an owning promoted CPU array, select closure/symbolic cotangents through typed `NARROW` and `RESHAPE`, broadcast captured inputs under vmap, emit exact ONNX `Cast`/`Unsqueeze`/`Concat`, and refuse `out=` plus tensor-plan/WebGPU until typed mutation and canonical variadic copy lowering exist. Make Grad consume the same fixture. | The callback hid promotion, axis insertion, and gradient selection from transforms/export, accepted an unbounded variadic allocation, and left backend decisions implicit. Reusing the registry-bound variadic contract removes those correctness/security gaps without treating a host copy as device execution or adding a second source-shaped path. |
| D-136 | 2026-07-22 | accepted | Migrate `torch.nn.functional.pad` to typed constant `PAD`. Normalize a plain even-length last-dimension-first sequence into canonical rank-sized geometry; preserve a closed source dtype with an exact fill; bound rank, each output extent, and bytes before allocation; return an owning CPU result; extract closure/symbolic cotangents through typed `SLICE`; preserve a leading vmap axis; emit exact ONNX `Pad`; and refuse non-constant modes, negative cropping, tensor-plan, and WebGPU until their distinct semantics and lowerings exist. Make Grad consume the same fixture. | The callback hid padding geometry, fill conversion, and derivative selection, accepted coercive/resource-hostile inputs, and left export and device meaning implicit; Grad maintained a separately drifting NumPy path. One registry-bound constant-padding profile closes those correctness/security gaps without conflating host padding, cropping, boundary modes, or structural plan admission with portable device execution. |
| D-137 | 2026-07-22 | accepted | Migrate both `torch.sort` outputs together as typed `SORT_INDICES` and `SORT_VALUES` over one exact normalized axis/descending/stable request. Bound rank, extents, selected-axis work, and combined output bytes; preserve value dtype and owning int64 indices; use stable ordering without dtype-changing negation; scatter floating cotangents through the exact paired permutation; shift the axis under vmap; export full-axis ONNX `TopK` plus `GatherElements`; and refuse `out=`, tensor-plan, and WebGPU until typed mutation and canonical portable ordering exist. Make Grad consume the same fixture. | Independent callbacks duplicated sorting work, could disagree, disconnected autograd, hid the paired-output relationship from transforms/export, and implemented descending order through reversal without an explicit stable-tie contract. One paired registry-bound ordering seam closes those correctness/security gaps without treating CPU sorting or ONNX exportability as portable device execution. |
| D-138 | 2026-07-22 | accepted | Migrate both `torch.topk` outputs together as typed `TOPK_INDICES` and `TOPK_VALUES` over one exact normalized axis/k/largest/sorted request. Bound rank, extents, selected-axis work, paired output, and conservative NumPy workspace; preserve value dtype and owning int64 indices; compute one negation-free partial selection and sort only the selected k when requested; scatter floating cotangents through the immutable selected permutation; shift the axis under vmap; export selected-k ONNX `TopK` plus `GatherElements`; and refuse `out=`, scalar input, tensor-plan, and WebGPU until typed mutation and canonical portable partial selection exist. Make Grad consume the same fixture and harden typed full sort with the same workspace ceiling. | Independent callbacks each performed a full sort, duplicated temporary allocation, disconnected autograd, hid their paired relationship from transforms/export, and used dtype-changing negation for largest selection; Grad separately cast values and indices to float32. One registry-bound partial-selection seam closes the performance, correctness, and resource-exhaustion gaps without treating a host selection or ONNX export as portable device execution. |
| D-139 | 2026-07-22 | accepted | Migrate overwrite `torch.scatter` to typed `SCATTER` over one exact normalized axis, same-rank int64 index, and exact matching tensor or scalar source. Require unique nonnegative in-range destinations and bounded target/output/workspace; preserve target dtype and return an owning result; split closure/symbolic VJP into target cotangent with overwritten positions zeroed and tensor-source cotangent gathered through typed `INDEX`; broadcast captured operands under leading-axis vmap; export opset-17 ONNX `ScatterElements` with scalar `Expand`; and refuse reductions, tensor-plan, and WebGPU until their distinct typed semantics and canonical portable lowerings exist. Make Grad consume the same fixture and snapshot the index for backward. | The callback coerced axes and sources, inherited NumPy negative-index behavior, accepted nondeterministic duplicate writes with incorrect gradients, disconnected JIT autograd, and hid overwrite meaning from transforms/export; Grad maintained a separately drifting path. One registry-bound unique-overwrite seam removes those correctness and resource-exhaustion gaps while matching ONNX reduction-none's unique-index precondition and avoiding a false portable-device claim. |
| D-140 | 2026-07-22 | accepted | Migrate string-equation `torch.einsum` to typed variadic `EINSUM` with allocation-free canonical parsing, explicit/implicit output, repeated-label diagonals, broadcast labels, different-rank ellipses, PyTorch ellipsis reduction, dimensioned promotion, and float32 half accumulation. Bound arity, equation bytes, rank, extents, resolved labels, output, cast/gradient workspace, and contraction domain before allocation; use one greedy CPU contraction; emit general closure and symbolic `EINSUM_VJP`; preserve mapped batch prefixes independently of user ellipses; export resolved opset-17 ONNX `Einsum`; and refuse tensor-plan/WebGPU until canonical contraction scheduling/lowering exists. Make Grad consume the same fixture and snapshot operands for backward. | The callback executed unchecked NumPy work during construction, reexecuted at realization, disconnected gradients and transforms, and left dtype/export/backend meaning implicit; Grad separately required an explicit arrow, capped arity at two, cast everything to float32, and could not differentiate general equations. One bounded registry-bound contraction removes those correctness, performance, and resource-exhaustion gaps without mistaking a host contraction for a portable device schedule. |
| D-141 | 2026-07-22 | accepted | Migrate `torch.nn.functional.l1_loss` to typed `L1_LOSS` with exact same-shape floating inputs, dimensioned promotion, float32 half accumulation, exact `none`/`sum`/`mean` reductions, and a transform-owned batch rank. Bound rank, extents, output, conservative workspace, and work before allocation; return owning CPU results; propagate both source cotangents through closure and internal symbolic `L1_LOSS_VJP` with zero subgradient at equality; preserve per-example vmap reductions; export opset-17 ONNX `Sub`/`Abs`/reduce; and refuse tensor-plan/WebGPU until canonical loss-reduction lowering exists. Make Grad consume the same fixture and snapshot the signed difference for backward. | The shared callback deferred shape, dtype, work, and allocation checks to NumPy, declared the first dtype without realized validation, disconnected target and functional gradients, and left transform/export/backend meaning implicit; Grad separately cast through float32 and disconnected the target. One bounded registry-bound reduction removes those correctness, performance, and resource-exhaustion gaps without treating host reduction as portable device execution or coupling unrelated loss functions. |
| D-142 | 2026-07-22 | accepted | Migrate `torch.nn.functional.smooth_l1_loss` to typed `SMOOTH_L1_LOSS` with exact same-shape floating inputs, promoted float32 half compute, finite non-negative compute-representable beta, safe zero-beta L1 semantics, exact reductions, and a transform-owned batch rank. Bound rank, extents, output, both retained gradients, masks, conservative workspace, and 32-visit piecewise work before allocation; propagate both source cotangents through closure and internal `SMOOTH_L1_LOSS_VJP`; preserve nested per-example vmap; emit an opset-17 piecewise ONNX decomposition; and refuse tensor-plan/WebGPU until canonical loss lowering exists. Make Grad consume the same fixture and snapshot the piecewise derivative. | The callback deferred shape, dtype, beta, work, and allocation checks to NumPy, disconnected the target and functional gradients, and left transforms/export/backend meaning implicit; Grad separately rejected the L1 limit, cast derivatives to float32, and disconnected the target. Reusing one bounded typed loss geometry closes those correctness, performance, and resource-exhaustion gaps without evaluating a zero-beta division or mistaking host reduction for portable device execution. |
| D-143 | 2026-07-22 | accepted | Migrate `torch.nn.functional.binary_cross_entropy` to typed `BINARY_CROSS_ENTROPY` with exact same-shape floating probabilities, promoted float32 half compute, finite closed `[0, 1]` runtime domains, PyTorch-compatible `-100` forward log floors, independent `1e-12` input-gradient denominator, unclamped target-logit derivative, exact reductions, and a transform-owned batch rank. Bound rank, extents, output, both retained gradients, casts, four compute buffers, one mask, and 48-visit work before allocation; propagate both source cotangents through closure and internal `BINARY_CROSS_ENTROPY_VJP`; preserve nested per-example vmap; refuse opset-17 ONNX when it cannot retain fail-closed runtime domain validation; and refuse tensor-plan/WebGPU until canonical probability-loss lowering exists. Make Grad consume the same fixture and snapshot both derivatives. | The callback clipped probabilities instead of logarithms, produced endpoint losses near 27.63 instead of 100, accepted invalid probability domains, disconnected target and functional gradients, and left transforms/export/backend meaning implicit; Grad duplicated the same drift. One bounded registry-bound probability loss closes those correctness, security, and resource-exhaustion gaps without clipping invalid data, exporting weaker semantics, or mistaking host reduction for portable device execution. |
| D-144 | 2026-07-22 | accepted | Migrate `torch.nn.functional.binary_cross_entropy_with_logits` to typed `BINARY_CROSS_ENTROPY_WITH_LOGITS` with exact same-shape floating inputs, promoted float32 half compute, overflow-safe `(1-target)*logits + softplus(-logits)` forward, stable sigmoid-minus-target input derivative, exact negative-logits target derivative, exact reductions, and a transform-owned batch rank. Bound rank, extents, output, both retained gradients, casts, four compute buffers, one mask, and 36-visit work before allocation; propagate both source cotangents through closure and internal `BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP`; preserve nested per-example vmap; emit an opset-17 stable `Neg`/`Softplus` decomposition; and refuse tensor-plan/WebGPU until canonical stable loss lowering exists. Make Grad consume the same fixture, snapshot both derivatives, expose all reductions, and retain explicit refusal of optional weighting. | The callback silently computed in float32 while declaring the logits dtype, differentiated only logits, accepted coercive targets, and left transforms/export/backend meaning implicit; Grad separately exposed mean only, used a direct overflow-prone sigmoid, cast through float32, and disconnected the target. One bounded registry-bound stable loss closes those correctness, performance, and resource-exhaustion gaps without mistaking probability BCE, unimplemented weighted broadcasting, or host reduction for the typed logits contract. |
| D-145 | 2026-07-22 | accepted | Migrate `torch.nn.functional.kl_div` to typed `KL_DIV` with exact same-shape floating inputs, promoted float32 half compute, exact probability-target `xlogy` or logarithmic-target algebra, exact `none`/`sum`/`mean`/`batchmean` reductions, and a transform-owned batch rank. Preserve native zero-target forward and derivative behavior, scalar batchmean, empty-batch division, and both source-dtype derivatives; bound rank, extents, output, retained gradients, casts, masks, conservative workspace, and 48-visit work before numerical execution; preserve nested per-example vmap; emit exact opset-17 decompositions for both target modes and batchmean; and refuse tensor-plan/WebGPU until canonical loss lowering exists. Delete the now-unused shared callback helper and make Grad consume the same fixture, snapshot both derivatives, and expose the exact public aliases. | The shared callback cast through float32, declared the input dtype without realized validation, treated batchmean as unavailable, differentiated only the input, and hid target representation, zero behavior, transforms, export, resource limits, and backend decisions. Grad duplicated the float32 and disconnected-target drift. One bounded registry-bound KL contract closes these correctness, performance, and resource-exhaustion gaps without clipping native zero semantics, weakening batchmean, adding deprecated reduction precedence, or treating host algebra as portable device execution. |
| D-146 | 2026-07-23 | accepted | Migrate `torch.nn.functional.nll_loss` to typed variadic `NLL_LOSS(input,target[,weight])` with exact unbatched/spatial class-axis geometry, int64 target and range checks, optional same-dtype nondifferentiable class weight, signed-int64 ignore index, exact `none`/`sum`/`mean` plus legacy reduction precedence, and a transform-owned batch rank. Preserve weighted denominators, ignored/empty behavior, source-dtype selected-class closure/symbolic VJP, mapped or captured weights under nested vmap, and exact unmapped opset-17 `NegativeLogLikelihoodLoss`; bound rank, extents, output, conservative workspace, and zero-hidden work before numerical execution; refuse mapped ONNX and tensor-plan/WebGPU until their semantics exist. Make Grad consume the same fixture and snapshot validity, indices, and selected weights. Partition trace-cache keys by session and bypass mutable module trees. | The callback supported only 2-D float32 mean loss, inherited unchecked NumPy indexing, hid weights/ignored targets/reduction/transform/export/resource/backend meaning, and pinned module weights to one session; Grad duplicated the narrow path. The trace cache also treated equal shape/dtype from distinct session buffer namespaces as interchangeable and could bypass mutable buffers. One bounded registry-bound indexed loss plus cache-isolation repair closes these correctness, performance, and resource-exhaustion gaps without differentiating discrete/weight inputs, weakening range checks, or treating host indexing as portable device execution. |
| D-147 | 2026-07-23 | accepted | Migrate `torch.nn.functional.cross_entropy` to typed variadic `CROSS_ENTROPY(input,target[,weight])` with stable promoted log-softmax, exact unbatched/spatial class-axis geometry, index or probability targets, optional same-dtype nondifferentiable class weight, signed-int64 ignore index for index targets, finite `[0,1]` label smoothing, exact `none`/`sum`/`mean` plus legacy reduction precedence, and a transform-owned batch rank. Preserve PyTorch-compatible weighted index means, probability-position means, all-ignored behavior, source-dtype logits closure/symbolic VJP, probability-target VJP, mapped/captured targets and weights under nested vmap, and exact unmapped index-target opset-17 `SoftmaxCrossEntropyLoss` when smoothing is zero; bound rank, extents, output, workspace, and work before numerical execution; refuse unsupported ONNX profiles and tensor-plan/WebGPU until their semantics exist. Make Grad consume the same fixture and immutable forward snapshots. Express ReLU through canonical typed comparison/where operands so closure and symbolic topology cannot diverge. | The callback exposed only a two-dimensional float32 index-target mean, used unstabilized exponentials, hid probability targets, weighting, ignoring, smoothing, reductions, derivatives, transforms, export, resources, and backend decisions, and duplicated narrow behavior in Grad. Its migration also made full MLP traces symbolic and exposed a pre-existing ReLU node whose three semantic operands disagreed with its one-input closure context. One bounded registry-bound class-axis loss plus the canonical ReLU expression closes those correctness, performance, resource-exhaustion, and graph-integrity gaps without differentiating discrete targets or weights, weakening runtime domains, or treating host softmax as portable device execution. |
| D-148 | 2026-07-23 | accepted | Migrate `torch.nn.functional.dropout` to typed `DROPOUT` and `DROPOUT_VJP` with exact probability/training/inplace validation, identity/no-RNG branches, deterministic dtype-preserving drop-all, and one immutable unsigned 63-bit seed per active stochastic operation. Regenerate the same mask for CPU replay, closure/symbolic VJP, functional grad, and checkpoint recomputation; bound rank, extents, output, work, and RNG/mask workspace before key consumption; refuse typed mutation, stochastic vmap without an explicit randomness policy, ONNX inference export, and tensor-plan/WebGPU execution. Exclude opaque/random graphs from the trace cache and make Grad preserve output/gradient dtype through one delegated functional implementation. | The callback resampled on every realization and replaced mutable closure state, so an observed forward and its gradient could use different masks; trace caching could also freeze one stochastic key. Grad separately validated after identity, rejected p=1, cast active values to float32, and duplicated module behavior. One keyed registry-bound operation closes replay, checkpoint, dtype, cache, transform, export, and resource-exhaustion gaps without claiming Philox parity, silently choosing vmap randomness, or weakening training randomness into inference/device semantics. |
| D-149 | 2026-07-23 | accepted | Keep the release harness fail-closed when compiler WebGPU artifacts are optional in the static result type: the runtime E2E rubric must explicitly reject missing `wgsl` or `wgslProgram` before inspecting them. Make the separately installed dogfood directory a valid one-package pnpm workspace while retaining its exclusion from the root workspace. | The Node 25 release run exposed three strict-nullability errors in an existing rubric and a final pnpm-version-sensitive `packages field missing or empty` failure after every source/distribution browser corpus gate had passed. Explicit artifact rejection preserves runtime semantics, and `packages: ["."]` makes the isolated published-package consumer portable across the root-inherited pnpm invocation without workspace-linking its dependencies. |
| D-150 | 2026-07-23 | accepted | Migrate `nn.BatchNorm1d` to typed `BATCH_NORM_1D`, `BATCH_NORM_1D_STATS_UPDATE`, and `BATCH_NORM_1D_VJP` under an exact float32 rank-2/rank-3 contract. Use biased variance for normalization and unbiased variance for persistent state; register running mean, running variance, and int64 batch count; support fixed/cumulative momentum; order lazy updates through predecessor dependencies; and commit each session-owned effect atomically once while preserving public buffer identity. Snapshot tracked eval state, remap checkpoint VJP authority to cloned forward nodes, bound metadata/workspace, and refuse vmap/export/device routes whose state semantics are absent. Converge Grad on the same fixture and immutable backward state. | The callback replayed running-state mutation during backward, saved derivatives in mutable closure state, persisted the wrong variance, omitted counting/cumulative averaging, and hid transform/export/resource/backend decisions. Grad duplicated those defects and replaced public buffers. One registry-bound state effect closes correctness, replay, checkpoint, state-dict, mutation, resource-exhaustion, and support-reporting gaps without pretending host state is a portable device effect. |
| D-151 | 2026-07-23 | accepted | Export semantic-core `/capability` only with concrete immutable capability/backend definitions and program/artifact-scoped lowering decisions. Generate runtime definitions byte-for-byte from the architecture vocabulary, but construct a support view only from at least one actual subject-bound decision. Positive decisions require a registered preservation level; conditional decisions require an exact feature, limit, or runtime guard; negative/unknown decisions require a reason and cannot claim preservation. Reject unknown or duplicate capability/backend pairs. | Static definitions, environment requirements, framework method presence, and evidence file paths are not program support. One canonical decision protocol prevents those facts from being flattened into booleans while giving the next generated platform view an honest program-scoped input. Terminal execution evidence remains separate. |
| D-152 | 2026-07-23 | accepted | Project JIT's JavaScript and platform support records from the same generated 36-operation JSON consumed by Python executable validators. Let runtime compose one complete provider-bound requirement environment, one raw program-support input, and one to sixteen bounded framework-neutral sources while retaining each fact separately. Reject open fields, malformed versions/IDs, oversized sources, and duplicate framework/operation identities; preserve the exact ten framework decision categories and never synthesize one support boolean. Keep runtime and JIT dependency-independent and prove their structural contract through a fresh packed consumer. | Platform UI needs one deterministic payload, but runtime must stay tensor-agnostic and the executable JIT table cannot be copied into runtime. Structural composition plus a packed integration test preserves package boundaries, avoids a hand-maintained support table, and prevents requirement availability, framework eligibility, program lowering, or evidence from substituting for one another. |
| D-153 | 2026-07-23 | accepted | Generate Grad's 22 framework-neutral platform records byte-for-byte from the exact schema-v2 compatibility inventory. Admit only verified target contracts; map its closed CPU/refusal, eager autograd, residency, and materialization facts; report symbolic transforms, export, tensor planning, and WebGPU as explicit eager non-applicability or NumPy-reference refusal; and reject stale output, duplicate behavior, or unknown mappings in the architecture gate. Prove the detached public source and the combined Grad/JIT runtime view through packed consumers. | Grad's source- and fixture-bound inventory is the executable eager contract; method presence or a parallel support table would create a weaker truth and overclaim lazy/device capabilities. Mechanical generation closes the initial Gate 6 exit while keeping runtime framework-independent and terminal evidence separate. |
| D-154 | 2026-07-26 | accepted | Admit external header-distribution approval only through one bounded canonical host policy and one canonical DSSE/in-toto P-256 P1363 statement over the exact current package review subject. The subject must commit the current build lock, header-input projection, all 17 output identities, package resource, output verification and reproducibility identities, and exact license-inventory hash and length. Reauthenticate the exact current package evidence before verification; retain the verified authority opaquely; and upgrade only the legal-review and distribution claims. Keep full-output reproducibility, producer trust, Worker execution, lowering, backend execution, and release false. | A package-generated file map or signature cannot establish independent approval, while a detached approval that is not rebound to current package bytes can become stale or authorize the wrong distribution. The policy and exact-subject seam makes the software decision auditable and fail-closed without pretending the synthetic fixture is a production policy, external reviewer key, or external legal decision. |
| D-155 | 2026-07-26 | accepted | Derive external builder signing bytes only from the exact opaque manifest, build-lock, and Worker authorities plus one admitted producer policy, prepared trust store, and allowlisted builder/key. Emit canonical DSSE/in-toto format-only material and keep every trust, execution, distribution, and release claim false. Reuse the existing signature-binding and producer-trust verifiers as the only authority transitions. | Manually assembled external statements can drift from the current promoted build or bind stale policy inputs. One exact signing-request seam makes the bytes to be signed deterministic and auditable without self-signing, retaining signer authority, or confusing a request with independent producer trust. |
| D-156 | 2026-07-26 | accepted | Treat every browser-required package policy file as a deterministic member of the complete distribution plan. Add the diagnostic-normalization manifest, bind the semantic-adapter media type exactly, repin the build lock, and require all nine executable/runtime asset paths and media selections during lock admission. The complete verifier now recognizes 24 deterministic subjects plus one detached envelope. | The asset manifest required diagnostic-normalization bytes that the former 24-path closed distribution explicitly forbade, so no live complete tree could both satisfy asset acquisition and the build lock. Closing the 25-path plan makes the external-evidence handoff constructible and prevents future package policy assets from disappearing behind ambient serving assumptions. |
| D-157 | 2026-07-26 | accepted | Operationalize producer signing through one closed host exchange interface. Read exact normalized absolute, canonical, current-user-owned, single-link, non-writable files under stable non-writable parent identities; require the current package lock and Worker bytes; write canonical request and observation files with exclusive no-follow creation, synchronization, read-only mode, independent reread, no clobber, and owned-failure cleanup. Verification must consume the exact issued request, match its payload, key, and current input identities, and invoke the existing signature-binding plus independent producer-policy transitions in one process. Accept no private key and serialize only a host observation with reusable producer authority false. | A pure signing-material function leaves operators to assemble files, match responses, and persist results correctly. That gap permits stale package inputs, request substitution, symlink or clobber hazards, and accidental treatment of serialized JSON as opaque producer authority. One deep exchange module owns those correctness and filesystem-security obligations without creating another trust transition or allowing the package to self-sign. |
| D-158 | 2026-07-26 | accepted | Deepen the same closed external-evidence exchange to own independent header-distribution review. Reauthenticate canonical immutable approval-policy and trust-store files, bind the package-pinned header-distribution resource identity, require an allowlisted reviewer and key, and emit one format-only request without accepting private material. Verification must rederive that exact request from current package evidence, match the returned DSSE coordinates, invoke the existing opaque distribution-approval transition in process, and persist only a host observation with reusable approval authority false. | Producer provenance and redistribution review have different trust roots but identical hostile filesystem and request/response mechanics. One protocol-discriminated deep module preserves those distinct authority transitions while preventing a second security adapter, stale review subjects, request or key substitution, clobber/link races, and accidental promotion of serialized true observations into reusable legal authority. |
| D-159 | 2026-07-26 | accepted | Materialize the complete browser distribution in two explicit stages. The deterministic stage accepts only the exact current build lock, package-pinned Wasm and Worker, admitted producer policy, and independently verified 17-output header root; it reconstructs the canonical profile, manifest, cycle-free build subject, and exact 24 deterministic outputs without accepting a signature or private key. The finalizer independently reauthenticates the exact tree and separately handed-off immutable profile, then adds only the sole detached DSSE envelope after an in-process producer authority binds that exact subject. Require exclusive no-follow output creation, exact tree closure, immutable stable inputs, bounded reads, terminal rehash, file and parent synchronization, no clobber, and cleanup limited to files created by the failed transaction. | The verifier proved what a complete tree must contain but no hardened producer constructed one, leaving operators to copy large assets, recreate derived metadata, and attach provenance by hand. A two-stage deep module makes the release-shaped tree deterministic and transactional while ensuring the compiler never accepts private signing material or treats materialization as producer, legal, distribution, execution, or release authority. |
| D-160 | 2026-07-26 | accepted | Package-pin the exact two-root live complete-distribution observation and admit it only through a fresh opaque verifier. Require byte equality with the reviewed resource, exact current build-lock identity and 24+1 plan, exact header-subset authority, current reproducible Wasm, verified zero-import Worker, canonical build-lock/diagnostic/runtime-ABI/semantic-adapter bytes, exact asset-manifest binding, canonical build subject, recomputed byte totals, and recomputed full reproducibility identity. Retain the live policy scope as local engineering only and keep detached signature, external producer trust, legal review, distribution, Worker execution, lowering, backend execution, and release false. | A temporary successful run is not durable package evidence, and a copied JSON record must not mint authority. Rebinding every deterministic output to current package authorities makes the live result reviewable and fail-closed while preventing a local engineering signature or structurally forged record from being promoted into production trust or release status. |

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
| 2026-07-17 | Gate 3 BrowserGrad-owned LibTooling tracer source | Compiler build; build-lock Vitest; `pnpm --filter @unlocalhosted/browsergrad-compiler test:browser-clang-wasm-build-plan:run`; source/hash/CMake structural tests; independent P0/P1 review; `git diff --check` | Passed after review fixes: compiler build; lock 1 file/24 tests; planner/source 2 files/29 tests; TypeScript build-plan typecheck and focused oxlint; exact checked-in C++/CMake hashes and byte lengths match the canonical input lock. Review found two P1 gaps: source hashes were test-only and the generated sidecar was never materialized into the declared output. The lock now keeps both as explicit release blockers and the planner reports both readiness facts false. The source contains a real closed-VFS LibTooling action and exact eight-export C ABI state machine. No LLVM/Emscripten toolchain was materialized, no C++ source was compiled, and no Docker, Worker, browser, or WebGPU execution occurred. | Commit the source/build-integration slice. Next implement source-verifying execution/materialization, VFS, CUDA dual-pass, artifact-v3, Worker integration, and obtain the first independently inspected build. |
| 2026-07-17 | Gate 3 host-owned Worker controller boundary | Controller/protocol focused Vitest; strict compiler typecheck; focused oxlint; two-stage independent P0/P1 review; `git diff --check` | Passed after hardening: controller plus protocol 2 files/28 tests, including 19 controller cases; typecheck/lint/whitespace clean. Review closed invalid-start cleanup, AbortSignal registration race, overdue-timer task ordering, caller-selected module authority, production-shaped fake evidence, and mutable ambient-global authority. Production now capability-fails at `$.runtime` before reading caller input/options or browser globals, so no production issuer is reachable. Test execution returns only a disjoint simulation with execution false. No Worker, Blob, Clang-Wasm, C++, Docker, or WebGPU execution occurred. | Commit the controller boundary. Next build and internally pin the self-contained Worker/factory module and captured native platform adapter before deliberately enabling production. |
| 2026-07-17 | Gate 3 deterministic offline header-pack selection and assembly | Focused header/VFS Vitest; strict compiler typecheck; focused oxlint; compiler architecture check; two-stage independent P0/P1 review; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; serial `pnpm test:release-packages`; `git diff --check` | Passed after hardening: focused 2 files/19 tests; full compiler 59 files/1,316 tests; optional Docker 2 files/96 tests; Clang-Wasm planner/source 2 files/29 tests; both architecture checks, kernels/compiler builds, strict typecheck/lint, and all synthetic/corpus guards; release 19 hostile-archive and 35 Node security tests plus packed/fresh consumers. Review closed global inventory-budget ordering, mutable source-byte hashing, license self-authorization, VFS content-set hash limits, selection-level canonical limits, and the inspector's missing third full-pack copy in memory preflight. Final independent review found no P0/P1. No acquisition, real header/notice bytes, build, Docker process, Worker, browser C++, or WebGPU execution occurred. | Commit the bounded offline assembly slice. Next acquire and externally review the exact inputs, materialize the real packs, and bind their observed identities into build/asset provenance before Worker integration. |

| 2026-07-17 | Gate 3 VFS successful-read accounting | Focused VFS-session Vitest; compiler typecheck/oxlint; independent P0/P1 review; full compiler/release gates; `git diff --check` | Passed: focused 1 file/8 tests; full compiler 60 files/1,326 tests; optional Docker 2 files/96 tests; build-plan 3 files/47 tests; release 19 Python and 35 Node security tests plus packed/fresh consumers; whitespace clean. `openedFiles` now means unique exact files whose content read committed to stable Wasm memory. Open/stat/probe, failed reads, and zero-byte reads of nonempty files do not count. This remains successful-read observation, not semantic-use evidence. No Worker or C++ execution occurred. | Commit separately. Real Worker integration must compare this set with artifact pass `openedFileIds`; do not promote it directly into semantic or provenance claims. |
| 2026-07-17 | Gate 3 Worker input-frame and verifier closure | Focused Worker/controller/protocol/artifact Vitest; compiler typecheck/oxlint; independent P0/P1 review; full compiler/release gates; `git diff --check` | Passed: focused 4 files/65 tests; full compiler 60 files/1,326 tests; optional Docker 2 files/96 tests; build-plan 3 files/47 tests; release gates green. Coverage includes complete artifact-v3 ceilings, one request-bounded decode budget across all internal hashes, impossible-request rejection before executable copies/hashes, one cached canonical profile/request region pair, a 64-byte little-endian input frame, per-invocation memoization, defensive copies, async liveness rechecks, and terminal revocation. A 4,096-diagnostic artifact with more than 2 MiB rendered text passes only under its explicit admitted contract. No production issuer, Worker, or C++ ABI execution exists. | Commit separately. Positive caller-frame tests remain protocol consistency only; add a real ABI consumer test after an independently inspected Wasm build exists. |
| 2026-07-17 | Gate 3 source snapshot and Wasm sidecar materialization | Build-plan TypeScript/oxlint/Vitest; independent filesystem/authority P0/P1 review; full compiler/release gates; `git diff --check` | Passed: build-plan 3 files/47 tests covering exact recursive source closure, canonical source-set identity, owned read-only staged snapshots, inode/path rebinding, trusted lexical/resolved ancestry, same-UID single-writer scope, bounded Wasm-v1 sidecar admission, atomic no-clobber installation, cancellation commit semantics, exact-inode rollback, identity-swap refusal, and composite cleanup failures. Full compiler 60 files/1,326 tests and release gates pass. No process spawned; no source compiled; installed bytes prove no WebAssembly validity, ABI conformance, build/output identity, reproducibility, provenance, licensing, or release readiness. | Commit separately. Next executor step is the real pinned Emscripten build; distribution still requires inspected Wasm projections, factory bundling, license closure, reproducibility, and authenticated provenance. |
| 2026-07-17 | Gate 3 C++ imported VFS bridge | `corepack pnpm --filter @unlocalhosted/browsergrad-compiler test:browser-clang-wasm-build-plan`; build-lock Vitest; focused source review; compiler typecheck/oxlint; independent P0/P1 review; `corepack pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `corepack pnpm test:release-packages`; `git diff --check` | Passed after review fixes: build-plan/source 3 files/49 tests; lock 1 file/24 tests; full compiler 62 files/1,343 tests; optional Docker 2 files/96 tests; release 19 Python and 35 Node security tests plus packed/fresh consumers. The C++ source implements exact six-import, bounded, lazy, fail-stop `llvm::vfs` behavior. Review replaced eager directory materialization, poisoned the adapter after close failures, and added post-open RAII cleanup. The source and lock hashes are repinned. No pinned LLVM/Emscripten compile, Wasm output, Worker, Docker daemon, or browser C++ execution occurred. | Compile against the pinned toolchain and inspect the resulting raw Wasm before treating this bridge as ABI or runtime evidence. |
| 2026-07-17 | Gate 3 local Wasm runtime metrics | Focused metrics Vitest; related Worker/ABI/VFS Vitest; compiler typecheck/oxlint; independent P0/P1 review; full compiler/release gates; `git diff --check` | Passed after review fixes: metrics 1 file/6 tests; related boundary 9 files/136 tests; full compiler 62 files/1,343 tests; release gate green. Coverage includes exact page/capacity observations, fixed 72-byte allocator records, global requested-byte counters, fixed-stack/current-page/profile coexistence, profile output ceiling, arithmetic/count consistency, captured intrinsics, and explicit local/unverified labels. Review corrected allocator ceilings, stack coexistence, profile memory/output enforcement, impossible cumulative relationships, mutable ambient intrinsics, and false Worker/reviewed-producer wording. Runtime ABI v1 has no metrics pointer/export; no producer emitted a record. | Add and repin a metrics ABI extension, then instrument and validate it in executed C++/Worker code. |
| 2026-07-17 | Gate 3 package Worker runtime input binding | Focused runtime-binding Vitest; related Worker/controller/protocol/VFS/ABI Vitest; compiler typecheck/oxlint; independent P0/P1 review; full compiler/release gates; `git diff --check` | Passed after review fixes: runtime binding 1 file/11 tests; related boundary 9 files/136 tests; full compiler 62 files/1,343 tests; release gate green. Exact invocation, frame, Wasm bytes, and VFS session are single-use, copied, rehashed, liveness-checked, and terminalized with aggregate cleanup. Review closed forged getter reads, first-cause-only cleanup, overclaimed intrinsics, mutable ambient operations, weak race coverage, and hostile-proxy error typing. Start remains capability-blocked before instantiation because package Worker bytes, generated factory, and first-build projections are absent. No Worker or C++ execution occurred. | Bundle and pin the Worker/factory only after the real build; preserve the closed no-loader/no-fetch authority surface. |
| 2026-07-17 | Gate 3 runtime ABI 1.1 metrics contract and realm-boundary audit | Focused ABI/metrics/assets/build-lock/Wasm/profile Vitest; strict compiler typecheck/oxlint; build-plan gate; two-stage independent P0/P1 review; `corepack pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `corepack pnpm test:release-packages`; stale-pin search; `git diff --check` | Passed after fixes: focused 6 files/144 tests; build-plan 3 files/49 tests; full compiler 62 files/1,349 tests; optional Docker 2 files/96 tests; both architecture checks, kernels/compiler builds, strict typecheck/lint, and all synthetic/corpus guards; release 19 Python and 35 Node security tests plus packed/fresh consumers. Review prevented silent ABI 1.0 mutation, an undefined linker export, and ambiguous zero-byte/realloc counters. ABI 1.1 now pins the ninth pointer export and exact record/event semantics; current ABI 1.0 C++ remains a named release blocker. Parallel architecture audit found the existing VFS/runtime binding is same-realm only and must be reconstructed/bound inside the Worker. No C++ compile, Wasm output, Worker, Docker daemon, or browser execution occurred. | Commit the ABI design slice. Next split the C++ extractor, implement and behaviorally prove the metrics producer, and correct Worker-local VFS ownership before factory bundling. |
| 2026-07-17 | Gate 3 C++ modular boundary and VFS mount/bind split | VFS/runtime focused Vitest; build-plan/source/executor and build-lock Vitest; compiler typecheck/lint/build; native strict C++ syntax checks; independent P0/P1 reviews; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `pnpm test:release-packages`; `git diff --check` | Passed after review fixes: VFS/runtime 2 files/24 tests; build-plan/source/executor 3 files/51 tests; build-lock 1 file/24 tests; full compiler 62 files/1,356 tests; optional Docker contract 2 files/96 tests; release 19 Python and 35 Node security tests plus packed/fresh consumers. Review removed an empty metrics module, captured authority/reflection/memory/cleanup intrinsics, and made cleanup terminal before destructive work. The extractor split preserves exact ABI 1.0/fail-closed behavior; VFS mount/bind remains explicitly same-realm. No complete LLVM/Emscripten compile, Wasm output, Worker spawn, Docker daemon, or browser-local C++ execution occurred. | Commits `9ca6c0ae` and `11d0fd3c`. Next implement the real ABI 1.1 metrics producer and the transferable Worker-local reconstruction path. |

| 2026-07-17 | Gate 3 pre-bind VFS imports and ABI 1.1 metrics producer | Focused VFS, runtime-ABI, build-lock, raw-Wasm metrics, native behavioral, source-closure, and build-plan tests; strict native/fake-Emscripten C++ syntax; compiler typecheck/lint/build; independent P0/P1 review; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `pnpm test:release-packages`; `git diff --check` | Passed after repinning transitive profile/asset fixtures: ABI/build-lock 3 files/76 tests; build-plan/source/native model 4 files/54 passed plus one Darwin ASan skip; full compiler 62 files/1,356 tests; optional Docker contract 2 files/96 tests; both architecture checks and all trailing compiler guards; release 19 Python and 35 Node security tests plus packed/fresh consumers. Stable pre-bind imports close the instantiation cycle. ABI 1.1 producer source/native behavior is real, but no pinned Emscripten build, final allocator call graph, executed Wasm record, Worker, or browser C++ execution exists. | Commits `916982a2` and `c0820ba6`. Next reconstruct every Worker-local authority from transferred canonical bytes, and run the exact pinned build when the daemon/toolchain becomes available. Keep the metrics release blocker until object/final-Wasm and executed-ABI evidence exist. |
| 2026-07-17 | Gate 3 canonical Worker transfer and realm-local runtime adoption | Focused transfer/runtime Vitest; compiler typecheck/oxlint; independent adversarial P0/P1 review; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `pnpm test:release-packages`; `git diff --check` | Passed after review fixes: focused 2 files/31 tests; full compiler 64 files/1,386 tests; optional Docker contract 2 files/96 tests; build-plan/source/native model 4 files/54 passed plus one Darwin ASan skip; release 19 Python and 35 Node security tests plus packed/fresh consumers. Coverage includes one-reservation/one-materialization host ownership, pre-clone ceilings, unique destructive buffers, hostile objects/accessors/aliases/shared or detached storage, exact receiver reconstruction, one-time adopt/discard, frame/Wasm continuity, stable pre-bind imports, phase-accurate cleanup, and one transfer-to-runtime integration seam with lower build authority deliberately mocked. Final independent review found no P0/P1. No Worker spawned; no C++, Wasm ABI, Docker runtime, browser execution, or lowering authority is claimed. | Commits `912aca22` and `647c814e`. Next make an internally pinned package Worker entry and controller consume this exact message, then instantiate only the reviewed generated factory after first-build projections exist. |
| 2026-07-17 | Gate 3 canonical Worker entry and typed terminal protocol | Focused entry/controller/transfer/runtime Vitest; strict compiler typecheck and oxlint; real Chromium module-Worker Vitest; independent P0/P1 review and re-review; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `pnpm test:release-packages`; `git diff --check` | Passed after review fixes: focused 4 files/62 tests; Chromium module Worker 1 file/1 test; full compiler 65 files/1,398 tests; optional Docker contract 2 files/96 tests; browser build-plan/source/native model 4 files/54 passed plus one Darwin ASan skip; both architecture checks and all synthetic/corpus guards; release 19 Python and 35 Node security tests plus packed/fresh consumers. Review closed pre-adoption authority leakage, post-adoption double-discard risk, failure-only terminal shape, string-only controller failure details, mock-only module-Worker behavior, and side-effect metadata drift. Final independent re-review found no P0/P1. Commit `9ef76452` is pushed to `origin/main`. The Chromium case proves module load, one-shot removal, and pre-identity error routing only; no reviewed Clang-Wasm, valid compile launch, user C++, Wasm ABI, or lowering executed. | Keep production capability-blocked. Next execute and inspect the pinned build, bundle the exact generated factory into the package Worker, then enable one valid browser-local layout-only compile after CUDA dual-pass/artifact-v3 completion. |
| 2026-07-17 | Gate 3 ABI 1.1 module-owned result lifecycle | Strict native C++ lifecycle and UBSan model using TypeScript-verifier-accepted canonical artifact-v3 bytes; source-closure/build-lock Vitest; compiler build/typecheck/lint; independent adversarial review and final re-review; `pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler`; `pnpm test:release-packages`; `git diff --check` | Passed after review fixes: native lifecycle 2 passed/1 Darwin ASan skip; build-plan/source/native models 5 files/56 passed/2 Darwin ASan skips; focused ABI/build-lock/Worker 3 files/81 tests; full compiler 65 files/1,398 tests; optional Docker 2 files/96 tests; release 19 Python and 35 Node security tests plus packed/fresh consumers. Review closed status-zero with a live review blocker, noncanonical success-fixture bytes, global-only output limits, sink-failure status masking, and inaccurate poison cleanup modeling. Runtime success now requires no blocker, one strictly bound invocation ceiling, one committed nonempty module-owned allocation, healthy metrics, and disjoint nonwrapping Wasm32 ranges. Exact input free preserves the immutable result; reset releases live owners while poisoned modules rely on mandatory Worker/module disposal. Final independent review found no P0/P1. Commit `f6431c9a` is pushed to `origin/main`. The real artifact callback remains the status-106 CUDA-dual-pass placeholder; no Wasm build or ABI execution occurred. | Next decode the canonical profile/request regions into a closed compile session and bind its `maxOutputBytes` before real CUDA passes/writer integration. |
| 2026-07-17 | Gate 3 locale-free cross-language wire ordering | Semantic-core canonical/hash tests; focused profile/request/provenance/artifact Vitest; semantic-core and compiler typecheck/build; focused oxlint; `git diff --check` | Passed: semantic-core 12 files/108 tests; compiler 4 files/72 tests; request-only 1 file/12 tests; both package typechecks, both builds, focused lint, and whitespace clean. Shared UTF-16 code-unit comparison now owns set-like wire order and canonical object-key sorting; all C++/CuTe production `localeCompare` use is removed. Unicode vectors cover composed/decomposed text, supplementary code points, and private-use code points. Commit `449fc9d1` is pushed to `origin/main`. | Reuse the same code-unit comparison in the native canonical JSON/session decoder. Do not treat this TypeScript prerequisite as native frame-decode evidence. |

| 2026-07-17 | Gate 3 bounded native canonical/identity foundation | Runtime-ABI/build-lock/profile/assets/Worker Vitest; compiler build/typecheck/lint; Clang-Wasm build-plan/source/native gate; optimized native C++ plus UBSan and platform-gated ASan; independent schema and primitive audits; architecture check; `git diff --check` | Passed after hardening: full compiler 65 files/1,403 tests; build-plan/native 7 files/61 passed/3 Darwin skips; strict runtime/build identity gates, compiler build/typecheck/lint, architecture, and whitespace clean. Review added a compiled parser-depth ceiling, standalone implementation linking, ABI-exact native limits, independent SHA padding/binary vectors, accurate direct-validation policy naming, and Linux CI execution. Commit `719cca9b` is pushed to `origin/main`. This proves native primitives and source closure only: no typed profile/request decode, VFS access, Clang action, Emscripten build, Wasm execution, browser C++, or artifact output occurred. | Resolve the six recorded compile-session contract gaps, then implement the immutable typed decoder and identity/VFS binding before CUDA passes. Run full `verify:compiler` and release gates after that integrated producer slice. |
| 2026-07-17 | Gate 3 closed semantic adapter and native policy primitives | Full compiler Vitest/typecheck/lint; semantic-adapter canonical/hash/hostile-input tests; deterministic native-policy codegen check; Clang-Wasm build-plan/source/executor/native gate; optimized/UBSan/platform-gated ASan tests; three read-only architecture/API audits; `git diff --check` | Passed: full compiler 68 files/1,426 tests; build-plan/native 10 files/73 passed/6 platform skips; exact 21-file extractor source closure; typecheck/lint/whitespace green. Commits `83aee3a9` and `9b9fd341` are pushed to `origin/main`. Profile 2.5/contract 1.1 and browser assets bind the exact canonical Clang 22.1.8 semantic-adapter resource. Native policy-to-argv, temporal-macro callbacks, and pass observations are bounded and source-locked. The gate also caught and fixed a stale 16-file executor ceiling. No production Clang action uses these helpers yet; no pinned toolchain build, Wasm instance, browser C++, or artifact output occurred. | Implement `DecodedCompileSession`, then a sealed full invocation builder. Wire fresh policy/observer/diagnostic state into device-first/host-second Clang actions before replacing the artifact-v3 placeholder. |

## Failure and Recovery Log

Record failures that may matter after context loss. Include the exact failing
command, concise error, suspected cause, resolution or next experiment, and
whether any files may be left partially changed.

- The semantic-adapter draft initially followed stale Clang 20.1.8 fixtures
  while the canonical build lock pins LLVM/Clang 22.1.8. Full compiler tests
  exposed the cross-resource drift. The manifest, profile, forced resource
  path, native API audit, and every dependent identity are now aligned to exact
  22.1.8. No 20.1.8 semantic-adapter claim remains.
- The first full compiler rerun after final policy repinning failed one
  deterministic artifact-byte SHA assertion. The artifact semantic hash and
  input-closure hashes were already correct; the canonical envelope byte pin
  was recomputed from the production verifier. The complete 68-file/1,426-test
  suite then passed.
- The first 21-file source-closure gate failed all 18 executor cases because
  its explicit source-file-count ceiling remained 16. The executor ceiling is
  now 32 while the aggregate one-MiB byte ceiling remains unchanged. The full
  build-plan/native gate passes 73 tests with six platform sanitizer skips.
- Compile-session schema audit found six pre-decoder contract gaps: unused
  request headers conflict with artifact opened-source closure; temporal macros
  have no reject/pin policy; warning-policy IDs have no closed Clang mapping;
  TypeScript and native virtual-path predicates disagree on C0/DEL bytes;
  imported VFS has no pass-scoped successful-read/include-edge observer; and
  Clang diagnostics have no frozen normalization/mapping contract. Do not let
  the native decoder or dual-pass actions invent these policies. Resolve and
  version them in producer-neutral contracts first.
- First build-plan/native rerun after adding four extractor files failed two
  deterministic fixture assertions (`fileCount: 12` and the old 12-path list).
  Executor/plan fixtures now pin all 16 source files; the complete 7-file gate
  passes 61 tests with three Darwin sanitizer skips.
- First full compiler test after ABI/build-lock repinning failed only stale
  profile and asset identity goldens. Canonical values were recomputed through
  production preparation paths and repinned; 65 files/1,403 tests pass.
- Primitive review found caller-controlled recursion depth, a canonical test
  that included its implementation directly, ABI/test limit drift, missing
  independent SHA boundary vectors, and no normal Linux CI route. The parser
  now rejects depth ceilings above 256, tests link the standalone translation
  unit, runtime-v1 limits are source-locked, fixed SHA vectors cover 55/57/63/
  64/65-byte and binary inputs, and CI runs the native/build-plan gate.
- The first canonical native fixture compile failed on a C++ vexing-parse
  vector declaration. Brace initialization fixed the test; optimized and
  UBSan cases pass. No production source or generated Wasm was left partial.
- The first ABI-result native run reported one live allocation after reset.
  Runtime ownership was correct; the test allocator stopped at an older freed
  record when `malloc` reused the same native address. Release lookup now
  selects only live records. Both optimized and UBSan cases pass.
- Independent review found the first result-sink draft could return status zero
  while retaining `kCanonicalArtifactV3Unavailable`, used an incomplete JSON
  success fixture, and enforced only the ABI-wide 8 MiB maximum. Success now
  requires an absent blocker, a once-bound invocation ceiling, and canonical
  bytes accepted by the strict TypeScript verifier. The sink enforces the
  invocation ceiling below the ABI maximum. Production stays fail-closed until
  strict profile/request decode supplies that ceiling.
- The first pointer-overflow test failed for the wrong reason: its fake
  allocator rejected the forced near-4-GiB address before runtime range
  validation. Forced test addresses now reach the Wasm32 encoder, proving the
  internal-error path rather than allocator resource failure.
- Source edits intentionally invalidated the extractor source set, recipe,
  resource hash, and build-lock ID while review was active. The final stable
  source closure was recomputed once, repinned to source set
  `8652de75e259ce63eada90b42ae6337a417eb76891578d46b549ada9df088fcf`,
  and the strict lock/source gates pass. No generated Wasm existed to mutate.
- Compile-session audit found set-like profile/request ordering depended on
  ambient `localeCompare`, preventing reproducible C++ parity for Unicode
  virtual paths. Shared canonical code-unit ordering replaced every production
  C++/CuTe locale comparison before native decoder work began.
- The first focused Worker-entry oxlint run reported two
  `no-control-regex` warnings in bounded failure-path validation. The path
  check now scans UTF-16 code units explicitly and rejects C0/DEL without a
  control-character regular expression. No external state changed.
- The first full `pnpm --filter @unlocalhosted/browsergrad-compiler
  verify:compiler` run reached the compiler unit suite and failed 1 of 1,398
  tests: `BG-COMPILER-CPP-CUTE-BROWSER-WORKER-ENTRY-INTERNAL` was absent from
  the public compatibility registry. The fix registers the complete transfer,
  runtime, controller, and entry error families rather than only the code the
  source scanner happened to detect. The focused core suite then passed
  180/180 and the complete compiler rerun passed 65 files/1,398 tests. The
  failed run made no external changes and no partial commit or push occurred.
- The first full compiler gate after the metrics source/lock repin passed every
  build-plan test, then found only stale deterministic browser profile and asset
  identities: profile hash, manifest ID, asset-set hash, and manifest hash. Each
  value was re-derived through the production constructors, explicitly repinned,
  and the complete compiler/release gates passed. The metrics commit was amended
  to `c0820ba6`; no runtime behavior or external state was involved.
- Final transfer/runtime review found that explicit caller abandonment was
  recorded as `worker-unavailable`, which would falsely imply a launch attempt.
  Runtime terminalization now accepts an exact typed reason: preparation failure
  and explicit discard use `abandoned`, while only capability-blocked start uses
  `worker-unavailable`. Focused 31-test/type/lint checks and independent review
  passed; no partial external state or live Worker existed.
- The first full native metrics gate timed out only in Apple clang ASan before
  `main`. Sampling showed a recursive lock in ASan/dyld initialization; an
  independently compiled empty `main` reproduced the same hang. UBSan remains
  mandatory locally and passes; the address-sanitizer case is retained for
  non-Darwin CI. No product code executes before this failure, so it is recorded
  as runner/toolchain evidence rather than hidden by a longer timeout.
- Strict native syntax exposed `metrics_realloc` as unused when Emscripten-only
  strong overrides are absent. Marking that internal helper `maybe_unused`
  preserves the production path and strict host syntax. Its byte/hash change
  was propagated through the exact source-set, recipe, lock, and resource
  identities before all focused gates passed again.
- A live pinned Emscripten build could not start because the local Docker CLI
  could not connect to `/Users/v0s06lr/.docker/run/docker.sock`. No container,
  compiler, or build output existed. This delays local pinned-build/call-graph
  evidence but does not put Docker in the browser runtime; Worker transfer and
  source-side work remain available.
- VFS integration review found an instantiation cycle: imports required bound
  memory while exported memory exists only after instantiation. Commit
  `916982a2` creates stable fail-closed imports from the mount before binding
  and activates those exact references after binding exported memory.
- Initial VFS mount/bind review found three P1 boundaries: live
  `WeakMap.prototype` methods could fabricate opaque authority, mutable memory
  and reflection intrinsics could corrupt admission, and mutable destructive
  methods could interrupt cleanup before terminalization. Captured intrinsics,
  exact same-realm memory checks, authority-first terminal state, and hostile
  poisoning tests close them. The API still does not claim cross-realm Worker
  reconstruction.
- The first C++ modular split included an unused metrics header/translation
  unit that owned no behavior. It was removed from CMake, the exact source
  closure, declarations, tests, and every lock hash before commit. Metrics will
  re-enter only with the ABI 1.1 producer. No build output or external state
  existed to clean up.
- Initial header-pack review found that per-pack validation could hash before
  enforcing one global file/metadata budget, source bytes were not yet copied
  and rehashed at the final writer boundary, and build-lock notice policy could
  be mistaken for file-level license authority. Preparation now enforces the
  global budgets before inventory hashing; assembly re-inspects, copies, and
  hashes exact unshared bytes immediately before encoding; and
  `licenseReviewComplete`, output, build, reproducibility, and release claims
  remain literal false with exact notice bytes and external file-license review
  as blockers. No real asset or external state was involved.
- Final scale review reproduced two generic-schema failures and one memory
  undercount. A 20,000-entry content set exceeded semantic-core's default
  cumulative-string budget; a matching large selection could then exceed the
  default selection node/string/document budgets; and inspection retained the
  original pack, its snapshot, and `sha256Hex`'s full input copy while the
  preflight counted only two packs. Content-set and selection canonicalization
  now use explicit compatible closed limits with translated header/VFS errors;
  regressions pass through a matching 20,004-file selection; and the
  conservative byte-copy projection includes the three-pack minimum plus
  canonical/index/source phases. Focused tests pass 19/19 and final review found
  no P0/P1. No partially generated files or external state remain.
- After adding source-verification and distributed-materialization blockers,
  the focused build-lock suite failed 8 of 24 tests because the canonical lock
  ID still named the preceding body. The decoder supplied the required new ID
  `ed7a6929...`; ID, resource hash/length, and exact tests were repinned. The
  settled lock suite passes all 24 tests. No external state changed.
- One concurrent `pnpm --filter @unlocalhosted/browsergrad-compiler build`
  failed while two agents raced the shared `dist` clean/build directory
  (`rm: dist: Directory not empty`). No source was damaged. The command was
  rerun alone and passed; package builds must remain serialized in this shared
  worktree.
- The first combined controller typecheck was run while the parallel header-
  pack file was between edits and reported unused imports, missing policy
  fields, and a removed VFS-copy symbol in that uncommitted file. Controller
  tests themselves passed. After the header lane completed, the combined
  3-file/36-test run, package typecheck, focused lint, and whitespace check all
  passed. No generated or external state was involved.
- Initial controller review found that an invalid start clock could leak an
  already-created Worker, abort could race listener registration, an overdue
  message task could beat the timer task, caller-selected module bytes could
  masquerade as package code, and fake-platform output still had production-
  shaped fields. Start timing now precedes effects and consumes on failure;
  abort and absolute-deadline checks are explicit; production is hard-gated;
  and test output is a disjoint non-authoritative simulation. Final review
  found no remaining P0/P1.

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

### 2026-07-17 — Gate 3 BrowserGrad-owned LibTooling tracer source

- Added the first real BrowserGrad-owned Clang LibTooling source. It resolves
  an anchored `cute::Layout` variable through the AST and retains canonical
  declaration/type/callee observations behind a closed-VFS review action.
- Implemented the exact eight-export C ABI allocation/compile/result/reset
  state machine. Structurally valid input still returns the canonical internal
  failure; `artifact-ready` is unreachable until the VFS bridge, explicit
  device-first/host-second CUDA actions, and artifact-v3 writer exist.
- Pinned the exact C++ and CMake source set into the build-input lock. The
  planner materializes those paths and names the generated Emscripten `.mjs`
  factory plus `.wasm` sidecar separately from the eventual distributed Wasm.
- Added an explicit deterministic Worker-factory-bundle blocker. The Worker
  must consume the exact generated factory with host-supplied Wasm
  instantiation; no relative or ambient sidecar fetch is authorized.
- Independent review found that checked-in tests alone did not make source
  verification a build fact and that no step materialized the generated Wasm
  into the declared distributed path. Both facts now remain false in the
  planner and are explicit release blockers until a real executor enforces
  source hash/length checks and a hash-verified materialization step.
- The repinned build lock is `ed7a6929...`; its canonical resource is
  `2e515985...` (20,263 bytes). These identify selected inputs only and do not
  authorize a build or release.
- Compiler build, 29 planner/source tests, 24 build-lock tests, focused lint,
  and whitespace checks pass. No LLVM/Emscripten inputs were materialized, no
  Wasm was built, and no Docker, Worker, browser C++, or WebGPU execution ran.

### 2026-07-17 — Gate 3 host-owned Worker controller boundary

- Added the host lifecycle controller around the prepared Worker protocol. It
  rehashes the module before Blob use, binds one invocation/nonce, owns the
  first terminal event, checks control/artifact ceilings before copies, and
  retires timer/listeners/Worker/Blob on every terminal path.
- Enforced wall time twice: the host timer cancels an active Worker, while the
  terminal handler compares its monotonic end time with the absolute prepared
  deadline so browser task-source ordering cannot accept a late result.
- Closed start-clock and cancellation races before evidence: invalid timing
  consumes before Blob/Worker effects, and abort is rechecked after listener
  registration before timer or postMessage.
- Kept the missing runtime truthful. Production unconditionally fails with a
  capability error before reading caller input/options or ambient browser
  globals. Removing that gate requires internally pinned package Worker bytes,
  a transferred Emscripten factory, and captured native platform intrinsics.
- Split fake-platform results into `test-platform-simulation` with execution
  false. It has a separate type, brand, WeakMap, and unwrap path and cannot be
  interpreted as production evidence.
- Controller/protocol 2 files/28 tests, strict typecheck, focused lint, and
  whitespace pass. Final independent review found no P0/P1. No Blob or Worker
  was created outside the fake test platform; no browser C++ execution ran.

### 2026-07-17 — Gate 3 deterministic offline header-pack assembly

- Added one opaque complete-profile selection over compiler-resource, libc++,
  CUDA, CuTe/CUTLASS, and Linux-sysroot roots. It binds the exact prepared build
  lock/profile, expected content-set identities, mounted paths, versions,
  revisions, and deterministic output roles without acquiring any bytes.
- Added offline assembly from an exact caller-supplied file inventory. It
  rejects additions, omissions, byte substitution, shared/hostile arrays,
  overlong mounts, and cross-authority drift; hashes exact owned snapshots;
  writes the canonical closed VFS pack; and independently re-inspects every
  result before returning copied opaque bytes.
- Kept license and release state truthful. Build-lock notices are only a
  derived policy projection; exact notice bytes and an externally reviewed
  distributed-file license map remain explicit blockers. Selection and
  assembly always report license/output/build/reproducibility/release authority
  false.
- Closed scale gaps before commit. Global file/logical-metadata budgets run
  before inventory hashing. Content-set and top-level selection
  canonicalization use explicit compatible node/string/document limits and
  translate failures at the header/VFS boundary. A matching 20,004-file
  selection now hashes and canonicalizes beyond generic defaults.
- Added a conservative assembly byte-copy preflight. It accounts for retained
  outputs, owned source/path/index buffers, canonical JSON plus SHA copies, and
  the inspector's original pack, full snapshot, and full-pack SHA input copy.
  A regression rejects a peak that the former two-pack projection admitted.
- Focused header/VFS verification passes 2 files/19 tests. Full compiler
  verification passes 59 files/1,316 tests plus optional Docker 96 tests and
  Clang-Wasm planner/source 29 tests. Release-package verification passes 19
  hostile-archive and 35 Node security tests plus packed/fresh consumers. Final
  independent review found no P0/P1. No network, real header/notice asset,
  build, Docker process, Worker, browser C++, or WebGPU execution occurred.

### 2026-07-17 — Gate 3 VFS successful-read accounting

- Replaced open-call evidence with post-commit content-read evidence. Failed or
  out-of-range reads cannot enter the terminal set. Partial and repeated reads
  reserve/account one exact full-file identity; genuinely empty files can be
  observed by a successful zero-length read.
- Kept ABI field name `openedFiles` for compatibility, but narrowed internal
  vocabulary to `successfullyReadPaths`/`observesFileContent`. This does not
  prove Clang preprocessing or Sema used the bytes.
- Focused VFS session verification passes 1 file/8 tests. Full compiler passes
  60 files/1,326 tests. Final independent review found no P0/P1.

### 2026-07-17 — Gate 3 Worker input-frame and verifier closure

- Initial review found one request-to-artifact limit mapping named a nonexistent
  artifact field, request ceilings could exceed fixed verifier ceilings, and
  internal artifact hashes fell back to generic JSON limits. Preparation now
  derives one complete explicit verifier contract and threads one admitted
  decode budget through every artifact projection and hash.
- Added an exact 64-byte little-endian `BGCCABI1` frame over cached canonical
  profile/request regions. Preparation rejects impossible budgets before
  executable copying or hashing, memoizes one frame per live invocation, and
  rechecks liveness after asynchronous hashing and at every copy boundary.
  Terminalization revokes outstanding frames.
- Focused Worker/controller/protocol/artifact verification passes 4 files/65
  tests, including 4,096 diagnostics and more than 2 MiB rendered diagnostic
  text under an explicit admitted contract. Final independent review found no
  P0/P1. No Worker, Clang-Wasm, or C++ execution occurred.

### 2026-07-17 — Gate 3 source snapshot and Wasm sidecar materialization

- Added a lock-derived executor boundary. It verifies exact checked-in source
  bytes and recursive closure, stages owned read-only inode-bound snapshots,
  and exposes no caller-supplied build-plan authority.
- Added bounded no-clobber sidecar installation with lexical and resolved POSIX
  ancestry admission, same-UID single-writer scope, stable open-file/path
  rebinding, directory sync, exact-inode rollback, abort-before-commit behavior,
  and independent close/temp cleanup settlement.
- Adversarial review found writable-ancestor replacement, identity-swap
  rollback, and composite-cleanup risks. Trusted-ancestry checks, exact linked
  inode rollback, destructive-cleanup refusal after identity replacement, and
  typed aggregate cleanup failures close those gaps.
- Build-plan verification passes 3 files/47 tests; full compiler passes 60
  files/1,326 tests; release-package verification passes 19 Python and 35 Node
  security tests plus packed/fresh consumers. Final independent review found no
  P0/P1. No process, Docker daemon, compiler, or generated artifact ran.

### 2026-07-17 — Gate 3 C++ imported VFS bridge

- Initial review found eager directory materialization could duplicate the
  complete mounted tree, a failed imported close allowed later calls to amplify
  leaked handles, and metadata/construction failures after open could leak the
  acquired handle. Directory iteration is now lazy, any close failure poisons
  the shared adapter, and a post-open RAII guard closes on every failure path.
- Source-level structural and build-lock tests pass, and the exact source/build
  hashes are repinned. The Docker daemon and pinned LLVM/Emscripten toolchain
  remain unavailable locally, so no compile/link or raw-import inspection has
  occurred. Do not diagnose source/API success from the structural tests.

### 2026-07-17 — Gate 3 local Wasm runtime metrics

- Review found the initial allocator cap omitted ABI input/output coexistence,
  fixed stack could exceed current pages, profile `maxMemoryBytes` and
  `maxOutputBytes` were not both enforced, count/cumulative relations could be
  impossible, and labels overstated Worker/reviewed-producer evidence. The
  observer now checks the complete profile/ABI arithmetic and explicitly labels
  its source local and unverified.
- Captured reflection, memory, numeric, timer, freeze, and WeakMap intrinsics
  replace mutable ambient paths. At this slice runtime ABI 1.0 had no metrics
  pointer/export, so the implementation was a verifier/observer contract only;
  D-092 later defines the still-unimplemented ABI 1.1 producer contract.

### 2026-07-17 — Gate 3 package Worker runtime input binding

- Review found forged frame accessors could run before opaque validation,
  cleanup retained only the first cause and could hide the second failure,
  generic native-intrinsic claims were too broad, ambient WeakMap/RegExp/string
  behavior remained mutable, reservation/race tests were weak, and hostile
  proxy failures were not normalized. Validation order, typed aggregate cleanup,
  captured intrinsics, reservation tests, and error translation close those
  gaps.
- The binding deliberately stops after terminalizing the VFS and invocation.
  Repeated capability failure is expected until package-owned Worker bytes,
  generated Emscripten factory code, and independently reviewed first-build
  projections exist; it is not a Docker or browser-runtime failure.

### 2026-07-17 — Gate 3 runtime ABI 1.1 and Worker-realm correction

- The first metrics-ABI draft silently added a strict export to ABI 1.0 and
  also inserted that undefined symbol into the locked linker flags while the
  C++ extractor remained ABI 1.0. The contract is now explicitly ABI 1.1; the
  current eight-export build command remains executable; and one release
  blocker names the missing export, producer, and behavioral conformance.
- Focused validation initially failed four deterministic fixtures: unsupported
  minor-version mutation still used the now-valid minor, the synthetic Wasm
  start index did not shift with the ninth function, and profile/asset hashes
  were stale. These were fixture/pin failures, not runtime observations; exact
  identities were recomputed from canonical bytes and all gates now pass.
- Final review found cumulative allocator counters still allowed incompatible
  realloc and zero-byte interpretations. ABI 1.1 now defines requested-byte
  basis, zero-byte creation, null free, all realloc branches, count changes,
  overflow, and no-op/failure behavior. The C++ producer blocker cannot clear
  without behavioral conformance cases.
- Parallel Worker audit found the prepared VFS session is already bound to one
  exact memory and host-realm opaque WeakMaps. Such objects cannot serve as a
  cross-realm launch protocol. Production work must reconstruct local
  authorities from verified transferred bytes, then bind a memory-independent
  mount once to exported Worker-owned memory. Do not bundle the factory first.

### 2026-07-17 — Gate 3 modular extractor and same-realm VFS bind

- The C++ extractor now separates runtime/lifecycle, imported VFS,
  LLVM-version-sensitive Clang action, and the wired fail-closed artifact
  coordinator. The top-level source contains only the exact eight ABI 1.0
  exports. CMake, executor staging, source closure, and every build-lock hash
  were repinned atomically.
- An empty metrics module was removed during review. It would have created dead
  build surface and implied ownership before any ABI 1.1 behavior existed. The
  real producer will add that unit together with allocator hooks and behavioral
  conformance tests.
- VFS setup now prepares a memory-independent same-realm mount and binds it once
  to exact unshared `WebAssembly.Memory`. Opaque clones and memory subclasses
  fail closed. Captured authority/reflection/memory/cleanup intrinsics and
  terminal-before-cleanup state prevent same-realm poisoning from forging
  authority or leaving reusable live state.
- The split does not make authority transferable. A dedicated Worker must still
  receive canonical bytes, reconstruct installation/request/runtime-ABI/pack
  authorities locally, instantiate verified Clang-Wasm, and bind the mount to
  `instance.exports.memory`.
- Full compiler verification passes 62 files/1,356 tests, including 51 locked
  browser-build tests and 96 optional Docker-parity contract tests. Release
  verification passes 19 Python and 35 Node security tests plus packed/fresh
  consumers. No pinned LLVM/Emscripten compile or browser execution occurred.

### 2026-07-17 — Gate 3 pre-bind VFS imports and ABI 1.1 metrics producer

- Commit `916982a2` breaks the Wasm-instantiation cycle: one frozen six-function
  VFS import table exists before memory, fails closed while unbound, and becomes
  active without changing function references after exact exported-memory bind.
- Commit `c0820ba6` adds the exact ABI 1.1 ninth export and 72-byte module-global
  allocator record. Strong libc entrypoint overrides use the pinned
  `emscripten_builtin_*` bypasses; a raw open-addressed requested-size table
  excludes its own metadata; overflow/corruption sticky-poison the producer and
  force runtime internal failure.
- The native behavioral model covers malloc/calloc/free, zero-byte creation,
  every realloc branch, aligned allocation, table growth/backshift deletion,
  allocator and metadata failure, counter overflow, and untracked-pointer
  poison. Contract/source tests require the exact supported/forbidden ingress
  sets and forbid direct bypass use outside the metrics unit.
- Independent review found no P0/P1. This is not Wasm conformance: the pinned
  Emscripten build, object/final-Wasm call graph, executed pointer/record, Worker,
  and browser execution remain unproved and the release blocker stays active.

### 2026-07-17 — Gate 3 canonical Worker transfer and runtime adoption

- Commit `912aca22` adds the closed host-to-realm transfer: exact canonical
  control regions, unique destructive asset/source buffers, pre-clone ceilings,
  one invocation reservation, one message materialization, and explicit discard.
- Commit `647c814e` reconstructs all opaque compile authorities in the receiving
  realm and makes the package runtime adopt the exact invocation/frame/Wasm and
  pre-bind VFS mount/import tuple once. Cleanup is deterministic and terminal.
- Focused 2 files/31 tests, full compiler 64 files/1,386 tests, optional Docker
  contract 96 tests, and release-package verification pass. Independent final
  review found no P0/P1.
- This is transfer/reconstruction authority, not execution evidence. No actual
  Worker entry or controller consumes the message; the pinned Emscripten build,
  generated factory, Wasm ABI execution, browser C++, and lowering remain open.

### 2026-07-17 — Gate 3 canonical Worker entry and terminal outcomes

- Commit `9ef76452` adds the package module entry and makes the controller test
  shell post the exact canonical transfer rather than a parallel launch shape.
  One controller protocol carries transferable success bytes or bounded typed
  infrastructure failure; authenticated failure fields remain machine-readable.
- Review found pre-adoption realm authority could leak, adopted authority could
  be double-discarded, and a failure-only sink would require redesign when the
  runtime starts succeeding. Entry cleanup now occurs only while input remains
  prepared, runtime retains post-adoption cleanup, and one discriminated sink
  handles both terminal outcomes.
- A real Chromium module Worker proves module loading, one-shot listener
  removal, and pre-identity error routing. Focused 4 files/62 tests, full
  compiler 65 files/1,398 tests, release gates, and independent final re-review
  pass with no P0/P1.
- Production remains blocked before Worker creation. The browser test does not
  contain a valid build, instantiate Clang-Wasm, execute user C++, call the ABI,
  produce an artifact, or mint lowering authority.

### 2026-07-17 — Gate 3 ABI 1.1 result ownership

- Commit `f6431c9a` is pushed to `origin/main`.
- The runtime, not the artifact producer, allocates and owns result storage.
  A producer may bind one strictly decoded invocation output ceiling, request
  one exact allocation, fill it synchronously, and commit once. It cannot
  supply or resize storage; runtime adoption occurs only after healthy metrics,
  nonempty length, nonwrapping Wasm32 range, and input/result disjointness.
- `artifact-ready` is illegal while any review blocker remains. The real wired
  producer still returns `internal-error` with the CUDA-dual-pass blocker. The
  native positive path is injection-only and copies exact canonical bytes that
  already passed the TypeScript artifact-v3 verifier.
- Exact input free after success preserves immutable result pointer/length;
  invalid calls hide but do not leak the result until reset. Sticky allocator
  poison forbids success and requires complete Worker/module disposal because
  production metrics intentionally refuse further tracked frees.
- This closes ABI ownership mechanics, not C++ extraction. The next producer
  slice must strictly decode canonical profile/request regions into one closed
  compile session, derive `maxOutputBytes` from that session, then run explicit
  device-first/host-second actions and write one complete artifact.

### 2026-07-17 — Gate 3 locale-free cross-language ordering

- Commit `449fc9d1` is pushed to `origin/main`.
- Replaced every production C++/CuTe `localeCompare` ordering decision with
  one semantic-core UTF-16 code-unit comparator. Canonical object keys and
  set-like profile/request/artifact/provenance collections now share explicit,
  locale-free ordering.
- Unicode vectors cover composed/decomposed text, supplementary code points,
  and private-use code points. This is a prerequisite for the native decoder;
  it is not native frame validation or executed-Wasm evidence.

### 2026-07-17 — Gate 3 bounded native canonical and identity foundation

- Commit `719cca9b` is pushed to `origin/main`.
- Runtime ABI 1.1 now owns exact canonical-region decoder ceilings,
  accounting, number policy, and direct canonical-byte validation policy.
- Added allocation-free native canonical JSON validation and incremental
  SHA-256. Canonical parsing is strict UTF-8, locale-free UTF-16 ordered,
  duplicate-free, safe-integer-only, and bounded by both ABI budgets and a
  compiled recursion ceiling.
- Added optimized/sanitized hostile fixtures, independent SHA boundary and
  binary vectors, exact extractor source/build identities, transitive
  profile/asset pins, and a Linux CI gate.
- Parallel compile-session audit recorded six contract gaps that must close
  before typed decoder/Clang work: opened-source subset semantics, temporal
  macros, warning mapping, virtual-path parity, pass-scoped VFS observation,
  and diagnostic normalization.
- This slice is a prerequisite, not browser C++ execution. No VFS read, Clang
  action, Emscripten build, Wasm instance, browser compile, or artifact occurred.

### 2026-07-17 — Gate 3 closed semantic adapter and native policy primitives

- Commits `83aee3a9` and `9b9fd341` are pushed to `origin/main`.
- Added one strict canonical semantic-adapter resource for exact Clang 22.1.8.
  Profile 2.5 and compilation contract 1.1 bind its reject-only temporal-macro
  policy, compiler-default warning baseline, closed warning registry, and exact
  asset identity.
- Generated the native temporal/warning tables from the canonical TypeScript
  manifest. Native command materialization accepts typed macro/warning options
  and returns owning argv elements; it exposes no raw shell or argv authority.
- Added reject-only temporal-macro PPCallbacks and a bounded pass-scoped VFS
  observer that records only complete successful reads and admitted resolved
  include edges. These helpers were re-audited against LLVM/Clang 22.1.8.
- Repinned the deterministic build to the exact 21-file extractor closure and
  raised only the explicit source-count ceiling from 16 to 32. The one-MiB
  aggregate source-byte ceiling remains unchanged.
- Full compiler tests pass 68 files/1,426 tests. The build-plan/native gate
  passes 10 files/73 tests with six platform sanitizer skips. Compiler
  typecheck, source lint, build-plan TypeScript/lint, codegen parity, and
  whitespace checks pass.
- Capability remains unchanged for users: the Clang action does not yet use
  the new observer/callback/policy helpers, the runtime does not decode a typed
  compile session, and artifact-v3 production remains the fail-closed
  placeholder. No pinned build, Wasm instance, browser C++, or artifact ran.

### 2026-07-17 — Goal paused and complete worktree checkpointed

- The user paused the implementation goal and explicitly requested that the
  complete worktree be committed and pushed to `main`.
- `cf94cc97` remains the last verified green baseline. The paused code
  checkpoint ends at `7b7757fa` and is known red.
- Checkpoint commits are `0adc5a97` (platform direction), `74e8061f`
  (validated runtime regions), `d45077bd` (diagnostic/profile/assets),
  `b8061280` (sealed native invocation), and `7b7757fa` (typed compile session).
- The focused TypeScript checkpoint has 2 of 5 files passing and 56 of 74 tests
  passing. Profile hashes and browser-asset installation fixtures are stale.
- The focused native checkpoint has 2 of 3 files passing. Invocation and
  runtime pass; compile-session strict and UBSan cases fail at profile offset
  4667 because native decode trails `resourceDirectoryVirtualPath`.
- No real Clang-Wasm build, valid Worker compile, browser-local C++, production
  Artifact V3, or C++-originated WebGPU execution occurred.
- Resume from
  [`package-requirements-handover-2026-07-17.md`](./package-requirements-handover-2026-07-17.md),
  which records exact file ownership, failures, decisions, and recovery order.

### 2026-07-18 — Gate 3 resumed through production native Artifact V3

- Commit `bf353f5f` recovers the red handover and closes the native producer:
  strict canonical session decode, every bound identity, exact VFS source-byte
  admission, sealed CUDA device-first/host-second actions, pass-scoped policy
  and observations, stable diagnostic normalization, and accepted/rejected
  canonical Artifact V3 output.
- Commit `467d2141` hardens file bindings against Linux/Node inode-number reuse
  by including version identity. CI run `29648699782` passed every job.
- This verifies the native producer slice only. It does not prove Emscripten
  compilation, Wasm ABI conformance, Worker execution, browser-local source
  compatibility, or WebGPU convergence.

### 2026-07-18 — Locked isolated Clang-Wasm build and evidence authorities

- Commit `caae5c65` adds the exact no-shell four-step build executor with clean
  argv/cwd/environment, process-group cancellation, four-hour and 16-MiB
  per-stream bounds, immutable logs, input/source rebinding, generated factory
  validation, raw WebAssembly validation, and atomic sidecar materialization.
- Commit `3343226a` adds the manual pinned workflow and runner. The workflow
  verifies the LLVM archive and Emscripten platform/config digests, then uses a
  networkless, read-only, capability-free, no-new-privileges container with
  read-only inputs and one private writable work mount.
- Commits `05679a77`, `2b5604a3`, and `b421f327` resolve three fail-closed image
  admission mismatches without weakening writable-input rejection: immutable
  foreign ownership, immutable ancestry, and image-layer mode bits are accepted
  only when the effective kernel mount is read-only.
- Commit `13d2fb54` records bounded streaming SHA-256 identities for native
  `clang-tblgen` and `llvm-tblgen`. Commit `a615a850` adds a strict two-record
  comparator for actual files, paths, commands/environments, tools, factory,
  Wasm, link map, and logs. Commit `77f41e94` runs the clean builds on separate
  runners and compares their bounded downloaded closures. Commit `1eac0a1c`
  adds a raw-Wasm projection producer against the independent runtime ABI;
  mismatches remain review observations and cannot become conformance.
- CI is green through `1eac0a1c`; the focused boundary contains 19 files and
  126 enumerated cases. The first admitted real build, workflow `29650211276`
  at `b421f327`, has verified acquisition, image identity, and isolation and is
  executing the exact build. It is not capability evidence until it terminates
  and its uploaded outputs are reviewed.

### 2026-07-19 — Compiler harness iteration and quality audit

- The original cold diagnostic run `29658164083` spent 97 minutes 5 seconds in
  isolated execution: roughly 9 minutes 36 seconds producing native TableGen,
  86 minutes 9 seconds producing reusable Wasm LLVM/Clang dependencies, and
  about 67 seconds on BrowserGrad translation units and link at that revision.
  This was unsuitable as an ordinary edit loop.
- Cached run `29667784478` first proved a stable no-migration diagnostic path in
  7 minutes 57 seconds. Commits `0d94e47c`, `49c6afce`, and `0e6186eb` then
  decoupled final link policy from the toolchain cache, restricted fast mode to
  the non-native verification boundary, and used the public Emscripten `-O2`
  link mode to preserve ABI names while retaining `-O3` object compilation.
  Run `29668611133` completed in 5 minutes 3 seconds. Final proof
  `29668822793` at `056aaf02` completed in 4 minutes 39 seconds: JavaScript
  verification took 27 seconds, cache restoration took 2 seconds, isolated
  executor work took 3 minutes 14 seconds, and ABI inspection took 3 seconds.
  The primary key hit directly, cache staging/saving were skipped, and exact-
  source CI `29668819938` passed at the same `056aaf02` revision. Commits
  `5a8a1cf9` and `056aaf02` removed both temporary cache-migration paths after
  their one-time use.
- The fast lane is diagnostic only. Its cache bytes remain untrusted and are
  admitted only inside the networkless, capability-free build container. Clean
  validation and two-build reproducibility never restore that cache, and the
  ordinary CI/clean lanes retain the full Clang sanitizer harness. The narrowed
  toolchain key at that checkpoint bound LLVM, Emscripten, recipe, compiler
  flags, selected libraries, extractor CMake, and platform identity; only final
  link flags iterated independently. The later CMake-stable cache checkpoint
  below narrows this further without granting cache authority.
- Commit `70af1732` made the independent inspector scale to the real production
  module without weakening its structural budgets. Commit `f801bff8` runs that
  review after every successful fast, clean, or reproducibility build and
  uploads the canonical report. The ABI-preserving module is 31,327,956 bytes;
  static inspection completes in about 3 seconds. Its memory export and six
  explicit `browsergrad_vfs_v1` imports are correctly named, but exact
  conformance remains false with ten mismatches.
- The observed module has 92 additional generated imports: 35 `invoke_*`
  wrappers and 57 other imports. The latter include clocks, randomness,
  process/environment operations, descriptors/syscalls, memory mapping, and
  ambient time-zone/file services. These contradict the runtime ABI capability
  ceiling and block Worker construction. The table/support-export inventory and
  target-feature declarations also differ. The ABI must not be repinned around
  this surface; forbidden ambient services must be closed inside the module or
  removed by an equally explicit build/link contract first.
- Audit verdict: the authority design is strong. Exact source/runtime closure,
  no-shell execution, read-only/networkless/capability-free isolation, bounded
  immutable logs, inode-version rebinding, independent Wasm parsing, native
  `WebAssembly.validate`, and separate diagnostic/clean/release authorities all
  fail closed. The maintenance shape still carries material debt. The executor
  is 2,237 lines against a 2,300-line ratchet; native `CompileSession.cpp` and
  `ArtifactWriter.cpp` are 2,121 and 1,784 lines; several semantic TypeScript
  modules are 5,000 to 8,000 lines. Cold provisioning remains roughly 40
  minutes, and cached links regenerate about 29 seconds of Emscripten system
  libraries. These require decomposition and a separately scoped system-library
  cache, but not at the expense of the current authority boundaries.
- Local verification passed the architecture and package gates, the optional
  Docker-shell contract with 96 tests, the complete native build-plan boundary
  with 169 passed and 9 intentional platform skips, and the full compiler suite
  with 1,446 tests. This evidence proves the harness changes; it does not prove
  a clean artifact, reproducibility, ABI conformance, a Worker launch, licensed
  header distribution, browser-local C++, or WebGPU convergence.

### 2026-07-19 — Generated Wasm surface closure and exact ABI conformance

- Capability commits from `08c25c5e` through `edbe26cf` closed forbidden clock,
  entropy, environment, sleep, descriptor, path, memory-map, process, and
  retained filesystem imports inside the module. The total function-import
  surface fell from 98 to 58: six explicit `browsergrad_vfs_v1` functions plus
  52 generated functions. The remaining generated set is exactly 48
  JavaScript-exception/control-flow shims, two bounded memory-growth helpers,
  one stack-overflow trap, and output-only `fd_write`; unlisted imports remain
  forbidden and the observed module cannot extend the allowlist.
- Commits `bb847df7`, `4949f149`, `35366b71`, `e5bb1d1c`, and `6999f9c8`
  independently pinned those generated imports, 29 worker-internal support
  functions, the single fixed `funcref` exception-dispatch table, advisory
  target metadata, and the single-memory `memory.copy`/`memory.fill` subset.
  Production run `29673488166` built a 31,307,834-byte module whose uploaded
  report records raw-Wasm verification, 58 imports, zero mismatches, and exact
  interface conformance. This authority deliberately leaves Worker execution
  and release false.
- The warm diagnostic lane remained stable across the capability series:
  production runs completed in 4 minutes 45 seconds, 4 minutes 30 seconds, 4
  minutes 59 seconds, and 4 minutes 27 seconds without changing the toolchain
  cache key. Source-only syscall seams used an included implementation fragment;
  at that checkpoint, adding a CMake translation unit was rejected because
  CMake source-list drift changed the expensive toolchain key.
- The audit also found that independently invoking a build-lock check and the
  fast suite concurrently can race because both touch package `dist`. The
  canonical `verify:browser-clang-wasm:fast` command now builds once, runs the
  no-rebuild lock check, and executes all 266 fast harness tests sequentially.
  The expanded set includes runtime ABI, browser profile, and browser asset
  identity propagation. Measured end-to-end local runs complete in 21.74 to
  26.37 seconds on Node 25.
- Commit `c12a8242` advances ABI 1.7 by separating browser-visible required
  features from the inspector-only `bulk-memory-opt` marker. Test fixtures now
  derive the required list from the canonical resource instead of copying it.
  Full compiler unit passes 69 files/1,446 tests; the broader local native gate
  passes 170 tests with 9 platform skips in 91.91 seconds. Commit `348d7373`
  separately makes `rename`/`renameat` host shims inherit the selected libc's
  exception specification, closing the glibc/libc++ portability split without
  platform-name conditionals.
- Remaining quality debt is explicit: cold provisioning is still roughly 40
  minutes, the cached link regenerates about 29 seconds of Emscripten system
  libraries, the executor is 2,237 lines against a 2,300-line ratchet, native
  producer files remain large, and several semantic TypeScript modules remain
  5,000 to 8,000 lines. These are decomposition/cache tasks, not reasons to
  collapse diagnostic, clean, reproducibility, Worker, or release authority.

### 2026-07-19 — Exact frontend work, canonical Worker results, and current clean attempt

- `20e851e3` advances C ABI 1.2 with one exact 96-byte frontend-work record.
  The native producer measures include, preprocessing, AST, constexpr,
  template, and CUDA-pass work and fails closed on overflow or prepared-profile
  ceilings. The Worker reader verifies generation and idle/complete/reset
  lifecycle instead of trusting caller counters.
- `ef51de56` replaces ten ad hoc compile-session limit accessors with one
  versioned semantic-limit index. `CompileSession.cpp` is back under its
  architecture ratchet at 2,112 lines; the architecture check remains green
  without raising either native-file budget.
- `be703a05` makes the Worker-local runtime build bounded canonical result
  control from independently verified Artifact V3 bytes plus exact runtime,
  frontend-work, and VFS observations. At that commit the production host
  controller remained blocked on self-contained package Worker bytes and minted
  no Worker-execution or lowering authority; the exact-bundle section below
  supersedes that packaging blocker without enabling production.
- Before the exact-bundle checks were added, the canonical local command passed
  333 fast tests in the established 21-to-27-second range. Full compiler unit
  passes 73 files/1,464 tests; Worker
  focused coverage passes 7 files/89 tests; typecheck, lint, build-lock
  authoring, and architecture checks pass on Node 25.
- Historical clean run `29674887505` completed in 39 minutes 29 seconds, with a
  36-minute-25-second locked build step, but predates the current ABI.
  Current-source clean run `29678087663` failed after 46 minutes 25 seconds.
  CMake generated patched upstream `ExprConstant.cpp` into the binary tree, so
  its relative private include `ByteCode/Context.h` no longer resolved without
  an explicit `clang/lib/AST` include. The failure record points to
  `$.steps[3]`; no factory, Wasm, ABI review, clean, or release authority was
  produced.
- Reproducibility run `29676333678` completed both clean builds but failed its
  final exact-tree check because each build correctly contained the new
  `clang-wasm-runtime-abi-review.v1.json` sidecar while the comparator still
  declared only the older output set. `eda1ad9d` admits, binds to the Wasm
  identity, canonical-checks, and compares that sidecar with a regression test.
  The failed run grants no reproducibility authority.

### 2026-07-19 — Exact package Worker bundle

- Vite/Rolldown authoring now builds the package Worker entry in two separate
  sequential passes, requires one ES entry chunk, rejects every static or
  dynamic import and ambient source-map/path reference, and exact-renders one
  checked package resource. The module at that checkpoint was 574,770 UTF-8 bytes with
  SHA-256 `4b1565e4ca8df2332cc6faabb6643226aa5cb26bcd5d9a8ac6d97d94972797a5`.
- The runtime strict-decodes and hashes those package bytes into an opaque
  authority; copies are isolated and structural forgeries fail closed. This
  proves package identity and a self-contained module graph, not Worker
  execution or release readiness.
- A real headless Chromium test creates a Blob module Worker from the exact
  verified bytes and observes the one-shot pre-identity error boundary. It does
  not invoke Clang, the generated factory, or the C ABI and grants no production
  Worker authority.
- The canonical fast gate now includes bundle staleness/determinism checks and
  passes 30 files/337 tests in 29.24 seconds end to end on Node 25. Focused
  controller/runtime/bundle coverage passes 3 files/41 tests; the exact-byte
  Chromium smoke passes 1 file/2 tests; typecheck, lint, build-lock authoring,
  architecture ratchets, and whitespace checks remain green.
- Production stays capability-blocked before reading caller input or ambient
  browser globals. Its remaining executable seam is the captured native
  platform adapter plus package-owned invocation composition; caller-supplied
  profile or module bytes cannot authorize code.

### 2026-07-19 — Private Clang include preflight

- The external extractor target now resolves and attaches the exact
  `clang/lib/AST` source directory needed by the generated patched
  `ExprConstant.cpp`. It still retains Clang's public source and generated
  binary include roots.
- Configured-target review independently requires all three canonical include
  roots in generated `CXX_INCLUDES` before the expensive extractor build. Its
  fixture omits each root in turn, and executor fixtures must model the same
  generated topology.
- The reviewed build-input lock advances to
  `bg.cpp.browser-build-input-lock.sha256.27ee2aa9e9f7c533a61ffa1803f205f0eeda10101f33c008e08fff05507cdd03`.
  This is input-selection authority only; a cached validation and then a clean
  build are still required.
- The canonical Node 25 fast gate passes 30 files/338 tests in 28.86 seconds;
  full compiler coverage passes 74 files/1,467 tests. Typecheck, source lint,
  architecture ratchets, exact Worker-bundle authoring, and build-lock
  authoring are green.

### 2026-07-19 — CMake-stable diagnostic toolchain cache

- Fast-validation run `29679722745` was deliberately cancelled after its cache
  projection changed from the populated key solely because the private-include
  CMake edit changed `CMakeLists.txt`. Continuing would have converted a
  diagnostic run into another cold LLVM/Clang build.
- Diagnostic cache selection now binds only inputs capable of changing the
  reusable LLVM/Clang layer: exact upstream and builder identities, recipe
  compiler inputs, platform, and the independently selected Clang libraries.
  Extractor CMake and final-link flags are excluded because every admitted cache
  is mandatorily reconfigured, all BrowserGrad target objects are invalidated,
  and generated target flags are independently reviewed before compilation.
- One exact legacy key is derived from the same current reusable inputs plus the
  last populated CMake identity. The workflow may restore only that compatible
  full key, then saves the successful reconfigured tree under the new primary
  key. There is no broad or trust-granting fallback.
- Migration run `29680036963` exposed that `actions/cache/restore` reports
  `cache-hit: false` for a compatible restore-key match even after downloading
  it. The admission step therefore ignored the restored tree and began another
  cold build; the run was cancelled at roughly 20 minutes. Admission now uses
  the pinned action's `cache-matched-key` output, while cache staging still uses
  the exact-hit signal so a compatible restore can be saved under the primary
  key.
- The new primary key is
  `bg.cpp.clang-wasm-toolchain-cache.sha256.eb130b274f93b28885dcecd1147c69763630b900c9178a6fc056c54b4f7a42f3`;
  its exact compatible legacy key is
  `bg.cpp.clang-wasm-toolchain-cache.sha256.ba2867dd05bdc99485e0de219510c5f87ec5c5a459d2ee607bca40be549de41a`.
  The canonical Node 25 fast gate remains 30 files/338 tests and passes in
  29.20 seconds.

### 2026-07-19 — Exact-primary timing and runtime ABI 1.8 review

- Migration run `29680686426` completed in 4 minutes 14 seconds, including 2
  minutes 29 seconds for the isolated build, and saved the CMake-stable primary
  cache. Exact-primary run `29680831101` completed in 4 minutes 17 seconds:
  JavaScript verification 32 seconds, cache restore 3 seconds, isolated
  compile/link 2 minutes 25 seconds, and raw-Wasm review 2 seconds. Exact hits
  correctly skip cache staging and saving.
- The resulting 31,641,378-byte module has SHA-256
  `20cf9ee448af03cd91395eb098e29a9c04be741097f4fb8c7d41fc602b68fc0a`.
  Raw parsing passed, but comparison against the prior ABI exposed 14 new
  Emscripten `invoke_*` signatures and a `funcref` table increase from 14,549
  to 15,166 entries. No prior generated import disappeared.
- Each new function is present in the exact generated factory and uses the
  same bounded exception bridge: stack snapshot, indirect table call, catch
  restricted to Emscripten's exception marker, stack restoration, and
  `setThrew`. The additions expose no clock, randomness, network, process,
  environment, or ambient filesystem capability.
- Runtime ABI 1.8 now pins 66 generated imports, including 62 exception-control
  functions, and the exact 15,166-entry table. Its manifest ID is
  `bg.cpp.browser-runtime-abi.sha256.4e870314f139bd8af2f0e45a7410d8d3c6eaa2a8d358064bffd534ae7a91cb18`.
  Detached local review of the exact run artifact reports raw-Wasm verification,
  exact interface conformance, and zero mismatches. Worker execution and
  release authority remain false.
- The build lock advances to
  `bg.cpp.browser-build-input-lock.sha256.bf62353c9421b955cd1a07e14e13c5e3417b5431e2be4555283acdacc0ee7def`
  because the native extractor embeds the runtime-ABI resource SHA. The warm
  lane must compile this repinned source before any clean run.
  The complete local fast gate passes 30 files/338 tests in 24.7 seconds.
- Exact-source warm run `29681607575` completed successfully at commit
  `640c6d51` in 5 minutes. It restored the 84,062,973-byte primary cache in 2
  seconds, completed the isolated build in 3 minutes 19 seconds, skipped cache
  staging/saving, and raw-reviewed the output in 3 seconds. The output Wasm is
  31,641,377 bytes with SHA-256
  `5fc425bbc051a2f5be588c2acbb164efb5e43f949afb48a373f3ed022c3b8758`;
  exact interface conformance is true with zero mismatches.

### 2026-07-19 — Clean admission, package Worker controller, and current factory

- `52e1da4a` admits one exact current-lock clean artifact tree only after every
  generated output, distributed Wasm, link map, bounded log, execution record,
  runtime closure, and raw-Wasm report agrees. `ec24e689` exposes only an
  immutable, no-clobber factory candidate outside the admitted tree and
  preserves clean-build, reproducibility, provenance, Worker, and release
  authorities as separate facts.
- `5733afe2` removes caller-provided Worker code from production input and
  composes one package-owned invocation with a fresh verified bundle copy.
  `bc429d5e` captures Blob, Worker, object-URL, monotonic-clock, timer, and
  EventTarget effects at module evaluation. `4f4e5814` uses those two seams in
  the production controller, rechecks Worker bytes immediately before Blob
  construction, authenticates exactly one terminal frame, and closes the
  Worker, URL, timer, and listeners before returning an opaque observation.
  It deliberately mints no lowering authority.
- `9a32ac6c` moves the package invocation, Worker entry, production controller,
  and controller lifecycle suites into the ordinary fast gate. The gate now
  covers 34 files/381 tests and completed in 25.26 seconds end to end on Node
  25, including a 17.05-second Vitest phase.
- `6ef19487` adds one strict two-argument command that admits a downloaded clean
  tree and writes its factory candidate. Exercising that command against the
  real artifact exposed two fixture-only assumptions: configure steps were
  incorrectly expected to run from their build directories, and the verifier's
  one-MiB report ceiling was below the raw inspector's own bounded output
  contract. `e31ce141` aligns configure CWD with the production source root,
  shares the inspector's 64-MiB/one-million-node canonical bounds, rereads the
  admitted Wasm, independently reruns the raw inspector, and requires the
  uploaded report to match byte for byte. The admitted current report is
  1,678,025 bytes; the independent review takes about 1.6 seconds locally.
- Clean run `29681845216` at `aca7ee4e` completed in 49 minutes 2 seconds. Its
  isolated build took 45 minutes 37 seconds and produced factory SHA-256
  `796a548237420df7f5eca0c0260d3cbe752aeca155d9c7182c6ad0f5491dfb12`
  at 27,125 bytes plus Wasm SHA-256
  `5fc425bbc051a2f5be588c2acbb164efb5e43f949afb48a373f3ed022c3b8758`
  at 31,641,377 bytes. `96ad7b16` packages that exact factory and regenerates
  the zero-import 577,480-byte Worker bundle at SHA-256
  `db0b0fd8f622d8c5febf7dc2c4daa75d68bf8be879f2d4f3d3670b56836f71d7`.
  Full unit passes 76 files/1,474 tests; real Chromium Worker bundle/platform
  smoke passes 1 file/3 tests; the broader build-plan gate passes 187 tests with
  9 intentional platform skips in 77.84 seconds; full `verify:compiler`,
  typecheck, lint, architecture, exact authoring, and whitespace checks are
  green.
- Reproducibility run `29683677087` completed both cache-free builds at exact
  commit `96ad7b16`. Their Wasm and factories match exactly. The original v2
  comparator rejected only the deliberately distinct absolute build roots in
  the non-distributed linker maps, so that failed comparator still granted no
  reproducibility authority by itself.

### 2026-07-19 — Exact header-pack inventory, materialization, and notices

- `6351bcd4` adds a strict recursive inventory for the five complete locked
  header roles. It canonicalizes portable paths, rejects links and special
  files, records exact identities/lengths/SHA-256 values, detects mutation, and
  mints no distribution or release authority. `b5e84b91` requires the exact
  current build lock, output role/path, resource identity, and notice-policy
  projection for every role.
- `dab2fd49` retains source access only through an opaque same-process authority
  and exposes bounded exact rereads; copied or serialized projections cannot
  recover paths or bytes. `029f675b` feeds those rereads into the existing
  canonical VFS writer, installs outputs into a private no-clobber tree, then
  independently rereads and inspects the persisted pack before returning.
  `2e2960f1` keeps that output authority opaque as well.
- `730e63dc` begins exact notice-resource verification; `b64d8974` completes
  the package set for every one of the ten notices already approved by the
  current lock. The standalone verification command rehashes and length-checks
  all ten exact checked-in resources and emits one bounded canonical record.
  CUDA 12.6.3 and the Linux sysroot remain explicit unresolved distribution
  policies, and a separate externally reviewed per-file license map remains
  required. No code infers legal approval from build-lock policy.
- The current Node 25 fast gate passes 38 files/406 tests in 27.49 seconds end
  to end, including a 19.08-second Vitest phase. Strict typecheck, source lint,
  architecture ratchets, build-lock authoring, Worker-bundle authoring, and
  whitespace checks are green. Tests use synthetic tiny trees; no real header
  pack, network acquisition, external legal review, Worker compile, or release
  evidence is claimed.
- Run `29683677087` ultimately completed both clean builds. The exact result and
  the repaired current-verifier admission are recorded below.

### 2026-07-19 — Reproducible extractor evidence without a second rebuild

- Run `29683677087` produced two exact cache-free copies of the 31,641,377-byte
  Wasm and 27,125-byte factory. Build 1 took 48 minutes 33 seconds, including a
  44-minute-57-second locked build. Build 2 took 27 minutes 37 seconds,
  including a 24-minute-51-second locked build. The original v2 comparison
  failed only at `$comparison.linkMap`: its raw linker maps retained their
  intentionally different isolated build-root paths.
- `c01b5f25` introduces reproducibility v3. It preserves both raw linker-map
  hashes but compares a strict root-canonical semantic projection. Only the six
  recorded roots may be substituted, replacements are boundary-aware, and
  foreign build roots, reserved placeholders, invalid UTF-8/NUL, and semantic
  drift remain fatal.
- `c1a79c0d` adds a verifier-only workflow that admits immutable artifacts only
  from the exact repository, event, source workflow, commit, and two successful
  clean jobs. Run `29685632925` completed that path in 1 minute 7 seconds. Its
  exact 3,470-byte v3 evidence has SHA-256
  `6c7aebe1376edf0f9a526b55bacd930e7e9e0fd454a95213d97931117548f31a`;
  its independently regenerated 1,678,025-byte runtime-ABI review has SHA-256
  `2da68fceb93f95c840b7974482b03ec6426f2339198b77e449140d8c51198884`.
- `2d4a6fa5` package-pins the exact v3 evidence, verifies it byte for byte behind
  an opaque authority, marks only extractor reproducibility true in the factory
  record, and regenerates the zero-import Worker bundle. The current bundle is
  577,876 bytes with SHA-256
  `652356abb90ad3ac80f2751e9db774185a3be529a93be597b3fe54921de910cf`.
  At that checkpoint the fast gate passed 37 files/400 tests in 29.98 seconds,
  and real Chromium
  exact-bundle/platform smoke passes 1 file/3 tests.
- This evidence is deliberately narrow. It proves reproducibility of the
  extractor Wasm/factory only; real header packs, the complete distribution,
  signed provenance, valid Worker compilation, lowering, and release authority
  remain false.

### 2026-07-19 — Exact locked source-archive admission

- `3acf5b65` adds a caller-policy inspection seam plus a separate package-lock
  authority for exactly LLVM 22.1.8 and CUTLASS 3.7.0. It accepts only
  current-user-owned canonical regular files in non-writable canonical
  directories, rejects links and duplicate inodes, streams bounded files
  through SHA-256 with before/after identity checks, and retains local paths
  only behind opaque same-process authority.
- The no-clobber copy path rechecks the live source inode, streams and syncs the
  destination, independently rehashes the persisted bytes, and removes only
  the exact inode it created on failure. Serialized or forged projections
  cannot copy an archive. The command itself performs no network access,
  extraction, tree verification, legal review, build, or release transition.
- The two archives acquired from the current lock's URLs were admitted locally:
  CUTLASS is 29,728,321 bytes at SHA-256
  `dfcafb7435a1b114ce32faee4f3257e276caf08f55fea04fa8bf3efa3a83c814`;
  LLVM is 167,061,596 bytes at SHA-256
  `922f1817a0df7b1489272d18134ee0087a8b068828f87ac63b9861b1a9965888`.
  The path-free 196,789,917-byte set has admission ID
  `bg.cpp.current-source-archive-admission.sha256.fe50109343304f2826967f2fc2438a7f1b9540edc3aa45e437fee1d66c75cf7b`.
- The Node 25 fast gate now passes 38 files/406 tests in 27.49 seconds end to
  end, including a 19.08-second Vitest phase. Strict typecheck, focused and
  package lint, architecture, and whitespace checks remain green.

### 2026-07-19 — Exact seven-archive header-pack pipeline

- `1213cdea` and `fae32ded` replace the partial archive set with one exact
  current-lock plan for LLVM 22.1.8, CUTLASS 3.7.0, CUDA 12.6.3 CCCL/cudart/nvcc,
  and Ubuntu Noble glibc/Linux UAPI cross-development packages. The admission
  streams and verifies 252,406,685 bytes under
  `bg.cpp.browser-header-source-archive-admission.sha256.b365f8f65931a94728775dc56b173ac7673a2a528f1e5fe12de7451e76881d20`.
  Every selected subtree is bound to its intended asset and license component;
  plan ID is
  `bg.cpp.browser-header-source-plan.sha256.360bfef6bc046ebbed5cc51b3a585a10ad0189b107d92f14df89772b265c1162`.
- `1480b079` adds a bounded streamed PAX/ustar selector that rejects malformed
  checksums/padding/PAX, traversal, links, special entries, duplicates, glob
  hazards, truncation, and all configured byte/file ceilings. `638264ef`
  removes per-header `fsync` while retaining directory-level durable commit;
  CUTLASS's 682-file materialization falls from about 4.6 seconds to 0.4
  seconds. `48ac1055` wraps a caller-selected `bsdtar` in a closed no-shell
  process boundary and uses Node 25's bounded Zstandard stream for Debian
  `data.tar.zst`, removing an undeclared ambient `zstd` CLI dependency.
- Default macOS cannot represent case-distinct Linux names directly in one
  directory. The normalizer therefore stores files under collision-free hashes
  while retaining portable virtual paths in authenticated metadata. This
  preserves real pairs including `xt_CONNMARK.h` and `xt_connmark.h` and makes
  extraction independent of host case folding.
- `e9b1e1a5` composes admission, private staging, strict normalization, and
  archive-copy removal. `91fc4610` keeps all opaque source authorities alive
  through exact inventory and canonical VFS materialization. It observes seven
  archives, eight selected subtrees, 5,769 files, and 67,092,008 content bytes;
  its five independently inspected packs total 67,495,319 bytes. Pipeline ID is
  `bg.cpp.browser-header-pack-pipeline.sha256.431f8b7c20bfdb4b6f129a04fb22ee24c5bcdd53cf29f2dd3b953abf007d39e2`.
- Exact pack SHA-256/byte pairs are: Clang resource source
  `1d05ba63268f4ba0f19efd52e17f13b52978f5e3319479e178b688a7dda8b2ba` /
  7,729,504; CUDA
  `f795494ab3d97cbed3e8dab374daeb90574bbd52f0d462b3466bd89e2aa11a77` /
  16,848,942; CUTLASS
  `4f1c39b73f2fa7252628a253f7bb5b1411bdfdada872c5ff733b1b9008d89555` /
  21,403,975; libc++
  `f66f128496f99e535e9d461fc9cf4b8b18f7a4e75406982470dd3451616f4fc2` /
  12,585,828; Linux sysroot
  `d04a460dc605703b8e8a104cc5c043e6a7020ca7201991470c615273d43e7ae4` /
  8,927,070. A second composed run produced the same identities.
- The exact end-to-end pipeline completed in 16.97 seconds. The ordinary Node
  25 harness passes 45 files/425 tests in 29.86 seconds, including a 21.24-second
  Vitest phase; architecture reports zero cycles, leaks, or failures. The old
  90-minute-class LLVM provisioning path is no longer an edit-loop dependency.
  Cache-free extractor builds remain a separate 25-to-49-minute evidence lane.
- Authority remains deliberately narrow. The observed `bsdtar` executable is
  hash/version recorded but not independently attested or package-pinned;
  generated Clang resource headers are incomplete; external file-level license
  review and remaining distribution review are unresolved. These packs are
  real source-derived non-release observations, not approved product assets.

### 2026-07-19 — Configured Clang resource output closure

- `7db7078b` binds the exact 25,049-byte LLVM 22.1.8
  `clang/lib/Headers/CMakeLists.txt` at SHA-256
  `6fbe03bc7a1ae8309451851c666a76fe0929d12a9e140be18b53428574cdbd35`
  to the locked `clang-extractor-wasm` stage. That stage sets
  `LLVM_TARGETS_TO_BUILD=WebAssembly` and `CLANG_ENABLE_HLSL=OFF`; the pinned
  upstream rules generate resource headers only for ARM/AArch64/RISC-V target
  selections, so the configured generated-file set is exactly empty.
- Extraction verifies that manifest identity before resolving the generated
  header blocker. Inventory rereads it again, omits it as build metadata, and
  distributes the remaining 277 configured resource files. The configured
  Clang pack is 7,704,399 bytes at SHA-256
  `fd7fb977130d1181c5ce0e038472a45e30623b54e0249b167f5f2ed228b51977`;
  its content-set SHA-256 is
  `5d257f0f00612d5af5f11458445102a19771eed92804cbffa72559e33e51dd9b`.
- The complete five-pack output now contains 5,768 files and 67,470,214 bytes.
  Current plan, admission, extraction, inventory, and pipeline IDs are
  `bg.cpp.browser-header-source-plan.sha256.f4c97df4ad6e8413bf8c66d488bd12c3f175778054e5ed80e6c124e8790ceb4f`,
  `bg.cpp.browser-header-source-archive-admission.sha256.1b42a2b4c2729facb403195d85aab884a5640cc43de10e00143c4c0d58216392`,
  `bg.cpp.browser-header-source-extraction.sha256.a2d38003f89079b453355cd57db6f91dc5f5ea2254835950b3d0d69f9377f3cb`,
  `bg.cpp.browser-header-pack-source-inventory.sha256.87c98a652c4d6d8168c6b27de527e1163e1da6ce1d7c24a70e46c42a257f72ec`,
  and
  `bg.cpp.browser-header-pack-pipeline.sha256.15540563830401f2f2822bdd1ed745a177e8943ccbf090e61bd6ea6549584a8c`.
- The composed CLI now creates its own private no-clobber pack root, cleans that
  exact owned root after failure, and prints a bounded causal diagnostic chain.
  This fixes the prior mismatch where the runbook showed an absent output path
  but the lower materializer silently required a pre-created directory.
- The real exact command completed in 22.51 seconds including package build.
  The ordinary Node 25 gate passes 45 files/426 tests in 28.60 seconds, with a
  20.17-second Vitest phase; architecture reports zero cycles, leaks, or
  failures. Normalizer attestation and external distribution/license review
  remain unresolved, so release authority stays false.

### 2026-07-19 — Package-pinned archive normalization closure

- `b0b4e3cf` narrows the release-shaped header pipeline to one reviewed Darwin
  arm64 builder identity. It requires `/usr/bin/bsdtar` SHA-256
  `2806c6e01f077f360f4046e597ef1a62d96c772eb937b5c35852ad97c9d0a625`
  at 195,680 bytes and exact version output. Debian Zstandard normalization
  additionally requires the Node 25.9.0 executable SHA-256
  `4b3fe8b384e30ee917e28a9f5b79a3ca64b72b13b70d9ab2273e6e9a823f4cbf`
  at 133,274,256 bytes, Zstandard 1.5.7, no `execArgv`, and no `NODE_OPTIONS`.
  Other platforms and runtime/tool substitutions fail closed; a future builder
  needs its own independently reviewed identity.
- The resulting environment ID is
  `bg.cpp.pinned-archive-normalization-environment.sha256.d9461759522fbe616b0244ab63267854eb249f546a1b8560c7b7b0cd6b6df818`.
  A package command verifies it before expensive work. Generic synthetic
  normalizer tests remain portable but cannot mint this release-shaped
  authority.
- The final exact extraction, inventory, and pipeline IDs are
  `bg.cpp.browser-header-source-extraction.sha256.fa3fa59be5d79c6a30fcb1240f75a3df5862aca355c760dbfbceac5874c5bc39`,
  `bg.cpp.browser-header-pack-source-inventory.sha256.b3bf111bfbe9d45c2e492b98953f15cf84d534ef43e05efaa5d9d6f6b1caa495`,
  and
  `bg.cpp.browser-header-pack-pipeline.sha256.b204eda9f2f8f7cab60396bb3e60f85a6abe422e63197a85e4b091d90f91ff3c`.
  All five pack hashes and byte lengths remain unchanged. Direct exact
  materialization completed in 22.70 seconds.
- The final fast corpus passes 45 files/427 tests in about 28 seconds on Node
  25. The focused JSON-reporter run passed 427/427 tests in 23.53 seconds, the
  exact environment verifier passed, and architecture reports zero dependency
  cycles, legacy leaks, semantic-IR leaks, frontend leaks, or coverage
  failures.
- This closes deterministic normalizer/decompressor package identity and the
  configured header-universe engineering blocker. It deliberately does not
  claim independent third-party tool attestation, external per-file license or
  distribution approval, signed provenance, Worker execution, or release
  authority.

### 2026-07-19 — Exact header distribution review input

- `20c93839` extends the strict selected-tar seam from subtrees to an explicit
  `subtree | file` policy and extracts the exact license/copyright bytes from
  the same seven admitted archives without widening the distributed header
  universe. The current plan and extraction bind eight evidence files totaling
  250,207 bytes: three CUDA Toolkit license copies, CUTLASS, Clang, libc++, and
  the two Debian source-package copyright files. Exact-file selection rejects
  every unselected sibling and preserves the existing five pack hashes.
- `b371db1a` admits the package-pinned 49,142-byte CUDA 12.6.3 redistribution
  index from a current-user private directory. It verifies SHA-256
  `9c598598457a6463eb92889080c16b2b9dc04150e501b8bfc1536d403ba70aaf`,
  release metadata, all three selected component versions, archive paths,
  archive hashes/lengths, and the upstream `CUDA Toolkit` license references.
  Index authority is exact-plan metadata evidence only; it performs no network
  access and decides no redistribution policy.
- The one-process pipeline now binds the live extraction, full 5,768-file
  inventory, independently reread persisted packs, all ten exact package
  notices, the eight extracted evidence files, and the CUDA index. It writes
  the build-lock-declared
  `assets/browsergrad-cpp-cute/license-inventory.json` through a bounded,
  private, no-clobber path. The exact 1,203,103-byte output has SHA-256
  `0c809a69554731e3b78acc8c0717c1c0939883b8df811637120a067e28be97b7`
  and review-input ID
  `bg.cpp.header-distribution-review-input.sha256.28346afa0239011e988de9cd40818b9eff6bbeeecc6e7e4bf508137697f4dc82`.
  Its ordinals cover 0 through 5,767 with 5,768 unique pack/path pairs and the
  exact five-component set: Clang, CUDA headers, CUTLASS, libc++, and Linux
  sysroot.
- Two final exact runs produced identical review-input and pipeline identities.
  The current pipeline is
  `bg.cpp.browser-header-pack-pipeline.sha256.fbc8e39bb2bf49a5124a10bce434e121a5638d42d60ed7df6075086842a9a154`
  and completed directly in 22.39 seconds. The fast test phase passes 47 files
  and 431/431 tests in 23.55 seconds; strict TypeScript, oxlint, whitespace,
  package lock/Worker checks, and architecture all pass with zero cycles,
  boundary leaks, or coverage failures.
- This completes the package-owned engineering input for external header
  distribution review. It explicitly leaves the CUDA, Linux-sysroot, and
  per-file review blockers active and keeps `licenseReviewComplete`,
  `distributionAuthorized`, and `releaseReady` false. No package-generated
  manifest can approve its own legal conclusions.

### 2026-07-19 — Exact distribution notice materialization

- `061ce246` retains each verified package notice byte snapshot behind the live
  notice authority and returns only isolated copies for exact component IDs.
  Serialized or copied metadata cannot select a path, provide replacement
  bytes, or recover the retained snapshot. Mutation of a returned copy cannot
  affect later materialization.
- A shared bounded output primitive replaces the duplicated review-input file
  harness. It validates the complete expected file and directory tree, rejects
  links, writable or multiply linked existing files, traversal, unexpected
  empty directories, byte drift, and clobber. It writes `0400` files under
  private `0700` parents, syncs and independently rereads them, verifies the
  exact final tree, and removes only the exact created identities after error.
- Header pipeline v4 materializes all ten build-lock component-license outputs
  from those retained bytes plus one deterministic aggregate. The component
  files total 111,010 bytes. The aggregate format
  `browsergrad.compiler.cpp-cute.distribution-notices.v1` records ten components
  and nine third-party components, embeds every untouched notice, and produces
  115,316 bytes at SHA-256
  `9933f791012ac5662cc87a63cdf56d794893e59d49ac8a013c05f29709b4e30b`.
  All eleven new outputs total 226,326 bytes. Notice materialization ID is
  `bg.cpp.browser-header-notice-materialization.sha256.a560867d2614feeb1c460c2192435b5472780c5538ab8d0c11c2cf6207df430a`;
  generic output materialization ID is
  `bg.cpp.distribution-output-file-materialization.sha256.4aadc8426edd1ce838edf77b615a451a080b877be146615654417fea4f73f6bf`.
- The pipeline identity initially exposed an output-root dependency during the
  first independent rerun. Hashing the opaque notice-materialization identity
  instead of its path-bearing report fixed that defect. Three subsequent exact
  runs under distinct private roots produced stable pipeline ID
  `bg.cpp.browser-header-pack-pipeline.sha256.35e0d574519cf799844dfb3f72c2b6e6ae00532224ee34ddbcab2ba3a6e03556`
  in 23.12, 23.52, and 25.34 seconds while preserving all five pack and review-
  input hashes.
- The Node 25 fast gate passes 49 files/434 tests in 37.26 seconds including a
  25.59-second Vitest phase. The broader native/sanitizer gate passes 44 files
  and 239 tests with 9 intentional skips in 106.12 seconds. Focused typecheck,
  oxlint, five files/10 tests, whitespace, build-lock/Worker authoring, and
  architecture all pass; architecture reports zero cycles, boundary leaks, or
  coverage failures. External CUDA/Linux/per-file review, distribution
  approval, signed provenance, browser mounting, and release remain false.

### 2026-07-19 — Two-root header-distribution reproducibility

- `e2fdbaeb` makes the pipeline result a live opaque authority and adds one
  parser-issued two-run input. Both runs must share byte-identical archive,
  CUDA-index, and host-tool inputs while all four source/pack roots use
  canonical absolute spellings and remain pairwise distinct and non-overlapping.
  Forged or copied inputs and pipeline reports fail before they can mint
  reproducibility.
- The shared output verifier now checks the exact file and directory tree both
  before and after hashing, hashes every non-writable single-link current-user
  file through no-follow handles, and terminally rechecks the same inode
  identities. Materialization uses the same stronger boundary for its initial
  and final tree, eliminating a post-hash replacement interval from the harness.
- The comparator concurrently reverifies both completed roots and compares 17
  path/hash/length records totaling 68,899,643 bytes. Output-verification ID is
  `bg.cpp.distribution-output-file-verification.sha256.6eb7779eee72c9f65589656ac28b02f10caa9d9362a98d1d981f198de13a3937`;
  header-distribution reproducibility ID is
  `bg.cpp.browser-header-distribution-reproducibility.sha256.986bcd7b462b1bac0653f33e61dba69403295fd69df4fdb1780bb27092de9337`.
  Two independent direct commands completed in 44.45 and 46.84 seconds with the
  same IDs; the package entrypoint including one build completed in 56.51
  seconds.
- The Node 25 fast gate passes 50 files/437 tests in 35.32 seconds, including a
  24.55-second Vitest phase. Focused typecheck, oxlint, five files/9 tests,
  whitespace, lock/Worker authoring, and architecture pass with zero cycles,
  leaks, or coverage failures. The authority deliberately reports
  `fullDistributedOutputSetReproducible: false`: external legal review,
  Wasm/factory/Worker integration into one release set, detached provenance,
  browser execution, and release remain unproved.

### 2026-07-19 — Package-pinned header reproducibility evidence

- `7ba2d32e` records the path-independent result from the exact two-root run as
  one immutable package resource. It contains the current build-input lock,
  pipeline, 17 sorted output path/hash/length records, output-verification ID,
  reproducibility ID, and the original verifier source revision, but excludes
  both private host roots.
- Admission accepts only the exact 4,042 resource bytes at SHA-256
  `eb85b8fb54d1bad1b932bd4bfb3ffc7aff8d66d6185c9926ddd3c9f6084d918a`.
  It snapshots unshared input, verifies the current package build lock and
  resource digest, validates path/hash/length ordering and the 68,899,643-byte
  total, then independently rederives both domain-separated identities before
  issuing one opaque authority. Copied objects, mutated/truncated bytes, shared
  buffers, and non-byte inputs cannot mint authority.
- The Node 25 fast gate passes 51 files/440 tests in 37.19 seconds, including a
  26.22-second Vitest phase. Strict TypeScript, focused oxlint, whitespace,
  lock/Worker authoring, and architecture all pass with zero cycles, leaks, or
  coverage failures.
- This converts the local reproducibility observation into package-consumable
  technical evidence only. External license review, distribution approval,
  signed provenance, the complete release-output set, valid Worker execution,
  and release authority remain false.

### 2026-07-19 — Parallel JavaScript and compiler verification

- `6b8c6f05` moves the complete JavaScript build-plan test suite into one
  independent workflow job that begins concurrently with the Clang-Wasm build
  matrix. Every compiler job still installs and builds the exact package
  closure before staging its immutable runtime workspace; only the duplicated
  serial test phase moved.
- Reproducibility verification now has an explicit dependency on both the
  JavaScript verification job and the two clean build jobs. A failure in either
  branch still fails the workflow, and neither branch can grant the other's
  authority.
- The prior reproducibility run spent 86 and 134 seconds in the duplicated
  pre-build JavaScript phase. The new dependency graph removes that time from
  both compiler critical paths by construction. A fresh remote run is still
  required to record end-to-end timing; no unobserved speedup is claimed.
- The Node 25 local gate at that commit passed 51 files/441 tests in 34.96
  seconds, including a 24.48-second Vitest phase. Workflow structure tests
  prove the expensive jobs do not depend on the verification job while the
  final comparator depends on both.

### 2026-07-19 — Mode-scoped build concurrency

- `bd3935b8` replaces the ref-wide non-cancelling workflow group with an exact
  ref-plus-mode group. Fast validation, clean validation, and reproducibility
  may now occupy independent hosted runners instead of queueing behind one
  another.
- A new fast-validation dispatch cancels only a superseded fast-validation run.
  Clean and reproducibility runs remain non-cancelling, so incomplete evidence
  can never be mistaken for a completed observation. Every run still owns
  isolated acquisition/build roots and content-addressed cache admission.
- The Node 25 local gate passes 51 files/442 tests in 33.89 seconds, including a
  23.41-second Vitest phase. YAML parsing, focused workflow structure coverage,
  TypeScript, script lint, build-lock/Worker authoring, and the full fast gate
  are green.

### 2026-07-19 — Parallel fast-lane proof and stdout harness repair

- Remote run `29694786607` proved the new jobs started together. Its cached
  compiler branch succeeded in 4 minutes 38 seconds with a 3-minute-11-second
  locked executor even after the independent JavaScript branch failed in 53
  seconds. That failure was useful evidence, not a compiler-build failure.
- `ac371e19` removed shell `printf` as binary fixture transport and added a
  256-KiB backpressure success case. Run `29695115301` still failed only the
  small fast-exiting case while the larger case passed, isolating the defect to
  the production stream-start boundary rather than tar generation or strict
  parsing.
- `85e8b20f` attaches a roughly one-MiB bounded `PassThrough` immediately,
  transfers child stdout into it under pipeline backpressure, and independently
  awaits transfer, consumer, child-close, and bounded-stderr settlement.
  Consumer rejection destroys the bridge, terminates the detached process
  group, and retains identity-scoped output cleanup. A malformed one-block tar
  regression proves rejection settles without weakening the parser.
- Exact-source Linux/Node-24 run `29695343749` is green. The JavaScript branch
  passed in 56 seconds, the locked cached build took 3 minutes 15 seconds, and
  the complete compiler job took 4 minutes 30 seconds. The local Node-25 gate
  passes 51 files/444 tests with a 26.97-second Vitest phase.
- Cache-free clean/repro evidence still recompiles the pinned LLVM/Clang graph.
  The current build already uses four parallel compile jobs on a four-vCPU
  runner, and the two reproducibility ordinals already run on separate runners.
  A larger pinned runner is the next low-risk cold-build lever; deleting the
  narrow LibTooling closure is not an order-of-magnitude substitute.

### 2026-07-19 — Bounded harness parallelism and retained authority

- `f758eecc` retains the exact observed-execution, invocation, request,
  profile, asset-manifest, VFS, runtime-ABI, and package-Worker chain behind a
  validated browser result. Validation remains caller-frame consistency only;
  it does not authorize lowering.
- `66bf2b28` parallelizes independent fast-harness files while preserving
  per-file isolation. `5b6c5618` splits source and distribution real-world
  corpus lanes; `4f6563c0` isolates the required-native harness so those
  independent surfaces no longer serialize one another.
- `bdecb04d` preserves all 17 comprehensive compiler commands behind three
  serial prerequisites, then schedules four bounded fail-fast lanes with
  process-group termination and per-lane serialization. The complete local
  command passed in 126.39 seconds. Exact-source CI `29697264202` passed every
  shard in about 4 minutes 15 seconds; pre-sharding run `29695555899` took
  about 6 minutes 14 seconds.
- Build-producing commands are intentionally not run concurrently against one
  package directory. A verification attempt that overlapped the package-clean
  fast build with a unit suite made the unit suite lose `dist/index.js`; the
  isolated rerun passed all 80 files/1,500 tests. This is retained as evidence
  for output ownership, not a product failure.

### 2026-07-19 — Browser artifact and build-signature boundaries

- `68a1775e` defines strict canonical DSSE/in-toto build-subject syntax but
  grants no signature, producer, asset, legal, execution, or release authority.
- `912195fa` hardens Artifact V3 view-copy verification with bounded
  non-recursive, target-intrinsic cycle detection and corrects build-subject
  versus provenance terminology. The exact package Worker was regenerated
  without a Clang-Wasm rebuild.
- `332ddf7a` advances the asset manifest to v1.4, binds its build-signature
  policy into asset-set identity, and verifies one P-256 signature against the
  exact profile, manifest, build lock, Worker/factory, and cycle-free subject.
  The result is deliberately named a build-signature binding and returns
  `manifestSignaturePolicyMatched=true` with `producerTrusted=false`. Direct
  crypto inputs are intrinsically bounded before decoding or copying.
- Local proof passed compiler typecheck, strict focused lint, package-Worker
  byte verification, architecture checks, a focused fast-harness Vitest phase
  of 56 files/509 tests in 11.39 seconds, and 80 compiler files/1,500 tests in
  11.27 seconds after build. The
  signing fixture uses an ephemeral synthetic private key and proves verifier
  behavior only.
- Exact-source run `29698350889` passed Node 20/24/25, required-native,
  source/dist real-world corpus, real Chromium/WebGPU, and Pyodide lanes in
  about 4 minutes 12 seconds.

### 2026-07-19 — Controller result-authority authentication

- `32fd4c47` makes the production controller re-unwrap the exact strict
  protocol-issued validated result frame immediately before it mints browser
  execution evidence. It cross-binds the validation, invocation, request,
  profile, request-binding, artifact identity, artifact bytes, and outcome.
- A structural copy of an otherwise matching validated frame now fails closed
  at `$.validatedResultFrame`. The change creates no browser-to-common-lowering
  transition and does not change the package Worker graph.
- Node 25 proof passed compiler typecheck, focused strict lint, 3 focused files
  and 36 tests, the 56-file/510-test fast harness in 22.48 seconds wall time
  with an 11.19-second Vitest phase, and the complete 80-file/1,501-test
  compiler suite in 12.51 seconds after build. Architecture and exact Worker
  bundle checks remained green; bundle SHA-256 is
  `d9bd0eea4b9084eb7dd0768b35fadd0f14667b3a09dd4662a3bc052fb331c4e9`.
- Independent adversarial review confirmed that the earlier draft lowering
  transition was unsafe because caller-prepared profiles, manifests, and Wasm
  do not establish a trusted compiler producer. That draft was removed before
  commit. Producer trust still requires an independently admitted package
  policy and an externally issued exact-build statement.

### 2026-07-20 — Verifier evidence and exact corpus feedback

- `5db4878a` mints raw-Wasm conformance only after the package-owned verifier
  Worker completes its host-observed lifecycle and the exact report/cleanup
  contract passes. Caller bytes or effects cannot mint that authority.
- `1288b034` binds a bounded canonical derivative of that exact authority into
  the compiler-Worker invocation and transfer. The receiving Worker can
  reconstruct only a non-production evidence-region binding; the host
  controller rechecks the original retained verifier authority before issuing
  compiler-Worker execution evidence. Compiler execution, lowering, backend
  authorization, producer trust, legal approval, and release remain false.
- `b08429ae` through `1669d1a8` make exact corpus provisioning explicit,
  regression-gated, gitlink-aware, dead-lease recoverable, cached without a
  fetch, and independent of fixed `/usr/bin` tool paths. The host-toolchain
  capability probes the selected Git/Python and required filesystem primitives
  once and fails closed.
- `6c671559` runs the four independent corpus audits concurrently, retains all
  existing browser flags and thresholds, bounds captured subprocess output,
  and uploads versioned timing evidence.
- Local verifier/Worker proof passed 9 focused files/94 tests plus typecheck,
  lint, exact bundle regeneration/checking, and 65 fast-harness files/597 tests
  in 10.92 seconds. The compiler Worker is 559,512 bytes at SHA-256
  `3fdc7d9a82fd91fa9eb61b0ac0b07fa95aed41cb89607a9cc8212e748c93468a`;
  the verifier Worker remains 151,555 bytes at SHA-256
  `df514398f671b2124d0df58babc49f03ed86d70504227d33acac1e4802abbb65`.
- Main CI `29768391553` passed all eight jobs in 4 minutes 4 seconds. Source
  and distribution workspace builds took 16/18 seconds, provisioning took
  14/15 seconds, parallel audit groups were bounded by 33.55/35.71 seconds,
  browser phases took 111.91/118.20 seconds, and complete real-world gates
  took 145.48/153.92 seconds.
- `9e20a610` prepares an accepted layout from the exact retained
  Worker/verifier/frame/artifact lineage through the shared Gate 2 semantic
  seam. The opaque result remains an observed candidate with producer trust,
  lowering authorization, backend execution, and release readiness false.
- `3b400a86` keeps each source/dist job single-owner but runs two bounded browser
  child shards after the four parallel audits. Exact-source CI `29769844668`
  passed complete 159-case coverage as 80/79 in each bundle. Source shards took
  96.57/96.69 seconds and distribution shards 96.48/97.10 seconds concurrently;
  complete verifier times were 131.15/132.17 seconds. Aggregation rejects
  missing, duplicate, failed, skipped, or unexpected outcomes.
- `6e4901ca` reclaims only canonical self-owned interrupted corpus snapshots or
  reservations for the same checkout target while holding its lease. Each
  operation inspects at most 32 candidates, removes at most four, and traverses
  at most 4,096 entries per candidate. Descriptor-relative no-follow traversal,
  exact snapshot Git-blob identities, and UID/root/target/process binding retain
  ambiguous state. A new test-scope rule ensures provisioning, ownership, and
  reclamation changes select the focused regression gate automatically.
- Current Node 25 proof passes the 66-file/600-test fast Vitest phase in 10.62
  seconds and the full fast command in 21.80 seconds. The complete post-build
  compiler suite passes 88 files/1,559 tests in 9.81 seconds. Corpus
  provisioning/reclamation and scope tests, syntax, oxlint, and whitespace
  checks pass locally. Exact-source CI `29802518928` passed all eight jobs in
  about 3 minutes 46 seconds, including both Linux source/dist corpus gates,
  Node 20/24/25, required-native checks, Chromium/WebGPU, and Pyodide.

### 2026-07-21 — Independent producer trust and browser layout authorization

- `22522508` adds a closed canonical host-only trust-policy authority for the
  exact predicate, trust-store digest, builder IDs, key IDs, and policy version.
  It composes only with the existing exact opaque signature binding and does
  not mutate that binding's `producerTrusted=false` claim. The resulting
  producer authority still keeps exact asset bytes, complete distributed-output
  reproducibility, legal/distribution approval, Worker execution, lowering,
  backend execution, and release false.
- `c806214c` cross-binds that exact producer to one exact observed Worker layout
  candidate and enters the existing producer-neutral frontend authorization and
  Gate 2 layout-lowering seam. Profile, manifest, asset set, package Worker,
  invocation, request, request binding, artifact, selected entry, and semantic
  layout identity must all agree; structural copies fail closed. Backend,
  distribution, and release authority remain false.
- Non-authoritative layout preparation moved into
  `cpp_cute_layout_semantics.ts`, leaving one canonical lowering path and zero
  compiler dependency cycles. Authorization options use captured intrinsics,
  reject accessors and hostile inspection without invoking caller code, accept
  immutable plain records, and check cancellation before minting.
- Local Node 25 proof passes 4 focused files/22 tests, the 69-file/614-test fast
  Vitest phase in 10.58 seconds, the complete 91-file/1,573-test compiler suite
  in 10.82 seconds, compiler typecheck, focused strict lint, exact build-lock and
  Worker-bundle checks, architecture with zero cycles/leaks, and whitespace
  checks. CI runs `29803438651` for producer trust and `29804453681` for
  browser layout authorization each passed all eight jobs.
- The trust tests use ephemeral synthetic key and policy material. No
  package-controlled production policy, externally issued exact-build
  statement, valid production compiler-Worker compile, browser-local C++, CPU
  convergence, real WebGPU execution, licensed distribution, or release
  authority is claimed.
- Clean-run attribution remains separate from the seconds-scale feedback loop.
  Successful run `29681845216` spent about 40 minutes 6 seconds of its
  45-minute-37-second locked build in four-way Wasm compile/link; acquisition
  took about 37 seconds. The dependency-bound native TableGen and Wasm stages
  cannot be naively overlapped. The next safe experiments are a provisioned
  larger pinned runner with repinned parallelism, then a pinned Ninja graph.

### 2026-07-21 — Fast-gate closure and authorized CuTe view-copy lowering

- `2093faa4` replaces the manually maintained browser-test allowlist with one
  bounded `cpp_cute_browser_*` family glob. Seven previously omitted files and
  63 tests entered the fast gate without increasing the measured Vitest phase:
  76 files/677 tests passed in 10.60 seconds versus the prior 69 files/614 tests
  in 10.55 to 10.58 seconds. Exact-source CI `29805119479` passed all eight
  jobs.
- `85631464` adds the producer-neutral authorized C++/CuTe `view-copy`
  transition. It accepts no semantic IDs or implicit storage authority from the
  caller: only the selected entry plus explicit source/destination allocation
  lengths and byte offsets. The pass reuses the canonical static affine-layout
  conversion, independently checks positive rank-2 address spans, and calls
  semantic-core's sole verified view-copy constructor.
- The initial profile is exact f32/32-bit device ABI, distinct non-null global
  parameter pointers, source-only const qualification, synchronous portable
  32-bit copy, exact read/write effects, reject-invalid-source, and
  forbid-overlap. The pinned semantic hashes are `5ade6e06...` for layout and
  `64dc9d67...` for kernel. The canonical CPU reference copies the `(3,2)`
  transpose bit-for-bit with nonzero offsets while preserving root canaries.
- Focused provenance/artifact/layout/view-copy proof passes 4 files/53 tests.
  The fast gate also discovers both C++/CuTe lowering families and passes 78
  files/692 tests in 11.67 seconds; the entire fast command completes in 22.78
  seconds. The complete compiler suite passes 92 files/1,578 tests in 11.16
  seconds. Build, typecheck, focused lint, architecture, and whitespace checks
  pass. Exact-source CI `29806349465` passed all eight jobs.
- Independent test commands remain parallelizable, but package `build` and the
  browser fast gate must be dependency-ordered in one checkout: `build`
  intentionally removes `dist/`, while build-script tests import exact files
  from `dist/`. A deliberately concurrent local verification exposed that
  shared-output race; rerunning the canonical sequential fast command passed.
- This capability uses synthetic AOT provenance only. No observed browser
  view-copy candidate, browser producer composition, real compiler-Worker
  compile, real WebGPU execution, licensed distribution, or release authority
  is claimed.

### 2026-07-21 — Observed Worker view-copy authorization

- `a8a861e2` prepares one opaque observed Worker `view-copy` candidate from the
  exact retained Worker/frame/profile/request-binding/Wasm-conformance lineage,
  then composes only that candidate and the independently admitted exact
  producer into the existing canonical authorized frontend artifact.
- Candidate and authorization identities bind the prepared semantic subject but
  deliberately exclude source/destination allocation sizes and byte offsets.
  Those explicit storage facts remain a later lowering input, so two storage
  bindings can share one authorization identity while producing distinct final
  canonical semantic hashes.
- Backend execution, distribution, and release authority remain false. The
  browser source is a synthetic authority fixture rather than a real production
  Worker compile, and no real-WebGPU evidence is claimed.
- The Node 25 fast harness passes 80 files/700 tests in 11.54 seconds. Focused
  candidate/authorization/lowering tests, typecheck, lint, build, architecture,
  canonical fast checks, and whitespace checks pass. Exact-source CI
  `29807986545` passed all eight jobs.
- At that checkpoint, the next software slice was required real-WebGPU convergence using the exact
  canonical authorized payload. It must not reconstruct layouts, offsets, or
  effects independently in the backend or upgrade distribution/release
  authority.

### 2026-07-21 — Required CuTe view-copy WebGPU convergence

- `c4e2d799` pins one realm-neutral canonical rank-2 fixture reproduced exactly
  by the Node authorization-to-lowering chain, then executes the same layout
  and kernel artifacts through the CPU reference and shared WebGPU view-copy
  backend. The browser evidence binds semantic, input, prepared-backend, WGSL,
  specialization, device-profile, and complete-destination hashes and verifies
  both nonzero-offset canaries.
- `08f4b102` makes success terminal: device acquisition, preparation,
  execution, and drains are bounded; device loss and late uncaptured errors
  race every post-acquisition stage; a final queue drain plus macrotask occurs
  before pass emission; and a dedicated validator rejects contradictory or
  incomplete retained evidence.
- CI initially exposed a harness defect rather than a compiler defect. Four
  Vitest configs placed Chromium arguments on an ignored instance `launch`
  field, so advisory suites silently skipped for lack of an adapter and the
  required suite failed. `6355289c` moves those arguments to the Playwright
  provider's `launchOptions`, after which CI acquired SwiftShader and executed
  207 compiler WebGPU tests before exposing two ungated `shader-f16` cases and
  one Metal-calibrated accumulated-math tolerance.
- `e546a124` gates the two f16-required cases on the adapter capability and
  uses a documented `5e-4` absolute bound for the fixture's 17 accumulated
  transcendental/arithmetic operations. Capable devices still execute the f16
  cases; unsupported devices no longer fail for a feature they do not expose.
- Local Apple Metal 3 passes 6 files/210 browser tests and all four affected
  package browser suites. Exact-source CI `29810488096` passes all eight jobs;
  its required SwiftShader record has `actualWebGpuExecution=true`, exact CPU
  and GPU specialization equality, complete destination equality, preserved
  canaries, and `productionBrowserCompileObserved=false`.
- This closes exact canonical fixture-payload CPU/WebGPU convergence only. It
  does not prove a valid production compiler-Worker compile and does not mint
  backend, distribution, or release authority. At that checkpoint, the next
  software capability was static positive-affine rank-3 view-copy lowering
  through the same seam.

### 2026-07-21 — Static rank-3 CuTe view-copy convergence

- `dca33be5` extends the same producer-neutral CuTe view-copy semantic seam from
  rank 2 to equal-rank static positive-affine rank 2 or rank 3. It replaces the
  rank-2-only address-span projection with one bounded rank-aware projection;
  dynamic or non-positive layouts, rank mismatch, and rank 4 still fail closed.
- The pinned rank-3 case has logical shape `(2,3,4)`, source strides `(1,2,6)`,
  destination strides `(12,4,1)`, explicit 104-byte source/destination
  allocations, and 4-byte view offsets. The CPU reference copies all 24 f32
  words bit-for-bit and preserves both root canaries. Its layout hash is
  `c2b5e8a0489bd2ee5a54d15399af95b91d9fe102aab63e450361500ffa946a6f`;
  its kernel hash is
  `e335ea9d9e9a38f591c80c737b8a33401578739e02d6892e5f1907e6b76e6ff2`.
- Candidate preparation, producer authorization, canonical lowering, fixture
  validation, and terminal browser evidence now cover both pinned ranks. The
  required actual-WebGPU lane executes two cases and requires exact CPU/GPU
  specialization equality, complete destination equality, and preserved
  nonzero-offset canaries.
- The complete compiler suite passes 95 files/1,591 tests. Exact-source CI run
  `29811673981` completed with all eight jobs green, including the two-case
  required actual-WebGPU lane.
- Authority remains deliberately narrow: the browser authority is synthetic,
  every terminal record retains `productionBrowserCompileObserved=false`, and
  no backend, distribution, release, or valid production compiler-Worker
  authority is minted. With the bounded rank-2/rank-3 view-copy software slice
  complete, the next software capability is Gate 4's frontend-neutral logical
  GEMM tile, represented independently from any physical schedule or backend
  mapping.

### 2026-07-21 — Gate 4 tiled GEMM and schedule separation

- `2cb7cea3` and `6e4e05b1` add the canonical backend-neutral logical-GEMM
  schedule artifact and its separate bounded specialization. Workgroup mapping,
  cooperative staging, two acquire-release workgroup barriers, active boundary
  participation, scalar vectors, and zero-fill/suppressed-store masks are
  derived from the requested physical tile and then strictly reverified. The
  logical kernel artifact retains no backend, WGSL, workgroup, subgroup,
  vectorization, staging, or physical-schedule fields.
- `8a54547d` proves that packed consumers receive the same schedule seam.
  `69031da9` adds an exact-input certificate that copies and retains caller
  bytes, binds logical and schedule-specialization hashes, rejects hostile or
  excessive inputs before authority, and admits only finite nonnegative
  integral f32 values whose products and subset sums are exactly representable.
  `b135c6a6` removes the last backend-shaped name from that shared certificate.
- `20bd9d3a` admits one closed typed compiler GEMM fact and lowers it through
  the canonical semantic-core constructor. `b97a9f27` proves that this path and
  direct semantic-core construction have identical logical and kernel hashes.
  This is typed-artifact convergence only: production source extraction and
  browser-local compilation remain Gate 3 blockers.
- `a77c1ed4` executes the exact prepared specialization and certificate-retained
  snapshots through a portable WGSL realization. Every invocation participates
  in both uniform barriers; out-of-range loads write zero into workgroup
  storage; destination effects are masked; accumulation follows increasing K.
  Caller mutation after certification cannot change uploaded operands.
- The required browser evidence runs M=17, K=23, N=19 under independent 8x8x8
  and 16x16x16 schedules. It compares complete destination bytes with one CPU
  result and across schedules and retains semantic, schedule, specialization,
  WGSL, certificate, environment, device-limit, and comparison-policy hashes.
  The scalar 4-byte profile exercises row pitches of 92 and 76 bytes, which are
  not aligned to wider vector/tile boundaries. `5af2a4d4` names the execution
  and preservation tiers; `b77d626b` isolates the evidence into a required CI
  job that fails when WebGPU is unavailable.
- `fe5581d3` makes the exact semantic-GEMM evidence commit a kernels publish
  prerequisite and orders both release workflows to produce and retain it
  before staging. Semantic-core passes 16 files/138 tests; kernels passes 18
  files/147 tests and the 20 focused semantic-GEMM tests; the compiler passes
  96 files/1,602 tests; packed-release checks pass. CI `29818182317` completed
  its dedicated required semantic-GEMM WebGPU job successfully.
- That run's required-native job correctly rejected the stale package Worker
  pin left by the typed artifact parser's larger module graph. `343523fe`
  regenerates the bundle twice, repins its exact 571,098-byte SHA-256 identity
  `01a4c1d10d606773bfa241284160f3af787dec856e1a17e22edd7c34dae043a3`,
  and keeps both static and dynamic import counts at zero. The exact bundle
  check, authoring test, architecture check, and complete compiler verifier pass
  locally. The failed superseded run grants no whole-CI success authority.
- Gate 4 is verified only for the named dense row-major, 4-byte-aligned,
  certified exact-input f32 portable scalar profile. Evidence reports
  `portable-webgpu-core`, `portable-relegalized`, and
  `bit-exact-on-certified-inputs`. It does not prove general f32, other dtypes
  or layouts, nonzero base offsets, resident-buffer provenance, vectorized or
  native MMA, preserved CUDA/CuTe schedules, source compatibility,
  distribution, or release authority.

### 2026-07-21 — Gate 5 attention-forward semantic baseline

- Semantic-core now defines one closed, backend-neutral
  `browsergrad.kernel.attention-forward@1` artifact over an exact verified
  layout hash. The canonical constructor derives four disjoint dense rank-4
  f32 Q/K/V/output views, upper-left causal or non-causal mask meaning, and the
  exact f32 `1/sqrt(queryDepth)` scale.
- The verifier independently rederives the scale from the verified query-depth
  dimension, rejects arbitrary positive replacement bits, limits all dimensions
  to the portable u32 profile and query/value depths to 256, and rejects
  schedule/backend fields. It binds stable-softmax phases, finite input/score/
  online-state requirements, a frozen `1e-4` absolute-or-relative comparison
  policy, pairwise-disjoint effects, and explicit forward-only VJP refusal.
- The semantic-core package passes 17 files/147 tests, build, typecheck, lint,
  architecture, canonical-byte round-trip, hostile mutation tests, and the
  packed-tarball consumer gate on Node 25.9.0. This slice adds no CPU executor,
  schedule, WGSL, frontend convergence, device evidence, performance record,
  or FlashAttention claim.
- Commit `4cfacd75` passed all CI jobs in run `29820268289`.
- A second capability adds the separate closed
  `browsergrad.schedule.attention-online-kv-tile@1` artifact. Two physical
  schedules (8x8 and 8x16 query/key rows) bind the same logical hash while
  retaining distinct schedule hashes. Both causal and non-causal logical
  artifacts accept the same physical recurrence but remain distinct through
  their referenced semantic hash.
- The schedule fixes increasing tile and within-tile key traversal,
  cooperative single-buffered K/V staging, a tile-wise running maximum,
  denominator, and weighted-value recurrence, two all-invocation uniform
  barriers per key tile, and scalar memory vectors. Invalid tail keys and
  logical-mask keys are excluded before the tile maximum or any online-state
  update; zero-filled staging is only a memory-safety action.
- Semantic-core now passes 18 files/155 tests plus build, typecheck, lint,
  architecture, and packed-release consumption. Gate 5 remains `partial`:
  this schedule grants no target legality, CPU or WebGPU execution,
  preservation, performance, frontend convergence, or named fused-attention
  claim. The next capability is the independent CPU reference and numerical
  conformance contract.
- The schedule-independent CPU capability prepares the exact logical layout and
  kernel without accepting a schedule. It proves the dense rank-4 row-major
  addresses once and binds layout, kernel, bindings, dimensions, allocation
  offsets/sizes, and bounded work into the specialization hash.
- Execution admits only exact enumerable Q/K/V/destination data properties and
  direct fixed unshared non-overlapping `Uint8Array` ranges with exact lengths
  and alignment. Q/K/V bytes are copied privately before the first yield;
  destination identity is rechecked and no write occurs until every row has
  completed successfully. GEMM CPU execution now consumes the same native
  binding helper instead of a separate parser.
- The oracle rejects non-finite inputs, scaled scores, exponentials,
  denominators, or outputs; implements upper-left causal exclusion before the
  score set; rounds products, reductions, scale, exponential results,
  probabilities, and weighted-value accumulation to f32; and records exact
  valid-score/multiply-add/byte counts. The comparator applies the named
  `1e-4` absolute-or-relative rule, ignores signed-zero differences, and rejects
  non-finite values.
- Semantic-core now passes 19 files/164 tests plus build, typecheck, lint, and
  architecture checks. Focused evidence covers composed-reference agreement,
  causal rectangular rows, hostile bindings, shared/subclass/wrong-length/
  overlap rejection, finite-domain atomicity, comparison boundaries, resource
  limits, cancellation, module-captured intrinsic resistance, and unchanged
  GEMM behavior. Packed-release execution is also required. Gate 5 remains
  `partial`; the next capability is schedule
  specialization and block-tiled WebGPU lowering.
- Schedule specialization now accepts only the module-authorized logical proof,
  its exact verified kernel, and a schedule bound to that kernel hash. It
  derives query/key tile sizes, workgroup invocations, K/V staging elements and
  bytes, query/output private elements, key-tile count, and x/y/z dispatch
  workgroups, then binds the resolved projection into a distinct specialization
  hash.
- Independent 8x8 and 8x16 schedules reuse the same logical specialization but
  produce 896/1,792 staging bytes, three/two key tiles, and distinct schedule
  specialization hashes for the `(B=2,H=3,Sq=17,Sk=23,D=16,Dv=12)` test.
  Copied logical objects, cross-kernel schedules, excess workgroup size,
  staging, private state, dispatch, and key-tile count fail before authority.
- Semantic-core now passes 19 files/166 tests and packed consumers exercise the
  same geometry seam. This still grants no target legality, WebGPU execution,
  numerical preservation, performance, or named fused-attention claim. The
  next capability is block-tiled K/V-staged WebGPU lowering.
- Kernels now prepares `browsergrad.webgpu.attention.block-tiled-online-softmax-f32@1`
  from the exact logical and schedule authorities. Generated WGSL uses one
  query row per invocation, private Q/output rows, cooperative zero-filled K/V
  staging, two uniform barriers per key tile, increasing-key tile recurrence,
  and suppressed boundary stores. Causal and invalid-tail keys are excluded
  before the tile maximum or any running-state update.
- Backend preparation hashes the semantic and schedule specializations, exact
  WGSL, algorithm profile, and workgroup storage. It bounds WGSL bytes,
  workgroup invocations/storage, private elements, key-tile count, dispatch
  workgroups, and aggregate transient host/GPU bytes before returning an
  immutable plan. Closed request capture rejects accessors, prototypes,
  symbols, and unknown backend fields without invoking them.
- Kernels passes 19 files/151 tests plus build, typecheck, lint, architecture,
  and packed-release consumption. This code-generation slice grants no actual
  WebGPU execution, numerical preservation, performance, resident-buffer,
  FlashAttention-v2, or frontend claim. The next capability is finite input
  snapshot/upload and device execution through the exact prepared plan.
- Host execution now rejects copied preparations before input/device effects,
  captures Q/K/V as exact enumerable direct `Uint8Array` data properties,
  requires fixed unshared non-overlapping aligned allocations, copies every
  input before the first yield, and validates every f32 value as finite under
  native abort/time bounds. Caller mutation cannot change submitted bytes.
- The device boundary admits allocation/binding/workgroup/staging/dispatch
  limits before pipeline work, enforces one semantic-attention operation in
  flight per device, drains pipeline and dispatch validation/OOM/internal
  scopes, races device loss/cancellation/timeout, destroys the prepared
  sequence on every terminal path, and publishes only a complete finite
  destination. Release and publish gates require exact-commit attention
  evidence separately from view-copy, GEMM, and JIT evidence.
- Required headed Chromium on Apple Metal 3 executes causal and non-causal
  `(B=1,H=2,Sq=9,Sk=11,D=4,Dv=6)` inputs under 8x8 and 8x16 schedules. All four
  complete outputs pass the semantic-core `1e-4` absolute-or-relative CPU
  comparator; each second schedule also passes a same-mask complete-output
  comparison. Kernels now passes 19 files/154 tests, Node/browser typecheck,
  build, lint, architecture, packed-release verification, and the required
  device lane. No FlashAttention-v2 or global numerical claim is made.
- A separate required performance lane measures the production block-tiled
  host API and frozen row-wise baseline on
  `(B=1,H=2,Sq=256,Sk=256,D=Dv=32)`. It performs an untimed CPU-correctness
  preflight, 16 warmups, and 20 alternating paired `performance.now` samples,
  each ending after complete output readback and queue drain. The terminal
  record retains raw samples, named device/browser/configuration, validation
  and buffer-lifecycle differences, and an explicit no-superiority/no-threshold
  claim. Correctness remains in its own required lane.
- Shared WebGPU evidence helpers now own the environment record, semantic
  device-limit projection, bounded timeout, and next-task drain used by GEMM
  and attention correctness/performance lanes. CI runs correctness and
  performance in separate parallel jobs; release/publish staging requires both
  exact-commit markers. This closes the initial f32 Gate 5 profile.

### 2026-07-22 — Gate 6 typed `Tensor.expand` migration

- Public `TensorProxy.expand` now parses dimensions through the integer-index
  protocol, rejects booleans and float truncation, permits `-1` only on an
  existing aligned dimension, checks rank and every non-singleton extent, and
  emits `BROADCAST_TO` with a closed resolved-shape argument.
- The shared typed validator checks exact arity, plain/closed arguments,
  optional VJP metadata type, nonnegative integer extents, node/argument shape
  identity, dtype preservation, rank direction, and aligned broadcast
  compatibility. Construction, CPU realization, VJP, vmap, ONNX, and GPU-plan
  scheduling all invoke it, so post-construction argument mutation fails before
  a path can reinterpret the node.
- CPU realization returns a writable owning copy. Closure backward remains
  compatible. The new symbolic VJP reduces leading and expanded singleton
  dimensions. Vmap retains its outer batch axis, ONNX emits `Expand`, and the
  default tensor plan contains `BUFFER,LOAD,BROADCAST_TO` with no opaque op.
  Both materializing and resident production bridge routes execute the public
  surface in the integration harness; the resident root remains unmaterialized
  until `.numpy()`.
- Grad eager `Tensor.expand` now applies the same integer-index, `-1`, rank,
  and aligned-extent rules before NumPy execution and preserves the input
  dtype for float16 and integer tensors. A shared cross-package fixture proves
  identical valid values, dtype outcomes, and invalid-shape diagnostic classes
  in eager and lazy surfaces. Grad deliberately retains an owning contiguous
  result, so the frozen compatibility record still reports view-alias debt.
- ADR-0002 records the required exception to ADR-0001. The exact frozen opaque
  inventory is narrowed from 36 constructor calls/39 operations to 35/38;
  `tensor.expand`, `jit.custom.expand.v0`, and the `expand` callback fixture are
  removed. The architecture validator now derives the exact total from frozen
  per-file counts rather than embedding the historical number, while retaining
  exact site, label, operation, source-definition, fixture, and hash checks.
- Final verification passes JIT build, strict typecheck, lint, 4 unit files/24
  tests, and 24 integration files/238 tests in 40.67 seconds; Grad build,
  strict typecheck, lint, 2 unit files/30 tests, and 34 integration files/324
  tests in 55.50 seconds; the semantic architecture check; 19 hostile-archive
  and 35 Node release-security tests plus packed/fresh consumers; and a full
  seven-package workspace build in 8.31 seconds. One intermediate
  WebGPU-realizer assertion still expected the pre-resident generic-plan count;
  it failed 2-versus-1 and was corrected to account for the deliberately added
  resident execution.
- Gate 6 remains `in-progress`. This slice reuses earlier actual-device
  `BROADCAST_TO` primitive evidence but does not claim a new public-surface
  actual-device record or a generated support table. The next slice must make
  framework operation decisions one executable registry consumed by public
  support reporting before the remaining opaque families migrate through it.

### 2026-07-22 — Gate 6 executable framework-operation registry

- Added package-owned schema `browsergrad.jit.framework-operation-contracts`
  version 1.0. Its first record binds `Tensor.expand`, typed
  `BROADCAST_TO`, the retired `jit.custom.expand.v0` ID, and explicit CPU,
  closure-autograd, symbolic-VJP, functional-grad, vmap, ONNX, tensor-plan,
  WebGPU-profile, residency, and materialization decisions.
- The runtime loader caps registry bytes, requires immutable input, decodes
  strict UTF-8 JSON with duplicate-key rejection, enforces exact fields and
  closed enums, rejects bool-shaped versions and duplicate identities, and
  fails import unless every record has exactly one executable validator.
- UOp construction and all existing expand boundaries now resolve validation
  through the executable registry. Public `framework_operation_support()`
  projects the same immutable records into fresh deterministic dictionaries;
  hostile mutation of one returned table cannot change later calls.
- The architecture gate independently verifies the package registry, declared
  non-`CUSTOM` opcodes, executable binding strings, closed decision values,
  deterministic contract ordering, and the exact union of current opaque and
  retired typed IDs against the original 39-operation baseline.
- Verification passes JIT build, strict typecheck, lint, 4 unit files/24 tests,
  the registry/expand/WebGPU-realizer group at 3 files/42 tests, and 25 full
  integration files/240 tests in 42.15 seconds; the semantic architecture
  check; compiler strict typecheck/lint and 96 files/1,602 tests; 19
  hostile-archive and 35 Node release-security tests plus packed/fresh
  consumers; and a full seven-package workspace build in 8.15 seconds.
- Gate 6 remains `in-progress`. The next bounded migration is the pure unary
  elementwise family; each admitted typed operation must extend this registry
  and its public table in the same capability commit.

### 2026-07-22 — Gate 6 typed `Tensor.abs` and `Tensor.sign`

- Added typed `ABS` and `SIGN` opcodes and registry records. Their shared
  validator requires one input, a plain closed argument dictionary, exact
  shape/dtype preservation, and a real numeric dtype; bool fails during UOp
  construction instead of inheriting NumPy behavior.
- CPU handlers revalidate and return owning arrays cast to the declared dtype.
  Closure autograd retains the zero-at-origin abs subgradient and zero sign
  derivative. Symbolic VJP emits typed `SIGN` for abs and an explicit zero for
  sign, enabling functional `grad` without `CUSTOM`.
- Vmap preserves the leading batch axis and revalidates the unary contract.
  ONNX emits direct opset-17 `Abs`/`Sign` nodes for exporter-supported dtypes.
  Tensor-plan and WebGPU paths explicitly refuse both operations because this
  slice adds no portable plan lowering or kernel; the declared profile remains
  host-materialized.
- Post-construction mutation tests cover CPU, VJP, vmap, and ONNX boundaries.
  The public support table is generated from the three validator-bound records
  `ABS`, `BROADCAST_TO`, and `SIGN`, while detached-return mutation cannot alter
  later calls.
- ADR-0004 retires `tensor.abs`, `tensor.sign`, `jit.custom.abs.v0`, and
  `jit.custom.sign.v0`. The current opaque inventory narrows from 35
  constructor calls/38 operations to 33/36, while the exact original set
  remains partitioned across 36 opaque IDs and three typed registry records.
- Final verification passes JIT build, strict typecheck, lint, 4 unit files/24
  tests, the Gate0/unary/registry/VJP group at 4 files/25 tests, and 26 full
  integration files/243 tests in 43.19 seconds. The architecture check and its
  focused compiler suite pass 1 file/24 tests; the full compiler suite passes
  96 files/1,602 tests. Release verification passes 19 Python archive tests,
  35 Node security tests, and every packed/fresh consumer. The seven-package
  workspace build passes under Node 25.9.0 in 8.34 seconds.
- The first full JIT run exposed two stale expectations: the frozen opcode
  count still said 55, and the WebGPU backward refusal expected abs to remain a
  closure-only `CUSTOM` node. They now require 57 opcodes including `ABS` and
  `SIGN`, and the exact typed `SIGN` tensor-plan refusal reached through abs
  symbolic VJP. The focused rerun and full suite pass.
- Gate 6 remains `in-progress`. The next bounded migration is the coupled
  `Tensor.sin`/`Tensor.cos` pair so each symbolic derivative remains typed.

### 2026-07-22 — Gate 6 typed `Tensor.sin` and `Tensor.cos`

- Added typed `SIN` and `COS` opcodes and registry records under the shared
  typed-unary validator. The floating profile accepts float16/32/64 only,
  preserves exact shape/dtype, and rejects bool/integer inputs at construction.
  This removes the frozen defect where an integer UOp declared integer output
  while NumPy realized float64.
- CPU handlers revalidate and return owning arrays in the declared floating
  dtype. Closure autograd retains cosine and negative-sine derivatives.
  Symbolic VJP emits typed `COS` for sin and typed `SIN` plus primitive negate
  for cos, so functional `grad` contains no `CUSTOM` derivative.
- Vmap preserves a leading batch axis. ONNX emits direct opset-17 `Sin`/`Cos`
  for exporter-supported dtypes. Tensor-plan and WebGPU remain explicit
  refusals because no portable lowering or kernel is admitted; residency stays
  host-materialized.
- Contract tests cover all three floating dtypes, known-angle values and
  closure gradients, owning materialization, symbolic graphs, functional grad,
  vmap, ONNX, explicit plan refusals, early non-floating rejection, and
  post-construction mutation at CPU/VJP/vmap/ONNX boundaries.
- ADR-0005 retires `tensor.sin`, `tensor.cos`, `jit.custom.sin.v0`, and
  `jit.custom.cos.v0`. The current opaque inventory narrows from 33
  constructor calls/36 operations to 31/34; the executable registry contains
  five typed retirements and still partitions the exact original 39 IDs.
- Final verification passes JIT build, strict typecheck, lint, 4 unit files/24
  tests, the trig/support/VJP/IR group at 4 files/47 tests, the updated
  Gate0/trig/support/VJP group at 4 files/25 tests, and 27 full integration
  files/246 tests in 44.33 seconds. The architecture guard passes; the full
  compiler suite passes 96 files/1,602 tests. Release verification passes 19
  Python archive tests, 35 Node security tests, and every packed/fresh
  consumer. The seven-package workspace build passes under Node 25.9.0 in
  8.09 seconds.
- Gate 6 remains `in-progress`. The next bounded migration is `Tensor.clamp`,
  including exact optional-bound validation and its piecewise VJP contract.

### 2026-07-22 — Gate 6 typed `Tensor.clamp`

- Added typed `CLAMP` and its executable registry record. The public builder
  requires at least one finite ordered bound, accepts only exact built-in or
  supported NumPy real scalar types, rejects bool/complex/string/non-finite
  values, and never invokes arbitrary `__float__` conversion hooks. Admitted
  bounds are normalized once into a closed `{min, max}` record.
- The initial profile accepts float16/32/64 inputs and preserves exact shape
  and dtype. Construction, CPU, symbolic VJP, vmap, and ONNX revalidate arity,
  plain/closed arguments, optional finite-float bounds, ordering, shape, and
  dtype, so post-construction integer/open-field mutations fail locally.
- CPU uses NumPy clip and returns an owning declared-dtype array. Closure and
  symbolic gradients use the same inclusive-bound mask; symbolic VJP emits
  typed comparisons, bool mask composition, cast, and multiply, enabling
  functional `grad` without `CUSTOM`.
- Vmap preserves the leading batch axis. ONNX opset 17 emits `Clip` with typed
  one-element initializers and correct empty optional-input placement for
  min-only, max-only, and two-bound calls. Tensor-plan/WebGPU remain explicit
  refusals; `clip` and `clamp_min` reuse this same contract.
- Contract tests cover all floating dtypes, owning results, normalized NumPy
  scalar bounds, aliases, inclusive-bound closure/functional gradients, typed
  symbolic graph, vmap, every ONNX bound form, plan refusal, hostile scalar
  non-execution, early dtype/coercion failures, invalid contracts, and mutation
  at every consumer boundary.
- ADR-0006 retires `tensor.clamp` and `jit.custom.clamp.v0`. The current opaque
  inventory narrows from 31 constructor calls/34 operations to 30/33; the
  registry contains six typed retirements and preserves the original 39-ID
  partition.
- Final verification passes JIT build, strict typecheck, lint, 4 unit files/24
  tests, the Gate0/clamp/support/VJP/IR group at 5 files/52 tests, and 28 full
  integration files/249 tests in 46.27 seconds. The architecture guard passes;
  the full compiler suite passes 96 files/1,602 tests. Release verification
  passes 19 Python archive tests, 35 Node security tests, and every
  packed/fresh consumer. The seven-package workspace build passes under Node
  25.9.0 in 7.89 seconds.
- Gate 6 remains `in-progress`. The next bounded migration is materializing
  `Tensor.flip`, with normalized axes, involutive VJP, vmap, ONNX, and explicit
  negative-stride/device refusal decisions.

### 2026-07-22 — Gate 6 typed `Tensor.flip`

- Added typed `FLIP` and its executable registry record. The public builder
  accepts one exact built-in or fixed-width NumPy integer scalar, rejects bool,
  float, string, and arbitrary conversion hooks without invoking them,
  normalizes one negative axis, and rejects scalar-rank or out-of-range axes
  instead of wrapping modulo rank.
- The contract requires one input, a plain closed axis argument plus optional
  VJP provenance, exact shape/dtype preservation, and a normalized
  nonnegative axis inside the input rank. Construction, CPU, symbolic VJP,
  vmap, and ONNX boundaries all revalidate the contract, so argument mutation
  fails locally.
- CPU realization copies NumPy's reverse view into owning storage for every IR
  dtype. Closure and symbolic gradients apply the same involutive reversal;
  functional `grad` therefore contains typed `FLIP`. Vmap shifts the logical
  axis by one past the leading batch axis.
- ONNX opset 17 admits float32/int32/int64/bool graphs and emits `Slice` with
  exact signed-int64 start `-1`, end `INT64_MIN`, normalized axis, and step
  `-1`; other graph dtypes fail explicitly. The contract test decodes the
  protobuf and verifies initializer order, dtype, size, and signed values.
  Tensor-plan and WebGPU explicitly refuse the negative-stride profile, so the
  Gate 2 positive-stride contract is unchanged.
- ADR-0007 retires `tensor.flip` and `jit.custom.flip.v0`. The current opaque
  inventory narrows from 30 constructor calls/33 operations to 29/32; the
  executable registry contains seven typed retirements and preserves the
  original 39-ID partition.
- Final verification passes JIT build/codegen, strict typecheck, lint, 4 unit
  files/24 tests, the Gate0/flip/support/VJP/IR group at 5 files/52 tests, and
  29 full integration files/252 tests in 50.85 seconds. The architecture guard
  and whitespace check pass. The complete compiler suite passes 96 files/1,602
  tests. Release verification passes 19 Python archive tests, 35 Node security
  tests, and every packed/fresh consumer. The seven-package workspace build
  passes under Node 25.9.0 in 7.84 seconds.
- Gate 6 remains `in-progress`. The next bounded migration is
  `Tensor.repeat`, replacing the shared opaque-refusal fixture with typed tile
  multipliers, a reduction VJP, transform/export decisions, and an explicit
  device profile.

### 2026-07-22 — Gate 6 typed `Tensor.repeat` and eager conformance

- Added typed `REPEAT` and its executable registry record. JIT and Grad accept
  only exact built-in or fixed-width NumPy integer factors, never invoke
  arbitrary conversion hooks, require one through 32 repeat axes and at least
  the input rank, and bound every factor to `[0, 2^30]`. The independent
  ceilings reject resource-hostile empty-output requests that an element-count
  check alone would miss.
- The JIT contract requires one input, a plain closed repeats tuple plus
  optional VJP provenance, exact dtype preservation, left-one rank padding,
  and an output shape derived by axis-wise multiplication. Construction, CPU,
  VJP, vmap, and ONNX boundaries revalidate it, so post-construction mutation
  fails locally.
- CPU returns an owning dtype-preserving NumPy tile. Closure and symbolic VJP
  reshape upstream gradients into interleaved repeat/source axes, reduce every
  repeat axis, and remove leading rank padding. Functional tests cover ordinary,
  leading-rank, scalar, and zero-repeat gradients. Vmap prepends a unit factor
  and therefore never tiles its batch axis.
- ONNX opset 17 emits `Tile` with one exact signed-int64 repeat vector for
  float32/int32/int64/bool graphs and rejects other exporter dtypes. Tensor-plan
  and WebGPU explicitly refuse the operation until canonical tile/index layout
  semantics exist; no backend-shaped modulo handler or plan opcode is added.
- Grad consumes the same cross-package values/dtype/refusal fixture and now
  returns owning float16, int32, and bool results without the former silent
  float32 cast.
- ADR-0008 retires `tensor.repeat` and `jit.custom.repeat.v0`. The current
  opaque inventory narrows from 29 constructor calls/32 operations to 28/31;
  the registry contains eight typed retirements and preserves the original
  39-ID partition. The shared opaque-refusal fixture now exercises
  `repeat_interleave`.
- Final verification passes JIT build/codegen, strict typecheck, lint, 4 unit
  files/24 tests, the Gate0/repeat/support/VJP/IR group at 5 files/53 tests,
  and 30 full integration files/256 tests in 56.46 seconds. Grad build/codegen,
  strict typecheck, lint, 2 unit files/30 tests, the shared eager conformance
  test, and 35 full integration files/325 tests in 63.13 seconds pass. The
  architecture guard passes; the complete compiler suite passes 96 files/1,602
  tests. Release verification passes 19 Python archive tests, 35 Node security
  tests, and every packed/fresh consumer. The seven-package workspace build
  passes under Node 25.9.0 in 8.04 seconds.
- Gate 6 remains `in-progress`. The next bounded migration is
  `Tensor.repeat_interleave`, with strict axis/count normalization, selected-
  axis block-sum VJP, transform/export decisions, and the same cross-package
  dtype/refusal conformance discipline.

### 2026-07-22 — Gate 6 typed `Tensor.repeat_interleave` and eager conformance

- Added typed `REPEAT_INTERLEAVE` and its executable registry record. JIT and
  Grad accept only exact built-in or fixed-width NumPy integer repeat/axis
  scalars, never invoke arbitrary conversion hooks, normalize one negative
  axis, reject scalar-rank and out-of-range axes, and bound repeat counts to
  `[0, 2^30]`. The independent ceiling rejects hostile empty-output requests.
- The JIT contract requires one input, a plain closed axis/repeats record plus
  optional VJP provenance, exact dtype preservation, and an output shape
  derived by multiplying only the selected axis. Construction, CPU, VJP,
  vmap, and ONNX boundaries revalidate it, so post-construction mutation fails
  locally.
- CPU returns an owning dtype-preserving NumPy result. Closure and symbolic
  VJP reshape the expanded axis into `(source-extent, repeats)` and reduce the
  repeat block, including zero-repeat gradients. Vmap shifts the selected axis
  past its leading batch axis and therefore never repeats the batch dimension.
- ONNX opset 17 emits exact `Unsqueeze`, `Tile`, and `Reshape` nodes with
  signed-int64 axis, repeat-vector, and output-shape initializers for
  float32/int32/int64/bool graphs and rejects other exporter dtypes.
  Tensor-plan and WebGPU explicitly refuse the operation until canonical
  selected-axis replication layout semantics exist.
- Grad consumes the same cross-package values/dtype/refusal fixture and now
  returns owning float16, int32, and bool results while preserving backward
  dtype instead of silently casting results and gradients to float32.
- ADR-0009 retires `tensor.repeat-interleave` and
  `jit.custom.repeat-interleave.v0`. The current opaque inventory narrows from
  28 constructor calls/31 operations to 27/30; the registry contains nine
  typed retirements and preserves the original 39-ID partition. The shared
  opaque-refusal fixture now exercises `prod`.
- Final verification passes JIT build/codegen, strict typecheck, lint, 4 unit
  files/24 tests, the Gate0/repeat-interleave/support/VJP/IR group at 5
  files/53 tests, and 31 full integration files/260 tests in 53.47 seconds.
  Grad build/codegen, strict typecheck, lint, 2 unit files/30 tests, the shared
  eager conformance test, and 36 full integration files/326 tests in 59.99
  seconds pass. The architecture guard passes; the complete compiler suite
  passes 96 files/1,602 tests. Release verification passes 19 Python archive
  tests, 35 Node security tests, and every packed/fresh consumer. The
  seven-package workspace build passes under Node 25.9.0 in 7.89 seconds.
- Gate 6 remains `in-progress`. The next bounded migration is `Tensor.prod`,
  with strict reduction-axis/dtype semantics, scalar-result normalization,
  typed VJP, transform/export decisions, and shared eager conformance.

### 2026-07-22 — Gate 6 typed `Tensor.prod` and eager conformance

- Added typed `PROD` and its executable registry record. JIT and Grad accept
  only exact built-in or fixed-width NumPy integer axes, never invoke arbitrary
  conversion or iteration hooks, normalize negative axes, reject explicit
  empty, duplicate, scalar-rank, and out-of-range axes, and require exact
  boolean keep-dimension flags.
- The JIT contract requires one input, a plain closed canonical axes/keepdims
  record plus optional VJP provenance, exact dtype preservation, and the exact
  derived reduced shape. Construction, CPU, VJP, vmap, and ONNX boundaries
  revalidate it, so post-construction mutation fails locally.
- CPU returns an owning dtype-preserving scalar or tensor result, including
  empty-domain identity reductions. Closure and symbolic VJP implement the
  zero-aware product rule: quotient for zero-free groups, the nonzero product
  only at the sole zero, and zero for groups containing multiple zeros. Vmap
  shifts every declared reduction axis past its leading batch axis.
- ONNX opset 17 emits `ReduceProd` with exact axes/keepdims for
  float32/int32/int64 graphs and rejects other exporter dtypes. Tensor-plan and
  WebGPU explicitly refuse the operation until a portable product-reduction
  lowering exists; no host callback is relabeled as device support.
- Grad consumes the same cross-package values/dtype/refusal fixture, returns
  owning float16, int32, and bool results, preserves backward dtype, and fixes
  the former division-only derivative at zero.
- ADR-0010 retires `tensor.prod` and `jit.custom.prod.v0`. The current opaque
  inventory narrows from 27 constructor calls/30 operations to 26/29; the
  registry contains ten typed retirements and preserves the original 39-ID
  partition. The shared opaque-refusal fixture now exercises `gather`.
- Final verification passes JIT build/codegen, strict typecheck, lint, 4 unit
  files/24 tests, the Gate0/product/support/VJP/IR group at 5 files/53 tests,
  and 32 full integration files/264 tests in 55.17 seconds. Grad
  build/codegen, strict typecheck, lint, 2 unit files/30 tests, the shared eager
  conformance test, and 37 full integration files/327 tests in 61.65 seconds
  pass. The architecture guard passes; the complete compiler suite passes 96
  files/1,602 tests. Release verification passes 19 Python archive tests, 35
  Node security tests, and every packed/fresh consumer. The seven-package
  workspace build passes under Node 25.9.0 in 7.93 seconds.
- Gate 6 remains `in-progress`. The next bounded migration is `Tensor.gather`,
  with strict index/axis semantics, scatter-add VJP, transform/export
  decisions, and shared eager conformance.

### 2026-07-22 — Gate 6 typed `Tensor.gather` and eager conformance

- Added typed `INDEX` and its executable `Tensor.gather` registry record. JIT
  and Grad accept only exact built-in or fixed-width NumPy integer axes and an
  actual int64 tensor index. They normalize one negative axis and reject bool,
  floating, hostile-conversion, scalar-rank, dtype, rank, and non-gather extent
  violations before execution.
- The JIT contract requires source and index inputs, a plain closed normalized
  axis record plus optional VJP provenance, an index-shaped output, source
  dtype preservation, same nonzero rank, and bounded non-gather extents.
  Construction, CPU, VJP, vmap, ONNX, and tensor-plan boundaries revalidate
  those facts, so post-construction mutation fails locally.
- CPU checks every nonempty index value against the gather-axis extent and
  rejects negative NumPy wrapping. Exact coordinate tensors preserve the
  PyTorch rule that non-gather index dimensions may be smaller than the source;
  the result is an owning source-dtype array and empty indices remain valid.
- Closure and symbolic VJP use deterministic `SCATTER_ADD` into a zero
  source-shaped tensor, so duplicate indices accumulate and the discrete index
  has no gradient. The scatter handler consumes the same coordinate/range
  semantics and a separate closed structural validator. Paired vmap requires
  source and index to share the leading mapped axis and shifts the gather axis.
- ONNX opset 17 emits `GatherElements` for float32/int32/int64/bool source
  graphs with int64 indices. Tensor-plan and WebGPU explicitly refuse until a
  deterministic bounds-checked index/scatter lowering exists; the old
  structural plan allowlist is not reported as device capability.
- Grad consumes the same cross-package values/dtype/refusal fixture, supports
  smaller non-gather dimensions and duplicate-index accumulation, returns an
  owning result, and preserves source/output/backward dtype through the shared
  eager gradient accumulator. The full regression exposed that its legacy
  indexing and reshape path converted an otherwise valid int64 token tensor to
  float32 before gather. The accepted freeze now records dtype-preserving
  indexing, reshape, transpose, and permute with compatible NumPy aliasing;
  the reasoning-workshop sequence-logprob and policy-gradient paths pass.
- ADR-0011 retires `tensor.gather` and `jit.custom.gather.v0`. The current
  opaque inventory narrows from 26 constructor calls/29 operations to 25/28;
  the registry contains eleven typed retirements and preserves the original
  39-ID partition. The shared opaque-refusal fixture now exercises `var`.
- Verification passes the semantic architecture guard; JIT focused
  gather/support/Gate0/IR/VJP group at 5 files/53 tests, 4 unit files/24 tests,
  and 33 full integration files/268 tests in 66.41 seconds; Grad focused
  gather/Gate0/reasoning group at 3 files/23 tests, 2 unit files/30 tests, and
  38 full integration files/328 tests in 60.80 seconds. Strict typecheck and
  lint pass for both packages. The complete compiler gate passes 96 files and
  1,602 tests. Release verification passes 19 Python archive tests, 35 Node
  security tests, and every packed/fresh consumer. The seven-package workspace
  build passes under Node 25.9.0 in 7.90 seconds.
- Gate 6 remains `in-progress`. The next bounded migration is `Tensor.var`,
  with canonical correction/axis semantics, stable reduction VJP,
  transform/export decisions, and shared eager conformance.

### 2026-07-22 — Gate 6 typed `Tensor.var` and eager conformance

- Added typed `VAR` and its executable `Tensor.var` registry record. JIT and
  Grad accept canonical normalized static axes, exact signed 32-bit correction,
  exact keepdims, and float16/32/64 inputs. Default correction is one; the
  legacy boolean `unbiased` alias remains available only when correction is
  absent. Bool, floating, hostile-conversion, empty, duplicate, out-of-range,
  ambiguous, and non-floating requests fail before execution.
- CPU realization returns owning dtype-preserving scalar/tensor arrays.
  Closure and symbolic VJP use the centered correction-aware derivative and
  preserve the source dtype. Scalar and zero-denominator floating behavior is
  explicit and covered rather than being rejected by the old callback path.
- Vmap shifts every reduction axis past a leading mapped dimension. ONNX
  opset 17 decomposes the exact float32 profile into `ReduceMean`, `Sub`, `Mul`,
  `ReduceSum`, and `Div`; float16/float64 export fail explicitly. Tensor-plan
  and WebGPU refuse until a portable variance reduction exists.
- Construction, CPU, VJP, vmap, ONNX, and tensor-plan boundaries consume the
  same closed validator, including mutation tests. Grad consumes the shared
  eager/lazy fixture and no longer casts variance results or gradients to f32.
- ADR-0012 retires `tensor.var` and `jit.custom.var.v0`. The current opaque
  inventory narrows from 25 constructor calls/28 operations to 24/27; the
  registry contains twelve typed retirements and preserves the original
  39-ID partition. The shared opaque-refusal fixture now exercises
  `masked_fill`.
- Verification passes the semantic architecture guard; JIT passes 34 full
  integration files/272 tests in 58.51 seconds and 4 unit files/24 tests; Grad
  passes 39 full integration files/329 tests in 64.88 seconds and 2 unit
  files/30 tests. Strict typecheck and lint pass for both packages. The
  compiler gate passes 96 files/1,602 tests. Release verification passes 19
  Python archive tests, 35 Node security tests, and every packed/fresh
  consumer. The seven-package Node 25.9.0 workspace build passes in 8.09
  seconds.
- One attempted overlap of the workspace build and compiler verification
  failed because both commands clean and materialize the same compiler `dist`
  directory. The unchanged compiler verification passed when rerun alone.
  Those artifact-writing commands are therefore serialized; independent JIT
  and Grad regressions remain safely parallel.
- Gate 6 remains `in-progress`. The next bounded migration is
  `Tensor.masked_fill`, with closed mask/broadcast semantics, selection VJP,
  transform/export decisions, and shared eager conformance.

### 2026-07-22 — Gate 6 typed `Tensor.masked_fill` and eager conformance

- Replaced the public `Tensor.masked_fill` opaque callback with typed `WHERE`
  and added its executable framework-operation registry record. The operation
  requires an actual bool tensor mask whose shape broadcasts into, but never
  enlarges, the source shape. Fill values are normalized without invoking
  hostile conversion hooks and must be exactly representable by the source
  dtype; float sources retain explicit `inf`/`nan` behavior.
- CPU realization returns an owning source-shaped, source-dtype array. Closure
  and symbolic VJP select the incoming gradient on the mask complement, while
  Grad's compatibility in-place spelling retains the same dtype-preserving
  behavior as the out-of-place result.
- Vmap supports a leading mapped source with captured or mapped broadcast
  masks. ONNX opset 17 lowers the exact float32/int32/int64/bool profile to
  `Where`. Tensor-plan and WebGPU refuse explicitly because the current tensor
  backend has no portable masked-selection operation.
- Construction, generic `WHERE`, CPU, VJP, vmap, ONNX, and tensor-plan
  boundaries share closed validation. Grad and JIT consume the same eager/lazy
  conformance fixture, including strict mask, broadcast, scalar, dtype,
  gradient, and hostile-input cases.
- ADR-0013 retires `tensor.masked-fill` and
  `jit.custom.masked-fill.v0`. The current opaque inventory narrows from 24
  constructor calls/27 operations to 23/26; the registry contains thirteen
  typed retirements and preserves the original 39-ID partition. The shared
  opaque-refusal fixture now exercises `Tensor.tril`.
- Verification passes the semantic architecture guard; JIT passes 35 full
  integration files/276 tests in 60.41 seconds and 4 unit files/24 tests; Grad
  passes 40 full integration files/330 tests in 66.41 seconds and 2 unit
  files/30 tests. Strict typecheck and lint pass for both packages. The
  compiler gate passes 96 files/1,602 tests. Release verification passes 19
  Python archive tests, 35 Node security tests, and every packed/fresh
  consumer. The seven-package Node 25.9.0 workspace build passes in 8.14
  seconds.
- Gate 6 remains `in-progress`. The next bounded migration is `Tensor.tril`,
  with typed triangular-selection semantics, stable selection VJP, eager
  conformance, and explicit transform/export/device decisions.

### 2026-07-22 — Gate 6 typed `Tensor.tril` and eager conformance

- Replaced the public `tril` opaque callback with typed `TRIL` and added its
  executable framework-operation registry record. Both JIT and Grad now expose
  equivalent instance and top-level spellings over the final two axes of a
  matrix or batch of matrices.
- Construction requires rank at least two, a supported real-numeric or boolean
  dtype, and an exact built-in or fixed-width NumPy integer diagonal. Boolean,
  floating, string, and hostile conversion objects fail before execution. The
  diagonal saturates into the unique matrix-derived
  `[-rows, columns - 1]` semantic range for nonempty matrices (zero for empty
  matrices), keeping arbitrarily large Python integers away from NumPy and
  ONNX while preserving their all-zero/all-input meaning.
- CPU realization returns an owning source-shaped, source-dtype array,
  including empty matrix dimensions. Closure and symbolic VJP apply the same
  idempotent lower-triangular selection to the incoming gradient. Leading-axis
  vmap preserves the final matrix axes and refuses a captured-only source.
- ONNX opset 17 emits exact `Trilu` input wiring, scalar int64 diagonal, and
  `upper=0` for float32/int32/int64/bool. Tensor-plan and WebGPU refuse until a
  portable triangular-selection lowering exists. Construction and every CPU,
  VJP, vmap, ONNX, and plan boundary consume the same closed validator and the
  test harness parses the emitted protobuf instead of relying on a substring.
- ADR-0014 retires `tensor.tril` and `jit.custom.tril.v0`. The current opaque
  inventory narrows from 23 constructor calls/26 operations to 22/25; the
  registry contains fourteen typed retirements and preserves the original
  39-ID partition. The shared opaque-refusal fixture now exercises
  `Tensor.triu`.
- Verification passes the semantic architecture guard; JIT passes 36 full
  integration files/280 tests in 62.80 seconds and 4 unit files/24 tests; Grad
  passes 41 full integration files/331 tests in 68.59 seconds and 2 unit
  files/30 tests. Strict typecheck and lint pass for both packages. The
  compiler gate passes 96 files/1,602 tests in 59.64 seconds. Release
  verification passes 19 Python archive tests, 35 Node security tests, and
  every packed/fresh consumer. The seven-package Node 25.9.0 workspace build
  passes in 7.96 seconds.
- Gate 6 remains `in-progress`. The next bounded migration is `Tensor.triu`,
  with the same typed upper-triangular selection seam and explicit
  transform/export/device decisions.

### 2026-07-22 — Gate 6 typed `Tensor.triu` and consolidated triangular harness

- Replaced the final public triangular `CUSTOM` callback with typed `TRIU` and
  added its executable framework-operation registry record. JIT and Grad now
  expose equivalent instance and top-level `triu` over the final two axes of a
  matrix or batch of matrices.
- `TRIL` and `TRIU` share one construction-time diagonal normalizer, supported
  dtype profile, and closed executable triangular validator instead of
  accumulating two spelling-shaped semantic paths. Upper selection uses the
  exact nonempty canonical range `[1 - rows, columns]` (zero for empty matrix
  dimensions), so arbitrarily large integers saturate to the unique all-input
  or all-zero representative before NumPy or ONNX.
- CPU realization returns an owning source-shaped/source-dtype array. Closure
  and symbolic VJP apply the same idempotent upper selection. Leading-axis
  vmap preserves the final matrix axes and refuses captured-only sources.
  ONNX opset 17 emits exact `Trilu` wiring, scalar int64 diagonal, and
  `upper=1` for float32/int32/int64/bool. Tensor-plan and WebGPU refuse until a
  portable triangular-selection lowering exists.
- Replaced the separate lower-only JIT/Grad fixture files with one shared
  two-variant triangular conformance module and parameterized package tests.
  The same four JIT cases now prove both variants across batched values,
  saturation, empty matrices, dtypes, closure/symbolic gradients, vmap, exact
  parsed ONNX protobuf fields, hostile inputs, boundary mutation, and backend
  refusal without duplicating the protobuf parser or assertion matrix.
- ADR-0015 retires `tensor.triu` and `jit.custom.triu.v0`. The current opaque
  inventory narrows from 22 constructor calls/25 operations to 21/24; the
  registry contains fifteen typed retirements and preserves the original
  39-ID partition. The shared opaque-refusal fixture now exercises `cumsum`.
- Verification passes the semantic architecture guard; JIT passes 36 full
  integration files/280 tests in 62.09 seconds and 4 unit files/24 tests; Grad
  passes 41 full integration files/331 tests in 68.31 seconds and 2 unit
  files/30 tests. Strict typecheck and lint pass for both packages. The
  compiler gate passes 96 files/1,602 tests in 60.02 seconds. Release
  verification passes 19 Python archive tests, 35 Node security tests, and
  every packed/fresh consumer in 8.82 seconds. The seven-package Node 25.9.0
  workspace build passes in 7.93 seconds.
- Gate 6 remains `in-progress`. The next bounded migration is `Tensor.cumsum`,
  with canonical axis/dtype accumulation, scan VJP, eager conformance, and
  explicit transform/export/device decisions.

### 2026-07-22 — Gate 6 typed `Tensor.cumsum`

- Replaced the public cumulative-sum `CUSTOM` callback with typed `CUMSUM` and
  added its executable framework-operation registry record. JIT and Grad share
  instance and top-level inclusive-scan semantics over one exact normalized
  built-in or fixed-width NumPy integer axis.
- The closed dtype contract preserves floating defaults, promotes integral and
  boolean defaults to int64, and casts before accumulation for an explicit
  supported BrowserGrad/PyTorch dtype token. CPU realization returns an owning
  array in the exact declared dtype, including empty inputs. Unsupported source
  dtypes, coercive axis/dtype objects, malformed IR, scalar input, and `out=`
  mutation fail at their semantic boundary.
- Closure and symbolic autograd use an inclusive scan in the opposite
  direction and cast back to source dtype where required. Edges are admitted
  only for floating source and output dtypes. Vmap shifts the scan axis past a
  leading mapped dimension. ONNX opset 17 emits an exact scalar int64 axis,
  explicit `exclusive=0`/direction attributes, and a preceding `Cast` when the
  supported float32/int32/int64 output differs from input. Tensor-plan and
  WebGPU explicitly refuse until portable scan lowering exists.
- One shared cross-package fixture covers inclusive values on two axes,
  negative-axis normalization, empty dimensions, float16 preservation,
  integral/bool promotion, int32 overflow prevention, explicit dtype aliases,
  closure and symbolic gradients across dtype conversion, owning results,
  hostile inputs, mutation refusal, vmap, exact parsed ONNX protobuf fields,
  and backend boundaries.
- ADR-0016 retires `tensor.cumsum` and `jit.custom.cumsum.v0`. The opaque
  inventory narrows from 21 constructor calls/24 operations to 20/23; the
  registry contains sixteen typed retirements and preserves the original
  39-ID partition. The shared opaque-refusal fixture now exercises `cat`.
- Verification passes the semantic architecture guard; JIT passes 37 full
  integration files/283 tests in 64.27 seconds and 4 unit files/24 tests; Grad
  passes 42 full integration files/332 tests in 69.94 seconds and 2 unit
  files/30 tests. Strict typecheck and lint pass for both packages. The
  compiler gate passes 96 files/1,602 tests. Release verification passes 19
  Python archive tests, 35 Node security tests, and every packed/fresh consumer
  in 8.77 seconds. The seven-package Node 25.9.0 workspace build passes in
  7.87 seconds.
- Gate 6 remains `in-progress`. The next bounded migration is `Tensor.cat`,
  with canonical variadic shape/dtype promotion, split VJP, eager conformance,
  and explicit transform/export/device decisions.

### 2026-07-22 — Gate 6 typed `cat`

- Replaced the top-level concatenation `CUSTOM` callback with typed variadic
  `CONCAT` and added its executable framework-operation registry record. JIT
  and Grad now share one strict existing-axis shape, dtype-promotion,
  legacy-empty, ownership, mutation, and resource contract.
- The public boundary accepts only a nonempty plain tuple/list of exact tensor
  values, caps arity at 1,024 and output storage at 256 MiB, and normalizes one
  exact built-in or fixed-width NumPy integer axis without arbitrary conversion
  hooks. Substantive inputs must match outside that axis; the rank-1 `(0,)`
  compatibility empty contributes zero elements while still participating in
  dtype promotion.
- The closed bool/uint8/signed-int/float promotion rule follows dimensioned-
  tensor category precedence. CPU casts inputs to the exact declared promoted
  dtype and returns an owning copy. Closure and symbolic VJP split the
  cotangent at static segment boundaries through typed internal `NARROW`,
  reshape compatibility-empty gradients, and cast floating gradients back to
  source dtype.
- Vmap shifts the concatenation axis past a leading batch and broadcasts
  captured inputs. ONNX opset 17 emits exact per-input `Cast` plus `Concat` for
  float32/int32/int64/bool and omits only a rank-mismatched `(0,)` compatibility
  input. Tensor-plan/WebGPU and `out=` fail explicitly until canonical variadic
  copy lowering and a typed mutation/effect contract exist.
- One shared cross-package fixture covers zero-width and legacy-empty inputs,
  all-empty concatenation, six promotion combinations, owning output,
  closure/symbolic mixed-dtype gradients, captured-input vmap, exact parsed
  ONNX protobuf fields, hostile inputs, resource ceilings, post-construction
  mutation, and backend refusal.
- ADR-0017 retires `tensor.cat` and `jit.custom.cat.v0`. The opaque inventory
  narrows from 20 constructor calls/23 operations to 19/22; the registry
  contains seventeen typed retirements and preserves the original 39-ID
  partition. The shared opaque-refusal fixture now exercises `stack`.
- Verification passes the semantic architecture guard; JIT passes 38 full
  integration files/286 tests in 62.42 seconds and 4 unit files/24 tests; Grad
  passes 43 full integration files/333 tests in 67.81 seconds and 2 unit
  files/30 tests. Strict typecheck and lint pass for both packages. The
  compiler gate passes 96 files/1,602 tests. Release verification passes 19
  Python archive tests, 35 Node security tests, and every packed/fresh consumer
  in 8.76 seconds. The seven-package Node 25.9.0 workspace build passes in
  7.94 seconds.
- Gate 6 remains `in-progress`. The next bounded migration is top-level
  `stack`, reusing the typed variadic promotion/splitting foundation while
  giving inserted-axis shape, vmap, export, and backend behavior its own exact
  contract.

### 2026-07-22 — Gate 6 typed `stack`

- Replaced the top-level stacking `CUSTOM` callback with typed variadic
  `STACK` and added its executable framework-operation registry record. JIT
  and Grad now share one strict inserted-axis shape, dtype-promotion,
  ownership, mutation, and resource contract.
- The public boundary accepts only a nonempty plain tuple/list of exact
  same-session tensor values, caps arity at 1,024 and output storage at 256
  MiB, requires identical input shapes, and normalizes one exact built-in or
  fixed-width NumPy integer axis over the output rank without arbitrary
  conversion hooks. Scalar and identically shaped empty inputs are supported.
- Stack reuses cat's closed bool/uint8/signed-int/float dimensioned-tensor
  promotion rule. CPU casts inputs to the exact declared promoted dtype and
  returns an owning copy. Closure and symbolic VJP select each static source
  index through typed internal `NARROW`, remove the inserted axis with
  `RESHAPE`, and cast floating gradients back to source dtype.
- Vmap shifts the stack axis past a leading batch and broadcasts captured
  inputs. ONNX opset 17 emits exact per-input `Cast` and `Unsqueeze` plus one
  `Concat` for float32/int32/int64/bool. Tensor-plan/WebGPU and `out=` fail
  explicitly until canonical variadic copy lowering and a typed mutation/effect
  contract exist.
- One shared cross-package fixture covers three-way and negative axes, six
  promotion combinations, scalar and empty inputs, owning output,
  closure/symbolic mixed-dtype gradients, captured-input vmap, exact parsed
  ONNX protobuf fields, hostile inputs, resource ceilings, post-construction
  mutation, and backend refusal.
- ADR-0018 retires `tensor.stack` and `jit.custom.stack.v0`. The opaque
  inventory narrows from 19 constructor calls/22 operations to 18/21; the
  registry contains eighteen typed retirements and preserves the original
  39-ID partition. The shared opaque-refusal fixture now exercises still-opaque
  `pad`.
- Verification passes the semantic architecture guard; focused JIT and Grad
  conformance passes 64 tests. JIT passes 39 full integration files/289 tests
  in 82.44 seconds and 4 unit files/24 tests; Grad passes 44 full integration
  files/334 tests in 89.76 seconds and 2 unit files/30 tests. Strict typecheck
  and lint pass for both packages. The compiler gate passes 96 files/1,602
  tests. Release verification passes 19 Python archive tests, 35 Node security
  tests, and every packed/fresh consumer. The seven-package Node 25.9.0
  workspace build passes in 10.09 seconds.
- Gate 6 remains `in-progress`. The next bounded migration is `F.pad`, closing
  the existing typed `PAD` seam across its public builder, validator, VJP,
  export, backend decisions, and eager conformance.

### 2026-07-22 — Gate 6 typed `torch.nn.functional.pad`

- Replaced the public constant-padding `CUSTOM` callback with typed `PAD` and
  added its executable framework-operation registry record. JIT and Grad share
  one strict trailing-dimension geometry, exact fill, dtype, ownership,
  gradient, and resource contract.
- The public boundary accepts an exact tensor and a plain even-length
  tuple/list of built-in or fixed-width NumPy integers in PyTorch's
  last-dimension-first order. It normalizes these into a canonical rank-sized
  first-dimension-first tuple and supports nonnegative `constant` padding only.
  Rank is capped at 32, every output extent at 268,435,456 elements, and total
  output storage at 256 MiB. The independent extent ceiling prevents a zero
  dimension from masking an otherwise unbounded host integer in the byte
  product.
- The closed bool/uint8/signed-int/float dtype profile preserves source dtype.
  `None` becomes exact zero; bool requires bool; integral fills must be exact
  and in range; floating fills must remain finite after destination conversion.
  CPU returns an owning exact-dtype copy. Closure and symbolic VJP extract the
  static interior, with symbolic differentiation emitting typed `SLICE` and no
  `CUSTOM`.
- Vmap prepends a no-op batch padding pair. ONNX opset 17 emits one exact int64
  pads initializer and one exact scalar fill initializer for
  float32/int32/int64. Tensor-plan and WebGPU explicitly refuse until canonical
  padding/layout lowering and a corresponding kernel exist. Non-constant modes
  and negative cropping remain distinct explicit refusals.
- One shared cross-package fixture covers asymmetric multi-axis padding,
  float16/int64/bool/uint8 preservation, zero and empty shapes, owning results,
  closure and symbolic gradients, vmap, exact parsed ONNX protobuf fields,
  hostile inputs, post-construction mutation, rank/byte/per-axis resource
  bounds, and portable-backend refusal.
- ADR-0019 retires `functional.pad` and `jit.custom.pad.v0`. The opaque
  inventory narrows from 18 constructor calls/21 operations to 17/20; the
  registry contains nineteen typed retirements and preserves the original
  39-ID partition. The shared opaque-refusal fixture now exercises still-opaque
  `l1_loss`.
- Verification passes the semantic architecture guard; focused JIT coverage
  passes 6 files/70 tests and focused Grad coverage passes the new conformance
  file. JIT passes 40 full integration files/292 tests in 97.94 seconds while
  Grad passes 45 files/335 tests in 105.25 seconds; the two lanes complete in
  parallel in 105.86 seconds. JIT passes 4 unit files/24 tests and Grad passes
  2 files/30 tests. Strict typecheck and lint pass for both packages. The
  compiler gate passes 96 files/1,602 tests in 82.38 seconds. Release
  verification passes 19 Python archive tests, 35 Node security tests, and all
  packed/fresh consumers in 11.15 seconds. The seven-package Node 25.9.0
  workspace build passes in 10.22 seconds.
- Gate 6 remains `in-progress`. The next bounded audit is the paired
  `Tensor.sort` values/indices surface, which must close ordering, tie, dtype,
  VJP, transform/export, and backend semantics together rather than migrate
  either opaque output independently.

### 2026-07-22 — Gate 6 typed `torch.sort`

- Replaced the two public sort `CUSTOM` callbacks with paired typed
  `SORT_INDICES` and `SORT_VALUES`. Both outputs bind the exact same source,
  normalized axis, descending flag, stable flag, and permutation. JIT and Grad
  share one ordering, dtype, ownership, gradient, hostile-input, and resource
  conformance fixture.
- The closed bool/uint8/signed-int/float profile preserves value dtype and
  returns owning int64 indices. CPU ordering is deterministic and stable. The
  descending algorithm avoids negation and is covered for stable ties,
  unsigned values, minimum int64, and NaNs. `stable=False` intentionally uses
  the same deterministic order while satisfying its weaker tie contract.
- Rank is capped at 32, every extent at 268,435,456, the selected axis at
  1,048,576 elements, and combined values-plus-indices storage at 256 MiB.
  Scalar and empty shapes are defined; zero-size tensors cannot hide an
  oversized nonselected extent. Coercive dimensions and flags, unsupported
  dtypes, invalid axes, output mutation, and mismatched paired IR fail before
  execution.
- Values use an immutable permutation-scatter closure and typed symbolic
  `SCATTER_ADD` VJP; indices are discrete. Vmap shifts the selected axis past
  the leading batch dimension. ONNX opset 17 emits full-axis `TopK` and
  `GatherElements` for float32/int32/int64 and explicitly refuses scalar,
  empty-axis, or unsupported-dtype export. Tensor-plan and WebGPU validate and
  refuse until canonical portable ordering exists.
- ADR-0020 retires `tensor.sort-indices`, `tensor.sort-values`, and their two
  opaque IDs. The frozen inventory narrows from 17 constructor calls/20
  operations to 15/18; the registry contains 21 typed retirements and
  preserves the original 39-ID partition. Gate 0 retains four representative
  forward-only callbacks: `einsum`, `scatter`, and paired `topk` outputs.
- Verification passes the semantic architecture guard; focused JIT coverage
  passes 6 files/70 tests and focused Grad coverage passes the new conformance
  file. Full JIT integration passes 41 files/295 tests in 91.66 seconds while
  Grad passes 46 files/336 tests in 98.19 seconds; the two lanes complete in
  parallel in 98 seconds. JIT passes 4 unit files/24 tests and Grad passes 2
  files/30 tests. Strict typecheck and lint pass for both packages. The
  compiler gate passes 96 files/1,602 tests in 79.82 seconds. Release
  verification passes 19 Python archive tests, 35 Node security tests, and all
  packed/fresh consumers in 11.14 seconds. The seven-package Node 25.9.0
  workspace build passes in 10.08 seconds.
- Gate 6 remains `in-progress`. The next coherent migration is paired
  `Tensor.topk`, reusing the ordering/permutation seam while closing the
  selected-`k`, largest, sorted, tie, gradient, transform, export, and backend
  contracts together.

### 2026-07-22 — Gate 6 typed `torch.topk`

- Replaced the two public top-k `CUSTOM` callbacks with paired typed
  `TOPK_INDICES` and `TOPK_VALUES`. Both outputs bind the exact same source,
  normalized axis, exact k, largest flag, sorted flag, and selected
  permutation. JIT and Grad share one selection, dtype, ownership, gradient,
  hostile-input, mutation, and resource conformance fixture.
- The closed bool/uint8/signed-int/float profile preserves value dtype and
  returns owning int64 indices. CPU computes one negation-free `argpartition`
  selection. Sorted requests order only the selected k values; unsorted
  requests retain partial-selection order. Tie identity is deliberately not a
  public stability promise, while the pinned NumPy runtime is deterministic.
  Unsigned values, minimum int64, NaNs, ties, k=0, and the `(4, 50257) ->
  (4, 10)` workshop shape are covered.
- Rank is capped at 32, every extent at 268,435,456, the selected axis at
  1,048,576 elements, paired output storage at 256 MiB, and a conservative
  full-permutation plus selected-buffer workspace projection at 256 MiB. Typed
  full sort now enforces an equivalent conservative workspace ceiling.
  Coercive dimensions/k/flags, unsupported dtypes, scalar input, invalid k or
  axes, output mutation, oversized work, and malformed/mismatched paired IR
  fail before allocation or execution.
- Values use an immutable selected-permutation scatter closure and typed
  symbolic `SCATTER_ADD` VJP; indices are discrete. Vmap shifts the selected
  axis past the leading batch dimension. ONNX opset 17 emits selected-k `TopK`
  and `GatherElements` for float32/int32/int64, while k=0 and unsupported
  exporter dtypes fail explicitly. Tensor-plan and WebGPU validate and refuse
  until canonical portable partial selection exists.
- ADR-0021 retires `tensor.topk-indices`, `tensor.topk-values`, and their two
  opaque IDs. The frozen inventory narrows from 15 constructor calls/18
  operations to 13/16; the registry contains 23 typed retirements and
  preserves the original 39-ID partition. Gate 0 retains `einsum` and
  `scatter` as its forward-only callback representatives.
- Focused coverage passes JIT 7 files/73 tests and Grad 5 files/63 tests. Full
  JIT integration passes 42 files/298 tests in 91.42 seconds while Grad passes
  47 files/337 tests in 97.62 seconds; the lanes run in parallel. JIT passes 4
  unit files/24 tests and Grad passes 2 files/30 tests. Strict typecheck and
  lint pass for both packages. The compiler gate passes 96 files/1,602 tests
  in 74.62 seconds. Release verification passes 19 Python archive tests, 35
  Node security tests, and all packed/fresh consumers in 12.01 seconds. The
  seven-package Node 25.9.0 workspace build passes in 9.75 seconds.
- Gate 6 remains `in-progress`. The next coherent migration is
  `Tensor.scatter`, reusing typed index/scatter primitives while closing exact
  axis, index bounds, duplicate-write/reduction, dtype, gradient, transform,
  export, and backend decisions together.

### 2026-07-22 — Gate 6 typed overwrite `torch.scatter`

- Replaced the public overwrite-scatter `CUSTOM` callback with typed `SCATTER`
  over `(target, int64 index, source)` and one normalized axis. JIT and Grad
  share the exact same shape, dtype, ownership, gradient, hostile-input,
  mutation, and resource conformance fixture.
- The closed profile accepts a nonzero same-rank target/index pair through rank
  32. Index extents cannot exceed target extents, every runtime destination is
  nonnegative and in range, and destinations along the selected axis must be
  unique. Tensor source has exactly the index shape and target dtype; exact
  built-in or fixed NumPy scalars remain scalar IR instead of allocating an
  index-shaped temporary. Bool, uint8, signed integer, and float16/32/64 targets
  preserve dtype and return an owning result.
- Output storage is bounded at 256 MiB. The conservative validation projection
  also bounds output plus sorted-int64 and adjacent-equality duplicate-check
  workspace at 256 MiB before CPU allocation. CPU copies the target once,
  rejects malformed destinations, then performs one `put_along_axis` overwrite.
  Coercive axes/sources, bool index, negative/out-of-range or duplicate index,
  incompatible source shape/dtype, reductions, mutated IR, and oversized work
  fail explicitly.
- Closure and symbolic differentiation preserve untouched target cotangents,
  zero overwritten target locations through typed `SCATTER`, and gather a
  tensor-source cotangent through typed `INDEX`; scalar source and index are
  discrete. Grad snapshots index state for backward. Vmap broadcasts captured
  target/index/source operands and shifts the axis after the leading batch
  dimension. ONNX opset 17 emits `ScatterElements` for the admitted
  float32/int32/int64/bool profile and inserts `Expand` for scalar source.
  Tensor-plan and WebGPU validate then refuse until canonical portable
  overwrite indexing exists.
- ADR-0022 retires `tensor.scatter` and `jit.custom.scatter.v0`. The frozen
  inventory narrows from 13 constructor calls/16 operations to 12/15; the
  executable registry contains 24 typed retirements and preserves the original
  39-ID partition. `einsum` is now the only forward-only callback in the Gate 0
  representative set.
- Focused coverage passes JIT 6 files/70 tests and Grad 2 files/22 tests. Full
  JIT integration passes 43 files/301 tests in 94.07 seconds while Grad passes
  48 files/338 tests in 99.74 seconds; those lanes run in parallel. JIT passes
  4 unit files/24 tests and Grad passes 2 files/30 tests. Strict typecheck and
  lint pass for both packages. The compiler gate passes 96 files/1,602 tests in
  73.62 seconds. Release verification passes 19 Python archive tests, 35 Node
  security tests, and all packed/fresh consumers in 10.86 seconds. The
  seven-package Node 25.9.0 workspace build passes in 10.94 seconds.
- An attempted compiler/release/workspace-build fan-out exposed a shared-output
  harness boundary: compiler build cleans and rematerializes `dist`, while the
  release consumer reads that same tree. The concurrent run failed in the
  package factory/release consumer; the dependency-ordered rerun passed. JIT
  and Grad remain safely parallel, but compiler build, workspace build, and
  release packing must be serialized or moved to isolated output roots before
  the harness may overlap them.
- Gate 6 remains `in-progress`. The next coherent migration is `einsum`,
  replacing the final public TensorProxy callback with typed contraction
  parsing, shape/dtype inference, VJP, vmap/export, resource, and backend
  decisions.

### 2026-07-22 — Gate 6 typed general `torch.einsum`

- Replaced the public `CUSTOM` callback and eager Grad's restricted path with
  one bounded string-equation contraction contract. JIT and Grad share exact
  fixtures for explicit/implicit output, arbitrary admitted arity,
  repeated-label diagonals, label broadcasting, different-rank ellipses,
  PyTorch ellipsis reduction, scalars, uppercase ordering, promotion,
  ownership, gradients, hostile input, and resource refusal.
- Construction parses and infers solely from metadata. It admits 1 through 64
  operands, equations through 4,096 UTF-8 bytes, rank through 32, at most 52
  resolved NumPy labels, and the closed bool/real-numeric dtype set. Output,
  conservative output/cast/contraction/largest-gradient workspace, and
  contraction-domain work are bounded before allocation. CPU uses one numeric-label greedy
  contraction; float16 accumulates in float32 before an owning float16 store.
- Closure and symbolic VJP handle arbitrary admitted operands, target labels
  absent from derivative inputs, broadcast reduction, and diagonal scatter.
  Eager Grad snapshots operands before forward execution. Vmap owns a distinct
  leading batch prefix and broadcasts captured inputs, preserving mapped axes
  even when the user equation reduces its ellipsis. ONNX opset 17 resolves the
  complete equation to lower-case labels, casts mixed inputs, and refuses bool
  or more than 26 resolved exporter labels. Tensor-plan/WebGPU validate and
  refuse pending canonical contraction scheduling/lowering.
- ADR-0023 retires `tensor.einsum` and `jit.custom.einsum.v0`. The frozen
  inventory narrows from 12 constructor calls/15 operations to 11/14; the
  executable registry contains 25 typed retirements and preserves the original
  39-ID partition. No public tensor callback remains in the former
  forward-only NumPy policy.
- Final verification passes the semantic architecture guard; focused JIT
  conformance passes 4 files/47 tests and focused Grad conformance passes 2
  files/13 tests. Full JIT integration passes 44 files/304 tests in 94.12
  seconds and full Grad integration passes 49 files/340 tests in 101.43
  seconds. JIT and Grad build/codegen, strict typecheck, lint, and unit suites
  pass. The compiler gate passes 96 files/1,602 tests. Release verification
  passes 19 Python archive tests, 35 Node security tests, and all packed/fresh
  consumers. The seven-package Node 25.9.0 workspace build passes in 9.90
  seconds.
- Gate 6 remains `in-progress`. The next smallest pure callback migration is
  `l1_loss`; the other loss operations can reuse its closed reduction and
  dtype seams without being grouped into one opaque capability commit.

### 2026-07-22 — Gate 6 typed `torch.nn.functional.l1_loss`

- Replaced the shared `CUSTOM` callback and eager Grad's float32-only,
  input-gradient-only path with one strict same-shape floating loss contract.
  JIT and Grad share exact fixtures for `none`/`sum`/`mean`, scalars, empty
  tensors, mixed float16/32/64 promotion, output ownership, both input and
  target derivatives, equality zero subgradient, hostile arguments, and
  resource refusal.
- Construction derives shape and dtype from metadata and never executes NumPy.
  Rank is capped at 32, each extent at 256 Mi elements, output and conservative
  workspace at 256 MiB, and element visits at `2^28`; zero extents cannot hide
  hostile sibling dimensions. Float16 differences and reductions compute in
  float32 before an owning half store. Empty none/sum/mean behavior is explicit.
- JIT closure autograd derives the signed difference from immutable registered
  buffers while eager Grad snapshots it at forward construction. Symbolic
  autograd emits typed `L1_LOSS_VJP` for both operands, normalizes target zero,
  and returns source-dtype cotangents. Vmap retains per-example reductions across a leading
  batch prefix. ONNX opset 17 emits float32 half-compute and other promotion
  casts, `Sub`, `Abs`, the exact non-batch reduction, and an output cast.
  Tensor-plan/WebGPU validate and refuse pending a
  canonical loss-reduction lowering.
- ADR-0024 retires `jit.custom.l1-loss.v0`. The shared helper remains one
  constructor site for three other loss operations, so the frozen inventory
  remains at 11 constructor calls while narrowing from 14 to 13 operations.
  The executable registry contains 26 typed retirements and preserves the
  original 39-ID partition.
- Focused integration passes 5 JIT files/55 tests and 2 Grad files/11 tests;
  the JIT surface unit passes 8 tests. Full JIT integration passes 45
  files/307 tests in 98.82 seconds and full Grad integration passes 50
  files/342 tests in 103.59 seconds. Both packages pass codegen/build, strict
  typecheck, lint, and unit suites. The compiler gate passes 96 files/1,602
  tests. Release verification passes 19 Python archive tests, 35 Node security
  tests, and every packed/fresh consumer. The seven-package Node 25.9.0 build
  passes in 9.83 seconds.
- Gate 6 remains `in-progress`. The next smallest callback using the same
  reduction/resource seam is `smooth_l1_loss`; it remains a separate
  capability commit with its own beta, derivative, transform, and export
  contract.

### 2026-07-22 — Gate 6 typed `torch.nn.functional.smooth_l1_loss`

- Replaced the final shared piecewise-loss `CUSTOM` caller and Grad's
  float32/input-only path with typed `SMOOTH_L1_LOSS` and internal
  `SMOOTH_L1_LOSS_VJP`. One cross-package fixture covers exact branch-boundary
  values, beta `1`, `0.5`, and `0`, all reductions, scalar/empty inputs,
  mixed float16/32/64 promotion, both gradients, aliases, ownership, eager
  mutation snapshots, and hostile/resource inputs.
- Beta accepts only exact real scalar types, normalizes negative zero, and is
  finite and non-negative. A positive value that becomes zero or infinity in
  the promoted compute dtype fails before UOp construction or NumPy work.
  Zero beta takes the exact L1 path and never divides. Positive beta uses the
  quadratic branch only for strict `abs(difference) < beta`; the boundary is
  linear.
- JIT L1 and Smooth L1 now share one metadata-only geometry, exact runtime
  array validator, reduction, upstream-cotangent expansion, and conservative
  workspace calculation. Rank is capped at 32; output/workspace and total
  element visits are capped at `2^28`. Smooth L1 uses a conservative 32-visit
  factor and counts casts, output, three compute buffers, one mask, and both
  retained source-dtype gradients. The refactor also corrects L1 workspace to
  count the retained upstream compute buffer.
- Closure and symbolic autograd return both source-dtype cotangents and
  normalize target zero. Nested vmap increments the owned batch rank without
  reducing mapped axes. ONNX opset 17 emits promoted `Sub`/`Abs`, strict
  `Less`, quadratic `Mul`/`Div`, linear `Sub`, `Where`, exact reduction, and
  output cast; zero beta emits the smaller L1 form. Tensor-plan/WebGPU validate
  and refuse pending a canonical piecewise loss lowering.
- ADR-0025 retires `jit.custom.smooth-l1-loss.v0`. The shared helper remains
  one constructor site for BCE and KL divergence, so the inventory remains at
  11 constructor calls while narrowing from 13 to 12 operations. The typed
  registry contains 27 retirements and preserves the original 39-ID
  partition. Gate 0 now uses still-opaque BCE for representative refusals.
- Focused JIT Smooth L1/L1/Gate-0/registry integration passes 4 files/13 tests,
  and focused Grad Smooth L1/L1 integration passes 2 files/4 tests. Full JIT
  integration passes 46 files/310 tests in 110.93 seconds and full Grad
  integration passes 51 files/344 tests in 115.08 seconds. Both packages pass
  codegen/build, strict typecheck, lint, and unit suites. The architecture
  guard and compiler 96-file/1,602-test gate pass. Release verification passes
  19 Python archive tests, 35 Node security tests, and all packed/fresh
  consumers. The seven-package Node 25.9.0 build passes in 10.05 seconds.
- One deliberately over-parallel verification attempt exposed the known shared
  output boundary: release packing raced the compiler's clean rebuild and saw
  a transient missing source map. After the compiler lane completed, the
  unchanged release gate passed. Compiler build and release packing therefore
  remain serialized unless run in isolated worktrees; JIT, Grad, and compiler
  test lanes remain independently parallelizable.
- Gate 6 remains `in-progress`. The next pure callback sharing this reduction
  family is `binary_cross_entropy`; it requires its own probability-domain,
  numerical-policy, derivative, transform, and export contract.

### 2026-07-22 — Gate 6 typed `torch.nn.functional.binary_cross_entropy`

- Replaced the shared probability-loss `CUSTOM` caller and Grad's clipped,
  input-only derivative path with typed `BINARY_CROSS_ENTROPY` and internal
  `BINARY_CROSS_ENTROPY_VJP`. One cross-package fixture covers interior and
  endpoint values, all reductions, scalar/empty inputs, mixed float16/32/64
  promotion, both gradients, aliases, ownership, eager mutation snapshots,
  invalid probability domains, and hostile resource requests.
- Exact runtime input and target arrays must be finite and within `[0, 1]`
  before logarithms or result allocation. Forward clamps `log(p)` and
  `log(1-p)` to `-100`, not `p`, so valid contradictory endpoints produce
  loss 100. The input derivative independently uses
  `(p-target)/max((1-p)*p, 1e-12)`; the target derivative is the unclamped
  `log(1-p)-log(p)` and retains signed infinity at valid endpoints.
- The shared loss geometry bounds rank at 32 and caps output, conservative
  workspace, and total element visits at `2^28`. BCE's 48-visit profile counts
  domain scans, logarithms, reductions, input casts, output, four compute
  buffers, one mask, and both retained source-dtype gradients. Zero extents
  cannot hide hostile sibling capacity.
- Closure and symbolic autograd return both source-dtype cotangents; eager Grad
  snapshots both derivatives. Nested vmap preserves per-example reductions.
  ONNX opset 17 refuses the profile because an arithmetic decomposition cannot
  retain fail-closed runtime probability-domain validation. Tensor-plan and
  WebGPU validate and refuse pending a canonical probability-loss lowering.
- ADR-0026 retires `jit.custom.binary-cross-entropy.v0`. KL divergence retains
  the shared helper constructor, so the inventory remains at 11 constructor
  calls while narrowing from 12 to 11 operations. The typed registry contains
  28 retirements and preserves the original 39-ID partition. Gate 0 now uses
  still-opaque KL divergence for representative refusals.
- Focused JIT BCE/Gate-0/registry/IR/VJP integration passes 5 files/52 tests in
  10.63 seconds, and focused Grad BCE integration passes 1 file/2 tests in 2.03
  seconds. Full JIT integration passes 47 files/313 tests in 104.08 seconds and
  full Grad integration passes 52 files/346 tests in 107.40 seconds. Both
  packages pass codegen/build, strict typecheck, lint, and unit suites. The
  architecture guard and compiler 96-file/1,602-test gate pass. Release
  verification passes 19 Python archive tests, 35 Node security tests, and all
  packed/fresh consumers. The seven-package Node 25 build passes in 9.84
  seconds.
- Gate 6 remains `in-progress`. The next adjacent loss callback is
  `binary_cross_entropy_with_logits`; it requires a stable logits-domain
  forward, exact optional weighting surface, both derivatives, transform,
  export, and backend contract rather than reuse of probability-domain BCE.

### 2026-07-22 — Gate 6 typed `torch.nn.functional.binary_cross_entropy_with_logits`

- Replaced the dedicated logits-loss `CUSTOM` constructor and Grad's
  mean-only, input-only path with typed `BINARY_CROSS_ENTROPY_WITH_LOGITS` and
  internal `BINARY_CROSS_ENTROPY_WITH_LOGITS_VJP`. One cross-package fixture
  covers stable finite extremes at `+/-1000`, all reductions, scalar/empty
  inputs, mixed float16/32/64 promotion, both gradients, aliases, ownership,
  eager mutation snapshots, unsupported weighting, and hostile resources.
- Forward evaluates `(1-target)*logits + softplus(-logits)` through
  `max(-logits,0)+log1p(exp(-abs(logits)))`; it never evaluates
  `exp(logits)`. The logits derivative derives a stable sigmoid from
  `exp(-abs(logits))` and subtracts target. The target derivative is exactly
  `-logits`. PyTorch's native logits loss does not enforce probability target
  bounds, so this unweighted profile preserves the native algebra instead of
  claiming probability-BCE validation.
- The shared loss geometry bounds rank at 32 and caps output, conservative
  workspace, and total element visits at `2^28`. The 36-visit profile counts
  stable forward, both derivatives, reductions, input casts, output, four
  compute buffers, one mask, and both retained source-dtype gradients. Zero
  extents cannot hide hostile sibling capacity.
- Closure and symbolic autograd return both source-dtype cotangents; eager Grad
  snapshots both derivatives. Nested vmap preserves per-example reductions.
  ONNX opset 17 emits promoted `Neg`/`Softplus`, `Sub`/`Mul`/`Add`, exact
  non-batch reduction, and output cast. Tensor-plan and WebGPU validate and
  refuse pending canonical stable loss lowering.
- Optional `weight`, `pos_weight`, `size_average`, and `reduce` remain explicit
  signature failures. The prior JIT surface never admitted them; implementing
  their broadcast and weighted derivative semantics requires a separately
  versioned extension rather than silently ignoring them. Grad now exposes the
  exact `binary_cross_entropy_with_logits` alias and all three reductions.
- ADR-0027 retires `jit.custom.binary-cross-entropy-with-logits.v0` and its
  unique constructor site. The inventory narrows from 11/11 to 10 constructor
  calls/10 operations; the typed registry contains 29 retirements and
  preserves the original 39-ID partition. Gate 0 continues to use still-opaque
  KL divergence for representative refusals.
- Focused JIT BCE-with-logits/Gate-0/registry/IR/VJP integration passes 5
  files/52 tests in 10.65 seconds, focused Grad BCE-with-logits/compatibility
  integration passes 2 files/20 tests in 3.98 seconds, full JIT integration
  passes 48 files/316 tests in 106.76 seconds, and full Grad integration passes
  53 files/348 tests in 109.45 seconds. The semantic architecture check, the
  96-file/1,602-test compiler gate, 19 Python archive tests, 35 Node release
  security tests, packed/fresh consumers, and the seven-package Node 25.9.0
  build all pass; the build completes in 9.83 seconds.
- Gate 6 remains `in-progress`. The next pure callback using the remaining
  shared elementwise-loss helper is `kl_div`; it requires exact `log_target`,
  reduction including `batchmean`, zero-target, both-derivative, transform,
  export, and backend semantics.

### 2026-07-22 — Gate 6 typed `torch.nn.functional.kl_div`

- Replaced the final shared elementwise-loss `CUSTOM` callback and Grad's
  input-only float32 path with typed `KL_DIV` and internal `KL_DIV_VJP`. One
  cross-package fixture covers probability and log targets, all four
  reductions, scalar and distinct empty geometries, mixed float16/32/64
  promotion, both gradients, public aliases, ownership, eager mutation
  snapshots, malformed arguments, mutated IR, and hostile resources.
- Probability targets evaluate native
  `xlogy(target,target)-target*input`, preserving an exact zero forward
  contribution and the current native NaN target derivative at zero. Log
  targets evaluate `exp(target)*(target-input)`. Neither path substitutes a
  clipped target or finite subgradient. Both derivatives preserve their source
  dtype.
- `batchmean` sums and divides by the first user dimension for non-scalar
  input; scalar batchmean is scalar sum. A zero batch returns NaN while a
  nonzero batch with zero support returns zero. Transform-owned leading axes
  remain outside mean and batchmean denominators through nested vmap.
- The shared loss geometry bounds rank at 32 and caps output, conservative
  workspace, and total visits at `2^28`. The 48-visit profile counts both
  target representations, both derivatives, reductions, casts, masks, output,
  retained cotangents, and compute buffers. Zero extents cannot hide hostile
  sibling capacity.
- Closure and symbolic autograd return both cotangents; eager Grad snapshots
  both derivatives. ONNX opset 17 emits promoted exact decompositions for both
  target modes, explicit zero-target `Equal`/`Where`, exact batchmean/scalar
  reduction, and output casts. Tensor-plan and WebGPU validate and refuse
  pending canonical loss lowering.
- `log_target` must be an exact bool. Deprecated `size_average` and `reduce`
  remain explicit signature failures because the prior surface never admitted
  their reduction precedence. Grad exposes both `kl_div_loss` and the exact
  `kl_div` alias.
- ADR-0028 retires `jit.custom.kl-div.v0` and deletes the now-unused shared
  callback helper. The inventory narrows from 10/10 to 9 constructor calls/9
  operations; the typed registry contains 30 retirements and preserves the
  original 39-ID partition. Gate 0 now uses still-opaque `nll_loss` for its
  representative refusal.
- Focused JIT KL/Gate-0/registry/IR/VJP integration passes 5 files/52 tests in
  10.64 seconds, and focused Grad KL/loss/compatibility integration passes 3
  files/29 tests in 5.80 seconds. Full JIT integration passes 49 files/319
  tests in 84.02 seconds and full Grad integration passes 54 files/350 tests
  in 119.08 seconds. Both packages pass codegen/build, strict typecheck, lint,
  and unit suites. The semantic architecture check and compiler
  96-file/1,602-test gate pass. Release verification passes 19 Python archive
  tests, 35 Node security tests, and all packed/fresh consumers. The
  seven-package Node 25.9.0 build passes in 9.77 seconds.
- One JIT run was invalidated when the host wall clock jumped forward during
  Pyodide initialization and Vitest fired the 120-second hook timeout. The
  isolated expand fixture passed immediately, followed by the complete
  49-file rerun above; no code or fixture change was required.
- One attempted concurrent compiler/release run was invalid: the compiler
  build's intentional `dist` clean raced the packed-consumer install and
  produced a missing compiler `dist/index.js`. The unchanged release gate
  passed after the compiler gate completed. These two shared-output gates must
  remain serial unless isolated build roots are introduced; independent JIT
  and Grad integration lanes remain safe to run concurrently.
- Gate 6 remains `in-progress`. The next adjacent typed loss migration is
  `nll_loss`, followed by its coupled `cross_entropy` consumer.

### 2026-07-23 — Gate 6 typed `torch.nn.functional.nll_loss`

- Replaced the dedicated NLL `CUSTOM` callback and Grad's two-dimensional
  float32-only path with variadic typed `NLL_LOSS` and internal
  `NLL_LOSS_VJP`. One cross-package fixture covers unbatched and spatial
  geometry, exact int64 target selection, optional class weights,
  `ignore_index`, all reductions, legacy reduction precedence, mixed
  float16/32/64, ownership, empty and all-ignored behavior, eager mutation
  snapshots, malformed runtime state, and hostile resource shapes.
- Every non-ignored target is range checked before indexed access. Weighted
  mean divides by the selected class-weight sum; unweighted mean divides by
  valid target count. CPU half compute uses float32 and returns an owning
  source-dtype result. Target and weight are explicitly non-differentiable;
  closure/symbolic VJP write only the selected-class input cotangent.
- Vmap moves the class axis past transform-owned leading dimensions, keeps
  reductions per example, and accepts captured or mapped weight through nested
  maps. Unmapped ONNX opset 17 emits `NegativeLogLikelihoodLoss`; rank-one
  input uses exact `Unsqueeze`/`Squeeze`. Mapped export, tensor-plan, and
  WebGPU validate then refuse pending canonical indexed-loss lowering.
- Rank, extents, output, conservative workspace, and 32-visit work are bounded
  before numerical execution; zero dimensions cannot hide hostile sibling
  capacity. `nn.NLLLoss` validates and materializes its bounded static weight
  as a module buffer, reconstructing the leaf in the input session.
- Trace caching now includes session identity in each input signature and
  bypasses module trees with mutable buffers. This prevents same-shaped
  cross-session buffer reuse and stale buffer snapshots while retaining hits
  for immutable same-session inference graphs.
- ADR-0029 retires `jit.custom.nll-loss.v0`. The inventory narrows from 9/9
  to 8 constructor calls/8 operations; the typed registry contains 31
  retirements and preserves the original 39-ID partition. Gate 0 now uses
  still-opaque `cross_entropy` for representative callback refusals.
- Focused NLL/Gate-0/registry/IR/trace integration passes 5 files/48 tests.
  Full JIT integration passes 50 files/324 tests in 85.65 seconds and full
  Grad integration passes 55 files/352 tests in 90.96 seconds. Both packages
  pass codegen, build, strict typecheck, lint, and unit suites. The semantic
  architecture check and compiler 96-file/1,602-test gate pass. Release
  verification passes 19 Python archive tests, 35 Node security tests, and all
  packed/fresh consumers. The seven-package Node 25.9.0 build passes in 8.10
  seconds.
- The first complete compiler run exposed a stale architecture-test mutation
  fixture that still targeted the retired NLL constructor site. The fixture
  now mutates still-opaque `cross_entropy`; the focused architecture test and
  complete compiler rerun pass.
- Gate 6 remains `in-progress`. The next coupled loss migration is typed
  `cross_entropy`, reusing the admitted class-index semantics without hiding
  log-softmax or label-smoothing behavior in a callback.

### 2026-07-23 — Gate 6 typed `torch.nn.functional.cross_entropy`

- Replaced the dedicated cross-entropy `CUSTOM` callback and Grad's
  two-dimensional float32-only path with variadic typed `CROSS_ENTROPY` and
  internal `CROSS_ENTROPY_VJP`. One cross-package fixture covers unbatched and
  spatial geometry, index and probability targets, optional class weights,
  `ignore_index`, label smoothing, all reductions, legacy reduction
  precedence, float16/32/64, ownership, empty and all-ignored behavior, eager
  mutation snapshots, malformed runtime state, and hostile resource shapes.
- The forward path uses max-shifted log-softmax and promoted float32 half
  compute. Index-target mean divides by selected class-weight mass;
  probability-target mean divides by the number of positions. Smoothing
  follows the weighted class-uniform term, and discrete target/weight inputs
  remain explicitly nondifferentiable.
- Closure and symbolic VJP preserve the logits source dtype and propagate the
  exact probability-target derivative when the target is floating. Vmap moves
  the class axis past transform-owned leading dimensions and accepts captured
  or mapped targets and weights under nested maps.
- Unmapped zero-smoothing index-target ONNX opset 17 emits
  `SoftmaxCrossEntropyLoss`. Probability targets, label smoothing, mapped
  export, tensor-plan, and WebGPU validate then refuse where opset or canonical
  portable lowering cannot retain the typed contract.
- Rank, extents, output, conservative workspace, and 64-visit work are bounded
  before numerical execution. `nn.CrossEntropyLoss` validates and materializes
  its bounded static weight as a module buffer and reconstructs the leaf in
  the input session.
- The registry is now 32,238 bytes, leaving only 530 bytes below its former
  32-KiB envelope. The package loader and independent architecture gate move
  together to a still-bounded 64-KiB ceiling, with the hostile oversized-byte
  fixture moved to the exact new boundary.
- Full symbolic MLP tracing exposed a pre-existing ReLU graph-integrity defect:
  its hand-built three-input `WHERE` node retained only one closure operand.
  ReLU now composes the canonical typed comparison and `where` operations, so
  closure and symbolic operand topology agree.
- ADR-0030 retires `jit.custom.cross-entropy.v0`. The inventory narrows from
  8/8 to 7 constructor calls/7 operations; the typed registry contains 32
  retirements and preserves the original 39-ID partition. Gate 0 now uses
  still-opaque `interpolate` for representative callback refusals.
- Focused cross-entropy, Gate-0, registry, IR, VJP, MLP/fusion, and semantic
  architecture verification passes. Full JIT integration passes 51 files/327
  tests in 100.50 seconds and full Grad integration passes 56 files/354 tests
  in 95.24 seconds. Both packages pass codegen, build, strict typecheck, lint,
  and unit suites. The architecture check and compiler 96-file/1,602-test gate
  pass. Release verification passes 19 Python archive tests, 35 Node security
  tests, and all packed/fresh consumers. The seven-package Node 25.9.0 build
  passes in 9.73 seconds.
- The first complete Grad run exposed two existing `CrossEntropyLoss` callers
  that pass NumPy index arrays. The strict new functional API remains
  Tensor-only, while the compatibility module delegates through the bounded
  legacy wrapper that converts the array to an int64 Tensor; the focused
  20-test rerun and complete 354-test rerun pass.
- Gate 6 remains `in-progress`. The next opaque migration is `dropout`, whose
  stateful mask/replay contract must be typed before any backend claim.

### 2026-07-23 — Gate 6 typed `torch.nn.functional.dropout`

- Replaced the mutable dropout `CUSTOM` callback with public typed `DROPOUT`
  and internal `DROPOUT_VJP`. One immutable per-operation seed now regenerates
  the exact mask for repeated CPU realization, closure backward, symbolic VJP,
  functional gradient, and checkpoint recomputation.
- Validation precedes every branch. Evaluation, zero probability, and empty
  input return the exact input without consuming RNG. Probability one is an
  owning dtype-preserving drop-all operation. Active stochastic training is
  limited to float16/32/64 and preserves exact output and gradient dtype.
  `inplace=True` refuses until typed mutation and alias semantics exist.
- Rank, per-axis extents, output, projected visits, and complete output/RNG/mask
  workspace are bounded before key consumption or allocation. Forward and VJP
  revalidate exact runtime ndarray, shape, dtype, and immutable arguments.
- Deterministic drop-all maps across a leading batch axis. Stochastic vmap
  refuses until the public transform owns an explicit same/different randomness
  policy. ONNX inference export and tensor-plan/WebGPU validate then refuse
  because those routes cannot preserve the keyed training contract.
- Trace caching now excludes `CUSTOM`, `RANDOM`, and `DROPOUT` graphs, while
  identity dropout remains cacheable because it emits no operation. Checkpoint
  defaults to `preserve_rng_state=True` and accepts either exact boolean because
  stochastic replay belongs to immutable IR rather than ambient mutable RNG.
- Grad consumes the shared conformance profile, admits `p=1`, preserves
  float16/32/64 values and cotangents, snapshots the forward mask, and delegates
  `nn.Dropout` to the functional implementation.
- ADR-0031 retires `jit.custom.dropout.v0`. The inventory narrows from 7/7 to
  6 constructor calls/6 operations; the typed registry contains 33 retirements
  and preserves the original 39-ID partition. Gate 0 now records only the
  remaining BatchNorm replay defect in its stateful callback case.
- Focused dropout, checkpoint, trace-cache, Gate-0, registry, IR, VJP, and Grad
  integration tests pass. Full JIT codegen/build/typecheck/browser-typecheck/
  lint/unit/integration passes 52 files/332 integration tests in 109.41
  seconds. Full Grad codegen/build/typecheck/lint/unit/integration passes 57
  files/356 integration tests in 109.30 seconds. The architecture mutation
  suite passes 24 tests, and `verify:compiler` passes its 96-file/1,602-test
  compiler suite plus the parallel browser-build-plan and Docker-shell lanes.
- The Node 25.9.0 release bodies pass workspace build/typecheck/unit/lint,
  19 Python archive tests, 35 Node release-security tests, required WebGPU
  kernel/layout/view-copy lanes, all four real-world CUDA compile audits, and
  both source and packed 159-case browser fixture/corpus lanes with zero skips.
  The first final dogfood invocation exposed an invalid config-only isolated
  pnpm workspace; D-149 fixes it, and the exact dogfood node 5-file/44-test and
  browser 16-file/98-test suites then pass with one expected failure. The
  complete release wrapper was not restarted after this terminal harness-only
  correction because all preceding gate bodies had already passed.
- Gate 6 remains `in-progress`. The next stateful callback migration is
  `BatchNorm1d`; runtime/profile UI consumption and remaining Grad view/dtype
  convergence also remain open.

### 2026-07-23 — Gate 6 typed `nn.BatchNorm1d`

- Replaced the stateful `CUSTOM` callback with public typed
  `BATCH_NORM_1D` and internal `BATCH_NORM_1D_STATS_UPDATE` /
  `BATCH_NORM_1D_VJP`.
- The closed v1 profile accepts exact float32 `(N,C)` or `(N,C,L)` input,
  exact float32 affine/state values, bounded positive channels, finite
  non-negative epsilon, `None` or finite `[0,1]` momentum, and exact boolean
  flags. Batch-stat modes reject one or fewer samples per channel.
- Training normalizes with biased variance and updates persistent variance with
  the unbiased estimator. Running mean, running variance, and scalar int64
  `num_batches_tracked` are registered buffers. `momentum=None` uses cumulative
  averaging, and reset/load/state-dict paths retain buffer identity.
- Each tracked call reserves the next sequence on one BufferTable-owned stream
  bound to the exact three target buffers. Reserved/applied watermarks make
  replay metadata constant-memory per module instead of per forward call. The
  realizer validates every target identity, shape, dtype, session, and sequence
  before committing the three updates in place. Predecessor edges preserve
  construction order; replay, symbolic backward, functional gradients, and
  checkpoint clones cannot apply the same effect twice.
- Running-state buffer names use minted module tokens rather than recyclable
  Python object IDs, so collected short-lived modules cannot collide with
  retained buffers in a long-lived Pyodide session.
- Tracked evaluation synchronizes pending state and owns an immutable snapshot.
  Forward, closure VJP, symbolic VJP, and Grad share the same batch/running-stat
  derivative. Checkpoint rewriting now remaps typed `vjp_of` authority to its
  cloned forward node rather than mixing cloned operands with an original
  source.
- Vmap, ONNX, tensor-plan, and WebGPU validate and refuse until they own
  batch-axis, stable export-snapshot, normalization, and ordered device-state
  semantics. Trace caching excludes the state-update opcode.
- Grad registers matching state buffers, persistent count, cumulative
  averaging, unbiased running variance, in-place identity preservation, exact
  float32 validation, and immutable affine/statistic backward snapshots.
- ADR-0032 retires `jit.custom.batch-norm-1d.v0`. The inventory narrows from
  6/6 to 5 constructor calls/5 operations; the executable registry contains 34
  retirements and preserves the original 39-ID partition. The obsolete Gate 0
  stateful-replay fixture is deleted.
- Focused JIT and Grad BatchNorm conformance, JIT module/state/checkpoint,
  lifecycle, wrong-target, architecture, registry, IR, and VJP checks pass in
  2–4 seconds. The seven-package build passes in 11.22 seconds on Node 25.9.0.
  JIT build/typecheck/browser-typecheck/lint and 4-file/24-test unit gates pass;
  its isolated full integration run passes 53 files/335 tests in 130.41 seconds.
  Grad build/typecheck/lint and 2-file/30-test unit gates pass; full integration
  passes 58 files/359 tests in 165.96 seconds. The architecture mutation suite
  passes 24 tests, and `verify:compiler` passes its 96-file/1,602-test compiler
  suite plus parallel browser-build-plan and Docker-shell lanes. The preceding
  dropout commit retains the latest complete release-body evidence.
- Gate 6 remains `in-progress`. The next opaque CPU callback is
  `torch.nn.functional.interpolate`; the three constructor-only/accelerator
  surfaces and intentional user-kernel boundary follow separately.

### 2026-07-23 — Gate 6 typed `torch.nn.functional.interpolate`

- Replaced the final framework-owned NumPy `CUSTOM` callback with public typed
  `INTERPOLATE_2D` and internal `INTERPOLATE_2D_VJP`.
- The closed v1 profile accepts exact rank-four `(N,C,H,W)` float16/32/64
  input, nearest or bilinear mode, and exactly one bounded positive scalar/pair
  size or scale. Scale-derived extents use floor. Nearest requires
  `align_corners=None`; bilinear owns half-pixel or aligned-corner geometry.
  Scale recomputation is explicit, and antialiasing remains a loud refusal.
- CPU forward uses vectorized gather and returns a fresh declared-dtype array.
  The transpose uses bounded flattened scatter-add instead of nested
  output-pixel Python loops; float16 computes in float32 and casts back.
  Runtime arrays, input/output extents and bytes, projected element visits, and
  conservative workspace are validated before execution.
- Closure and symbolic VJP consume the same immutable geometry. Functional
  gradients, leading-axis vmap, vmap of symbolic gradients with a captured
  cotangent, checkpoint replay, dtype preservation, ownership, and non-integral
  scale recomputation are covered by the shared conformance fixture.
- ONNX opset 17 emits `Resize`: asymmetric/floor nearest, half-pixel bilinear,
  or aligned-corner bilinear. Explicit non-recomputed scales use the scales
  input; size and recomputed requests use sizes. VJP export refuses.
  Tensor-plan and WebGPU validate then explicitly refuse until canonical
  spatial-resampling lowering and kernels exist.
- Grad now shares the strict geometry, dtype, ownership, resource, forward, and
  transpose contract. Its previous forced-float32 results and nested nearest/
  bilinear backward loops are removed.
- ADR-0033 retires `jit.custom.interpolate.v0`,
  `functional.interpolate`, and the last legacy NumPy-callback policy. The
  current opaque inventory narrows from 5/5 to 4 constructor calls/4
  operations, and the executable registry contains 35 retirements while
  preserving the original 39-ID partition. Gate 0 retains one executable case
  for the two explicit accelerator routes and two constructor-only surfaces.
- Focused interpolation, registry, Gate 0, IR, and VJP integration passes JIT
  5 files/48 tests; the new Grad contract plus the pre-existing interpolation
  suite passes 2 files/13 tests. The semantic architecture guard passes.
  Node 25.9.0 builds all seven packages in 10.86 seconds. JIT build, typecheck,
  browser typecheck, lint, and 4-file/24-test unit gates pass; full integration
  passes 54 files/335 tests in 144.93 seconds. Grad build, typecheck, lint, and
  2-file/30-test unit gates pass; full integration passes 59 files/361 tests in
  141.75 seconds. Those integration suites and `verify:compiler` ran
  concurrently, completing the expensive feedback wave in roughly 2.5 minutes
  wall-clock rather than serially. The compiler gate passes its
  96-file/1,602-test suite plus parallel 46-file/250-test build-plan and
  2-file/96-test Docker-shell lanes; its nine native-environment skips remain
  explicitly non-evidence for native execution.
- Gate 6 remains `in-progress`, but no framework-owned NumPy callback remains.
  The remaining opaque identities are `flash_attention`, `transformer_block`,
  `webnn_matmul`, and the intentional `user` kernel extension. Next resolve the
  three advertised compatibility surfaces through typed semantics, accurate
  removal, or an explicitly non-portable boundary; then close the Grad
  view/dtype and generated runtime/profile support-table exit criteria.

### 2026-07-23 — Gate 6 retirement of constructor-only WebNN matmul

- Removed `bg.experimental.webnn.matmul` instead of relabeling ordinary
  `MATMUL` as WebNN execution. The old constructor created
  `CUSTOM(op="webnn_matmul")`, disconnected autograd, omitted dtype/fallback
  contracts, and was rejected by every CPU, WebGPU, and WebNN route.
- `bg.experimental.webnn.is_available` remains a narrow `navigator.ml`
  presence probe. It does not claim context creation, operation support,
  profitable dispatch, or device execution. PRD-011 remains a future
  graph-level backend consuming typed IR through explicit partitioning,
  lowering, fallback, and capability evidence.
- The architecture freeze now preserves three disjoint classifications for the
  original 39 operation IDs: 35 typed retirements, 3 current opaque identities,
  and 1 removed unsupported surface. Mutation checks reject a missing,
  duplicate, overlapping, or non-namespaced removed-surface classification.
- ADR-0034 records the removal. The opaque inventory narrows from 4/4 to three
  constructor calls and three operations: `flash_attention`,
  `transformer_block`, and the intentional `user` kernel extension. The
  executable Gate 0 fixture no longer treats a non-realizable constructor as
  a shipped backend spike.
- Focused WebNN-surface and Gate 0 integration passes 2 files/11 tests in 4.94
  seconds. The compiler architecture mutation suite passes 1 file/24 tests in
  1.69 seconds. JIT codegen/build, TypeScript and browser typechecks, lint, and
  4-file/24-test unit gates pass concurrently; the semantic architecture and
  generated-source checks pass.
- Gate 6 remains `in-progress`. The remaining advertised opaque compatibility
  surfaces are `flash_attention` and `transformer_block`; Grad dtype/view and
  generated runtime/profile support consumers remain open.

### 2026-07-23 — Gate 6 retirement of constructor-only transformer block

- Removed `bg.kernels.transformer_block` instead of retaining a call that
  constructed `CUSTOM(op="transformer_block")` with no CPU, autograd,
  tensor-plan, legacy-WebGPU, or other executable route.
- The removal does not silently substitute an ordinary primitive transformer
  sequence for the promised megakernel. PRD-012c remains a draft until typed
  graph-pattern recognition or an exact typed operation, code generation, and
  actual execution evidence exist.
- ADR-0035 records the decision. The original 39 operation IDs now partition
  exactly into 35 typed retirements, 2 current opaque identities, and 2 removed
  unsupported surfaces. The current identities are the legacy
  `flash_attention` accelerator route and the intentional `user` kernel
  extension.
- Focused Gate 0 and deferred-PRD integration pass 2 files/10 tests in 4.87
  seconds. The compiler architecture mutation suite passes 1 file/24 tests in
  1.81 seconds. JIT codegen/build, TypeScript and browser typechecks, lint, and
  4-file/24-test unit gates pass concurrently; the semantic architecture and
  generated-source checks pass.
- Gate 6 remains `in-progress`. `flash_attention` is the only remaining
  advertised opaque compatibility surface; Grad dtype/view and generated
  runtime/profile support consumers remain open.

### 2026-07-23 — Gate 6 typed attention-forward boundary

- Added typed `ATTENTION_FORWARD` IR and the accurately named
  `bg.kernels.attention_forward` entrypoint. The existing
  `bg.kernels.flash_attention` API remains only as a compatibility alias that
  constructs the same typed node; it no longer enters `CUSTOM`.
- The exact v1 JIT profile admits positive dense rank-4 float32 `(B,H,S,D)`
  Q/K/V tensors with matching batch/head dimensions, matching K/V sequence
  length, identical depth, depth at most 64, canonical float32
  inverse-square-root scaling, finite runtime values, and bounded
  output/work/workspace. CPU execution returns an owning stable NumPy result.
- Gradient-bearing inputs, masks, and custom scales fail at construction.
  Vmap, ONNX, and the portable tensor plan revalidate the typed contract and
  explicitly refuse until their own attention contracts exist. The legacy
  direct-WebGPU route uses the row-wise online-softmax bridge; it is not the
  independently verified Gate 5 block-tiled implementation and makes no
  FlashAttention-v2 claim.
- ADR-0036 records the decision. The original 39 operation IDs now partition
  exactly into 36 typed retirements, one current intentional opaque
  user-authored WGSL extension, and two removed unsupported surfaces.
  `_custom_kernel.py` is the sole remaining `OP_CUSTOM` constructor caller.
- Focused attention, registry, Gate 0, tensor-plan, and WebGPU integration
  passes 5 files/47 tests. Full JIT integration passes 55 files/336 tests in
  135.00 seconds. The seven-package Node 25.9.0 build completes in 15.49
  seconds; the compiler architecture mutation suite passes 1 file/24 tests,
  the full compiler gate passes 96 files/1,602 tests plus its parallel
  46-file/250-test build-plan and 2-file/96-test shell lanes, and the semantic
  architecture check passes.
- Gate 6 remains `in-progress`, but its advertised opaque-operation migration
  is complete. Remaining work is Grad dtype/view convergence, runtime/profile
  consumption of executable requirement and capability records, and generated
  cross-framework/platform support views.

### 2026-07-23 — Gate 6 explicit Grad bfloat16 refusal

- Removed the silent `bf16`/`bfloat16` to float32 alias. Both spellings now
  fail with `NotImplementedError` before allocation or conversion and state
  that real bfloat16 storage/conversion is unavailable.
- `torch.bfloat16` is the distinct token `"bfloat16"` rather than
  `"float32"`. `torch.tensor(..., dtype=torch.bfloat16)`,
  `Tensor(..., dtype="bf16")`, and `Tensor.to(torch.bfloat16)` converge on the
  same refusal.
- `nn.Linear` and `nn.Embedding` no longer ignore `dtype`. They accept only
  their exact float32 parameter storage/computation profile, reject other
  floating or non-floating parameter dtypes, and propagate the bfloat16
  refusal through model configuration.
- The Grad compatibility inventory advances to schema v2 with an exact
  unsupported-dtype map, revised behavior identities, frozen source
  definitions, and hostile mutation coverage. ADR-0037 records the baseline
  transition.
- Focused Gate 0 and reasoning-workshop integration passes 2 files/23 tests;
  full Grad integration passes 59 files/362 tests in 128.58 seconds; build,
  typecheck, lint, 30 unit tests, semantic architecture, and its 24-test
  mutation suite pass.
- Remaining Grad Gate 6 debt is view/materialization and conversion behavior:
  `contiguous`, detach, cross-dtype `to`, invalid dtype/device disambiguation,
  and NumPy interop. Real bfloat16 remains future work requiring distinct
  storage, rounding, serialization, and backend contracts.

### 2026-07-23 — Gate 6 truthful Grad contiguous materialization

- Replaced the unconditional `Tensor.contiguous()` no-op with two exact
  branches: already C-contiguous storage returns the same tensor; any other
  layout produces an independent owning C-order copy.
- The copy branch preserves shape and storage dtype, isolates mutations in
  both directions, and attaches an identity-gradient edge when the source
  requires gradients. The identity branch preserves the original graph.
- The versioned compatibility fixture separately proves the float32
  non-contiguous copy/backward path, the contiguous identity path, and
  float16 dtype-preserving materialization. Workshop
  `transpose().contiguous().view()` patterns pass through the real copy.
- ADR-0038 records the baseline transition. The Grad schema v2 behavior
  `grad.materialization.contiguous.v1` replaces the name-only
  `grad.materialization.contiguous-noop.v0`; source hashes and hostile mutation
  checks are repinned.
- Focused Gate 0 and workshop integration passes 3 files/48 tests; full Grad
  integration passes 59 files/362 tests in 137.16 seconds; build, typecheck,
  lint, 30 unit tests, semantic architecture, and its 24-test mutation suite
  pass.
- Remaining Grad convergence is detach, cross-dtype conversion, invalid
  dtype/device disambiguation, and NumPy interop. Runtime/profile generated
  support consumption remains the next Gate 6 domain after those eager
  contracts.

### 2026-07-23 — Gate 6 truthful Grad detach aliasing

- Replaced copying `Tensor.detach()` with a distinct tensor object sharing the
  exact source NumPy storage, dtype, shape, strides, and contiguity.
- The detached tensor is a leaf with `requires_grad=False` and no graph
  context. Bidirectional mutation proves storage sharing without implying an
  in-place `detach_()` surface.
- The versioned compatibility fixture separately proves float32 alias
  mutation, float16 dtype preservation, and non-contiguous stride
  preservation. ADR-0039 replaces `grad.materialization.detach-copy.v0` with
  the conformant `grad.view.detach.v1` baseline.
- Focused Gate 0 and workshop integration passes 4 files/70 tests in 11.07
  seconds; full Grad integration passes 59 files/362 tests in 130.11 seconds.
  Build, typecheck, lint, 30 unit tests, semantic architecture, and its
  24-test hostile mutation suite pass.
- Remaining Grad convergence is cross-dtype conversion, invalid dtype/device
  disambiguation, and NumPy interop. Runtime/profile generated support
  consumption follows those eager contracts.

### 2026-07-23 — Gate 6 differentiable Grad cross-dtype casts

- Cross-dtype `Tensor.to()` among float16, float32, and float64 now records a
  cast graph edge and converts the incoming VJP back to source storage dtype.
- Target storage remains independent and follows NumPy order-preserving
  conversion, including non-contiguous float16 input. `no_grad()` suppresses
  the edge, while bool/integer endpoint casts remain detached.
- ADR-0040 replaces the old detached behavior with separate conformant
  floating and nonfloating records. The fixture proves float32-to-float64
  VJP, float16 non-contiguous layout/VJP, no-grad behavior, and float-to-int
  truncation/detachment.
- Focused Gate 0 and reasoning-workshop integration passes 2 files/23 tests in
  5.00 seconds; full Grad integration passes 59 files/362 tests in 132.88
  seconds. Build, typecheck, lint, 30 unit tests, semantic architecture, and
  its 24 hostile mutation tests pass.
- Remaining Grad convergence is invalid dtype/device disambiguation and NumPy
  interop. Runtime/profile generated support consumption follows those eager
  contracts.

### 2026-07-23 — Gate 6 honest Grad tensor device requests

- Replaced the catch-all `Tensor.to()` parser with an exact CPU-device and
  dtype request boundary. Invalid strings, unsupported keywords, duplicate
  requests, excessive arguments, and ambiguous forms fail before allocation.
- `to("cpu")`, `to(device="cpu")`, and `cpu()` preserve identity. CPU plus a
  supported dtype composes with the frozen conversion/autograd contract.
- CUDA, MPS, XPU, Meta, indexed CPU, and other non-CPU requests reject before
  execution. `cuda()` also rejects instead of returning the CPU/Pyodide tensor
  and implying a transfer that never occurred.
- ADR-0041 adds exact device-placement/refusal records and freezes
  `Tensor.to`, `Tensor.cpu`, and `Tensor.cuda`. Focused Gate 0, torch
  compatibility, and reasoning-workshop integration passes 3 files/41 tests
  in 5.44 seconds; full Grad integration passes 59 files/362 tests in 102.62
  seconds. Build, typecheck, lint, 30 unit tests, semantic architecture, and
  its 24 hostile mutations pass.
- Remaining Grad device convergence is `torch.tensor(device=...)` and
  `nn.Module.to(...)`, followed by NumPy interop and runtime/profile generated
  support consumption.

### 2026-07-23 — Gate 6 closed Grad constructor/module placement

- `torch.tensor(device=...)` now admits only no device or exact CPU before
  applying its frozen dtype/default constructor semantics. Non-string and
  non-CPU requests reject before tensor allocation.
- The compatibility `nn.Module.to(...)` shim preserves identity only for no
  request or CPU. It rejects unavailable devices, dtype conversion,
  unsupported keywords, duplicates, and ambiguous forms without reading or
  mutating parameters.
- ADR-0042 extends the Grad architecture freeze to the limited compatibility
  source and adds exact constructor/module device records. Focused Gate 0,
  limited-pile, and reasoning-workshop integration passes 3 files/34 tests in
  5.26 seconds; full Grad integration passes 59 files/362 tests in 99.50
  seconds. Build, typecheck, lint, 30 unit tests, semantic architecture, and
  its 24 hostile mutations pass.
- Remaining eager Grad compatibility convergence is NumPy interop, followed
  by runtime/profile generated support consumption.

### 2026-07-23 — Gate 6 coherent Grad NumPy interop

- `from_numpy` now wraps writable arrays in the closed twelve-dtype eager storage
  set without copying. Exact ndarray identity, dtype, negative/non-contiguous
  strides, layout, and bidirectional mutation are preserved.
- Non-array, read-only, complex, and object inputs reject before wrapping.
  This avoids silent float32 conversion and unsafe object/read-only aliases.
- `.numpy()` and `np.asarray(tensor)` now share one owning `order="K"`
  snapshot implementation. Dtype requests remain explicit; mutations are
  isolated; `copy=False` cannot expose live tensor storage.
- ADR-0043 replaces the three inconsistent interop records with exact input
  alias and output snapshot contracts. Focused Gate 0, torch compatibility,
  reasoning, and LLM-workshop integration passes 4 files/67 tests in 6.97
  seconds; full Grad integration passes 59 files/362 tests in 101.14 seconds.
  Build, typecheck, lint, 30 unit tests, semantic architecture, and its 24
  hostile mutations pass.
- Remaining eager debt is NumPy-delegated dtype fallback,
  constructor-default classification, and owning expand materialization.
  Runtime/profile generated support consumption follows those contracts.

### 2026-07-23 — Gate 6 closed Grad eager dtype registry

- `_resolve_dtype` now admits strings only through the frozen
  BrowserGrad/PyTorch alias table. NumPy abbreviation strings and every unknown
  parser spelling reject before allocation.
- NumPy dtype objects and scalar types remain supported for the exact twelve
  eager storage dtypes, including uint16/32/64. Complex, object, structured,
  datetime, and other storage specifications reject before allocation.
- ADR-0044 replaces NumPy parser delegation with an executable package-owned
  registry contract and hostile source mutations for both string admission and
  physical-storage admission.
- Focused Gate 0, torch compatibility, reasoning, and LLM-workshop integration
  passes 4 files/67 tests in 7.30 seconds. Full Grad integration passes 59
  files/362 tests in 99.19 seconds; build, typecheck, lint, 30 unit tests,
  semantic architecture, and its 24 hostile mutations pass.
- Remaining eager debt is constructor-default classification and owning expand
  materialization. Runtime/profile generated support consumption follows those
  contracts.

### 2026-07-23 — Gate 6 converged Grad constructor semantics

- Direct `Tensor(data, dtype=None)` now has its own BrowserGrad-defined
  float32-default, conditionally aliasing contract rather than sharing one
  ambiguous record with the torch compatibility factory.
- `torch.tensor` now always owns a leaf copy. Python boolean/integer/floating
  data infer bool/int64/float32; admitted NumPy arrays/scalars and existing
  Tensor inputs preserve dtype unless an explicit admitted dtype is requested.
- Complex, object, string, structured, datetime, malformed `requires_grad`,
  and gradient-bearing nonfloating construction reject before allocation.
  Constructor copies cannot alias or inherit autograd history from their input.
- ADR-0045 replaces the combined compatibility-debt record with separate
  executable direct and torch constructor contracts.
- Focused Gate 0, torch compatibility, reasoning, and LLM-workshop integration
  passes 4 files/67 tests in 7.38 seconds. Full Grad integration passes 59
  files/362 tests in 99.60 seconds; build, typecheck, lint, 30 unit tests,
  semantic architecture, and its 24 hostile mutations pass.
- Remaining eager Grad compatibility debt is owning expand materialization.
  Runtime/profile generated support consumption follows that contract.

### 2026-07-23 — Gate 6 converged Grad expand view

- `Tensor.expand` now returns a storage-sharing view. Expanded singleton axes
  use zero strides; non-expanded and non-contiguous source strides, dtype, and
  source writeability are preserved.
- Source and result mutations propagate bidirectionally without an
  output-sized allocation. The existing unbroadcast VJP still reduces every
  expanded axis to the original input shape.
- Exact shape validation precedes stride construction, so the bounded
  `as_strided` view cannot address beyond the source. Float16, int32, uint64,
  and non-contiguous layout fixtures prove the physical contract.
- ADR-0046 replaces the final Grad compatibility-debt record with a
  PyTorch-compatible view contract. No Grad inventory behavior remains marked
  `compatibility-debt`.
- Focused Gate 0, expand conformance, torch compatibility, reasoning, and
  LLM-workshop integration passes 5 files/68 tests in 9.06 seconds. Full Grad
  integration passes 59 files/362 tests in 99.46 seconds; build, typecheck,
  lint, 30 unit tests, semantic architecture, and its 24 hostile mutations
  pass.
- Remaining Gate 6 work is generated runtime/profile support consumption and
  cross-framework/platform views from executable contracts.

### 2026-07-23 — Gate 6 provider-bound runtime requirements

- Semantic-core now exports a concrete `/requirement` protocol. Immutable
  definitions retain version, kind, owner, lifecycle, meaning, and an optional
  explicit semantic-capability link. Available environment resolutions require
  one provider ID, closed provider mode, and deterministic evidence IDs;
  unavailable records invent neither provider nor evidence.
- Runtime generates all 53 definitions from the architecture vocabulary and
  resolves every definition exactly once per environment. Definition presence
  alone cannot satisfy a profile; only explicit available providers enter the
  compatibility capability evaluator.
- Architecture validation compares the generated registry byte-for-byte with
  the vocabulary. The mutation suite rejects stale content, duplicate IDs, and
  malformed generation input. ADR-0047 records the compatibility bridge and
  explicitly keeps semantic lowering and backend support out of requirement
  resolution.
- Semantic-core build/typecheck/lint and its full suite pass; runtime
  build/typecheck/lint and 12 files/129 tests pass. Semantic architecture and
  its 25-test mutation suite pass. Focused registry/resolution tests complete
  in under one second. The release-package gate passes 19 hostile archive and
  35 Node security tests plus packed and fresh npm consumers of both new
  protocol surfaces.
- The first packed-gate attempt correctly failed because the local
  `node_modules` workspace links had not been refreshed after the lockfile-only
  dependency update. A pinned pnpm 10.34.5 offline install restored the declared
  runtime-to-semantic-core link; the unchanged gate then passed.
- Remaining Gate 6 work is direct resolution consumption by run-plan, report,
  handoff, matrix, and profile UI paths; program-specific capability/lowering
  records; and generated cross-framework/platform support views.

### 2026-07-23 — Gate 6 direct runtime resolution consumption

- `AssignmentReadinessEnvironment` preserves compatibility with bare capability
  environments while allowing provider-bound environments to enter every
  run-plan, preflight, benchmark-matrix, and JavaScript profile-runner path
  directly.
- Resolution inputs are fully validated and canonically reconstructed before
  evaluation. Each plan retains only the available/unavailable records
  referenced by that profile; reports, handoffs, external-runner requests, and
  benchmark rows carry the same frozen provider/mode/evidence records.
- Architecture now freezes six direct consumer signatures and five public
  resolution carriers. Mutations that restore a bare capability signature or
  replace a resolution array with strings fail locally. The same fast gate now
  compares every exported architecture helper and optional-parameter position
  with its checked TypeScript declaration.
- ADR-0048 records the migration and compatibility window. Focused direct
  consumer/typecheck/lint tests complete in 2.2 seconds; runtime passes 12
  files/130 tests, semantic architecture passes, and its mutation suite passes
  27 tests. The release-package gate passes 19 hostile archive and 35 Node
  security tests plus a fresh npm consumer that gives the provider-bound
  environment directly to `createAssignmentRunPlan`.
- Strict compiler typecheck initially exposed that the architecture declaration
  predated both the limited Grad compatibility parameter and the new registry
  helpers. The declaration is corrected, and the new parity mutation prevents
  that drift from recurring.
- Remaining Gate 6 work is artifact/program-specific capability and lowering
  records, followed by generated cross-framework/platform support views.

### 2026-07-23 — Gate 6 program-scoped lowering decisions

- Semantic-core now exports a concrete `/capability` protocol. Immutable
  capability/backend definitions contain only versioned static meaning and
  evidence IDs. Lowering decisions bind one canonical program ID or 64-hex
  semantic artifact hash to one registered capability, backend, execution
  tier, and support state.
- Positive decisions require a capability-owned preservation level.
  Conditional decisions retain at least one feature, limit, or runtime guard.
  Unsupported, unknown, and not-applicable decisions require a reason and
  cannot claim preservation.
- Runtime generates the one current semantic capability and three registered
  backend definitions byte-for-byte from the architecture vocabulary.
  `createProgramCapabilitySupportView()` accepts only actual decisions for one
  subject, rejects empty, unknown, or duplicate capability/backend pairs, and
  includes only referenced definitions.
- ADR-0049 records why static definitions, requirement availability, method
  presence, and evidence paths cannot become support claims. Architecture
  regeneration and its mutation suite reject stale or malformed program
  registries.
- Semantic-core build/typecheck/lint and 21 files/174 tests pass. Runtime
  build/typecheck/lint and 13 files/134 tests pass. Semantic architecture,
  strict compiler typecheck, and 28 architecture mutations pass. The first
  mutation invocation used root `pnpm exec`, where Vitest is intentionally
  absent; the corrected compiler-scoped command passed in 2.03 seconds.
- The release-package gate passes 19 hostile archive and 35 Node security
  tests plus packed and fresh consumers of `/capability` and the runtime
  program-support view. All seven publishable packages build in 8.50 seconds
  on Node 25.9.0.
- Remaining Gate 6 work is generated cross-framework/platform support views
  derived from the executable JIT registry and these program decisions.
  Terminal execution evidence remains a separate protocol fact.

### 2026-07-23 — Gate 6 framework/platform support composition

- JIT's JavaScript root now exports `frameworkOperationSupport()` and
  `frameworkPlatformSupportSource()`. Both parse the same generated
  36-operation JSON loaded by the Python executable validators, preserve the
  exact ten CPU/autograd/transform/export/plan/WebGPU/residency/materialization
  decisions, and return detached records.
- Runtime now exports `createFrameworkPlatformSupportView()`. It canonically
  reconstructs a complete provider-bound requirement environment and raw
  program-support input, then validates, bounds, sorts, and freezes one to
  sixteen framework-neutral sources with at most 256 operations each.
- The output keeps requirement resolutions, program lowering decisions, and
  framework contract strings in separate fields. It rejects open decision
  maps, duplicate identities, malformed versions/IDs, and empty or oversized
  sources; it creates no generic support or availability boolean.
- ADR-0050 records the structural package seam. Runtime imports no framework
  package, JIT imports no runtime package, and a fresh packed npm consumer
  proves that the generated JIT source satisfies runtime's public input type
  and runtime validator.
- JIT build/typecheck/lint, 5 files/27 unit tests, and the focused generated
  source tests pass. Runtime build/typecheck/lint and 14 files/137 tests pass.
  Semantic architecture and its 28 mutations pass. The release-package gate
  passes 19 hostile archive and 35 Node security tests plus the new
  runtime/JIT packed composition consumer.
- All seven publishable packages build in 8.33 seconds on Node 25.9.0.
  Remaining Gate 6 work is a generated Grad platform source derived from its
  frozen executable compatibility contracts. Terminal execution evidence
  remains separate.

### 2026-07-23 — Gate 6 generated Grad platform support and closure

- Grad now exports `frameworkPlatformSupportSource()` with all 22 verified
  eager compatibility records generated from the exact schema-v2 inventory.
  CPU/refusal, dtype, eager autograd, residency, and materialization facts are
  preserved; symbolic transforms, export, tensor plans, and WebGPU remain
  explicit non-applicability or refusal records.
- The generator accepts only the closed inventory identity and exact behavior
  shape, requires verified target contracts, rejects duplicate or unregistered
  mappings, and emits deterministic operation order. Architecture checking
  regenerates and exact-compares the checked-in source.
- ADR-0051 records the contract. Runtime still imports no framework package,
  and a packed fresh consumer composes Grad's 22 records with JIT's 36 records
  through the public runtime view.
- Grad's full unit suite passes 3 files/33 tests in 0.99 seconds. Semantic
  architecture and its 29 mutations pass. The release-package gate passes 19
  hostile-archive and 35 Node security tests plus the combined packed
  consumer. All seven publishable packages build in 8.36 seconds on Node
  25.9.0.
- Gate 6 is `verified` for the initial closed framework-convergence profile.
  New operations, broader dtype/layout coverage, and backend profiles must add
  executable contracts and regenerate the sources. Terminal execution
  evidence remains a separate authority.

### 2026-07-23 — Clang 22 production-invocation root cause removed

- The strict browser observation exposed a producer terminal
  `internal-error`. Replaying its exact device/host argument vectors against
  native Clang 22.1.8 identified the pre-semantic root cause: the shared
  compile plan still emitted
  `-fno-experimental-new-constant-interpreter`, which this driver does not
  support.
- The sole compile-plan builder no longer emits that option. The required
  native lane now executes both production-shaped CUDA semantic passes.
  Diagnostic capture also owns its error count across the whole
  `ToolInvocation`, so driver errors remain bounded and observable when the
  frontend action does not complete instead of collapsing to a zero-error
  internal failure.
- The new exact source/build lock is
  `bg.cpp.browser-build-input-lock.sha256.1b747a53be87251e85fdedb0d43dd48ba53ae83717436abcb86b46f874d33f0e`
  with resource SHA-256
  `ec61fb9d270df159c341fb8261ac2fb6f6d31d7e0e10b07a3787be10c8d850bd`.
  A fresh two-root pipeline rehashed all 17 header-distribution outputs. The
  five VFS packs remain byte-identical; the lock-bound inventory and aggregate
  notice received new identities. The package-pinned subset is 4,042 bytes at
  SHA-256
  `cb0afdf5bc616ab326b5dc577957ba29d12534206cf3588ecde8bafb3eaed574`
  with reproducibility ID
  `bg.cpp.browser-header-distribution-reproducibility.sha256.c4295b8226eac800cf37b5fcf92b8064b33167d67c0e6c94096f477d9e3dc4bb`.
- Verification: required native/build-plan lane 47 files, 253 passed and 9
  platform skips; complete compiler suite 96 files and 1,611 tests. The pinned
  Wasm still represents the preceding source lock. A new isolated Wasm build,
  raw-ABI review, reproducibility evidence, and strict browser observation are
  required before claiming one valid browser-local C++/CuTe compile.
- External file-level license approval, externally rooted producer trust,
  distribution authorization, lowering/backend execution from a production
  result, and release remain false.

### 2026-07-23 — Fast browser diagnostic authority and real-header work convergence

- The real-browser runner now admits an exact locally hashed fast-build Wasm
  only when `--allow-untrusted-diagnostic-wasm` is explicit. The evidence marks
  that input `untrustedDiagnosticWasm=true` and
  `pinnedReproducibleWasmMatched=false`. The strict compile gate requires both
  the package-pinned two-clean-build Wasm and a compiled Artifact V3, so the
  diagnostic lane cannot become release or reproducibility authority.
- Worker controller protocol v2 carries one bounded sanitized deepest failure
  detail. Before failed-execution cleanup, the C-ABI seam snapshots exact
  frontend-work lifecycle/counters, WebAssembly pages and allocator counters,
  plus bounded VFS file/miss counters. These are diagnostic-only records and
  cannot mint Worker execution or lowering authority.
- The first live signal proved that the former `internal-error` stopped at
  exactly 8,193 macro expansions because the real-browser request inherited an
  artificial 8,192 ceiling. Frontend work ceilings are now independent of the
  bounded number of retained Artifact V3 macro/template records. This removes
  a false coupling between compilation work and semantic projection size.
- With profile-bound work ceilings, unchanged C++17/CuTe source in Chromium
  reached 56,150 macro expansions, 238,041 preprocessed tokens, 49,690 AST
  nodes, 11 constexpr steps, and one completed CUDA semantic pass. The
  observation remained about 7.8 seconds end to end. WebAssembly remained at
  4,096 pages, allocator peak requested bytes were 17,499,914, and allocation
  failures were zero.
- Verification passed the full compiler gate with 96 files and 1,613 tests,
  the required native/build-plan lane with 47 files, 253 passes, and 9
  intentional platform skips, and the fast harness with 82 files and 724
  tests. Typecheck, lint, architecture checks, deterministic Worker authoring,
  build-lock verification, and the real Chromium observation also passed.
- The remaining compile blocker is narrowed to the one-pass
  diagnostic/Artifact V3 or second-pass boundary. The fast-build Wasm,
  browser observation, and failure diagnostics remain untrusted local evidence;
  producer trust, external license approval, Worker execution, lowering,
  backend authorization, distribution, and release are false.

### 2026-07-23 — Native first-cause diagnostics

- The native producer, Artifact V3 composer, runtime, and allocator tracker now
  preserve fixed first-cause codes instead of collapsing every unexpected
  branch into status 106. Allocator poisoning is sticky and records only its
  first cause, including untracked free/reallocation, reentrant hook, table,
  counter, and pointer-integrity failures.
- Codes use only the already-pinned bounded stderr capability. They contain no
  caller source, virtual path, rendered diagnostic, or dynamic detail and do
  not change ABI 1.2, Artifact V3, Worker-execution, lowering, or release
  authority.
- The source lock is
  `bg.cpp.browser-build-input-lock.sha256.c0c7ee66b05951ee3b977ccac0de68e5e417106d6c8894e1152771a4a775dbf3`
  at resource SHA-256
  `31c550f63668f896b728ff2b8beae45ca72abf76f582c0b1e237be2a0f47902f`.
  A fresh two-root run completed in 34.26 seconds and reverified all 17
  distribution outputs under reproducibility ID
  `bg.cpp.browser-header-distribution-reproducibility.sha256.9f98374557977e35bff019faa9d09cae1e1aa19fc6397ec703810d68dae987cf`.
- The 82-file/724-test fast gate passes in 12.55 seconds, the full compiler
  suite passes 96 files/1,613 tests, and the required native/build-plan lane
  passes 253 tests with 9 intentional platform skips. A fresh diagnostic Wasm
  and browser observation are still required to identify and remove the live
  producer blocker.

### 2026-07-23 — Strict raw-Wasm ABI validation

- Diagnostic build `30022629591` kept the exact 72-import capability surface
  but grew the JavaScript exception-dispatch table from 15,166 to 15,167
  entries. The former workflow persisted that mismatch as useful discovery
  evidence but still returned success, so the browser verifier became the
  first failing boundary.
- Runtime ABI 1.9 now pins the independently reviewed 15,167-entry table at
  manifest ID
  `bg.cpp.browser-runtime-abi.sha256.dee3716770d8521f478505323adfd771737778dc917fc5c9777dadd9b8f71056`
  and resource SHA-256
  `f4892b5f3a09d0543d48ab8da4ebadcc00476223470fab95b0ba2b06dff4a886`.
  No host import, support export, memory policy, feature, or source C ABI was
  widened.
- The raw-Wasm reviewer retains its observation-only mode, but
  `--require-exact-interface` converts an authentic nonconforming report into a
  fixed validation failure. Both cached validation and clean reproducibility
  use that strict mode after persisting the bounded report.
- The current build lock is
  `bg.cpp.browser-build-input-lock.sha256.27b4735cf1ca2970e0acbcc2c0f9b7b73bc35aeab84007b1a9ae4ca9ef9855a9`
  at resource SHA-256
  `3dadc3dcec2d846f3c359c5cc798eee8e18ee031bbc75b01ccccf8389abb1ffe`.
  Fresh two-root header materialization reproduced all 17 outputs under
  `bg.cpp.browser-header-distribution-reproducibility.sha256.00ec67e05c5c950557e3963502e3bc772d36978467398c063e91a285dcaa05e2`;
  its package resource SHA-256 is
  `8802767f3a1f6ec66031762a0f4a86be01595bab7e2f6d3e74208c57b844c79d`.
- Node 25 passes the 82-file/725-test fast gate in 10.55 seconds and the
  96-file/1,613-test full compiler suite in 10.02 seconds. Strict local review
  of the exact 31,649,344-byte diagnostic Wasm reports zero mismatches. Those
  bytes still embed the preceding request-manifest hash, so they are review
  evidence only; one cached current-lock rebuild is required before the next
  browser execution. The required native/build-plan lane passes 254 tests with
  9 intentional platform skips.

### 2026-07-24 — Closed CUDA headers and first accepted browser CuTe artifact

- `ea08e8fc` adds the exact NVIDIA `libcurand` 10.3.7.77 archive to the CUDA
  12.6.3 header plan. Admission verifies its 81,729,748 archive bytes, exact
  SHA-256, 17 selected headers, and the same exact CUDA Toolkit license bytes
  already required by the other NVIDIA components.
- `7476fd48` admits the exact upstream Clang 22.1.8
  `__clang_cuda_runtime_wrapper.h` by its 18,624-byte identity and materializes
  one deterministic profile-bound derivative. The transformation scopes the
  header-provided unsupported-libc++ escape hatch to the wrapper, removes one
  unmatched upstream `__USE_FAST_MATH__` pop, and undefines `__host__` only
  immediately before Clang's two intentional redefinitions. Global
  macro-redefinition errors remain enabled.
- A clean two-root materialization from commit `7476fd48` rehashed the exact
  17-output tree in both private roots. The result totals 71,114,813 bytes:
  reproducibility ID
  `bg.cpp.browser-header-distribution-reproducibility.sha256.43f703672ddbeaf1e6e6d544e3ed50721a2585e947b5d0a1e624293cac80d449`,
  pipeline ID
  `bg.cpp.browser-header-pack-pipeline.sha256.80a29abc734fcf3183c98fbd3bce5c23005a045f06e6837b80231845fdf09b71`,
  and output-verification ID
  `bg.cpp.distribution-output-file-verification.sha256.1cc298cf70ed624df258a14b0eb687c6a0666a14cdd4e5d208674f6c0f7fb3df`.
  `b66d9026` package-pins that observation at resource SHA-256
  `8dafa7484a7ca7a5c12e6cb2128cfbbc22d2692a96ebc1581c81e48953ad8620`.
- The clean no-rebuild Chromium lane then installed all five packs and 5,788
  files, verified the raw Wasm in its separate Worker, and compiled unchanged
  C++17/CuTe source inside the compiler Worker. Compilation took 22.369
  seconds and total browser execution took 25.596 seconds. The accepted
  Artifact V3 is
  `bg.artifact.cpp-cute-frontend.sha256.4489656ea0da6faef2a37164fd73e36e201e15f4fba640fa88395a46deb81991`;
  Worker evidence is
  `bg.cpp.browser-worker-execution.sha256.afe0b8f2df66c3f29789afb423fb6433c686803370b93817c4859217b97558fd`.
  Its rank-2 candidate retains layout semantic hash
  `9c4ad2f7a3f05e21511c98873155d689ae2e6f253ef29ab1022945f0e2198be0`.
- This is the first accepted Artifact V3 from unchanged C++/CuTe source in the
  real browser Worker, but the exact 31,653,752-byte Wasm remains an explicitly
  untrusted diagnostic local observation. It is not the package-pinned
  two-clean-build Wasm. Header license approval, producer trust, lowering,
  backend execution, and release readiness therefore remain false.
- At that checkpoint the fast gate passed 84 files/736 tests plus typecheck, deterministic
  Worker verification, build-lock verification, and the zero-cycle
  architecture audit. The ordinary edit loop remains seconds-scale; the
  approximately one-minute two-root command is release evidence, not an
  every-edit dependency.

### 2026-07-24 — Reproducible producer and strict browser observation

- Two-clean-build run `30047077419` at `c41ab6a8` produced byte-identical
  31,653,752-byte Wasm and 27,285-byte factories. The locked build steps took
  35 minutes 22 seconds and 45 minutes 14 seconds. The Wasm SHA-256 is
  `7950c52270fdac4ea8cae36fbaafbde56cb61720242e10ea5881becf2fe4cfd4`;
  the factory SHA-256 is
  `f64d5239d5c258f44e859834b57e1ea330b7efdf7a405dead3126b53330a5534`.
  The run's overall failure was isolated to the native VFS test's missing
  `<array>` include; both clean producer jobs and their strict raw-Wasm reviews
  succeeded.
- `df6da6a4` closes that native test input, and `5bd6ae6c` lets the
  reproducibility comparator retain two successful clean builds even when an
  independent JavaScript verification job fails. Corrected cached validation
  run `30047968064` is green. The required native/build-plan lane passes 261
  tests with 9 intentional platform skips across 49 files in about 55 seconds.
- `cc7856ee` makes detached review require exact raw-Wasm interface
  conformance. Verifier-only run `30049923259` at `2d3cd52b` reused the two
  immutable clean artifact trees and succeeded without rebuilding LLVM. Its
  canonical 3,470-byte evidence has SHA-256
  `974bcaae92e88522f2a8ed91874c50269fbe0a84ec00823508495e3f034ac047`;
  the local comparator and remote verifier emitted byte-identical evidence.
- `559a0586` package-pins that current reproducibility record and derives the
  generated-factory authority from it. The rebuilt zero-import package Worker
  is 582,580 bytes with SHA-256
  `eb7df701054a82f59486c011e9a861e1565525c688e402fc1d3fe1724f2530f6`.
- The strict no-rebuild Chromium lane then required the package-pinned
  reproducible Wasm and all five package-pinned header packs. It installed
  5,788 files, opened one source and 1,104 headers, compiled in 21.133 seconds,
  and completed in 24.331 seconds. The accepted Artifact V3 remains
  `bg.artifact.cpp-cute-frontend.sha256.4489656ea0da6faef2a37164fd73e36e201e15f4fba640fa88395a46deb81991`;
  strict Worker evidence is
  `bg.cpp.browser-worker-execution.sha256.fbff539a3f5a3ad532e21d24ab07665f1f7b5b434b8aec74d57b7ba5e3b69019`;
  the observed semantic candidate is
  `bg.cpp.browser-worker-layout-candidate.sha256.72f3b5933de96569359767f19fdaaed3eaadadc56ca5c297238abcc149c0d34d`.
- `0ac82726` package-pins the exact 2,886-byte strict observation at SHA-256
  `bf4d378a92eda260a120da15651deac8d42c7324de490ad1009224a3e7761496`
  and admits it only when the current reproducibility authority, header
  reproducibility authority, package Worker, artifact, and truth claims all
  match. The current fast gate passes 85 files/738 tests in about 12.5 seconds.
- This closes strict reproducible layout-only browser execution. It does not
  establish externally rooted producer trust, legal approval for the header
  distribution, real source-produced dynamic Tensor/view-copy semantics,
  lowering authority, backend execution, or release readiness; every one of
  those claims remains false.

### 2026-07-24 — Word32 view-copy expansion and fail-closed ABI repin

- Capability commits `85778e8f` and `0d42f33e` extend the Clang AST producer
  and canonical execution seam without adding source-shaped lowering. Exact
  `float`, `int`, and `unsigned int` device ABI facts lower to f32/i32/u32
  storage, and equal-rank view copies cover ranks 1 through 4, positive affine
  layouts, nonzero offsets, positive slices, read-only broadcast, padding, and
  zero extent.
- Required Chromium on Apple Metal 3 executes 13 word32 cases through WebGPU.
  The terminal evidence has artifact SHA-256
  `50ea47971af5ff9c5876718c3cfc2ea2cbc8d47e512a33dbeb4551d0bafc3443`
  and case-set SHA-256
  `44e912bac8dd90c66071169a014f0f629c8050683ad27ab27c3109d1cead7a28`.
  This proves the shared semantic/backend profile, not current source-producer
  promotion.
- Both clean builders in run `30067229885` compiled byte-identical
  31,841,010-byte expanded extractors with SHA-256
  `35cf6d2576875da094de39d6f226c37d49e542fb16a876973347647a85c62002`.
  Build execution took about 39.6 and 46.7 minutes. The strict post-build
  reviewer then failed both jobs because the JavaScript exception-dispatch
  table shrank from the pinned 15,304 entries to 15,301. Imports, exports,
  memory, tags, and generated runtime support still matched exactly.
- Commit `9479fcdf` introduces runtime ABI 1.17 for that smaller reviewed table
  and current build lock
  `bg.cpp.browser-build-input-lock.sha256.5a96def9bac1db052108142dfe4c82e729f4b41f450d459406a4f3c5227daad7`.
  The downloaded Wasm passes the new exact review with zero mismatches. The
  compiler and verifier Workers were deterministically regenerated at SHA-256
  `3d1692b959f5ce1b61cd9a1810641f7a7aabad08e56b992c9297368c875ef3b1`
  and
  `06ffb66e4e808e9df030cc3fe2981fa3adddf13d03780680abb091cbcbd4b9eb`.
  The old four-case strict observation now fails closed against the changed
  Worker rather than being silently relabeled.
- Local verification passes 86 fast files/748 tests, the complete 97-file/
  1,624-test compiler suite, exact build-lock and zero-import Worker checks,
  and the semantic architecture guard. Current two-clean-build run
  `30069614333` is active. The strict eight-case source matrix cannot run until
  those bytes become the exact package-pinned reproducibility authority; the
  runner correctly rejected an attempted diagnostic substitution.

### 2026-07-24 — Current extractor promotion and strict eight-case source proof

- Workflow run `30069614333` at exact source
  `9479fcdfba172f56fff93498f72ea33bd449ac7e` is green. Its two cache-free,
  distinct-path builds produced byte-identical 31,841,008-byte Wasm with
  SHA-256
  `19edd5622461b2308e83f10fb90f9f029241a5ba706e4c1741b194cb52a82138`.
  The isolated build steps took 40 minutes 30 seconds and 45 minutes 55
  seconds concurrently; complete job durations were 41 minutes 45 seconds and
  47 minutes 28 seconds. The 46-second comparison job verified Wasm, factory,
  canonical linker-map projection, native TableGen tools, runtime closure, and
  exact runtime-ABI review identity across both builds.
- Commit `6691bde4` package-pins the canonical 3,470-byte reproducibility
  resource at SHA-256
  `b8ab918d667d68a8effcdcd14a79691a92e7d3466e9041906e5039c9993e028d`.
  It regenerates the zero-import 584,660-byte compiler Worker at SHA-256
  `a3b8610fc116c7b4949379dbcdcdd55e06ce5f9f59f11bf692abd689b0f17916`.
  The separate 158,314-byte verifier Worker remains SHA-256
  `06ffb66e4e808e9df030cc3fe2981fa3adddf13d03780680abb091cbcbd4b9eb`.
- The strict Chromium matrix then compiled eight unchanged C++17/CuTe sources
  with that exact Wasm and the five package-pinned header packs: f32 ranks 1,
  2, 3, and 4; positive f32 strided-slice; read-only f32 broadcast; i32 rank-2;
  and u32 broadcast. Per-case compile time was 23.869–26.319 seconds and total
  browser time was 26.988–29.587 seconds. Every case opened one source and
  1,168 headers, produced one accepted Artifact V3, and prepared a distinct
  view-copy candidate.
- Commit `9ce2d5b0` package-pins the canonical 26,213-byte matrix at SHA-256
  `5d60c80c5aec6b2164b80769f800a6fda931efabf43859c7a22db06910f0768d`
  and source revision `6691bde4137efd6f0522e5f86f2863e46461549f`. Admission
  re-verifies the exact resource bytes, current reproducibility authority,
  complete header distribution, both Worker identities, eight source/dtype/
  layout facts, and unique evidence/artifact/candidate identities. A
  deterministic authoring check is now part of the fast gate.
- Final local verification passes the 87-file/751-test fast gate in 15.67
  seconds, compiler typecheck, the 97-file/1,624-test suite in 16.79 seconds,
  and the no-cycle/no-leak architecture guard. This closes current extractor
  promotion and strict source compilation. It does not self-approve producer
  trust or redistribution, authorize lowering/backend execution, prove the
  complete distributed output set, or mint release authority; those claims
  remain false.

### 2026-07-24 — Exact compile-profile ownership and strict matrix repin

- Commit `894c1e83` replaces the browser compile test's dependency on the
  synthetic frontend-profile fixture with a package-owned production-shape
  constructor. It pins exact Clang 22.1.8, CUDA 12.6.3, CUTLASS 3.7.0 at
  `b78588d1630aa6643bf021613717bafb705df4ef`, libc++ 22.1.8, Linux sysroot,
  source-root, header-content, Worker, Wasm, semantic-adapter, runtime-ABI, and
  extraction-limit fields. The focused contract prevents fixture placeholder
  revisions from returning to the real browser lane.
- The first regeneration compiled all eight sources successfully but the
  matrix aggregator rejected case-varying asset-set identities. Commit
  `e9062c20` fixes the model: user main-source bytes remain request-bound,
  while the profile owns one exact empty project-header include-root manifest
  at SHA-256
  `6076ac6ed221c1ce33a656d14113c1099c60bd6781ae65928cdb85ed55ab9c91`.
  This preserves one compiler/header asset set across arbitrary main sources
  without weakening per-request source identity.
- The strict rerun at source
  `e9062c20f2a070774743e4d839c275c05df47225` passed all eight isolated
  Chromium compilations. Compile times were 23.228–24.768 seconds; all cases
  used asset set
  `d5e3d24dd105d66555ac857c75dc7c66416cdd6e86b181592476a622c7313cbe`
  and retained unique Worker evidence, Artifact V3, and semantic candidate
  identities.
- Commit `6341c956` package-pins the resulting 26,213-byte resource at SHA-256
  `4d8b956050834e550405b15f1c7d52b16927ee3e8d4bc7b7da4035d430edc80a`.
  The fast gate passes 89 files/754 tests in 13.74 seconds, the complete
  compiler suite passes 99 files/1,627 tests in 14.42 seconds, typecheck
  passes, and architecture reports zero cycles or leaks. Trust, legal,
  lowering, backend, complete-output reproducibility, and release claims
  remain false.

### 2026-07-24 — Complete distribution reproducibility verifier

- Commit `8eea3660` adds one package-owned verifier for the build lock's complete
  24-path distributed-output plan. It rehashes both exact private trees before
  and after detached-evidence inspection and rejects missing, additional,
  writable, linked, changed, overlapping, or noncanonical output state.
- All 23 `deterministic-subject` outputs must have identical SHA-256 and byte
  length across the roots. The sole
  `assets/browsergrad-cpp-cute/build-provenance.dsse.json` output may have
  different envelope bytes, but both envelopes must strict-decode, bind the
  exact prepared build-lock identity and recipe, and name the same build
  subject.
- The result is opaque verifier-issued reproducibility authority. Its claims
  explicitly keep signature verification, license review, distribution
  approval, producer trust, Worker execution, lowering, backend execution,
  and release readiness false. Structural copies cannot carry authority.
- Focused typecheck, lint, and four adversarial tests pass. The complete fast
  lane passes 90 files/758 tests in 15.49 seconds, the compiler suite passes 99
  files/1,627 tests in 12.91 seconds, and both compiler and workspace
  architecture checks report zero cycles or leaks.
- This closes the software verifier, not the release evidence. The current
  repository has not materialized and verified two complete live 24-output
  distributions, so `fullDistributedOutputSetReproducible` remains false for
  the promoted production build.

### 2026-07-26 — External header-distribution approval seam

- Commit `b1caf988` adds one bounded canonical host policy and one exact
  distribution-approval verifier. The policy binds its derived identity,
  trust-store hash, reviewer allowlist, key allowlist, and resource limits.
- The verifier reauthenticates the current package-pinned 17-output header
  distribution and derives one exact review subject over the current build
  lock, header-input projection, every output identity, package resource,
  output verification and reproducibility identities, and the exact
  `license-inventory.json` hash and length.
- Approval requires one canonical DSSE/in-toto statement and one allowlisted
  P-256 P1363 signature. Its opaque result upgrades only the external
  file-license, notice, CUDA-index, upstream-evidence, license-review, and
  distribution claims for that exact policy and subject. Signing-request
  material grants no authority.
- Seven focused adversarial tests cover exact success, policy admission,
  defensive copying and opaque authority, malformed/open/oversized policies,
  unsafe buffers, wrong roots, subject drift, unsigned and hostile envelopes,
  forged authorities, and cancellation. Strict typecheck and lint pass. The
  fast lane passes 91 files/765 tests in 14.88 seconds, the complete compiler
  suite passes 100 files/1,634 tests in 14.50 seconds, and compiler/workspace
  architecture checks report zero cycles or leaks.
- This closes the software admission seam, not production approval. The
  repository contains only a synthetic test policy, ephemeral test key, and
  test statement. No package-controlled production policy, externally
  controlled reviewer key, or externally issued decision has been installed.

### 2026-07-26 — Exact external build signing request

- Commit `e722434c` derives canonical DSSE/in-toto signing material only from
  the exact prepared asset manifest, build-input lock, verified Worker,
  admitted producer policy, prepared trust store, builder, and key.
- The result is format-only: all signature, producer-trust, execution,
  distribution, lowering, backend, and release claims remain false. The
  existing exact signature-binding and producer-trust transitions remain the
  only authority-producing seams.
- Six adversarial tests cover exact output and defensive bytes, downstream
  external signing, policy/root/builder/key drift, forged or cross-bound opaque
  inputs, hostile accessors/proxies, and cancellation.
- The fast lane passes 92 files/771 tests in 14.23 seconds, the complete
  compiler suite passes 101 files/1,640 tests in 14.03 seconds, and compiler
  and workspace architecture checks report zero cycles or leaks.
- No production policy, external builder key, or externally issued exact-build
  statement is present. This capability makes the external handoff exact; it
  does not establish production producer trust.

### 2026-07-26 — Closed browser distribution asset plan

- Commit `628c718a` fixes the closed distribution plan discovered during the
  live-evidence audit: the asset manifest required diagnostic-normalization
  bytes while the former 24-path plan prohibited that file.
- The repinned build lock now contains 24 deterministic subjects plus one
  detached DSSE envelope. Lock admission requires all nine package executable,
  VFS, and runtime-policy asset outputs with exact paths, roles, and selected
  media types. The semantic-adapter output now retains its versioned media type.
- The complete two-root verifier and declaration contract now require all 25
  outputs. Four adversarial tests pass; the narrow authority still grants only
  reproducibility and keeps signing, trust, licensing, execution, lowering,
  backend, and release false.
- Commit `df51179f` keeps the full suite reliable under parallel load by giving
  two intentional resource-ceiling stress cases explicit budgets and reducing
  the above-default selection to the minimum 16,385 entries. The focused pair
  passes 24 tests in 3.32 seconds.
- The fast lane passes 92 files/771 tests in 17.34 seconds; the complete suite
  passes 101 files/1,640 tests in 14.45 seconds. Typecheck, lint, build-lock
  authoring, both architecture checks, and the full compiler verifier pass.

### 2026-07-26 — No-clobber external producer exchange

- Commit `3cf11a47` adds one host exchange interface with explicit
  `author:browser-build-provenance-signing-request` and
  `verify:browser-build-provenance-envelope` commands. The first derives and
  persists the exact format-only request; the second requires that same request,
  rebinds the returned payload and key, and runs the package signature and
  independent producer-policy transitions in one process.
- All profile, manifest, lock, Worker, policy, trust-store, request, and
  envelope inputs must be canonical immutable single-link files with stable
  current-user-owned path and parent identities. The lock and Worker must equal
  the current package bytes. Outputs use exclusive no-follow creation, file and
  directory synchronization, read-only mode, independent reread, no clobber,
  and owned-file cleanup on failure.
- Private keys are not accepted. A successful verification writes only
  `host-verification-observation-only` JSON with reusable producer authority
  false; the actual producer authority remains opaque and process-local.
- Six adversarial interface tests cover deterministic canonical requests,
  actual external-style signing, exact verification, read-only persistence,
  clobber and alias refusal, writable/link/package drift, request/payload/key/
  signature drift, hostile arguments, and cancellation.
- The fast lane passes 93 files/777 tests in 17.04 seconds; the complete
  compiler suite passes 101 files/1,640 tests in 14.72 seconds; and the required
  native boundary passes 53 files/280 tests with nine explicit skips in 58.54
  seconds. The full compiler verifier, strict typecheck/lint, and compiler plus
  workspace architecture checks pass with zero cycles or leaks.
- This closes the operational file-exchange seam, not external evidence. No
  package-controlled production policy, externally controlled builder key, or
  externally issued exact-build envelope exists in the repository.

### 2026-07-26 — Unified external distribution-review exchange

- Commit `23d8bb60` deepens and renames the producer-only runner into one
  protocol-discriminated external-evidence exchange. Existing producer commands
  retain their exact semantics; new
  `author:browser-distribution-approval-signing-request` and
  `verify:browser-distribution-approval-envelope` commands reuse the same
  hardened immutable-input, exclusive-output, synchronization, no-clobber, and
  owned-cleanup boundary.
- Review-request authoring reauthenticates the exact canonical approval policy
  and trust store, binds the package-pinned header-distribution resource, and
  requires an allowlisted reviewer/key pair. It accepts no private key and keeps
  signature, external review, licensing, distribution, trust, execution,
  backend, and release claims false.
- Verification consumes the exact issued request, rederives it from the current
  policy and package review subject, matches the returned DSSE payload and key,
  and invokes the opaque distribution-approval verifier in the same process.
  Persistence records only `host-verification-observation-only`; reusable
  distribution-approval authority remains false.
- Nine focused adversarial cases pass in 2.60 seconds, including both producer
  and reviewer exchanges, real P-256 signing, deterministic canonical output,
  policy/trust/request/signature/path drift, immutable input and output rules,
  no-clobber behavior, hostile arguments, and cancellation.
- The fast lane passes 93 files/780 tests in 14.87 seconds; the complete compiler
  suite passes 101 files/1,640 tests in 14.62 seconds; and the required native
  boundary passes 53 files/283 tests with nine explicit skips in 63.24 seconds.
  Full compiler verification and both architecture checks pass with zero cycles
  or leaks.
- This closes the package-owned reviewer handoff, not the external decision. No
  production approval policy, externally controlled reviewer key, externally
  issued statement, or reusable legal/distribution authority is present.

### 2026-07-26 — Exact full-distribution materialization

- Commit `0840801a` adds a pure deterministic metadata seam plus one closed
  host materializer. It reconstructs the exact profile, asset manifest,
  cycle-free build subject, and 24 deterministic outputs from the current
  build lock, package-pinned reproducible Wasm, verified package Worker, five
  inspected header packs, and admitted producer policy.
- Deterministic materialization accepts no signing material and writes no DSSE
  envelope. Finalization separately rereads the exact tree and immutable
  profile and adds only
  `assets/browsergrad-cpp-cute/build-provenance.dsse.json` after the existing
  opaque producer transition binds the same build subject. Admission mode can
  reauthenticate an existing deterministic tree without writing.
- Inputs must be immutable, single-link, exact current files under stable
  private roots. Outputs use exclusive no-follow creation, fixed file and byte
  ceilings, independent rereads, exact tree closure, file and parent
  synchronization, and no clobber. Failure cleanup removes only exact files
  created by that transaction.
- Two independent exact-source reruns materialized and finalized distinct
  roots. Concurrent header pipelines took about 20.75 seconds each;
  deterministic materialization took about 0.76 seconds per root and
  finalization about 0.65 seconds per root. Each final root contains 25 files
  and 103,637,461 bytes. All 24 deterministic subjects are byte-identical.
- The deterministic metadata ID is
  `bg.cpp.browser-distribution-metadata.sha256.a276fcac4a1feea108c007e410ad35388f0f9f316c52e95414c0edd05d63b1ab`;
  the build subject is
  `bg.cpp.browser-build-subject.sha256.f54e7af965f09b75b49e1db00a5a94d5ba6efc3771c0deff483b0c4417bc010c`;
  the complete output-verification ID is
  `bg.cpp.distribution-output-file-verification.sha256.a6a0762ca6bfb1718a32511946594d82aff4be536c1e8a7aaf12a69604091306`;
  and the full reproducibility ID is
  `bg.cpp.browser-full-distribution-reproducibility.sha256.b9b6bab28354bc3ddb7466886f958413d253aab65c00bcd61ef13eeaa074c2bf`.
- The live producer policy, key, request, and envelope were deliberately
  labeled local engineering reproducibility only. They prove the finalizer and
  verifier flow, not externally rooted production trust. Signature, legal,
  distribution, execution, lowering, backend, and release claims remain false.

### 2026-07-26 — Package-pinned live full-distribution evidence

- Commit `c00769c6` records the exact live two-root observation in one
  8,951-byte canonical resource at SHA-256
  `4637cf9624ad00dd833b72b832f2c9e25ee0a8ffcdf62d8b95b732791d36a65a`.
- Its package verifier admits only exact unshared bytes, then independently
  decodes the current build lock and requires the exact 24+1 plan. It
  reauthenticates the 17 header outputs, reproducible extractor, zero-import
  Worker, current lock bytes, diagnostic-normalization, runtime-ABI, and
  semantic-adapter resources; binds the asset-manifest output and detached
  build subject; recomputes both root totals; and recomputes the full
  reproducibility identity.
- The resulting authority is opaque and copy-resistant. Its scope is
  `package-pinned-full-distribution-reproducibility-only`; structural copies
  and modified, truncated, shared, or non-byte records fail. The recorded
  producer policy remains local engineering only, so external producer trust,
  license review, distribution approval, execution, lowering, backend
  execution, and release readiness remain false.
- Focused verification passes two files/seven tests. The fast gate passes 96
  files/789 tests in 14.14 seconds. The required native boundary passes 54
  files/286 tests with nine explicit skips in 52.38 seconds, and the complete
  compiler suite passes 103 files/1,646 tests in 11.52 seconds. Strict
  typecheck, lint, Docker-shell 2 files/96 tests, compiler/workspace
  architecture, and the full compiler verifier pass with zero cycles or
  leaks.

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

Extractor-output reproducibility, exact configured header-pack materialization,
the package-reviewed normalization environment, the deterministic complete
header review input, exact notice output materialization, and package-pinned
two-root header-subset reproducibility are now independent narrow authorities.
The host-only distribution-approval transition now binds those exact current
facts to a separately admitted policy and externally signed review statement.
Its unified no-clobber handoff derives the exact format-only request and verifies
the returned envelope without serializing reusable authority; the synthetic test
fixture still establishes no production approval. Preserve these authorities
while installing the package-controlled production policy, externally
controlled reviewer key, and externally issued exact-review statement. Neither
reproducible bytes nor a package-generated file map, notice bundle, signing
request, signature, or observation can substitute for independent approval.

The current complete 25-output distribution is now materialized, independently
rehashed across two distinct roots, and package-pinned as narrow
complete-output reproducibility authority. Preserve its exact current build
lock, 24 deterministic identities, detached build subject, local-engineering
policy scope, and false external/release claims. A build-lock, Worker, Wasm,
header, policy-resource, asset-manifest, materializer, or output-plan change
invalidates this observation and requires a new exact two-root run and reviewed
package resource; unchanged external trust or legal work does not.

The manifest-pinned build-signature verifier, policy-scoped signing request,
host-only trust-policy transition, and no-clobber request/response exchange are
implemented. Producer authority still requires the exact opaque signature
binding plus an independently admitted predicate, trust-store hash, builder ID,
key ID, and policy version; neither a request nor its serialized verification
observation carries that authority. The next checkpoint is external: install
the package-controlled production policy, issue the exact immutable request,
sign only through the externally controlled key, and verify the returned
envelope in the same process. Fixture or package-generated signatures remain
synthetic and cannot establish current production status.

The controller, package invocation, and package-owned raw-Wasm verifier are
wired. The compiler-Worker invocation binds the exact canonical derivative
evidence region, while the execution-evidence boundary re-authenticates both
the original host verifier authority and the exact protocol-issued result
authority rather than accepting structural copies. The package now admits the
current eight-case f32/i32/u32 rank-1-through-rank-4 matrix from the promoted
reproducible extractor. That exact lineage prepares accepted view copies
through shared semantics as observed candidates, but candidate preparation is
not lowering authority. The next execution checkpoint is externally rooted
producer admission followed by production-path CPU plus required real-WebGPU
convergence over those exact candidates.

The producer-neutral authorized artifact seam lowers exact f32/i32/u32
same-dtype view copies at ranks 1 through 4 into the canonical layout/kernel
artifacts and CPU reference. An observed browser view-copy candidate plus the
independently admitted exact producer converge into that same canonical
authorization while retaining the full Worker/profile/request/artifact
lineage. Candidate and authorization identities exclude allocation lengths and
offsets; those remain later runtime-binding facts and are not folded into
producer trust or inferred from CuTe `cosize`. The required actual-WebGPU lane
passes 13 word32 layout cases without minting source-producer, distribution, or
release authority.
The synthetic browser authority fixture retains
`productionBrowserCompileObserved=false` and still does not substitute for a
real production Worker compile. Gate 4 now owns the first frontend-neutral
logical GEMM tile, a separate verified schedule, exact certified-input CPU and
portable-WebGPU execution, and honest re-legalized preservation reporting.
Gate 5's initial closed f32 profile is verified. It owns the separate
frontend-neutral attention-forward contract, an independently verified tiled
schedule, schedule-independent CPU oracle, authority-bound schedule
specialization, scalar block-tiled WGSL, bounded production host execution,
causal/non-causal two-schedule correctness, and a separate observational
performance record against the frozen row-wise baseline. The exact execution
trace still requires declared-policy comparison; the performance record still
makes no superiority claim; no FlashAttention-v2, additional dtype/layout,
resident-buffer provenance, or frontend schedule-preservation claim is implied.
Gate 6 is verified for its initial closed convergence profile. All thirty-six
formerly opaque advertised JIT operations have typed executable decisions;
only the intentional user-authored WGSL extension boundary remains opaque.
JIT's generated source projects those 36 validator-bound records, while Grad's
generated source projects all 22 source- and fixture-bound eager compatibility
records. Runtime composes both with provider-bound requirements and
subject-bound lowering decisions without framework dependencies or inferred
support booleans. Broader dtype/layout coverage and new framework/backend
profiles remain capability work, not unrecorded widening of this profile.
Terminal execution evidence remains separately authoritative.

Install the production distribution-approval policy, issue the exact immutable
request through the new exchange, and have the external reviewer sign only those
bytes. Verify the returned envelope through the same exchange, then bind the
independently reread approved file map, CUDA index, source evidence, notice, and
license-inventory identities into the asset/provenance chain before mounting
those exact packs in the package Worker.
No ambient builder header may enter the parsed-program sysroot.

The next executable product checkpoint is the independent external signer
transition over the exact promoted build subject. Once that authority exists,
the exact package-pinned eight-case observed payload must lower through the
shared word32 seam and match CPU plus required real WebGPU. The external
file-level redistribution decision now has a software admission seam but no
production evidence. It, externally rooted producer trust, exact-payload CPU
and required real-WebGPU convergence, and final release composition remain
separate prerequisites; current live complete-output reproducibility is
already satisfied and package-pinned without satisfying any of them.
