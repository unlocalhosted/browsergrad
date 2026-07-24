# Compiler Harness Quality Audit — 2026-07-18

## Verdict

As of 2026-07-24, the BrowserGrad compiler harness is strong on correctness,
security boundaries, and iteration speed. It separately proves parser,
native-Clang, raw-Wasm ABI, two-clean-build, browser Worker, semantic lowering,
CPU, and real-WebGPU tiers; a lower tier cannot silently satisfy a higher one.
The ordinary compiler loop is seconds-scale and no longer invokes clean LLVM/
Clang production.

Maintainability remains the weak dimension. Several native and orchestration
modules are large, generated identity changes require coordinated authoring,
and the clean evidence lane still takes roughly 40–47 minutes per concurrent
builder. Line ratchets, dependency-cycle checks, generated-resource checks, and
focused test routing constrain that debt, but do not make it disappear.

Overall rating: A for fail-closed evidence integrity, A- for feedback topology,
B for maintainability, and incomplete for production authority. Externally
rooted producer signing, file-level redistribution approval, full release
output reproducibility, and final release authorization remain false.

## 2026-07-19 follow-up

The original speed verdict is now materially better. Cached source iteration
stabilized between 4 minutes 27 seconds and 5 minutes 4 seconds, while the
canonical local fast gate moved ordinary harness feedback off the native build
path. Before bundle checks it measured 21.74 to 26.37 seconds; the expanded
30-file/337-test gate, including deterministic Worker-bundle authoring, measures
29.24 seconds end to end on Node 25. Clean proof remains cache-free and is not
part of the edit loop; the first successful historical clean run completed in
39 minutes 29 seconds.

Maintainability also improved without relaxing ratchets: compile-session limit
access was consolidated and the file is now 2,112 lines against its 2,121-line
ceiling, while the 1,784-line artifact writer remains exactly capped. Exact
frontend-work records replaced estimate-only compiler work counters, and the
reproducibility comparator now admits and compares the ABI-review sidecar that
the workflow itself produces. One exact 574,770-byte zero-import Worker graph
is now authored twice, hash-pinned, checked in the fast gate, and loaded from
verified Blob bytes in Chromium. Production Worker execution, current-ABI clean
proof, licensed header packs, and two-build reproducibility remain open.

The next current-source clean run exposed one quality hole rather than closing
that proof. Run `29678087663` failed after 46 minutes 25 seconds because a
patched upstream Clang source generated into the build tree retained the
private relative include `ByteCode/Context.h`, but the external target carried
only Clang's public source and generated include roots. The target now adds the
canonical private `clang/lib/AST` root, and the generated-flags review requires
all three roots before compilation. This exact regression is therefore moved
from a late clean-build failure to the configuration boundary. The failed run
grants no output, ABI, clean, reproducibility, or release authority.
The expanded gate now passes 30 files/338 tests in 28.86 seconds on Node 25.

The first attempted cached validation of that repair exposed a separate speed
policy defect: hashing the entire extractor CMake file into the reusable
toolchain key converted any target-wiring edit into a cold LLVM/Clang build.
Run `29679722745` was cancelled rather than allowed to continue cold. The key
now binds only true reusable-layer inputs. Exact CMake is always reapplied,
BrowserGrad target objects are invalidated, and generated flags are reviewed
before compile. One exact compatible legacy key migrates the already populated
untrusted cache to the CMake-stable primary key; no broad fallback or clean
authority was added. The expanded local gate remains green at 30 files/338
tests in 29.20 seconds.

The first migration attempt then exposed a workflow admission bug: the pinned
cache action returns `cache-hit: false` for a restore-key match even when it has
downloaded a compatible cache. Run `29680036963` therefore ignored those bytes
and was cancelled at roughly 20 minutes into another cold build. Admission now
uses the separate `cache-matched-key` output; exact-hit state remains separate
so a compatible restore is staged and saved under the new primary key.

That repair is now proved. Migration run `29680686426` completed in 4 minutes
14 seconds, including 2 minutes 29 seconds for the isolated build, and saved
the migrated primary cache. Exact-primary run `29680831101` then completed in
4 minutes 17 seconds: the JavaScript boundary took 32 seconds, exact cache
restore took 3 seconds, isolated compile/link took 2 minutes 25 seconds, raw
ABI review took 2 seconds, and cache staging/saving was correctly skipped.
The original 90-to-97-minute build is therefore no longer the engineering
iteration loop.

The exact-primary output also demonstrated why raw interface review remains a
separate gate. The 31,641,378-byte Wasm was structurally valid but added 14
Emscripten `invoke_*` exception-control-flow signatures and grew the fixed
exception dispatch table from 14,549 to 15,166 entries. Each new import was
traced to the generated Emscripten factory's same bounded stack-save,
table-dispatch, Emscripten-exception-only catch, stack-restore, and `setThrew`
bridge; none adds ambient authority. Runtime ABI 1.8 now pins all 66 generated
imports and the exact table projection. Detached local review of the same raw
Wasm reports zero mismatches and exact interface conformance. Exact-source run
`29681607575` then rebuilt the repinned source successfully in 5 minutes: the
isolated build took 3 minutes 19 seconds, the primary cache hit exactly, and
raw review reported zero mismatches. It produced a 31,641,377-byte module with
SHA-256 `5fc425bbc051a2f5be588c2acbb164efb5e43f949afb48a373f3ed022c3b8758`.
The repinned 30-file/338-test local gate completes in 24.7 seconds on Node 25.

## 2026-07-20 corpus and Worker follow-up

The ordinary workspace build is no longer a meaningful cold-path bottleneck.
Main-branch CI `29768391553` passed all eight concurrent jobs at exact revision
`6c6715591c64a9dc657c6431fc723ec292c1ba0a` in 4 minutes 4 seconds. The two
real-world CUDA jobs built the whole workspace in 16 and 18 seconds, provisioned
all pinned corpora in 14 and 15 seconds, and completed their authoritative
source/distribution gates in 2 minutes 26 seconds and 2 minutes 35 seconds.
Those are clean runner measurements, not warm local estimates.

The corpus gate itself now emits versioned timing artifacts. Four independent
compile/codegen audits execute concurrently and preserve the same corpus
selection, limits, failure policy, and browser thresholds. In the successful
run, the parallel audit group was bounded by `llm.c` at 33.55 seconds for source
and 35.71 seconds for distribution. A same-machine comparison measured 46.22
seconds serial versus 22.01 seconds parallel for the audit phase, a 52.4 percent
wall-time reduction. Commit `3b400a86` then split the dominant browser corpus
into two child processes without duplicating runner setup, install, workspace
build, or provisioning. Exact-source CI `29769844668` passed all 159 cases as
an 80/79 split for both source and distribution. The source shards overlapped
at 96.57/96.69 seconds and the distribution shards at 96.48/97.10 seconds;
complete verifier time was 131.15/132.17 seconds. The gate rejects incomplete,
duplicate, failed, skipped, or unexpected shard outcomes before aggregation.

Corpus provisioning is now an explicit gate rather than hidden verifier setup.
It admits the pinned LeetCUDA gitlinks as non-audit metadata while binding their
exact path, commit, and physical state; recovers only an unambiguous dead-owner
lease; skips fetch/mutation for an already confirmed checkout; and probes a
canonical Git/Python host-toolchain capability instead of assuming `/usr/bin`
paths. The focused fixture provisions locally in about 0.44 to 0.48 seconds.
Commit `6e4901ca` additionally gives interrupted snapshots and reservations
canonical target-scoped ownership records and bounded reclamation: at most 32
candidates inspected, four removed, and 4,096 entries considered per candidate
under the target lease. Descriptor-relative no-follow traversal, exact Git-blob
hashes, and UID/root/target/process binding keep ambiguous or foreign residue in
place. These
contracts close the CI gitlink failure, permanent post-`SIGKILL` busy state,
unnecessary cached fetch, missing regression-gate, and undeclared host-tool
layout findings without claiming safety against hostile same-UID leaf swaps in
the final unlink interval. Exact-source CI `29802518928` passed all eight jobs
in about 3 minutes 46 seconds, including both Linux source/dist corpus gates,
Node 20/24/25, required-native checks, Chromium/WebGPU, and Pyodide.

The production compiler controller now runs the exact package-owned raw-Wasm
verifier before preparing the compiler Worker invocation. It transfers a
canonical, hash-bound derivative evidence region, rejects caller-supplied raw
conformance or verifier evidence, and rebinds the exact retained host verifier
authority before minting Worker execution evidence. The regenerated zero-import
compiler Worker bundle is 559,512 bytes with SHA-256
`3fdc7d9a82fd91fa9eb61b0ac0b07fa95aed41cb89607a9cc8212e748c93468a`.
The focused integration passes 9 files/94 tests. The exact authenticated
Worker/verifier/frame/artifact lineage can now prepare one accepted layout
through the shared semantic seam as an opaque observed candidate, while keeping
producer trust, lowering, backend, and release flags false. The current fast
harness passes 66 files/600 tests in 10.62 seconds and completes end to end in
21.80 seconds; the complete post-build compiler suite passes 88 files/1,559
tests in 9.81 seconds. This does not claim a valid compiler-Worker C++ compile,
shared lowering authority, producer trust, legal approval, or release readiness.

The harness is improved, not finished. Provisioning is now a 2,073-line module
plus a 271-line descriptor-relative reclamation helper, still rereads corpus
bytes across multiple admission/snapshot/cleanup passes, lacks a corpus-scale
I/O budget/benchmark, and retains an unavoidable final stat-to-unlink interval
under its cooperative same-UID threat model. Browser/WebGPU remains the
dominant verifier cost even after sharding. These are current quality findings,
not reasons to weaken the semantic or real-device gates.

## 2026-07-24 producer and feedback follow-up

The 90-minute compiler feedback loop is closed. On current `main`,
`verify:browser-clang-wasm:fast` builds the package, verifies the exact build
lock, verifies both zero-import Worker bundles, verifies the deterministic
strict-observation authoring projection, and passes 87 files/751 tests; the
latest Vitest portion takes 15.67 seconds. The complete compiler unit suite
passes 97 files/1,624 tests in 16.79 seconds. The intentionally broader native
boundary, including optimized, UBSan, and platform-gated ASan work, passes 50
files/267 tests with nine explicit skips in 67.62 seconds. Cache-free LLVM/Clang
reproducibility remains an evidence-production lane and runs as two concurrent
isolated jobs; it is not an edit loop.

The browser-local producer is also no longer a parser-only claim. Package-pinned
evidence already covers unchanged C++17/CuTe rank-2/rank-3, strided-slice, and
broadcast source compiling inside the real compiler Worker. The current
extractor source additionally emits exact f32/i32/u32 32-bit ABI facts and the
shared view-copy lowerer executes ranks 1 through 4 through the same canonical
semantic-core and kernels seams. Required headed Chromium on Apple Metal 3
passes a 13-case bit-exact CPU/WebGPU matrix under
`browsergrad.webgpu.view-copy.word32@2`.

Current-lock run `30069614333` closes the producer promotion that was open at
the start of this follow-up. Its two cache-free build steps ran concurrently
for 40 minutes 30 seconds and 45 minutes 55 seconds and produced byte-identical
31,841,008-byte Wasm. The 46-second comparison job verified exact Wasm,
factory, canonical linker-map, native TableGen, runtime closure, and ABI-review
identity. The resulting package Worker was deterministically re-authored.

The strict Chromium lane then compiled eight unchanged C++17/CuTe cases with
the promoted Wasm: f32 ranks 1 through 4, positive strided-slice, read-only
broadcast, i32 rank-2, and u32 broadcast. Individual compilation took
23.869–26.319 seconds. The canonical matrix is package-pinned, cross-binds both
Worker identities and its exact source revision, and is re-derived by the fast
gate. This is a real current-source proof, not a parser or diagnostic
substitute.

This does not erase the remaining production boundaries. Externally rooted
producer signing, file-level redistribution approval, exact-candidate CPU/
real-WebGPU convergence, full distributed-output reproducibility, and final
release authority remain separate and false. The large compiler modules and
native producer files also remain maintainability debt even though exact line
ratchets, dependency-cycle checks, and single-path architecture prevent further
ungoverned growth.

## Why feedback was slow

The locked recipe originally performed a clean LLVM/Clang 22.1.8 build with
`parallelJobs: 1`, no network, no compiler cache, and no reusable build tree.
The first real run reached 97 percent and failed after roughly 86 minutes
because BrowserGrad used C++ exceptions while the target inherited
`-fno-exceptions`. Before this audit, validation and two-build reproducibility
also shared the same expensive workflow shape.

The final-link run's timestamps make the split concrete: native TableGen took
about 9 minutes 36 seconds, reusable Wasm LLVM/Clang dependencies took about 86
minutes 9 seconds, and all BrowserGrad translation units plus the final link
took about 67 seconds. The dominant cost was therefore repeatedly rebuilding
the reusable toolchain, not BrowserGrad iteration, GitHub setup, or artifact
transfer.

A second validation ran for 94 minutes 45 seconds inside the isolated build.
It cleared the exception failure, built the selected Clang libraries, compiled
the first three BrowserGrad translation units, and then exposed a separate
external-project wiring error: the target did not inherit Clang's source and
generated-header include directories. The target now declares both directories,
and the configured-target review rejects either missing `-I` path before the
expensive Wasm build begins.

The next validation exercised that preflight and stopped after 9 minutes 36
seconds of isolated execution, before the Wasm build step existed. Code and
configuration inspection identified CMake's preserved source spelling
`llvm/../clang/include` while the reviewer required its canonical path. Both
verified include candidates are
now resolved with CMake `REAL_PATH` before target attachment. This converted the
same class of integration defect from an 85-to-95-minute discovery into a
roughly ten-minute discovery.

Exact-source CI run `29658935991` is green at `3e17a744`, including Node 20,
24, and 25, the exact-Clang native harness checks, Pyodide integration, and the
real Chromium/WebGPU CUDA corpus. Replacement build run `29658164083` compiled
every BrowserGrad translation unit and reached the final Wasm link after 97
minutes 5 seconds of isolated execution. The link correctly failed closed: the
LLVM/Clang libraries were still built without RTTI while the extractor required
it, and LLVM Support referenced POSIX user, process, resource, and signal-stack
services that a browser Worker must not acquire implicitly. No factory or Wasm
was admitted.

Commit `df849d95` replaces the remaining two-package runtime mount for future
runs with a staged closure containing exact JavaScript/package files and the
lock-listed extractor files. The stager uses no shell, refuses non-private
or overlapping roots, copies through no-follow handles with before/after file
identity checks, bounds file/node/byte counts, seals the tree, and then
re-enumerates and re-hashes it. The container verifies the same closure again;
build-execution evidence v2 records every file identity, and reproducibility
evidence v2 requires the two runtime-closure observations to match. The focused
boundary at `37dacbf6` passes 25 files with 161 tests and 9 intentional platform
skips over a 37-file extractor closure. Run `29658164083` predates this
capability and remains evidence only for the prior package-tree mount.

Commit `37dacbf6` closes the observed link boundary without enabling ambient
host authority. The Wasm LLVM/Clang libraries are built with RTTI; a dedicated
BrowserGrad translation unit resolves unsupported user-database, process,
resource, and signal-stack calls inside the module with deterministic
identity-free or `ENOSYS` behavior; and the linker now reports every unresolved
symbol. The configured-target reviewer independently reads `CMakeCache.txt` and
rejects `LLVM_ENABLE_RTTI=OFF` before compilation. Validation `29661850267`
stopped in the JavaScript verification boundary before root allocation,
acquisition, or build: Linux system headers correctly mark several POSIX
arguments non-null, while the new native test passed null and the shim made
tautological defensive checks under `-Werror`. Commit `60a3d9f9` makes the test
obey those contracts, removes the tautological checks, and advances the exact
build lock to
`bg.cpp.browser-build-input-lock.sha256.74a783b4b283f533ca599b0bf3197346d20f9f2bf90ad612a1f557b5e1df662b`.
The full required-native harness again passes 25 files with 161 tests and 9
intentional platform skips. Replacement validation `29662148150` was cancelled
during its JavaScript boundary, before root allocation or acquisition, when
fast iteration became the immediate priority.

Commits `d0f970ee` and `09dc21a2` separate engineering feedback from release
proof. Clean builds now use four locked jobs, advancing the build-input lock to
`bg.cpp.browser-build-input-lock.sha256.15a71eedf31da3ec752332f90ce74c3d2f7308bb81d3eeba698fc466b683fe14`.
The default manual mode restores only the native TableGen and Wasm LLVM/Clang
build directories under exact cache key
`bg.cpp.clang-wasm-toolchain-cache.sha256.9bd5cebfd49382887f2ee4c2f8841b477446a907e1d27634185ea6b1d4f93b0d`.
The key excludes ordinary extractor implementation edits but includes the
locked LLVM source, Emscripten builder, build recipe, selected libraries, and
extractor CMake graph. Restored contents are treated as untrusted, run only
inside the existing networkless/capability-free container, use no prefix
fallback, and emit a distinct diagnostic schema with `cleanBuild: false` and
`cacheContentsTrusted: false`. Clean validation and two-clean-build
reproducibility remain cache-free. Cold provisioning run `29663494490` is
active; no warm timing or build output is claimed yet.

The same capability moves executor option admission into the exact 26-file
runtime closure, reducing the executor from 2,237 to 2,198 lines, and makes the
local native harness prefer installed Clang 22.1.8 over Apple Clang 15. The full
required-native boundary passes 26 files with 166 tests and 9 intentional skips
in 89.31 seconds locally.

The first exact-source CI run containing that closure exposed an independent
Grad reproducibility defect: parameterized layer constructors used fresh NumPy
generators, so `torch.manual_seed` did not control initialization and a training
fixture could cross its accuracy threshold. Commit `81984eda` routes those
initializers through the seeded global generator and adds all-family contract
coverage. CI `29660471267` has passed the full Pyodide/Grad lane and the
Node 20/24/25 surfaces, including the exact native compiler harness, plus the
real Chromium/WebGPU corpus. This is not Clang-Wasm output evidence, but it
closes a real source of whole-workspace validation noise.

That cost is partly an honest reproducibility cost and partly a harness design
failure: configuration mismatches should not require an almost-complete LLVM
build to discover.

## Scorecard

| Area | Rating | Evidence | Required improvement |
| --- | --- | --- | --- |
| Locked input integrity | Strong for the promoted extractor; release incomplete | Exact LLVM archive, builder image/config, recipe, ABI, extractor source, runtime closure, factory, Wasm, and canonical reproducibility record are content-bound. Run `30069614333` reproduced the current extractor across two clean roots. | Extend reproducibility to the complete distributed output set and retain independent external approval. |
| Build isolation | Strong and exercised | Both current clean builds ran networkless with a read-only root and declared inputs, private work mounts, zero capabilities, and no-new-privileges; source and build paths were distinct. | Keep diagnostic caches outside clean/reproducibility authority and preserve the sealed closure as the recipe evolves. |
| Native and browser semantic coverage | Good for the declared word32 slice | Exact Clang 22 native behavioral/UBSan/ASan lanes coexist with eight real compiler-Worker C++/CuTe cases and 13 bit-exact CPU/WebGPU word32 cases. | Bind exact promoted candidates through the externally rooted producer transition and shared CPU/WebGPU seam; add new dtype/layout profiles only with matching proof. |
| Evidence quality | Strong for narrow authorities; release incomplete | Bounded build records, exact ABI reviews, two-build reproducibility, separate Worker identities, strict source observations, artifacts, semantic candidates, and real-device evidence remain distinct. Trust, legal approval, lowering, backend authorization for the exact source payload, complete-output reproducibility, and release remain false. | Complete only the missing independent authorities; do not collapse build, verifier, compiler-Worker, legal, producer-trust, backend, or release evidence. |
| Feedback speed | Strong engineering loop; intentionally expensive proof lane | The fast gate passes 87 files/751 tests with a 15.67-second Vitest portion; the complete compiler unit suite passes 97 files/1,624 tests in 16.79 seconds. Two cache-free clean builds take 40.5 and 45.9 minutes concurrently, while the strict eight-case browser matrix takes about four minutes. | Keep ordinary edits on focused/fast paths; run clean builds and the complete matrix only when output identities or producer semantics change. |
| Maintainability | Fair with controlled high-risk debt | Architecture checks report zero dependency cycles, legacy-backend leaks, semantic-IR leaks, or C++/CuTe frontend legacy leaks. Exact ratchets contain the 7,972-line WGSL lowerer, 7,178-line semantic IR, 6,999-line reference, 2,235-line build executor, 2,162-line native session, and 1,829-line artifact writer. | Extract existing semantic/effect and harness seams without creating parallel paths, and lower each ratchet as modules split. |
| Delivery truthfulness | Strong | Package admission authenticates exact reproducibility, header, compiler-Worker, verifier-Worker, source, artifact, and candidate identities while explicitly retaining trust, legal, lowering, backend, and release flags as false. | Preserve the separated authority model through external signing, legal approval, exact-payload execution, and final release. |

## Findings closed during this audit

1. The Emscripten exception model is now explicit: BrowserGrad uses
   JavaScript-based `-fexceptions`; native Wasm exception tags remain
   forbidden by the runtime ABI.
2. One-build validation is separate from two-clean-build reproducibility, so a
   configuration failure no longer pays for two full failures.
3. CMake's generated BrowserGrad target flags are reviewed before the Wasm
   compile. The build rejects `-fno-exceptions`, `-fwasm-exceptions`,
   `-fno-rtti`, or either missing Clang include root, and rechecks the generated
   commands after compilation.
4. The generated-target review is enforcement only. It is not serialized as
   durable evidence because the reviewed CMake files are not uploaded.
5. Native test compiler discovery is shared, exact, absolute, executable, and
   required in the primary CI lane.
6. C++ driver invocation preserves the validated `clang++` path instead of
   resolving away the driver basename and accidentally linking as C.
7. Clang classification uses the `__clang__` predefined macro rather than a
   vendor-specific version banner. Required sanitizer lanes cannot silently
   turn into non-Clang skips.
8. CI uses the same exact Clang 22.1.8 API surface as the locked source and
   installs its exact sanitizer runtimes.
9. Main and Clang-build workflow actions are pinned by full commit and use
   Node-24-capable action releases.
10. Grad integration and the kernels real-WebGPU suite are blocking. The
    nondeterministic Grad classifier data is explicitly seeded.
11. Build-lock authoring identities are derived by a checked tool instead of a
    manual sequence of ad hoc hash edits.
12. Architecture governance includes the production harness, native extractor
    closure, tests, dependency cycles, source coverage, and line budgets.
13. Failed builds emit a canonical failure-only observation that binds the
    typed error to every available immutable partial log. Version 2 preserves
    a bounded four-error typed cause chain, marks cycles/truncation, and does
    not invoke accessors. It explicitly grants no successful-build, output,
    ABI, reproducibility, or release authority.
14. Node 25 has a dedicated compatibility lane for the workspace build,
    architecture checks, ordinary test suites, compiler typecheck and lint, and
    the Node-only Docker-shell contract. Exact Clang and sanitizer semantics
    remain in the separately locked Node 24 lane.
15. The LLVM external target explicitly binds Clang's public source, generated
    binary, and private AST source include roots and depends on Clang's
    generated-header targets. Missing include wiring now fails
    configured-target review before the expensive Wasm compile instead of after
    the selected Clang libraries have built.
16. The four largest compiler-core modules, build executor, native compile
    session, and artifact writer have exact file-specific line ratchets.
    Duplicate, invalid, or stale ratchet entries fail closed.
17. The build container no longer mounts the whole checkout. The intermediate
    two-package boundary removed unrelated repository read authority, and the
    runner observed both effective mount modes before execution.
18. The VFS call-ceiling test no longer performs one million full memory-
    validated calls. The exact ABI-owned budget transition has direct boundary
    coverage, while live-handle and aggregate-byte ceilings remain integration
    tests. Exact-source Node 20 CI passes the revised suite.
19. Future builds no longer mount either complete package tree. An exact staged
    runtime closure contains 25 required JavaScript/package files plus the 36
    lock-listed extractor files. It rejects symlinks, changes, missing files,
    undeclared files/directories, and resource overruns; proves the isolated
    runner imports from that tree; and binds the closure into build and
    reproducibility evidence v2.

## Recommended decomposition order

Refactoring should preserve the current single execution path and extract
existing responsibilities in this order:

1. Split the build executor's source/input snapshot and trust checks from its
   process/log/output sealing, then split atomic sidecar installation from both.
   These are already distinct authority boundaries; they do not require a new
   build path or schema.
2. Split the native compile session's bounded canonical-JSON parser from typed
   profile decoding and typed request decoding. Keep one shared budget resource
   and one `DecodedCompileSession` construction point.
3. Split the artifact writer's canonical JSON serializer, source/identity graph
   projection, and diagnostic projection from final artifact assembly. Keep the
   existing writer as the sole entry point until byte-for-byte parity tests pass.
4. Lower the installed file-specific ratchets after each extraction so
   decomposition cannot regress into a second set of monoliths.

## Broader compiler shape

The current architecture scan reports zero dependency cycles, zero legacy
backend leaks, zero semantic-IR representation leaks, and zero C++/CuTe legacy
leaks. Those are meaningful structural strengths. They do not make the module
shape healthy.

The largest production modules are `semantic_wgsl.ts` at 7,972 lines,
`semantic_ir.ts` at 7,178, `semantic_reference.ts` at 6,999, and `analyzer.ts`
at 5,437. The general source budget was reduced from 8,500 to 8,000, and each of
these modules is now pinned to its exact current line count. The executor is
pinned at 2,237 lines, compile session at 2,121, and artifact writer at 1,784.
These ratchets prevent further growth but are still not design-quality targets.
The modules need seam-specific decomposition after the active Gate 3 execution
chain is stabilized; a broad rewrite during the first Wasm/Worker integration
would mix correctness risk with organizational work.

The build container's writable authority is narrow, and its unrelated checkout
and complete-package read authority have been removed for future runs. The
workflow stages only the exact runner dependency graph and lock-listed C++
source, mounts that tree read-only, independently enumerates it inside the
container, and records every file identity. No completed build has yet exercised
this v2 boundary, so the older active validation cannot retroactively grant it
execution evidence. Release-grade output identity still requires a successful
v2 build, independent raw-Wasm review, and two matching clean v2 builds.

## Open high-priority findings

1. Obtain external approval for the exact header-pack file map, notices, CUDA
   index, and upstream evidence, then bind the approved identities into the
   asset chain.
2. Admit the browser-build signer through an independent package-owned trust
   root and verify one external exact-build statement; synthetic package keys
   remain insufficient.
3. Execute one valid browser-local C++ request through the verifier Worker,
   compiler Worker, verified Artifact V3, shared semantic lowering, and real
   WebGPU.
4. Complete the full release-output reproducibility set without widening the
   already separated build, ABI, legal, producer-trust, or release authorities.
5. Add bounded descriptor-relative reclamation for owned failure residue,
   close or explicitly bound the final validated-inode unlink interval, and
   reduce repeated corpus-byte passes with corpus-scale evidence.
6. Shard the serial browser/WebGPU case phase while preserving aggregate case
   coverage, skip policy, timeouts, and performance thresholds.
7. Decompose the corpus provisioning module, build executor, compile session,
   artifact writer, and the four largest compiler-core modules along existing
   authority and semantic seams.

## Capability boundary after the audit

The project can now fail fast on the known exception/RTTI/include wiring class,
preserve actionable bounded failure causes, prevent the seven named monoliths
from growing, run the owned native producer against exact Clang 22, enforce its
sanitizer lanes, derive build-lock authoring identities, distinguish validation
from reproducibility, execute the package-owned verifier Worker, and provision
exact real-world corpora through an explicit portable host-toolchain contract.
It still cannot honestly claim that the C++/CuTe frontend is available in the
browser product. That claim begins only after approved exact packs and producer
trust feed one valid compiler-Worker C++ request through verified Artifact V3,
shared lowering, and real WebGPU.

## 2026-07-23 real-browser observation addendum

The harness now has a distinct post-build lane for already-produced exact
Clang-Wasm and header packs. Host preflight rejects non-canonical or symbolic
paths, hashes all six files concurrently, rechecks each open file identity
after hashing, and admits only the package-pinned two-clean-build Wasm. A
dedicated Vitest/Chromium configuration exposes those exact files through
bounded no-store routes and runs only the real C++/CuTe observation.

The package Worker now projects the deepest bounded authenticated error in its
cause chain. VFS observations retain a capped, deterministic set of not-found
paths, and compile-status failures include the last bounded Emscripten module
log lines when any exist. Focused unit tests cover the error projection,
lookup-miss accounting, truncation, disposal receipt, and unchanged success
authority.

The exact current inputs complete preflight in under one second and the browser
observation in about seven seconds. Chromium verifies the 31,641,377-byte Wasm,
installs five packs containing 5,768 files, verifies the raw interface in the
separate verifier Worker, launches the exact compiler Worker, and receives one
authenticated failure terminal. The terminal is now precise:
`BG-COMPILER-CPP-CUTE-BROWSER-WASM-COMPILER-COMPILE-STATUS` at
`$.runtime.compile`. A diagnostic-only same-realm run established that Clang
opened the source plus 109 installed headers and completed one semantic pass;
its 180 VFS misses are normal ordered search probes rather than a missing
required header. That diagnostic run is not retained as production evidence.

This changes the active engineering blocker without widening capability claims.
The build, asset transport, raw-Wasm verifier, VFS installation, and Worker
failure protocol are no longer the unknowns. The remaining local blocker is
inside the producer or rejected Artifact V3 path. The observation command
accepts and records that exact current blocker; the separate strict verification
command fails until an authenticated Artifact V3 terminal is produced.
