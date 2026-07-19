# Compiler Harness Quality Audit — 2026-07-18

## Verdict

The BrowserGrad compiler harness is not a fake execution path, but its current
shape is uneven. Its container containment is strong and its locked input
integrity is good but incomplete. Its evidence vocabulary is deliberately
conservative. Its feedback loop and maintainability are not yet at the same
standard.

The "hacked upon" impression is justified by accumulated orchestration,
manual authoring steps, duplicated native-test discovery, and several large
modules. It is not justified as a claim that successful parser tests are being
presented as real Clang, Wasm, Worker, or WebGPU execution. Those authorities
remain explicitly separate and incomplete.

No completed Clang-Wasm build, raw-Wasm ABI conformance, two-build
reproducibility result, valid production Worker compile, or browser-local C++
execution is claimed by this audit.

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
the workflow itself produces. One exact 572,755-byte zero-import Worker graph
is now authored twice, hash-pinned, checked in the fast gate, and loaded from
verified Blob bytes in Chromium. Production Worker execution, current-ABI clean
proof, licensed header packs, and two-build reproducibility remain open.

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
| Locked input integrity | Good but incomplete | Exact LLVM archive, builder manifest/config, recipe, ABI, extractor source, and 26-file harness runtime closure are checked and content-bound in build evidence v2. Successful factory/Wasm identities and reviewed ABI projections remain unproven. | Complete a build under the v2 boundary, then bind reviewed outputs before distribution. |
| Build isolation | Strong containment; new boundary not yet exercised by a completed build | No network, read-only root and declared inputs, private work mount, zero capabilities, and no-new-privileges are observed. Future runs mount only the sealed exact runtime/extractor closure instead of checkout or package trees. | Preserve the same closure in a successful validation and both reproducibility builds. |
| Native semantic coverage | Good but incomplete | Exact Clang 22 pass integration plus native behavioral, UBSan, and ASan lanes are required in CI. | Keep the exact-version lane blocking and add executed-Wasm coverage. |
| Evidence quality | Mixed | Bounded immutable logs, raw-Wasm inspection, authority-specific records, and a failure-only observation for available partial logs exist. | Complete successful and reproducible build evidence. |
| Feedback speed | Good for local harness work; native link remains minutes | The canonical Node 25 gate passes 30 files/337 tests in 29.24 seconds. Cached source validation is 4 minutes 27 seconds to 5 minutes 4 seconds; clean/reproducibility gates stay cache-free and authoritative. | Reduce the cached Emscripten system-library regeneration cost without allowing diagnostic cache state to satisfy clean proof. |
| Maintainability | Weak but improving | The build executor is down to 2,198 lines; native compile session is 2,121 lines; artifact writer is 1,784 lines. Existing compiler-core modules range from roughly 5,400 to 8,000 lines. Exact per-module ratchets prevent all named monoliths from growing. | Continue extracting effect boundaries and semantic subcomponents without creating parallel execution paths, lowering each ratchet as code moves out. |
| Delivery truthfulness | Good | Production Worker/controller paths remain capability-blocked; CPU, parser, Wasm, Worker, and WebGPU claims are distinct. | Do not relax blockers until reviewed factory/Wasm bytes execute in the package Worker. |

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
15. The LLVM external target explicitly binds Clang's source and generated
    include roots and depends on Clang's generated-header targets. Missing
    include wiring now fails configured-target review before the expensive Wasm
    compile instead of after the selected Clang libraries have built.
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

1. Complete and independently inspect one locked Clang-Wasm build.
2. Compare the raw Wasm interface with the runtime ABI, then repin only the
   independently observed exact imports, exports, tables, globals, tags, and
   custom sections.
3. Run two distinct clean builds and prove byte equality for the factory,
   Wasm, link map, and admitted output set.
4. Reconcile the pinned bundle with the active current-source clean factory,
   then wire only a captured production platform adapter and package-owned
   invocation issuer; do not accept caller-supplied code or ambient fetch authority.
5. Execute one valid browser-local C++ request through the Worker, verified
   Artifact V3, shared semantic lowering, and real WebGPU.
6. Acquire the exact header packs and close per-file license and notice
   review.
7. Decompose the build executor, compile session, artifact writer, and the four
   largest compiler-core modules along existing authority and semantic seams.
8. Exercise the content-bound runtime closure in a successful validation and
   both clean reproducibility builds before granting output-identity,
   provenance, or release authority.

## Capability boundary after the audit

The project can now fail fast on the known exception/RTTI/include wiring class,
preserve actionable bounded failure causes, prevent the seven named monoliths
from growing, run the owned native producer against exact Clang 22, enforce its
sanitizer lanes, derive build-lock authoring identities, and distinguish
validation from reproducibility. It still cannot honestly claim that the
C++/CuTe frontend is available in the browser product. That claim begins only
after the successful build, ABI, Worker, semantic-lowering, and real-WebGPU
chain is complete.
