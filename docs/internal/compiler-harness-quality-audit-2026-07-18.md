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

## Why feedback was slow

The locked recipe intentionally performs a clean LLVM/Clang 22.1.8 build with
`parallelJobs: 1`, no network, no compiler cache, and no reusable build tree.
The first real run reached 97 percent and failed after roughly 86 minutes
because BrowserGrad used C++ exceptions while the target inherited
`-fno-exceptions`. Before this audit, validation and two-build reproducibility
also shared the same expensive workflow shape.

The failed run's timestamps make the split concrete: workflow setup, verified
source acquisition, and builder acquisition took 2 minutes 55 seconds; the
isolated build step took 86 minutes 45 seconds. Its Wasm CMake configure report
accounted for about 111 seconds. The dominant cost was therefore the clean,
single-job LLVM/Clang compile, not GitHub setup or artifact transfer.

A second validation ran for 94 minutes 45 seconds inside the isolated build.
It cleared the exception failure, built the selected Clang libraries, compiled
the first three BrowserGrad translation units, and then exposed a separate
external-project wiring error: the target did not inherit Clang's source and
generated-header include directories. The target now declares both directories,
and the configured-target review rejects either missing `-I` path before the
expensive Wasm build begins.

That cost is partly an honest reproducibility cost and partly a harness design
failure: configuration mismatches should not require an almost-complete LLVM
build to discover.

## Scorecard

| Area | Rating | Evidence | Required improvement |
| --- | --- | --- | --- |
| Locked input integrity | Good but incomplete | Exact LLVM archive, builder manifest/config, extractor source closure, recipe, ABI, and resource identities are checked. The success receipt does not yet bind the harness implementation revision. | Bind the harness revision, successful output, and reviewed runtime projections before distribution. |
| Build isolation | Strong containment; incomplete hermeticity | No network, read-only root and inputs, private work mount, zero capabilities, and no-new-privileges are observed. The container can still read the whole checkout and installed workspace. | Minimize or independently bind the read-only workspace closure and preserve the same boundary in reproducibility runs. |
| Native semantic coverage | Good but incomplete | Exact Clang 22 pass integration plus native behavioral, UBSan, and ASan lanes are required in CI. | Keep the exact-version lane blocking and add executed-Wasm coverage. |
| Evidence quality | Mixed | Bounded immutable logs, raw-Wasm inspection, authority-specific records, and a failure-only observation for available partial logs exist. | Complete successful and reproducible build evidence. |
| Feedback speed | Weak | The clean serial toolchain build dominates iteration time. | Use cheap target-configuration review, then validate a safe higher parallelism only through reproducibility evidence. |
| Maintainability | Weak | The build executor remains over 2,200 lines; native compile session is 2,121 lines; artifact writer is 1,784 lines. Existing compiler-core modules range from roughly 5,400 to 8,000 lines. | Extract effect boundaries and semantic subcomponents without creating parallel execution paths; replace the permissive 8,500-line source ceiling with ratcheting per-module budgets. |
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
    typed error to every available immutable partial log. It explicitly grants
    no successful-build, output, ABI, reproducibility, or release authority.
14. Node 25 has a dedicated compatibility lane for the workspace build,
    architecture checks, ordinary test suites, compiler typecheck and lint, and
    the Node-only Docker-shell contract. Exact Clang and sanitizer semantics
    remain in the separately locked Node 24 lane.
15. The LLVM external target explicitly binds Clang's source and generated
    include roots and depends on Clang's generated-header targets. Missing
    include wiring now fails configured-target review before the expensive Wasm
    compile instead of after the selected Clang libraries have built.

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
4. Tighten line budgets after each extraction so decomposition cannot regress
   into a second set of monoliths.

## Broader compiler shape

The current architecture scan reports zero dependency cycles, zero legacy
backend leaks, zero semantic-IR representation leaks, and zero C++/CuTe legacy
leaks. Those are meaningful structural strengths. They do not make the module
shape healthy.

The largest production modules are `semantic_wgsl.ts` at 7,972 lines,
`semantic_ir.ts` at 7,178, `semantic_reference.ts` at 6,999, and `analyzer.ts`
at 5,437. The general source budget was 8,500 lines and has been ratcheted to
8,000, leaving only 28 lines of headroom for the largest module. This prevents
further unbounded growth but is still not a design-quality target. These modules
need seam-specific decomposition and per-module ratchets after the active Gate
3 execution chain is stabilized; a broad rewrite during the first Wasm/Worker
integration would mix correctness risk with organizational work.

The build container's writable authority is narrow, but its readable authority
is broader than the lock: the workflow mounts the complete checkout and
installed workspace read-only so the verification code and package graph can
execute. Network isolation prevents remote exfiltration, and current receipts
grant no output-identity or release authority. Even so, release-grade
hermeticity requires either a minimal staged harness closure or an independently
recorded revision/content identity for every readable harness input.

## Open high-priority findings

1. Complete and independently inspect one locked Clang-Wasm build.
2. Compare the raw Wasm interface with the runtime ABI, then repin only the
   independently observed exact imports, exports, tables, globals, tags, and
   custom sections.
3. Run two distinct clean builds and prove byte equality for the factory,
   Wasm, link map, and admitted output set.
4. Bundle the reviewed Emscripten factory and Worker module as package-owned
   bytes; do not accept caller-supplied code or ambient fetch authority.
5. Execute one valid browser-local C++ request through the Worker, verified
   Artifact V3, shared semantic lowering, and real WebGPU.
6. Acquire the exact header packs and close per-file license and notice
   review.
7. Decompose the build executor, compile session, artifact writer, and the four
   largest compiler-core modules along existing authority and semantic seams.
8. Minimize or content-bind the read-only workspace/harness closure and record
   its revision before granting output-identity, provenance, or release
   authority.

## Capability boundary after the audit

The project can now fail fast on the known exception/RTTI wiring class, run
the owned native producer against exact Clang 22, enforce its sanitizer lanes,
derive build-lock authoring identities, and distinguish validation from
reproducibility. It still cannot honestly claim that the C++/CuTe frontend is
available in the browser product. That claim begins only after the successful
build, ABI, Worker, semantic-lowering, and real-WebGPU chain is complete.
