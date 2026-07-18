# Compiler Harness Quality Audit — 2026-07-18

## Verdict

The BrowserGrad compiler harness is not a fake execution path, but its current
shape is uneven. Its input-integrity and container-isolation boundaries are
strong. Its evidence vocabulary is deliberately conservative. Its feedback
loop and maintainability are not yet at the same standard.

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

That cost is partly an honest reproducibility cost and partly a harness design
failure: configuration mismatches should not require an almost-complete LLVM
build to discover.

## Scorecard

| Area | Rating | Evidence | Required improvement |
| --- | --- | --- | --- |
| Locked input integrity | Strong | Exact LLVM archive, builder manifest/config, extractor source closure, recipe, ABI, and resource identities are checked. | Bind successful output and reviewed runtime projections before distribution. |
| Build isolation | Strong | No network, read-only root and inputs, private work mount, zero capabilities, and no-new-privileges are observed. | Preserve the same boundary in reproducibility runs. |
| Native semantic coverage | Good but incomplete | Exact Clang 22 pass integration plus native behavioral, UBSan, and ASan lanes are required in CI. | Keep the exact-version lane blocking and add executed-Wasm coverage. |
| Evidence quality | Mixed | Bounded immutable logs, raw-Wasm inspection, and authority-specific records exist. | Add a structured failed-build receipt and complete successful/reproducible build evidence. |
| Feedback speed | Weak | The clean serial toolchain build dominates iteration time. | Use cheap target-configuration review, then validate a safe higher parallelism only through reproducibility evidence. |
| Maintainability | Weak | The build executor remains over 2,200 lines; native compile session is 2,121 lines; artifact writer is 1,784 lines. | Extract effect boundaries and semantic subcomponents without creating parallel execution paths. |
| Delivery truthfulness | Good | Production Worker/controller paths remain capability-blocked; CPU, parser, Wasm, Worker, and WebGPU claims are distinct. | Do not relax blockers until reviewed factory/Wasm bytes execute in the package Worker. |

## Findings closed during this audit

1. The Emscripten exception model is now explicit: BrowserGrad uses
   JavaScript-based `-fexceptions`; native Wasm exception tags remain
   forbidden by the runtime ABI.
2. One-build validation is separate from two-clean-build reproducibility, so a
   configuration failure no longer pays for two full failures.
3. CMake's generated BrowserGrad target flags are reviewed before the Wasm
   compile. The build rejects `-fno-exceptions`, `-fwasm-exceptions`, or
   `-fno-rtti`, and rechecks the generated commands after compilation.
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
7. Decompose the build executor, compile session, and artifact writer along
   existing authority and semantic seams.
8. Add a canonical failed-build receipt that binds the typed failure to every
   immutable partial log without granting successful-build authority.

## Capability boundary after the audit

The project can now fail fast on the known exception/RTTI wiring class, run
the owned native producer against exact Clang 22, enforce its sanitizer lanes,
derive build-lock authoring identities, and distinguish validation from
reproducibility. It still cannot honestly claim that the C++/CuTe frontend is
available in the browser product. That claim begins only after the successful
build, ABI, Worker, semantic-lowering, and real-WebGPU chain is complete.
