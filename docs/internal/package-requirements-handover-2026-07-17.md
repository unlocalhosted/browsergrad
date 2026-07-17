# Package Requirements Implementation Handover — 2026-07-17

- **State:** paused by user
- **Goal:** implement the architecture in
  [`package-requirements-lld.md`](../platform/package-requirements-lld.md)
- **Durable status ledger:**
  [`package-requirements-implementation-ledger.md`](./package-requirements-implementation-ledger.md)
- **Branch:** `main`
- **Last verified green baseline:** `cf94cc97`
- **Paused code checkpoint:** `7b7757fa` (committed on `main`; known red)
- **Active gate:** Gate 3 — real C++/CuTe frontend slice
- **Gate state:** in progress; exit criteria are not met

This document is the recovery point for resuming the paused goal. It separates
the pushed, verified baseline from incomplete local work. The LLD remains the
normative target; the implementation ledger remains the chronological record.

## Runtime Direction — Do Not Reinterpret

The portable path is:

```text
user C++/CUDA/CuTe source
  -> pinned Clang/extractor running as Wasm in a dedicated browser Worker
  -> verified BrowserGrad semantic artifact
  -> shared semantic verification/lowering
  -> CPU reference or WGSL/WebGPU execution
```

The user's C++ program does **not** execute as Wasm. The Clang-based compiler
and extractor execute as Wasm and emit verified portable meaning. Docker is
optional maintainer-side build/reproducibility or native-AOT parity machinery;
it is not in the browser runtime graph and cannot satisfy Gate 3.

## Gate State

| Gate | State at pause | Meaning |
|---|---|---|
| Gate 0 | `verified` | Freeze and inventory complete. |
| Gate 1 | `verified` | Value/layout core and wire foundation complete. |
| Gate 2 | `verified` | Initial shared compiler/JIT/CPU/WebGPU view slice complete. |
| Gate 3 | `in-progress` | Foundations are pushed; typed producer integration is unfinished. |
| Gates 4–7 | `not-started` | No implementation in this workstream. |

## Pushed Baseline

Before the explicit checkpoint push, `HEAD` and `origin/main` both pointed to
`cf94cc97`. Its latest coherent green slices are:

| Commit | Slice |
|---|---|
| `cf94cc97` | Record the Gate 3 semantic-policy foundation. |
| `9b9fd341` | Add native compile-policy primitives. |
| `83aee3a9` | Close the semantic-adapter policy. |
| `d1146cea` | Reject reentrant runtime reset. |
| `34f4e5a9` | Unify virtual-path validation. |
| `a7e811ec` | Bind the exact source-closure subset. |
| `7ddddf41` | Record the native-decode foundation. |
| `719cca9b` | Add bounded native-decode primitives. |

The pushed baseline last proved:

- Compiler suite: 68 files and 1,426 tests passed.
- Build-plan/native suite: 10 files and 73 tests passed, with six documented
  platform sanitizer skips.
- Compiler typecheck, lint, codegen parity, and whitespace checks passed.
- A real Chromium module Worker shell test passed 1 file and 1 test.

This evidence proves bounded components and contracts. It does not prove a
real Clang-Wasm build, a valid Worker compile launch, browser-local C++ source
compilation, semantic-artifact production, or C++-originated WebGPU execution.

The paused work was then preserved in these coherent commits:

| Commit | Checkpoint slice |
|---|---|
| `0adc5a97` | Align repository documentation with the browser-local semantics-first direction. |
| `74e8061f` | Isolate validated profile/request frame regions at the runtime callback. |
| `d45077bd` | Checkpoint diagnostic normalization, profile, and asset contracts. |
| `b8061280` | Add the sealed typed native compiler invocation. |
| `7b7757fa` | Checkpoint the incomplete typed native compile session. |

These commits preserve resumable work; they do not upgrade the evidence or
capability tier of the green baseline.

## Committed Pause Checkpoint

The checkpoint deliberately contains incomplete Gate 3 work and is known red.
It was split into the commits above for recovery. Do not treat its presence on
`main` as capability or release evidence.

### Coherent components that were independently green

1. **Validated runtime input regions**
   - The compile callback receives only validated profile/request regions.
   - Raw frame headers, padding, and unvalidated bytes are withheld.
   - The holder is non-default-constructible and non-copyable/non-movable.
   - Focused native result before pause: 1 file, 2 tests passed, 1 platform
     AddressSanitizer skip.

2. **Sealed native invocation**
   - Owns one ordered, typed compiler-option sequence matching profile order.
   - Enforces exact ordinals, one syntax-only option, and one error-limit
     option while preserving option interleaving.
   - Exposes no raw command string, shell, or caller-controlled argv.
   - Pins exact device-pass then host-pass prefixes.
   - The invocation native test remained green in the final focused run.

3. **TypeScript diagnostic-normalization authority**
   - Pins exact Clang `22.1.8` behavior and a versioned normalization resource.
   - Diagnostic IDs are recomputable only from serialized Artifact V3 fields,
     compilation-contract hash, and owner pass.
   - Exact duplicates collapse; built-in Clang codes use
     `clang:diag-<uint32>`.
   - The focused diagnostic test was green in the final targeted run, but the
     surrounding profile/asset integration remained red.

### Interrupted or internally inconsistent components

1. **Frontend profile and compilation contract**
   - Local profile is at `2.6`; compilation contract is at `1.2`.
   - Diagnostic-normalization identity is bound into the language contract.
   - `toolchain.compiler.resourceDirectoryVirtualPath` was added immediately
     before pause.
   - The compiler-resource include root must equal
     `${resourceDirectoryVirtualPath}/include` and must be a system include.
   - Exact profile and manifest hashes have not all been repinned.

2. **Browser asset contract**
   - Partially requires a `diagnostic-normalization-manifest` asset.
   - Installation fixtures and assertions were not completed.
   - Asset ordering changed, invalidating at least one index-based assertion.

3. **Native compile session**
   - A large bounded canonical profile/request decoder exists locally.
   - It was being extended for compiler version, resource-directory path, and
     extraction limits when work stopped.
   - Its native schema currently trails the TypeScript profile and fails at
     profile offset 4667.

4. **Native diagnostics generation**
   - Only partial TypeScript declaration/JavaScript codegen scripts exist.
   - There is no generated policy include, native diagnostics normalizer, or
     native diagnostics test yet.
   - No `BrowserGradCppCuteDiagnostics.h/.cpp` exists.

5. **Producer integration**
   - Invocation, compile session, runtime, VFS observer, temporal-macro policy,
     diagnostic state, and the review-only Clang action are not wired into one
     producer path.
   - Artifact V3 writing is still a fail-closed placeholder.
   - CMake/source-lock/build-lock identities do not include the checkpoint WIP.

## Current Verification State

The verified baseline at `cf94cc97` is green. The committed checkpoint through
`7b7757fa` is red by construction.

### Targeted TypeScript run

```sh
pnpm --filter @unlocalhosted/browsergrad-compiler exec vitest run \
  tests/compiler/cpp_cute_diagnostic_normalization.test.ts \
  tests/compiler/cpp_cute_frontend_profile.test.ts \
  tests/compiler/cpp_cute_frontend_policy.test.ts \
  tests/compiler/cpp_cute_browser_assets.test.ts \
  tests/compiler/cpp_cute_browser_asset_installation.test.ts
```

Result: 5 files total; 2 passed and 3 failed. Of 74 tests, 56 passed and 18
failed.

Known failures:

- Two stale exact profile-hash expectations after adding the compiler resource
  directory:
  - AOT: `1a91c95ac11638b04bd62c8f7c060152d6f654434274fc6bfdf815daecdad752`
  - browser:
    `21465742ae328a68adad4a23a7930f4581f8dc9f5040d281dcf8daf5cdc75e07`
- Browser-asset manifest identity changed to
  `bg.cpp.browser-assets.sha256.ab13e8c3e323f31909b745cf96e5d168a0111a056bb5ffd66a59759c31e81938`.
- The runtime-ABI asset moved from fixture index 7 to index 8.
- A root-mount mutation now violates the exact
  `resourceDirectoryVirtualPath/include` invariant.
- All 13 asset-installation cases use fixtures that lack the newly required
  diagnostic-normalization manifest.

### Focused native run

```sh
pnpm --filter @unlocalhosted/browsergrad-compiler exec vitest run \
  --config scripts/cpp_cute_browser_build/vitest.config.ts \
  scripts/cpp_cute_browser_build/cpp_cute_browser_invocation_native.test.ts \
  scripts/cpp_cute_browser_build/cpp_cute_browser_compile_session_native.test.ts \
  scripts/cpp_cute_browser_build/cpp_cute_browser_runtime_native.test.ts
```

Result: 3 files total; 2 passed and 1 failed. Of 9 tests, 4 passed, 2 failed,
and 3 were platform sanitizer skips.

- Invocation and runtime pass.
- Compile-session strict and UndefinedBehaviorSanitizer cases fail with an
  invalid frame/schema status at profile offset 4667 because the native decoder
  does not yet consume the new compiler resource-directory field.

Do not use the earlier concurrent full-package run as current proof. Run the
focused lanes first after restoring schema consistency, then run the complete
package gates.

## Checkpoint File Ownership

### User-owned unrelated changes

These existed before or outside the Gate 3 slice and were committed in
`0adc5a97` only because the user explicitly requested that everything be
pushed. Do not mix future changes to them into compiler implementation slices:

```text
AGENTS.md
DEVELOPMENT.md
README.md
RTK.md
docs/internal/vision.md
docs/platform/agent-consumption-guide.md
docs/platform/cuda-compatibility-spine.md
docs/platform/curriculum-platform-architecture.md
packages/browsergrad-compiler/README.md
```

### Gate 3 checkpoint files

```text
packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_extractor_source.test.ts
packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_runtime_native_test.cpp
packages/browsergrad-compiler/scripts/cpp_cute_browser_build/extractor/BrowserGradCppCuteArtifactV3.cpp
packages/browsergrad-compiler/scripts/cpp_cute_browser_build/extractor/BrowserGradCppCuteArtifactV3.h
packages/browsergrad-compiler/scripts/cpp_cute_browser_build/extractor/BrowserGradCppCuteRuntime.cpp
packages/browsergrad-compiler/scripts/cpp_cute_browser_build/extractor/BrowserGradCppCuteRuntime.h
packages/browsergrad-compiler/src/cpp_cute_browser_assets.ts
packages/browsergrad-compiler/src/cpp_cute_frontend_profile.ts
packages/browsergrad-compiler/tests/compiler/cpp_cute_browser_assets.test.ts
packages/browsergrad-compiler/tests/compiler/cpp_cute_frontend_policy.test.ts
packages/browsergrad-compiler/tests/compiler/cpp_cute_frontend_profile.test.ts
packages/browsergrad-compiler/tests/compiler/support/cpp_cute_browser_asset_fixtures.ts
packages/browsergrad-compiler/tests/compiler/support/cpp_cute_frontend_fixtures.ts
packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_compile_session_native.test.ts
packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_compile_session_native_test.cpp
packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_diagnostics_policy_codegen.d.mts
packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_diagnostics_policy_codegen.mjs
packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_invocation_native.test.ts
packages/browsergrad-compiler/scripts/cpp_cute_browser_build/cpp_cute_browser_invocation_native_test.cpp
packages/browsergrad-compiler/scripts/cpp_cute_browser_build/extractor/BrowserGradCppCuteCompileSession.cpp
packages/browsergrad-compiler/scripts/cpp_cute_browser_build/extractor/BrowserGradCppCuteCompileSession.h
packages/browsergrad-compiler/scripts/cpp_cute_browser_build/extractor/BrowserGradCppCuteInvocation.cpp
packages/browsergrad-compiler/scripts/cpp_cute_browser_build/extractor/BrowserGradCppCuteInvocation.h
packages/browsergrad-compiler/src/cpp_cute_diagnostic_normalization.ts
packages/browsergrad-compiler/src/resources/cpp_cute_diagnostic_normalization_v1.ts
packages/browsergrad-compiler/tests/compiler/cpp_cute_diagnostic_normalization.test.ts
```

The worktree should be clean after the checkpoint push. Before resuming, run
`git status --short`; any new dirty files need ownership classification before
editing or staging.

## Decisions That Must Survive Resume

- Browser-local Clang/extractor Wasm is the primary producer; native and Docker
  lanes are optional parity/build aids only.
- Diagnostic normalization is a versioned semantic resource bound by profile
  `2.6` and compilation contract `1.2`.
- The compiler resource directory is explicit canonical data. Never infer it
  by taking the parent of an include-root string.
- Compiler options are typed and preserve exact canonical profile order. Do not
  expose a raw argv or reconstruct ordering by option family.
- Stable diagnostic IDs use only serialized, reproducible Artifact V3 fields,
  compilation-contract identity, and owner pass. Exact duplicates collapse.
- The browser asset manifest must bind the exact diagnostic-normalization
  resource and its identity.
- Native virtual-path validation is duplicated across components. Consolidate
  it behind one source-closed native helper before adding more consumers.
- Each Clang pass needs fresh preprocessor, VFS-observation, include-edge,
  diagnostic, and semantic state. Device pass runs before host pass.
- No producer success is legal until the canonical Artifact V3 writer commits
  one fully verified bounded result.

## Resume Order

1. Run `git status --short`; preserve all user-owned files above.
2. Finish `resourceDirectoryVirtualPath` through the native compile-session
   decoder and all TypeScript/native fixtures.
3. Repin profile identities only after schema parity is green.
4. Complete the diagnostic-normalization asset in the browser manifest,
   installation fixtures, and order-independent tests.
5. Finish deterministic diagnostic-policy codegen and native normalization;
   add strict native parity/adversarial tests.
6. Re-run the focused diagnostic/profile/asset and compile-session lanes.
7. Complete compile-session accessors and connect the sealed invocation to the
   validated runtime callback.
8. Consolidate native canonical virtual-path validation.
9. Add every completed producer source to CMake and the exact source closure,
   then repin build identities. Do this only after the slice is internally
   green.
10. Wire fresh policy/VFS/diagnostic/semantic state into device-first then
    host-second Clang passes.
11. Replace the placeholder with the canonical Artifact V3 writer and verify
    strict decode/ownership/failure behavior.
12. Acquire and independently review licensed header packs; execute two clean
    pinned Clang-Wasm builds and repin observed identities.
13. Bundle the reviewed Emscripten factory and self-contained Worker bytes.
14. Prove one unmodified browser-local fixture with network and Docker absent:
    source -> Clang-Wasm -> verified semantic artifact -> shared lowering ->
    real WebGPU.

Commit and push each coherent green slice independently. Never mix the
user-owned files with Gate 3 commits.

## Validation Sequence After Restoring Green

```sh
pnpm --filter @unlocalhosted/browsergrad-compiler exec vitest run \
  tests/compiler/cpp_cute_diagnostic_normalization.test.ts \
  tests/compiler/cpp_cute_frontend_profile.test.ts \
  tests/compiler/cpp_cute_frontend_policy.test.ts \
  tests/compiler/cpp_cute_browser_assets.test.ts \
  tests/compiler/cpp_cute_browser_asset_installation.test.ts

pnpm --filter @unlocalhosted/browsergrad-compiler exec vitest run \
  --config scripts/cpp_cute_browser_build/vitest.config.ts \
  scripts/cpp_cute_browser_build/cpp_cute_browser_invocation_native.test.ts \
  scripts/cpp_cute_browser_build/cpp_cute_browser_compile_session_native.test.ts \
  scripts/cpp_cute_browser_build/cpp_cute_browser_runtime_native.test.ts

pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler
pnpm --filter @unlocalhosted/browsergrad-compiler run typecheck
```

Run the build-plan/native gate, source lint, codegen-parity check, browser
Worker gate, and real-WebGPU gate when the files those lanes own are touched.
For the eventual real browser capability proof, require WebGPU and retain exact
source revision, input identities, Worker/Wasm identities, artifact identity,
and terminal evidence.

## Claims That Remain False

At this checkpoint, do not claim any of the following:

- Gate 3 complete.
- A real pinned Clang-Wasm extractor has been built.
- A valid browser Worker compile has run.
- User C++ has executed in the browser.
- User C++ is compiled into and executed as Wasm.
- A production Artifact V3 has been emitted by Clang.
- C++/CuTe source has converged through real WebGPU.
- Header packs, Worker bytes, Wasm bytes, or package assets are release-ready.
- Docker or native tests prove the portable browser product.

Three parallel audit/implementation lanes were interrupted when the goal was
paused. Their partial filesystem changes are included in the WIP inventory;
there is no running sub-agent work to wait for.
