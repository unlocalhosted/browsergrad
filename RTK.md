# Repo Task Kit

This is the short operational runbook for future agents.

## First Five Minutes

1. Run `git status --short --branch`.
2. Read `AGENTS.md`, `AGENTS-MAP.md`, and the package README for the area you are touching.
3. Search with `rg` before editing.
4. Identify whether the change touches editable source, generated source, or published `dist/`.
5. Choose the smallest test command that exercises the change.

For compiler, kernel, JIT realization, dtype/device, or capability-language
work, also read `docs/platform/package-requirements-lld.md`. It is the
normative semantics-first contract: a parse result, CPU reference, portable
WebGPU execution, and native execution are separate claims and must be named
separately.

## Editing Rules

- Use `apply_patch` for manual file edits.
- Avoid unrelated refactors.
- Do not revert user changes.
- Keep Python package generated files synchronized through codegen.
- Keep public errors descriptive and explicit when functionality is unsupported.
- Do not resolve a missing semantic abstraction by adding source-spelling
  handlers, a renamed CPU fallback, a silent dtype conversion, or an opaque
  callback in a claimed core execution path.

## Pyodide Probe Pattern

When checking installed Python behavior from Node, run from a package directory that can resolve `pyodide`, for example:

```sh
cd packages/browsergrad-jit
node --input-type=module
```

Then import package dist installers and node adapters:

```js
import { loadPyodide } from "pyodide";
import { installJit } from "./dist/index.js";
import { createNodePyodideTarget } from "./dist/node-adapter.js";
```

For `browsergrad-grad` from the JIT directory, use sibling imports:

```js
import { installGrad } from "../browsergrad-grad/dist/index.js";
import { createNodePyodideTarget as createGradTarget } from "../browsergrad-grad/dist/node-adapter.js";
```

## Curriculum Compatibility Work

Before changing package APIs for a course/lab:

- Check whether need is reusable across curricula.
- Put lab-specific facts in `docs/internal/` or lab manifests.
- Add package APIs only when they match BrowserGrad's general PyTorch-shaped/browser-safe contract.
- Add focused package tests for reusable APIs.
- Keep native-only upstream harness assumptions out of root package code.

## Current CUDA-Lite Compiler Workflow

`compileCudaLiteKernel*()` is the workflow for the current CUDA-lite frontend.
It is not authorization to model real C++/CuTe compatibility as a growing list
of parser cases. Before extending layouts, tensors, tiled algorithms, barriers,
or pointer/view behavior, identify the canonical `TensorView`, `IndexMap`,
`Tile`, uniformity, or host-graph semantic change required by the LLD.

Keep the existing architecture gate: parser facts, semantic IR, CPU reference,
and WGSL lowering must agree. A clearer error is useful only after that shared
model can state exactly what is unavailable.

Track active bugbash state in `docs/internal/compiler-bugbash-progress.md`.
It should show the latest green gates, remaining probes, and exact next command
before claiming progress.

For browser-local Clang-Wasm build, ABI, profile, or asset-harness edits, use
the package-owned fast gate as the ordinary edit loop:

```sh
pnpm --filter @unlocalhosted/browsergrad-compiler run verify:browser-clang-wasm:fast
```

It builds once, runs the no-rebuild lock check, and exercises the build-plan,
runtime-ABI, browser-profile, and browser-asset identity chain sequentially.
At the current 2026-07-24 checkpoint it passes 736 tests across 84 files in
about 14 seconds end to end on Node 25.
The set includes package invocation,
Worker entry, production-controller lifecycle, eight-archive admission, strict
tar/Debian normalization, collision-free source extraction, exact header-tree
inventory/materialization, exact CUDA redistribution-index admission, the
complete per-file header distribution review input, exact component-license
and aggregate-notice materialization, notice verification, and package-pinned
two-clean-build reproducibility coverage.
Use `test:browser-clang-wasm-build-plan` for the broader native/sanitizer
pre-commit gate; its same-checkpoint run passed 261 tests with 9 intentional
platform skips across 49 files in about 55 seconds. Do not run package
entrypoints that clean `dist` concurrently in one worktree.

Remote cached validation is diagnostic and ordinarily takes four to five
minutes. Uncached clean and two-build reproducibility modes remain deliberately
separate release-evidence authorities; their provisioning cost is not an edit
loop and must not be placed on routine source iteration.

When exact Clang-Wasm and five materialized pack files already exist, use the
real-browser lane without rebuilding either input:

```sh
pnpm --filter @unlocalhosted/browsergrad-compiler \
  run preflight:browser-cpp-cute-real-compile -- \
  --wasm=/absolute/clang-extractor.wasm \
  --pack-root=/absolute/materialized-pack-root

pnpm --filter @unlocalhosted/browsergrad-compiler \
  run observe:browser-cpp-cute-real-compile -- \
  --wasm=/absolute/clang-extractor.wasm \
  --pack-root=/absolute/materialized-pack-root \
  --evidence-output=/absolute/new-evidence.json
```

`preflight` verifies canonical non-symlink paths, stable file identities, all
six exact hashes, and the package-pinned reproducible Wasm before starting a
browser. `observe` runs the raw-Wasm verifier and compiler Workers in Chromium
and emits either `compiled` or the exact authenticated blocker. Use
`verify:browser-cpp-cute-real-compile` with the same arguments for the strict
Artifact V3 gate; it rejects a `blocked` observation. At the 2026-07-24
checkpoint, preflight takes under one second and an unchanged CuTe layout
compile takes about 21 seconds inside the Worker and about 25 seconds end to
end.

For source-iteration only, an exact locally hashed fast-build Wasm may be
observed by adding `--allow-untrusted-diagnostic-wasm`. Its evidence explicitly
records `untrustedDiagnosticWasm=true` and
`pinnedReproducibleWasmMatched=false`; strict verification refuses it even if
compilation succeeds. Worker protocol v2 failure details report bounded
frontend-work, allocator, and VFS state before cleanup. The current diagnostic
Wasm completes both CUDA semantic passes over the closed five-pack VFS and
returns one accepted Artifact V3 for unchanged C++17/CuTe layout source. That
observation proves real browser Worker execution and source-derived layout
semantics only; producer trust, header-license approval, lowering authority,
backend execution, and release authority remain false.

To turn the eight exact locked archives into the five source-derived VFS packs,
use the one-process command so its opaque authorities remain live:

```sh
pnpm --filter @unlocalhosted/browsergrad-compiler run materialize:browser-header-packs-from-archives -- \
  --cuda-cccl-linux-x86-64=/absolute/cccl.tar.xz \
  --cuda-cudart-linux-x86-64=/absolute/cudart.tar.xz \
  --cuda-libcurand-linux-x86-64=/absolute/libcurand.tar.xz \
  --cuda-nvcc-linux-x86-64=/absolute/nvcc.tar.xz \
  --cutlass=/absolute/cutlass.tar.gz \
  --llvm-project=/absolute/llvm.tar.xz \
  --ubuntu-noble-libc6-dev-amd64-cross=/absolute/libc6-dev.deb \
  --ubuntu-noble-linux-libc-dev-amd64-cross=/absolute/linux-libc-dev.deb \
  --bsdtar=/usr/bin/bsdtar \
  --cuda-redistribution-index=/absolute/private/redistrib_12.6.3.json \
  --output-root=/absolute/private-source-output \
  --pack-output-root=/absolute/private-pack-output
```

The CUDA index and all archives must already be canonical files inside
current-user private directories. Repeated current direct runs complete in
23.1 to 25.4 seconds; the package command additionally performs one package
build. It verifies the locked WebAssembly-only Clang configuration has an
empty generated-header set and excludes the upstream build manifest from the
distributed pack. It also writes the deterministic
`assets/browsergrad-cpp-cute/license-inventory.json` review input: every one of
the 5,788 distributed files is bound to its exact pack identity, component,
package notice, upstream license/copyright evidence, and CUDA index record.
It then materializes all ten build-lock-declared component-license files plus
the deterministic 115,316-byte
`assets/browsergrad-cpp-cute/THIRD_PARTY_NOTICES.txt` aggregate. All eleven
notice outputs are written without clobber, reread independently, and included
in the exact final private tree. These are distribution review artifacts, not
legal approval. The exact Darwin arm64 builder is
package-pinned to Node 25.9.0, Node's Zstandard 1.5.7 closure, and
`/usr/bin/bsdtar` 3.5.3. Check
that identity before materialization with:

```sh
pnpm --filter @unlocalhosted/browsergrad-compiler run verify:browser-header-normalization-environment
```

An unreviewed host or runtime fails closed instead of inheriting authority from
a version string. Outputs remain non-release observations until external
file-level licensing, distribution approval, signed provenance, approved asset
binding, and browser execution are closed.

To materialize the same exact input closure twice under distinct roots and
independently rehash all 17 outputs in both trees, use:

```sh
pnpm --filter @unlocalhosted/browsergrad-compiler run verify:browser-header-distribution-reproducibility -- \
  --cuda-cccl-linux-x86-64=/absolute/cccl.tar.xz \
  --cuda-cudart-linux-x86-64=/absolute/cudart.tar.xz \
  --cuda-libcurand-linux-x86-64=/absolute/libcurand.tar.xz \
  --cuda-nvcc-linux-x86-64=/absolute/nvcc.tar.xz \
  --cutlass=/absolute/cutlass.tar.gz \
  --llvm-project=/absolute/llvm.tar.xz \
  --ubuntu-noble-libc6-dev-amd64-cross=/absolute/libc6-dev.deb \
  --ubuntu-noble-linux-libc-dev-amd64-cross=/absolute/linux-libc-dev.deb \
  --bsdtar=/usr/bin/bsdtar \
  --cuda-redistribution-index=/absolute/private/redistrib_12.6.3.json \
  --first-source-output-root=/absolute/private/source-a \
  --first-pack-output-root=/absolute/private/packs-a \
  --second-source-output-root=/absolute/private/source-b \
  --second-pack-output-root=/absolute/private/packs-b
```

All four output roots must be absent, canonical, distinct, and non-overlapping
under private parents. The current direct two-run proof takes 44 to 47 seconds.
It proves the header-distribution subset only; it does not prove the complete
release asset set, licensing, provenance, or release readiness.

Use the smallest WebGPU loop that covers the suspected bug class:

```sh
pnpm --filter @unlocalhosted/browsergrad-compiler run verify:changed
pnpm --filter @unlocalhosted/browsergrad-compiler run verify:changed:plan -- --scope atomic
pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:last-failures
pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:case -- --case atomic:helper-rmw
pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:hot-case:gate -- --cases texture-surface:roundtrip
pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:compile
pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:smoke
pnpm --filter @unlocalhosted/browsergrad-compiler run e2e:webgpu:fast
```

- `verify:changed`: changed files to owning tests; use as default edit loop.
- `verify:changed:plan`: print scoped plan without running; use `--scope atomic|storage|pointer|vector|control|texture|runtime|real-world`.
- `e2e:webgpu:last-failures`: rerun cases persisted in `.tmp/cuda-lite-last-failures.json`.
- Add `--timing-json .tmp/test-scope-timing.json` to `cuda-lite-test-scope.mjs --run` when tuning slow loops.
- `verify:changed:compile`: same scoped tests plus cached fast corpus shader-module compile only.
- `verify:changed:fast`: same scoped tests plus cached fast corpus for compiler source edits.
- `e2e:webgpu:case`: focused repro, no build, fail fast.
- `e2e:webgpu:warm-case`: repeats focused cases in one browser/device session to expose warm pipeline behavior.
- `e2e:webgpu:hot-case:gate`: repeats focused cases and fails if warm speedup drops below the configured floor.
- `e2e:webgpu:compile`: fast auto-corpus WGSL shader-module validation without dispatch/readback.
- `e2e:webgpu:smoke`: representative hand fixtures for storage, vectors, atomics, barriers, and texture/surface lowering.
- `e2e:webgpu:fast`: cached auto-corpus fast profile. Use before claiming compiler progress.
- `verify:real-world-cuda` uses fast auto-corpus smoke by default; pass `--auto-corpus-smoke-profile full` only for exhaustive smoke.
- Full corpus gates stay for commit/release confidence, not every edit.
- Filtered WebGPU runs should not load unrelated corpus fixture sources. If focused cases slow down, check source-loading first.

## Release And Publish

Never claim npm publication from local source state. Verify the packed artifact
and the registry.

Before tagging or dispatching publish:

```sh
pnpm -r build
pnpm test:release-packages
node scripts/publish-missing-npm.mjs --dry-run
```

Use `pnpm publish`, not `npm publish`, for workspace packages. `pnpm publish`
rewrites `workspace:*` dependencies to concrete package versions in the tarball.
After CI publishes, check npm directly:

```sh
npm view @unlocalhosted/browsergrad-kernels version exports --json
npm view @unlocalhosted/browsergrad-compiler version dependencies --json
```
