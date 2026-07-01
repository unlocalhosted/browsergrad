# Repo Task Kit

This is the short operational runbook for future agents.

## First Five Minutes

1. Run `git status --short --branch`.
2. Read `AGENTS.md`, `AGENTS-MAP.md`, and the package README for the area you are touching.
3. Search with `rg` before editing.
4. Identify whether the change touches editable source, generated source, or published `dist/`.
5. Choose the smallest test command that exercises the change.

## Editing Rules

- Use `apply_patch` for manual file edits.
- Avoid unrelated refactors.
- Do not revert user changes.
- Keep Python package generated files synchronized through codegen.
- Keep public errors descriptive and explicit when functionality is unsupported.

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

## CUDA-Lite Compiler Iteration

Track active bugbash state in `docs/internal/compiler-bugbash-progress.md`.
It should show the latest green gates, remaining probes, and exact next command
before claiming progress.

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
