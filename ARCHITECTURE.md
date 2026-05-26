# Architecture deepening tracker

Tracking the five-candidate architectural refactor identified post-v0.5.0.
Each candidate was researched (dossier) and grilled (verdict + plan) before
implementation. This doc records the load-bearing decisions and links each
to its commit, so the rationale survives the codebase.

For broader PyTorch-coverage progress, see [PROGRESS.md](PROGRESS.md).

## Summary

| # | Refactor | Verdict | Status | Commit |
|---|---|---|---|---|
| 5 | Single-source the version | Add `resolveJsonModule`; import `pkg` in 3 files; don't flip `verbatimModuleSyntax` | ✅ done | `27cf321` |
| 4 | Python source as `.py` files | Pre-tsc codegen script; commit generated files; base64-IIFE emission | ✅ done | `a8b926a` |
| 3 | NodePyodideTarget adapter | Factory at `./node-adapter` subpath; `pyodide` as optionalPeerDep | ✅ done | (this commit) |
| 2 | Split torch_compat into real/limited/impossible | Self-installing modules; runtime `is`-identity assertion pins the latent coupling | ⏳ pending | — |
| 1 | Split nn.ts into per-family TS chunks | Option A (TS-split, single `nn.py`); `NN_CHUNK_ORDER` constant enforces order | ⏳ pending | — |

## Methodology

Each refactor follows the same shape:
1. **Dossier** — research-only Explore pass producing files-touched + alternative shapes.
2. **Grilling** — adversarial design walk producing a verdict + concrete plan.
3. **Implementation** — small TDD-style commits, each landing all-green.
4. **Decision log** — load-bearing trade-offs captured below, not in code comments.

ADRs are not being introduced as a separate format. This doc IS the decision log;
if a decision survives long enough to be re-litigated, it gets hoisted into a
proper ADR. Don't create per-decision files prematurely.

Implementation order is intentional: #5 → #4 → #3 → #2 → #1. Rationale:
- #5 is small and validates the `resolveJsonModule` path that other refactors might need.
- #4 must precede #1 — splitting real `.py` files into a directory is mechanically simpler than splitting a 1628-line TS template literal.
- #3 is independent and unlocks the first real-Pyodide coverage of `installViaFs`.
- #2 benefits from #4 landing first (the source becomes a normal `.py` file).
- #1 has the biggest blast radius; everything else clears its path first.

---

## #5 — Single-source the version

### Verdict

Import `pkg` from `package.json` directly at the three duplicated touch-points:
`src/python/index.ts`, `src/install.ts`, `tests/surface.test.ts`. Do NOT introduce
a `src/version.ts` intermediary — three import sites doing the same 2-line import
isn't a Depth problem, just two lines of repetition. Do NOT flip
`verbatimModuleSyntax` to `false` — that flag enforces `import type` discipline
across the whole codebase, which is a real Locality signal worth keeping. Just
add `resolveJsonModule: true` alongside the existing `esModuleInterop: true`.

### Plan

1. `packages/browsergrad-grad/tsconfig.json` — add `"resolveJsonModule": true`.
2. `packages/browsergrad-grad/tsconfig.build.json` — add `"resolveJsonModule": true`.
3. `packages/browsergrad-grad/src/python/index.ts` — import `pkg`; replace literal `"0.5.0"` in INIT_PY template with `${pkg.version}`.
4. `packages/browsergrad-grad/src/install.ts` — import `pkg`; replace literal in smoke-check assert; upgrade error message to `f"expected ${pkg.version}, got {_bg_check.__version__}"`.
5. `packages/browsergrad-grad/tests/surface.test.ts` — import `pkg`; replace literal with `pkg.version`; strip version from the `it()` description.

### Validation gate

Hand-bump `package.json` to a sentinel version (e.g. `0.6.0-dry`), run typecheck + full test suite, confirm zero other files need editing. Revert. That single experiment proves the single-source property is real, not theoretical.

### Decisions

- **`resolveJsonModule: true` added only to dev tsconfig.json**, not also to `tsconfig.build.json` as the grilling verdict suggested. `tsconfig.build.json` extends the dev config so the flag is inherited — making it explicit in both would be redundant, not defensive.
- **No `assert { type: "json" }` import attribute** on the JSON imports. The grilling verdict used the older proposal syntax; current TC39 stage-3 is `with { type: "json" }`. Plain `import pkg from "../../package.json"` works once `resolveJsonModule: true` is set, and avoids picking a syntax form before TypeScript settles on one.
- **`verbatimModuleSyntax: true` preserved** in dev tsconfig. The flag does not block JSON value imports (it blocks type-only re-exports without `import type`/`export type`), so adding `resolveJsonModule` is sufficient.
- **Validation gate ran clean** on `0.6.0-dry`: typecheck + 25 unit tests + 27 integration files / 232 integration tests all green with zero other file edits. Property is real.

---

## #4 — Python source as `.py` files (pending)

### Verdict

Option A (pre-tsc codegen script emitting `*.generated.ts`), with generated files
**committed** (not gitignored). Option B (Vite plugin) would require introducing
a Vite build pipeline that doesn't exist today; the package builds with `tsc` only.

### Plan

1. Create `src/python/{tensor,functional,nn,optim,torch_compat,utils_data}.py` — copy Python content verbatim, auto-strip the `\`` escapes.
2. Create `scripts/build-python-sources.ts` — reads each `.py`, encodes via `TextEncoder + btoa` (same idiom as `pythonStringLiteral` in `install.ts:99-109`), emits `*.generated.ts` with a self-contained IIFE that decodes the base64. Header: `// @generated — do not edit by hand`.
3. Thin each existing `src/python/*.ts` to a single re-export: `export { TENSOR_PY } from "./tensor.generated.js"`. Preserves the import graph in `src/python/index.ts` without changing it.
4. Update `package.json` scripts: `"codegen": "tsx scripts/build-python-sources.ts"`, prepend it to `"build"`.
5. Diff-test (one-off, deleted after): assert byte-for-byte that each generated string equals the current template-literal value before deleting the original content.

### ruff + mypy follow-ons

Separate commits, in this order: migration → ruff → mypy. Bundling them poisons the migration diff with noise (expect 30–80 mypy errors in `tensor.py` alone on first `--strict` run). Use `uvx` to avoid requiring a local Python env.

### Decisions

- **`.mjs` over `.ts` for the codegen script.** Avoids the chicken-and-egg of needing TS tooling before the build step that produces the TS sources. The script is plain Node ESM JS, runs via `node scripts/build-python-sources.mjs`. Zero new devDeps.
- **One-time extraction script bug**: first version used `tsSource.indexOf("\`")` which matched the FIRST backtick in the JSDoc preamble (e.g., `tensor.ts` has `\`tsc\`` in its doc comment). Production tests all failed with `SyntaxError: invalid character '—'` because the extracted "Python" started inside a JSDoc paragraph. Fixed by anchoring on `export const FOO_PY = \`` and using the right closing `\`;` — extraction script verified before being deleted.
- **The one-time extraction script (`extract-python-once.mjs`) was deleted after use.** Source of truth is now `.py`; no need to maintain a script that reads the old TS form.
- **Re-export shim preserves the import graph in `src/python/index.ts`**: each `tensor.ts` is now a single `export { TENSOR_PY } from "./tensor.generated.js";`. No edit to `src/python/index.ts` needed.
- **Codegen wired into `build` script** as a prefix: `"build": "pnpm codegen && tsc -p tsconfig.build.json"`. Generated files are committed so `pnpm typecheck` and `pnpm test` work without running codegen first on a fresh clone.

---

## #3 — NodePyodideTarget adapter (pending)

### Verdict

Ship a `createNodePyodideTarget(pyodide)` factory at `./node-adapter` subpath.
Adds no required dependency — `pyodide` becomes `optionalPeerDependencies` with
the version range already in devDeps. Status quo is indefensible: `installViaFs`
is a published code path that has never run against real Pyodide. Option (a)
(drop GradTarget, take `Session`) rejected because it forces a peer dep on
`@unlocalhosted/browsergrad-runtime`, breaking the README's "no peer dep" claim
and the "install into raw Pyodide" use case (Deno, Jupyter, server Node).

### Plan

1. `src/node-adapter.ts` — exports `createNodePyodideTarget(pyodide: PyodideInterface): GradTarget`. Structural `PyodideInterface` declared locally (no hard pyodide import). Factory does NOT expose `.run<T>()` — that's test glue.
2. `package.json` — add `"./node-adapter"` to `exports`; add `optionalPeerDependencies: { "pyodide": "^0.26.4" }`.
3. `tests-integration/pyodide-host.ts` — replace `makeTarget(py)` in `getGradTarget` with `createNodePyodideTarget(py)`. Keep the local `.run<T>()` wrapper for tests.
4. `tests-integration/install-via-fs.test.ts` — new test asserting `py.FS.analyzePath('/lib/browsergrad_grad_src/browsergrad_grad/__init__.py').exists === true`. That's the proof installViaFs ran (not exec).
5. README — one section documenting `./node-adapter` and the optional peer dep.

`install.ts` itself: zero changes.

### Decisions

- **`PyodideInterface` declared locally**, not imported from the `pyodide` package. Mirrors how `GradTarget` itself is duck-typed — the Adapter accepts anything quacking like Pyodide. Keeps the Module free of any type-level dependency on the `pyodide` package, even though pyodide is an `optionalPeerDependency` at runtime.
- **`PyodideInterface` exported** alongside `createNodePyodideTarget` so consumers can write their own factory or extend it. Tiny export with high leverage for downstream users.
- **`makeTarget` in tests now spreads the Adapter** rather than re-implementing `exec` and `fs`. The `.run<T>()` test-glue helper sits on top via `{...adapter, run}`. Makes tests exercise the same code path consumers exercise.
- **Test name `install_via_fs.test.ts`** (snake_case to match the file's naming convention in this test suite) rather than `install-via-fs.test.ts` — every other integration test file uses snake_case.
- **Two assertions, not one**, in the new test: `__init__.py` exists (proves the basic write path) AND `utils/data.py` exists (proves `mkdirTree` ran for the nested subpackage). Without the second, a regression in the `mkdirTree` step would slip through.

---

## #2 — Split torch_compat into real/limited/impossible (pending)

### Verdict

Option B (self-installing modules with `def _install_pile_a(torch_mod, _bg, _types): ...`),
not Option A (string-concat). String-concat is indentation-fragile; any auto-formatter
silently shifts pile boundaries out of function scope. Pin the latent
`torch_nn.Module is _bg.nn.Module` coupling with a runtime assertion inside
`install_torch_alias()` — not a comment, not a new test. The assertion message
names the next file to look at.

### Plan

1. Create `src/python/torch_compat_real.ts` — exports `TORCH_REAL_PY: string` defining `_install_pile_a(torch_mod, _bg, _types)`.
2. Create `src/python/torch_compat_limited.ts` — exports `TORCH_LIMITED_PY: string` defining `_install_pile_b(torch_mod, _bg, _types, _ctxlib, _np)`.
3. Create `src/python/torch_compat_impossible.ts` — exports `TORCH_IMPOSSIBLE_PY: string` defining `_impossible(name, reason)` factory + `_install_pile_c(torch_mod, _types)`.
4. Thin `src/python/torch_compat.ts` — orchestrator that concatenates the three strings and defines `install_torch_alias()` calling each `_install_pile_*` in order, with the identity assertion between Pile A and Pile B.
5. No test file changes — existing tests (`torch_alias.test.ts`, `torch_compat_completeness.test.ts`, `piles_b_c.test.ts`) reach only through the Python torch namespace.

The `Module.to` monkey-patch fix (moving it into `nn.ts` as a proper method) is **deferred** — the assertion makes the cross-pile coupling loud, which is enough for now.

### Decisions

_(filled in during implementation)_

---

## #1 — Split nn.ts into per-family TS chunks (pending)

### Verdict

Option A (TS-split into 12 sibling `.ts` chunks producing one flat `nn.py`),
not Option B (real Python subpackage). The library's pedagogical contract is
`nn.Foo`, not `from browsergrad_grad.nn.linear import Linear`. Exposing sub-modules
would forever-couple the API to that import surface for no domain reason.

Mechanical enforcement: encode concat order as `NN_CHUNK_ORDER: readonly string[]`
in `src/python/nn/index.ts`. Missing chunks become silently absent, which the
existing content-assertion tests catch. Not a comment, not a convention.

### Prerequisite

Remove `_normalize_with_affine` (dead code, `nn.ts:1206`) as a separate prior
commit. Mixing dead-code removal into a 12-file refactor poisons the diff.

### Plan

1. (Prerequisite commit) Remove `_normalize_with_affine` from `nn.ts`.
2. Create `src/python/nn/{module,linear,conv,norm,pool,dropout,activation,embedding,recurrent,attention,loss,init}.ts` (12 chunk files).
3. Create `src/python/nn/index.ts` — assembler exporting `NN_PY` via `NN_CHUNK_ORDER.map(...).join("\n")` plus a build-time `length > 10000` floor assertion.
4. Delete `src/python/nn.ts`.
5. Update `src/python/index.ts` import — explicit `"./nn/index.js"` if tsconfig uses `node16`/`nodenext`.
6. Add 3 ordering-invariant tests to `tests/surface.test.ts` (Module before Linear, Linear before MultiHeadAttention, `sys.modules["browsergrad_grad.nn.init"]` registration after MultiHeadAttention).

### Decisions

_(filled in during implementation)_

---

## Methodology meta-decisions

These are choices made about HOW we work, not about specific refactors:

- **Dossier-then-grill-then-implement** is the standard pattern for any refactor expected to take >2 hours. For small one-file changes, skip directly to implementation.
- **Subagents fan out research and grilling**; implementation stays in the main thread because each step needs to be reviewable as a single coherent change.
- **One commit per refactor**, all-green. No mid-refactor commits.
- **This doc is the single source of architectural rationale.** PROGRESS.md is for PyTorch-coverage; CHANGELOG.md is for user-facing release notes; README is for what the library does. ARCHITECTURE.md is for why it's shaped this way.
