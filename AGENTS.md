# Agent Operating Notes

This repo is `browsergrad`, a pnpm monorepo for Pyodide-based ML education:

- `@unlocalhosted/browsergrad-runtime`: Pyodide-in-Worker execution, structured assertions/artifacts, lab manifest validation.
- `@unlocalhosted/browsergrad-grad`: eager NumPy-backed autograd with a broad PyTorch-shaped teaching surface.
- `@unlocalhosted/browsergrad-jit`: lazy UOp IR, symbolic backward, fusion, AMP, checkpointing, ONNX, WebGPU realizer bridge.
- `@unlocalhosted/browsergrad-kernels`: WGSL kernels and the production WebGPU realizer bridge.
- `@unlocalhosted/browsergrad-compiler`: browser-native CUDA-lite parser/analyzer, Kernel IR, CPU reference, WGSL/WebGPU runner, and real-world corpus gates.
- `@unlocalhosted/browsergrad-primitives`: canonical facade for small browser-safe primitives: text, data, evaluation, simulation, hosted-training, and RL math.
- `packages/browsergrad-dogfood`: cross-package and published-module compatibility tests.

Start by reading:

1. `AGENTS-MAP.md` for the source map and commands.
2. `ARCHITECTURE.md` for package responsibilities and seams.
3. `RTK.md` for the short task runbook.
4. `docs/internal/` or `docs/prd/` for task-specific design context.

## Repository Rules

- Prefer `rg` and `rg --files` for search.
- Keep generated files in sync. For Python package sources, edit the `.py` source or `nn_chunks/*` source, then run the package `codegen` script instead of hand-editing generated `.ts`.
- Do not change package boundaries casually. Runtime must stay tensor-agnostic. Kernels must stay Python-agnostic. `grad` and `jit` should compose through public installers and the runtime.
- Preserve the clear-failure contract. Unsupported PyTorch aliases should fail loudly with specific errors, not silently return wrong values.
- Use focused tests. Broad workspace tests are useful before release, but package-level integration tests are usually the fastest confidence loop.
- Do not use `npx convex deploy` from this repo unless explicitly asked for production deployment.

## Common Commands

```sh
pnpm install
pnpm -r run build
pnpm -r run typecheck
pnpm -r run test
pnpm --filter @unlocalhosted/browsergrad-grad test:integration
pnpm --filter @unlocalhosted/browsergrad-jit test:integration
pnpm --filter @unlocalhosted/browsergrad-kernels test:browser
pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler
pnpm --filter @unlocalhosted/browsergrad-compiler verify:real-world-cuda -- --skip-fetch --require-webgpu
pnpm --filter @unlocalhosted/browsergrad-primitives test
pnpm --filter @unlocalhosted/browsergrad-dogfood test:node
```

Run codegen after changing Python sources:

```sh
pnpm --filter @unlocalhosted/browsergrad-grad run codegen
pnpm --filter @unlocalhosted/browsergrad-jit run codegen
```

## Curriculum And Platform Direction

Keep BrowserGrad general-purpose. It should support many in-browser ML labs,
course slices, lecture companions, notebooks, and framework-shaped exercises.
The platform vision is a guided-lab layer for many classes and videos, not a
single-course clone. Do not shape root agent guidance around one assignment or
one downstream app.

When adding curriculum support:

- Read `docs/platform/curriculum-platform-architecture.md` for the multi-course
  architecture.
- Read `docs/platform/kernel-lab-foundation.md` before adding GPU-programming,
  CUDA-like, Triton-like, or distributed systems lab support.
- Read `docs/platform/research-gated-prd-workflow.md` before creating new PRDs.
- Keep reusable runtime/library capability in packages.
- Put small reusable helpers behind `browsergrad-primitives` first. Split a new
  package only when implementation weight or release cadence proves a real seam.
- Put assignment-specific findings in `docs/internal/` or lab manifests, not in root repo rules.
- Prefer platform adapters/rubrics over hard-coding course assumptions into `grad`, `jit`, `runtime`, or `kernels`.
- Preserve small, explicit compatibility surfaces. Unsupported PyTorch APIs should fail clearly.

## Convex Development Rules

If future work introduces or edits `convex/` directories, follow these rules.

### Security

- All public `query`, `mutation`, and `action` functions must define `args` and `returns` validators.
- Every public function accessing user data must verify auth via `ctx.auth.getUserIdentity()`.
- Always verify resource ownership before reads/writes. Never trust client-provided user IDs.
- Prefer `convex-helpers` `customQuery` and `customMutation` wrappers such as `authedQuery`, `authedMutation`, and `adminQuery`.

### Performance

- Use `.withIndex()` instead of `.filter()` for database queries.
- Index all foreign keys.
- Do not call `Date.now()` in queries. Pass time as an argument or use status fields.
- Use cursor-based pagination for large datasets. Never `.collect()` unbounded queries.

### Async And Errors

- Await every `ctx.db.patch`, `ctx.db.insert`, `ctx.scheduler.runAfter`, and similar promise.
- Throw descriptive errors such as `"Not authenticated"`, `"Task not found"`, or `"Unauthorized"`.
- Return `null` for missing data in queries.

### Schema Design

- Use flat documents with ID references, not deeply nested documents.
- Use arrays only when bounded below Convex limits.
- Model enums with `v.union(v.literal("a"), v.literal("b"))`.
- Store timestamps as `v.number()` milliseconds since epoch.
- Prefer single-field indexes for simple lookups and compound indexes for filtered queries.

### Function Organization

- Keep query, mutation, and action wrappers thin. Put business logic in plain TypeScript functions.
- Schedule only `internal.*` functions, never `api.*`.
- Files with `"use node"` may contain only actions. Keep Node actions separate from queries and mutations.

### Code Quality

- Keep TypeScript strict.
- Avoid `any`.
- Use `@convex-dev/eslint-plugin` and run ESLint on Convex code.
- Use Convex components for modular reusable features when appropriate.

### Development

- Use `npx convex dev` for development.
- `npx convex deploy` is production-only.
- Run `npx convex codegen` after schema changes.
