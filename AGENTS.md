# Agent Operating Notes

This repo is `browsergrad`, a browser-native systems and ML execution platform
with Pyodide-based ML education, guided labs, notebooks, and demos as demanding
consumers of its capabilities:

- `@unlocalhosted/browsergrad-runtime`: Pyodide-in-Worker execution, structured assertions/artifacts, lab manifest validation.
- `@unlocalhosted/browsergrad-grad`: eager NumPy-backed autograd with a broad PyTorch-shaped teaching surface.
- `@unlocalhosted/browsergrad-jit`: lazy UOp IR, symbolic backward, fusion, AMP, checkpointing, ONNX, WebGPU realizer bridge.
- `@unlocalhosted/browsergrad-kernels`: WGSL kernels and the production WebGPU realizer bridge.
- `@unlocalhosted/browsergrad-compiler`: current CUDA-lite frontend plus canonical semantic Kernel IR, CPU reference, WGSL/WebGPU runner, and real-world corpus gates. Its target direction is real C++/CuTe source compatibility lowered into shared tensor/layout semantics.
- `@unlocalhosted/browsergrad-primitives`: canonical facade for browser-safe text, data, evaluation, simulation, hosted-training, and RL math primitives.
- `packages/browsergrad-dogfood`: cross-package and published-module compatibility tests.

Start by reading:

1. `AGENTS-MAP.md` for the source map and commands.
2. `ARCHITECTURE.md` for package responsibilities and seams.
3. `RTK.md` for the short task runbook.
4. `docs/platform/package-requirements-lld.md` for the semantic systems requirements and terminology contract.
5. `docs/internal/` or `docs/prd/` for task-specific design context.

## Engineering Direction: Semantics First

BrowserGrad is not a source-pattern collection or a simplified stand-in for
systems work. Build reusable semantic capabilities that can power real
programs, then expose those capabilities in labs and applications.

- Preserve real user source syntax when it belongs to a supported frontend.
  Do not demand BrowserGrad-specific replacement syntax because a compiler
  abstraction is missing.
- Treat `TensorView`, layout/index semantics, tiles, synchronization, dtype,
  and execution graphs as core engineering domains. Do not solve their absence
  with spelling-specific parser handlers, hand-written special kernels, or
  opaque host callbacks.
- A CPU reference, portable WebGPU backend, and native companion backend have
  different contracts. State exactly which tier an API or test proves.
- Never silently substitute semantics: `bf16` is not `f32`; a CPU reference is
  not GPU execution; a row-wise online-softmax kernel is not block-tiled
  FlashAttention; a parser acceptance result is not source compatibility.
- Clear unsupported diagnostics remain necessary, but only after the compiler
  has represented the program honestly and identified the unavailable semantic
  operation or backend capability. Diagnostics are not a substitute for the
  missing abstraction.
- Extend existing canonical seams—semantic IR, CPU reference, WGSL lowering,
  real-device tests, and runtime capability records—rather than creating a
  second source-shaped execution path.

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
course slices, lecture companions, notebooks, framework-shaped exercises, and
systems workloads. The platform vision is a guided-lab layer for many classes
and videos built on serious reusable execution capability, not a single-course
clone or an assignment-specific emulator. Do not shape root agent guidance
around one assignment or one downstream app.

When adding curriculum support:

- Read `docs/platform/curriculum-platform-architecture.md` for the multi-course
  architecture.
- Read `docs/platform/kernel-lab-foundation.md` before adding GPU-programming,
  CUDA-like, Triton-like, or distributed systems lab support.
- Read `docs/platform/package-requirements-lld.md` before changing compiler,
  kernels, JIT realization, dtype/device behavior, or platform capability
  claims.
- Read `docs/platform/research-gated-prd-workflow.md` before creating new PRDs.
- Keep reusable runtime/library capability in packages.
- Put reusable browser-safe helpers behind `browsergrad-primitives` first.
  Split a new package only when implementation weight or release cadence proves
  a real seam.
- Put assignment-specific findings in `docs/internal/` or lab manifests, not in root repo rules.
- Prefer platform adapters/rubrics over hard-coding course assumptions into `grad`, `jit`, `runtime`, or `kernels`.
- Preserve explicit, versioned compatibility surfaces. Unsupported PyTorch,
  CUDA, CuTe, or backend APIs should fail clearly at the true semantic or
  execution boundary; do not silently emulate a different contract.

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
