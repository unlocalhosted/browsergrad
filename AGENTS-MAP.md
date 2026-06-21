# Agent Map

Use this as a fast navigation layer before diving into files.

## Top-Level Context

| Path | Purpose |
| --- | --- |
| `README.md` | Product overview, install snippets, package summary, test matrix. |
| `ARCHITECTURE.md` | Package responsibilities, data flow, core seams, testing strategy. |
| `DEVELOPMENT.md` | Development notes. |
| `docs/internal/` | Internal vision, progress, status, and compatibility notes. |
| `docs/platform/` | Platform architecture and authoring guides for multi-course guided labs, profiles, rubrics, fixtures, and browser-safe gates. |
| `docs/prd/` | Design records and roadmap PRDs. |
| `packages/` | Workspace packages. |

## Package Map

| Package | Read First | Primary Source | Tests |
| --- | --- | --- | --- |
| `browsergrad-runtime` | `packages/browsergrad-runtime/README.md` | `src/client.ts`, `src/worker/`, `src/lab.ts` | `tests/`, `tests-integration/`, dogfood runtime tests |
| `browsergrad-grad` | `packages/browsergrad-grad/README.md` | `src/python/tensor.py`, `src/python/functional.py`, `src/python/optim.py`, `src/python/nn_chunks/`, `src/python/_torch_compat_*.py` | `tests/`, `tests-integration/` |
| `browsergrad-jit` | `packages/browsergrad-jit/README.md` | `src/python/_ir.py`, `_tensor_proxy.py`, `_realize.py`, `_vjp.py`, `_functional.py`, `_nn.py`, `_optim.py`, `_torch_compat.py` | `tests/`, `tests-integration/` |
| `browsergrad-kernels` | `packages/browsergrad-kernels/README.md` | `src/realizer.ts`, `src/kernels/` | `tests/`, `tests-browser/` |
| `browsergrad-tokenizers` | `packages/browsergrad-tokenizers/README.md` | `src/index.ts` | `tests/` |
| `browsergrad-dogfood` | `packages/browsergrad-dogfood/README.md` | `tests-node/`, `tests/` | cross-package published compatibility |

## Generated Source Rules

- `browsergrad-jit/src/python/*.py` are the editable Python source files. Generated siblings are built by `packages/browsergrad-jit/scripts/build-python-sources.mjs`.
- `browsergrad-grad/src/python/tensor.py`, `functional.py`, `optim.py`, `_torch_compat_*.py`, and `nn_chunks/*` are editable Python source. `nn.generated.ts` is assembled from chunks.
- Do not manually patch `*.generated.ts` or `dist/` unless the task is explicitly about build output inspection. Run codegen/build instead.

## Important Seams

| Seam | Why It Matters |
| --- | --- |
| `createSession` and `session.exec` | Host-to-Pyodide execution boundary. |
| Runtime structured assertions/artifacts | Platform grading and UI feedback channel. |
| `installGrad` / `installJit` | Mounts Python package sources into Pyodide. |
| `install_torch_alias()` | Allows PyTorch-shaped imports, but only for supported surfaces. |
| `bg.register_webgpu_bridge(bridge)` | Connects JIT Python IR to JS/WebGPU kernels. |
| Lab manifest `requires_browsergrad` | Version gate for platform exercises. |

## Curriculum Compatibility Pointers

- Keep root package behavior course-agnostic.
- Read `docs/platform/curriculum-platform-architecture.md` before adding new course or lecture companion work.
- Use `docs/internal/` for assignment-specific compatibility records.
- Use runtime lab manifests and rubrics for platform packaging.
- Use `browsergrad-grad` for stable eager teaching surfaces.
- Use `browsergrad-jit` for lazy IR, fusion, symbolic backward, and GPU-oriented labs.
- Use `browsergrad-tokenizers` for browser-safe tokenizer/BPE oracles and streaming rubric gates.
- Move broadly useful gaps into package source/tests; keep one-off course glue outside package internals.

## Test Selection

Use the narrowest useful command first:

```sh
pnpm --filter @unlocalhosted/browsergrad-runtime test
pnpm --filter @unlocalhosted/browsergrad-grad test
pnpm --filter @unlocalhosted/browsergrad-grad test:integration
pnpm --filter @unlocalhosted/browsergrad-jit test
pnpm --filter @unlocalhosted/browsergrad-jit test:integration
pnpm --filter @unlocalhosted/browsergrad-kernels test
pnpm --filter @unlocalhosted/browsergrad-kernels test:browser
pnpm --filter @unlocalhosted/browsergrad-tokenizers test
pnpm --filter @unlocalhosted/browsergrad-dogfood test:node
```

Before release-level confidence:

```sh
pnpm -r run build
pnpm -r run typecheck
pnpm -r run test
```
