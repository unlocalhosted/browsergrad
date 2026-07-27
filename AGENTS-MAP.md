# Agent Map

Use this as a fast navigation layer before diving into files.

## Top-Level Context

| Path | Purpose |
| --- | --- |
| `README.md` | Product overview, install snippets, package summary, test matrix. |
| `ARCHITECTURE.md` | Package responsibilities, data flow, core seams, testing strategy. |
| `DEVELOPMENT.md` | Development notes. |
| `docs/platform/package-requirements-lld.md` | Normative multi-level semantic architecture, low-level contracts, migration gates, current-state corrections, and acceptance evidence. Read before compiler/kernel/JIT/dtype work. |
| `docs/platform/consuming-browsergrad.md` | Production npm consumption guide and import matrix. |
| `docs/platform/agent-consumption-guide.md` | Agent-facing package selection and import rules. |
| `docs/platform/release-readiness.md` | Publish workflow, packed tarball checks, and npm verification. |
| `docs/platform/resource-metrics.md` | Correctness contract for runtime/WebGPU resource metrics and budgets. |
| `docs/internal/` | Internal vision, progress, status, and compatibility notes. |
| `docs/platform/` | Platform architecture and authoring guides for multi-course guided labs, profiles, rubrics, fixtures, and browser-safe gates. |
| `docs/prd/` | Design records and roadmap PRDs. |
| `packages/` | Workspace packages. |

## Package Map

| Package | Read First | Primary Source | Tests |
| --- | --- | --- | --- |
| `browsergrad-runtime` | `packages/browsergrad-runtime/README.md` | `src/client.ts`, `src/worker/`, `src/lab.ts` | `tests/`, `tests-integration/`, dogfood runtime tests |
| `browsergrad-grad` | `packages/browsergrad-grad/README.md` | `src/python/tensor.py`, `src/python/functional.py`, `src/python/optim.py`, `src/python/nn_chunks/`, `src/python/_device.py`, `src/python/_torch_compat_*.py`, `src/kernel-device.ts` | `tests/`, `tests-integration/` |
| `browsergrad-jit` | `packages/browsergrad-jit/README.md` | `src/python/_ir.py`, `_tensor_proxy.py`, `_realize.py`, `_vjp.py`, `_functional.py`, `_nn.py`, `_optim.py`, `_torch_compat.py` | `tests/`, `tests-integration/` |
| `browsergrad-kernels` | `packages/browsergrad-kernels/README.md` | `src/realizer.ts`, `src/kernels/` | `tests/`, `tests-browser/` |
| `browsergrad-compiler` | `docs/platform/package-requirements-lld.md`, `packages/browsergrad-compiler/README.md`, `docs/platform/cuda-lite-compiler-architecture.md` | `src/parser.ts`, `src/analyzer.ts`, `src/semantic_ir_types.ts`, `src/semantic_ir.ts`, `src/semantic_reference.ts`, `src/semantic_wgsl.ts`, `src/runner.ts`, `scripts/cuda-lite-source-normalizer.mjs` | `tests/`, `tests-browser/`, corpus/e2e scripts |
| `browsergrad-primitives` | `packages/browsergrad-primitives/README.md` | `src/index.ts`, `src/text.ts`, `src/data.ts`, `src/evaluation.ts`, `src/scaling.ts`, `src/simulation.ts`, `src/rl.ts` | `tests/` |
| `browsergrad-dogfood` | `packages/browsergrad-dogfood/README.md` | `tests-node/`, `tests/` | cross-package published compatibility |

## Generated Source Rules

- `browsergrad-jit/src/python/*.py` are the editable Python source files. Generated siblings are built by `packages/browsergrad-jit/scripts/build-python-sources.mjs`.
- `browsergrad-grad/src/python/tensor.py`, `functional.py`, `optim.py`, `_device.py`, `_torch_compat_*.py`, and `nn_chunks/*` are editable Python source. `nn.generated.ts` is assembled from chunks. `src/kernel-device.ts` is the TS adapter from `@unlocalhosted/browsergrad-kernels` to Python's explicit eager `device=` bridge.
- Do not manually patch `*.generated.ts` or `dist/` unless the task is explicitly about build output inspection. Run codegen/build instead.

## Important Seams

| Seam | Why It Matters |
| --- | --- |
| `createSession` and `session.exec` | Host-to-Pyodide execution boundary. |
| Runtime structured assertions/artifacts | Platform grading and UI feedback channel. |
| `installGrad` / `installJit` | Mounts Python package sources into Pyodide. |
| `install_torch_alias()` | Allows PyTorch-shaped imports, but only for supported surfaces. |
| `bg.register_webgpu_bridge(bridge)` | Connects JIT Python IR to JS/WebGPU kernels. |
| `compileCudaLiteKernel*()` | Browser-native CUDA-lite frontend to Kernel IR/WGSL/reference/WebGPU. |
| Browser-local C++/CuTe controller | Runs the pinned Clang-Wasm extractor in a verified Worker and lowers accepted Artifact V3 facts through semantic-core. |
| Lab manifest `requires_browsergrad` | Version gate for platform exercises. |

CUDA-lite remains the shipping default frontend. A closed pre-release
browser-local C++/CuTe profile now runs a real Clang-Wasm extractor and lowers
accepted view-copy facts through the same semantic-core CPU/WebGPU seams; it is
not a generic full-C++ claim. Before adding syntax support, read the semantic
requirements: extend the versioned frontend through layouts,
`Tensor<Engine, Layout>`, views, index maps, tiles, kernel meaning, schedules,
and host graphs. Do not add source-spelling handlers as a substitute for those
abstractions.

## Curriculum Compatibility Pointers

- Keep root package behavior course-agnostic.
- Treat education and guided labs as consumers of exact platform semantics, not
  as permission to replace them with assignment-specific emulation.
- Read `docs/platform/curriculum-platform-architecture.md` before adding new course or lecture companion work.
- Read `docs/platform/kernel-lab-foundation.md` before adding browser-native GPU/kernel lab work.
- Read `docs/platform/package-requirements-lld.md` before changing compiler,
  kernels, JIT realization, dtype/device behavior, or capability terminology.
- Read `docs/platform/package-consolidation-audit.md` before proposing or adding
  a package.
- Read `docs/platform/research-gated-prd-workflow.md` before creating or publishing PRDs.
- Use `docs/internal/` for assignment-specific compatibility records.
- Use runtime lab manifests and rubrics for platform packaging.
- Use `browsergrad-grad` for stable eager teaching surfaces.
- Use `browsergrad-jit` for lazy IR, fusion, symbolic backward, and GPU-oriented labs.
- Use `browsergrad-primitives` first for browser-safe text, data, evaluation,
  hosted-training, simulation, and RL helpers.
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
pnpm --filter @unlocalhosted/browsergrad-kernels test:browser:semantic-host-graph:required
pnpm --filter @unlocalhosted/browsergrad-compiler run verify:browser-clang-wasm:fast
pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler
pnpm --filter @unlocalhosted/browsergrad-compiler verify:real-world-cuda -- --skip-fetch --require-webgpu
pnpm --filter @unlocalhosted/browsergrad-primitives test
pnpm --filter @unlocalhosted/browsergrad-dogfood test:node
```

Before release-level confidence:

```sh
pnpm -r run build
pnpm test:release-packages
pnpm -r run typecheck
pnpm -r run test
```
