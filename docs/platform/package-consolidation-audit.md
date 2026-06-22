# Package Consolidation Audit

BrowserGrad's package shape should serve the browser ML library first. Labs,
assignments, and course ports are benchmark probes over that library.

## Package Classes

| Package | Class | Rationale |
| --- | --- | --- |
| `browsergrad-runtime` | Keep as a real seam | Owns host/worker/Pyodide execution, filesystem mounts, cancellation, assertions, artifacts, and assignment/profile preflight. Deleting it would push worker and platform protocol code into every caller. |
| `browsergrad-grad` | Keep as a real seam | Eager PyTorch-shaped Python library. Deleting it would mix tensor/autograd teaching source into runtime or JIT. |
| `browsergrad-jit` | Keep as a real seam | Lazy UOp IR, realization, symbolic backward, codegen, and backend integration. Separate from eager `grad` by architecture and release cadence. |
| `browsergrad-kernels` | Keep as a real seam | WGSL/WebGPU, CPU references, CUDA-shaped simulator/lowering, and browser GPU bridge. Python-agnostic GPU package. |
| `browsergrad-primitives` | Keep as a real seam | Owns small generic references, comparators, fixtures, simulators, parsers, data-cleaning helpers, tokenizer references, hosted-training fixtures, and RL math. The previous small shard packages were shallow: deleting them concentrated complexity instead of spreading it. |
| `browsergrad-dogfood` | Private verification package | Not product surface. Keeps cross-package and published-module compatibility checks out of shipped libraries. |

## Rules For New Work

- New small reusable helpers enter `browsergrad-primitives` first.
- New heavy execution substrates enter an existing real seam unless they force a
  new durable interface.
- Course-specific facts stay in `docs/internal/`, profile JSON, fixtures,
  rubrics, and platform glue.
- Profile bridge objects may use snake_case or JSON strings for Pyodide, but
  package exports should use generic reference/comparator/fixture/simulator
  vocabulary.
- New package proposals must pass the deletion test in
  `docs/platform/primitive-package-architecture.md`.

## Current Package Graph

Teach this package graph:

```text
runtime      = host execution and lab/profile preflight
grad         = eager Python tensor/autograd library
jit          = lazy Python IR/compiler library
kernels      = WebGPU/WGSL and CUDA-shaped browser GPU core
primitives   = small generic browser-safe references
dogfood      = private release verification
```

Any future split from `browsergrad-primitives` must show two real adapters or a
hard packaging constraint. Course demand alone is not enough.
