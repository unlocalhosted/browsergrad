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
| `browsergrad-primitives` | Canonical facade | Public home for small generic references, comparators, fixtures, simulators, parsers, and data-cleaning helpers. |
| `browsergrad-tokenizers` | Compatibility / implementation shard | Byte-BPE implementation remains useful, but new public guidance should import through `browsergrad-primitives/text` unless bundle policy requires direct import. |
| `browsergrad-data` | Compatibility / implementation shard | Data-cleaning implementation remains useful, but generic consumers should learn `browsergrad-primitives/data` first. |
| `browsergrad-snapshots` | Compatibility / implementation shard | Snapshot comparison is small enough for the primitive facade; keep direct package only for compatibility or bundle splits. |
| `browsergrad-scaling` | Compatibility / implementation shard | Hosted-training fixtures and scaling-law math should be taught through `browsergrad-primitives/scaling` until release cadence proves a real seam. |
| `browsergrad-simulators` | Compatibility / implementation shard | Deterministic mesh/task/SIMD simulators are generic primitives; split only if worker-mesh/native-runner code makes the implementation heavy. |
| `browsergrad-alignment` | Compatibility / implementation shard | RL/alignment math references belong behind `browsergrad-primitives/rl` for now. |
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

## Migration Target

Teach this package graph:

```text
runtime      = host execution and lab/profile preflight
grad         = eager Python tensor/autograd library
jit          = lazy Python IR/compiler library
kernels      = WebGPU/WGSL and CUDA-shaped browser GPU core
primitives   = small generic browser-safe references
dogfood      = private release verification
```

Treat the other small packages as compatibility import paths and implementation
shards unless a future PRD proves a real seam.
