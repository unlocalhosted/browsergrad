# @unlocalhosted/browsergrad-jit

[![npm version](https://img.shields.io/npm/v/@unlocalhosted/browsergrad-jit.svg)](https://www.npmjs.com/package/@unlocalhosted/browsergrad-jit)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A PyTorch-shaped Python tensor library that runs in the browser via Pyodide. Same API surface as [`@unlocalhosted/browsergrad-grad`](../browsergrad-grad/) — **lazy by default**: every op builds a UOp IR node, and computation defers until you call `.numpy()`, `.tolist()`, `.item()`, `.backward()`, or trip one of the documented Python protocol triggers.

Designed as the substrate for downstream optimization passes — fusion, symbolic backward, WGSL codegen, WebNN routing, megakernels, function transforms, ONNX export. See the [PRD index](../../docs/prd/README.md) for the full roadmap.

## Status

**Pre-1.0 — Week 1 of PRD-005.** What ships in this version:

- The UOp IR (`_ir.py`) — 23 opcodes, frozen-dataclass nodes, shape-validated construction, topological-sort helper.
- The `TensorProxy` stub — metadata-only attributes (`.shape`, `.dtype`, `.ndim`, `len()`, `__repr__`) work. Realization triggers raise `JitNotImplementedError` until Week 3.
- The per-session `BufferTable` — the lifecycle primitive that keeps two concurrent models from colliding on buffer ids.
- The install pipeline + codegen tooling, mirroring [`browsergrad-grad`](../browsergrad-grad/)'s shape exactly.

Roadmap weeks 2-10 land arithmetic, the NumPy realizer, autograd, and the `nn.Module` / `optim.*` compat layer in subsequent releases. Follow [PRD-005](../../docs/prd/PRD-005-jit-foundation.md) for details.

## Compatibility contract

| Surface | Stability |
| --- | --- |
| `TensorProxy` attributes & methods | Semver-stable across the `0.x` line |
| `nn.*`, `optim.*`, `functional.*` shapes (ship in later weeks) | Semver-stable |
| Public errors (`ShapeError`, `JitError`, etc.) | Semver-stable |
| `_ir` module, opcode strings, `UOp` dataclass | **Internal.** Changes freely across minor releases. |
| `BufferTable`, `Session` internals | Internal. The `Session` *class* is stable; its private slots are not. |
| IR serialization, trace cache format | **Not promised.** Do not depend on these on disk. |
| Per-opcode numerical match vs `browsergrad-grad` | Within `1e-4` for fp32 (documented per-PRD). |

If you reach for anything in the `Internal` row, expect breakage. File an issue describing what you're trying to do and we'll either lift the seam into the public API or document the supported equivalent.

## Installation

```bash
npm install @unlocalhosted/browsergrad-jit
```

In a browser tab via [`@unlocalhosted/browsergrad-runtime`](../browsergrad-runtime/):

```ts
import { createSession } from "@unlocalhosted/browsergrad-runtime";
import { installJit } from "@unlocalhosted/browsergrad-jit";

const session = await createSession({
  pyodideIndexURL: "/pyodide/v0.26.4/",
  packages: ["numpy"],
});
await installJit(session);
```

In Node (CI, smoke tests):

```ts
import { loadPyodide } from "pyodide";
import { installJit } from "@unlocalhosted/browsergrad-jit";
import { createNodePyodideTarget } from "@unlocalhosted/browsergrad-jit/node-adapter";

const py = await loadPyodide();
await py.loadPackage(["numpy"]);
await installJit(createNodePyodideTarget(py));
```

After install:

```python
import browsergrad_jit as jit
print(jit.__version__)              # the package version
sess = jit.new_session()            # opt-in per-loop isolation
```

## Coexistence with browsergrad-grad

Both packages can be installed in the same Pyodide worker. They mount to distinct sys.path entries (`/lib/browsergrad_grad_src` vs `/lib/browsergrad_jit_src`) and use distinct top-level module names (`browsergrad_grad` vs `browsergrad_jit`). The `torch` alias shim (PRD-005 Week 6) refuses to install if the other package already owns it — see [`TorchAliasConflict`](src/python/_errors.py).

This is the supported migration path: install both, run grad as the correctness oracle, exercise jit through the public API only, and assert numerical equivalence with `np.allclose(..., atol=1e-4)`.

## License

[MIT](LICENSE).
