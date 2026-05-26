# @unlocalhosted/browsergrad-jit

[![npm version](https://img.shields.io/npm/v/@unlocalhosted/browsergrad-jit.svg)](https://www.npmjs.com/package/@unlocalhosted/browsergrad-jit)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A PyTorch-shaped Python tensor library that runs in the browser via Pyodide. Same API surface as [`@unlocalhosted/browsergrad-grad`](../browsergrad-grad/) — **lazy by default**: every op builds a UOp IR node, and computation defers until you call `.numpy()`, `.tolist()`, `.item()`, `.backward()`, or trip one of the documented Python protocol triggers.

Designed as the substrate for downstream optimization passes — fusion, symbolic backward, WGSL codegen, WebNN routing, megakernels, function transforms, ONNX export. See the [PRD index](../../docs/prd/README.md) for the full roadmap.

## Status

**0.1.0 — PRD-005 minimum-viable.** The elementwise + MLP scope is real:

- The 23-opcode UOp IR (`_ir.py`).
- The NumPy realizer (`_realize.py`) — one dispatch handler per opcode.
- `TensorProxy` with full arithmetic surface, reductions, shape ops, dtype casts, broadcasting via NumPy.
- Closure-based autograd. `.backward()` populates leaf parameter gradients per PyTorch semantics.
- `nn.Module`, `nn.Linear`, `nn.Sequential`, `nn.Dropout`, activation + loss modules.
- `nn.functional`: `relu`, `softmax`, `cross_entropy`, `mse_loss`, `nll_loss`, `linear`, etc.
- `optim.SGD`, `optim.Adam`, `optim.AdamW`.
- `install_torch_alias()` with the owner-token protocol so `browsergrad-jit` and `browsergrad-grad` coexist cleanly.
- Per-session `BufferTable` for multi-loop isolation.

**Not yet in 0.1.0** — slipping to 0.1.x patches per PRD-005's revised plan:

- `nn.Conv1d`/`Conv2d`/`Conv3d`, pooling, batch/layer/group norm, embedding, RNN/LSTM/GRU, MultiHeadAttention.
- Symbolic backward (PRD-007), kernel fusion (PRD-006), pipeline caching (PRD-008), WGSL (PRD-002 extends into jit), mixed precision (PRD-010), function transforms (PRD-014), WebNN (PRD-011), megakernels (PRD-012), ONNX export (PRD-016).

Follow [PRD-005](../../docs/prd/PRD-005-jit-foundation.md) and the other PRDs in `docs/prd/` for the roadmap.

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
