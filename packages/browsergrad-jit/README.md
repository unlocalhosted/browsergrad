# @unlocalhosted/browsergrad-jit

[![npm version](https://img.shields.io/npm/v/@unlocalhosted/browsergrad-jit.svg)](https://www.npmjs.com/package/@unlocalhosted/browsergrad-jit)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A PyTorch-shaped Python tensor library that runs in the browser via Pyodide. Same API surface as [`browsergrad-grad`](../browsergrad-grad/) — but **lazy by default**: every op builds a UOp IR node, and computation defers until you call `.numpy()`, `.tolist()`, `.item()`, `.backward()`, `optimizer.step()`, or trip one of the documented Python protocol triggers.

The IR is the substrate for everything downstream: fusion, symbolic backward, AMP cast-insertion, gradient checkpointing as IR rewrite, functional transforms (vmap/grad/vjp), ONNX export, and pluggable backends (NumPy today, WebGPU via the realizer-bridge seam).

## What's shipped (v0.8.0)

| PRD | Surface | Status |
|---|---|---|
| 005 — IR + tracer + NumPy realizer | 28-opcode IR, `TensorProxy`, `nn.Module`, `nn.functional`, `optim.SGD/Adam/AdamW`, `install_torch_alias()` | ✅ |
| 006 — Kernel fusion | Elementwise chain + softmax DAG fusion; `bg.jit.debug_fused_kernels()` | ✅ |
| 007 — Symbolic backward | 13 VJP rules; closure backward kept as safety net | ✅ |
| 008 — Trace cache + safetensors | `bg.save_safetensors(state) -> bytes` (browser-friendly), `bg.load_safetensors(blob)` | ✅ |
| 009 — Gradient checkpointing | `bg.utils.checkpoint.checkpoint(fn, *args)`; IR rewrite at backward time | ✅ |
| 010 — Mixed precision | `bg.amp.autocast`, `bg.amp.GradScaler`, fp32 accumulator inside fp16 matmul | ✅ |
| 011.5 — WebGPU realizer seam | `bg.realize_webgpu(tensor)`, `bg.register_webgpu_bridge(bridge)` | ✅ |
| 012a — Tiled GEMM + fused codegen | Via the kernels package; bridge dispatches `matmulTiledDirect` + `fusedElementwiseDirect` | ✅ |
| 013 — Lab manifest + harness | `bg.lab.{assert_pytorch_match, assert_shape_match, assert_no_nan_inf}` | ✅ |
| 014 — Functional transforms | `bg.func.{grad, vjp, functional_call, vmap}` + partial vmap rules | ✅ |
| 015 — Custom WGSL kernels | `@bg.custom_kernel(wgsl=..., ...)` decorator | ✅ |
| 016 — ONNX export | `bg.onnx.export_inference(tensor, input_buffers=(...))` | ✅ |

For per-PRD architectural rationale (and what was deferred and why) see [PROGRESS.md](../../PROGRESS.md), [FEEDBACK.md](../../FEEDBACK.md), and the per-PRD docs under `docs/prd/`.

## Quick start

### Install

```bash
npm install @unlocalhosted/browsergrad-jit
```

### In a browser tab (via the runtime)

```ts
import { createSession } from "@unlocalhosted/browsergrad-runtime";
import { installJit } from "@unlocalhosted/browsergrad-jit";

const session = await createSession({
  pyodideIndexURL: "/pyodide/v0.26.4/",
  packages: ["numpy"],
});
await installJit(session);
```

### In Node (CI, tests)

```ts
import { loadPyodide } from "pyodide";
import { installJit } from "@unlocalhosted/browsergrad-jit";
import { createNodePyodideTarget } from "@unlocalhosted/browsergrad-jit/node-adapter";

const py = await loadPyodide();
await py.loadPackage(["numpy"]);
await installJit(createNodePyodideTarget(py));
```

### Train a model

```python
import browsergrad_jit as bg
import numpy as np

bg.manual_seed(0)
model = bg.nn.Sequential(
    bg.nn.Linear(8, 16),
    bg.nn.ReLU(),
    bg.nn.Linear(16, 4),
)
opt = bg.optim.SGD([p for p in model.parameters()], lr=0.01)

x = bg.from_numpy(np.random.randn(32, 8).astype(np.float32))
y = bg.from_numpy(np.random.randn(32, 4).astype(np.float32))

for _ in range(10):
    opt.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()
```

### Mixed precision

```python
with bg.amp.autocast(device_type="webgpu", dtype="float16"):
    pred = model(x)
    loss = ((pred - y) ** 2).mean()

scaler = bg.amp.GradScaler()
scaler.scale(loss).backward()
scaler.step(opt)
scaler.update()
```

### Gradient checkpointing

```python
from browsergrad_jit.utils.checkpoint import checkpoint

def block(x):
    return model_layers(x)

y = checkpoint(block, x)        # forward intermediates recomputed at backward
```

### Functional transforms

```python
# Functional gradient — does NOT mutate .grad
g = bg.func.grad(lambda t: (t * t).sum())(x)

# vjp — useful for non-scalar outputs
out, vjp_fn = bg.func.vjp(lambda t: t * 2.0, x)
(g_t,) = vjp_fn(bg.from_numpy(np.ones_like(x.numpy())))

# vmap — JAX-style batching (17 op rules; SCATTER_ADD/INDEX/MASK/etc. raise)
batched = bg.from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4))
per_sample = bg.func.vmap(lambda x: x.sum())(batched)   # shape [3]

# functional_call — stateless module evaluation (for MAML etc.)
out = bg.func.functional_call(model, {"weight": new_w, "bias": new_b}, (x,))
```

### Save / load weights

```python
state = {"w1": model[0].weight, "b1": model[0].bias}
blob = bg.save_safetensors(state)         # bytes — browser-friendly
restored = bg.load_safetensors(blob)
model[0].weight = restored["w1"]            # from_numpy accepts TensorProxy identity
```

### Export to ONNX

```python
y = (x @ w + b).relu()
onnx_bytes = bg.onnx.export_inference(y, input_buffers=(x,))
# Pipe onnx_bytes to a Blob URL + <a download> in JS
```

The encoder is a hand-rolled pure-Python proto3 writer (~250 LOC). No `onnx` package needed; runs identically in Pyodide and in Node. 14 ops mapped; opcodes outside the supported set raise `bg.onnx.OnnxUnmappableOp`.

### Run on real WebGPU

```ts
// JS side
import { createDevice, createWebGpuRealizerBridge } from "@unlocalhosted/browsergrad-kernels";

const device = await createDevice();
const bridge = createWebGpuRealizerBridge(device);
pyodide.registerJsModule("_bg_webgpu_bridge", bridge);
```

```python
# Python side
import browsergrad_jit as bg
from js import _bg_webgpu_bridge
bg.register_webgpu_bridge(_bg_webgpu_bridge)

out = bg.realize_webgpu(x @ w + b)   # ndarray, materialised at the seam
```

Forward-only in v0; backward stays on the NumPy realizer. Supported opcodes: `BUFFER, LOAD, CONST, CAST, MATMUL, FUSED_ELEMENTWISE, CUSTOM`. Anything else raises `JitNotImplementedError` with a pointer back to `bg.realize()` (NumPy).

### Custom WGSL kernel

```python
double_each = bg.custom_kernel(
    wgsl="""
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&A)) { return; }
  Out[i] = A[i] * 2.0;
}
""",
    name="double_each",
    workgroup_size=(64, 1, 1),
    output_shape_fn=lambda s0: s0,
    dispatch_shape_fn=lambda s0: (s0[0], 1, 1),
    num_inputs=1,
)
y = double_each(x)
out = bg.realize_webgpu(y)
```

SHA-256 of the WGSL = pipeline cache key. Forward only — `.backward()` through user-kernel graphs raises `NoBackwardError`.

## Coexistence with browsergrad-grad

Both packages can be installed in the same Pyodide worker. They mount to distinct sys.path entries (`/lib/browsergrad_grad_src` vs `/lib/browsergrad_jit_src`) and use distinct top-level module names. The `torch` alias shim uses an owner-token protocol — see [`TorchAliasConflict`](src/python/_errors.py) — so calling `install_torch_alias()` on both raises a clear error rather than silently overwriting.

Migration path: install both, exercise jit through the public API only, assert numerical equivalence with `np.allclose(..., atol=1e-4)`.

## Compatibility contract

| Surface | Stability |
| --- | --- |
| `TensorProxy` attributes & methods | Semver-stable across `0.x` |
| `nn.*`, `optim.*`, `nn.functional.*` shapes | Semver-stable |
| `bg.func.*`, `bg.amp.*`, `bg.utils.checkpoint.*`, `bg.lab.*`, `bg.onnx.*`, `bg.kernels.*`, `bg.custom_kernel` | Semver-stable |
| `bg.realize_webgpu`, `bg.register_webgpu_bridge`, `bg.webgpu_supported_opcodes()` | Semver-stable; supported-opcodes set may grow |
| Public errors (`ShapeError`, `JitError`, `JitNotImplementedError`, `NoBackwardError`, `TorchAliasConflict`, `RealizationError`, `BufferTableError`, `OnnxUnmappableOp`, `LabRuntimeMismatch`) | Semver-stable |
| `_ir` module, opcode strings, `UOp` dataclass | **Internal.** Changes freely across minor releases. |
| `BufferTable`, `Session` internals | The `Session` *class* is stable; private slots are not. |
| IR serialization, trace cache format | **Not promised.** Do not rely on these on disk. |
| Per-opcode numerical match vs `browsergrad-grad` | Within `1e-4` for fp32 |

Anything in the **Internal** row will break across minor releases. File an issue if you need an Internal surface lifted to public — we'll either lift it or document a supported equivalent.

## Known issues

See [`STATUS.md`](../../STATUS.md#known-issues) for the live list. Currently: FA-v2 kernel has a deterministic numerical bug on real WebGPU (tracked); `vmap(grad(fn))` composition has shape-broadcasting subtleties (PRD-014b polish).

## License

[MIT](LICENSE).
