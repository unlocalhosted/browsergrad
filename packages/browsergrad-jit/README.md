# @unlocalhosted/browsergrad-jit

[![npm version](https://img.shields.io/npm/v/@unlocalhosted/browsergrad-jit.svg)](https://www.npmjs.com/package/@unlocalhosted/browsergrad-jit)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A PyTorch-shaped Python tensor library that runs in the browser via Pyodide. **Lazy by default** — arithmetic builds a UOp graph; nothing executes until you call `.numpy()`, `.tolist()`, `.item()`, `.backward()`, or `optimizer.step()`.

The IR is the substrate for everything downstream: fusion, symbolic backward, AMP cast-insertion, gradient checkpointing IR rewrites, functional transforms (`grad`/`vjp`/`vmap`/`functional_call`), ONNX export, custom WGSL kernels, and pluggable backends (NumPy today, WebGPU via the realizer bridge).

## Install

```sh
npm install @unlocalhosted/browsergrad-jit
```

## Hello world

```python
import browsergrad_jit as bg
import numpy as np

bg.manual_seed(0)
model = bg.nn.Sequential(
    bg.nn.Linear(8, 16),
    bg.nn.ReLU(),
    bg.nn.Linear(16, 4),
)
opt = bg.optim.SGD(model.parameters(), lr=0.01)

x = bg.from_numpy(np.random.randn(32, 8).astype(np.float32))
y = bg.from_numpy(np.random.randn(32, 4).astype(np.float32))

for _ in range(10):
    opt.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()
```

## Public surface

### Tensor core
- `TensorProxy` (alias `Tensor`) — lazy tensor
- Factory: `tensor`, `from_numpy`, `zeros`, `ones`, `randn`, `arange`
- Arithmetic + reductions + shape ops + comparisons + dtype casts
- Autograd: `requires_grad`, `.backward()`, `.grad`

### Neural networks
- `nn.Module`, `nn.Sequential`
- `nn.Linear`, `nn.Dropout`, activation modules
- `nn.functional`: `relu`, `softmax`, `cross_entropy`, `mse_loss`, `nll_loss`, `linear`
- `optim.SGD`, `optim.Adam`, `optim.AdamW`

### Mixed precision
```python
with bg.amp.autocast(device_type="webgpu", dtype="float16"):
    pred = model(x)
    loss = ((pred - y) ** 2).mean()

scaler = bg.amp.GradScaler()
scaler.scale(loss).backward()
scaler.step(opt); scaler.update()
```

### Gradient checkpointing
```python
from browsergrad_jit.utils.checkpoint import checkpoint

def block(x): return model_layers(x)
y = checkpoint(block, x)   # forward intermediates recomputed at backward
```

### Data loading
```python
from browsergrad_jit.utils.data import TensorDataset, DataLoader

ds = TensorDataset(x, y)
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
```

`DataLoader` is intentionally single-process in Pyodide. `num_workers > 0`
raises with a browser-specific explanation.

### Functional transforms
```python
g = bg.func.grad(lambda t: (t * t).sum())(x)
out, vjp_fn = bg.func.vjp(lambda t: t * 2.0, x)
per_sample = bg.func.vmap(lambda t: t.sum())(batched)
out = bg.func.functional_call(model, {"weight": w, "bias": b}, (x,))
```

### Save / load
```python
state = {"w1": model[0].weight, "b1": model[0].bias}
blob = bg.save_safetensors(state)        # bytes — browser-friendly
restored = bg.load_safetensors(blob)
model[0].weight = restored["w1"]          # from_numpy accepts TensorProxy
```

### ONNX export
```python
y = (x @ w + b).relu()
onnx_bytes = bg.onnx.export_inference(y, input_buffers=(x,))
```

Hand-rolled pure-Python proto3 encoder. No `onnx` wheel needed. Opcodes outside the supported set raise `bg.onnx.OnnxUnmappableOp`.

### WebGPU realizer
```python
bg.register_webgpu_bridge(bridge)         # bridge built by browsergrad-kernels
out = bg.realize_webgpu(x @ w + b)        # ndarray, materialized at the seam
```

Forward-only in v0. Supported opcodes: `BUFFER`, `LOAD`, `CONST`, `CAST`, `MATMUL`, `FUSED_ELEMENTWISE`, `CUSTOM`. Other opcodes raise with a pointer back to `bg.realize()` (NumPy).

### Custom WGSL kernel
```python
double_each = bg.custom_kernel(
    wgsl="...",
    name="double_each",
    workgroup_size=(64, 1, 1),
    output_shape_fn=lambda s0: s0,
    dispatch_shape_fn=lambda s0: (s0[0], 1, 1),
    num_inputs=1,
)
y = double_each(x)
out = bg.realize_webgpu(y)
```

SHA-256 of the WGSL is the pipeline cache key. Forward-only.

### Lab harness (optional)
```python
bg.lab.assert_pytorch_match("forward_correct", actual, expected, rtol=1e-4)
bg.lab.assert_shape_match("shape_ok", t, (32, 10))
bg.lab.assert_no_nan_inf("clean_grads", w_grad)
```

Routes through the runtime's structured assertion protocol when run inside `@unlocalhosted/browsergrad-runtime`; falls back to structured stdout otherwise.

## PyTorch alias

```python
bg.install_torch_alias()
import torch, torch.nn, torch.func, torch.amp, torch.utils.checkpoint, torch.utils.data
```

The shim covers `torch.nn`, `torch.optim`, `torch.nn.functional`, `torch.func`, `torch.amp`, `torch.utils.checkpoint`, and `torch.utils.data`. Anything not implemented raises `AttributeError`, not silent wrong behavior.

## Coexists with browsergrad-grad

Both libraries can be installed in the same Pyodide worker. They mount to distinct `sys.path` entries and the `torch` alias uses an owner-token protocol so calling `install_torch_alias()` from both raises a clear error rather than silently overwriting.

## Compatibility contract

| Surface | Stability |
|---|---|
| `TensorProxy` attributes & methods | Semver-stable across `0.x` |
| `nn.*`, `optim.*`, `nn.functional.*` shapes | Semver-stable |
| `bg.func.*`, `bg.amp.*`, `bg.utils.checkpoint.*`, `bg.utils.data.*`, `bg.onnx.*`, `bg.kernels.*`, `bg.custom_kernel`, `bg.lab.*` | Semver-stable |
| `bg.realize_webgpu`, `bg.register_webgpu_bridge`, `bg.webgpu_supported_opcodes()` | Semver-stable; supported-opcode set may grow |
| Public errors (`ShapeError`, `JitError`, `JitNotImplementedError`, `NoBackwardError`, `TorchAliasConflict`, `RealizationError`, `BufferTableError`, `OnnxUnmappableOp`) | Semver-stable |
| Per-opcode numerical match vs `browsergrad-grad` | Within `1e-4` (fp32) |
| `_ir`, `_realize`, opcode strings, `UOp` dataclass | **Internal.** Changes freely. |
| IR serialization, trace cache format | **Not promised.** Do not depend on these on disk. |

Anything in the **Internal** row will break across minor releases. File an issue if you need an internal surface lifted to public.

## Testing

```sh
pnpm test                        # surface tests (no Pyodide)
pnpm test:integration            # Python correctness via Pyodide-in-node
```

170+ integration scenarios cover every public surface (training loops, gradient checkpointing, AMP, ONNX, functional transforms, etc.).

## License

[MIT](LICENSE).
