# @unlocalhosted/browsergrad-grad

[![npm](https://img.shields.io/npm/v/@unlocalhosted/browsergrad-grad.svg)](https://www.npmjs.com/package/@unlocalhosted/browsergrad-grad)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A small, readable tensor + autograd library that runs inside Pyodide.

```python
import browsergrad_grad as grad
import browsergrad_grad.functional as F

x = grad.Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad.tolist())   # [2.0, 4.0, 6.0]
```

> **Status: v0.5.2 — stable.** Comprehensive layer set for CNNs and Transformers: ConvTranspose2d, Conv3d/2d/1d, BatchNorm3d/2d/1d, GroupNorm/InstanceNorm2d, LayerNorm, MaxPool/AvgPool, AdaptiveAvgPool2d, Dropout/Dropout2d, Embedding, MultiHeadAttention, RNN/LSTM/GRU, Flatten + all common activations. Optimizers: SGD/Adam/AdamW plus LR schedulers. Module.train()/eval(), hooks, state_dict/load_state_dict, torch-alias compatibility shims, and end-to-end training checks for MLP, CNN, sequence-CNN, and transformer-block.
>
> **The lazy-IR successor is [`browsergrad-jit`](../browsergrad-jit/)** — same PyTorch surface, but ops build a UOp graph that fusion / symbolic backward / AMP / gradient checkpointing / functional transforms / ONNX export / WebGPU realizer-bridge all hook into. Use grad for stable curriculum content; use jit when you want fusion + GPU acceleration + the broader toolkit. Both coexist in the same Pyodide session.

## What this is

PyTorch-flavored API, NumPy-backed, **deliberately not PyTorch.** The Python module is named `browsergrad_grad`, not `torch`; call `install_torch_alias()` when you want the supported `torch` shim. Unsupported PyTorch APIs fail loudly instead of pretending to work.

The library is meant to be **legible source code**. Tensor/autograd, functional ops, optimizers, and `nn` chunks live as editable Python files under `src/python/`; codegen embeds them into TypeScript for installation into Pyodide.

## What this is not

- ❌ PyTorch. We don't try to match its full API.
- ❌ A full polyfill. `install_torch_alias()` supports common tutorial code, but CUDA, distributed, compile/fx/jit, ONNX, quantization, and multi-process loaders remain explicit browser-scope refusals.
- ❌ Production-fast. NumPy-on-CPU by default. A forward-only `device=` escape hatch can dispatch matmul / softmax / layernorm / unmasked 2D attention through `@unlocalhosted/browsergrad-kernels`, but throughput-oriented model execution still belongs in [`browsergrad-jit`](../browsergrad-jit/) via its WebGPU realizer-bridge.
- ❌ A general framework. It's a **teaching artifact** sized to fit in your head.

## Install

```sh
npm install @unlocalhosted/browsergrad-grad
```

The npm package includes `@unlocalhosted/browsergrad-kernels` as a runtime
dependency so the explicit `device=` bridge is available without a second
install. Kernels owns its compatible `@unlocalhosted/browsergrad-semantic-core`
dependency; applications do not need to install semantic-core directly.

Pyodide is a standard optional peer. Applications using the direct Node
adapter install a compatible `pyodide@^0.26.4`; `installGrad` itself continues
to accept runtime-managed and other duck-typed Pyodide targets.

## Usage

```ts
import { createSession } from "@unlocalhosted/browsergrad-runtime";
import { installGrad } from "@unlocalhosted/browsergrad-grad";

const session = await createSession({
  pyodideIndexURL: "/pyodide/v0.26.4/",
  packages: ["numpy"],
});
await installGrad(session);

await session.exec({
  code: `
    import browsergrad_grad as grad
    import browsergrad_grad.functional as F

    # Tiny regression: y = 3x + 1, learn it.
    X = grad.randn(32, 1, seed=0)
    y_true = X * 3.0 + 1.0

    model = grad.nn.Linear(1, 1)
    opt = grad.optim.SGD(model.parameters(), lr=0.1)

    for step in range(200):
        opt.zero_grad()
        y_hat = model(X)
        loss = F.mse_loss(y_hat, y_true)
        loss.backward()
        opt.step()

    print(f"learned: y ≈ {model.weight.item():.2f} x + {model.bias.item():.2f}")
  `,
  onStdout: (s) => console.log(s),
});
```

Works with any Pyodide target — not just our runtime. Anything with an async `exec({code})` method works:

```ts
await installGrad({
  exec: async ({ code }) => pyodide.runPythonAsync(code),
});
```

For Node scripts and CI where you `loadPyodide()` directly, use the shipped adapter at the `./node-adapter` subpath — it wraps Pyodide's `FS.writeFile + FS.mkdirTree` to go through `installViaFs` (faster than `installViaExec`):

```ts
import { loadPyodide } from "pyodide";
import { installGrad } from "@unlocalhosted/browsergrad-grad";
import { createNodePyodideTarget } from "@unlocalhosted/browsergrad-grad/node-adapter";

const py = await loadPyodide();
await py.loadPackage(["numpy"]);
await installGrad(createNodePyodideTarget(py));
```

`pyodide` is an optional peer — direct-adapter consumers bring their own
compatible version. The adapter has no other dependencies.

Optional WebGPU forward dispatch:

```ts
import { createDevice } from "@unlocalhosted/browsergrad-kernels";
import { createGradKernelDeviceBridge } from "@unlocalhosted/browsergrad-grad/kernel-device";

const device = await createDevice();
const gradDevice = createGradKernelDeviceBridge(device);
py.globals.set("grad_device", gradDevice);

await py.runPythonAsync(`
import browsergrad_grad as grad
import browsergrad_grad.functional as F

a = grad.Tensor([[1., 2.], [3., 4.]])
b = grad.Tensor([[5., 6.], [7., 8.]])
y = grad.matmul(a, b, device=grad_device)
p = F.softmax(y, dim=-1, device=grad_device)
`);
```

`device=` is intentionally small: 2D `grad.matmul` / `grad.mm`, last-dim
`F.softmax`, last-dim `nn.LayerNorm(..., device=...)`, and unmasked default
`F.scaled_dot_product_attention(..., device=...)` for 2D Q/K/V. Backward still
uses BrowserGrad's CPU formulas after the GPU forward result is materialized.

## Python API surface (v0.5)

```python
import browsergrad_grad as grad

# Construction
t = grad.Tensor([1, 2, 3], requires_grad=False)
z = grad.zeros(3, 4)
o = grad.ones(2, 2)
r = grad.randn(5, 5, seed=42)

# Properties
t.shape, t.ndim, t.size, t.data    # numpy view
t.numpy(), t.tolist(), t.item()    # exports
t.detach()                         # storage-sharing leaf, no autograd
t.to("float64")                    # owning differentiable floating cast
t.to("cpu"), t.cpu()               # CPU identity
t.to("cuda"), t.cuda()             # explicit NotImplementedError

# Arithmetic — broadcasts in v0.2
a + b, a - b, a * b, a / b, -a
a @ b                              # any rank ≥ 2, batch dims broadcast
a ** 2.0                           # scalar power only
a.exp(), a.log()                   # elementwise

# Shape
a.reshape(*shape), a.view(*shape), a.transpose(d0, d1), a.T   # 2D only
a.contiguous()                     # identity if C-order, otherwise owning copy

# Reductions (axis-aware)
t.sum(), t.sum(axis=1, keepdims=True)
t.mean(axis=-1)

# Autograd
loss.backward()                    # accumulates into .grad of every leaf

# Functional
import browsergrad_grad.functional as F
F.relu(x), F.leaky_relu(x, 0.01), F.sigmoid(x), F.tanh(x), F.gelu(x)
F.softmax(x, dim=-1), F.log_softmax(x, dim=-1)
F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
F.mse_loss(y_hat, y)               # regression
F.cross_entropy_loss(logits, targets)   # classification (fused, stable)
F.nll_loss(log_probs, targets)

# Neural net building blocks
import browsergrad_grad.nn as nn
nn.Module                          # base — auto-tracks Tensor params
nn.Linear(in_features, out_features, bias=True)
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
nn.MaxPool2d(kernel_size, stride=None, padding=0)
nn.AvgPool2d(kernel_size, stride=None, padding=0)
nn.AdaptiveAvgPool2d(output_size)
nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True)
nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True)   # (N,C) or (N,C,L)
nn.BatchNorm3d(num_features, eps=1e-5, momentum=0.1, affine=True)
nn.GroupNorm(num_groups, num_channels, eps=1e-5, affine=True)
nn.InstanceNorm2d(num_features, eps=1e-5, affine=False)
nn.LayerNorm(normalized_shape, eps=1e-5, device=None)
nn.Embedding(num_embeddings, embedding_dim)
nn.MultiHeadAttention(embed_dim, num_heads, bias=True)              # (N, S, D)
nn.RNN(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False)
nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False)
nn.GRU(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False)
nn.Dropout(p=0.5)
nn.Dropout2d(p=0.5)                                                 # channel-wise
nn.Flatten(start_dim=1, end_dim=-1)
nn.Sequential(m1, m2, m3)
nn.ReLU(), nn.LeakyReLU(0.01), nn.Sigmoid(), nn.Tanh(), nn.GELU()
# Mode control (cross-cutting):
model.train()    # train-mode behavior (BN uses batch stats; Dropout drops)
model.eval()     # eval-mode behavior  (BN uses running stats; Dropout identity)

# Optimization
import browsergrad_grad.optim as optim
optim.SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0)
optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
```

## Current capability status

Done since the original v0.3 cut:

- **Conv2d** now uses im2col + batched matmul, supports tuple `kernel_size` / `stride` / `padding`, `dilation`, and `groups`.
- **ConvTranspose2d**, **Conv3d/2d/1d**, **BatchNorm3d/2d/1d**, **GroupNorm**, **InstanceNorm2d**, **Dropout/Dropout2d**, **RNN/LSTM/GRU** (multi-layer + bidirectional), **Embedding**, and **MultiHeadAttention** are in.
- **Norm backward correctness** now covers full statistics-aware input gradients for BatchNorm3d, GroupNorm, and InstanceNorm2d.
- **Module ergonomics** now cover `train()` / `eval()`, hooks, buffers, `state_dict`, `load_state_dict`, and torch compatibility shims.

- **WebGPU forward dispatch** is in as an explicit `device=` path over `@unlocalhosted/browsergrad-kernels` for matmul, softmax, layernorm, and attention.

Remaining explicit limits:

- **Direct eager GPU scope.** `device=` is forward-only and intentionally explicit. It does not make all eager ops GPU-backed, and it materializes results before CPU autograd.

## Design notes

- **No `_ctx`-mutability shenanigans.** Each op captures the data it needs at forward time and binds it in a closure. Backward functions are pure.
- **Contiguous means contiguous.** Non-C-order storage is copied into owning
  C-order storage with a dtype-preserving identity-gradient edge.
- **Global gradient context exists.** `grad.no_grad()` disables graph
  construction for inference sections; `.detach()` returns a distinct
  storage-sharing leaf with no autograd history.
- **Floating casts stay differentiable.** `.to()` casts among
  float16/float32/float64 retain an autograd edge and restore the source dtype
  in backward. Bool/integer casts are detached.
- **Device requests are truthful.** Eager tensor storage is CPU/Pyodide-backed.
  CPU requests preserve identity; unavailable CUDA/MPS/XPU/Meta requests fail
  before execution and never masquerade as a transfer.
- **Reverse-mode only.** No forward-mode, no functional transforms (vmap, etc.).
- **`Tensor.__slots__`.** Slot-based attribute layout to keep memory predictable for tensors in long training loops.

## API reference

See [`src/python/*.ts`](./src/python/) — every Python module is embedded as a `*_PY` template literal in its own TS file. That's where the source code lives; reading those files is the documentation.

## License

MIT
