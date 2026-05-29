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

> **Status: v0.4.6 — stable.** Comprehensive layer set for CNNs and Transformers: Conv2d/Conv1d, BatchNorm2d/1d, LayerNorm, MaxPool/AvgPool, AdaptiveAvgPool2d, Dropout/Dropout2d, Embedding, MultiHeadAttention, Flatten + all common activations. Optimizers: SGD/Adam/AdamW. Module.train()/eval() mode system. **140 tests green** (25 surface + 115 Pyodide-in-node integration), with end-to-end training checks for MLP, CNN, sequence-CNN, and transformer-block.
>
> **The lazy-IR successor is [`browsergrad-jit`](../browsergrad-jit/)** — same PyTorch surface, but ops build a UOp graph that fusion / symbolic backward / AMP / gradient checkpointing / functional transforms / ONNX export / WebGPU realizer-bridge all hook into. Use grad for stable curriculum content; use jit when you want fusion + GPU acceleration + the broader toolkit. Both coexist in the same Pyodide session.

## What this is

PyTorch-flavored API, NumPy-backed, **deliberately not PyTorch.** The Python module is named `browsergrad_grad`, not `torch` — because pretending to be PyTorch traps you into PyTorch's full surface area, and we want to stay small enough to read.

The library is meant to be **legible source code**. If you `print(inspect.getsource(grad.Tensor))` you should be able to follow what's happening. The whole package is ~450 lines of Python.

## What this is not

- ❌ PyTorch. We don't try to match its API exactly.
- ❌ A polyfill. Don't expect `import torch` to work.
- ❌ Production-fast. NumPy-on-CPU. **GPU acceleration lives in [`browsergrad-jit`](../browsergrad-jit/)** via the WebGPU realizer-bridge (PRD-011.5) — if you need throughput, migrate to jit; if you want stable curriculum semantics, stay here.
- ❌ A general framework. It's a **teaching artifact** sized to fit in your head.

## Install

```sh
npm install @unlocalhosted/browsergrad-grad
```

No dependencies. Pyodide is not a peer dep — `installGrad` works with any
duck-typed Pyodide target.

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

`pyodide` is an `optionalPeerDependencies` — bring your own version. The adapter has no other dependencies.

## Python API surface (v0.2.0)

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
t.detach()                         # leaf copy, no autograd

# Arithmetic — broadcasts in v0.2
a + b, a - b, a * b, a / b, -a
a @ b                              # any rank ≥ 2, batch dims broadcast
a ** 2.0                           # scalar power only
a.exp(), a.log()                   # elementwise

# Shape
a.reshape(*shape), a.view(*shape), a.transpose(d0, d1), a.T   # 2D only

# Reductions (axis-aware)
t.sum(), t.sum(axis=1, keepdims=True)
t.mean(axis=-1)

# Autograd
loss.backward()                    # accumulates into .grad of every leaf

# Functional
import browsergrad_grad.functional as F
F.relu(x), F.leaky_relu(x, 0.01), F.sigmoid(x), F.tanh(x), F.gelu(x)
F.softmax(x, dim=-1), F.log_softmax(x, dim=-1)
F.mse_loss(y_hat, y)               # regression
F.cross_entropy_loss(logits, targets)   # classification (fused, stable)
F.nll_loss(log_probs, targets)

# Neural net building blocks
import browsergrad_grad.nn as nn
nn.Module                          # base — auto-tracks Tensor params
nn.Linear(in_features, out_features, bias=True)
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
nn.MaxPool2d(kernel_size, stride=None, padding=0)
nn.AvgPool2d(kernel_size, stride=None, padding=0)
nn.AdaptiveAvgPool2d(output_size)
nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True)
nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True)   # (N,C) or (N,C,L)
nn.LayerNorm(normalized_shape, eps=1e-5)
nn.Embedding(num_embeddings, embedding_dim)
nn.MultiHeadAttention(embed_dim, num_heads, bias=True)              # (N, S, D)
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

## What's NOT yet in (v0.3+ targets)

These are documented as deferred — additive when they land:

- **Conv1d / Conv3d.** v0.4 if needed; v0.3 ships Conv2d only.
- **Tuple kernel/stride/padding shapes, dilation, groups, ConvTranspose2d.** v0.4+.
- **im2col + matmul optimization** for Conv2d. v0.3 ships naive nested loops (correct + readable). Refactor candidate when perf matters.
- **BatchNorm / GroupNorm.** Mostly needed for older CNN architectures.
- **Dropout, RNN, LSTM, GRU.** Less central for transformer-focused curriculum.
- **WebGPU dispatch.** v0 is pure NumPy. v0.3 will add an optional `device` arg
  that dispatches matmul / softmax / layernorm / attention to a `KernelDevice`
  from `@unlocalhosted/browsergrad-kernels` when provided.

## Design notes

- **No `_ctx`-mutability shenanigans.** Each op captures the data it needs at forward time and binds it in a closure. Backward functions are pure.
- **No global gradient context.** No `torch.no_grad()` yet — to detach from autograd, call `.detach()` to get a fresh leaf.
- **Reverse-mode only.** No forward-mode, no functional transforms (vmap, etc.).
- **`Tensor.__slots__`.** Slot-based attribute layout to keep memory predictable for tensors in long training loops.

## API reference

See [`src/python/*.ts`](./src/python/) — every Python module is embedded as a `*_PY` template literal in its own TS file. That's where the source code lives; reading those files is the documentation.

## License

MIT
