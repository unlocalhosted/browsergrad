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

> **Status: v0.2.0.** Broadcasting throughout autograd, higher-rank matmul, Adam/AdamW, cross-entropy, GELU/softmax/log_softmax, LayerNorm, Embedding. Enough to write a transformer block by hand.

## What this is

PyTorch-flavored API, NumPy-backed, **deliberately not PyTorch.** The Python module is named `browsergrad_grad`, not `torch` — because pretending to be PyTorch traps you into PyTorch's full surface area, and we want to stay small enough to read.

The library is meant to be **legible source code**. If you `print(inspect.getsource(grad.Tensor))` you should be able to follow what's happening. The whole package is ~450 lines of Python.

## What this is not

- ❌ PyTorch. We don't try to match its API exactly.
- ❌ A polyfill. Don't expect `import torch` to work.
- ❌ Production-fast. v0 is NumPy-on-CPU. WebGPU acceleration via `@unlocalhosted/browsergrad-kernels` is planned for v0.3.
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
nn.LayerNorm(normalized_shape, eps=1e-5)
nn.Embedding(num_embeddings, embedding_dim)
nn.Sequential(m1, m2, m3)
nn.ReLU(), nn.LeakyReLU(0.01), nn.Sigmoid(), nn.Tanh(), nn.GELU()

# Optimization
import browsergrad_grad.optim as optim
optim.SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0)
optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
```

## What's NOT yet in (v0.3+ targets)

These are documented as deferred — additive when they land:

- **Conv1d / Conv2d.** Highest-priority v0.3 target for CNN labs.
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
