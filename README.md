# browsergrad

> A small, readable, well-tested library family for running Python and machine-learning workloads directly in the browser.

```
@unlocalhosted/browsergrad-runtime   Pyodide-in-Worker host with a structured assertion/artifact protocol
@unlocalhosted/browsergrad-kernels   WGSL compute-shader catalog for ML primitives (matmul, attention, …)
@unlocalhosted/browsergrad-grad      PyTorch-flavored tensor + autograd library that runs inside Pyodide
```

Each package is **independently consumable** — they share an organization scope on npm but no runtime dependency. Take one or all.

## Install

```sh
npm install @unlocalhosted/browsergrad-runtime pyodide
npm install @unlocalhosted/browsergrad-kernels
npm install @unlocalhosted/browsergrad-grad
```

`pyodide` is a peer dependency of the runtime — you install it directly so you control the version and the asset-sync story (assets must be served same-origin).

## Quick start

### Run Python in a Worker

```ts
import { createSession } from "@unlocalhosted/browsergrad-runtime";

const session = await createSession({
  pyodideIndexURL: "/pyodide/v0.26.4/",
  packages: ["numpy"],
});

await session.exec({
  code: `
    import numpy as np
    print(np.arange(10).sum())
  `,
  onStdout: (chunk) => console.log(chunk),
});

await session.dispose();
```

### Train a model with browsergrad-grad

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
    import browsergrad_grad.nn as nn
    import browsergrad_grad.optim as optim
    import browsergrad_grad.functional as F

    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
    opt = optim.Adam(model.parameters(), lr=1e-2)
    # ... training loop ...
  `,
});
```

### Or write vanilla PyTorch code

After `grad.install_torch_alias()`, the `torch` namespace is registered:

```python
import browsergrad_grad as grad
grad.install_torch_alias()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Standard PyTorch user code from here on
model = nn.Linear(4, 2)
opt = optim.Adam(model.parameters(), lr=1e-2)
loss = F.cross_entropy(model(torch.tensor(X)), y)
loss.backward()
opt.step()
```

The shim covers the subset of `torch` that browsergrad-grad implements. Anything unsupported (e.g. `torch.cuda`) raises `AttributeError` rather than silently faking success.

### Use a WGSL kernel directly

```ts
import { createDevice, kernels, tensor } from "@unlocalhosted/browsergrad-kernels";

const device = await createDevice();
const A = tensor([2, 3], new Float32Array([1, 2, 3, 4, 5, 6]));
const B = tensor([3, 2], new Float32Array([7, 8, 9, 10, 11, 12]));
const C = await kernels.matmul(device, A, B);
```

Pure-JS reference implementations are at the `/reference` subpath for environments without WebGPU:

```ts
import { reference } from "@unlocalhosted/browsergrad-kernels/reference";
const C = reference.matmul(A, B);
```

## Packages

| Package | What it does |
|---|---|
| [`browsergrad-runtime`](./packages/browsergrad-runtime) | Spawns Pyodide in a Web Worker, exposes `exec` / `fs` / structured assertion + artifact protocol, AbortSignal cancellation. |
| [`browsergrad-kernels`](./packages/browsergrad-kernels) | WGSL kernels for ML primitives — matmul, softmax, relu, gelu, layernorm, attention — each shipped with a pure-JS reference impl for conformance and CPU fallback. |
| [`browsergrad-grad`](./packages/browsergrad-grad) | Tensor + reverse-mode autograd in Python, runs inside any Pyodide-shaped target. PyTorch-flavored API. Every standard NN layer for CNN + transformer work, all standard optimizers + LR schedulers, full train/eval mode system. Optional `torch` namespace shim for unmodified PyTorch user code. |

Each package has its own README and CHANGELOG with full API details.

## Documentation

- [STATUS.md](./STATUS.md) — current state, test counts, supported APIs, known limitations
- [DEVELOPMENT.md](./DEVELOPMENT.md) — how the library is developed; TDD methodology; quality gates
- [CONTRIBUTING.md](./CONTRIBUTING.md) — how to contribute
- [SECURITY.md](./SECURITY.md) — how to report vulnerabilities

## License

MIT. See [LICENSE](./LICENSE).
