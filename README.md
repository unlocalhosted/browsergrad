# browsergrad

**PyTorch-shaped deep learning in the browser.** Lazy IR with fusion, symbolic backward, AMP, gradient checkpointing, functional transforms, WGSL kernels, ONNX export.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI status](https://img.shields.io/badge/tests-427%20passing-brightgreen.svg)](#testing)
[![browser](https://img.shields.io/badge/runs-in%20the%20browser-blue.svg)](#)

```python
import browsergrad_jit as bg
import numpy as np

model = bg.nn.Sequential(
    bg.nn.Linear(784, 128),
    bg.nn.ReLU(),
    bg.nn.Linear(128, 10),
)
opt = bg.optim.Adam(model.parameters(), lr=1e-3)

x = bg.from_numpy(np.random.randn(64, 784).astype(np.float32))
y = bg.from_numpy(np.random.randint(0, 10, size=(64,)).astype(np.int64))

for _ in range(100):
    opt.zero_grad()
    loss = bg.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
```

That code runs unmodified inside Pyodide in a browser tab. Same surface as PyTorch; no CUDA, no native compile step, no install — just `<script type="module">` and a Web Worker.

## Why

- **Same API as PyTorch.** `nn.Module`, `optim.Adam`, `nn.functional.cross_entropy`, `.backward()`, `torch.func.{grad, vjp, vmap, functional_call}`, `torch.amp.autocast`, `torch.utils.checkpoint`. Toggle a single flag and existing PyTorch code runs unchanged.
- **Lazy by default.** Arithmetic builds a UOp graph; nothing realizes until you ask for `.numpy()` or call `.backward()`. Enables fusion, AMP cast-insertion, gradient-checkpointing IR rewrites, and pluggable backends.
- **GPU when you want it.** Plug a WGSL backend via a small bridge protocol. Forward inference runs on the GPU; backward stays correct on NumPy. No CUDA. No driver install.
- **Real autograd, two paths.** Symbolic VJP rules emit IR; closure backward is the safety net. Verified against finite differences and hand-derived oracles.
- **Save and ship.** safetensors for weights (returns bytes — browser-friendly), ONNX export for inference graphs (pure-Python proto3 encoder, no protobuf wheel).
- **Honest about scope.** Per-PRD design reviews kill speculative ambitions before they ship. What's listed is what works. Limitations are listed too.

## Install

```sh
npm install @unlocalhosted/browsergrad-runtime pyodide
npm install @unlocalhosted/browsergrad-jit
npm install @unlocalhosted/browsergrad-kernels        # optional: WGSL kernels + WebGPU bridge
npm install @unlocalhosted/browsergrad-grad           # optional: eager-autograd alternative
npm install @unlocalhosted/browsergrad-tokenizers     # optional: tokenizer/BPE + streaming primitives
npm install @unlocalhosted/browsergrad-simulators     # optional: parallel/distributed execution simulators
npm install @unlocalhosted/browsergrad-snapshots      # optional: JSON/numeric snapshot comparison
npm install @unlocalhosted/browsergrad-data           # optional: HTML/dedupe/PII/data-quality primitives
npm install @unlocalhosted/browsergrad-scaling        # optional: scaling-law + hosted API mock primitives
npm install @unlocalhosted/browsergrad-alignment      # optional: DPO/GRPO/alignment math primitives
```

`pyodide` is a peer dependency. Asset-sync into `public/pyodide/v0.26.4/` so the runtime is served same-origin.

## Quick start

### Boot Pyodide in a Worker

```ts
import { createSession } from "@unlocalhosted/browsergrad-runtime";
import { installJit } from "@unlocalhosted/browsergrad-jit";

const session = await createSession({
  pyodideIndexURL: "/pyodide/v0.26.4/",
  packages: ["numpy"],
});
await installJit(session);

await session.exec({
  code: `
    import browsergrad_jit as bg
    import numpy as np

    x = bg.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
    print((x @ x.T).numpy())
  `,
});
```

### Use the PyTorch alias

```python
import browsergrad_jit as bg
bg.install_torch_alias()

import torch
import torch.nn as nn
import torch.func

x = torch.from_numpy(...)
g = torch.func.grad(lambda t: (t * t).sum())(x)
```

Anything browsergrad doesn't implement raises `AttributeError`, not silent wrong behavior.

### Run on real WebGPU

```ts
import { createDevice, createWebGpuRealizerBridge } from "@unlocalhosted/browsergrad-kernels";

const device = await createDevice();
const bridge = createWebGpuRealizerBridge(device);
pyodide.registerJsModule("_bg_webgpu_bridge", bridge);
```

```python
from js import _bg_webgpu_bridge
bg.register_webgpu_bridge(_bg_webgpu_bridge)

out = bg.realize_webgpu(x @ w + b)   # tiled GEMM, fused elementwise, custom WGSL
```

## Packages

| Package | What it does |
|---|---|
| [`browsergrad-runtime`](./packages/browsergrad-runtime) | Pyodide-in-Worker host. `createSession`, `exec`, structured assertion + artifact protocol, AbortSignal cancellation, optional lab-manifest validator. |
| [`browsergrad-jit`](./packages/browsergrad-jit) | Lazy-IR PyTorch-shape library. 28-opcode IR, fusion, symbolic VJP, AMP, gradient checkpointing, `bg.func.*`, custom WGSL kernels, ONNX export. |
| [`browsergrad-kernels`](./packages/browsergrad-kernels) | WGSL compute-shader catalog, CUDA-shaped 1D program authoring, and pure-JS references for attention, tensor kernels, and GPU teaching subsets. |
| [`browsergrad-grad`](./packages/browsergrad-grad) | Eager-autograd alternative. PyTorch-flavored, NumPy-backed, closure backward. Stable. |
| [`browsergrad-tokenizers`](./packages/browsergrad-tokenizers) | Pure TypeScript tokenizer/BPE primitives and streaming gates. |
| [`browsergrad-simulators`](./packages/browsergrad-simulators) | Deterministic simulators for SIMD/thread/task traces, worker-mesh collectives, DDP/FSDP, and sharded optimizer behavior. |
| [`browsergrad-snapshots`](./packages/browsergrad-snapshots) | Browser-safe JSON/numeric snapshot comparison for structured and tensor-like outputs. |
| [`browsergrad-data`](./packages/browsergrad-data) | Browser-safe data primitives for HTML extraction, exact/near dedupe, quality rules, and PII masking. |
| [`browsergrad-scaling`](./packages/browsergrad-scaling) | Browser-safe hosted API mocks, budget schedulers, and scaling-law fitters. |
| [`browsergrad-alignment`](./packages/browsergrad-alignment) | Browser-safe DPO, GRPO, rollout reward, parser, and policy-gradient math primitives. |

Each package is independently consumable; they share an npm scope but no runtime dependency. Take one or all.

## Testing

Workspace tests cover package surfaces, Pyodide integration, and browser WebGPU:

```sh
pnpm -r test
pnpm -r test:integration                                    # 311 Pyodide-in-node tests
pnpm --filter @unlocalhosted/browsergrad-kernels test:browser    # 7 real-Chromium WebGPU tests
pnpm --filter @unlocalhosted/browsergrad-simulators test
pnpm --filter @unlocalhosted/browsergrad-snapshots test
pnpm --filter @unlocalhosted/browsergrad-data test
pnpm --filter @unlocalhosted/browsergrad-scaling test
pnpm --filter @unlocalhosted/browsergrad-alignment test
```

The browser-mode suite runs the WGSL kernels and the realizer bridge against an actual `GPUDevice` via Playwright + Chromium. It catches shader-level bugs that NumPy mocks miss.

## Documentation

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — how the packages compose; data flow; design principles
- [`CHANGELOG.md`](./CHANGELOG.md) — release history
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — how to contribute
- [`SECURITY.md`](./SECURITY.md) — vulnerability reporting
- [`docs/`](./docs) — design documents, PRDs, and internal notes

Per-package READMEs cover the package-level API surface and stability contracts.

## License

[MIT](LICENSE).
