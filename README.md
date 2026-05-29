# browsergrad

> A small, readable, well-tested library family for running Python and machine-learning workloads directly in the browser.

```
@unlocalhosted/browsergrad-runtime   Pyodide-in-Worker host + structured assertion/artifact protocol + lab manifest
@unlocalhosted/browsergrad-kernels   WGSL kernels for ML primitives + WebGpuRealizerBridge for jit
@unlocalhosted/browsergrad-grad      PyTorch-flavored tensor + autograd library (eager, NumPy-backed). Stable.
@unlocalhosted/browsergrad-jit       Lazy-IR successor: same PyTorch surface + fusion + symbolic backward
                                      + AMP + checkpointing + functional transforms + ONNX + WebGPU seam
```

Each package is **independently consumable** — they share an organization scope on npm but no runtime dependency. Take one or all.

## Install

```sh
npm install @unlocalhosted/browsergrad-runtime pyodide
npm install @unlocalhosted/browsergrad-kernels
npm install @unlocalhosted/browsergrad-grad   # eager autograd
npm install @unlocalhosted/browsergrad-jit    # lazy IR + fusion + GPU seam
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

### Train a model with browsergrad-jit (lazy IR + fusion)

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
    loss.backward()      # symbolic backward via VJP rules
    opt.step()

# Save weights — browser-friendly bytes, no filesystem required
state = {"w1": model[0].weight, "b1": model[0].bias,
         "w2": model[2].weight, "b2": model[2].bias}
blob = bg.save_safetensors(state)
```

The jit library ships fusion, AMP, gradient checkpointing, functional transforms (`bg.func.{grad, vjp, vmap, functional_call}`), custom WGSL kernels (`@bg.custom_kernel`), and ONNX export (`bg.onnx.export_inference`). See [`packages/browsergrad-jit/README.md`](./packages/browsergrad-jit/README.md) for the full surface.

### Run on real WebGPU (jit + kernels seam)

The jit library realizes through a pluggable bridge. The kernels package ships the production WebGPU bridge:

```ts
// JS side — create the device + bridge
import { createDevice, createWebGpuRealizerBridge } from "@unlocalhosted/browsergrad-kernels";

const device = await createDevice();
const bridge = createWebGpuRealizerBridge(device);

// Hand to Pyodide so Python can call into it
pyodide.registerJsModule("_bg_webgpu_bridge", bridge);
```

```python
# Python side — register and realize
import browsergrad_jit as bg
from js import _bg_webgpu_bridge
bg.register_webgpu_bridge(_bg_webgpu_bridge)

# Now graphs realize on the GPU
out = bg.realize_webgpu(x @ w + b)   # ndarray, materialised at the seam
```

Backward stays on the NumPy realizer in v0; the GPU path is forward-inference for now.

### Or use the eager library (browsergrad-grad)

```python
import browsergrad_grad as grad
import browsergrad_grad.nn as nn
import browsergrad_grad.optim as optim

model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
opt = optim.Adam(model.parameters(), lr=1e-2)
# ... training loop ...
```

### Or write vanilla PyTorch code

After `install_torch_alias()`, the `torch` namespace is registered against either library:

```python
import browsergrad_jit as bg
bg.install_torch_alias()

import torch
import torch.nn as nn
import torch.func        # → bg.func
import torch.amp         # → bg.amp
import torch.utils.checkpoint   # → bg.utils.checkpoint

model = nn.Linear(4, 2)
with torch.amp.autocast(device_type="webgpu", dtype=torch.float16):
    pred = model(x)
loss = ((pred - y) ** 2).mean()
loss.backward()
```

Anything unsupported raises `AttributeError` rather than silently faking success.

### Use a WGSL kernel directly

```ts
import { createDevice, kernels, tensor, matmulTiled } from "@unlocalhosted/browsergrad-kernels";

const device = await createDevice();
const A = tensor([2, 3], new Float32Array([1, 2, 3, 4, 5, 6]));
const B = tensor([3, 2], new Float32Array([7, 8, 9, 10, 11, 12]));
const C = await matmulTiled(device, A, B);   // tiled 16×16 GEMM
```

Pure-JS reference implementations live at the `/reference` subpath for environments without WebGPU.

## Browser testing

The kernels package has real-Chromium tests against an actual `GPUDevice`:

```sh
pnpm --filter @unlocalhosted/browsergrad-kernels test:browser
```

This launches Chromium via Playwright with WebGPU enabled and exercises the tiled GEMM, fused-elementwise codegen, residency contract, and the realizer-bridge end-to-end. On macOS the browser is headed (Metal driver only exposed when visible); on Linux CI set `BG_BROWSER_HEADLESS=1`.

The browser-test harness was added when the FEEDBACK loop demanded it — the NumPy mocks pass everything green but only a real GPU surfaces shader-level bugs (we found a deterministic FA-v2 kernel issue this way; tracked in [STATUS.md](./STATUS.md)).

## Using inside craftingattention

The lab platform consumes these packages directly:

1. **Pin the runtime version** in each lab's `manifest.json`:
   ```json
   {
     "id": "single-neuron-relu",
     "version": "1.0.0",
     "requires_browsergrad": "^0.8.0",
     "required_ops": ["MATMUL", "ADD", "WHERE"],
     "rubric_path": "rubric.py",
     "starter_path": "starter.py",
     "reference_path": "reference.py",
     "datasets": []
   }
   ```
   The runtime validates the manifest at boot via `parseManifest()` and refuses to run a lab whose pin doesn't satisfy the live runtime semver (`assertCompatibleRuntime` throws `LabRuntimeMismatch`).

2. **Author rubrics in Python** using the harness primitives:
   ```python
   import browsergrad_jit as bg
   # student's code runs first, producing `student_out`...

   bg.lab.assert_pytorch_match("forward_matches_reference",
                               student_out, reference_out, rtol=1e-4)
   bg.lab.assert_shape_match("output_shape_correct", student_out, (32, 10))
   bg.lab.assert_no_nan_inf("no_nan_in_gradient", w_grad)
   ```
   Each call routes through the runtime's `browsergrad` assertion module — craftingattention's UI receives them via the existing `onAssertion` callback.

3. **Wire the WebGPU bridge** if the lab needs GPU acceleration (any matmul-heavy workload):
   ```ts
   // In craftingattention's worker host
   import { createWebGpuRealizerBridge, createDevice } from "@unlocalhosted/browsergrad-kernels";

   const device = await createDevice();
   const bridge = createWebGpuRealizerBridge(device);
   pyodide.registerJsModule("_bg_webgpu_bridge", bridge);
   // Then in the lab's setup code:
   //   from js import _bg_webgpu_bridge
   //   bg.register_webgpu_bridge(_bg_webgpu_bridge)
   ```

4. **Export trained models for download** via `bg.onnx.export_inference`. Returns bytes — pipe to a `Blob` URL + `<a download>` in JS.

The craftingattention repo lives separately; the integration surface is small enough that no changes here are required to wire it up.

## Packages

| Package | What it does |
|---|---|
| [`browsergrad-runtime`](./packages/browsergrad-runtime) | Spawns Pyodide in a Web Worker, exposes `exec` / `fs` / structured assertion + artifact protocol, AbortSignal cancellation, lab manifest validator. |
| [`browsergrad-kernels`](./packages/browsergrad-kernels) | WGSL kernels for ML primitives — naive + tiled matmul, softmax, layernorm, attention, Flash Attention v2, elementwise activations. Each shipped with a pure-JS reference impl. Plus `createWebGpuRealizerBridge` for browsergrad-jit. |
| [`browsergrad-grad`](./packages/browsergrad-grad) | Eager tensor + reverse-mode autograd in Python. Stable. Used by curriculum content today. |
| [`browsergrad-jit`](./packages/browsergrad-jit) | Lazy-IR successor. PyTorch-shaped surface; fusion, symbolic backward, AMP, gradient checkpointing, functional transforms (`bg.func.{grad, vjp, vmap, functional_call}`), custom WGSL kernels, ONNX export, GPUBuffer-backed WGSL realizer seam. |

Each package has its own README and CHANGELOG with full API details.

## Documentation

- [STATUS.md](./STATUS.md) — current state, test counts, supported APIs, known issues, deliberate deferrals
- [ARCHITECTURE.md](./ARCHITECTURE.md) — how the packages compose; data flow; design principles
- [FEEDBACK.md](./FEEDBACK.md) — what end-to-end exercising the library surfaced; perf baselines; ranked improvement priorities
- [VISION.md](./VISION.md) — what this is for; why each package exists; what we won't build
- [PROGRESS.md](./PROGRESS.md) — PRD log
- [DEVELOPMENT.md](./DEVELOPMENT.md) — TDD methodology, quality gates
- [CONTRIBUTING.md](./CONTRIBUTING.md) — how to contribute
- [SECURITY.md](./SECURITY.md) — how to report vulnerabilities

## License

MIT. See [LICENSE](./LICENSE).
