# Status

Living document. Reflects the current state of each package, what's tested, and what's known-deferred.

## Package versions

| Package | Version | Surface tests | Integration tests |
|---|---|---|---|
| `@unlocalhosted/browsergrad-runtime` | `0.1.1` | 11 | 23 (Pyodide-in-node) |
| `@unlocalhosted/browsergrad-kernels` | `0.1.0` | 26 (incl. JS-reference numerical checks) | — (WebGPU required) |
| `@unlocalhosted/browsergrad-grad` | `0.4.6` | 25 | 115 (Pyodide-in-node) |

**Total: 200 tests green**, every gradient verified against finite differences or hand-derived oracles.

## Surface inventory — `browsergrad-grad`

### Tensor + autograd

- `Tensor(data, requires_grad=False)` — f32 NumPy-backed
- Element-wise ops with NumPy-style broadcasting (forward + backward)
- Higher-rank matmul (batched)
- Reductions with `axis`/`keepdims` for `sum` and `mean`
- Element-wise unary: `exp`, `log`
- Shape ops: `reshape`, `view`, `transpose(d0, d1)`, `.T`
- Multi-tensor ops: `grad.cat([...], dim)`, `grad.stack([...], dim)`
- Reverse-mode autograd via topological sort
- `grad.no_grad()` context for inference
- Scalar conversion: `t.item()`, `int(t)`, `float(t)`, `bool(t)`
- Constructors: `zeros`, `ones`, `randn(seed=)`
- `Tensor.argmax(dim=None)`
- `grad.install_torch_alias()` — registers `torch`, `torch.nn`, `torch.nn.functional`,
  `torch.optim` in `sys.modules` so vanilla PyTorch user code runs unmodified

### Neural network modules (`browsergrad_grad.nn`)

- `Module` base — auto-tracks parameters, recursive `train()`/`eval()`/`training` flag
- `Linear`
- `Conv1d`, `Conv2d` (im2col + matmul backend)
- `Embedding`
- `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`
- `BatchNorm1d` (accepts `(N, C)` and `(N, C, L)`), `BatchNorm2d`
- `LayerNorm`
- `Dropout`, `Dropout2d`
- `MultiHeadAttention` (batch-first `(N, S, D)`)
- `Flatten`, `Sequential`
- Activation modules: `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `GELU`

### Functional (`browsergrad_grad.functional`)

- Activations: `relu`, `leaky_relu`, `sigmoid`, `tanh`, `gelu`
- `softmax`, `log_softmax`
- Losses: `mse_loss`, `cross_entropy_loss` (fused softmax + NLL), `nll_loss`

### Optimization (`browsergrad_grad.optim`)

- Optimizers: `SGD` (with momentum, weight_decay), `Adam`, `AdamW`
- LR schedulers: `StepLR`, `CosineAnnealingLR`

## Surface inventory — `browsergrad-kernels`

WGSL compute-shader catalog. Every kernel ships with a pure-JS reference impl that doubles as a conformance oracle and a CPU fallback:

- `matmul` (naive `[M,K]·[K,N]`)
- `softmax` (stable, along last axis)
- `relu`, `gelu` (elementwise)
- `layernorm` (along last axis, optional gamma/beta)
- `attention` (single-head scaled-dot-product; `[S, D]` shapes)

Pipeline cache per device, buffer cleanup on each invocation.

## Surface inventory — `browsergrad-runtime`

- `createSession({ pyodideIndexURL, packages, worker?, disableInterruptBuffer? })`
- `session.fs.write/read` (MEMFS via Emscripten)
- `session.exec({ code, timeoutMs, signal, onStdout, onStderr, onAssertion, onArtifact })`
- `session.interrupt()` + `session.canInterrupt` (SAB + cross-origin isolation)
- `session.clearNamespace()`, `session.dispose()`
- Structured assertion + artifact protocols emitted from Python via `import browsergrad as bg`

## End-to-end checks

These run as part of the integration suite — they validate the full pipeline composes correctly:

- MLP on linear regression converges to known coefficients
- 2-class CNN (Conv → ReLU → MaxPool → Linear) reaches >95%
- 2-class CNN with BatchNorm reaches >95% across `model.train()`/`model.eval()` switches
- Conv1d sequence classifier reaches >95%
- MLP with Dropout reaches >90% in both train and eval modes
- Transformer block (Embedding → MHA → LayerNorm → FFN) trains a copy task to low loss
- 4-class CNN "kitchen sink" (Conv → BN → Dropout2d → Pool → Flatten → Linear → Dropout → Linear) reaches >95%
- Two-branch model with `grad.cat([h1, h2])` trains to low loss
- Vanilla-PyTorch MLP (using `import torch`) reaches >95%

Every gradient formula is verified against either finite differences or a hand-derived analytic oracle. No test compares the implementation against itself.

## Known deferred / out of scope

| Item | Reason |
|---|---|
| Runtime browser-context tests | Web Workers in node behave differently; covered by Pyodide-in-node integration tests for the protocol; full Worker isolation needs a real browser. |
| Kernels WGSL conformance against the JS reference | Requires real WebGPU; node WebGPU bindings (`@google/dawn` etc.) are not stable enough to depend on. |
| WebGPU dispatch from `browsergrad-grad` | Depends on kernels conformance being testable. |
| ConvTranspose / Conv3d / dilated conv / groups | Out of v0 scope. |
| RNN / LSTM / GRU | Largely superseded by transformers for the curriculum-shaped use cases. |
| `torch.cuda.*`, `torch.compile`, `torch.fx` | Out of scope for `install_torch_alias`. Raises `AttributeError`. |

When these become blocking for a real consumer, file an issue.
