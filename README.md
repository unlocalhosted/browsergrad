# browsergrad

Open-source library family for running Python and ML workloads in the browser.

```
@unlocalhosted/browsergrad-runtime   v0.1.0   Pyodide-in-Worker host: exec, fs, AbortSignal cancel
@unlocalhosted/browsergrad-kernels   v0.1.0   WGSL kernel catalog (matmul, softmax, attention, ...)
@unlocalhosted/browsergrad-grad      v0.4.1   Tensor + autograd library, PyTorch-flavored, runs in Pyodide
```

Each package is **independently consumable** — they share an organization scope on npm but no runtime dependency. Take one or all.

## Status

| Package | Surface tests | Behavior tests | What works |
|---|---|---|---|
| **browsergrad-runtime** | 11 ✅ | — (browser env) | Pyodide worker, exec, fs, assertion/artifact protocol, cooperative cancel |
| **browsergrad-kernels** | 26 ✅ (incl. JS reference correctness) | — (WebGPU) | 6 WGSL kernels + matching JS references: matmul, softmax, relu, gelu, layernorm, attention |
| **browsergrad-grad** | 23 ✅ | 79 ✅ (real Python in Pyodide-in-node) | See below |

**Total: 139 tests green** across the workspace. Every grad gradient verified against finite differences or hand-derived oracles; every WGSL kernel verified against its pure-JS counterpart.

## What `browsergrad-grad` covers (v0.4.1)

Tensor + autograd:
- Element-wise ops with NumPy-style broadcasting (incl. in backward)
- Higher-rank matmul (batched), reductions with axis/keepdims, exp/log/reshape/transpose
- Reverse-mode autograd via topological sort

Neural network modules:
- Linear, Conv1d, Conv2d, Embedding
- MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
- BatchNorm1d (handles 2D & 3D), BatchNorm2d, LayerNorm
- Dropout, Dropout2d
- MultiHeadAttention (batch-first, multi-head SDP attention)
- Flatten, Sequential
- Activation modules: ReLU, LeakyReLU, Sigmoid, Tanh, GELU
- `Module.training` / `.train()` / `.eval()` mode system

Functional:
- relu, leaky_relu, sigmoid, tanh, gelu
- softmax, log_softmax
- mse_loss, cross_entropy_loss (fused softmax+NLL), nll_loss

Optimizers: SGD (+ momentum, weight_decay), Adam, AdamW.

### End-to-end checks that run in CI

- MLP on linear regression converges to known coefficients
- 2-class image classifier (Conv → ReLU → MaxPool → Linear) reaches >95%
- 2-class image classifier with BatchNorm reaches >95% with mode switching
- Conv1d sequence classifier reaches >95%
- MLP with Dropout reaches >90% in both train and eval modes
- Transformer block (Embedding → MHA → LayerNorm → FFN) trains a copy task to low loss

Every gradient formula is verified against either finite differences or a hand-derived analytic oracle. No test compares the implementation against itself.

## Design

Each package exposes **primitives, not workflows.** No baked-in concept of a "notebook," "lab," or "lesson." Consumers compose their own product on top.

- `browsergrad-runtime` doesn't know what tests are.
- `browsergrad-kernels` doesn't know what a tensor is — it dispatches WGSL with typed buffers.
- `browsergrad-grad` is the tensor library; it can be installed into *any* Pyodide-shaped target with `exec({code})`, not just our runtime.

This is deliberate. Packages can be used in isolation or together.

## Repository layout

```
browsergrad/
├── packages/
│   ├── browsergrad-runtime/     start here
│   ├── browsergrad-kernels/
│   └── browsergrad-grad/        most-developed; v0.4.1
├── package.json                 pnpm-workspace root
└── LICENSE                      MIT
```

## Quality gates

- `pnpm typecheck` — TypeScript strict mode on all 3 packages
- `pnpm test` — fast surface tests across all packages
- `pnpm -F @unlocalhosted/browsergrad-grad test:integration` — Pyodide-in-node integration tests (boots Pyodide, runs Python, verifies math). ~2-3s per file after the first numpy load.

## License

MIT. See [LICENSE](./LICENSE).
