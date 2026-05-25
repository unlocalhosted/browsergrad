# browsergrad

Open-source library family for running Python and ML workloads in the browser.

```
@unlocalhosted/browsergrad-runtime   v0.1.1   Pyodide-in-Worker host: exec, fs, AbortSignal cancel, structured assertions/artifacts
@unlocalhosted/browsergrad-kernels   v0.1.0   WGSL kernel catalog (matmul, softmax, attention, ...) + JS reference impls
@unlocalhosted/browsergrad-grad      v0.4.3   Tensor + autograd, every standard NN layer, full train/eval mode system
```

Each package is **independently consumable** — they share an organization scope on npm but no runtime dependency. Take one or all.

## Status

| Package | Surface | Integration | What works |
|---|---|---|---|
| **browsergrad-runtime** | 11 ✅ | 23 ✅ | Pyodide worker, exec, fs, assertion/artifact protocol, cooperative cancel; Python bridge + JS message routing both verified |
| **browsergrad-kernels** | 26 ✅ (incl. JS reference correctness) | — (WebGPU needed) | 6 WGSL kernels + JS references: matmul, softmax, relu, gelu, layernorm, attention |
| **browsergrad-grad** | 23 ✅ | 89 ✅ | See below |

**Total: 172 tests green** across the workspace. Every gradient verified against finite differences or hand-derived oracles; every WGSL kernel verified against its JS counterpart; the entire pipeline verified end-to-end via a 4-class CNN classifier kitchen-sink test.

## What `browsergrad-grad` covers (v0.4.3)

Tensor + autograd:
- Element-wise ops with NumPy-style broadcasting (incl. in backward)
- Higher-rank matmul (batched), reductions with axis/keepdims, exp/log/reshape/transpose
- Reverse-mode autograd via topological sort
- `grad.no_grad()` context for inference

Neural network modules:
- Linear, Conv1d, Conv2d (im2col + matmul backend), Embedding
- MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
- BatchNorm1d (handles 2D & 3D), BatchNorm2d, LayerNorm
- Dropout, Dropout2d
- MultiHeadAttention (batch-first scaled dot-product, multi-head)
- Flatten, Sequential
- Activation modules: ReLU, LeakyReLU, Sigmoid, Tanh, GELU
- `Module.training` / `.train()` / `.eval()` mode system (recursive)

Functional:
- relu, leaky_relu, sigmoid, tanh, gelu
- softmax, log_softmax
- mse_loss, cross_entropy_loss (fused softmax+NLL), nll_loss

Optimizers: SGD (+ momentum, weight_decay), Adam, AdamW.

Tensor extras: argmax, item/int/float/bool scalar conversions, detach, numpy, tolist.

### End-to-end checks that run in CI

- MLP on linear regression converges to known coefficients
- 2-class image classifier (Conv → ReLU → MaxPool → Linear) reaches >95%
- 2-class image classifier with BatchNorm reaches >95% with mode switching
- Conv1d sequence classifier reaches >95%
- MLP with Dropout reaches >90% in both train and eval modes
- Transformer block (Embedding → MHA + LayerNorm + FFN) trains a copy task
- **Kitchen-sink 4-class CNN** combining Conv2d/BN2d/Dropout2d/MaxPool/Flatten/Linear/Dropout/Adam reaches >90% accuracy, eval-mode differs from train-mode, predictions via no_grad+argmax

Every gradient formula is verified against either finite differences or a hand-derived analytic oracle. No test compares the implementation against itself.

### What real bugs were caught by these tests

- **`bg.emit_json` / `bg.log` / `bg.emit_image` / `bg.assert_error` `NameError`** in the runtime's PY_PREAMBLE (helpers' closures looked up deleted globals at call time). Caught the first run of `tests-integration/python-bridge.test.ts`. Fixed by capturing helpers in keyword-only default args. Without this test, every production session would have crashed when user code called any of those functions.
- A test scaffolding bug (`clearNamespace` deleted its own iteration variable mid-loop) caught by the original grad integration suite.
- A mistaken assertion that Adam < SGD at small step counts; fixed to match the real behavior (Adam needs ~20 steps to warm up its bias correction).

## Design

Each package exposes **primitives, not workflows.** No baked-in concept of a "notebook," "lab," or "lesson." Consumers compose their own product on top.

- `browsergrad-runtime` doesn't know what tests are.
- `browsergrad-kernels` doesn't know what a tensor is — it dispatches WGSL with typed buffers.
- `browsergrad-grad` is the tensor library; it can be installed into *any* Pyodide-shaped target with `exec({code})`, not just our runtime.

This is deliberate. Packages can be used in isolation or together.

## Methodology

Every layer in `browsergrad-grad` was developed via strict TDD:

1. Write one behavior test against an **independent oracle** (NumPy result, hand-derived math identity, or finite-difference numerical check).
2. Watch it fail.
3. Write the minimum implementation to make it pass.
4. Never refactor while red.
5. Repeat.

Refactors (like the Conv2d im2col rewrite) are validated *only* by existing tests — no new tests written for them; the existing safety net catches drift.

## Repository layout

```
browsergrad/
├── packages/
│   ├── browsergrad-runtime/     Pyodide-in-Worker host (v0.1.1)
│   ├── browsergrad-kernels/     WGSL kernel catalog (v0.1.0)
│   └── browsergrad-grad/        most-developed; tensor + autograd + nn (v0.4.3)
├── package.json                 pnpm-workspace root
└── LICENSE                      MIT
```

## Quality gates

- `pnpm typecheck` — TypeScript strict mode on all 3 packages
- `pnpm test` — fast surface tests across all packages (~1s)
- `pnpm -F @unlocalhosted/browsergrad-grad test:integration` — full Pyodide-in-node integration suite (~15s; 89 tests across 9 files)
- `pnpm -F @unlocalhosted/browsergrad-runtime test:integration` — runtime Python-bridge + client-routing tests (~3s; 23 tests)

## License

MIT. See [LICENSE](./LICENSE).
