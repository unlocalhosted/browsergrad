# Changelog

All notable changes to `@unlocalhosted/browsergrad-grad`.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-25

Initial release. Tensor + reverse-mode autograd, basic neural-network primitives,
SGD. Enough to train a tiny MLP end to end.

### Added

#### Host (TypeScript)

- `installGrad(target, options?)` — installs the `browsergrad_grad` Python
  package into any Pyodide-shaped target. Two install paths:
    - `fs.write` path: writes .py files to the virtual FS, adds the mount root
      to `sys.path`, smoke-tests by importing. Preferred when available.
    - exec-only fallback: a single `exec` call materializes the package
      contents from base64-embedded TypeScript constants. Works with raw
      Pyodide setups that don't expose a fs helper.
- `GradTarget` — duck-typed install interface (`exec({code}) → Promise`,
  optionally `fs.write(path, content)`).
- `GradInstallError` for install-time failures.
- Python source re-exported at the `./source` subpath for tools that want
  to install grad through their own bootstrap.

#### Python (browsergrad_grad)

- `Tensor(data, requires_grad=False)` — f32 NumPy-backed.
  - Properties: `shape`, `ndim`, `size`, `data`.
  - Conversions: `.numpy()`, `.tolist()`, `.item()`, `.detach()`, `.zero_grad()`.
  - Arithmetic ops: `+`, `-`, `*`, `/`, `-` (unary), `@`, `** float`.
  - Reductions: `.sum()`, `.mean()`.
  - `.backward(grad=None)` — reverse-mode autograd via topological sort.
- Constructors: `zeros`, `ones`, `randn` (with seed).
- `functional`: `relu`, `sigmoid`, `tanh`, `mse_loss`.
- `nn`: `Module` (base with auto-parameter discovery), `Linear`, `Sequential`.
- `optim`: `SGD` with momentum and weight-decay support.

#### Quality

- 14 vitest tests covering: public exports, Python module bundle contents,
  fs-path install behavior, exec-only fallback, custom mount roots, the
  `skipImportCheck` option, and error wrapping.
- TypeScript declarations + source maps for every emitted module.

### Deferred

- **Tensor-tensor broadcasting.** v0 supports same-shape ops + scalar broadcasting.
- **Higher-rank matmul.** v0 is 2D × 2D only.
- **Adam, AdamW, RMSprop.** Only SGD in v0.
- **CrossEntropyLoss, BCE, NLL.** Only MSE in v0.
- **GELU, softmax, more activations.** v0 is relu/sigmoid/tanh.
- **Conv/BatchNorm/Embedding/Dropout/RNN/LSTM/GRU.** All v0.2+.
- **WebGPU dispatch via `@unlocalhosted/browsergrad-kernels`.** v0 is NumPy.
  When integrated (v0.3 target), matmul / softmax / layernorm / attention
  will dispatch to GPU when a `KernelDevice` is provided.
- **Pyodide integration tests.** Python correctness tests need a real Pyodide
  runtime. Surface tests cover the host API; correctness tests come in v0.2
  via `@vitest/browser` + the runtime package.

### Known limitations

- The library is f32-only. NumPy operations occasionally promote to f64;
  every Tensor cast back via `astype(np.float32)`. Some accumulation drift
  is expected in long training loops.
- Binary ops with mismatched shapes throw rather than auto-broadcast.
  This is deliberate — broadcasting in backward requires summing across
  the broadcasted dims, which is easy to get wrong. v0.2 will add it.
- `Module._parameters` discovery happens on `__setattr__`. If you mutate a
  `requires_grad` flag after assignment, the module won't notice. Set
  `requires_grad` at construction time.

[0.1.0]: https://github.com/unlocalhosted/browsergrad/releases/tag/grad%40v0.1.0
