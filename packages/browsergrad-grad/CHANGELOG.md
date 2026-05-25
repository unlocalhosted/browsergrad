# Changelog

All notable changes to `@unlocalhosted/browsergrad-grad`.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-25

Broadcasting, transformer parts, Adam. Same public API as 0.1.0 — broadcasting
just starts working in every binary op. Old code continues to type-check and
run, only the `_check_shapes`-raising error path goes away.

### Added

#### Tensor

- **Tensor-tensor broadcasting in every binary op.** NumPy-style
  right-to-left dim matching, with `_unbroadcast` in each backward to
  sum-collapse gradients back to the parent's original shape.
- **Higher-rank matmul.** `a @ b` now accepts any rank ≥ 2 with broadcasted
  batch dimensions, matching `numpy.matmul`. Backward swaps the last two
  axes (`np.swapaxes(-1, -2)`) instead of `.T`.
- **New methods**: `.exp()`, `.log()`, `.reshape(*shape)`, `.view(*shape)`
  (alias), `.transpose(dim0, dim1)`, `.T` (2D only).
- **Axis-aware reductions.** `.sum(axis=, keepdims=)` and `.mean(axis=,
  keepdims=)` — passes through to NumPy with proper gradient broadcast
  back to the input shape.

#### functional

- `leaky_relu(x, negative_slope=0.01)`
- `gelu(x)` — tanh-approximation (GPT-2 / BERT variant), with its full
  analytic derivative (not the cheap approximation).
- `softmax(x, dim=-1)` — stable, with the standard
  `s_i * (g_i - sum_j(s_j * g_j))` backward.
- `log_softmax(x, dim=-1)` — numerically stable.
- `cross_entropy_loss(logits, targets)` — fused softmax + NLL. Targets
  are class indices (numpy int array). Gradient simplifies to
  `(softmax - one_hot) / N`, bypassing the need for differentiable indexing.
- `nll_loss(log_probs, targets)` — for when you already have log-probs.

#### nn

- **`LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)`** —
  with the standard fused backward formula (avoids accumulating numerical
  noise through naive `mean → var → sqrt → divide` chains).
- **`Embedding(num_embeddings, embedding_dim)`** — int index lookup with
  scatter-add gradient on the weight table.
- **Activation modules**: `ReLU`, `LeakyReLU(negative_slope=)`, `Sigmoid`,
  `Tanh`, `GELU`. Slot into `Sequential` directly.
- `Linear` simplified — drops the v0.1 bias-broadcasting workaround
  (`broadcast_to_rows`) now that the Tensor supports real broadcasting.

#### optim

- **`Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)`**
  — Kingma & Ba 2014. Matches `torch.optim.Adam`'s update with
  `amsgrad=False`.
- **`AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)`**
  — decoupled weight decay (Loshchilov & Hutter 2017). The standard
  optimizer for transformer pre-training.

### Changed

- `Linear.forward` now uses real broadcasting for bias addition. The
  `Tensor.broadcast_to_rows` helper added in v0.1 to work around the lack of
  broadcasting has been removed — Sequential nets that used it transparently
  continue to work, but anyone calling `tensor.broadcast_to_rows` directly
  will need to use `+` instead.
- `__version__` bumped to `"0.2.0"`. The installer's smoke-check expects
  this version exactly.

### Still deferred (v0.3+)

- **Conv2d / Conv1d.** im2col + the gradient gymnastics around stride /
  padding / dilation. Highest-priority v0.3 target since the curriculum's
  CNN labs depend on it.
- **BatchNorm / GroupNorm.** Mostly needed for older CNN architectures —
  modern transformer work uses LayerNorm exclusively.
- **Dropout, RNN, LSTM, GRU.** Less central given the curriculum's transformer
  emphasis; will land when needed.
- **WebGPU dispatch via `@unlocalhosted/browsergrad-kernels`.** All v0.2
  ops still run on CPU through NumPy. A future minor will add an optional
  `device` arg that dispatches matmul / softmax / layernorm / attention
  to a `KernelDevice` when provided.
- **Pyodide integration tests.** Surface tests verify host wiring; Python
  correctness tests await the browser test setup.

### Known limitations

- Integer dtypes aren't a first-class concept. Embedding's `indices` is a
  numpy int array, not a `Tensor`. cross_entropy_loss's `targets` likewise.
  This is on purpose: integer tensors don't participate in autograd, so
  conflating them with the f32 Tensor would create surprises.
- `LayerNorm` requires `normalized_shape` to match the last D dims of the
  input exactly; we don't support PyTorch's "elementwise_affine over a
  prefix of dims" generalization (it's almost never used in practice).

## [0.1.0] — 2026-05-25

Initial release. Tensor + reverse-mode autograd, basic neural-network
primitives, SGD. See git history (`fd3ddde` initial, `f7d0df9` grad).

[0.2.0]: https://github.com/unlocalhosted/browsergrad/releases/tag/grad%40v0.2.0
[0.1.0]: https://github.com/unlocalhosted/browsergrad/releases/tag/grad%40v0.1.0
