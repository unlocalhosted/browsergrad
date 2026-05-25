# Changelog

All notable changes to `@unlocalhosted/browsergrad-grad`.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] — 2026-05-25

**`nn.BatchNorm2d`** plus the **`Module.training` train/eval mode system**.
Same TDD discipline (7 cycles). The training flag is a cross-cutting addition
to `Module` — every existing subclass keeps working unchanged.

### Added

- `Module.training: bool` — default `True`. Recursively flipped by
  `model.train(mode=True)` / `model.eval()`. Existing Modules (Linear,
  Conv2d, MaxPool/AvgPool, LayerNorm, Embedding, activations) ignore the
  flag — same behavior as before. BatchNorm reads it to branch.
- `nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)`
  - Train mode: forward uses batch stats, updates `running_mean` / `running_var`
    via the EMA `r ← (1 - momentum) * r + momentum * batch`.
  - Eval mode: forward uses `running_mean` / `running_var`, doesn't update them.
  - Affine: learnable `weight` (shape `(C,)`, init 1.0) and `bias` (init 0.0).
  - Buffers (`running_mean`, `running_var`) are plain numpy arrays — NOT in
    `.parameters()`; the optimizer doesn't touch them.
  - Backward uses the fused standard formula in train mode (same shape as
    LayerNorm's: `inv_std/N * (N*dxhat - sum(dxhat) - x_hat * sum(dxhat * x_hat))`),
    and the simpler `grad_x = grad_x_hat * inv_std` in eval mode.
  - All three gradients (input via finite-diff, β = `sum(grad_out)` hand-derived,
    γ via finite-diff) verified end-to-end.
- End-to-end test in `tests-integration/batchnorm.test.ts`:
  `Conv2d(1, 4, 3, padding=1) → BatchNorm2d(4) → ReLU → MaxPool2d(2) → reshape → Linear(36, 2)`
  trains to >95% accuracy in 60 Adam steps, with `model.eval()` used during
  the accuracy probes to exercise the eval-mode path.

### Still deferred (v0.4+)

- `nn.Dropout` and `nn.Dropout2d` (now feasible — both need the `training` flag).
- BatchNorm1d / BatchNorm3d.
- `GroupNorm`, `InstanceNorm`.

## [0.3.1] — 2026-05-25

**`nn.MaxPool2d` and `nn.AvgPool2d`**, TDD'd in 10 cycles. Same discipline
as Conv2d: one test, one minimum impl, public-interface-only oracles
(NumPy windowed reductions for forward, hand-derived argmax-routing
for MaxPool backward, finite differences for both gradients).

### Added

- `nn.MaxPool2d(kernel_size, stride=None, padding=0)` — square kernel,
  isotropic stride/padding. `stride` defaults to `kernel_size`. No
  learnable params. Forward records the argmax index per output cell;
  backward scatters `grad_out` to those positions.
- `nn.AvgPool2d(kernel_size, stride=None, padding=0)` — same shape contract.
  Backward distributes `grad_out / (K*K)` evenly across each window.
- End-to-end CNN integration test in `tests-integration/pool2d.test.ts`:
  `Conv2d(1, 2, 3, padding=1) → ReLU → MaxPool2d(2) → reshape → Linear(18, 2)`
  trains from random init to >95% accuracy in 60 Adam steps on a 64-sample
  top-vs-bottom synthetic classification task. The classifier passing
  proves *all* of Conv2d backward, MaxPool2d backward, reshape, Linear,
  cross-entropy, and Adam compose correctly.

### Still deferred (v0.4+)

- `padding > 0` for pooling. Currently rejected by the window-bounds math.
- Tuple kernel/stride/padding shapes for pooling (matches the same Conv2d
  scope rule).
- `AdaptivePool` and global pooling helpers.

## [0.3.0] — 2026-05-25

**`nn.Conv2d`**, developed strictly via TDD. One test, one cycle of minimum
implementation, repeat. Eleven cycles total:

| Cycle | Test (oracle) | Resulting impl change |
|---|---|---|
| 1 | tracer: 1×1 kernel → `w*x + b` (hand-derived) | Constructor + 1×1 forward |
| 2 | 3×3 vs numpy triple-loop reference | General loop forward |
| 3-5 | multi out-ch, multi in-ch, batch (already passed) | regression coverage |
| 6 | `stride=2` → output at every-other position | `(H-K)//S+1` + offset math |
| 7 | `padding=1` preserves spatial dims | `np.pad` input |
| 8 | `d/d(bias) = grad_out.sum((0,2,3))` (hand-derived) | bias gradient |
| 9 | `d/d(weight)` matches finite differences | weight gradient (accumulating loop) |
| 10 | `d/d(input)` matches finite differences | input gradient (scatter-add) |
| 11 | end-to-end: trains to recover an edge-detection kernel | (no new code — full pipeline check) |

### Added

- `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)`
  - Input shape: `(N, C_in, H, W)` → output: `(N, C_out, H_out, W_out)` where
    `H_out = (H + 2*padding - kernel_size) // stride + 1`.
  - Weight shape `(C_out, C_in, K, K)`, Kaiming-uniform init.
  - Forward: naive 4-deep nested loop with NumPy slicing inside; correct first,
    optimizable later. Readability over speed for v0.3 — the code is intentionally
    lesson-material.
  - Backward: closure captures `x_padded` and `weight.data` at forward time; one
    scatter-add pass produces `grad_w` and `grad_x_padded`; bias gradient is
    `grad_out.sum((0, 2, 3))`. All three gradients are verified end-to-end
    against finite differences (`eps=1e-3`, `tol=5e-3`).
- 11 new integration tests in `tests-integration/conv2d.test.ts`.

### Still deferred (v0.4+)

- Tuple kernel/stride/padding shapes (e.g. `kernel_size=(3, 5)`).
- Dilation.
- Groups / depthwise convolution.
- Transposed convolution (`ConvTranspose2d`).
- `Conv1d`, `Conv3d`.
- **im2col + matmul optimization.** Refactor candidate: identical interface,
  10–100× faster, sacrifices the readable nested-loop pedagogy.
- WebGPU dispatch via `@unlocalhosted/browsergrad-kernels`. Will need a
  matmul-based Conv2d to actually benefit from GPU.

## [0.2.0] — 2026-05-25

Broadcasting, transformer parts, Adam, **and the first real Python-execution
tests for the autograd**. Same public API as 0.1.0 — broadcasting just starts
working in every binary op. Old code continues to type-check and run, only the
`_check_shapes`-raising error path goes away.

### Test coverage (new in this release)

- **`tests-integration/`** boots Pyodide in node, installs the library via the
  public `installGrad` API, and runs real Python that exercises:
    - Autograd basics: `d(x²)/dx = 2x`, `d(mean(x²))/dx`, `d(x³)/dx`, `d(exp(x))`, `d(log(x))`
    - Broadcasting backward: bias-into-batch sums correctly, scalar broadcasts roundtrip
    - 2D and batched 3D matmul gradients
    - Functional ops: relu masking, softmax sums to 1, cross-entropy gradient
      matches the `(softmax - one_hot)/N` formula, MSE gradient
    - nn modules: Linear shapes, LayerNorm normalizes to mean≈0 var≈1,
      Embedding scatter-add accumulates duplicate indices correctly
    - End-to-end: SGD converges to y=3x+1, Adam converges, MLP trains
  Run via `pnpm test:integration`. Surface tests in `tests/` still run via
  `pnpm test` for fast feedback.

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
