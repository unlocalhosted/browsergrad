# Changelog

All notable changes to `@unlocalhosted/browsergrad-grad`.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] — 2026-06-02

Dogfood pass on the published 0.5.0 tarball surfaced two issues. Both fixed.

### Fixed

- **Raw-Node ESM import broke.** `dist/install.js` and
  `dist/python/index.js` did `import pkg from "./package.json"` without
  the `with { type: "json" }` attribute required by Node 20+ ESM. The
  package worked under Vite/webpack/vitest (transformed) but failed in
  raw Node, edge runtimes, and any SSR framework hand-rolling Node ESM
  (`TypeError: Module ".../package.json" needs an import attribute of
  "type: json"`). Both source files now use the required attribute.
- **PyTorch ergonomic gaps.** `grad.nn.functional` was unreachable as
  an attribute; `from browsergrad_grad.nn import functional` failed
  (`'browsergrad_grad.nn' is not a package`); `F.cross_entropy` didn't
  exist (only the native `cross_entropy_loss`). All three resolved:
  `grad.F` is the standard shorthand, `grad.nn.functional` is wired as
  an attribute (and registered in `sys.modules` so the `import`
  statement also works), and `cross_entropy`/`nll`/`bce_with_logits`
  are added as PyTorch-name aliases alongside the native `_loss` names.
  Code copied from PyTorch tutorials now runs without
  `install_torch_alias()`.

## [0.5.0] — 2026-05-26

**PyTorch parity push (Piles A, B, C) + the PyTorch-conformance suite.**

Pile A (possible-just-unbuilt) — completed:
- Tensor indexing (`x[mask]`, slices, fancy) + comparison ops returning
  bool-dtype tensors (eq/ne/lt/gt/le/ge).
- Multi-dtype: `Tensor(data, dtype=...)`, `.dtype` property, `.long() / .int() /
  .float() / .double() / .bool() / .to(dtype)`, PyTorch aliases (long, double, …).
- `torch.utils.data.{Dataset, DataLoader, TensorDataset}` — single-process
  loader with shuffle/drop_last/tuple-collate; `num_workers > 0` raises
  with a Pyodide-specific reason.
- `state_dict` / `load_state_dict` / `grad.save` / `grad.load` (+ torch.save/load).
- `nn.init.{zeros_, ones_, uniform_, normal_, constant_, kaiming_uniform_,
  xavier_uniform_}` registered on `sys.modules` so `import
  browsergrad_grad.nn.init` works.
- Tensor math: `abs, sign, sqrt, pow, clamp/clip, topk, where` (top-level),
  shape ops `expand, repeat, flip` (rest deferred).
- `F.pad` (constant), `F.interpolate` (nearest + bilinear),
  `F.normalize` (L2), `F.cosine_similarity`,
  `F.scaled_dot_product_attention` (with bool/float mask + is_causal).
- More losses: `BCELoss`, `L1Loss`, `SmoothL1Loss`, `KLDivLoss` (with
  mean/batchmean/sum/none reductions, shared reduction helper).
- More optimizers: `RMSprop`, `Adagrad`, `Adadelta`.
- More schedulers: `MultiStepLR`, `ExponentialLR`, `ReduceLROnPlateau`,
  `OneCycleLR`. Exposed under `optim.lr_scheduler`.
- `grad.einsum` with autograd backward (1- and 2-operand).
- `nn.RNN / nn.LSTM / nn.GRU` (single-layer, batch_first) with BPTT
  through the autograd graph.
- `nn.GroupNorm`, `nn.InstanceNorm2d`, `nn.BatchNorm3d` (forward; backward
  through affine + scale).
- Module `register_forward_hook` / `remove_forward_hook`.

Pile B (possible but limited):
- `torch.amp.autocast` — no-op context manager.
- `torch.linalg.{norm, inv, det, svd, eigh, solve, pinv}` — wrap numpy.linalg.
- `nn.Module.to(device)` — no-op shim returning self.

Pile C (physically impossible in browser; LOUD failure with reason):
- `torch.compile`, `torch.fx.symbolic_trace`, `torch.jit.{script, trace}`,
  `torch.cuda.{current_device}`, `torch.distributed.{init_process_group,
  all_reduce}`, `torch.onnx.export`, `torch.quantization.quantize` — all
  raise `NotImplementedError` with an architectural reason.
- `torch.cuda.is_available()` returns `False` (legitimate).
- `torch.cuda.device_count()` returns 0.
- `torch.distributed.is_initialized()` returns False.

★ PyTorch-conformance fixture suite:
- `scripts/gen_pytorch_fixtures.py` runs real torch and serializes
  forward + backward fixtures to `tests-integration/fixtures/`.
- `tests-integration/pytorch_conformance.test.ts` loads the fixtures
  and verifies browsergrad matches real PyTorch within 1e-4 on Linear,
  CrossEntropy, LayerNorm, Softmax, ReLU.

Quality:
- `Tensor.__array__` so `np.asarray(tensor)` just works.
- `installViaExec` now mkdirs parent directories per file (lets us ship
  nested packages like `browsergrad_grad.utils.data`).
- Shared `_reduce_loss` helper across L1/BCE/SmoothL1/KLDiv.
- Friendlier dtype-spec error: lists supported aliases.

27 integration test files, 232 tests, all green. Source-of-truth oracle
is real PyTorch via the new conformance suite.

## [0.4.7] — 2026-05-26

**PyTorch-compat completeness pass.** Audit identified ~15 idioms that real
deep-ml-style PyTorch labs use but the v0.4.6 shim didn't cover. All now
work — `tensor.unsqueeze/squeeze/permute/size/to/cpu/cuda`, top-level
`torch.from_numpy/manual_seed/matmul/mm/bmm/exp/log/sum/mean/argmax`, loss
Modules (`nn.CrossEntropyLoss`, `nn.MSELoss`, `nn.NLLLoss`,
`nn.BCEWithLogitsLoss`), `nn.MultiheadAttention` lowercase alias,
`F.one_hot`, `F.dropout`, `F.bce_with_logits_loss`.

17 new integration tests cover every gap individually plus a realistic
end-to-end PyTorch lab using `nn.CrossEntropyLoss`, `torch.from_numpy`,
`torch.argmax` under `with torch.no_grad():`, and `model.train()/eval()`
transitions. Trains to >95% accuracy in 60 Adam steps.

## [0.4.6] — 2026-05-26

`install_torch_alias()` — vanilla PyTorch code runs against browsergrad_grad.

### Added

- **`grad.install_torch_alias()`** — registers a `torch` namespace shim
  into `sys.modules` so user code can `import torch`, `import torch.nn as nn`,
  `import torch.nn.functional as F`, `import torch.optim as optim` and have
  the calls run against browsergrad_grad transparently.
- PyTorch-name aliases inside the shim where our internal names differ:
  `F.cross_entropy` → `cross_entropy_loss`, `F.nll` → `nll_loss`.
- New Python module: `browsergrad_grad.torch_compat`. Exported from the
  package's `__init__` as `install_torch_alias`.

### Why this addition

Use case: deep-ml-style problems where the user writes vanilla PyTorch
code (`import torch.nn as nn` etc.) and expects it to just run. Without
the alias, a problem author would need to write `import browsergrad_grad
as torch` everywhere, which doesn't match how PyTorch problems are
typically authored or shared. The alias lets a platform turn that on
once per session.

### Verified

- 8 tests cover namespace plumbing (torch / torch.nn / torch.nn.functional /
  torch.optim) plus an end-to-end PyTorch-style classifier that trains a
  2-layer MLP to >95% accuracy on a linearly separable problem — using
  only `torch.*` calls, no `browsergrad_grad` references in user code.

### Limitations

The shim covers only the subset of torch's API that browsergrad_grad
implements. Notably absent and not faked:
- `torch.cuda.*` (no GPU device concept in v0)
- `torch.compile`, `torch.fx`, `torch.jit`
- dtype objects beyond the f32 we use internally (the shim exposes
  `torch.float32` etc. as strings, which work for the `dtype=` kwarg
  paths we support, but they're not full dtype objects)

## [0.4.5] — 2026-05-25

LR schedulers — closes the last commonly-expected gap in `optim`.

### Added

- `optim.StepLR(optimizer, step_size, gamma=0.1)` — decay lr by `gamma`
  every `step_size` scheduler steps. Matches `torch.optim.lr_scheduler.StepLR`.
- `optim.CosineAnnealingLR(optimizer, T_max, eta_min=0.0)` — cosine
  schedule from `optimizer.lr` to `eta_min` over `T_max` steps. Matches
  the PyTorch formula `eta_min + 0.5 * (base - eta_min) * (1 + cos(t/T_max·π))`.
  Verified at sampled epochs against the closed-form.
- `optim._LRScheduler` base class — used by both schedulers; subclasses
  override `_compute_lr(step) → float`. Internal but exported so users
  can write their own.

### Verified

- 5 tests cover: StepLR's per-step decay pattern across 10 steps, StepLR
  on Adam (any Optimizer subclass), CosineAnnealingLR matching the
  documented formula at every t ∈ [0, T_max], CosineAnnealingLR reaching
  exactly `eta_min` at t = T_max, and the integration check — scheduler
  step updates the lr that the optimizer's next step uses.

## [0.4.4] — 2026-05-25

Compositional multi-tensor ops — `cat` and `stack`.

### Added

- `grad.cat(tensors, dim=0)` — concatenate along an existing axis. All
  inputs must agree on every dim except `dim`. Backward splits the
  gradient along `dim` (`np.split` with cumulative-size cuts) and
  distributes one section to each parent.
- `grad.stack(tensors, dim=0)` — concatenate along a *new* axis inserted
  at `dim`. All inputs must have identical shape. Backward indexes the
  gradient along that new axis to recover each parent's slice.

### Verified

- 6 tests cover: cat along axis 0 and 1, cat backward gradient
  distribution; stack along axis 0 and inner axes; stack backward; plus
  a compositional check — a two-branch `Linear(cat([h1, h2]))` model
  trains end-to-end (confirming gradient flows through `cat`).

## [0.4.3] — 2026-05-25

Inference ergonomics: `no_grad()` context + `Tensor.argmax` + scalar conversions.

### Added

- **`grad.no_grad()`** context manager — disables autograd graph building
  inside the `with` block. Matches `torch.no_grad()`. Required to keep
  inference memory-flat for any model (otherwise every eval forward keeps
  the whole computation graph live).
- **`Tensor.argmax(dim=None)`** — returns indices of the max along a dim.
  Mostly used for classification accuracy probes; non-differentiable
  (no `_ctx`). Result is a float-backed tensor; users call `.tolist()` or
  use them as numpy-int directly.
- **`Tensor.__int__`, `Tensor.__float__`, `Tensor.__bool__`** — make
  Python's `int(tensor)`, `float(tensor)`, `bool(tensor)` work on scalar
  (0-d or size-1) tensors. Matches PyTorch.

### Verified

- 5 tests check no_grad behavior: ops inside the block don't build graphs,
  ops outside still do, the block restores prior state, backward through
  pre-block tensors still works.
- 5 tests check argmax: global (no axis), axis=0 (per-column), axis=-1
  (per-row), result is int-coercible, and integrates with a Linear classifier.
- 88 integration tests total green across 9 files (was 79 in 8 files).

## [0.4.2] — 2026-05-25

Pure refactor — same surface, no behavior change, big perf win.

### Changed

- **`nn.Conv2d` now uses im2col + matmul** in both forward and backward
  instead of the v0.3.0 naive 4-deep nested loops.
  Forward gathers each K×K window into a column matrix once (two outer
  loops) then does a single batched matmul against the flattened weight.
  Backward decomposes symmetrically: grad_weight via `grad_out_flat @ cols.T`
  (summed over batch), grad_input via `weight_flat.T @ grad_out_flat`
  scattered back via col2im.
  All 11 Conv2d tests (forward correctness against numpy, all 3 gradients
  vs finite differences, end-to-end training) pass unchanged — TDD
  safety net working as designed. Same numerical result to f32 tolerance.

## [0.4.1] — 2026-05-25

Round out the standard layer surface for sequence models and CNN→FFN transitions.

### Added
- `nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)` — input shape `(N, C_in, L)`, naive nested-loop forward with capture-and-scatter backward (same pattern as Conv2d). All three gradients verified against finite differences.
- `nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)` — accepts both 2D `(N, C)` and 3D `(N, C, L)` inputs; reduce axes computed from rank. Same fused-backward formula as BatchNorm2d, parameterized.
- `nn.Flatten(start_dim=1, end_dim=-1)` — composes on `Tensor.reshape` so backward is automatic.
- End-to-end Conv1d sequence classifier test (>95% accuracy in 80 Adam steps).

## [0.4.0] — 2026-05-25

The headline release for attention.

### Added
- `nn.MultiHeadAttention(embed_dim, num_heads, bias=True)` — PyTorch-conformant scaled dot-product attention, batch-first convention `(N, S, D)`. Built entirely from autograd primitives (Linear + matmul + softmax + transpose + reshape) — backward is automatic and correct by construction.
- `nn.AdaptiveAvgPool2d(output_size)` — adaptive pooling matching the PyTorch bin-boundary formula. Handles non-evenly-divisible cases (e.g. 7×7 → 3×3). Backward distributes gradient over each bin's area.
- End-to-end transformer-block test (Embedding → MHA → LayerNorm → FFN) trains a copy task to low loss.

## [0.3.3] — 2026-05-25

Dropout family. Uses the `Module.training` flag added in v0.3.2.

### Added
- `nn.Dropout(p=0.5)` — PyTorch-conformant inverted dropout. Train: keep with prob `1-p`, scale kept by `1/(1-p)` so `E[y] = x`. Eval: identity.
- `nn.Dropout2d(p=0.5)` — channel-wise dropout for `(N, C, H, W)`; whole channels are zeroed/kept together.
- 9 integration tests cover: tracer, eval identity, statistical drop rate, scale-preservation, backward (zero pattern matches kept pattern, scale matches `1/(1-p)`), end-to-end MLP-with-dropout trains.

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
