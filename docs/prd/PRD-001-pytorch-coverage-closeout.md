# PRD-001: PyTorch Coverage Closeout (P0.1 + Cheap Fills)

**Status**: Draft
**Owner**: [Assignee TBD]
**Parent roadmap doc**: [PRD.md](../../PRD.md) §7 P0.1
**Progress tracker**: [PROGRESS.md](../../PROGRESS.md) Pile A rows 13, 14, 16, 17, 18
**Estimated duration**: 2 weeks
**Target version**: v0.6.0
**Last updated**: 2026-05-26

---

## TL;DR

Eleven specific PyTorch ops and two container types are blocking concrete educational labs from running end-to-end in browsergrad. This PRD specifies exactly what to implement, where each piece of code goes in the chunked Python source, what the acceptance criteria are, and how to verify each op against a real PyTorch reference fixture within 1e-4 tolerance. Everything here is a real implementation — no stubs except for `torch.utils.checkpoint.checkpoint_sequential`, which gets an explicit-error stub pointing to PRD-009. Shipping this PRD makes nanoGPT's `model.py` importable without modification and closes the last two gaps in the deep-ml problem catalog.

---

## Background

### Why each gap is load-bearing

**nanoGPT (`model.py`)**

Karpathy's nanoGPT is the canonical "train a real language model from scratch" reference. As of May 2026, browsergrad runs the attention, normalization, dropout, embedding, and optimizer paths. The four remaining blockers are:

- `nn.ModuleDict` (line 102 of `model.py`): the GPT model builds its entire submodule tree via `self.transformer = nn.ModuleDict(dict(...))`. Without this, the class body raises `AttributeError` on instantiation.
- `nn.ModuleList` (line 105): transformer blocks are stored as `nn.ModuleList([Block(config) for _ in range(config.n_layer)])`. Without this, indexing into the block list silently stops tracking parameters.
- `torch.tril` (line 48): `CausalSelfAttention.__init__` calls `torch.tril(torch.ones(...))` to pre-compute the causal mask buffer. Without this, any attempt to instantiate CausalSelfAttention raises `AttributeError`.
- `torch.topk` as a module-level function accepting `dim` argument (line 343): the generation loop calls `torch.topk(logits, min(top_k, logits.size(-1)))` on a 2-D logit tensor, returning `(values, _)`. The current `Tensor.topk` method exists but only handles 1-D tensors (`tensor.py:323–330`); `torch.topk` as a standalone function with dim support is missing.
- `torch.multinomial` (line 347): `idx_next = torch.multinomial(probs, num_samples=1)` samples the next token from a probability distribution. Entirely absent.

Source: [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py), lines 48, 102, 105, 343, 347.

**deep-ml problem catalog**

The deep-ml curriculum (`Open-Deep-ML/DML-OpenProblem`) includes activation function problems that explicitly list `softsign` in their expected API surface. browsergrad's `functional.py` currently has relu, leaky_relu, sigmoid, tanh, and gelu, but not softsign. This is the last functional gap to deep-ml parity (the other was gradient checkpointing, addressed via the stub in this PRD).

**Universal beginner idiom (in-place ops)**

Beginner PyTorch code frequently uses:
```python
x = torch.randn(3, 3)
x.requires_grad_(True)   # common idiom from every beginner tutorial
```
and the zeroing / accumulation patterns:
```python
w.zero_()
w.add_(delta)
```
PyTorch raises `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation` when you call any mutating method on a leaf tensor that has `requires_grad=True`. The `requires_grad_()` setter must work; the `zero_()`, `add_()`, `mul_()` methods must error correctly on such leaves. Currently none of these three in-place mutation methods exist on `Tensor`, and `requires_grad_()` does not exist either, causing `AttributeError` when beginner code follows the standard idiom.

Source: [PyTorch autograd docs](https://docs.pytorch.org/docs/2.12/notes/autograd.html), [PyTorch forum discussion](https://discuss.pytorch.org/t/write-to-data-instead-runtimeerror-a-leaf-variable-that-requires-grad-has-been-used-in-an-in-place-operation/41049).

**Module backward hook**

`register_backward_hook` is the standard API for inspecting gradient flow through `nn.Module` instances. The current `module.py` has `_forward_hooks` and `register_forward_hook` (line 35–41) but no backward hook infrastructure. Any lab that teaches gradient-based debugging hits `AttributeError` when calling `module.register_backward_hook(fn)`.

**Conv3d**

Conv3d is the natural generalization of Conv2d's im2col approach to volumetric input `(N, C, D, H, W)`. It blocks any video or 3D medical-imaging lab. The existing Conv2d implementation at `nn_chunks/conv.py:1–115` has the exact im2col kernel that can be extended to three spatial dimensions. PROGRESS.md row 16 marks Conv3d as "deferred" — this PRD re-activates it as a cheap fill because the structural work is already done.

**InstanceNorm1d, InstanceNorm3d**

`nn_chunks/norm.py` already has `InstanceNorm2d` (lines 356–400). Extending to 1d `(N, C, L)` and 3d `(N, C, D, H, W)` is a parameterization of the same per-instance normalization logic; only the reduction axes differ. PROGRESS.md row 17 marks these as pending.

---

## User Stories

**Story 1 — The nanoGPT student.** A student follows Karpathy's "Let's build GPT" walkthrough. They copy `model.py` verbatim, swap `import torch` for `import browsergrad_grad as torch`, and run it in a browser-based lab. Without this PRD, instantiation of `GPT` crashes at `nn.ModuleDict` (line 102). With this PRD, the model instantiates, a forward pass runs, and a short training loop on synthetic data converges — all in Pyodide.

**Story 2 — The deep-ml activation problem grinder.** A student works through every activation function problem in the deep-ml catalog. They reach the softsign problem. Without this PRD, `F.softsign` raises `AttributeError`. With this PRD, the reference implementation is there for the student to verify against.

**Story 3 — The gradient debugging lab.** A course author writes a lab that teaches students to inspect gradient magnitudes through layers using backward hooks. They write:
```python
def hook(module, grad_input, grad_output):
    print(f"grad_output norm: {grad_output[0].data.norm()}")
model.layers[0].register_backward_hook(hook)
```
Without this PRD, `register_backward_hook` raises `AttributeError`. With this PRD, the hook fires after backward and the lab works.

---

## Goals and Non-Goals

### Goals

1. Implement all eleven ops listed in this PRD as real, numerically correct implementations verified against PyTorch fixtures within 1e-4.
2. Make nanoGPT's `model.py` importable and runnable (forward + one training step) in Pyodide without source modification.
3. Achieve 100% coverage of the deep-ml sampled problem catalog (closing the softsign and checkpoint-stub gaps).
4. Add PyTorch-conformance fixtures for all new differentiable ops.
5. Maintain the existing "all 234 integration tests green" invariant throughout.

### Non-Goals

1. Real `torch.utils.checkpoint.checkpoint_sequential` implementation — that is PRD-009. This PRD ships a stub-with-explicit-error only.
2. `ConvTranspose3d` — no curriculum currently requires it.
3. Multi-device dispatch for any new op — all new ops run on the NumPy CPU path.
4. `torch.topk` with `largest=False` or `sorted=False` arguments.
5. `torch.multinomial` with `replacement=False` and edge-case distributions — uncertain behavior.

---

## Architecture

### File placement map

| Op | File | Strategy |
|---|---|---|
| `Tensor.requires_grad_()` | `src/python/tensor.py` | New method on `Tensor`; mutates `self.requires_grad` in place; returns `self` |
| `Tensor.zero_()`, `add_()`, `mul_()` | `src/python/tensor.py` | Raise `RuntimeError` if `self._is_leaf and self.requires_grad`; otherwise mutate `.data` |
| `nn.ModuleList`, `nn.ModuleDict` | `src/python/nn_chunks/module.py` | New classes; use existing `Module.__setattr__` wiring |
| `torch.tril` | `src/python/tensor.py` (module-level) | Wraps `np.tril`; non-differentiable |
| `torch.topk` (function form, any dim) | `src/python/tensor.py` (module-level) | Generalizes existing `Tensor.topk` (1-D only); uses `np.argsort` |
| `torch.multinomial` | `src/python/tensor.py` (module-level) | Uses `np.random.choice` with prob weights; returns int64 |
| `F.softsign` | `src/python/functional.py` | New function in activations section; ~15 LOC |
| `register_backward_hook` on `Module` | `src/python/nn_chunks/module.py` | Add `_backward_hooks` list; fire from `Module.__call__` via thin `_build_ctx` wrapper on output |
| `Conv3d` | `src/python/nn_chunks/conv.py` | Extend Conv2d's im2col to 3 spatial dims |
| `InstanceNorm1d`, `InstanceNorm3d` | `src/python/nn_chunks/norm.py` | Generalize `InstanceNorm2d`'s per-instance reduction |
| `checkpoint_sequential` stub | New `src/python/nn_chunks/checkpoint.py` | Raises `NotImplementedError` with message pointing to PRD-009 |

### Conv3d im2col extension

```
Conv2d im2col layout:
  cols: (N, C_in * K_h * K_w, H_out * W_out)
  Each column i*W_out+j holds the flattened K_h×K_w window at position (i,j)

Conv3d im2col extension:
  cols: (N, C_in * K_d * K_h * K_w, D_out * H_out * W_out)
  Each column k*H_out*W_out + i*W_out + j holds the flattened K_d×K_h×K_w
  window at depth position k, spatial position (i,j)

  Triple loop:
  for k in range(D_out):          # new depth loop
    for i in range(H_out):
      for j in range(W_out):
        d0, h0, w0 = k*S_d, i*S_h, j*S_w
        cols[:, :, k*H_out*W_out + i*W_out + j] =
          x_padded[:, :, d0:d0+K_d, h0:h0+K_h, w0:w0+K_w].reshape(N, -1)

  weight_flat: (C_out, C_in * K_d * K_h * K_w)
  out_flat   = weight_flat @ cols
  out_data   = out_flat.reshape(N, C_out, D_out, H_out, W_out)
```

The backward (col2im) for Conv3d mirrors Conv2d's exactly — the only change is the `D_out` outer loop and the 5-D gradient accumulation buffer.

**Memory note**: for a single Conv3d layer with `C_in=32, K=3, N=4, D=16, H=16, W=16`, the `cols` matrix is `4 * (32 * 27) * (14 * 14 * 14) = ~27M float32 values = ~108MB`. This is the im2col memory cost and is the primary risk factor (see Risks).

### register_backward_hook design

Signature (matching `register_full_backward_hook`, the non-deprecated form, aliased to both names for compatibility):

```python
hook(module, grad_input: tuple[Tensor], grad_output: tuple[Tensor]) -> None | tuple[Tensor]
```

The hook fires after the module's backward pass. Implementation: in `Module.__call__`, after computing `output = self.forward(*args, **kwargs)`, if `self._backward_hooks` is non-empty, wrap `output` in a thin `_build_ctx` identity node that captures the backward hook invocations. When `.backward()` reaches that node, it calls each registered hook with `(module, grad_inputs, grad_outputs)` before propagating further.

This is simpler than PyTorch's C++ engine approach and correct for the educational use case. It does not support hooks that return a modified `grad_input` to reroute gradients in v0 (return value is ignored).

---

## API Surface

### Tensor in-place methods

```python
import browsergrad_grad as torch
import numpy as np

# requires_grad_() — in-place setter
x = torch.Tensor(np.array([1.0, 2.0, 3.0]))
x.requires_grad_(True)   # returns self; x.requires_grad is now True

# zero_(), add_(), mul_() — safe on non-leaf or no-grad leaves
w = torch.Tensor(np.ones((3, 3)), requires_grad=False)
w.zero_()          # fills w.data with zeros, returns self
w.add_(1.0)        # w.data += 1.0
w.mul_(2.0)        # w.data *= 2.0

# Error on leaf with requires_grad=True
param = torch.Tensor(np.ones(3), requires_grad=True)
param.zero_()      # raises RuntimeError("a leaf Variable that requires grad ...")
```

### torch.tril, torch.topk, torch.multinomial

```python
# torch.tril — causal mask
mask = torch.tril(torch.ones(4, 4))
# Tensor([[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1]])

# torch.topk — top-k along last dim
logits = torch.Tensor(np.array([[0.1, 0.9, 0.5, 0.3]]))
values, indices = torch.topk(logits, k=2)
# values:  Tensor([[0.9, 0.5]])
# indices: Tensor([[1, 2]])  (int64)

# torch.multinomial
probs = torch.Tensor(np.array([[0.1, 0.6, 0.3]]))
idx = torch.multinomial(probs, num_samples=1)
# idx: Tensor of shape (1, 1) with sampled index
```

### F.softsign

```python
import browsergrad_grad.functional as F

x = torch.Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), requires_grad=True)
y = F.softsign(x)
# y.data ≈ [-0.667, -0.500, 0.000, 0.500, 0.667]
y.sum().backward()
# x.grad.data ≈ [0.111, 0.250, 1.000, 0.250, 0.111]  (= 1/(1+|x|)^2)
```

### nn.ModuleList and nn.ModuleDict

```python
# ModuleList — integer-indexed
layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
params = list(layers.parameters())  # 3*2 = 6 parameter tensors
for layer in layers:
    x = layer(x)

# ModuleDict — string-keyed
model_parts = nn.ModuleDict({
    "wte": nn.Embedding(50257, 768),
    "drop": nn.Dropout(0.1),
    "ln_f": nn.LayerNorm(768),
})
```

### Conv3d, InstanceNorm1d, InstanceNorm3d

```python
conv = nn.Conv3d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
x = torch.Tensor(np.random.randn(2, 3, 8, 8, 8).astype(np.float32), requires_grad=True)
y = conv(x)  # y.shape == (2, 8, 8, 8, 8)
y.sum().backward()  # x.grad.shape == (2, 3, 8, 8, 8)

# 1D: (N, C, L)
norm1d = nn.InstanceNorm1d(num_features=16, affine=False)

# 3D: (N, C, D, H, W)
norm3d = nn.InstanceNorm3d(num_features=8, affine=False)
```

### register_backward_hook

```python
fc = nn.Linear(4, 2)
grad_norms = []

def hook(module, grad_input, grad_output):
    grad_norms.append(float(np.linalg.norm(grad_output[0].data)))

handle = fc.register_backward_hook(hook)
y = fc(x)
y.sum().backward()
handle.remove()
```

### checkpoint_sequential stub

```python
import torch.utils.checkpoint
torch.utils.checkpoint.checkpoint_sequential(model, segments=2, input=x)
# NotImplementedError: checkpoint_sequential is not implemented in browsergrad v0.
# Gradient checkpointing requires the tracing JIT (PRD-009, planned for P1.7).
```

---

## Implementation Plan

### Week 1 (Days 1–5): Tensor methods + functional + containers

**Day 1–2: Tensor in-place ops + tril + topk + multinomial**

File: `src/python/tensor.py`

- Add `Tensor.requires_grad_(requires_grad=True) -> Tensor`. Body: `self.requires_grad = bool(requires_grad); return self`.
- Add `Tensor.zero_() -> Tensor`. Body: check `if self._is_leaf and self.requires_grad: raise RuntimeError(...)`, then `self.data[...] = 0.0`.
- Add `Tensor.add_(other) -> Tensor`. Same leaf guard.
- Add `Tensor.mul_(other) -> Tensor`. Same leaf guard.
- Add module-level `tril(input, diagonal=0) -> Tensor`. Wraps `np.tril`.
- Add module-level `topk(input, k, dim=-1, largest=True, sorted=True)`. Uses `np.argsort` on the specified dim.
- Add module-level `multinomial(input, num_samples, replacement=True) -> Tensor`. Uses `np.random.choice` per row. Returns int64.

Write tests: `tests-integration/inplace_ops.test.ts` covering all four in-place methods, the leaf guard error, `tril` shape/values, `topk` dim behavior, `multinomial` output shape and valid index range.

**Day 3: F.softsign**

File: `src/python/functional.py`, insert after `gelu` (~line 50).

```python
def softsign(x: Tensor) -> Tensor:
    """softsign(x) = x / (1 + |x|). Smooth bounded activation.
    Derivative: 1 / (1 + |x|)^2"""
    abs_x = np.abs(x.data)
    out_data = x.data / (1.0 + abs_x)
    out = Tensor(out_data.astype(np.float32))
    denom = (1.0 + abs_x) ** 2
    return _build_ctx(out, (x,), lambda g: (g.data / denom,))
```

Write tests: `tests-integration/softsign.test.ts` covering forward, backward, and PyTorch conformance.

**Day 4–5: nn.ModuleList, nn.ModuleDict**

File: `src/python/nn_chunks/module.py`, append after `Module`.

`ModuleList`: `__init__`, `add_module`, `__getitem__`, `__setitem__`, `__len__`, `__iter__`, `append`, `forward` (raises NotImplementedError).

`ModuleDict`: `__init__`, `add_module`, `__getitem__`, `__setitem__`, `__contains__`, `keys`, `values`, `items`, `forward` (raises NotImplementedError).

Write tests: `tests-integration/module_containers.test.ts` covering parameter collection, state_dict round-trip, direct module access.

### Week 2 (Days 6–10): Conv3d + InstanceNorm + backward hook + fixtures

**Day 6–7: Conv3d**

File: `src/python/nn_chunks/conv.py`, append after `Conv1d`.

Key parameters: `in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True` (isotropic in v0).

Forward: triple loop over `(D_out, H_out, W_out)` filling `cols` of shape `(N, C_in * K^3, D_out * H_out * W_out)`. Then `weight_flat @ cols` reshaped to `(N, C_out, D_out, H_out, W_out)`.

Backward: col2im triple loop scatters gradient back to `(N, C_in, D+2P, H+2P, W+2P)` then slices padding.

Add memory guard: if `N * C_in * (K**3) * D_out * H_out * W_out * 4 > 512 * 1024 * 1024`, raise `MemoryError` with a clear message.

**Day 8: InstanceNorm1d, InstanceNorm3d**

File: `src/python/nn_chunks/norm.py`, append after `InstanceNorm2d`.

`InstanceNorm1d`: Input `(N, C, L)`. Reduction axes `(2,)`. Affine reshape `(1, C, 1)`.

`InstanceNorm3d`: Input `(N, C, D, H, W)`. Reduction axes `(2, 3, 4)`. Affine reshape `(1, C, 1, 1, 1)`.

**Day 9: register_backward_hook**

File: `src/python/nn_chunks/module.py`.

1. In `__init__`, add `self._backward_hooks: list = []`.
2. Add `register_backward_hook(self, fn)` (aliased as `register_full_backward_hook`): appends to `self._backward_hooks`, returns a handle with `.remove()`.
3. In `__call__`, after `output = self.forward(*args, **kwargs)`: if `self._backward_hooks` non-empty, wrap output's autograd node to capture backward hook invocations.

**Day 10: Fixtures + checkpoint stub + integration**

1. Add fixtures to `scripts/gen_pytorch_fixtures.py`: `make_softsign`, `make_tril`, `make_topk`, `make_conv3d`, `make_instancenorm1d`, `make_instancenorm3d`.
2. Add `torch.utils.checkpoint` stub as new chunk `nn_chunks/checkpoint.py` assembled after `init.py`.
3. Run full integration suite. Fix any regressions.
4. Verify nanoGPT `model.py` instantiation + one forward pass.

---

## Acceptance Criteria

Each op must satisfy:
1. Forward output agrees with real `torch` within 1e-4 on the fixture.
2. Backward (where applicable) agrees within 1e-4 OR a finite-difference check passes with step 1e-3.
3. The integration test for the op is green.
4. All 234 existing integration tests remain green.

### Per-op fixtures

**F.softsign** — Input `[-3.0, -1.0, 0.0, 1.0, 3.0]`. Forward exact: `[-0.75, -0.5, 0.0, 0.5, 0.75]`. Backward: `[0.0625, 0.25, 1.0, 0.25, 0.0625]`.

**Tensor.requires_grad_()** — `x = Tensor([1.0]); assert not x.requires_grad; x.requires_grad_(True); assert x.requires_grad`.

**Tensor.zero_/add_/mul_ leaf guard** — `param = Tensor([1.0], requires_grad=True); param.zero_()` must raise `RuntimeError` with `"leaf Variable that requires grad"`.

**torch.tril** — `torch.ones(4, 4)` → `[[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1]]` (exact).

**torch.topk** — `logits = Tensor([[0.1, 0.9, 0.5, 0.3, 0.7]])`, k=3 → values `[0.9, 0.7, 0.5]`, indices `[1, 4, 2]` (int64). Tolerance: exact on indices, 1e-5 on values.

**torch.multinomial** — Stochastic: shape + range only. Deterministic case: `probs = [[0.0, 1.0, 0.0]]` → all outputs must be `1` over 10 runs.

**nn.ModuleList** — Model with `self.layers = ModuleList([Linear(4,4), Linear(4,4)])`: `len(list(M().parameters())) == 4`. state_dict keys include `"layers.0.weight"`, etc.

**nn.ModuleDict** — Model with `self.parts = ModuleDict({"a": Linear(4,4), "b": Linear(4,2)})`: `len(list(M().parameters())) == 4`. state_dict keys include `"parts.a.weight"`, etc.

**Conv3d** — Input `(2, 3, 8, 8, 8)`, `Conv3d(3, 4, 3, padding=1)`. Forward shape `(2, 4, 8, 8, 8)`. Values within 1e-4 of triple-loop NumPy oracle AND within 1e-4 of PyTorch fixture.

**InstanceNorm1d** — Input `(4, 16, 32)`, `affine=False`. Per-instance mean `< 1e-4`, std within 1e-3 of 1.0.

**InstanceNorm3d** — Input `(2, 8, 4, 4, 4)`, `affine=False`. Per-instance mean `< 1e-4`, std within 1e-3 of 1.0.

**register_backward_hook** — Hook fires once per backward. `grad_output[0].shape` matches module output shape. Return value `None` doesn't alter gradient. `handle.remove()` prevents subsequent firing.

---

## Test Strategy

### Unit (surface) tests — `tests/surface.test.ts`

Verify package surface exports:
- `torch.tril`, `torch.topk`, `torch.multinomial`
- `F.softsign`
- `nn.ModuleList`, `nn.ModuleDict`, `nn.Conv3d`, `nn.InstanceNorm1d`, `nn.InstanceNorm3d`
- `torch.utils.checkpoint.checkpoint_sequential` (present and callable)

### Integration tests (Pyodide-in-Node)

New test files:
1. `tests-integration/inplace_ops.test.ts` — `requires_grad_`, `zero_`, `add_`, `mul_`, leaf guard, `tril`, `topk`, `multinomial`.
2. `tests-integration/softsign.test.ts` — forward, backward, PyTorch conformance.
3. `tests-integration/module_containers.test.ts` — ModuleList, ModuleDict, state_dict round-trip.

Extended test files:
4. `tests-integration/conv3d_norms_hooks.test.ts` — add Conv3d, InstanceNorm1d/3d, register_backward_hook describes.
5. `tests-integration/pytorch_conformance.test.ts` — new fixture kinds.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Conv3d im2col memory blow-up: `cols` matrix > 512MB | High | High | Explicit memory guard; raise `MemoryError` with clear message about reducing batch/input size. |
| `register_backward_hook` thin wrapper changes autograd graph topology, breaking existing backward tests | Medium | Medium | Implement wrapper only when `_backward_hooks` non-empty (zero cost when no hooks). Run full conformance suite. |
| `torch.multinomial` with `replacement=False` on edge-case distributions — undefined vs PyTorch | Low | Low | Document in docstring; v0 only guarantees `replacement=True`. |
| `ModuleList`/`ModuleDict` parameter tracking drops on `self.parts[0] = new_module` reassignment | Medium | Medium | `__setitem__` must call `add_module()`, not assign directly. Test mutation after construction. |
| `topk` with `dim != -1` on multi-dim — current partial impl is 1-D only | Low | Medium | Test against 2-D, 3-D, negative-dim inputs. nanoGPT call pattern covered specifically. |
| Pyodide's NumPy `random.choice` seeding differs from host NumPy → multinomial fixture non-deterministic | High | Low | Multinomial fixture only checks shape + valid-range. Deterministic test uses one-hot prob. |

---

## Open Questions

1. **Conv3d: im2col vs direct loop.** im2col matches Conv2d (same code pattern, proven backward) and is faster for typical educational shapes, but the `cols` matrix can be very large for 3D. Direct loop uses O(1) extra memory but no vectorization. **Proposed**: use im2col with 512MB guard. Revisit with direct-loop fallback if the guard fires in practice. **Confirm before Day 6.**

2. **register_backward_hook vs register_full_backward_hook.** PyTorch deprecated the former. Since we're a clean-room implementation, could skip the deprecated form. But nanoGPT/tutorials still use the old name. **Proposed**: implement both names pointing to the same function. **Confirm before Day 9.**

3. **torch.topk unification.** Existing 1-D `Tensor.topk` method vs new general `torch.topk(input, k, dim=)` function. **Proposed**: keep both; `Tensor.topk` delegates to the module-level function with `dim=-1`. **Confirm before Day 1.**

4. **InstanceNorm1d on 2-D input `(N, C)`.** PyTorch's accepts both `(N, C)` and `(N, C, L)`. With `(N, C)` input there's nothing to normalize (each instance is scalar). PyTorch effectively does nothing — output equals input. **Mark as "returns input unchanged for 2-D input" and document.**

5. **checkpoint_sequential stub placement.** A 13th chunk `checkpoint.py` is cleaner than polluting `tensor.py`. **Proposed**: 13th chunk assembled after `init.py`. **Confirm before Day 10.**

---

## References

1. [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) — lines 48, 102, 105, 343, 347.
2. [PyTorch autograd notes — in-place ops](https://docs.pytorch.org/docs/2.12/notes/autograd.html).
3. [PyTorch Tensor.requires_grad_() docs](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.requires_grad_.html).
4. [PyTorch Module.register_full_backward_hook](https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html).
5. [PyTorch forum — in-place ops on leaves](https://discuss.pytorch.org/t/write-to-data-instead-runtimeerror-a-leaf-variable-that-requires-grad-has-been-used-in-an-in-place-operation/41049).
6. [Open-Deep-ML/DML-OpenProblem](https://github.com/Open-Deep-ML/DML-OpenProblem) — softsign + checkpoint gaps.
7. [Anatomy of a high-performance convolution (im2col walkthrough)](https://sahnimanas.github.io/post/anatomy-of-a-high-performance-convolution/).
8. [Parallel Multi-Channel Convolution using GEMM (arXiv:1704.04428)](https://arxiv.org/pdf/1704.04428).
9. [PRD.md §7 P0.1](../../PRD.md) — parent feature specification.
10. [PROGRESS.md](../../PROGRESS.md) — Pile A rows 13, 14, 16, 17, 18.

---

*This PRD is self-contained. A new engineer should be able to start on Day 1 using only this document and the four source files listed in the Architecture section. Open Questions 1–5 require human sign-off before the relevant implementation days.*
