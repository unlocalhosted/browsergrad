# PRD-005 — `browsergrad-jit` MVP: IR + Tracer + NumPy Backend

| Field | Value |
|---|---|
| **Status** | Draft v1 |
| **Author** | unlocalhosted maintainers |
| **Date** | 2026-05-26 |
| **PRD ID** | PRD-005 |
| **Phase** | P1 (Months 4–9 of the 14-month roadmap in PRD.md §6) |
| **Package** | `@unlocalhosted/browsergrad-jit` |
| **Companion docs** | [VISION.md](../../VISION.md) §4 Layer 2 + Layer 3 · [PRD.md](../../PRD.md) §3.9 · [ARCHITECTURE.md](../../ARCHITECTURE.md) |
| **Predecessor** | `@unlocalhosted/browsergrad-grad` v0.5.0 (the correctness oracle) |
| **Successor PRDs** | PRD-006 (fusion) · PRD-007 (symbolic backward) · PRD-008 (pipeline cache) |

---

## TL;DR

`browsergrad-jit` is a new package that replaces eager NumPy dispatch with a lazy graph that defers all computation until a realization trigger (`.numpy()`, `.backward()`, etc.), building a UOp-style intermediate representation instead. Version 0 of the package uses NumPy as its sole realization backend, meaning users observe no performance change — the architectural value is entirely in the foundation it lays: a fully traced IR that future PRDs can attach kernel fusion (PRD-006), symbolic differentiation (PRD-007), pipeline caching (PRD-008), megakernel codegen (PRD-012), function transforms (PRD-014), and ONNX export (PRD-016) to. The entire existing PyTorch-shaped API (`nn.Module`, `Tensor`, `F.*`) is re-expressed on top of the IR with zero user-visible surface change, and the existing PyTorch-conformance suite in `browsergrad-grad` serves as the cross-validation oracle.

---

## Background

### Why a JIT now

`browsergrad-grad` v0.5.0 executes every op eagerly: each Python call allocates a NumPy array, runs the computation, stores a backward closure, and returns a concrete value. This design is correct, readable, and a complete educational implementation. It has a fixed ceiling: because computation happens at op-call time, the library cannot see what follows any given op, so it cannot fuse, dead-code-eliminate, or choose a better execution order. Every `softmax` is three kernel roundtrips; every transformer block is 11 sequential dispatches.

VISION.md §3 quantifies the gap: a ResNet18 training step in the current eager NumPy path is approximately 500ms; the projected target after full JIT + fusion is 8ms. That 60× gap is not closed by micro-optimizing NumPy; it requires seeing the whole graph before executing any of it.

PRD.md §3.9 locked in the decision: the JIT ships as `@unlocalhosted/browsergrad-jit`, a parallel package. `browsergrad-grad` stays alive as the correctness oracle throughout all JIT development. The migration path is: build the IR layer, re-implement every op to produce IR nodes, realize through NumPy at first, then substitute better backends without touching user code.

### What tinygrad and JAX prove

tinygrad's UOps design (see [tinygrad/tinygrad — ops.py](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/ops.py)) demonstrates that the complete PyTorch op surface decomposes into approximately 15 primitive node types. The entire forward pass of a transformer — hundreds of Python ops — becomes a flat graph of these primitives, which a compiler can then re-order, fuse, and lower to any backend. The design is intentionally minimal: fewer node types mean simpler fusion passes and simpler backend adapters.

JAX's tracing model ([JAX JIT docs](https://docs.jax.dev/en/latest/jit-compilation.html)) shows that standard Python code can be traced without modifying user code: replace concrete array values with abstract tracers, run the function, collect the IR, cache the trace keyed on shape signatures. The same function body serves both eager execution (concrete arrays) and traced execution (tracer objects), controlled entirely by the type of the argument passed in. We adopt the same separation.

PyTorch's `torch.compile` / TorchInductor ([torch.compile docs](https://pytorch.org/docs/stable/generated/torch.compile.html)) traces the `nn.Module.forward()` graph by running it through a symbolic dispatch path, then lowers to Triton or C++ kernels. The important lesson: tracing is per-call-signature (shape + dtype), not per-module instance. A module called with `(B=32, seq=512)` produces a different trace than `(B=64, seq=1024)`. We match this behavior.

---

## User Stories

**U1 — CNN training loop, identical results.** A student writes a standard CNN training loop using `import browsergrad_jit as torch`. The forward pass, loss computation, `loss.backward()`, and `optimizer.step()` all execute and produce gradients matching the `browsergrad-grad` eager path within 1e-4. The student cannot tell from the API that they are on the JIT path.

**U2 — Debugging via eager fallback.** A course author suspects a numerical mismatch in a custom loss function. They add `browsergrad_jit.use_eager(True)` at the top of the script. Every op now executes immediately via NumPy, printing intermediate values are concrete, and the mismatch is located. Removing the flag returns to lazy mode with no other changes.

**U3 — nanoGPT model import.** An engineer porting Karpathy's nanoGPT to browsergrad replaces `import torch` with `import browsergrad_jit as torch`. The model instantiates, a forward pass with `(B=4, seq=64)` produces a loss tensor, `.backward()` populates all parameter `.grad` fields, and one `optimizer.step()` updates weights — all matching the `browsergrad-grad` conformance fixtures.

---

## Goals and Non-Goals

### Goals

1. Ship a traced IR for the full `browsergrad-grad` op surface: every op that produces a `Tensor` in `browsergrad-grad` produces a `TensorProxy` in `browsergrad-jit`.
2. Define a complete, stable UOp IR with ~15 node types sufficient to represent all current ops.
3. Realize the IR via a NumPy backend — same NumPy operations as `browsergrad-grad`, invoked from the IR walk rather than from Python op calls.
4. Pass the existing 234 integration tests and 5 PyTorch-conformance fixtures unchanged, via the IR path.
5. Provide `browsergrad_jit.use_eager(True)` as a one-line debug switch.
6. Establish the package scaffolding (build pipeline, codegen pattern, test harness) so PRD-006 can immediately attach a WGSL backend.

### Non-Goals

1. Any performance improvement over `browsergrad-grad` v0.5.0. NumPy is the v0 backend; performance parity is the target, not speedup.
2. WGSL kernel generation or dispatch. That is PRD-006.
3. Kernel fusion. That is PRD-006 and PRD-012.
4. Symbolic backward differentiation. The backward IR in v0 is generated by recording NumPy backward closures as IR nodes — the same mechanism as `browsergrad-grad`, but wrapped in IR form. Fully symbolic backward is PRD-007.
5. Pipeline caching (OPFS). That is PRD-008.
6. WebNN dispatch. That is PRD-011.
7. Changing the user-visible Python API in any way.

---

## Architecture

### IR Design

The IR is a directed acyclic graph of `UOp` nodes. Each `UOp` carries: an opcode (one of the ~15 types below), a tuple of input `UOp` nodes, a `shape: tuple[int, ...]`, a `dtype: str` (NumPy dtype name), and an opaque `arg` field for op-specific metadata (axis, padding, scalar value, etc.).

**File:** `packages/browsergrad-jit/src/python/ir.py`

| Opcode | Inputs | `arg` | Output shape | Semantics |
|---|---|---|---|---|
| `BUFFER` | none | `buffer_id: str` | as declared | Leaf node. Represents a named buffer holding concrete data. |
| `CONST` | none | `value: float \| int` | `()` | Scalar constant. Folded away during constant-propagation. |
| `LOAD` | `(buf: BUFFER)` | none | buf.shape | Read a buffer into the computation. |
| `STORE` | `(dst: BUFFER, src: UOp)` | none | `()` | Write realized values back into a named buffer. |
| `ADD` | `(a, b)` | none | broadcast(a.shape, b.shape) | Elementwise addition with NumPy broadcasting. |
| `MUL` | `(a, b)` | none | broadcast(a.shape, b.shape) | Elementwise multiplication. |
| `DIV` | `(a, b)` | none | broadcast(a.shape, b.shape) | Elementwise division. |
| `EXP` | `(x,)` | none | x.shape | Elementwise natural exponent. |
| `LOG` | `(x,)` | none | x.shape | Elementwise natural log. |
| `NEG` | `(x,)` | none | x.shape | Elementwise negation. |
| `MATMUL` | `(a, b)` | none | matmul_shape(a, b) | Matrix multiplication; supports batched matmul. |
| `REDUCE` | `(x,)` | `op`, `axis`, `keepdims` | reduced shape | sum/max/min along axis. mean = REDUCE sum + scalar DIV. |
| `CAST` | `(x,)` | `dtype: str` | x.shape | Change element dtype. Covers `.float()`, `.long()`, `.bool()`. |
| `RESHAPE` | `(x,)` | `new_shape` | new_shape | View reshape. No data movement. |
| `PERMUTE` | `(x,)` | `axes: tuple` | permuted shape | Axis permutation. Covers `.transpose()`, `.T`, `permute()`. |
| `PAD` | `(x,)` | `pad_width`, `mode` | padded shape | Constant/reflect-pad. Covers `F.pad`. |
| `SLICE` | `(x,)` | `slices` | sliced shape | Contiguous slice. Covers `__getitem__` with slice keys. |
| `GATHER` | `(x, idx)` | `dim` | gather shape | Index-based gather. Covers `Tensor[mask]`, `Tensor[int_array]`. |
| `WHERE` | `(cond, a, b)` | none | broadcast shape | Elementwise conditional. |

Total: 19 opcodes. The additional four over the nominal 15 (`DIV`, `NEG`, `PAD`, `WHERE`) are load-bearing for the existing op surface.

**Concrete IR example — `y = x @ W + b`:**

```
load_x   = LOAD(x_buf)                         # shape (B, in)
load_w   = LOAD(w_buf)                         # shape (in, out)
mm       = MATMUL(load_x, load_w)              # shape (B, out)
load_b   = LOAD(b_buf)                         # shape (out,)
out      = ADD(mm, load_b)                     # shape (B, out) — b broadcast
store    = STORE(out_buf, out)
```

**Concrete IR example — `softmax(x, dim=-1)`:**

```
load_x   = LOAD(x_buf)                         # shape (B, C)
max_x    = REDUCE(load_x, op="max", axis=-1, keepdims=True)   # (B, 1)
shifted  = ADD(load_x, NEG(max_x))             # (B, C)
exp_x    = EXP(shifted)                        # (B, C)
sum_exp  = REDUCE(exp_x, op="sum", axis=-1, keepdims=True)    # (B, 1)
softmax  = DIV(exp_x, sum_exp)                 # (B, C)
store    = STORE(out_buf, softmax)
```

The `UOp` class is a frozen dataclass. Instances are immutable and hashable; equality is structural.

```python
@dataclass(frozen=True)
class UOp:
    op: str                        # opcode string
    inputs: Tuple["UOp", ...]      # upstream nodes
    shape: Tuple[int, ...]
    dtype: str                     # numpy dtype name
    arg: Any = None                # op-specific metadata
```

### Tensor Proxy Design

**File:** `packages/browsergrad-jit/src/python/tensor_proxy.py`

`TensorProxy` matches `browsergrad-grad`'s `Tensor` API slot-for-slot: same `.shape`, `.dtype`, `.requires_grad`, `.grad`, `.numpy()`, `.tolist()`, `.item()`, `.backward()`, all arithmetic operators, all reduction methods.

Internally:

```python
class TensorProxy:
    __slots__ = ("_uop", "requires_grad", "grad")

    def __init__(self, uop: UOp, requires_grad: bool = False):
        self._uop = uop
        self.requires_grad = requires_grad
        self.grad: Optional["TensorProxy"] = None
```

Every op on a `TensorProxy` constructs a new `UOp` and wraps it in a new `TensorProxy`. No computation occurs. The `_uop` DAG is built purely by object construction.

**Shape and dtype** are computed eagerly at proxy construction time so that code like `x.shape[0]` works without triggering realization. This matches JAX's abstract evaluation.

**Leaf proxies** (parameters, batch inputs) are backed by `BUFFER` + `LOAD` UOps. A `BufferTable` (module-level dict mapping `buffer_id: str → np.ndarray`) holds the actual data.

### Realization Algorithm

**File:** `packages/browsergrad-jit/src/python/realize.py`

Realization is triggered by:
- `.numpy()`, `.tolist()`, `.item()`
- `.backward()`
- `optimizer.step()` (reads `.grad`)

Algorithm:
1. **Collect** the output UOp.
2. **Topological sort** (Kahn's algorithm).
3. **Execute**: walk sorted list. Per `UOp`, dispatch to NumPy via `_NUMPY_OPS` table. Store in `ValueTable: dict[UOp, np.ndarray]`.
4. **Return** for `.numpy()`: `ValueTable[root_uop]`.

**NumPy dispatch table** (excerpt):

```python
_NUMPY_OPS: dict[str, Callable] = {
    "LOAD":    lambda uop, vt, bt: bt[uop.inputs[0].arg],
    "ADD":     lambda uop, vt, bt: vt[uop.inputs[0]] + vt[uop.inputs[1]],
    "MUL":     lambda uop, vt, bt: vt[uop.inputs[0]] * vt[uop.inputs[1]],
    "MATMUL":  lambda uop, vt, bt: vt[uop.inputs[0]] @ vt[uop.inputs[1]],
    "REDUCE":  _dispatch_reduce,
    "RESHAPE": lambda uop, vt, bt: vt[uop.inputs[0]].reshape(uop.shape),
    "PERMUTE": lambda uop, vt, bt: np.transpose(vt[uop.inputs[0]], uop.arg["axes"]),
    # ... etc
}
```

**Backward realization.** In v0, the backward graph is built as `browsergrad-grad` does: each op that produces a `TensorProxy` also registers a backward closure. The closure, when called with the upstream gradient proxy, produces new IR nodes. PRD-007 will replace this with symbolic backward; in v0 it's the same `_build_ctx` pattern translated to IR.

### Compat Surface: nn.Module on Top of IR

**Files:** `packages/browsergrad-jit/src/python/nn/module.py`, `linear.py`, `conv.py`, etc.

The compatibility surface re-implements the same 12 chunked Python files (`nn_chunks/module.py`, `linear.py`, `conv.py`, `norm.py`, `attention.py`, `recurrent.py`, `dropout.py`, `pool.py`, `loss.py`, `activation.py`, `embedding.py`). The public API is identical. Internally, every `forward()` method builds IR nodes instead of calling NumPy directly.

Example: `nn.Linear.forward()` in `browsergrad-grad` calls `x.data @ self.weight.data.T + self.bias.data`. In `browsergrad-jit`, it calls `x @ self.weight.T + self.bias` where these are `TensorProxy` instances. The `@` and `+` operators produce `MATMUL` and `ADD` UOps.

**Parameter registration** follows the same `__setattr__` pattern. When a `TensorProxy` with `requires_grad=True` is assigned as a module attribute, it is registered in `_parameters`. Each parameter is backed by a `BUFFER` node whose `buffer_id` is set during the first call to `module.parameters()`.

### Eager Fallback Mechanism

**File:** `packages/browsergrad-jit/src/python/config.py`

```python
_EAGER_MODE: bool = False

def use_eager(enabled: bool) -> None:
    """Switch between lazy (default) and eager (debug) mode."""
    global _EAGER_MODE
    _EAGER_MODE = enabled
```

When `_EAGER_MODE` is `True`, every op method on `TensorProxy` still constructs a `UOp` and immediately calls `realize()` before returning. The returned object is still a `TensorProxy` (not a bare `np.ndarray`) so downstream code that type-checks continues to work.

`eager_context()` context manager for fine-grained sections:

```python
with browsergrad_jit.eager_context():
    debug_val = suspicious_tensor.numpy()
```

### Cross-Validation Against the Eager Oracle

**File:** `packages/browsergrad-jit/tests-integration/jit_vs_grad_conformance.test.ts`

Every integration test in `browsergrad-grad` has a parallel JIT variant. The test harness runs the same Python snippet twice: once via `browsergrad-grad` (the oracle) and once via `browsergrad-jit`. Both results are compared within 1e-4 for floats, exact equality for integers.

The PyTorch-conformance fixtures (`fixtures/pytorch_conformance.json`) are loaded into both paths. Any divergence is a blocking bug.

---

## API Surface

From the user's perspective, no import changes beyond the package name:

```python
# Before (eager)
import browsergrad_grad as torch

# After (JIT, v0 — same NumPy speed, foundational architecture)
import browsergrad_jit as torch

import browsergrad_jit.nn as nn
import browsergrad_jit.nn.functional as F
import browsergrad_jit.optim as optim

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

model = MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

x = torch.randn(32, 784)       # TensorProxy backed by BUFFER+LOAD UOp
y = torch.randint(0, 10, (32,))

logits = model(x)              # builds IR; nothing computed yet
loss = F.cross_entropy(logits, y)   # extends IR; still nothing computed

optimizer.zero_grad()
loss.backward()                # REALIZATION POINT — IR walked, NumPy executes
optimizer.step()               # reads .grad, another small realization

print(loss.item())             # REALIZATION POINT if not yet realized
arr = logits.numpy()           # returns np.ndarray

browsergrad_jit.use_eager(True)    # all subsequent ops execute immediately
x2 = torch.tensor([1.0, 2.0])
y2 = x2 * 2                        # NumPy runs immediately
browsergrad_jit.use_eager(False)
```

**Torch alias support**:

```python
import browsergrad_jit
browsergrad_jit.install_torch_alias()
import torch          # now points at browsergrad_jit
```

---

## Implementation Plan

### Week 1 — Package scaffolding + IR definition

- [ ] Create `packages/browsergrad-jit/` mirroring `browsergrad-grad` structure: same codegen + build + test scripts, package name `@unlocalhosted/browsergrad-jit`, Python module `browsergrad_jit`.
- [ ] Write `src/python/ir.py`: frozen `UOp` dataclass, 19 opcodes as string constants, `_arg_hash` helper, `toposort(root: UOp) -> list[UOp]`.
- [ ] Write unit tests for `ir.py`: topological sort on diamond graphs, hash stability, structural equality.
- [ ] Set up `vitest.integration.config.ts` pointing at `pyodide-host.ts`, extended to install `browsergrad-jit`.

### Week 2 — TensorProxy + leaf construction

- [ ] Write `src/python/tensor_proxy.py`: `TensorProxy` class, `__slots__`, property accessors delegating to `_uop.shape`/`_uop.dtype`, `__repr__` matching `browsergrad-grad`.
- [ ] Write `src/python/buffer_table.py`: `BufferTable` singleton dict. `new_buffer(name, array)` → registers, returns `BUFFER` UOp.
- [ ] Implement `torch.tensor()`, `torch.zeros()`, `torch.ones()`, `torch.randn()`, `torch.randint()` as factory functions.
- [ ] Integration test: `x = torch.tensor([1.0, 2.0]); assert x.shape == (2,)` — proxy shape correct before realization.

### Week 3 — NumPy realizer

- [ ] Write `src/python/realize.py`: `realize(uop) -> np.ndarray` with topological walk + `_NUMPY_OPS` dispatch.
- [ ] Implement all 19 opcode handlers. `REDUCE`, `PAD`, `GATHER` handlers ported from `browsergrad-grad`'s `functional.py`.
- [ ] Wire into `TensorProxy`: `.numpy()`, `.tolist()`, `.item()` all call `realize(self._uop)`.
- [ ] Integration test: `assert torch.tensor([1.,2.,3.]).sum().item() == 6.0`.

### Week 4 — Arithmetic + elementwise ops

- [ ] Implement all arithmetic on `TensorProxy`: `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__neg__`, `__matmul__`, `__pow__`. Broadcasting shape inference.
- [ ] Unary: `.exp()`, `.log()`, `.abs()`, `.sqrt()`.
- [ ] Reduction: `.sum()`, `.mean()`, `.max()`, `.min()`, `.argmax()`.
- [ ] Shape: `.reshape()`, `.view()`, `.transpose()`, `.permute()`, `.unsqueeze()`, `.squeeze()`, `.T`.
- [ ] Integration test: `y = x @ W + b` decomposes to expected 5-node IR. Assert IR structure by walking `y._uop`.

### Week 5 — Backward graph + autograd

- [ ] Port `_build_ctx` from `browsergrad-grad`. Backward closures return new `TensorProxy` objects built from IR nodes. Not symbolically differentiated yet — that's PRD-007.
- [ ] Implement `.backward()`: collect leaf proxies, reverse topological walk, accumulate `.grad`.
- [ ] Implement `no_grad` context manager.
- [ ] Integration test: `loss.backward()` on a 2-layer MLP matches `browsergrad-grad` values within 1e-4.

### Week 6 — nn.Module compat surface + full conformance

- [ ] Port all 12 `nn_chunks` files into `packages/browsergrad-jit/src/python/nn/`.
- [ ] Port `functional.py` entirely.
- [ ] Port `optim.py`: SGD, Adam, AdamW. Each `step()` builds and realizes a small IR graph per parameter.
- [ ] Port `torch_compat.py` shims (Pile A/B/C identical to `browsergrad-grad`).
- [ ] Run full 234-test integration suite. Target: zero failures.
- [ ] Run 5 PyTorch-conformance fixtures. Target: all within 1e-4.
- [ ] Implement `use_eager()` and `eager_context()`.
- [ ] Publish `@unlocalhosted/browsergrad-jit@0.1.0` to npm.

---

## Acceptance Criteria

| # | Criterion | Measurement |
|---|---|---|
| AC1 | All 234 `browsergrad-grad` integration tests pass via JIT path unchanged | `pnpm test:integration` green |
| AC2 | All 5 PyTorch-conformance fixtures pass within 1e-4 | `pytorch_conformance.test.ts` green on JIT path |
| AC3 | IR for `y = x @ W + b` is exactly 5 nodes (LOAD, LOAD, MATMUL, LOAD, ADD) | `ir_structure.test.ts` |
| AC4 | IR for `softmax(x, dim=-1)` is exactly 6 nodes | `ir_structure.test.ts` |
| AC5 | `loss.backward()` populates `.grad` within 1e-4 of `browsergrad-grad` | `jit_vs_grad_conformance.test.ts` |
| AC6 | `use_eager(True)` switches to immediate NumPy dispatch | `eager_mode.test.ts` |
| AC7 | JIT path no worse than 3× slower than `browsergrad-grad` eager on 2-layer MLP | Benchmark in CI (tracked, not blocking) |
| AC8 | `torch.tensor`, `torch.randn`, `torch.zeros`, `torch.ones`, `torch.randint` all work | Integration suite |
| AC9 | `TensorProxy.shape` available without triggering realization | `assert model(x).shape == (32, 10)` without `.numpy()` |
| AC10 | Package builds via `pnpm build` with zero TS errors | CI |

---

## Test Strategy

### Unit tests (`tests/` — Vitest, no Pyodide)

- `ir_unit.test.ts`: `UOp` hash stability, toposort correctness on 5 synthetic graphs.
- `codegen_unit.test.ts`: Python source files concatenated and exported correctly.

### Integration tests (`tests-integration/` — Vitest + real Pyodide-in-Node)

- `jit_vs_grad_conformance.test.ts` — the primary oracle test. For each of the 234 `browsergrad-grad` scenarios, runs the same code via `browsergrad-jit` and asserts numerical equality.
- `pytorch_conformance.test.ts` — re-uses `fixtures/pytorch_conformance.json`.
- `ir_structure.test.ts` — asserts on shape of built IR graphs before realization.
- `eager_mode.test.ts` — tests `use_eager(True/False)` and `eager_context()`.
- `backward_parity.test.ts` — gradient checks: every backward closure's output matches numerical finite-difference gradient within 1e-3.

### Cross-package conformance

The existing `cross-package-conformance.test.ts` in `browsergrad-grad` is extended to run every fixture through both packages and assert they agree, providing a single always-green signal.

---

## Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | `conv2d` backward through IR significantly more complex than forward — may block AC1 for 1-2 weeks | High | Medium | Implement `conv2d` forward + backward using `np.lib.stride_tricks` as opaque `CUSTOM` UOp in v0; replace with decomposed IR in PRD-006 |
| R2 | `TensorProxy.shape` shape-inference bugs in edge cases (dynamic shapes, empty tensors, -1 in reshape) cause silent wrong-shape IR | Medium | High | Add shape-inference unit tests for every opcode; fail loudly with `ShapeError` on unresolvable shape |
| R3 | Backward closures capturing NumPy arrays create memory leak — full forward NumPy state retained until `.backward()` | Medium | Medium | Benchmark peak memory on 5-layer MLP; if > 3× forward peak, switch to capturing only UOp references |
| R4 | `BufferTable` singleton not thread-safe; Pyodide's GIL is safe today but Web Worker parallelism would break it | Low | Medium | Document limitation; design `BufferTable` as per-session object from start |
| R5 | Performance regression > 3× vs eager NumPy makes JIT path unusable | Medium | Medium | Profile topological sort + dispatch loop on 50-node graph; cache sorted order keyed on root UOp hash if overhead > 10ms |
| R6 | Opaque UOp hash collisions cause silent wrong IR substitution | Low | High | Use SHA-256 of full serialized UOp tree, not Python's `hash()`, for any dict keying surviving across realizations |

---

## Open Questions

1. **Python package naming.** The npm package is `@unlocalhosted/browsergrad-jit`. The Python module inside Pyodide is `browsergrad_jit`. The torch alias maps to `browsergrad_jit`. Does `install_torch_alias()` in `browsergrad-jit` shadow `browsergrad-grad`'s if both are installed? Resolution: only one package calls `install_torch_alias()` per session; `browsergrad-jit` takes precedence.

2. **Eager-vs-JIT toggle semantics for module-level state.** `use_eager(True)` is a global flag. If two concurrent training loops run in the same Pyodide worker, flipping affects both. Resolution: add deprecation note that `use_eager` is process-global; recommend `eager_context()` for scoped use.

3. **Debugger story.** A student calling `print(x)` on `TensorProxy` gets `TensorProxy(shape=(32, 784))` — less informative than eager which prints values. Should `__repr__` trigger realization? Resolution for v0: no — surprises users who print tensors inside the traced graph. Instead provide `x.peek()` as explicit "realize and show first 5 values" debug method.

4. **`conv2d` in the IR decomposition.** A proper `CONV2D` UOp enables fusion with normalization. Opaque fallback works for v0 but leaves fusion potential on the table. Decision deferred to PRD-006, which needs a proper `CONV2D` opcode anyway.

5. **Grad accumulation across multiple `.backward()` calls.** `browsergrad-grad` accumulates `.grad` (PyTorch behavior). `BufferTable` design needs accumulation into `.grad` rather than overwrite. Confirm `STORE` for gradient writes uses `+=` semantics for non-leaf gradients.

---

## References

1. **tinygrad UOps design** — [tinygrad ops.py](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/ops.py). Authoritative reference for the ~15-opcode IR.

2. **JAX JIT compilation and tracing** — [JAX JIT docs](https://docs.jax.dev/en/latest/jit-compilation.html); [JAX `make_jaxpr`](https://docs.jax.dev/en/latest/jax.make_jaxpr.html). Tracer/proxy design and shape-inference-at-trace-time follow JAX's abstract evaluation.

3. **PyTorch `torch.compile` and TorchInductor** — [torch.compile overview](https://pytorch.org/docs/stable/generated/torch.compile.html); [TorchInductor design](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747). Per-shape-signature trace caching and `nn.Module.forward()` entry point model.

4. **Flash Attention** — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135). Canonical example of what IR-level fusion enables.

5. **WebGPU matmul performance ceiling** — [nuss-and-bolts.com WGSL matmul](https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel). NumPy v0 path will be superseded in PRD-006.

6. **Pyodide PyTorch compatibility block** — [pyodide/pyodide #1625](https://github.com/pyodide/pyodide/issues/1625). External constraint making `browsergrad-jit` the only viable PyTorch-shaped JIT-tracing runtime in Pyodide.

7. **browsergrad-grad source** — `packages/browsergrad-grad/src/python/tensor.py` (Tensor + `_build_ctx`), `functional.py` (NumPy implementations), `nn_chunks/module.py` (Module base class). Direct implementation sources for the compat layer.
