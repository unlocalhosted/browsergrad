# PRD-007 — Symbolic Backward: VJP Rules in the IR

| Field | Value |
|---|---|
| **Status** | Draft v1 |
| **Author** | unlocalhosted maintainers |
| **Date** | 2026-05-26 |
| **PRD ID** | PRD-007 |
| **Phase** | P1 (Months 4–9 of the 14-month roadmap in PRD.md §6) |
| **Package** | `@unlocalhosted/browsergrad-jit` |
| **Depends on** | PRD-005 (JIT MVP — IR + tracer + NumPy realizer) |
| **Enables** | PRD-006 (fusion of forward+backward), PRD-009 (gradient checkpointing), PRD-014 (torch.func / vmap) |

---

## TL;DR

`browsergrad-jit` v0.1 (PRD-005) executes backward by replaying the forward op-call closures — same `_build_ctx` pattern as `browsergrad-grad`, just translated to build IR nodes. The closures hold references to NumPy arrays of forward activations and run NumPy operations when invoked. This is *correct* but it leaves backward outside the IR: the compiler cannot fuse, cache, or eliminate work in the backward pass, and `vmap`/`grad` transforms have nothing to operate on. This PRD replaces the closure-based backward with **symbolic vector-Jacobian-product (VJP) rules**: every opcode has a registered rule that, given the upstream gradient UOp and the original forward inputs/output, emits new UOps for the input gradients. The result is a single unified IR for forward and backward, opening every downstream optimization (kernel fusion across forward/backward, gradient checkpointing, function transforms, ONNX export of the gradient graph).

---

## Background

### Why closures aren't enough

`browsergrad-grad` (and PRD-005's port) wires backward via per-op closures:

```python
# packages/browsergrad-grad/src/python/tensor.py:474 — _build_ctx
def _matmul(self, other):
    out = Tensor(self.data @ other.data, requires_grad=...)
    def _backward(grad):
        if self.requires_grad: self.grad += grad @ other.data.T
        if other.requires_grad: other.grad += self.data.T @ grad
    out._ctx = _backward
    return out
```

The closure captures `self.data` and `other.data` — concrete NumPy arrays from the forward pass. At `.backward()` time, the closure runs, mutating `.grad` fields in place via NumPy. PRD-005 ported this pattern to wrap the operations in `TensorProxy` construction, so the closures now build IR nodes. But the structural problem is the same: **backward computation is opaque to the IR walker until each closure fires.**

This blocks everything downstream:
- **Fusion (PRD-006)** can fuse a chain of forward elementwise ops into one kernel. But it cannot fuse the matching backward — there is no backward graph at fusion time, only a list of opaque closures.
- **Gradient checkpointing (PRD-009)** requires re-running a subgraph of the forward during backward. With closure-backward, you would have to manually nest closures inside closures — fragile and prevents proper IR-level analysis.
- **`torch.func.vmap` and `torch.func.grad` (PRD-014)** are graph-to-graph transformations. They need a graph. Closures cannot be transformed by IR passes.
- **ONNX export (PRD-016)** of a training graph requires both forward and backward as IR. Closures are not serializable.

### What JAX, PyTorch, and tinygrad do

**JAX** ([JAX autodiff docs](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html); [JAX VJP source](https://github.com/jax-ml/jax/blob/main/jax/interpreters/ad.py)) registers per-primitive VJP rules. Each primitive in the JAX lattice has a `defjvp_op` or `defvjp` function. When `jax.grad(f)` is called, JAX traces `f` to a Jaxpr, walks it in reverse topological order, invokes each primitive's VJP rule to produce new Jaxpr nodes for the gradients. The output is itself a Jaxpr — a graph, same kind as forward. Function transforms compose because they operate on graphs.

**PyTorch** ([PyTorch autograd internals](https://pytorch.org/docs/stable/notes/autograd.html); [PyTorch Function docs](https://pytorch.org/docs/stable/generated/torch.autograd.Function.html); [derivatives.yaml in PyTorch](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml)) declares derivatives in `derivatives.yaml`, a YAML file mapping every op to its gradient expression in PyTorch's own ATen language. Codegen converts these into `*Backward` classes registered with the autograd engine. The runtime engine treats backward as ordinary IR construction during the backward pass, enabling double backward (Hessian-vector products) through the same mechanism.

**tinygrad** ([tinygrad function.py](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/function.py)) defines `class Function` with `forward(ctx, *args)` and `backward(ctx, grad_output)` as classmethods. The forward returns a `LazyBuffer`; the backward returns one or more `LazyBuffer` operations. Backward is just more lazy IR.

All three approaches share the same property: **the backward graph is constructed from IR primitives, not from opaque callables.** Our PRD-007 implements the same model.

### The math: VJP rules per opcode

For each opcode `op` with forward function `y = op(x₁, x₂, ..., xₙ)`, the VJP rule computes `(∂L/∂x₁, ..., ∂L/∂xₙ)` given `∂L/∂y` (the upstream gradient).

Standard rules (proofs trivial from chain rule):

| Op | Forward | VJP for each input |
|---|---|---|
| `ADD` | `y = a + b` | `da = dy`, `db = dy` (with un-broadcast to original shape) |
| `MUL` | `y = a * b` | `da = dy * b`, `db = dy * a` (with un-broadcast) |
| `DIV` | `y = a / b` | `da = dy / b`, `db = -dy * a / (b*b)` |
| `NEG` | `y = -x` | `dx = -dy` |
| `EXP` | `y = exp(x)` | `dx = dy * y` (re-uses the forward output) |
| `LOG` | `y = log(x)` | `dx = dy / x` |
| `MATMUL` | `y = a @ b` | `da = dy @ b.T`, `db = a.T @ dy` (batched matmul: T on last 2 axes) |
| `REDUCE(sum, axis)` | `y = sum(x, axis)` | `dx = broadcast(dy, x.shape)` |
| `REDUCE(max, axis)` | `y = max(x, axis)` | `dx = where(x == broadcast(y), broadcast(dy), 0)` |
| `REDUCE(mean, axis)` | `y = mean(x, axis)` | `dx = broadcast(dy / N, x.shape)` where N = product of reduced dims |
| `RESHAPE(new_shape)` | `y = reshape(x, new_shape)` | `dx = reshape(dy, x.shape)` |
| `PERMUTE(axes)` | `y = permute(x, axes)` | `dx = permute(dy, inverse(axes))` |
| `PAD(pad_width)` | `y = pad(x, pad_width)` | `dx = slice(dy, inverse(pad_width))` |
| `SLICE(slices)` | `y = x[slices]` | `dx = pad_into(dy, x.shape, slices)` |
| `CAST(dtype)` | `y = cast(x, dtype)` | `dx = cast(dy, x.dtype)` |
| `GATHER(dim, idx)` | `y = gather(x, dim, idx)` | `dx = scatter_add(zeros(x.shape), dim, idx, dy)` |
| `WHERE` | `y = where(c, a, b)` | `da = where(c, dy, 0)`, `db = where(c, 0, dy)`, `dc = 0` (boolean) |

**Un-broadcasting** for `ADD`/`MUL`/`DIV` handles the case where inputs were broadcast to match shapes. The gradient comes back at the broadcast (larger) shape and must be summed along broadcast dimensions to match the original input shape. This is a small `REDUCE(sum)` insertion in the VJP.

---

## User Stories

**U1 — Unified IR view.** A maintainer prints `loss._uop` after `.backward()` is *constructed but not yet realized*. The output shows a single graph containing both the forward computation and all gradient computations as connected UOps — proving backward is no longer opaque.

**U2 — Backward fusion benefit.** A training loop that previously had K forward kernel dispatches and K backward dispatches drops to (K+K)/F dispatches after fusion is enabled in PRD-006, because PRD-006 can see and fuse the backward graph.

**U3 — Double backward (Hessian-vector product).** A user writes `g = torch.autograd.grad(loss, params, create_graph=True)` and then takes a second `.backward()` on `sum(g)`. Both gradients match `browsergrad-grad`'s eager double-backward within 1e-3.

**U4 — Closed numerical conformance.** All 234 integration tests and 5 PyTorch-conformance fixtures pass via the symbolic backward path, with gradient match within 1e-4 of the closure path.

---

## Goals and Non-Goals

### Goals

1. Register VJP rules for all 19 opcodes defined in PRD-005's IR.
2. Replace `_build_ctx` closure mechanism in `browsergrad-jit` with reverse-topological IR walk + per-op VJP rule invocation.
3. Pass the full conformance suite (234 integration tests, 5 PyTorch fixtures) with the same numerical tolerance (1e-4) as the closure path.
4. Provide `create_graph=True` support enabling double backward through the same mechanism.
5. Add unit tests asserting on the *structure* of the backward IR for each opcode in isolation.
6. Add a `viz_backward_graph(loss)` utility that emits Graphviz/DOT for the combined forward+backward DAG.

### Non-Goals

1. Performance speedup over closure-based backward in this PRD alone. Realized values still come from the NumPy backend — same arithmetic, just routed through IR construction. Speedup is unlocked by *enabling* PRD-006/009/012/014 to operate on the unified graph.
2. Custom user-defined backward via `torch.autograd.Function` — that is a separate small PRD (or P2 stretch); the symbolic rule registry can be exposed but ergonomic Python API is out of scope here.
3. `jvp` (forward-mode autodiff). Optional addition once VJP machinery is solid; defer to PRD-014 if needed for `torch.func.jvp`.
4. Higher-order derivatives beyond 2nd order. The mechanism supports them; explicit tests are not in scope.

---

## Architecture

### VJP Rule Registry

**File:** `packages/browsergrad-jit/src/python/vjp.py`

```python
# Signature: VJPRule = Callable[[UOp, UOp, tuple[UOp, ...]], tuple[Optional[UOp], ...]]
# Args:    (output, upstream_grad, original_inputs) -> tuple of input_grads
# Returning None for an input means "no gradient flows here" (e.g. WHERE condition input).

_VJP_RULES: dict[str, VJPRule] = {}

def register_vjp(op: str):
    def deco(fn): _VJP_RULES[op] = fn; return fn
    return deco

@register_vjp("ADD")
def _vjp_add(output, dy, inputs):
    a, b = inputs
    da = _maybe_unbroadcast(dy, a.shape)
    db = _maybe_unbroadcast(dy, b.shape)
    return (da, db)

@register_vjp("MUL")
def _vjp_mul(output, dy, inputs):
    a, b = inputs
    da = _maybe_unbroadcast(UOp("MUL", (dy, b), a.shape, a.dtype), a.shape)
    db = _maybe_unbroadcast(UOp("MUL", (dy, a), b.shape, b.dtype), b.shape)
    return (da, db)

@register_vjp("MATMUL")
def _vjp_matmul(output, dy, inputs):
    a, b = inputs
    bT = UOp("PERMUTE", (b,), _transpose_last_two(b.shape), b.dtype,
             arg={"axes": _last_two_swap_axes(len(b.shape))})
    aT = UOp("PERMUTE", (a,), _transpose_last_two(a.shape), a.dtype,
             arg={"axes": _last_two_swap_axes(len(a.shape))})
    da = UOp("MATMUL", (dy, bT), a.shape, a.dtype)
    db = UOp("MATMUL", (aT, dy), b.shape, b.dtype)
    return (da, db)
```

Each rule constructs new `UOp` nodes only — no NumPy execution. The output of `.backward()` is an extended IR graph rooted at the original outputs, with new edges into gradient subgraphs.

### Reverse-Topological Backward Pass

**File:** `packages/browsergrad-jit/src/python/autograd.py`

```python
def backward(loss_proxy: TensorProxy, create_graph: bool = False) -> None:
    """Build symbolic gradient subgraphs and accumulate into leaf .grad fields."""
    assert loss_proxy.shape == (), "loss must be scalar"

    # 1. Seed: dy for the loss is a CONST(1.0) UOp.
    seed = UOp("CONST", (), (), loss_proxy.dtype, arg={"value": 1.0})
    grads: dict[UOp, UOp] = {loss_proxy._uop: seed}

    # 2. Reverse topological walk.
    sorted_uops = toposort(loss_proxy._uop)
    for node in reversed(sorted_uops):
        if node not in grads:
            continue  # no gradient flowing to this node
        dy = grads[node]
        rule = _VJP_RULES.get(node.op)
        if rule is None:
            if node.op in ("BUFFER", "CONST", "LOAD"):
                continue  # leaves — accumulate into BufferTable grad slot
            raise NotImplementedError(f"No VJP rule for opcode {node.op!r}")
        input_grads = rule(node, dy, node.inputs)
        for inp, g in zip(node.inputs, input_grads):
            if g is None:
                continue
            # Accumulate: if multiple paths reach inp, sum the gradients.
            existing = grads.get(inp)
            grads[inp] = g if existing is None else UOp("ADD", (existing, g),
                                                       inp.shape, inp.dtype)

    # 3. Stash gradients onto leaf proxies' .grad fields.
    for proxy in _ALL_LEAF_PROXIES_WITH_GRAD:
        if proxy._uop in grads:
            proxy.grad = TensorProxy(grads[proxy._uop], requires_grad=create_graph)

    # 4. Realization deferred — .grad holds a TensorProxy whose _uop is the
    #    gradient subgraph. optimizer.step() triggers realization.
```

If `create_graph=True`, the resulting `proxy.grad` is itself a `TensorProxy` with `requires_grad=True`. A subsequent `.backward()` on a function of that grad triggers the same machinery recursively — VJP rules building VJP rules. Same trick PyTorch and JAX use.

### Un-broadcasting helper

```python
def _maybe_unbroadcast(grad: UOp, target_shape: tuple) -> UOp:
    """Sum gradient along axes that were broadcast in the forward."""
    if grad.shape == target_shape:
        return grad
    # Find axes to reduce: leading dims (added by broadcast) + dims where target is 1.
    extra_dims = len(grad.shape) - len(target_shape)
    reduce_axes = list(range(extra_dims))
    for i, (g_dim, t_dim) in enumerate(zip(grad.shape[extra_dims:], target_shape)):
        if t_dim == 1 and g_dim != 1:
            reduce_axes.append(extra_dims + i)
    if not reduce_axes:
        return grad
    reduced = UOp("REDUCE", (grad,), _reduce_shape(grad.shape, reduce_axes, keepdims=True),
                  grad.dtype, arg={"op": "sum", "axis": reduce_axes, "keepdims": True})
    return UOp("RESHAPE", (reduced,), target_shape, grad.dtype, arg={"new_shape": target_shape})
```

### Integration with TensorProxy

`TensorProxy.backward(create_graph: bool = False)` simply calls `autograd.backward(self, create_graph)`. The proxy's existing `.grad` slot now holds either `None` or a `TensorProxy` whose `_uop` is the gradient subgraph. `optimizer.step()` realizes the subgraph during the parameter update.

### Conformance Layer

**File:** `packages/browsergrad-jit/tests-integration/symbolic_vs_closure_parity.test.ts`

For every opcode, runs a small test that:
1. Constructs forward as `y = op(x)` with `x.requires_grad=True`.
2. Computes `y.sum().backward()` via the closure path (PRD-005 v0.1).
3. Computes the same via the symbolic VJP path (this PRD).
4. Asserts `np.allclose(x.grad_closure, x.grad_symbolic, atol=1e-5)`.

Extra finite-difference check for each opcode: numerical gradient via central difference within 1e-3.

---

## API Surface

External: zero change. `loss.backward()` works identically; `x.grad` is still a `Tensor`/`TensorProxy`. New optional argument `create_graph=True` enables double backward.

Internal-only additions:
```python
# Available for advanced use / debugging
browsergrad_jit.vjp.list_registered() -> list[str]   # ['ADD', 'MUL', ..., 'GATHER']
browsergrad_jit.viz_backward_graph(loss) -> str       # Graphviz DOT
browsergrad_jit.autograd.grad(loss, params, create_graph=False)  # JAX/PyTorch-style explicit
```

---

## Implementation Plan

### Week 1 — Foundations + simple opcodes

- [ ] Create `packages/browsergrad-jit/src/python/vjp.py` with `_VJP_RULES`, `register_vjp`, `_maybe_unbroadcast`, signature helpers.
- [ ] Implement VJP for trivially symmetric opcodes: `ADD`, `NEG`, `RESHAPE`, `PERMUTE`, `CAST`.
- [ ] Unit test (Vitest, no Pyodide): build `y = a + b`, walk grad, assert grad structure is `(LOAD seed, LOAD seed)` and shapes match.

### Week 2 — Multiplicative opcodes + broadcasting

- [ ] Implement VJP for `MUL`, `DIV`, `EXP`, `LOG`.
- [ ] Implement broadcasting un-projection in `_maybe_unbroadcast`.
- [ ] Tests for broadcast cases: `a: (32, 1)`, `b: (32, 10)` → `(a*b).sum().backward()` should produce `(REDUCE(sum, axis=1), b_grad)`.

### Week 3 — Matmul + reductions

- [ ] Implement VJP for `MATMUL` including batched dims.
- [ ] Implement VJP for `REDUCE(sum/mean/max/min)` with correct un-broadcasting.
- [ ] Integration test on the IR for a single-layer linear: `(x @ W + b).sum().backward()` produces expected nodes.

### Week 4 — Index/shape opcodes

- [ ] Implement VJP for `PAD`, `SLICE`, `GATHER`, `WHERE`.
- [ ] `GATHER` backward requires `SCATTER_ADD` — add it as a 20th opcode if needed (PRD-005 already reserves `GATHER`; the scatter inverse becomes a sibling opcode here).
- [ ] Tests for slicing through indexing: `y = x[:, 5:10]`, `y.sum().backward()` produces correct sparse grad.

### Week 5 — Reverse walk + accumulation

- [ ] Implement `autograd.backward()` reverse topological walk in `autograd.py`.
- [ ] Wire `TensorProxy.backward()` to dispatch to the new path.
- [ ] Add per-leaf accumulation: when grad reaches a `BUFFER` UOp, accumulate (sum) into `BufferTable` `.grad` slot.
- [ ] Cross-validation: run all 234 integration tests via the symbolic path. Target zero regressions.

### Week 6 — Double backward + viz + cleanup

- [ ] Implement `create_graph=True`: resulting gradient `TensorProxy` objects have `requires_grad=True` and participate in subsequent backward.
- [ ] Add integration test for Hessian-vector product: `H @ v` for a 2-layer MLP, compare to numerical finite-difference within 1e-3.
- [ ] Implement `viz_backward_graph(loss)` writing Graphviz DOT to a string, useful for course material.
- [ ] Remove the closure-based `_ctx` field from `TensorProxy`. (Closure path remains in `browsergrad-grad` only.)
- [ ] Publish `@unlocalhosted/browsergrad-jit@0.2.0` to npm.

---

## Acceptance Criteria

| # | Criterion | Measurement |
|---|---|---|
| AC1 | VJP rules registered for all 19 PRD-005 opcodes plus `SCATTER_ADD` if introduced | `vjp.list_registered()` returns full list |
| AC2 | All 234 integration tests pass via symbolic backward path within 1e-5 of closure path | `pnpm test:integration` green |
| AC3 | All 5 PyTorch conformance fixtures pass within 1e-4 | `pytorch_conformance.test.ts` green |
| AC4 | Double backward (`create_graph=True` then second `.backward()`) matches eager double-backward within 1e-3 | `double_backward.test.ts` |
| AC5 | Finite-difference gradient check passes for every opcode in isolation within 1e-3 | `per_op_finite_diff.test.ts` |
| AC6 | Backward IR is fully introspectable — `viz_backward_graph(loss)` produces valid DOT | `viz_smoke.test.ts` |
| AC7 | Closure `_ctx` field deleted from `TensorProxy`; no references remain in `browsergrad-jit` | grep check in CI |
| AC8 | Broadcasting un-projection correct for `ADD`/`MUL`/`DIV` with mixed shapes | `broadcast_grad.test.ts` |
| AC9 | `optimizer.step()` realizes backward subgraph correctly; loss decreases on 2-layer MLP over 100 steps | `mlp_training.test.ts` |
| AC10 | No `NotImplementedError` raised for any opcode used in the conformance suite | runtime error check |

---

## Test Strategy

### Unit (Vitest, no Pyodide)

- `vjp_structure.test.ts` — for each opcode, build a tiny forward graph, request gradient, assert the *exact* set of UOps produced (opcode names, shapes, dtypes). This is the cheapest regression guard.

### Integration (Vitest + Pyodide-in-Node)

- `symbolic_vs_closure_parity.test.ts` — runs every existing `browsergrad-grad` scenario through symbolic backward; numerical match within 1e-5.
- `per_op_finite_diff.test.ts` — opcode × small shape grid; central-difference numerical gradient compared to symbolic.
- `double_backward.test.ts` — second-order: Hessian-vector products on a 2-layer MLP.
- `broadcast_grad.test.ts` — all interesting broadcast shape patterns.
- `mlp_training.test.ts` — actual training loop, asserts loss decreases.
- `viz_smoke.test.ts` — `viz_backward_graph` produces parseable DOT.

### Per-opcode rule documentation

Each opcode's VJP rule lives in `vjp.py` with a docstring containing the math derivation. Course material in `craftingattention` can extract these to teach autodiff from this codebase directly.

---

## Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | `GATHER` backward requires `SCATTER_ADD` — a new opcode not in PRD-005. NumPy realizer doesn't have it. | High | Medium | Add `SCATTER_ADD` opcode + NumPy realizer (`np.add.at(out, (idx,), dy)`) in week 4 |
| R2 | Broadcasting un-projection has many edge cases (negative dims, mixed singleton positions). Easy to introduce shape bugs. | High | High | Property test: random shape pairs `(a_shape, b_shape)` → forward broadcast → backward → assert shape match. ~200 cases in CI. |
| R3 | `REDUCE(max)` backward needs to identify which elements achieved max. Tie-breaking convention may diverge from PyTorch. | Medium | Low | PyTorch distributes gradient equally among max-tied elements. Match by adding tie-counting in `WHERE` arg. |
| R4 | Memory: backward IR holds references to all forward UOps until realization. May 2× memory peak vs closure path. | Medium | Medium | Measure on 5-layer MLP; if regression > 2×, introduce on-the-fly realization of forward activations not used in backward (DCE pass). |
| R5 | Loss of debuggability — closure stack trace was easy to read in Python; IR walk is harder. | Medium | Low | Add `--debug-backward` flag that prints per-op grad shapes during the walk. |
| R6 | Double backward through `create_graph=True` exponentially blows up IR size for deep networks. | Low | Medium | Document the cost in PRIMER.md; PyTorch has the same limit. |
| R7 | `WHERE` condition input is bool — its "gradient" must be `None` (or zero) without raising. | Low | Low | Rule for `WHERE` returns `(da, db, None)`; reverse walk skips `None`. |

---

## Open Questions

1. **JVP support?** Some `torch.func` transforms benefit from forward-mode AD. Should this PRD register `JVP` rules in parallel with `VJP`? Resolution: defer to PRD-014; the symbol registry is extensible.

2. **Per-opcode rule organization.** Put all rules in one `vjp.py` (~600 lines) or one file per opcode? Resolution: one file (`vjp.py`) for v0; split if it exceeds 1200 lines.

3. **Should `create_graph=True` be opt-in or default-off when called from `torch.autograd.grad`?** PyTorch defaults to `create_graph=False`; we match.

4. **Custom `Function.apply`** — supporting user-defined backward via `torch.autograd.Function`. Resolution: out of scope for PRD-007; design separate small PRD if a course assignment needs it.

5. **Gradient name preservation** — Tensors have `.name` in some PyTorch versions for debugging. Backward IR currently loses names. Decision: add `arg={"src_op": forward_op}` on every VJP-produced UOp for `viz_backward_graph` readability.

---

## References

1. **JAX autodiff cookbook** — [JAX autodiff cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html). The canonical narrative for VJP/JVP design.

2. **JAX VJP implementation** — [`jax/interpreters/ad.py`](https://github.com/jax-ml/jax/blob/main/jax/interpreters/ad.py). Concrete reference for graph-to-graph backward construction.

3. **PyTorch autograd internals** — [PyTorch autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html); [PyTorch derivatives.yaml](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml). Per-op derivative declarations and codegen.

4. **PyTorch double backward** — [Higher-order gradients](https://pytorch.org/tutorials/intermediate/forward_ad_usage.html). Mechanism via `create_graph=True`.

5. **tinygrad Function class** — [`tinygrad/function.py`](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/function.py). Minimal symbolic backward in ~200 lines.

6. **Griewank & Walther, "Evaluating Derivatives"** — Section 3.2 on the chain rule for VJP. Mathematical foundation for the per-op rules.

7. **browsergrad-grad source** — `packages/browsergrad-grad/src/python/tensor.py:474` (`_build_ctx`), `:544` (`_matmul` closure). The closure pattern being replaced.

8. **PRD-005 IR definition** — opcode set this PRD writes VJP rules for.
