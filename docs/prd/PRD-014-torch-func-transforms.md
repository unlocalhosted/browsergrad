# PRD-014 — `torch.func` / `vmap` / `grad` / `jacrev`: JAX-Style Function Transforms on the IR

| Field | Value |
|---|---|
| **Status** | Draft v1 |
| **Author** | unlocalhosted maintainers |
| **Date** | 2026-05-26 |
| **PRD ID** | PRD-014 |
| **Phase** | P2 (Months 10–14 of the 14-month roadmap in PRD.md §6) |
| **Package** | `@unlocalhosted/browsergrad-jit` |
| **Depends on** | PRD-005 (IR + tracer + NumPy realizer), PRD-007 (symbolic VJP rules on the IR) |
| **Enables** | Per-sample gradients (Opacus-style private ML, influence functions); Hessian-vector products (second-order optimisers, SAM); meta-learning labs (MAML); Stanford CS336 assignment 4 (efficient LM training); fast.ai Ch 17 (autograd-from-scratch); PRD-012 megakernels see the post-`vmap` batched IR |
| **Companion docs** | [VISION.md](../../VISION.md) §4 Layer 3 · [PRD.md](../../PRD.md) §7 P2.4 |

---

## TL;DR

PRD-014 implements `torch.func` (formerly `functorch`) as **graph-to-graph function transforms over the IR from PRD-005**. `vmap`, `grad`, `jacrev`, `jacfwd`, `hessian`, and `functional_call` are not eager wrappers — they trace the user function once to obtain its UOp graph, then *rewrite the graph*: `vmap` adds a batched axis to every leaf and re-derives the shape/semantics of each opcode; `grad` invokes PRD-007's VJP machinery and extracts the gradient sub-DAG as the new output; `jacrev` does the same but seeds an identity matrix instead of `1.0`; `jacfwd` mirrors VJP with forward-mode JVP rules registered symmetrically in `jvp.py`; `hessian` is `jacfwd(jacrev(fn))`. Because every transform produces a *fresh IR graph*, transforms compose freely: `vmap(grad(fn))` is per-sample gradients (the Opacus use case, [arXiv:2109.12298](https://arxiv.org/abs/2109.12298)); `grad(vmap(fn))` is the gradient of a summed loss; `vmap(vmap(...))` adds two batch axes. Each result remains an IR graph that PRD-006's elementwise fusion, PRD-009's checkpointing, and PRD-012's megakernel codegen consume on equal footing with hand-written batched code. `functional_call` provides a stateless module application so transforms — which cannot capture `nn.Module.parameters()` as hidden state — work on PyTorch-shaped modules unchanged. Implementation budget: 8 weeks.

---

## Background

### Why function transforms have to be graph rewrites

PyTorch's classic `nn.Module.forward()` carries hidden state: the parameter tensors live as attributes, the autograd tape lives in `Tensor._ctx`, the RNG state is implicit. A naive `vmap` would have to clone the module, the tape, and the RNG per batch element — exponential complexity in transform depth. JAX rejected this design from day one ([Bradbury et al., *Composable transformations of Python+NumPy programs*, arXiv:1810.06367](https://arxiv.org/abs/1810.06367)): functions are pure, all state is an argument, and transforms operate on Jaxprs (JAX's IR). The composition `vmap(grad(fn))` is then *literally* graph-rewrite-of-graph-rewrite: `grad(fn)` produces a Jaxpr `g`, `vmap(g)` rewrites `g` into a batched Jaxpr `g'`. No closures, no recursion, no hidden state.

functorch ([He, Zou et al., *functorch: JAX-like composable function transforms for PyTorch*, arXiv:2207.06974](https://arxiv.org/abs/2207.06974)) ported the design to PyTorch by way of `__torch_dispatch__` interception and a parallel "FuncTorch" tape. Once stable, it was upstreamed as `torch.func` ([PyTorch torch.func docs](https://pytorch.org/docs/stable/func.html)). The lesson for us: even PyTorch — which started with the wrong substrate — needed an IR-shaped tape before it could ship composable transforms. We have PRD-005's UOp IR from day one of the JIT epoch, so the design is shorter for us than it was for functorch.

### What changed in PRD-005/PRD-007 that makes this tractable

- **PRD-005** turned every PyTorch op into a `UOp` constructor. A user function `fn(x, W)` calling `nn.Linear`-style code produces a deterministic UOp graph — a Jaxpr-equivalent — without any new tracing infrastructure. The shape and dtype of every UOp are known at graph-build time.
- **PRD-007** made backward a symbolic IR-to-IR transform. `loss.backward()` returns a graph whose roots are gradient UOps wired into the same DAG as the forward. That is exactly what `grad(fn)` needs to return as a callable.
- The composition rule then follows mechanically: `grad(fn)` traces `fn` to a graph `G`, runs PRD-007's reverse-topo VJP walk, and returns a new function whose body re-binds the input UOps and outputs the input-gradient UOps. `vmap(g)` runs over that graph the same way it runs over any other.

### What JAX's transforms actually do, opcode-by-opcode

`vmap` is "tracing in a non-standard interpretation." Instead of binding inputs to `BUFFER+LOAD` UOps of shape `s`, it binds them to `BUFFER+LOAD` UOps of shape `(B, *s)` and re-derives every downstream UOp's shape under the rule "the batch axis flows through pointwise; reductions over non-batch axes still happen; reductions over the batch axis are forbidden unless the user asked." Each opcode has a small **batching rule** ([JAX vmap batching rules](https://docs.jax.dev/en/latest/jax-101/03-vectorization.html)). For `ADD`, the rule is trivial: broadcast handles it. For `MATMUL`, the rule promotes to batched matmul. For `REDUCE`, the rule adjusts the axis index. For `RESHAPE` and `PERMUTE`, the rule shifts axes to leave the batch dim in place.

`grad(fn)` is "trace, then seed the backward graph with a scalar 1, then return the input gradient as the new output." `jacrev(fn)` differs only in seeding: instead of `dy = 1.0`, it seeds `dy = I` (identity matrix), giving one column of the Jacobian per row of `I` — implemented efficiently by `vmap` over the seed dimension ([JAX `jacrev` source](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py)).

`jacfwd(fn)` requires *forward-mode AD*. For each opcode, given the input UOps `x_i` and *their tangents* `dx_i`, the JVP rule emits a tangent UOp `dy` for the output: `MUL.jvp(a, b, da, db) = a*db + b*da`. This is the mathematical dual of VJP. PRD-007 declined to register JVP rules; PRD-014 registers them — same registry pattern, same one-file structure — so the symbolic-AD machinery is complete in both directions.

`hessian(fn)` is then literally `jacfwd(jacrev(fn))`. Composition. Free.

### Why per-sample gradients matter (the killer app)

In a standard PyTorch training loop the gradient of a minibatch is the *summed* per-sample gradient — exactly the gradient the optimiser wants. Several modern techniques want the *individual* per-sample gradients, not the sum:

1. **Differentially-private SGD** (Abadi et al., [arXiv:1607.00133](https://arxiv.org/abs/1607.00133); Opacus library, [opacus.ai](https://opacus.ai)). Each sample's gradient norm is clipped *before* aggregation. Without per-sample grads, the privacy guarantee evaporates.
2. **Influence functions** ([Koh & Liang, arXiv:1703.04730](https://arxiv.org/abs/1703.04730)) for "which training example influenced this prediction" — requires `∂loss_i / ∂θ` separately for each `i`.
3. **Sharpness-aware minimization** (SAM, [Foret et al., arXiv:2010.01412](https://arxiv.org/abs/2010.01412)) computes a gradient at the worst-case perturbation within an L2 ball — needs HVP, which composes as `vmap(grad(grad(...)))`.

The PyTorch-eager implementation of per-sample gradients reruns the forward N times per step (Opacus's "hooks" approach is a workaround). With `vmap(grad(fn))` over a batched input, it is **one** forward+backward pass over an IR rewritten with an extra leading batch axis — i.e., the same cost as a standard batched forward+backward, plus the per-sample dimension. This is the substantive performance reason `torch.func` exists; we get it for free once the transforms are IR rewrites and PRD-012's megakernel codegen sees the batched IR.

---

## User Stories

**U1 — Per-sample gradients in one line.** A student implementing Opacus-style differentially-private SGD writes:

```python
from browsergrad_jit.func import grad, vmap, functional_call

def compute_loss(params, x, y):
    logits = functional_call(model, params, (x,))
    return F.cross_entropy(logits, y)

per_sample_grad = vmap(grad(compute_loss), in_dims=(None, 0, 0))
grads = per_sample_grad(params, x_batch, y_batch)   # dict of per-sample tensors
```

Each value in `grads` has a leading batch dimension equal to `x_batch.shape[0]`. The student clips per-sample norms, sums, applies the Gaussian mechanism, and steps — full DP-SGD on browsergrad. Numerically identical (within 1e-4) to running `grad(compute_loss)` in a Python `for` loop over the batch.

**U2 — Hessian-vector product for SAM.** A user implementing sharpness-aware minimization needs `H @ v` where `H = ∇²_θ loss`. They write:

```python
from browsergrad_jit.func import grad, jvp

def loss_fn(params): return _loss(params, x, y)
g_fn = grad(loss_fn)
_, hvp = jvp(g_fn, (params,), (v,))   # H @ v in one trace
```

The HVP returns a structurally-identical pytree to `params`. Wall-clock is roughly 2× one backward pass, matching JAX's measured HVP cost.

**U3 — Full Jacobian of a small function.** A maintainer building a course example for Jacobian visualisation writes `J = jacrev(f)(x)` for `f: R^3 → R^3` and renders the 3×3 matrix. The result matches `torch.func.jacrev(f)(x)` to 1e-4. Switching to `jacfwd(f)(x)` produces the same matrix to 1e-5 via forward-mode AD — exercising the JVP registry independently of VJP.

**U4 — Stateless module application.** A user wants to vmap over a `nn.Module`. They cannot `vmap(model)` directly because `model` carries parameters as hidden state. They use `functional_call(model, params, args)` instead. `params` is a flat dict of `TensorProxy`s extracted via `dict(model.named_parameters())`. The result of `vmap(lambda p, x: functional_call(model, p, (x,)))(params, batched_x)` is identical to a standard batched forward pass — proving the transform sees the module as pure.

**U5 — Composed transforms, fused.** A user writes `vmap(grad(fn))` on a function whose body is a small MLP. The runtime traces, rewrites for vmap, rewrites for grad, hands the resulting batched-forward-and-backward IR to PRD-012's megakernel codegen. The megakernel runs the whole per-sample-grad computation in a single dispatch with workgroup-shared tiles. Wall-clock matches a hand-written batched gradient kernel within 10%.

---

## Goals and Non-Goals

### Goals

1. Ship `browsergrad_jit.func` with: `vmap`, `grad`, `jacrev`, `jacfwd`, `hessian`, `functional_call`, `vjp`, and `jvp` — eight callables matching PyTorch's `torch.func` signatures.
2. Implement `vmap` as an IR-rewrite pass: register a **batching rule** for every PRD-005 opcode; the rewrite walks the traced graph and produces a fresh batched graph.
3. Register forward-mode **JVP rules** in `jvp.py` symmetrically to PRD-007's `vjp.py`, for every PRD-005 opcode plus `SCATTER_ADD` if PRD-007 added it.
4. Implement `grad(fn)` as: trace → reverse-topo VJP (re-using PRD-007) → return new IR-backed callable whose output is the input gradient.
5. Implement `jacrev` and `jacfwd` as composed seeding strategies on top of `grad` / `jvp`.
6. Implement `hessian(fn) = jacfwd(jacrev(fn))` as a single line of composition.
7. Implement `functional_call(module, params, args)` so transforms can apply stateful modules without copying.
8. Composition: `vmap(grad(fn))`, `grad(vmap(fn))`, `jacrev(jacrev(fn))`, `vmap(vmap(fn))` all produce correct results without infinite recursion.
9. **Performance-by-default**: the IR produced by every transform is a normal UOp graph; PRD-006 fusion, PRD-009 checkpointing, and PRD-012 megakernels operate on it without special cases.
10. Numerical tolerance: every transform matches a reference (`torch.func` on the same input, or PyTorch eager + Python loop) within **1e-4** for fp32.

### Non-Goals

1. **Custom batching rules for user-defined ops.** PRD-014 covers every PRD-005 opcode; user-defined Functions (PRD-015 custom WGSL kernels) need to register their own batching rules — out of scope here.
2. **`torch.func.replace_all_batch_norm_modules_`.** BatchNorm in `vmap` requires per-sample running stats. PyTorch's helper rewrites BN modules to `GroupNorm` for compatibility. We surface the same helper but only as a thin shim; full per-sample BN with running stats is deferred to a future small PRD.
3. **Stateful RNG inside transformed functions.** JAX requires the user to thread PRNG keys explicitly. We match JAX semantics: `torch.func.vmap`'d functions must take RNG state as an argument (use `vmap(fn, in_dims=(..., 0))` with a stack of seeds). Implicit `torch.randn` inside a transformed function raises a clear error.
4. **`torch.func.linearize`** (just-in-time JVP). The mechanism is the same as `jvp` plus partial evaluation; we ship `jvp` and revisit `linearize` only if a curriculum chapter needs it.
5. **`make_fx`** / explicit FX graph extraction. The user already has the IR via `browsergrad_jit.jit.trace(fn)`; a separate `make_fx` API duplicates effort.
6. Higher-order derivatives beyond second order. The machinery supports them; explicit tests are not in scope.

---

## Architecture

The pipeline is four passes wrapped in three public callables. Every transform consumes a function or an IR graph and produces an IR graph.

```
user fn                                          new fn
   │                                                ▲
   ▼                                                │
1. Trace fn → UOp graph G   ──(reuse PRD-005)──┐    │
                                                ▼    │
2. Rewrite G under transform rules ─┬─ vmap ───┬──► G'
                                    ├─ grad ───┤
                                    ├─ jacrev ─┤
                                    ├─ jacfwd ─┘
                                                │
                                                ▼
3. Wrap G' in a new traceable callable ───────► new fn
                                                │
                                                ▼  (called by user)
4. Re-trace? No. G' is re-bound to fresh inputs and realised.
```

### 1. Tracing seam (`func/tracing.py`)

**File:** `packages/browsergrad-jit/src/python/func/tracing.py`

`trace(fn, example_inputs) -> TracedGraph` runs `fn` with `example_inputs` *as TensorProxies* (PRD-005's `BUFFER+LOAD` leaves). Every op inside `fn` builds UOps. We collect:

```python
@dataclass(frozen=True)
class TracedGraph:
    inputs: Tuple[UOp, ...]            # the LOAD UOps for example inputs
    outputs: Tuple[UOp, ...]            # output UOps of fn
    leaf_buffer_ids: Tuple[str, ...]    # parameter buffers reachable from outputs
    pytree_spec: PyTreeSpec             # output structure for re-packing
```

`pytree_spec` is the same flattening protocol JAX and `torch.func` use ([torch.utils._pytree](https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py)). Inputs and outputs are flattened to flat tuples of UOps; the spec lets us re-pack to dicts/lists/dataclasses on the way out.

`trace` is cache-keyed by `(fn_id, tuple(input_shape_dtype_signatures))` — same per-shape-signature caching as PRD-005's `nn.Module.forward()` tracing.

### 2. `vmap`: per-opcode batching rules (`func/vmap.py`)

**File:** `packages/browsergrad-jit/src/python/func/vmap.py`

`vmap(fn, in_dims=0, out_dims=0)` returns a new function `fn'` that, when called with batched inputs, behaves as if `fn` were called once per slice along `in_dims` and the results stacked along `out_dims`. Implementation:

```python
def vmap(fn, in_dims=0, out_dims=0):
    def fn_vmapped(*batched_args):
        # 1. Trace fn with sliced inputs (drop the batch dim per in_dims).
        unbatched_shapes = _drop_batch_dim(batched_args, in_dims)
        traced = trace(fn, unbatched_shapes)
        # 2. Walk traced.outputs back through the graph; apply batching rules.
        batched_graph = _apply_batching(traced, in_dims, out_dims, batch_size=B)
        # 3. Bind batched_args' UOps to batched_graph's inputs and return outputs.
        return _bind_and_unflatten(batched_graph, batched_args)
    return fn_vmapped
```

The heart is `_apply_batching`, which walks the traced UOp DAG and, for each node, looks up its **batching rule**:

```python
_BATCH_RULES: dict[str, BatchRule] = {}

def register_batch(op: str):
    def deco(fn): _BATCH_RULES[op] = fn; return fn
    return deco

@register_batch("ADD")
def _batch_add(node: UOp, inputs_bdim, batch_size):
    # All inputs already broadcast-compatible; just propagate the batch dim.
    out_bdim = _first_present(inputs_bdim)
    return UOp("ADD", node.inputs, _shape_with_batch(node, out_bdim, batch_size),
               node.dtype), out_bdim

@register_batch("MATMUL")
def _batch_matmul(node, inputs_bdim, B):
    a, b = node.inputs
    a_bd, b_bd = inputs_bdim
    # Case 1: both unbatched — unreachable inside vmap.
    # Case 2: one batched, one not — broadcast the unbatched one.
    # Case 3: both batched — batched matmul (NumPy/WGSL handle this natively).
    a = _move_batch_to_front(a, a_bd) if a_bd is not None else a
    b = _move_batch_to_front(b, b_bd) if b_bd is not None else b
    out_shape = _batched_matmul_shape(a.shape, b.shape)
    return UOp("MATMUL", (a, b), out_shape, node.dtype), 0

@register_batch("REDUCE")
def _batch_reduce(node, inputs_bdim, B):
    x_bd = inputs_bdim[0]
    if x_bd is None:
        return node, None
    # The user asked to reduce over a non-batch axis; shift the axis index.
    axis = node.arg["axis"]
    new_axis = axis + 1 if axis >= x_bd else axis
    new_arg = {**node.arg, "axis": new_axis}
    return UOp("REDUCE", node.inputs, _reduce_shape_with_batch(node, x_bd, new_axis),
               node.dtype, arg=new_arg), x_bd
```

The `inputs_bdim` argument is a tuple of "batch dim index or None per input." For each opcode we hand-write the rule that determines (1) the new output UOp, (2) where the batch dim ends up in the output. PRD-005's 19 opcodes get 19 batching rules; SCATTER_ADD (added in PRD-007 for GATHER backward) gets one too. JAX uses the same pattern with the same number of rules ([JAX `batching.py`](https://github.com/jax-ml/jax/blob/main/jax/interpreters/batching.py)).

**The `in_dims=None` case.** A user can vmap over only some arguments: `vmap(fn, in_dims=(0, None))(x_batched, params)` keeps `params` un-batched (broadcast across all batch elements). In the rule machinery, an unbatched input has `bdim = None` and the rule treats it as a constant that must be broadcast where it interacts with a batched input.

**Composition with `grad`.** When the input to `vmap` is itself an IR graph from `grad(fn)`, the rules apply identically — that graph contains the same UOps as any other. The composition is therefore transparent: `vmap(grad(fn))` traces `grad(fn)` once to get a graph, then runs the batching rewrite on it. No recursion into `grad` itself.

### 3. `grad` and `vjp`: re-using PRD-007 (`func/grad.py`)

**File:** `packages/browsergrad-jit/src/python/func/grad.py`

`grad(fn, argnums=0)` returns a function that computes `∂fn(*args) / ∂args[argnums]`:

```python
def grad(fn, argnums=0, has_aux=False):
    def fn_grad(*args):
        traced = trace(fn, args)
        assert traced.outputs[0].shape == (), "grad requires scalar output"
        # Seed the upstream gradient with CONST(1.0).
        seed = UOp("CONST", (), (), traced.outputs[0].dtype, arg={"value": 1.0})
        # Re-use PRD-007's reverse-topo walk; obtain a dict UOp → gradient UOp.
        grads = autograd.symbolic_backward(traced.outputs[0], seed)
        # Pick the input(s) at argnums; pytree-unflatten to user shape.
        wanted_inputs = _select_inputs_at_argnums(traced, argnums)
        out_uops = tuple(grads[u] for u in wanted_inputs)
        return _bind_and_unflatten(out_uops, args, pytree_spec=traced.input_spec_at(argnums))
    return fn_grad
```

`autograd.symbolic_backward` is PRD-007's existing reverse-topological VJP walk lifted to take an explicit seed UOp. The function returns the same `grads: dict[UOp, UOp]` map PRD-007 already builds — we just don't write it back to `.grad` slots. Instead we read the gradient at the input UOps and return them as the new function's outputs.

`vjp(fn, *args)` returns `(out, vjp_fn)` where `vjp_fn(cotangent)` computes the VJP. Same machinery, but the seed is left as a parameter:

```python
def vjp(fn, *args):
    traced = trace(fn, args)
    def vjp_fn(cotangent):
        seed_uop = _wrap_as_uop(cotangent, shape=traced.outputs[0].shape)
        grads = autograd.symbolic_backward(traced.outputs[0], seed_uop)
        return tuple(grads[u] for u in traced.inputs)
    return _bind_and_unflatten(traced.outputs, args), vjp_fn
```

### 4. `jvp` and JVP rule registry (`func/jvp.py`)

**File:** `packages/browsergrad-jit/src/python/func/jvp.py`

JVP (forward-mode AD) is the dual of VJP. For `y = op(x_1, ..., x_n)` with tangents `dx_i`, the JVP rule emits `dy`. Same registry pattern as PRD-007's `vjp.py`:

```python
_JVP_RULES: dict[str, JVPRule] = {}

def register_jvp(op: str):
    def deco(fn): _JVP_RULES[op] = fn; return fn
    return deco

@register_jvp("ADD")
def _jvp_add(output, inputs, tangents):
    da, db = tangents
    return UOp("ADD", (da, db), output.shape, output.dtype)

@register_jvp("MUL")
def _jvp_mul(output, inputs, tangents):
    a, b = inputs
    da, db = tangents
    return UOp("ADD",
               (UOp("MUL", (da, b), output.shape, output.dtype),
                UOp("MUL", (a, db), output.shape, output.dtype)),
               output.shape, output.dtype)

@register_jvp("MATMUL")
def _jvp_matmul(output, inputs, tangents):
    a, b = inputs
    da, db = tangents
    return UOp("ADD",
               (UOp("MATMUL", (da, b), output.shape, output.dtype),
                UOp("MATMUL", (a, db), output.shape, output.dtype)),
               output.shape, output.dtype)

@register_jvp("EXP")
def _jvp_exp(output, inputs, tangents):
    # d(exp(x)) = exp(x) * dx; output IS exp(x) — re-use it.
    (dx,) = tangents
    return UOp("MUL", (output, dx), output.shape, output.dtype)
```

Each rule emits a small subgraph of UOps representing the tangent. The forward function `y = fn(x)` traced as a UOp graph, walked in **topological order** (forward direction), with tangents accumulated at each node, yields the JVP at the output. This is the mirror of PRD-007's reverse walk:

```python
def jvp(fn, primals, tangents):
    traced = trace(fn, primals)
    # tangents[i] becomes the tangent UOp at traced.inputs[i].
    tangent_map: dict[UOp, UOp] = dict(zip(traced.inputs, _flatten(tangents)))
    for node in toposort(traced.outputs):
        if node in tangent_map: continue
        if node.op in ("BUFFER", "CONST", "LOAD"): continue
        rule = _JVP_RULES.get(node.op)
        if rule is None: raise NotImplementedError(f"No JVP rule for {node.op!r}")
        node_tangents = tuple(tangent_map.get(inp, _zeros_like_uop(inp)) for inp in node.inputs)
        tangent_map[node] = rule(node, node.inputs, node_tangents)
    primal_out = traced.outputs[0]
    tangent_out = tangent_map[primal_out]
    return primal_out, tangent_out
```

JVP rules cover the same 19 (+1) opcodes VJP rules cover. The two registries are independent; `WHERE`'s JVP is symmetric in `(a, b)` like its VJP, but `MAX_REDUCE`'s JVP needs the index-of-max from the forward — the same trick VJP uses, just on the other side.

### 5. `jacrev`, `jacfwd`, `hessian`

**File:** `packages/browsergrad-jit/src/python/func/jacobians.py`

`jacrev` is `grad` with a non-scalar seed. For `f: R^n → R^m` we seed once per output dim and stack:

```python
def jacrev(fn, argnums=0):
    def fn_jacrev(*args):
        traced = trace(fn, args)
        m = _product(traced.outputs[0].shape)
        # Efficient implementation: vmap over an identity-matrix seed.
        def vjp_at_row(seed_row):
            grads = autograd.symbolic_backward(traced.outputs[0], seed_row)
            return tuple(grads[u] for u in _select(traced.inputs, argnums))
        identity = UOp("EYE", (), (m, m), traced.outputs[0].dtype, arg={"m": m})
        # vmap over the leading axis of identity:
        return vmap(vjp_at_row, in_dims=0)(identity)
    return fn_jacrev
```

`vmap` over the seed dimension is *exactly* what makes `jacrev` efficient — one batched backward pass instead of `m` sequential ones. This is the same optimisation [JAX's `jacrev`](https://docs.jax.dev/en/latest/_autosummary/jax.jacrev.html) uses.

`jacfwd` is the dual:

```python
def jacfwd(fn, argnums=0):
    def fn_jacfwd(*args):
        traced = trace(fn, args)
        n = _product(_select(traced.inputs, argnums)[0].shape)
        identity = UOp("EYE", (), (n, n), traced.inputs[0].dtype, arg={"m": n})
        # vmap over the leading axis of the input tangent identity:
        def jvp_at_col(seed_col):
            _, tangent = jvp(fn, args, _tangents_with(seed_col, argnums))
            return tangent
        return vmap(jvp_at_col, in_dims=0)(identity)
    return fn_jacfwd
```

`hessian` is one line:

```python
def hessian(fn, argnums=0):
    return jacfwd(jacrev(fn, argnums=argnums), argnums=argnums)
```

The choice of `jacfwd ∘ jacrev` rather than `jacrev ∘ jacrev` is JAX's standard recipe — it minimises memory by alternating modes ([JAX advanced autodiff](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html#hessians)). The composition works because each transform produces an IR graph that the next transform consumes.

We add one new opcode for the seed: `EYE` (identity-matrix constant). Its NumPy realiser is `np.eye(m)`; its WGSL realiser is a small kernel. The opcode is also useful for the `eye()` factory function exposed by PRD-005.

### 6. `functional_call`: stateless module application

**File:** `packages/browsergrad-jit/src/python/func/functional_call.py`

`functional_call(module, params_dict, args)` runs `module.forward(*args)` but with parameters bound from `params_dict` instead of from `module._parameters`. Implementation: a context manager temporarily swaps the module's `_parameters` and `_buffers` dicts:

```python
def functional_call(module, parameter_and_buffer_dicts, args, kwargs=None, *, tie_weights=True):
    kwargs = kwargs or {}
    flat = _flatten_dicts(parameter_and_buffer_dicts)  # {param_name: TensorProxy}
    with _override_module_state(module, flat):
        return module(*args, **kwargs)

@contextmanager
def _override_module_state(module, overrides):
    saved = {}
    for name, value in overrides.items():
        parent, leaf = _resolve_dotted(module, name)
        saved[name] = parent._parameters.get(leaf) or parent._buffers.get(leaf)
        parent._parameters[leaf] = value if value.requires_grad else None
        parent._buffers[leaf] = value if not value.requires_grad else None
    try:
        yield
    finally:
        for name, original in saved.items():
            parent, leaf = _resolve_dotted(module, name)
            if original is None:
                parent._parameters.pop(leaf, None); parent._buffers.pop(leaf, None)
            elif original.requires_grad:
                parent._parameters[leaf] = original
            else:
                parent._buffers[leaf] = original
```

This matches PyTorch's `torch.func.functional_call` semantics ([torch.func.functional_call docs](https://pytorch.org/docs/stable/generated/torch.func.functional_call.html)). The key property for transforms: with `functional_call`, the parameter UOps become explicit inputs to the traced graph, so `vmap` / `grad` see them as graph inputs and rewrite them like any other UOp.

`dict(model.named_parameters())` returns the flat dict the user passes in. For `vmap`-over-parameters (the per-sample case), the user constructs a `params` dict whose values have a leading batch dim and passes `in_dims=(None, 0, 0)` to vmap; the batching rules see the parameter UOps as batched leaves and propagate correctly.

### 7. Composition machinery

The composition `T1(T2(fn))` works because each transform returns a *function* whose body, when traced, produces a *graph* that includes the inner transform's rewrites already baked in. Concretely:

- `inner = grad(fn)` returns a Python function whose call signature is `inner(*args) → grad_tensor`.
- `outer = vmap(inner)` calls `trace(inner, batched_args)` — re-tracing `inner` runs `grad`'s rewrite logic *again*, but only to produce the IR graph. The output of that trace is a fresh UOp graph containing both the forward-and-VJP UOps that `grad` produced. `vmap` then rewrites that graph.

The only correctness condition is that **each transform fully consumes the graph it sees**. We don't leave half-rewritten UOps. PRD-014's `_apply_batching` and `symbolic_backward` both produce closed IR graphs with no "in-progress" tags. Composition is then associative the way function composition is.

**Avoiding infinite recursion.** A naive implementation might `vmap` call `grad` call `vmap` recursively without ever terminating. We avoid this with two invariants:

1. Every transform calls `trace(fn, args)` exactly once per call. `trace` does not call into transforms; it just builds a UOp graph from primitive ops.
2. The output of every transform is a UOp graph, not a Python function that re-invokes transforms.

So `vmap(grad(fn))` traces `grad(fn)` once (which itself traces `fn` once + runs VJP + returns a graph), then rewrites that graph for batching — two traces total, never recursive.

### 8. Integration with PRD-006 / PRD-009 / PRD-012

The IR graph produced by every transform looks identical to a hand-written IR graph. Downstream passes don't need to know a transform happened:

- **PRD-006 (elementwise / softmax / layernorm fusion)** sees ordinary UOps and fuses them. A `vmap`'d softmax becomes a per-row softmax over the (batched) input — same fusion pattern as PRD-006 already handles.
- **PRD-009 (gradient checkpointing)** can wrap a transformed function: `checkpoint(grad(fn), x)` rewrites the gradient subgraph the same way it rewrites a forward subgraph. The `CheckpointRegion` machinery is opcode-agnostic.
- **PRD-012 (megakernels)** sees the post-vmap IR with a leading batch axis on every tensor and emits batched WGSL. The Flash-Attention-v2-style attention kernel already iterates over a `(B, H, S, D)` layout; `vmap` adds another batch axis, which the cost model treats as a fourth iteration dimension. As long as the tile size budget holds, megakernels for `vmap(grad(fn))` are the same kernels as for `grad(fn)` with a wider batch.

The performance-by-default property follows mechanically: there is no special path. The transforms produce IR; the IR is fused, checkpointed, megakerneled.

---

## API Surface

```python
# packages/browsergrad-jit/src/python/func/__init__.py

# Function transforms (the public eight)
from browsergrad_jit.func import (
    vmap,                # batched / vectorized function
    grad,                # gradient as a function (scalar output required)
    vjp,                 # (out, vjp_fn) pair, JAX-style
    jvp,                 # forward-mode AD: (out, tangent) pair
    jacrev,              # Jacobian via reverse mode
    jacfwd,              # Jacobian via forward mode
    hessian,             # = jacfwd(jacrev(fn))
    functional_call,     # stateless module application
)

# Torch-compat shim
torch.func.vmap = vmap                  # all eight re-exported
torch.func.grad = grad
torch.func.functional_call = functional_call
# ... etc
```

Example end-to-end (per-sample gradients):

```python
import browsergrad_jit as torch
import browsergrad_jit.nn as nn
import browsergrad_jit.nn.functional as F
from browsergrad_jit.func import grad, vmap, functional_call

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
params = dict(model.named_parameters())  # flat dict of TensorProxy

def loss_fn(params, x, y):
    logits = functional_call(model, params, (x,))
    return F.cross_entropy(logits, y)

per_sample_grad = vmap(grad(loss_fn), in_dims=(None, 0, 0))

x = torch.randn(64, 784)
y = torch.randint(0, 10, (64,))
grads = per_sample_grad(params, x, y)
# grads["0.weight"].shape == (64, 256, 784)  ← per-sample gradients
```

Debug / introspection:

```python
import browsergrad_jit.func as bf
bf.list_batching_rules()             # 20 entries, one per opcode
bf.list_jvp_rules()                  # 20 entries
bf.viz_transformed_graph(grad_fn, *example_args)  # Graphviz DOT
```

---

## Implementation Plan

8 weeks. Each week ships behind a feature flag until week 8, where everything goes public under `browsergrad_jit.func`.

### Week 1 — Tracing seam + pytree

- Create `packages/browsergrad-jit/src/python/func/` directory tree.
- Implement `tracing.py` with `trace(fn, example_inputs) → TracedGraph`; cache keyed on shape-dtype signatures (same scheme as PRD-005).
- Port `torch.utils._pytree` (BSD-licensed; ~600 LOC) into `func/_pytree.py` with attribution.
- Unit tests: trace a 3-layer MLP; assert input/output flattening round-trips through pytree.

### Week 2 — JVP rule registry + `jvp(fn, primals, tangents)`

- Create `func/jvp.py` with `_JVP_RULES`, `register_jvp`, `jvp(fn, primals, tangents)`.
- Implement JVP rules for all 19 PRD-005 opcodes + SCATTER_ADD if present. One rule per opcode, ~5 LOC each.
- Unit tests: for every opcode, finite-difference check `(fn(x+ε·dx) - fn(x)) / ε ≈ jvp(fn, x, dx)` to 1e-3.
- Cross-check: `jvp` of a 2-layer MLP matches PRD-007's `vjp` adjoint via the standard `<v, J·u> = <J^T·v, u>` identity.

### Week 3 — `grad` and `vjp` re-using PRD-007

- Create `func/grad.py` implementing `grad`, `vjp` on top of PRD-007's `autograd.symbolic_backward` lifted to take an explicit seed UOp.
- Argnums handling: `grad(fn, argnums=(0, 2))` returns gradients wrt multiple inputs as a tuple.
- `has_aux=True` flag: function returns `(scalar, aux)`; gradient computed wrt scalar, aux returned unchanged.
- Integration test: `grad(loss_fn)(params, x, y)` matches `loss_fn(params, x, y).backward(); params.grad` to 1e-4 on a 2-layer MLP.

### Week 4 — Batching rule registry + `vmap` core

- Create `func/vmap.py` with `_BATCH_RULES`, `register_batch`, `_apply_batching`.
- Implement batching rules for all 19 PRD-005 opcodes + SCATTER_ADD.
- Handle `in_dims` as int, tuple, or pytree; handle `out_dims` symmetrically.
- Unit tests: for each opcode, `vmap` over a 4-element batch produces the same result as a Python for-loop with stacking (within 1e-6).

### Week 5 — `jacrev`, `jacfwd`, `hessian` + `EYE` opcode

- Add `EYE` as a 21st (or 22nd) PRD-005 opcode; NumPy realiser is `np.eye`; ship a stub WGSL realiser for week 5 (replaced by a real fused kernel in PRD-012).
- Implement `jacrev`, `jacfwd`, `hessian` in `func/jacobians.py` using `vmap` over identity seeds.
- Integration tests: `jacrev(f)(x)` and `jacfwd(f)(x)` match each other to 1e-5 for 5 representative functions; `hessian(f)(x)` is symmetric to 1e-5.
- Conformance test: results match `torch.func.jacrev` from real PyTorch 2.12.0 on 3 fixture cases.

### Week 6 — `functional_call` + module integration

- Implement `functional_call` in `func/functional_call.py` with the `_override_module_state` context manager.
- Validate against PyTorch: `functional_call(model, params, (x,))` produces same output as `model(x)` when `params = dict(model.named_parameters())`.
- Per-sample gradient end-to-end test: `vmap(grad(loss_fn), in_dims=(None, 0, 0))` on a 64-sample batch matches a Python for-loop computing 64 individual gradients within 1e-4.

### Week 7 — Composition + fusion integration

- Test composition matrix: `vmap(grad)`, `grad(vmap)`, `jacrev(jacrev)`, `vmap(vmap)`, `vmap(hessian)` — all produce correct results, no infinite recursion.
- Verify PRD-006 fusion fires on `vmap(grad(fn))` output IR (debug log shows fused kernels).
- Verify PRD-009 `checkpoint(grad(fn), x)` correctly rewrites the gradient subgraph.
- Verify PRD-012 megakernel codegen emits a single dispatch for `vmap(grad(small_mlp))`.
- HVP test: `jvp(grad(loss_fn), (params,), (v,))` returns `H @ v` matching numerical finite-difference of grad to 1e-3.

### Week 8 — Conformance, docs, ship

- Run full 234 integration tests with `func` available; no regressions in the non-transformed path.
- Run new conformance fixtures: 5 transform-specific fixtures generated from real `torch.func` (PyTorch 2.12.0) — `vmap_softmax.json`, `grad_mlp.json`, `jacrev_attention.json`, `hessian_quadratic.json`, `vmap_grad_per_sample.json`.
- Write `docs/guides/torch-func-transforms.md` linking to the JAX cookbook and the functorch paper for theoretical background.
- Publish `@unlocalhosted/browsergrad-jit@0.5.0` with `browsergrad_jit.func` public.

---

## Acceptance Criteria

| # | Criterion | Measurement |
|---|---|---|
| AC1 | `vmap(fn)(batched_x)` matches a manual Python loop within 1e-4 across 5 representative `fn`s (linear, softmax, attention, recurrent step, custom) | `vmap_conformance.test.ts` |
| AC2 | `grad(fn)(x)` matches `fn(x).backward(); x.grad` within 1e-5 | `grad_parity.test.ts` |
| AC3 | `jacrev(f)(x)` matches `torch.func.jacrev(f)(x)` from real PyTorch 2.12.0 within 1e-4 on 3 fixtures | `jacobian_conformance.test.ts` |
| AC4 | `jacfwd(f)(x) == jacrev(f)(x)` within 1e-5 (forward and reverse-mode agree) | `jacfwd_vs_jacrev.test.ts` |
| AC5 | `hessian(f)(x)` is symmetric to 1e-5 and matches numerical finite-difference of `grad` to 1e-3 | `hessian_symmetry.test.ts` |
| AC6 | `vmap(grad(loss_fn), in_dims=(None,0,0))` per-sample grads match a Python for-loop's 64 grads within 1e-4 | `per_sample_grad.test.ts` |
| AC7 | `functional_call(model, params, args)` equals `model(args)` when `params = dict(model.named_parameters())` within bitwise tolerance | `functional_call.test.ts` |
| AC8 | Composition `vmap(vmap(fn))`, `grad(jacrev(fn))`, `vmap(hessian(fn))` all run without infinite recursion and produce results to 1e-4 | `composition.test.ts` |
| AC9 | JVP rules registered for all 19 PRD-005 opcodes plus EYE plus SCATTER_ADD | `jvp.list_registered()` returns 21 entries |
| AC10 | Batching rules registered for all 19 PRD-005 opcodes plus EYE plus SCATTER_ADD | `vmap.list_registered()` returns 21 entries |
| AC11 | PRD-006 elementwise fusion fires on `vmap(grad(fn))` output IR (debug log assertion) | `fusion_interop.test.ts` |
| AC12 | PRD-009 `checkpoint(grad(fn), x)` reduces peak memory ≥30% vs unwrapped | `checkpoint_grad.test.ts` |
| AC13 | PRD-012 megakernel codegen emits ≤2 dispatches for `vmap(grad(small_mlp))` | `megakernel_interop.test.ts` |
| AC14 | All 234 existing integration tests still pass with `func` available | full suite green |
| AC15 | `torch.func.vmap` / `.grad` / `.jacrev` etc. accessible via the torch-alias shim | `torch_alias.test.ts` |

---

## Test Strategy

### Unit (Vitest, no Pyodide)

- `pytree_unit.test.ts` — pytree flatten/unflatten round-trips for dict, list, tuple, dataclass, nested combinations.
- `jvp_rule_finite_diff.test.ts` — per-opcode finite-difference check of the JVP rule output.
- `batch_rule_unit.test.ts` — for each opcode, a small synthetic UOp graph batched along axis 0 produces the expected output UOp structure.
- `composition_unit.test.ts` — assert each composition produces a closed IR graph (no dangling rule applications).

### Integration (Vitest + Pyodide-in-Node)

- `vmap_conformance.test.ts` — 5 representative functions.
- `grad_parity.test.ts` — vs PRD-007 backward on 10 model shapes.
- `jacobian_conformance.test.ts` — vs `torch.func.jacrev` from PyTorch 2.12.0.
- `hessian_symmetry.test.ts` — quadratic forms, MLPs, attention blocks.
- `per_sample_grad.test.ts` — DP-SGD-style per-sample gradient check on MNIST mini-batch.
- `functional_call.test.ts` — equivalence to direct module call across 8 module types.
- `composition.test.ts` — full matrix of `T1 ∘ T2` for `T_i ∈ {vmap, grad, jacrev}`.
- `fusion_interop.test.ts` — assert PRD-006 fusion still fires post-transform.
- `checkpoint_grad.test.ts` — PRD-009 + PRD-014 interaction.
- `megakernel_interop.test.ts` — PRD-012 dispatch count for transformed graphs.
- `torch_alias.test.ts` — `import torch; torch.func.grad(fn)` works.

### Conformance (PyTorch oracle)

Five new fixtures generated from real `torch 2.12.0`:
- `vmap_softmax.json` — `vmap(F.softmax, in_dims=0)` on `(8, 32)` input.
- `grad_mlp.json` — `grad(loss_fn)` on a 2-layer MLP, asserts per-parameter gradient match.
- `jacrev_attention.json` — `jacrev` of a small attention block wrt Q.
- `hessian_quadratic.json` — Hessian of `f(x) = x^T A x` is `A + A^T`; exact match expected.
- `vmap_grad_per_sample.json` — `vmap(grad(loss_fn))` per-sample grads on 16-sample batch.

### Wall-time benchmarks (CI-tracked, not blocking)

- `per_sample_grad.bench.ts` — `vmap(grad)` vs Python-loop baseline; expect 10× speedup for batch 64.
- `hvp.bench.ts` — HVP cost vs two-pass backward; expect within 2.5× of single backward.

---

## Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | A PRD-005 opcode is missing a batching rule, breaking some `vmap` compositions silently | High | High | CI grep enforces: `len(_BATCH_RULES) >= len(_OPCODES)`; missing rules raise `NotImplementedError` at trace time, never at run time |
| R2 | JVP rules for `REDUCE(max)` / `WHERE` differ subtly from PyTorch's tie-breaking | Medium | Medium | Match PyTorch's "distribute gradient equally among tied indices" rule; add tie-breaking test case explicitly |
| R3 | Composition `vmap(vmap(...))` produces correct results but exponential graph size for deep nests | Low | Medium | Document that vmap nests deeper than 3 are pedagogical; flag in PRIMER.md with the same caveat JAX uses |
| R4 | `functional_call` parameter override is not thread-safe (Pyodide GIL covers single-thread, but Web Worker parallelism would break) | Low | Medium | Document as single-session; same caveat as PRD-005's `BufferTable` |
| R5 | `jacrev`'s vmap-over-identity-seed produces an IR graph too large for PRD-012's pattern recogniser to match | Medium | Medium | PRD-012 already supports unmatched groups falling back to PRD-006 fusion; verify path; document the perf cliff for huge Jacobians |
| R6 | `vmap` over a function containing data-dependent control flow (`if` on tensor value) fails like JAX | Medium | Low | Same constraint JAX imposes; documented; raise a clear error pointing at the JAX FAQ |
| R7 | Trace cache miss on every transform call because `trace(grad(fn))` re-traces `fn` inside `grad` | Medium | Medium | Cache key includes `(fn_id, args_signature, transform_chain)`; transforms register their identity in the cache key |
| R8 | RNG inside transformed functions breaks deterministic replay (the JAX/torch.func tension) | Medium | Medium | Mirror JAX: raise `RngInTransformError` when an implicit RNG op (`torch.randn` without explicit seed) is traced inside a transform |
| R9 | `EYE` opcode not yet supported by PRD-012's megakernel codegen — Jacobians stay on the PRD-006 fast path | Low | Low | Acceptable for v0; PRD-012 follow-up adds `EYE` to the fusible set |
| R10 | Per-sample gradient memory blows up on large models (batch × params bytes) | High | Medium | Document the trade-off; recommend `checkpoint(grad(fn), x)` for large models; pair the example with the gradient-checkpointing guide |
| R11 | PyTorch ships a torch.func change in 2.13+ that breaks our shim | Low | Low | Pin torch.func conformance to PyTorch 2.12.0; update on each major torch release |
| R12 | `BatchNorm` running stats break under vmap because per-sample stats overwrite each other | Medium | Medium | Surface `replace_all_batch_norm_modules_` helper as a thin shim that rewrites BN → GroupNorm; loud error if BN is detected inside a vmap'd function |

---

## Open Questions

1. **Should `vmap` randomness=`"different"` semantics ship in v0?** PyTorch's `vmap(fn, randomness="different")` allows per-sample independent random draws. JAX requires explicit per-sample PRNG keys. We default to JAX semantics (explicit keys) and document the PyTorch-style mode as a future addition. Resolution requested before week 4.

2. **`grad` argnums on a single tensor argument: int or 0-tuple?** PyTorch's `torch.func.grad(fn, argnums=0)` returns a single tensor; `argnums=(0,)` returns a 1-tuple. We match PyTorch exactly. Verify with conformance fixture.

3. **`functional_call` with tied weights.** PyTorch's `tie_weights=True` default identifies shared parameters by name and ensures gradients accumulate correctly. We ship the default; verify on a transformer with tied input embedding + output projection.

4. **Eager vs lazy execution of transformed functions.** When `browsergrad_jit.use_eager(True)` is set (PRD-005 debug fallback), do transforms execute eagerly (one slice at a time for vmap) or still produce IR? Resolution: transforms always produce IR; the IR is realized immediately under eager mode, which is acceptable since debugging.

5. **`hessian` for non-scalar functions.** `hessian(f)` for `f: R^n → R^m` returns a 3-tensor `(m, n, n)`. JAX supports this; PyTorch's `torch.func.hessian` raises. We match PyTorch (require scalar output) and point users to `jacfwd(jacrev(f))` for the general case.

6. **Should `jvp` rules live in `vjp.py` or a separate `jvp.py`?** Decision: separate file (`func/jvp.py`) so PRD-007's `vjp.py` stays focused. Cross-file refs are fine; the test harness asserts both registries are complete.

7. **Custom user-defined function support (`torch.autograd.Function.apply`).** PyTorch users can register custom forward+backward via `Function.apply` and expect `vmap`/`grad` to work on them. PRD-014 ships *no* custom-function support; users must register both a VJP rule (PRD-007) and a JVP rule (this PRD) for their op. A small future PRD wraps that in an ergonomic API.

---

## References

1. **JAX core paper** — Bradbury, J. et al., *Composable transformations of Python+NumPy programs*, [arXiv:1810.06367](https://arxiv.org/abs/1810.06367) (2018). The original statement of "transforms are graph rewrites over a pure IR."

2. **functorch paper** — He, R., Zou, R. et al., *functorch: JAX-like composable function transforms for PyTorch*, [arXiv:2207.06974](https://arxiv.org/abs/2207.06974) (2022). The PyTorch-flavoured port; documents the design tensions we avoid by starting from the IR.

3. **PyTorch `torch.func` docs** — [torch.func reference](https://pytorch.org/docs/stable/func.html); per-function pages for [vmap](https://pytorch.org/docs/stable/generated/torch.func.vmap.html), [grad](https://pytorch.org/docs/stable/generated/torch.func.grad.html), [jacrev](https://pytorch.org/docs/stable/generated/torch.func.jacrev.html), [jacfwd](https://pytorch.org/docs/stable/generated/torch.func.jacfwd.html), [hessian](https://pytorch.org/docs/stable/generated/torch.func.hessian.html), [functional_call](https://pytorch.org/docs/stable/generated/torch.func.functional_call.html). API surface we shadow.

4. **JAX vmap / grad / jacrev docs** — [jax.vmap](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html); [jax.grad](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html); [jax.jacrev](https://docs.jax.dev/en/latest/_autosummary/jax.jacrev.html); [vectorisation tutorial](https://docs.jax.dev/en/latest/jax-101/03-vectorization.html). Reference semantics.

5. **JAX autodiff cookbook** — [autodiff cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html). The narrative for why `hessian = jacfwd ∘ jacrev` and when to choose forward vs reverse mode.

6. **JAX `batching.py`** — [batching interpreter source](https://github.com/jax-ml/jax/blob/main/jax/interpreters/batching.py). Direct reference for the per-primitive batching rule pattern.

7. **JAX `ad.py`** — [AD interpreter source](https://github.com/jax-ml/jax/blob/main/jax/interpreters/ad.py). Forward-mode AD (JVP) implementation we mirror.

8. **PyTorch _pytree** — [torch.utils._pytree source](https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py). BSD-licensed; we vendor a subset.

9. **Opacus / DP-SGD** — Abadi, M. et al., *Deep Learning with Differential Privacy*, [arXiv:1607.00133](https://arxiv.org/abs/1607.00133) (2016); [Opacus library](https://opacus.ai); [Opacus per-sample gradients in PyTorch, arXiv:2109.12298](https://arxiv.org/abs/2109.12298). Primary use case for `vmap(grad(...))`.

10. **Sharpness-aware minimization** — Foret, P. et al., *Sharpness-Aware Minimization for Efficiently Improving Generalization*, [arXiv:2010.01412](https://arxiv.org/abs/2010.01412) (2020). HVP / second-order use case.

11. **Influence functions** — Koh, P. W. & Liang, P., *Understanding Black-box Predictions via Influence Functions*, [arXiv:1703.04730](https://arxiv.org/abs/1703.04730) (2017). Per-sample gradient use case for interpretability.

12. **Griewank & Walther, *Evaluating Derivatives*** (2008), SIAM, Chapter 3 (forward-mode AD), Chapter 4 (reverse-mode AD). Mathematical foundation for both rule registries.

13. **PRD-005 (`browsergrad-jit` MVP)** — `docs/prd/PRD-005-jit-foundation.md`. The IR substrate every transform consumes and produces.

14. **PRD-007 (Symbolic backward)** — `docs/prd/PRD-007-symbolic-backward.md`. `grad` re-uses its reverse-topo VJP walk; `jvp.py` registers the symmetric forward-mode rules.

15. **PRD-009 (Gradient checkpointing)** — `docs/prd/PRD-009-gradient-checkpointing.md`. Wraps the gradient subgraph that `grad(fn)` produces — same `CheckpointRegion` machinery, opcode-agnostic.

16. **PRD-012 (Megakernel codegen)** — `docs/prd/PRD-012-megakernel-codegen.md`. Consumes the IR `vmap` produces with no special-casing; the batched dimension is just another iteration axis to the cost model.

17. **Stanford CS336 Assignment 4** — *Efficient LM Training* (course-internal; covers per-sample gradient norms, gradient accumulation, mixed precision). Primary curriculum target.

18. **fast.ai Part 2 Chapter 17** — *Autograd from scratch* ([course.fast.ai/Lessons/part2.html](https://course.fast.ai/Lessons/part2.html)). Teaches function-transform style as the natural endpoint of building autograd; the chapter becomes a lab once `browsergrad_jit.func` ships.
