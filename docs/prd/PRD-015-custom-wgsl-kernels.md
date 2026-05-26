# PRD-015 — Custom WGSL Kernels from Python

| Field | Value |
|---|---|
| **Status** | Draft v1 |
| **Author** | unlocalhosted maintainers |
| **Date** | 2026-05-26 |
| **PRD ID** | PRD-015 |
| **Phase** | P2 (Months 10–14 of the 14-month roadmap in PRD.md §6) |
| **Package** | `@unlocalhosted/browsergrad-jit`, `@unlocalhosted/browsergrad-kernels` |
| **Depends on** | PRD-005 (IR), PRD-007 (symbolic backward — VJP registration surface), PRD-008 (OPFS pipeline cache) |
| **Companion docs** | [VISION.md](../../VISION.md) §4 Layer 3 · [PRD.md](../../PRD.md) §7 P2.5 |
| **Adjacent PRDs** | PRD-006 (fusion — custom kernels are fusion barriers by default), PRD-012 (megakernel codegen — fusion hint surface shared) |

---

## TL;DR

Fast.ai Part 2 Chapter 14 teaches "CUDA kernels from scratch": students hand-write a tile of CUDA C, register it with PyTorch via `torch.utils.cpp_extension`, and watch their hand-rolled matmul beat the naive Python version by orders of magnitude. The chapter is unteachable in the browser today because (a) there is no CUDA in the browser and (b) browsergrad-jit treats kernel codegen as a private implementation detail. PRD-015 closes the gap by exposing a real custom-kernel surface in Python: a `@bg.kernel("wgsl")` decorator accepts a Python function whose body returns a WGSL compute-shader source string, expands shape and dtype placeholders at first call, hands the result to `device.createShaderModule` ([WGSL spec §1](https://www.w3.org/TR/WGSL/)), captures any compile diagnostics as Python exceptions with line and column information, and inserts a new `CUSTOM_WGSL` UOp into the IR that the existing realizer (PRD-005) dispatches through the same WebGPU bridge used for built-in kernels. Custom kernels participate in the OPFS pipeline cache (PRD-008) and in autograd through an explicit `register_backward` hook that emits the gradient as either another custom kernel call or a composition of built-in UOps. By default the new opcode is a fusion barrier; advanced users can attach a `FusionDescriptor` so PRD-012's megakernel codegen can absorb the kernel into surrounding elementwise neighbours. Sandboxing is design-default: WGSL is memory-safe by browser construction — no `fetch`, no `eval`, no out-of-bounds reads — so the security model is "trust the browser, validate dispatch-time sizes." The pedagogical headline is that a student writes their own conv2d in fifty lines of WGSL inside a notebook cell and uses it as a drop-in replacement for `F.conv2d` in the rest of their model.

---

## Background

### Why custom kernels matter for the curriculum

The PRD.md §3.7 target curricula include fast.ai Part 2, which devotes an entire chapter ([fast.ai Part 2 Ch 14, "CUDA kernels"](https://course.fast.ai/Lessons/part2.html)) to writing a hand-rolled GPU kernel and registering it with the autograd engine. The Stanford CS336 syllabus has a parallel exercise ("write a Flash Attention kernel"). Karpathy's "Let's reproduce GPT-2" series has a planned advanced episode on hand-tuned matmul. All three are unteachable in the current browsergrad surface because the kernel layer is not user-addressable — every WGSL string is emitted by the codegen pipeline in PRD-006 / PRD-012 and never crosses the Python boundary.

The competing platforms split on this point. Deep-ml ([deep-ml.com](https://www.deep-ml.com/)) does not expose kernel authoring; its problems stop at the autograd-level API. tinygrad does ([tinygrad RAW op docs](https://docs.tinygrad.org/tensor/ops_low_level/)) — its `Tensor.from_url` and `KernelOp` machinery let users drop down to backend-specific source. PyTorch ships `torch.utils.cpp_extension` ([PyTorch C++ extension tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html)) for exactly the same purpose on the desktop side. Triton ([OpenAI Triton announcement](https://openai.com/research/triton/)) goes further by replacing CUDA C with a Python DSL, but the DSL itself is a research project; we do not match Triton's surface area in v0 of this PRD. The closest published in-browser analogue is the `pipeline.run_wgsl` escape hatch in WebLLM ([WebLLM repo](https://github.com/mlc-ai/web-llm)), which is private to TVM and not user-addressable.

### Why WGSL is a credible substitute for CUDA C

The WGSL specification ([w3.org/TR/WGSL](https://www.w3.org/TR/WGSL/)) covers the exact subset of shader-language features that the fast.ai chapter relies on: workgroup memory (`var<workgroup>`), thread-local registers (`var<private>`), atomic operations, barriers, vector and matrix arithmetic, and a `@compute` entry-point qualifier. Compute-shader limits are published in the WebGPU specification ([gpuweb.github.io/gpuweb/#limits](https://gpuweb.github.io/gpuweb/#limits)) — `maxComputeWorkgroupStorageSize` defaults to 16384 bytes, `maxComputeInvocationsPerWorkgroup` to 256, `maxComputeWorkgroupSizeX` to 256 — and Chrome 144+ has stabilised subgroup operations ([Chrome 144 WebGPU subgroups blog post](https://developer.chrome.com/blog/new-in-webgpu-144)) which give students access to warp-level primitives analogous to CUDA's `__shfl_xor_sync`. The translation from a CUDA tile to a WGSL tile is mechanical for the patterns the fast.ai chapter teaches; the conceptual content (tile sizes, bank conflicts, occupancy, online softmax) transfers verbatim.

### Why this is a P2 rather than a P0

Custom kernels are useless without an IR to hang them on. PRD-005 introduced the UOp graph; without `CUSTOM_WGSL` as a graph node, a hand-written kernel could only run as a one-shot eager dispatch with no autograd, no fusion, no caching. PRD-007 introduced symbolic VJP rules; without that surface, custom backward registration would have to fall back to closure-based autograd — workable, but inconsistent with every other op in browsergrad-jit. PRD-008 introduced the OPFS pipeline cache; without that, every page reload would re-compile the user's kernel even though it has not changed. The three prerequisites stack, and PRD-015 sits on top of all of them.

---

## User Stories

**U1 — Write your own matmul.** A fast.ai Part 2 student writes a tiled matmul kernel in fifty lines of WGSL inside a notebook cell. The decorator returns a callable; the student replaces `x @ W` with `my_matmul(x, W)` in their MLP's forward; training proceeds; gradients flow through their kernel because the student also wrote a `register_backward` that returns `dy @ W.T` and `x.T @ dy` using built-in UOps. The lab reads as a one-to-one analogue of the CUDA chapter.

**U2 — Custom activation with custom backward.** A course author defines a soft-relu approximation `f(x) = log(1 + exp(x))` in WGSL because the numerical-stability lesson requires a kernel-level view of the `exp` overflow. They register a backward that returns `sigmoid(x) * dy` using a single built-in UOp. The autograd test fixture (numerical gradient check) passes within 1e-4.

**U3 — Kernel compile error surfaces to Python.** A student writes a kernel with `@compute @workgroup_size(64)` but forgets to declare `@builtin(global_invocation_id)`. The browser returns a compile diagnostic at line 7, column 12. The Python harness raises `bg.WGSLCompileError` with `e.line == 7`, `e.column == 12`, and `e.source_excerpt` showing the offending line with a caret. The student sees the same error UX as a Python `SyntaxError`.

**U4 — Cached across reloads.** A student finishes their custom matmul and reloads the page to test convergence. The kernel hash is unchanged; the OPFS pipeline cache (PRD-008) returns the cached pipeline; the kernel runs without re-compiling. The dev tools panel shows `bg.cache_stats()` reporting a hit.

**U5 — Fusion-aware advanced user.** A library contributor writes a custom GeLU variant and attaches a `FusionDescriptor` declaring "I am elementwise, shape-preserving, my output depends only on my input at the same index." PRD-012's megakernel codegen absorbs the kernel into the surrounding FFN block; the contributor measures a 1.4× speedup over the same kernel as a fusion barrier.

---

## Goals and Non-Goals

### Goals

1. Ship a `@bg.kernel("wgsl")` decorator that takes a Python function returning a WGSL source string (optionally a template with `${shape}` / `${dtype}` placeholders) and returns a callable that builds a `CUSTOM_WGSL` UOp.
2. Introduce a new IR opcode, `CUSTOM_WGSL`, carrying `(wgsl_source_hash, input_shapes, output_shape, dtype, workgroup_size, entry_point)`, integrated into the realizer (PRD-005) and the OPFS pipeline cache (PRD-008).
3. Provide a backward-registration surface — `@my_kernel.register_backward(...)` — that hooks into the symbolic VJP machinery (PRD-007). The backward function may itself return a custom kernel call or a composition of built-in UOps.
4. Surface WGSL compile errors as `bg.WGSLCompileError` Python exceptions with `line`, `column`, `source_excerpt`, and `raw_browser_message` fields ([WebGPU error API](https://gpuweb.github.io/gpuweb/#errors)).
5. Provide a `FusionDescriptor` opt-in API so advanced users can let PRD-012 fuse their kernel with elementwise neighbours.
6. Document the security model in one place: WGSL has no `fetch`, no `eval`, no arbitrary-pointer arithmetic, and out-of-bounds buffer accesses are clamped or zero-returned by the WGPU runtime ([WGSL §16 "Memory access"](https://www.w3.org/TR/WGSL/#memory-access)). No additional sandbox is required.
7. Provide a notebook-ready debugging mode: `bg.kernel.debug(kernel)` prints the expanded WGSL with line numbers and the resolved shape / dtype substitutions.

### Non-Goals

1. A Python-DSL kernel language (Triton-style). Users write raw WGSL. The Triton DSL is cited as inspiration but explicitly out of scope.
2. Cross-kernel autotuning of user-supplied tile sizes. PRD-012's autotuner remains internal.
3. WebGPU-vendor-specific extensions beyond what the WGSL specification standardises. `shader-f16` is available transparently when the device supports it (PRD-010); subgroup ops are exposed when available ([Chrome 144 subgroups](https://developer.chrome.com/blog/new-in-webgpu-144)).
4. CPU fallback for custom kernels. If the user dispatches a custom WGSL kernel on a device without WebGPU, the runtime raises `bg.NoBackendError`. There is no transpiler to NumPy.
5. Multi-entry-point kernels. Each `@bg.kernel` is one `@compute` entry point.
6. Runtime mutation of the kernel source after first compile. The kernel is a frozen artifact post-decoration.
7. Sandbox for "untrusted user code" beyond what the browser already provides. WGSL is the sandbox.

---

## Architecture

### Decorator and template expansion

**File:** `packages/browsergrad-jit/src/python/kernel.py`

The decorator inspects the Python function's signature for argument names and (optionally) `bg.Tensor` annotations. The function body is expected to return a WGSL source string; the string is treated as a Python `string.Template` with `$shape0`, `$shape1`, `$dtype0`, `$dtype1`, `$output_shape`, `$output_dtype` placeholders. On first call with concrete tensors the placeholders are substituted; the resulting source is hashed (SHA-256) and registered.

```python
@bg.kernel("wgsl", workgroup_size=(16, 16, 1))
def tiled_matmul(x: bg.Tensor, w: bg.Tensor) -> bg.Tensor:
    return """
    @group(0) @binding(0) var<storage, read>       x: array<$dtype0>;
    @group(0) @binding(1) var<storage, read>       w: array<$dtype1>;
    @group(0) @binding(2) var<storage, read_write> y: array<$output_dtype>;

    const M: u32 = ${shape0_0}u;
    const K: u32 = ${shape0_1}u;
    const N: u32 = ${shape1_1}u;

    var<workgroup> xs: array<f32, 256>;
    var<workgroup> ws: array<f32, 256>;

    @compute @workgroup_size(16, 16, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>,
            @builtin(local_invocation_id)  lid: vec3<u32>) {
        // ... tiled matmul body ...
    }
    """
```

Output shape inference is the user's responsibility, declared via a separate `@tiled_matmul.shape_fn` hook:

```python
@tiled_matmul.shape_fn
def _(x_shape, w_shape):
    return (x_shape[0], w_shape[1])
```

If no `shape_fn` is registered the output shape defaults to the broadcast of the inputs and the decorator emits a warning.

### The `CUSTOM_WGSL` UOp

**File:** `packages/browsergrad-jit/src/python/ir.py`

Adds a new opcode to the 19-opcode IR enumerated in PRD-005:

| Opcode | Inputs | `arg` | Output shape | Semantics |
|---|---|---|---|---|
| `CUSTOM_WGSL` | `(x₁, ..., xₙ)` | `{source_hash, entry_point, workgroup_size, output_shape, output_dtype, fusion_descriptor?}` | `output_shape` | Dispatches the user-registered WGSL compute shader identified by `source_hash`. Treated as opaque by PRD-006 fusion unless `fusion_descriptor` is present. |

The `arg` payload is hashable; two `CUSTOM_WGSL` UOps with the same `source_hash`, input shapes, and dtype unify under structural equality, which lets the OPFS pipeline cache (PRD-008) key on `UOp.hash()` exactly as it does for built-in kernels.

### Realization

**File:** `packages/browsergrad-jit/src/python/realize.py` (extended)

The realizer's WebGPU dispatcher gets one new branch:

```python
def _dispatch_custom_wgsl(uop, vt, registry, gpu_device):
    cached = registry.get_pipeline(uop.arg["source_hash"])
    if cached is None:
        source = registry.get_source(uop.arg["source_hash"])
        try:
            module = gpu_device.createShaderModule(code=source)
        except WGSLCompileError as e:
            raise bg.WGSLCompileError.from_browser_error(e, source=source)
        cached = gpu_device.createComputePipelineAsync(
            layout="auto",
            compute={"module": module, "entryPoint": uop.arg["entry_point"]},
        )
        registry.store_pipeline(uop.arg["source_hash"], cached)
    # Bind input/output buffers via existing PRD-005 buffer-table machinery
    return _dispatch(cached, [vt[i] for i in uop.inputs], uop.arg["workgroup_size"])
```

The pipeline-cache key matches the format PRD-008 §6 defines (`{wgsl_hash, browser_fingerprint, layout_descriptor}`), so cached pipelines persist across page loads with no special-case code.

### Backward registration

**File:** `packages/browsergrad-jit/src/python/autograd.py` (extended)

A custom kernel's backward is a Python function consuming the original input UOps and the upstream gradient UOp, returning one gradient UOp per input. Internally it plugs into the PRD-007 VJP table keyed by `(opcode="CUSTOM_WGSL", source_hash=...)`:

```python
@tiled_matmul.register_backward
def _(x, w, dy):
    return dy @ w.T, x.T @ dy
```

The function builds standard UOps; the resulting backward graph is itself an IR rewrite that PRD-007's reverse-mode walker can consume. Double-backward is supported provided the user-supplied backward is itself differentiable through built-in UOps; double-backward through a custom backward kernel is a PRD-007 limitation, not a PRD-015 one.

If no backward is registered, attempting `.backward()` through the kernel raises `bg.NoBackwardError("custom kernel 'tiled_matmul' has no registered backward")`. Users opting out of autograd may pass `requires_grad=False` to all inputs explicitly.

### Fusion descriptor (opt-in)

**File:** `packages/browsergrad-jit/src/python/fusion.py` (extended from PRD-006/PRD-012)

```python
@tiled_matmul.fusion_descriptor
def _():
    return bg.FusionDescriptor(
        kind="elementwise",          # or "reduction", "matmul", "opaque"
        shape_preserving=False,
        per_index_function=None,     # required for "elementwise"
        memory_pattern="tile",       # informs PRD-012 cost model
    )
```

By default `CUSTOM_WGSL` is treated as `kind="opaque"`: PRD-006 stops fusion at the boundary, PRD-012 emits the kernel as its own megakernel boundary. With an explicit descriptor PRD-012's producer-consumer pass can attempt to inline the user's WGSL body into a surrounding megakernel — succeeding only if the cost model's workgroup-memory budget is respected. Failure to fuse is silent (the kernel still runs as a standalone dispatch).

### Compile-error surfacing

**File:** `packages/browsergrad-jit/src/python/errors.py`

The browser returns WGSL compile diagnostics through the WebGPU error scope mechanism ([WebGPU §13 "Errors"](https://gpuweb.github.io/gpuweb/#errors)). The runtime captures the diagnostic, parses the standard `:line:column: message` prefix, and constructs `WGSLCompileError`:

```python
class WGSLCompileError(Exception):
    line: int
    column: int
    source_excerpt: str      # the offending line plus caret
    raw_browser_message: str
    kernel_name: str
```

Python harnesses render the error similarly to a `SyntaxError`. Runtime errors (NaN propagation, division by zero) are **not** raised — WGSL guarantees no undefined behaviour and clamps out-of-bounds reads to zero ([WGSL §16](https://www.w3.org/TR/WGSL/#memory-access)) — so bad values silently propagate. This is documented as a known limitation; users debugging custom kernels are pointed at `bg.kernel.debug(...)` for inspection.

### Sandbox model

The security model is one paragraph in the user-facing docs: *"WGSL kernels cannot read or write memory outside their declared buffers, cannot call `fetch`, cannot `eval` strings, cannot escape the WebGPU sandbox. Out-of-bounds reads return zero; out-of-bounds writes are discarded. The browser is the sandbox; browsergrad adds no additional review."* The dispatch path performs one runtime check: buffer sizes declared in the binding layout must match the input tensor strides, refusing the dispatch with `bg.BufferLayoutError` otherwise. This is defence against accidental shape mismatches, not malicious code.

---

## API Surface

```python
import browsergrad_jit as bg

@bg.kernel("wgsl", workgroup_size=(64, 1, 1))
def my_relu(x: bg.Tensor) -> bg.Tensor:
    return """
    @group(0) @binding(0) var<storage, read>       x: array<$dtype0>;
    @group(0) @binding(1) var<storage, read_write> y: array<$output_dtype>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i >= ${shape0_0}u) { return; }
        y[i] = max(x[i], $dtype0(0));
    }
    """

@my_relu.shape_fn
def _(x_shape):
    return x_shape

@my_relu.register_backward
def _(x, dy):
    return dy * (x > 0).cast(x.dtype)

# Use it like any built-in op
import browsergrad_jit.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(my_relu(self.fc1(x)))

# Debugging
print(bg.kernel.debug(my_relu, sample_input=bg.randn(784)))
# Prints expanded WGSL with line numbers

# Inspect compile artifacts
print(bg.cache_stats().custom_kernels)
# {'my_relu': {'compiles': 1, 'dispatches': 1024, 'cache_hits_on_reload': 1}}
```

---

## Implementation Plan

### Week 1 — Decorator, template expansion, IR opcode

- [ ] Add `CUSTOM_WGSL` to `ir.py` and the 20-opcode shape-inference table.
- [ ] Write `kernel.py`: `@bg.kernel("wgsl")` decorator, `string.Template`-based placeholder expansion (`$shape*`, `$dtype*`, `$output_*`).
- [ ] Write `KernelRegistry` class: maps `source_hash → (source, layout, workgroup_size, shape_fn, backward_fn, fusion_descriptor)`.
- [ ] Unit tests: hash stability across decorator re-evaluation, structural equality of two `CUSTOM_WGSL` UOps with same hash and shapes.

### Week 2 — Realization through WebGPU

- [ ] Extend `realize.py`'s WebGPU dispatcher with `_dispatch_custom_wgsl`.
- [ ] Wire pipeline creation through PRD-008's OPFS pipeline cache. `KernelRegistry.get_pipeline()` consults the cache before recompiling.
- [ ] Implement `bg.WGSLCompileError`: parse browser diagnostic, attach `line`/`column`/`source_excerpt`.
- [ ] Integration test: a trivial `double_each` kernel round-trips through the harness, output matches `2 * x` exactly.

### Week 3 — Backward registration and autograd

- [ ] Extend `autograd.py`'s VJP table to look up `(opcode="CUSTOM_WGSL", source_hash=...)`.
- [ ] Implement `kernel.register_backward(fn)`. Backward functions return UOps; reverse-mode walker (PRD-007) consumes them like any other VJP rule.
- [ ] Implement `bg.NoBackwardError` for kernels invoked under autograd without a registered backward.
- [ ] Integration test: a `my_relu` kernel with registered backward passes numerical gradient check (finite-difference within 1e-3).

### Week 4 — Fusion descriptor, debug surface, error UX

- [ ] Add `FusionDescriptor` to `fusion.py`; PRD-006's elementwise pass consults it. Default `opaque` behaviour preserved.
- [ ] PRD-012's producer-consumer pass: if descriptor declares `kind="elementwise"` with `per_index_function`, inline the WGSL body into the surrounding megakernel; otherwise treat as boundary.
- [ ] Implement `bg.kernel.debug(kernel, sample_input)`: expand placeholders, return source with line numbers.
- [ ] Integration test: a custom GeLU with `kind="elementwise"` descriptor fuses with a surrounding `add → mul` chain (verified via `bg.jit.debug_fused_kernels()` from PRD-006).
- [ ] Publish docs page with the fast.ai-style "write your own conv2d" example.

---

## Acceptance Criteria

| # | Criterion | Measurement |
|---|---|---|
| AC1 | `@bg.kernel("wgsl")` registers a callable that builds a `CUSTOM_WGSL` UOp on first invocation | `kernel_registration.test.ts` |
| AC2 | Trivial `double_each` kernel produces output exactly equal to `2 * x` | `custom_kernel_roundtrip.test.ts` |
| AC3 | Compile error in user-supplied WGSL surfaces as `bg.WGSLCompileError` with line and column | `compile_error_surface.test.ts` |
| AC4 | `register_backward` enables `.backward()` through the custom kernel, gradients match finite-difference within 1e-3 | `custom_backward_gradcheck.test.ts` |
| AC5 | Two invocations of the same kernel with the same input shapes hit the OPFS pipeline cache on second page load | `custom_kernel_cache_hit.test.ts` |
| AC6 | A custom kernel with `FusionDescriptor(kind="elementwise")` is fused by PRD-006/PRD-012 with elementwise neighbours | `custom_fusion.test.ts` |
| AC7 | A custom kernel without a registered backward raises `bg.NoBackwardError` under autograd | `no_backward_error.test.ts` |
| AC8 | A custom matmul kernel substitutes for `F.linear` in a 2-layer MLP without API changes elsewhere | `custom_kernel_mlp_integration.test.ts` |
| AC9 | `bg.kernel.debug(kernel, sample_input)` returns the expanded WGSL with line numbers | `kernel_debug.test.ts` |
| AC10 | Dispatch on a device without WebGPU raises `bg.NoBackendError` (no NumPy fallback) | `custom_kernel_no_backend.test.ts` |
| AC11 | Fast.ai Ch 14 analogue notebook (write-your-own-conv2d) runs end-to-end and trains a small classifier | `notebooks/custom_conv2d.ipynb` regression run in CI |

---

## Test Strategy

### Unit tests (`tests/` — Vitest, no Pyodide)

- `kernel_registry_unit.test.ts`: source-hash stability, structural equality of two `CUSTOM_WGSL` UOps.
- `template_expansion_unit.test.ts`: `${shape*}`, `${dtype*}`, `${output_*}` substitution under multiple input ranks; warning emitted when an unrecognised placeholder is left in the source.
- `fusion_descriptor_unit.test.ts`: PRD-006 fusion pass skips `kind="opaque"` and inlines `kind="elementwise"`.

### Integration tests (`tests-integration/` — Vitest + real Pyodide-in-Node, WebGPU mock)

- `custom_kernel_roundtrip.test.ts` — trivial kernels round-trip; output matches expected.
- `compile_error_surface.test.ts` — deliberately broken WGSL surfaces as `WGSLCompileError` with parsed line/column.
- `custom_backward_gradcheck.test.ts` — finite-difference verification on five kernels (relu, gelu, soft-relu, log1pexp, custom mse).
- `custom_kernel_cache_hit.test.ts` — page reload via the test harness re-uses the cached pipeline.
- `custom_fusion.test.ts` — `bg.jit.debug_fused_kernels()` reports the custom kernel as inlined.
- `custom_kernel_mlp_integration.test.ts` — a custom matmul replaces `F.linear` in an MLP; the model trains and matches the built-in path within 1e-3 over 10 epochs.

### End-to-end notebook regression

- `notebooks/custom_conv2d.ipynb` — the fast.ai-shaped exercise. CI runs the notebook via `jupyter nbconvert --execute` and asserts the final accuracy is within 1% of a reference run.

---

## Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | WGSL compile diagnostics format varies across browsers (Chrome vs Safari vs Firefox), breaking `WGSLCompileError` parsing | High | Medium | Parse defensively: fall back to `raw_browser_message` when the `:line:column:` prefix is absent; integration tests run on all three engines via Playwright in CI |
| R2 | A user's custom kernel has a numerical bug (NaN propagation) that browsergrad silently passes through, blamed on the library | Medium | Medium | Document the "WGSL has no UB, bad values pass through" semantics prominently; provide `bg.kernel.debug` with `assert_no_nan=True` opt-in |
| R3 | `FusionDescriptor(kind="elementwise")` user lies about `per_index_function`, PRD-012 emits a wrong megakernel | Medium | High | Cross-check declared descriptor against actual kernel behaviour with a randomised input comparison during first call; raise `bg.FusionDescriptorMismatch` on divergence |
| R4 | Pipeline cache collision: two distinct user kernels hash to the same `source_hash` (SHA-256 collision) | Vanishingly low | High | Use full SHA-256 of source + layout descriptor + workgroup size; collision-resistance is the standard threat model |
| R5 | Custom kernel dispatched with mismatched buffer sizes corrupts an adjacent allocation | Low | High | WebGPU validates buffer sizes against binding-layout declarations at dispatch ([WebGPU §10.5](https://gpuweb.github.io/gpuweb/#dispatch-validation)); our `BufferLayoutError` is an additional defence-in-depth check |
| R6 | A teaching notebook accidentally allocates a 1 GB workgroup buffer, crashes the tab | Medium | Medium | Validate declared workgroup size and `var<workgroup>` total size against `maxComputeWorkgroupStorageSize` at compile time; raise `bg.WorkgroupLimitExceeded` with the offending limit name |
| R7 | A user expects CUDA-style `__shfl_xor_sync` semantics, gets surprised when subgroups are unavailable on older Chrome | Medium | Low | Detect `GPUDevice.features.has("subgroups")` at kernel-registration time when the WGSL source contains `subgroupShuffle*` calls; raise an explicit `bg.SubgroupsUnavailable` |
| R8 | Browser revokes the `GPUDevice` mid-dispatch (e.g., switching dGPU/iGPU), custom kernels lose their cached pipelines | Low | Medium | Re-register pipelines on `device.lost` event ([WebGPU §6 "Device loss"](https://gpuweb.github.io/gpuweb/#device-loss)); PRD-008's cache key includes device fingerprint so stale entries are invalidated |

---

## Open Questions

1. **Should `@bg.kernel("wgsl")` allow multiple entry points in one source string?** Triton allows it for sub-kernel composition. v0 says no — one decorator, one entry point — but a `@bg.kernel.entrypoints(["forward", "backward"])` variant is on the long tail. Decision deferred to a successor PRD once a real curriculum lab demands it.

2. **Subgroup-op exposure.** Chrome 144+ ships subgroups; Safari 26 ships them as `experimental-subgroups`. Do we expose them through a feature-flag (`@bg.kernel("wgsl", require=["subgroups"])`) or transparently? Tentative resolution: transparent at the source level (the WGSL compiler errors if the device doesn't support them), with a clearer `bg.SubgroupsUnavailable` exception so the student sees a sensible message rather than a raw WGSL diagnostic.

3. **Backward kernel sharing between forward calls of different shapes.** The forward kernel is shape-specialised; the backward kernel is the user's function. Should the backward be re-traced per forward shape or shared? Resolution for v0: re-trace per forward shape — same as forward. The forward and backward source hashes are independent; both participate in PRD-008's cache.

4. **Determinism guarantees.** WGSL does not guarantee bit-exact floating-point determinism across runs ([WGSL §4 "Floating-point semantics"](https://www.w3.org/TR/WGSL/#floating-point-semantics)). For a teaching context this is fine; for a grading context (deep-ml-style) it matters. Resolution: document the non-determinism explicitly; reserve a `@bg.kernel("wgsl", deterministic=True)` flag for a future PRD that would route the kernel through a deterministic-reduction codegen variant.

5. **Vector and matrix WGSL types.** Should `vec4<f32>` and `mat4x4<f32>` outputs be exposed at the Python boundary as separate tensor shapes, or always flattened to `array<f32>`? v0 supports `array<f32>`, `array<f16>`, `array<i32>`, and `array<u32>`; structured types are unwrapped on the Python side. Vector types remain available within the kernel body. Future PRD if a curriculum needs richer types at the Python boundary.

6. **Pedagogical pairing with a Python-DSL.** The Triton-inspired "tile-DSL" surface is appealing but a research project in its own right. Should we ship one alongside the raw-WGSL surface? Resolution: no — raw WGSL is the right pedagogical primitive for fast.ai Part 2 Ch 14; a DSL is for a future PRD aimed at advanced researchers.

---

## References

1. **WGSL specification** — [W3C WGSL](https://www.w3.org/TR/WGSL/). Authoritative reference for the shader language. §1 (overview), §4 (FP semantics), §16 (memory access) are load-bearing here.

2. **WebGPU compute-shader limits** — [WebGPU specification §3 Limits](https://gpuweb.github.io/gpuweb/#limits). Establishes `maxComputeWorkgroupStorageSize`, `maxComputeInvocationsPerWorkgroup`, etc., that the dispatch-time validation uses.

3. **WebGPU error model** — [WebGPU specification §13 Errors](https://gpuweb.github.io/gpuweb/#errors); [WebGPU device loss §6](https://gpuweb.github.io/gpuweb/#device-loss). Underpins `WGSLCompileError` and the cache-invalidation-on-device-loss path.

4. **Chrome 144 WebGPU subgroups** — [Chrome developer blog "New in WebGPU 144"](https://developer.chrome.com/blog/new-in-webgpu-144). Source for the subgroup-availability detection and the advanced-user opt-in.

5. **Triton DSL** — [OpenAI Triton announcement](https://openai.com/research/triton/); [Triton repo](https://github.com/triton-lang/triton). Cited as inspiration for tile-level Python-DSL custom kernels; we deliberately do not match this surface in v0.

6. **tinygrad RAW / low-level ops** — [tinygrad tensor ops_low_level](https://docs.tinygrad.org/tensor/ops_low_level/); [tinygrad function.py](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/function.py). Closest existing analogue for user-supplied backend code in a PyTorch-shaped library.

7. **PyTorch C++ extensions** — [PyTorch C++ extension tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html); [`torch.utils.cpp_extension` docs](https://pytorch.org/docs/stable/cpp_extension.html). Desktop analogue we are porting to the browser; `register_backward` mirrors `torch.autograd.Function`.

8. **Flash Attention v2** — [arXiv:2307.08691](https://arxiv.org/abs/2307.08691). Reference for the kind of advanced custom kernel a sufficiently motivated student will write inside this surface — workgroup-memory online softmax with tiled Q/K/V.

9. **fast.ai Part 2 Ch 14 — "CUDA kernels"** — [course.fast.ai/Lessons/part2.html](https://course.fast.ai/Lessons/part2.html). The headline curriculum dependency this PRD exists to satisfy.

10. **HuggingFace safetensors format** — [safetensors repo](https://github.com/huggingface/safetensors). Adjacent dependency; pretrained kernels shipped alongside user custom kernels follow the same memory-mapped pattern PRD-008 establishes.

11. **PRD-005 — JIT foundation** — `docs/prd/PRD-005-jit-foundation.md`. IR opcode table extended here.

12. **PRD-007 — Symbolic backward** — `docs/prd/PRD-007-symbolic-backward.md`. VJP-registration mechanism that `register_backward` plugs into.

13. **PRD-008 — Persistent caching** — `docs/prd/PRD-008-persistent-caching.md`. OPFS pipeline cache that custom kernels reuse with no special-casing.

14. **PRD-012 — Megakernel codegen** — `docs/prd/PRD-012-megakernel-codegen.md`. Consumer of the `FusionDescriptor` surface introduced here.
