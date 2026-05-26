# PRD-010 — Real Mixed Precision Training: `autocast` + `GradScaler` + WGSL f16

| Field | Value |
|---|---|
| **Status** | Draft v1 |
| **Author** | unlocalhosted maintainers |
| **Date** | 2026-05-26 |
| **PRD ID** | PRD-010 |
| **Phase** | P1 (Months 4–9 of the 14-month roadmap in PRD.md §6 — feature P1.8) |
| **Package** | `@unlocalhosted/browsergrad-jit`, `@unlocalhosted/browsergrad-kernels` |
| **Depends on** | PRD-005 (UOp IR + tracer — adds `CAST` and dtype tracking), PRD-006 (WGSL codegen — fused kernels must emit f16 ops), PRD-007 (symbolic backward — gradient scaling lives at the IR level) |
| **Enables** | Fast.ai Part 2 Ch 20 as a real lab, transformer training within memory budgets, future quantization work (PRD-017) |

---

## TL;DR

PyTorch's mixed-precision recipe — `torch.amp.autocast` for per-op dtype routing plus `GradScaler` for dynamic loss scaling — has been the production-default for training since ~2020 ([Micikevicius et al. 2017](https://arxiv.org/abs/1710.03740), [PyTorch AMP docs](https://pytorch.org/docs/stable/amp.html)). On hardware that supports it, this recipe cuts memory bandwidth in half, doubles arithmetic throughput on matmul/conv, and converges to within 0.05% of the fp32 loss curve. The browser now has the silicon path to do the same: WebGPU's [`shader-f16` extension](https://www.w3.org/TR/WGSL/#extension-f16) is enabled by default in Chrome 113+ and Edge 113+, on by default in Safari 26, and behind a flag in Firefox 145. PRD-010 ships a *real* mixed-precision stack — not a no-op — on top of the IR PRD-005 introduced. Specifically: (1) a `torch.amp.autocast(device_type="webgpu", dtype=torch.float16)` context manager that walks the IR at trace time and inserts `CAST` nodes around allowlist ops (matmul, conv) so they run in f16, while keeping blocklist ops (softmax, layernorm, loss, exp/log reductions) in f32; (2) a `GradScaler` whose `scale`/`unscale_`/`step` operations are themselves IR nodes so loss scaling is part of the traced graph rather than a Python wrapper; (3) WGSL codegen extensions in `browsergrad-kernels` that emit `enable f16;` and use `f16`/`vec4<f16>` types for fused elementwise + matmul kernels when the device exposes `shader-f16`; (4) feature detection with graceful, *loud* fallback to f32 when the extension is unavailable. The acceptance bar is concrete: a small CNN trains MNIST to ≥95% test accuracy in both fp32 and AMP modes, and AMP delivers ≥1.3× wall-clock speedup on a hidden=768 transformer block on M-class hardware.

---

## Background

### What mixed precision actually is

[Micikevicius et al. (2017)](https://arxiv.org/abs/1710.03740) is the founding paper. Three ideas combined:

1. **Store activations and the forward computation in fp16.** This halves memory bandwidth — the dominant cost on matmul-heavy workloads — and on hardware with dedicated f16 math units it ~doubles arithmetic throughput.
2. **Keep a master copy of weights in fp32**, and accumulate matmul/conv inner products in fp32 even when inputs are f16. fp16's 10-bit mantissa is enough for the multiplicand but not enough to accumulate a sum over thousands of terms; fp32 accumulators preserve the gradient signal.
3. **Scale the loss by a large constant S (typically 2¹⁶–2²⁴) before backward**, then divide gradients by S before the optimizer step. fp16's minimum positive normal value is ~6.1e-5; without scaling, the smallest gradients underflow to zero and learning stalls. Scaling pushes the gradient histogram into the representable range. Dynamic scaling adjusts S adaptively: double S after N consecutive non-NaN backward passes, halve S immediately on any NaN.

Tensor-core hardware (Nvidia V100+, AMD MI100+, Apple M-series Neural Engine) has dedicated multiply-accumulate units that implement exactly this pattern: f16 inputs, fp32 accumulator, fp16 or fp32 output. The PyTorch `autocast` API codifies which ops run in which dtype — see the official ["Op-Specific Behavior" table](https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16).

### Why the browser was a special case (and why that's ending)

PRD.md §8 originally listed mixed precision as a non-goal: "WebGPU's fp16 support is uneven; autocast stays a no-op." That was true in 2024. As of May 2026 it is no longer true:

- **Chrome 113+** ships `shader-f16` as a standard WebGPU extension, on by default. ([Chrome Status: shader-f16](https://chromestatus.com/feature/5180552310751232).)
- **Edge 113+** inherits Chromium support.
- **Safari 26.0** (Sept 2025) supports `shader-f16` by default on Apple Silicon Macs and iPad M-series.
- **Firefox 145** ships `shader-f16` behind `dom.webgpu.f16.enabled` in stable as of April 2026; landing as default is on the M147 roadmap.
- Historical note: Chrome had this behind `chrome://flags/#enable-unsafe-webgpu` until M113. We do **not** need that flag in 2026.

The [W3C WGSL §extension-f16 spec](https://www.w3.org/TR/WGSL/#extension-f16) defines the contract: a kernel opts in via `enable f16;`, gains the `f16`, `vec2<f16>`, `vec3<f16>`, `vec4<f16>`, `mat2x2<f16>` ... types, and the runtime must request the extension at device-creation time: `await adapter.requestDevice({ requiredFeatures: ["shader-f16"] })`. Devices without the feature throw on request; we feature-detect via `adapter.features.has("shader-f16")` and degrade.

PRD-006 ships fp32-only fused kernels. PRD-010 extends that codegen to emit f16 variants when the IR demands them.

### Why the no-op stub is now actively harmful

PRD.md §8 also said: "A real fp16 implementation beats a stub for the educational value of teaching mixed precision." Fast.ai Part 2 Ch 20 ("Mixed Precision") is *literally* about this recipe. Shipping a no-op `autocast` means the chapter cannot teach what it claims to teach: students see no speedup, no underflow, no GradScaler-fired skip step. Real implementation = real lesson.

There is also a meaningful runtime win: an f16 matmul kernel on M2-class hardware via WebGPU subgroups achieves measurably higher throughput than fp32 because (a) memory bandwidth halves and (b) Apple's f16 path on the M-series GPU has dedicated SIMD units. We project ≥1.3× on matmul-heavy workloads on M-series after fusion (PRD-006) is in place; this is conservative relative to the 1.5–2× tensor-core ratio on Nvidia, because WebGPU does not yet expose tensor-core paths directly.

---

## User Stories

**U1 — Fast.ai Ch 20 lab, real.** A student opens the Mixed Precision lab in craftingattention. They wrap the training step in `with torch.amp.autocast(device_type="webgpu"):` and add `scaler = torch.amp.GradScaler()`. They run side-by-side fp32 vs AMP cells. The AMP cell trains visibly faster (≥1.3× wall-clock on the M2-class hardware our benchmarks target), reaches the same final accuracy within 0.05%, and the `scaler` reports a scale value that adapts over training. The chapter teaches what it claims to teach.

**U2 — Underflow demonstration.** A course author builds a small interactive that disables `GradScaler` (`enabled=False`) and shows a tiny transformer training. The student sees training stall — loss flat-lines after a few hundred steps because gradients have underflowed to zero. They re-enable the scaler; training resumes. The pedagogy is in the observed failure.

**U3 — Graceful degradation on Firefox.** A student on Firefox 145 without the about:config flag opens an AMP lab. The runtime detects `!adapter.features.has("shader-f16")`, prints a single visible warning ("autocast unavailable on this browser: shader-f16 extension missing; falling back to fp32"), and silently treats `autocast` as a no-op. The training completes correctly, just slower. No errors thrown, no silent wrong-precision math.

**U4 — Memory headroom for bigger models.** A student tries to train a 6-layer transformer with hidden=512 in their browser tab. In fp32 the activation buffers OOM. In AMP, activations are f16, the memory budget is halved, the model fits. They reach the lab's target without changing their model code.

**U5 — Cross-validation.** A maintainer runs the conformance suite under `torch.amp.autocast(device_type="webgpu")`. Forward outputs and gradients match the fp32 reference within 1e-3 (looser than the 1e-4 elsewhere because f16 has 10-bit mantissa). The PyTorch AMP-vs-fp32 conformance fixture (newly added in this PRD) passes.

---

## Goals and Non-Goals

### Goals

1. Implement `torch.amp.autocast(device_type="webgpu", dtype=torch.float16, enabled=True)` as a real context manager that toggles a thread-local flag observed by the IR builder.
2. Implement a real IR pass — **`insert_cast_pass`** — that walks the UOp graph, identifies allowlist ops (`MATMUL`, `CONV2D`, batched matmul forms), and inserts `CAST` nodes to demote their inputs to f16 and re-promote outputs to f32. Blocklist ops (`REDUCE`, `EXP`, `LOG`, `DIV` in softmax, `LayerNorm`'s reductions, `cross_entropy`/`NLLLoss`/`MSELoss`) stay in f32.
3. Implement `torch.amp.GradScaler` as IR-traceable nodes: `scale(loss)` is an IR `MUL` by a `CONST` scalar held in a `BUFFER`; `unscale_(grad)` is a `DIV` by the same `CONST`; the NaN check is an IR-level `ISNAN` + `REDUCE(or)` producing a boolean realized through the existing realization pipeline.
4. Add an `ISNAN` opcode (20th UOp) with NumPy + WGSL realizers. Needed for the dynamic scaler.
5. Extend `browsergrad-kernels`' WGSL codegen to emit `enable f16;` and f16 types/intrinsics for fused elementwise and fused matmul kernels whose IR carries `dtype="f16"`.
6. Feature-detect `shader-f16` at device-request time; surface as `bg.amp.is_available()` Python API; fall back to f32 with a loud warning when unavailable.
7. Track weights in fp32 master copies; produce f16 *views* on the fly when the forward enters an `autocast` region. Optimizer reads f32 master, applies unscaled gradients, writes f32 master.
8. Pass a new PyTorch-conformance fixture: train a 4-layer CNN on a 5K-sample MNIST shard for 5 epochs, in fp32 and AMP, both reaching ≥95% test accuracy. AMP wall-clock ≥1.3× faster on M-series hardware where `shader-f16` is enabled.

### Non-Goals

1. **bfloat16.** [BF16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) is not part of the WGSL spec as of May 2026 ([WGSL §extension-f16](https://www.w3.org/TR/WGSL/#extension-f16) defines IEEE binary16 only). Defer until WebGPU exposes a `shader-bf16` extension. `torch.bfloat16` raises a clear `NotImplementedError`.
2. **Int8 / fp8 quantization.** Different problem; separate future PRD.
3. **Mixed-precision for *backward fusion* across the entire graph.** PRD-006 ships forward elementwise + softmax/layernorm fusion. Fusing the backward pass with f16 storage is layered on top of PRD-007's symbolic backward, addressed in PRD-012.
4. **Autocast for ops outside our IR.** Custom WGSL kernels (PRD-024 / P2.5) are opaque to the cast pass; the user owns dtype inside them.
5. **`torch.cuda.amp` legacy API path.** PyTorch 2.x deprecated `torch.cuda.amp.autocast` in favor of `torch.amp.autocast(device_type=...)`. We ship the new API only. A shim alias maps `torch.cuda.amp.autocast` to `torch.amp.autocast(device_type="webgpu", ...)` with a deprecation warning, matching upstream.
6. **Memory-format autotuning** (NHWC vs NCHW). PyTorch's `channels_last` interacts with AMP for conv; we ship NCHW only, matching the rest of browsergrad.

---

## Architecture

### High-level flow

```
User Python:
    with torch.amp.autocast(device_type="webgpu"):
        logits = model(x)        # builds IR; ops tagged with autocast=True
    loss = F.cross_entropy(logits, y)   # OUTSIDE autocast — runs in f32
    scaler.scale(loss).backward()       # IR mul-by-S, then PRD-007 VJP walk
    scaler.step(optimizer)              # unscale + nan-check + step
    scaler.update()                     # adaptive scale update

Trace time (every op called inside `with autocast(...)`):
    -> UOp constructed with `autocast_hint="f16"` (or whatever active dtype is)

Realization time, before codegen:
    -> insert_cast_pass(graph) walks UOps, applies the allowlist/blocklist
       rules, inserts CAST(f16) before allowlist ops and CAST(f32) after.
       Blocklist regions stay f32.

Codegen (PRD-006 path):
    -> Each fused kernel inspects its constituent UOps' dtypes; if any are f16
       and the device supports shader-f16, emits `enable f16;` and uses f16
       types. Matmul kernels still accumulate in f32.

Dispatcher:
    -> If shader-f16 unavailable: insert_cast_pass becomes a no-op (skip),
       autocast silently demotes to fp32. Warning logged once per session.
```

### Allowlist / blocklist rules (matches PyTorch AMP)

The cast pass uses three sets, mirroring the [PyTorch AMP CUDA op table](https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16):

```python
# packages/browsergrad-jit/src/python/amp/policy.py

# Run in f16 (downcast inputs, accumulate in fp32, upcast output if consumer is fp32)
AMP_ALLOWLIST_F16 = {
    "MATMUL",       # the headline win
    "CONV2D",       # not yet a primitive in PRD-005 — see Open Question 4
    "CONV1D",
    "CONV_TRANSPOSE2D",
}

# Always run in fp32 — accuracy-critical
AMP_BLOCKLIST_F32 = {
    "EXP", "LOG",         # softmax internals
    "REDUCE",             # sum/mean/max over many elements — accumulation precision
    "DIV",                # softmax normalize, layernorm rsqrt
    "POW",                # layernorm variance squaring
    "LOSS_NLL", "LOSS_CE", "LOSS_MSE",   # composite loss UOps
    "RSQRT",              # layernorm
}

# Promote: if any input is f32, promote others to f32 (else stay narrow)
AMP_PROMOTE = {"ADD", "SUB", "MUL", "WHERE", "PAD"}
```

The pass is one topological walk:

```python
def insert_cast_pass(graph: UOpGraph, active_dtype: str) -> UOpGraph:
    """If active_dtype=='f16' and shader-f16 available, rewrite graph."""
    new_uops: dict[UOp, UOp] = {}

    for node in topological_order(graph):
        if node.op in AMP_ALLOWLIST_F16:
            new_inputs = tuple(_cast_to(new_uops.get(i, i), active_dtype) for i in node.inputs)
            recast = UOp(node.op, new_inputs, node.shape, active_dtype, node.arg)
            promoted = UOp("CAST", (recast,), node.shape, "f32", arg={"dtype": "f32"})
            new_uops[node] = promoted

        elif node.op in AMP_BLOCKLIST_F32:
            new_inputs = tuple(_cast_to(new_uops.get(i, i), "f32") for i in node.inputs)
            new_uops[node] = UOp(node.op, new_inputs, node.shape, "f32", node.arg)

        elif node.op in AMP_PROMOTE:
            # If any input is f32 → all f32. Else stay in whatever they share.
            mapped = [new_uops.get(i, i) for i in node.inputs]
            target = "f32" if any(m.dtype == "f32" for m in mapped) else mapped[0].dtype
            new_inputs = tuple(_cast_to(m, target) for m in mapped)
            new_uops[node] = UOp(node.op, new_inputs, node.shape, target, node.arg)

        else:
            new_uops[node] = _rewire(node, new_uops)

    return _rebuild_graph(new_uops, graph.outputs)


def _cast_to(uop: UOp, dtype: str) -> UOp:
    if uop.dtype == dtype:
        return uop
    return UOp("CAST", (uop,), uop.shape, dtype, arg={"dtype": dtype})
```

`CAST` is already an IR opcode in PRD-005 §IR Design. The cast pass simply inserts them at boundaries.

A redundancy-elimination follow-up pass removes `CAST(f32) → CAST(f16) → CAST(f32)` runs that emerge when an allowlist op feeds another allowlist op directly:

```python
def collapse_redundant_casts(graph: UOpGraph) -> UOpGraph:
    # CAST(x, a) -> CAST(_, b)  ==>  CAST(x, b)
    # CAST(x, dtype) where x.dtype == dtype  ==>  x
    ...
```

This produces the canonical pattern: f32 weights/inputs → one downcast → contiguous f16 region (multiple allowlist ops, no upcast between) → one upcast → f32 blocklist region.

### `autocast` context manager

**File:** `packages/browsergrad-jit/src/python/amp/autocast.py`

```python
import threading
from contextlib import ContextDecorator

_TLS = threading.local()

def _active_dtype() -> str | None:
    return getattr(_TLS, "amp_dtype", None)

class autocast(ContextDecorator):
    def __init__(self, device_type: str = "webgpu",
                 dtype: "torch.dtype" = None, enabled: bool = True):
        if device_type not in ("webgpu", "cpu"):
            raise ValueError(f"autocast device_type must be 'webgpu' or 'cpu', got {device_type!r}")
        if dtype is None:
            dtype = torch.float16
        if dtype not in (torch.float16, torch.float32):
            raise NotImplementedError(f"autocast dtype={dtype}; only float16 supported "
                                      f"(bfloat16 requires shader-bf16 — not in WGSL yet)")
        self.enabled = enabled and _amp_available()
        self.dtype_str = "f16" if dtype is torch.float16 else "f32"
        self._prev = None

    def __enter__(self):
        self._prev = getattr(_TLS, "amp_dtype", None)
        if self.enabled:
            _TLS.amp_dtype = self.dtype_str
        return self

    def __exit__(self, *exc):
        _TLS.amp_dtype = self._prev
        return False
```

`TensorProxy.__init__` reads `_active_dtype()` and tags every UOp built inside the region with `arg["autocast_hint"]=self.dtype_str`. The cast pass uses this tag to gate which ops to rewrite, so we never rewrite ops the user constructed outside the region (e.g., a loss computed deliberately in f32).

### `GradScaler` — IR-native, not a Python wrapper

**File:** `packages/browsergrad-jit/src/python/amp/grad_scaler.py`

PyTorch's `GradScaler` in CUDA-land is a Python-level wrapper that calls `_amp_foreach_non_finite_check_and_unscale_`. We have no CUDA primitive to call; instead the scaler emits IR nodes so the entire scale/unscale dance ends up in the traced graph that PRD-007 differentiates and PRD-006 may later fuse.

```python
class GradScaler:
    def __init__(self,
                 init_scale: float = 2.0 ** 16,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000,
                 enabled: bool = True):
        self.enabled = enabled and _amp_available()
        # Held as a 1-element BUFFER so it lives in the IR like any tensor.
        self._scale_buf = new_buffer("amp_scale", np.array([init_scale], dtype=np.float32))
        self._growth_tracker = 0
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval

    def scale(self, loss: TensorProxy) -> TensorProxy:
        if not self.enabled:
            return loss
        scale_proxy = TensorProxy(UOp("LOAD", (self._scale_buf,), (1,), "f32"))
        return loss * scale_proxy             # IR MUL — traced into backward

    def unscale_(self, optimizer) -> None:
        if not self.enabled:
            return
        inv_scale = UOp("DIV",
                        (UOp("CONST", (), (), "f32", arg={"value": 1.0}),
                         UOp("LOAD", (self._scale_buf,), (1,), "f32")),
                        (1,), "f32")
        for p in optimizer.param_groups[0]["params"]:
            if p.grad is None:
                continue
            p.grad = TensorProxy(UOp("MUL", (p.grad._uop, inv_scale),
                                      p.grad.shape, "f32"))

    def step(self, optimizer) -> None:
        if not self.enabled:
            return optimizer.step()
        self.unscale_(optimizer)
        # Check for non-finite gradients via realized boolean.
        any_nan = self._any_nonfinite([p.grad for p in optimizer.param_groups[0]["params"]
                                       if p.grad is not None])
        if any_nan.item():        # realization point
            self._growth_tracker = 0
            self._scale_buf.set(self._scale_buf.get() * self._backoff_factor)
            return                # skip the optimizer step entirely
        optimizer.step()
        self._growth_tracker += 1
        if self._growth_tracker >= self._growth_interval:
            self._growth_tracker = 0
            self._scale_buf.set(self._scale_buf.get() * self._growth_factor)

    def update(self) -> None:
        """No-op in this design — growth/backoff happens inside .step()."""
        pass

    def _any_nonfinite(self, grads: list[TensorProxy]) -> TensorProxy:
        nans = [TensorProxy(UOp("ISNAN", (g._uop,), g.shape, "bool")) for g in grads]
        flat = [TensorProxy(UOp("REDUCE", (n._uop,), (), "bool",
                                arg={"op": "or", "axis": "all", "keepdims": False}))
                for n in nans]
        # Combine with successive OR.
        cur = flat[0]
        for f in flat[1:]:
            cur = TensorProxy(UOp("OR", (cur._uop, f._uop), (), "bool"))
        return cur
```

This adds two IR opcodes that PRD-005 didn't have: `ISNAN` (`x != x` semantics via NumPy `np.isnan`, WGSL `nan != nan`) and `OR` (bool, bool → bool, boolean reduce-friendly). Both are trivial NumPy and trivial WGSL.

The scale state lives in a `BUFFER`, so reading/writing it is the same mechanism as any tensor and the scaler value can be inspected with `scaler.get_scale()` → `self._scale_buf.get()`.

### WGSL f16 kernel emission (PRD-006 codegen extensions)

`browsergrad-kernels`' WGSL emitter today produces f32 kernels (PRD-006 §6.3). PRD-010 adds an f16 codepath. For an elementwise chain whose IR carries `dtype="f16"` after the cast pass:

```wgsl
enable f16;

@group(0) @binding(0) var<storage, read>       buf_x : array<f16>;
@group(0) @binding(1) var<storage, read>       buf_w : array<f16>;
@group(0) @binding(2) var<storage, read>       buf_b : array<f16>;
@group(0) @binding(3) var<storage, read_write> buf_out : array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&buf_x)) { return; }

    var v0 : f16 = buf_x[idx] * buf_w[idx];      // f16 multiply
    var v1 : f16 = v0 + buf_b[idx];              // f16 add
    var v2 : f16 = max(v1, f16(0.0));            // f16 relu

    buf_out[idx] = v2;
}
```

For matmul the codegen template specifically routes f16 inputs through fp32 accumulators, matching tensor-core semantics:

```wgsl
enable f16;

@group(0) @binding(0) var<storage, read>       A : array<f16>;
@group(0) @binding(1) var<storage, read>       B : array<f16>;
@group(0) @binding(2) var<storage, read_write> C : array<f16>;
struct Dims { M : u32, N : u32, K : u32 }
@group(0) @binding(3) var<uniform> dims : Dims;

@compute @workgroup_size(16, 16)
fn matmul_f16_acc_f32(@builtin(global_invocation_id) gid : vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= dims.M || col >= dims.N) { return; }

    var acc : f32 = 0.0;             // <-- accumulator in fp32
    for (var k : u32 = 0u; k < dims.K; k = k + 1u) {
        let a : f32 = f32(A[row * dims.K + k]);
        let b : f32 = f32(B[k * dims.N + col]);
        acc = acc + a * b;
    }
    C[row * dims.N + col] = f16(acc);    // downcast on store
}
```

The fp32 accumulator inside an f16-buffered matmul is the single most important detail in the entire PRD. Without it, every long-reduction matmul accumulates 10-bit-mantissa error and diverges. With it, we match what tensor cores do at the hardware level.

The full tiled matmul (workgroup-memory blocking, ~3× faster) is a follow-on; the version above is functionally correct and >1.3× faster than f32 because of the halved memory bandwidth alone.

### Weight storage policy: f32 master, f16 views on demand

Critical detail for the memory story. We do **not** store two copies of every weight on the GPU. We store one f32 master in a `BUFFER` and emit a `CAST(f16)` UOp on the forward path — the f16 view is a transient kernel-input buffer materialized only during the dispatch of the consuming f16 kernel.

```
Parameters (always):     fp32 master in BUFFER  (e.g. Linear.weight = 4MB at hidden=1024)
Forward inside autocast: fp32 → CAST(f16) → MATMUL(f16, acc=f32) → CAST(f32) → ...
Activations stored:      f16 (where allowlist-region outputs land)
Backward grads:          start f32 (loss is f32) → flow through CAST inverses → arrive
                         at parameter slots in f32 (where the .grad accumulates)
Optimizer step:          reads f32 master, applies f32 unscaled grads, writes f32 master
```

This matches PyTorch AMP's master-weights-in-fp32 design exactly ([AMP recipe](https://pytorch.org/docs/stable/notes/amp_examples.html)). Memory savings on activations: ~2× during the forward pass, which is where the budget actually pinches. Parameter storage doesn't shrink — and shouldn't, for stability.

### Feature detection

**File:** `packages/browsergrad-kernels/src/device/f16-feature.ts`

```typescript
let _f16Available: boolean | null = null;

export async function detectShaderF16(): Promise<boolean> {
  if (_f16Available !== null) return _f16Available;
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) {
    _f16Available = false;
    return false;
  }
  _f16Available = adapter.features.has("shader-f16");
  return _f16Available;
}

export async function requestDeviceWithF16(): Promise<GPUDevice> {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter");
  const wantsF16 = adapter.features.has("shader-f16");
  return adapter.requestDevice({
    requiredFeatures: wantsF16 ? ["shader-f16"] : [],
  });
}
```

Exposed to Python via the runtime bridge as `browsergrad_jit.amp.is_available() -> bool`. The `autocast` context manager calls this; if false, `__enter__` is a no-op and a single warning is logged per session.

---

## API Surface

```python
import browsergrad_jit as torch
import browsergrad_jit.amp as amp     # convenience alias for torch.amp

# Detect availability
if amp.is_available():
    print("Mixed precision available")

# The full PyTorch-compatible recipe
model = MyTransformer().to("webgpu")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler()

for x, y in loader:
    optimizer.zero_grad()
    with torch.amp.autocast(device_type="webgpu", dtype=torch.float16):
        logits = model(x)
        loss = F.cross_entropy(logits, y)    # cross_entropy auto-blocklisted: runs f32 inside autocast
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Disabling AMP for one section (e.g. numerically-sensitive eval metric)
with torch.amp.autocast(device_type="webgpu", enabled=False):
    eval_logits = model(eval_x)

# Inspecting the dynamic scale
print(scaler.get_scale())                  # e.g. 32768.0 after some backoffs

# Legacy CUDA API alias — deprecation warning + redirect
with torch.cuda.amp.autocast():
    ...   # warns and calls torch.amp.autocast(device_type="webgpu")
```

---

## Implementation Plan

### Week 1 — Feature detection + autocast context manager + IR tagging

- [ ] `packages/browsergrad-kernels/src/device/f16-feature.ts`: `detectShaderF16`, `requestDeviceWithF16`.
- [ ] Wire the kernels package to request `shader-f16` at device init when available; fall back cleanly when not.
- [ ] `packages/browsergrad-jit/src/python/amp/autocast.py`: `autocast` context manager + thread-local state.
- [ ] `TensorProxy.__init__` tags new UOps with `arg["autocast_hint"]` when active.
- [ ] `browsergrad_jit.amp.is_available()` Python API bridged via the runtime.
- [ ] Unit tests: enter/exit nesting, `enabled=False`, invalid `device_type`, invalid `dtype` raising `NotImplementedError` for bf16.

### Week 2 — Cast-insertion IR pass + redundant-cast collapse

- [ ] `packages/browsergrad-jit/src/python/amp/policy.py`: allowlist / blocklist / promote sets.
- [ ] `packages/browsergrad-jit/src/python/amp/cast_pass.py`: `insert_cast_pass`, `collapse_redundant_casts`.
- [ ] Wire into the realization pipeline between trace and codegen.
- [ ] Unit tests against synthetic IR graphs: matmul-then-softmax produces (cast-f16, matmul-f16, cast-f32, softmax-f32, ...) sequence with exactly the right cast count.
- [ ] Cross-validation: run the existing 234 integration tests with `autocast` *disabled* — must remain identical to PRD-007 behavior (pass must be no-op when no tagged UOps exist).

### Week 3 — `ISNAN`/`OR` opcodes + `GradScaler`

- [ ] Add `ISNAN` (input: any tensor, output: bool same shape) and `OR` (bool, bool → bool) to PRD-005's opcode registry.
- [ ] NumPy realizers: `np.isnan`, `np.logical_or`.
- [ ] WGSL realizers: emit `x != x` for `ISNAN`, `||` for `OR`. Trivial per-element fused-elementwise extensions.
- [ ] VJP registrations (PRD-007): both opcodes return `None` for all input gradients (boolean / non-differentiable).
- [ ] `packages/browsergrad-jit/src/python/amp/grad_scaler.py`: full implementation with adaptive scaling.
- [ ] Unit tests: artificially insert NaN, assert step is skipped, scale halves; run 2000 clean steps, assert scale doubles.

### Week 4 — WGSL f16 codegen (elementwise + matmul)

- [ ] Extend `packages/browsergrad-kernels/src/jit/codegen.ts` (PRD-006's emitter): detect any f16 dtype in the kernel's UOps; emit `enable f16;` header + `f16` array types.
- [ ] f16 matmul template with fp32 accumulator (above). Add to kernel registry behind a `dtype="f16"` discriminator.
- [ ] Conformance: drive a small set of fixtures (matmul 64×64 @ 64×64; matmul 256×256 @ 256×256; conv2d 32 channels in × 64 out × kernel 3) through f16 path; numerical match within 1e-3 of fp32 reference (relaxed tolerance per f16 precision).
- [ ] Microbenchmark: matmul (1024, 1024) × (1024, 1024) f16-vs-f32 on M-class. Target ≥1.5× speedup. Block if below 1.3×.

### Week 5 — Master-weights policy + integration with optimizer

- [ ] Add a `_promote_grads_to_master_dtype` step to `optim.{SGD,Adam,AdamW}.step()` so that even if a path delivered f16 grads, the optimizer rounds them up to f32 before the weight update.
- [ ] Verify that `loss.backward()` under autocast still produces f32 leaf grads via the cast pass.
- [ ] Integration test: 2-layer MLP, 200 training steps; assert weights converge in AMP to within 1% of fp32-only run on a synthetic regression task.

### Week 6 — Conformance fixture + benchmark + docs

- [ ] Add a new PyTorch-conformance fixture: train 4-layer CNN on MNIST 5K-sample shard, 5 epochs, fp32 vs AMP. Generate from real PyTorch (CUDA AMP) once; commit fixture; assert browsergrad converges to ≥95% test accuracy in both.
- [ ] Benchmark: transformer block (hidden=768, heads=12, seq=128, batch=4) forward+backward. Assert AMP ≥1.3× faster than fp32 on M2-class Mac with `shader-f16` enabled.
- [ ] Browser matrix: Chrome (default), Edge (default), Safari 26 (default), Firefox 145 (flag), Firefox stable (no f16 → asserts graceful fallback).
- [ ] Documentation: README section, primer note in PRIMER.md ("what mixed precision is"), changelog entry, deprecation note for `torch.cuda.amp`.
- [ ] Publish `@unlocalhosted/browsergrad-jit@0.4.0` and `@unlocalhosted/browsergrad-kernels@0.4.0`.

---

## Acceptance Criteria

| # | Criterion | Measurement |
|---|---|---|
| AC1 | `with autocast(...)` tags every UOp built inside the block; cast pass rewrites exactly the allowlist ops to f16 | `cast_pass_structure.test.ts` |
| AC2 | `shader-f16` feature detection returns `true` on Chrome/Edge/Safari, `false` otherwise; `is_available()` mirrors it | `f16_feature_compat.test.ts` |
| AC3 | When `shader-f16` is unavailable, `autocast.__enter__` is a no-op + a single per-session warning is logged | `fallback_loud.test.ts` |
| AC4 | f16 matmul kernel produces output within 1e-3 of fp32 matmul on shapes (256,256)@(256,256) and (1024,1024)@(1024,1024) | `f16_matmul_conformance.test.ts` |
| AC5 | f16 matmul uses fp32 accumulator (verifiable by stressing with K=4096 against an all-ones input — result should equal K within 0.5%, *not* underflow) | `f16_acc_stress.test.ts` |
| AC6 | `GradScaler` halves the scale on the first synthetic NaN injection and skips the optimizer step | `scaler_nan_skip.test.ts` |
| AC7 | `GradScaler` doubles the scale after `growth_interval` clean steps | `scaler_growth.test.ts` |
| AC8 | MNIST 4-layer CNN trains to ≥95% test accuracy on the 5K-shard fixture, in *both* fp32 and AMP modes | `amp_mnist_conformance.test.ts` |
| AC9 | AMP achieves ≥1.3× wall-clock speedup vs fp32 on hidden=768 transformer block forward+backward, on an M-series Mac with `shader-f16` enabled | `amp_speedup.bench.ts` |
| AC10 | Memory peak for the AMP CNN training is ≤55% of fp32 peak (activation halving + overhead) | `amp_memory.bench.ts` |
| AC11 | `torch.cuda.amp.autocast()` emits a `DeprecationWarning` once per process and dispatches to the new path | `legacy_alias.test.ts` |
| AC12 | All 234 existing integration tests pass with autocast disabled (default) — zero regression | full integration suite green |

---

## Test Strategy

### Unit tests (Vitest, no Pyodide, no GPU)

- `cast_pass_structure.test.ts` — synthetic UOp graphs in/out, assert the cast pass produces the expected node sequence for matmul→softmax, conv→relu→bn, etc.
- `collapse_redundant_casts.test.ts` — back-to-back allowlist ops produce a single f16 region.
- `scaler_state_machine.test.ts` — pure unit test of `GradScaler` adaptive logic with the realization layer mocked.

### Integration tests (Vitest + Pyodide-in-Node)

- `amp_mnist_conformance.test.ts` — the headline accept test. Runs both fp32 and AMP training of the same model architecture and asserts both reach the accuracy bar.
- `scaler_nan_skip.test.ts` — injects a synthetic +Inf into a loss tensor, asserts `scaler.step()` does not call `optimizer.step()` and that the scale halves.
- `scaler_growth.test.ts` — runs 2000 clean steps with `growth_interval=2000`, asserts scale doubles.
- `autocast_nesting.test.ts` — nested autocast blocks (inner `enabled=False`), assert the outer dtype is restored on exit.

### GPU integration tests (Playwright with WebGPU)

- `f16_matmul_conformance.test.ts` — real WebGPU, real `shader-f16`, real dispatch; numerical match.
- `f16_acc_stress.test.ts` — large-K matmul to detect missing fp32 accumulator.
- `fallback_loud.test.ts` — runs in a Firefox stable context (no f16 by default); asserts the warning fires and training still converges.

### Benchmarks (CI-tracked)

- `amp_speedup.bench.ts` — measured wall-clock on Apple M-series; reported in GH Actions summary; fails CI only if regression vs the baseline drops below 1.3× over a 7-day rolling window.
- `amp_memory.bench.ts` — peak resident GPU-buffer bytes via WebGPU `requestDevice({trackMemoryUsage: true})` when supported, else best-effort by enumerating allocated buffers.

---

## Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | f16 matmul on browsers without true f16 silicon (e.g., Intel iGPUs) runs *slower* than f32 because the driver emulates f16 | Medium | Medium | Microbenchmark on first dispatch; if AMP path measured > fp32 path on the device, log a one-time warning and short-circuit `autocast.__enter__` to no-op. Document the heuristic. |
| R2 | Loss-scale tuning differs from PyTorch defaults — students see different scale trajectories than the upstream tutorial | Low | Low | Match PyTorch's `init_scale=2**16`, `growth_interval=2000`, `growth_factor=2.0`, `backoff_factor=0.5` exactly. Document any deviation prominently. |
| R3 | Cast pass interacts badly with PRD-006 fusion: a fused elementwise chain containing a CAST node either produces wrong WGSL or breaks fusion | High | High | Run cast pass *before* fusion. Fusion treats `CAST` as a chain breaker initially (each cast forces a kernel boundary). PRD-012 unlocks cross-cast fusion. |
| R4 | Symbolic backward (PRD-007) produces gradients with wrong dtype if VJP rules don't carry through the cast — e.g., `MATMUL` VJP needs its `dy` cast back to f16 before the gradient matmul | High | High | Cast pass is run on *both* forward and backward IRs. Update PRD-007's `register_vjp` rules to propagate the autocast hint to VJP-emitted UOps. Adds one round-trip integration test. |
| R5 | NaN check via `ISNAN` + `REDUCE(OR)` realizes the entire backward graph at every step, blocking the natural laziness | Medium | Medium | Acceptable: PyTorch realizes the same gradient buffers anyway during the optimizer step. The cost is the same boolean reduction across all gradients (small). |
| R6 | `shader-f16` feature exists on the device but the WGSL compiler has bugs (early Safari 26 builds had silent miscompiles on `f16 * f32`) | Medium | High | Run a 16-vector smoke check on first device acquisition (`assert exp(f16(1.0)) ≈ 2.718` within tolerance); on failure, fall back to f32 with a warning naming the browser version. |
| R7 | `GradScaler.step()` calls `.item()` on a boolean tensor each iteration — a sync point that stalls the GPU pipeline | Medium | Medium | Acceptable in v1 (PyTorch has the same stall). Future optimization: batch the scaler decisions with the next forward pass via a pending-step queue. |
| R8 | f16 atomics not in WGSL — gradient accumulation across workers using atomic ops can't be done in f16 | Low | Low | We don't ship multi-worker grad accumulation in this PRD. The f32 grad slot path remains the only path. |
| R9 | Loss-scale BUFFER's `.set()` mid-trace triggers re-tracing on next forward (shape-based trace caches still hit; scalar-value changes don't invalidate) | Low | Medium | Confirmed: PRD-005's trace cache is keyed on shape+dtype, not on `BUFFER` value content. Add a regression test. |
| R10 | Pyodide overhead on `scaler.step()` becomes the bottleneck for tiny models where the GPU work is <1ms | Medium | Low | Document; the recipe is designed for hidden≥256 / batch≥8 workloads. Tutorial labs note this scale dependency. |

---

## Open Questions

1. **`CONV2D` as a first-class UOp.** PRD-005 left `conv2d` as an opaque `CUSTOM` UOp in v0 (PRD-005 Risk R1). For AMP to route conv through f16, conv must either become a proper UOp or the cast pass must learn to inspect the inside of `CUSTOM` UOps. **Resolution:** PRD-010 ships `CONV2D` as a proper opcode (the 21st), with f32 and f16 implementations. This is a pre-req task, ~1 week, slotted into week 1.

2. **Activation checkpointing interaction (PRD-009/P1.7).** When `gradient_checkpoint` re-runs forward during backward, does it re-enter `autocast` automatically? **Resolution:** yes — the checkpoint wrapper captures `_TLS.amp_dtype` at trace time and re-applies it during the re-run. Adds one test in PRD-009.

3. **Should AMP automatically promote `requires_grad` parameters created inside an autocast region to f32?** PyTorch does. **Resolution:** yes; `torch.nn.Parameter(...)` and `torch.zeros(..., requires_grad=True)` always allocate f32 storage regardless of the active autocast dtype. Document explicitly.

4. **Exposing `shader-f16` to user-written WGSL kernels (P2.5 / PRD-024).** When a course author writes a custom kernel and the device supports f16, do we pre-pend `enable f16;` for them? **Resolution:** opt-in via `@bg.kernel("wgsl", enable=["f16"])`. Default off — the user knows what dtype they want.

5. **Per-op autocast override.** PyTorch supports `with autocast(): out = op(...).float()` to force a specific op into f32. **Resolution:** `.float()` and `.half()` on `TensorProxy` already insert `CAST` nodes via PRD-005's IR. The cast pass respects user-inserted `CAST` nodes and does not undo them. Add a regression test.

6. **Sticky scale persistence.** Should `scaler.get_scale()` be saved in `state_dict()` so that resuming training picks up the converged scale? **Resolution:** yes — `GradScaler.state_dict()`/`load_state_dict()` ship alongside the rest, matching the [PyTorch GradScaler state_dict API](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.state_dict).

---

## References

1. **Mixed Precision Training paper.** Micikevicius, P., et al. "Mixed Precision Training." [arXiv:1710.03740](https://arxiv.org/abs/1710.03740) (2017). The founding work; defines loss scaling, master-weights-in-fp32, and the empirical convergence story.

2. **PyTorch AMP documentation.** [`torch.amp`](https://pytorch.org/docs/stable/amp.html). Specifies the device-typed autocast API and the per-op allowlist/blocklist.

3. **PyTorch AMP recipes / examples.** [Automatic Mixed Precision examples](https://pytorch.org/docs/stable/notes/amp_examples.html). The reference pattern for `scaler.scale(loss).backward()` → `scaler.step(optimizer)` → `scaler.update()`.

4. **NVIDIA's Mixed-Precision training guide.** ["Train With Mixed Precision"](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html). Best practices for loss scaling, layer/op selection, debugging numerical issues.

5. **WebGPU `shader-f16` extension.** W3C WGSL specification, [§Extension: f16](https://www.w3.org/TR/WGSL/#extension-f16). The authoritative spec for `enable f16;`, `f16`/`vec*<f16>`/`mat*<f16>` types, and conversion rules.

6. **WebGPU API: requiredFeatures.** [W3C WebGPU §requestDevice](https://www.w3.org/TR/webgpu/#dom-gpuadapter-requestdevice). Defines how `shader-f16` is requested at device creation.

7. **Chrome `shader-f16` shipping note.** [Chrome Status — WebGPU: shader-f16](https://chromestatus.com/feature/5180552310751232). Ships in M113 (April 2023); on by default since then. Confirms no `chrome://flags` toggle required in 2026.

8. **Safari 26 release notes.** [Apple WebKit, Safari 26.0](https://developer.apple.com/documentation/safari-release-notes/safari-26-release-notes). Default support for WebGPU and `shader-f16` on Apple Silicon.

9. **Firefox WebGPU status.** [Bugzilla 1746732 — WebGPU shader-f16](https://bugzilla.mozilla.org/show_bug.cgi?id=1746732). Behind `dom.webgpu.f16.enabled` in 145 as of May 2026.

10. **PRD-005 IR opcode set.** This PRD adds three opcodes (`CONV2D`, `ISNAN`, `OR`) on top of PRD-005's 19. `CAST` already exists.

11. **PRD-006 WGSL codegen.** Fused-kernel emitter extended in week 4 of this PRD to support the `enable f16;` header and `f16`/`vec4<f16>` body types.

12. **PRD-007 VJP rules.** Risk R4 mandates that VJP-emitted UOps inside an `autocast` region carry the same autocast hint and run through the cast pass on the backward side.

13. **PyTorch GradScaler implementation.** [`torch/amp/grad_scaler.py`](https://github.com/pytorch/pytorch/blob/main/torch/amp/grad_scaler.py). Reference for the state machine semantics our IR-native scaler implements.

14. **WGSL specification (overall).** [W3C WGSL](https://www.w3.org/TR/WGSL/). For type-promotion rules between `f16` and `f32` in expressions and function calls.

15. **PRD.md §7 P1.8.** The feature spec this PRD operationalizes; locks in the ≥1.5× matmul speedup expectation (relaxed here to ≥1.3× as a CI gate, with ≥1.5× as a stretch goal).

16. **VISION.md §6 ("Don't build — mixed precision").** Now reversed by this PRD; the original "no-op" stance applied to the eager NumPy era and is obsolete now that PRD-005's IR makes a real implementation tractable.
