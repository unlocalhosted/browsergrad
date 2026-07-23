# @unlocalhosted/browsergrad-jit

[![npm version](https://img.shields.io/npm/v/@unlocalhosted/browsergrad-jit.svg)](https://www.npmjs.com/package/@unlocalhosted/browsergrad-jit)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A curated PyTorch-shaped Python tensor library that runs in the browser via Pyodide. **Lazy by default** — arithmetic builds a UOp graph; nothing executes until you call `.numpy()`, `.tolist()`, `.item()`, `.backward()`, or `optimizer.step()`.

The IR is the substrate for everything downstream: fusion, symbolic backward, AMP cast-insertion, gradient checkpointing IR rewrites, functional transforms (`grad`/`vjp`/`vmap`/`functional_call`), ONNX export, custom WGSL kernels, and pluggable backends (NumPy today, WebGPU via the realizer bridge).

## Install

The core package installs Python sources into any Pyodide-shaped target and
does not require a particular execution backend:

```sh
npm install @unlocalhosted/browsergrad-jit
```

Install the optional peer that owns the integration you use:

```sh
# Direct Node/Pyodide usage through the exported node adapter
npm install @unlocalhosted/browsergrad-jit pyodide

# Production WebGPU realization through the BrowserGrad bridge
npm install @unlocalhosted/browsergrad-jit @unlocalhosted/browsergrad-kernels
```

`@unlocalhosted/browsergrad-kernels` supplies the production WebGPU bridge and
its compatible semantic-core dependency. JIT intentionally does not declare a
second semantic-core peer: it emits the versioned bridge request wire without
importing semantic-core at runtime. Package managers therefore do not install
semantic-core through JIT, and both optional peers may remain absent for
core-only consumers.

## Hello world

```python
import browsergrad_jit as bg
import numpy as np

bg.manual_seed(0)
model = bg.nn.Sequential(
    bg.nn.Linear(8, 16),
    bg.nn.ReLU(),
    bg.nn.Linear(16, 4),
)
opt = bg.optim.SGD(model.parameters(), lr=0.01)

x = bg.from_numpy(np.random.randn(32, 8).astype(np.float32))
y = bg.from_numpy(np.random.randn(32, 4).astype(np.float32))

for _ in range(10):
    opt.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()
```

## Public surface

### Tensor core
- `TensorProxy` (alias `Tensor`) — lazy tensor
- Factory: `tensor`, `from_numpy`, `zeros`, `ones`, `randn`, `arange`
- Arithmetic + reductions + shape ops + comparisons + dtype casts
- Metadata and graph boundaries: `shape`, `dtype`, `ndim`, `numel`,
  `requires_grad`, `is_leaf`, `grad_fn`, `detach()`, `clone()`
- Autograd: `.backward()`, `.grad`

Known tensor gaps: `device`, `is_contiguous()`, and tensor indexing
(`__getitem__`) are not implemented yet. They fail loudly instead of
pretending to match PyTorch.

### Neural networks
- `nn.Module`, `nn.Sequential`
- `nn.Linear`, `nn.Conv1d`, `nn.Conv2d`, `nn.ConvTranspose2d`, `nn.Conv3d`,
  `nn.Dropout`, `nn.LayerNorm`, `nn.BatchNorm1d`, activation modules
- `nn.Module` parameters, buffers, `state_dict()`, `load_state_dict()`,
  `train()`, `eval()`, `zero_grad()`
- `nn.functional`: `relu`, `sigmoid`, `tanh`, `gelu`, `softmax`,
  `cross_entropy`, `mse_loss`, `nll_loss`, `linear`, `layer_norm`,
  `conv1d`, `conv2d`, `conv_transpose2d`, `conv3d`, `interpolate`
- `optim.SGD`, `optim.Adam`, `optim.AdamW`

`Conv1d`, `Conv2d`, `ConvTranspose2d`, `Conv3d`, and `LayerNorm` are primitive
IR ops with NumPy handlers and symbolic VJPs. Their forward/backward roots
lower through generic tensor-plan WebGPU via `realize_tensor_plan_webgpu(...)`.
`nn.functional.scaled_dot_product_attention(...)` stays in primitive tensor IR:
`MATMUL` -> scale -> `FUSED_SOFTMAX` -> `MATMUL`, with no `CUSTOM` callback.
`loss.backward(device="webgpu")` realizes symbolic leaf-gradient roots through
the same tensor-plan bridge and refuses closure-only graphs instead of falling
back to CPU. `loss.backward(device="webgpu", resident=True)` stores leaf grads
as GPUBuffer-backed TensorProxy values until explicit `.numpy()` / `.item()`.
Default `.backward()` keeps CPU semantics for CPU-owned graphs, but selects the
resident WebGPU path when the graph already reads GPU-owned buffers.
`BatchNorm1d` uses typed forward, VJP, and exactly-once running-state IR. Its
v1 profile is float32 CPU execution with explicit vmap, ONNX, tensor-plan, and
WebGPU refusals until those boundaries own normalization/state semantics.
`nn.functional.interpolate` uses typed spatial-resampling forward and VJP IR.
Its v1 profile supports bounded rank-4 float16/32/64 nearest and bilinear CPU
execution, closure/symbolic gradients, leading-axis vmap, checkpoint replay,
and ONNX `Resize`; tensor-plan and WebGPU execution refuse until canonical
resampling layout and kernel semantics exist.
`bg.optim.sgd_update(...)`, `bg.optim.adam_update(...)`, and
`bg.optim.adamw_update(...)` are functional optimizer/update IR nodes and lower
through the same tensor-plan WebGPU path. They return updated tensors/state;
`Optimizer.step(device="webgpu")` uses those IR nodes for SGD without momentum,
Adam, and AdamW, then writes the materialized result back to CPU parameter/state
buffers. `SGD.step(device="webgpu", resident=True)` keeps no-momentum parameter
updates GPU-resident. `Adam.step(device="webgpu", resident=True)` and
`AdamW.step(device="webgpu", resident=True)` keep parameter, first-moment, and
second-moment buffers resident.

`bg.gpu_plan_summary(tensor)` and `bg.jit.gpu_plan.*` expose the first
compiler-facing tensor-IR execution plan: scheduled primitive UOps, buffer
liveness bytes, one explicit root materialization boundary, and the first
scheduler/codegen choice: linear elementwise chains become one
`FUSED_ELEMENTWISE` plan step, while canonical softmax DAGs become one
`FUSED_SOFTMAX` plan step. It refuses `CUSTOM` by default, keeping user/lab
kernels separate from core framework GPU lowering.

`bg.realize_tensor_plan_webgpu(tensor)` sends that canonical plan through one
generic WebGPU bridge call (`run_tensor_plan`) instead of walking legacy per-op
bridge methods. `bg.realize_tensor_plan_webgpu_resident(tensor)` uses
`run_tensor_plan_resident` and returns a TensorProxy whose root stays in a
registered GPUBuffer until `.numpy()` / `.item()` materialization. Current
runtime coverage is f32 BUFFER/LOAD/2-D MATMUL, `FUSED_ELEMENTWISE`
runtime WGSL codegen, `FUSED_SOFTMAX` last-axis direct softmax, RESHAPE,
PERMUTE, BROADCAST_TO, and REDUCE(sum/mean) rank <= 4, plus
Conv1d/Conv2d/ConvTranspose2d/Conv3d/LayerNorm forward/backward and functional
SGD/Adam/AdamW updates. `Optimizer.step(device="webgpu")` uses the same
tensor-plan path for SGD without momentum, Adam, and AdamW, then writes the
materialized result back to CPU buffers; `SGD.step(device="webgpu",
resident=True)` keeps the updated parameter buffer resident, and resident
Adam/AdamW keeps m/v state resident too.

`Tensor.expand(...)` now emits the typed `BROADCAST_TO` primitive rather than
an opaque NumPy callback. CPU realization returns an owning dtype-preserving
array; closure backward and symbolic VJP both sum expanded axes; vmap and ONNX
map the same node; and the materializing/resident tensor-plan routes accept its
current f32 rank-at-most-four WebGPU profile. Invalid, non-integral, rank-
reducing, or incompatible shapes fail before execution.

`Tensor.abs()` and `Tensor.sign()` emit typed `ABS` and `SIGN` primitives.
They preserve the exact shape and real numeric dtype, reject bool before
execution, return owning CPU arrays, support closure and symbolic gradients,
leading-axis vmap, and direct ONNX export. Their current backend profile is
host-materialized only: tensor-plan and WebGPU execution fail explicitly until
a portable lowering and kernel are admitted.

`Tensor.sin()` and `Tensor.cos()` emit typed `SIN` and `COS`. They accept
float16, float32, and float64 only, preserve exact shape/dtype, return owning
CPU arrays, and provide mutually typed closure/symbolic gradients, functional
grad, leading-axis vmap, and direct ONNX export. Integer and bool inputs fail
before execution instead of realizing a dtype different from the graph. Their
tensor-plan/WebGPU profile is also an explicit refusal.

`Tensor.clamp()` emits typed `CLAMP` for finite floating bounds and
float16/32/64 tensors. Bounds are normalized without invoking arbitrary scalar
conversion hooks; CPU returns an owning dtype-preserving array; closure and
symbolic gradients use the same inclusive-bound mask; vmap and ONNX `Clip`
consume the closed contract. `clip()` and `clamp_min()` reuse the same typed
builder. Integer inputs and hostile/non-finite bounds fail before execution,
and tensor-plan/WebGPU remain explicit refusals.

`Tensor.flip()` emits typed `FLIP` for one built-in or NumPy integer axis.
Negative axes normalize once; bool, floating, hostile-conversion, scalar-rank,
and out-of-range axes fail before execution. CPU realization is an owning
dtype-preserving copy, closure and symbolic gradients apply the same
involutive reversal, vmap shifts the logical axis past its leading batch axis,
and ONNX emits a signed-int64 `Slice` for float32, int32, int64, and bool
graphs. Other ONNX dtypes fail explicitly. Tensor-plan and WebGPU explicitly
refuse the negative-stride profile rather than widening the portable view contract.

`Tensor.gather()` emits typed `INDEX` for one normalized axis and an int64
same-rank tensor index. Non-gather index extents may be smaller than the source;
negative and out-of-range index values fail at the first value-observing
boundary. CPU returns an owning source-dtype result, closure and symbolic VJP
use deterministic duplicate-accumulating scatter-add, and paired vmap shifts
the logical axis past its leading batch dimension. ONNX emits
`GatherElements` for float32, int32, int64, and bool source graphs. Tensor-plan
and WebGPU explicitly refuse until a deterministic bounds-checked index/scatter
lowering exists.

`Tensor.masked_fill()` emits the canonical typed `WHERE(mask, fill, source)`
selection. The mask must be a bool tensor that broadcasts into, but never
enlarges, the source shape; the fill is normalized once into an exact scalar
constant of the source dtype without invoking conversion hooks. CPU returns an
owning source-dtype result, closure and symbolic gradients route through the
mask complement, and vmap supports a leading mapped source with a captured or
mapped broadcast mask. ONNX emits `Where` for float32, int32, int64, and bool.
Tensor-plan and WebGPU explicitly refuse until portable masked selection
exists.

`Tensor.tril()` and top-level `tril(input, diagonal)` emit typed `TRIL` over
the final two matrix axes. Inputs require rank at least two and a supported
real-numeric or boolean dtype. The diagonal must be an exact built-in or NumPy
integer and is saturated into the matrix's unique closed all-zero/all-input
semantic range before entering IR, so hostile conversion hooks and oversized
integers never reach NumPy or export. CPU returns an owning dtype-preserving result;
closure and symbolic VJP apply the same idempotent triangular selection; vmap
preserves the last two matrix axes while inserting its leading batch axis.
ONNX emits `Trilu` with `upper=0` for float32, int32, int64, and bool.
Tensor-plan and WebGPU explicitly refuse until portable triangular selection
exists.

`Tensor.triu()` and top-level `triu(input, diagonal)` emit the corresponding
typed `TRIU` over the same final two axes and share the exact rank, dtype, and
non-coercive diagonal admission contract with `TRIL`. Its canonical nonempty
diagonal range is `[1 - rows, columns]`, spanning the unique all-input through
all-zero representatives. CPU, closure/symbolic VJP, and leading-axis vmap
apply the same idempotent upper selection. ONNX emits `Trilu` with `upper=1`
for float32, int32, int64, and bool; tensor-plan and WebGPU retain the same
explicit portable-lowering refusal.

`Tensor.repeat()` emits typed `REPEAT` for bounded exact integer tile
multipliers. CPU realization returns an owning dtype-preserving array;
closure and symbolic gradients sum interleaved tile blocks; vmap prepends a
unit multiplier so it never repeats the batch axis. ONNX emits `Tile` for
float32, int32, int64, and bool graphs. Tensor-plan and WebGPU explicitly
refuse the operation until a canonical tile/index layout profile exists.

`Tensor.repeat_interleave()` emits typed `REPEAT_INTERLEAVE` for one bounded
exact repeat count and normalized selected axis. CPU realization returns an
owning dtype-preserving array; closure and symbolic gradients sum each
selected-axis repeat block; vmap shifts the logical axis past its batch axis.
ONNX emits an exact `Unsqueeze`/`Tile`/`Reshape` decomposition for float32,
int32, int64, and bool graphs. Tensor-plan and WebGPU explicitly refuse the
operation until a canonical selected-axis replication profile exists.

`Tensor.prod()` emits typed `PROD` for a canonical static set of normalized
reduction axes and an exact boolean keepdims flag. CPU realization always
returns an owning input-dtype array, including scalar results. Closure and
symbolic gradients correctly distinguish zero-free, one-zero, and multi-zero
reduction groups; vmap keeps the leading batch axis outside the reduction.
ONNX emits `ReduceProd` for float32, int32, and int64 graphs. Tensor-plan and
WebGPU explicitly refuse the operation until a portable product-reduction
lowering exists.

`Tensor.var()` emits typed `VAR` for canonical static reduction axes, an exact
signed 32-bit correction, and an exact keepdims flag. It accepts float16,
float32, and float64, preserves dtype in owning CPU scalar/tensor results, and
supports the legacy boolean `unbiased` alias only when `correction` is absent.
Closure and symbolic gradients use the centered correction-aware derivative;
vmap shifts reduction axes past the leading batch. ONNX opset 17 decomposes
float32 variance into `ReduceMean`/`Sub`/`Mul`/`ReduceSum`/`Div`; other dtypes
fail export explicitly. Tensor-plan and WebGPU refuse until a portable
variance-reduction lowering exists.

`bg.framework_operation_support()` returns the versioned public decision table
for framework operations admitted to typed execution. The table is generated
from the same package registry that binds executable validators at CPU,
autograd, transform, export, and plan boundaries; each call returns detached
data. Backend entries name eligibility profiles only. Check runtime/device
availability and execution evidence separately. Operations absent from this
table have no admitted typed framework-operation contract.

JavaScript platforms can consume the same registry without launching Pyodide:

```ts
import {
  frameworkOperationSupport,
  frameworkPlatformSupportSource,
} from "@unlocalhosted/browsergrad-jit";

const table = frameworkOperationSupport();
const source = frameworkPlatformSupportSource();
```

Both functions return detached records. The platform source maps the exact
contract/opcode/legacy identities into runtime's framework-neutral input shape;
it does not infer device availability, program lowering, or execution
evidence.

### Gradient control
```python
with bg.no_grad():
    validation_logits = model(x)

with bg.inference_mode():
    probs = bg.sigmoid(validation_logits)
```

The PyTorch alias exposes the same contexts as `torch.no_grad()` and
`torch.inference_mode()`.

### Mixed precision
```python
with bg.amp.autocast(device_type="webgpu", dtype="float16"):
    pred = model(x)
    loss = ((pred - y) ** 2).mean()

scaler = bg.amp.GradScaler()
scaler.scale(loss).backward()
scaler.step(opt); scaler.update()
```

### Gradient checkpointing
```python
from browsergrad_jit.utils.checkpoint import checkpoint

def block(x): return model_layers(x)
y = checkpoint(block, x)   # forward intermediates recomputed at backward
```

### Data loading
```python
from browsergrad_jit.utils.data import TensorDataset, DataLoader

ds = TensorDataset(x, y)
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
```

`DataLoader` is intentionally single-process in Pyodide. `num_workers > 0`
raises with a browser-specific explanation.

### Functional transforms
```python
g = bg.func.grad(lambda t: (t * t).sum())(x)
out, vjp_fn = bg.func.vjp(lambda t: t * 2.0, x)
per_sample = bg.func.vmap(lambda t: t.sum())(batched)
out = bg.func.functional_call(model, {"weight": w, "bias": b}, (x,))
```

### Save / load
```python
state = {"w1": model[0].weight, "b1": model[0].bias}
blob = bg.save_safetensors(state)        # bytes — browser-friendly
restored = bg.load_safetensors(blob)
model[0].weight = restored["w1"]          # from_numpy accepts TensorProxy
```

The PyTorch alias also exposes minimal `torch.save()` / `torch.load()` for
pickle-safe BrowserGrad objects such as state dicts. It is not a full PyTorch
checkpoint compatibility layer.

### ONNX export
```python
y = (x @ w + b).relu()
onnx_bytes = bg.onnx.export_inference(y, input_buffers=(x,))
```

Hand-rolled pure-Python proto3 encoder. No `onnx` wheel needed. Opcodes outside the supported set raise `bg.onnx.OnnxUnmappableOp`.

### WebGPU realizer
```python
bg.register_webgpu_bridge(bridge)         # bridge built by browsergrad-kernels
out = bg.realize_webgpu(x @ w + b)        # ndarray, materialized at the seam
```

Explicit materialization in v0. Supported opcodes: `BUFFER`, `LOAD`, `CONST`,
`CAST`, `MATMUL`, `CONV1D`, `CONV1D_BACKWARD_INPUT`,
`CONV1D_BACKWARD_WEIGHT`, `CONV1D_BACKWARD_BIAS`, `CONV2D`,
`CONV2D_BACKWARD_INPUT`, `CONV2D_BACKWARD_WEIGHT`, `CONV2D_BACKWARD_BIAS`,
`FUSED_ELEMENTWISE`, `CUSTOM`. The newer tensor-plan path additionally covers
ConvTranspose2d/Conv3d/LayerNorm/optimizer-update roots. Supported `CUSTOM`
bridge ops include FlashAttention forward and
custom WGSL kernels. Other opcodes raise with a pointer back to `bg.realize()`
(NumPy).

### Custom WGSL kernel
```python
double_each = bg.custom_kernel(
    wgsl="...",
    name="double_each",
    workgroup_size=(64, 1, 1),
    output_shape_fn=lambda s0: s0,
    dispatch_shape_fn=lambda s0: (s0[0], 1, 1),
    num_inputs=1,
)
y = double_each(x)
out = bg.realize_webgpu(y)
```

SHA-256 of the WGSL is the pipeline cache key. Forward-only.

### Lab harness (optional)
```python
bg.lab.assert_pytorch_match("forward_correct", actual, expected, rtol=1e-4)
bg.lab.assert_shape_match("shape_ok", t, (32, 10))
bg.lab.assert_no_nan_inf("clean_grads", w_grad)
```

Routes through the runtime's structured assertion protocol when run inside `@unlocalhosted/browsergrad-runtime`; falls back to structured stdout otherwise.

## PyTorch alias

```python
bg.install_torch_alias()
import torch, torch.nn, torch.func, torch.amp, torch.utils.checkpoint, torch.utils.data
```

The shim covers the curated BrowserGrad subset of `torch.nn`, `torch.optim`,
`torch.nn.functional`, `torch.func`, `torch.amp`, `torch.utils.checkpoint`, and
`torch.utils.data`. It also exposes dtype tokens such as `torch.float32` and
`torch.int64`, plus `torch.save` / `torch.load` for BrowserGrad state dicts.
Anything not implemented raises `AttributeError`, not silent wrong behavior.

## Coexists with browsergrad-grad

Both libraries can be installed in the same Pyodide worker. They mount to distinct `sys.path` entries and the `torch` alias uses an owner-token protocol so calling `install_torch_alias()` from both raises a clear error rather than silently overwriting.

## Compatibility contract

| Surface | Stability |
|---|---|
| `TensorProxy` attributes & methods | Semver-stable across `0.x` |
| `nn.*`, `optim.*`, `nn.functional.*` shapes | Semver-stable |
| `bg.func.*`, `bg.amp.*`, `bg.utils.checkpoint.*`, `bg.utils.data.*`, `bg.onnx.*`, `bg.kernels.*`, `bg.custom_kernel`, `bg.lab.*` | Semver-stable |
| `bg.realize_webgpu`, `bg.register_webgpu_bridge`, `bg.webgpu_supported_opcodes()` | Semver-stable; supported-opcode set may grow |
| Public errors (`ShapeError`, `JitError`, `JitNotImplementedError`, `NoBackwardError`, `TorchAliasConflict`, `RealizationError`, `BufferTableError`, `OnnxUnmappableOp`) | Semver-stable |
| Per-opcode numerical match vs `browsergrad-grad` | Within `1e-4` (fp32) |
| `_ir`, `_realize`, opcode strings, `UOp` dataclass | **Internal.** Changes freely. |
| IR serialization, trace cache format | **Not promised.** Do not depend on these on disk. |

Anything in the **Internal** row will break across minor releases. File an issue if you need an internal surface lifted to public.

### Explicit non-goals in current release

- Full PyTorch API parity.
- CUDA device emulation.
- Fully scheduled GPU-resident training loop. Explicit
  `loss.backward(device="webgpu", resident=True)` and
  `Optimizer.step(device="webgpu", resident=True)` keep supported grads,
  params, and optimizer state in GPUBuffer storage until explicit `.numpy()` /
  `.item()`; default `.backward()` / `.step()` select that resident path when
  inputs are already GPU-owned. Full memory-planned training-loop scheduling
  remains future work.
- Tensor layout/stride compatibility for contiguity teaching.
- PyTorch-native checkpoint file compatibility.

## Testing

```sh
pnpm test                        # surface tests (no Pyodide)
pnpm test:integration            # Python correctness via Pyodide-in-node
pnpm run build
pnpm run typecheck
pnpm run lint
```

180 integration scenarios cover the supported public surface (training loops,
gradient checkpointing, AMP, ONNX, functional transforms, PyTorch alias
compatibility, etc.).

## License

[MIT](LICENSE).
