# Status

Living document. Reflects the current state of each package, what's tested, and what's deliberately deferred.

## Package versions

| Package | Version | Surface tests | Integration tests | Browser tests |
|---|---|---|---|---|
| `@unlocalhosted/browsergrad-runtime` | `0.1.1` | 27 | 23 (Pyodide-in-node) | — |
| `@unlocalhosted/browsergrad-kernels` | `0.1.2` | 35 (incl. JS-reference numerical checks, FUSED WGSL codegen) | — | 42 (real Chromium + WebGPU) |
| `@unlocalhosted/browsergrad-grad` | `0.5.1` | 30 | 321 (Pyodide-in-node) | — |
| `@unlocalhosted/browsergrad-jit` | `0.8.2` | 8 | 223 (Pyodide-in-node, incl. feedback + perf benches) | — (via kernels) |

`browsergrad-grad` current local gates: 30 surface/unit tests and 321
Pyodide-in-node integration tests green. Workspace-wide totals drift as
compiler/WebGPU bug-bash work lands; prefer package-level commands as source of
truth.

Every gradient verified against finite differences or a hand-derived oracle. Every realizer numerical result verified against a NumPy or JS-reference oracle. No test compares the implementation against itself.

## Two-library story

- `browsergrad-grad` is the **closure-autograd** library. PyTorch-shaped, NumPy-backed, eager. Used today by curriculum content. Stable.
- `browsergrad-jit` is the **lazy IR** successor. Same PyTorch surface, but ops build a UOp graph that gets realized through a backend (NumPy today, WebGPU via PRD-011.5's seam). Fusion + symbolic backward + AMP + gradient checkpointing + functional transforms + ONNX export + custom WGSL kernels all live here.

Both ship under the same `unlocalhosted` npm scope. They can coexist in the same Pyodide session (separate `install_torch_alias()` namespaces, owner-token protocol prevents collision).

GPU-native target: keep `grad` as educational CPU/reference baseline; make
`jit` the canonical tensor IR owner; promote core ops out of permanent
`CUSTOM` callbacks into primitive tensor IR; compile tensor IR to WebGPU; keep
CUDA-lite compiler focused on labs and user kernels. `.numpy()` / `.item()` are
the intended GPU materialization boundaries.

## Surface inventory — `browsergrad-jit` (v0.8.0)

### Core (PRD-005)
- 55-opcode IR (`_ir.py`): BUFFER, LOAD, STORE, CONST, RANDOM, CAST, ADD, MUL, DIV, NEG, EXP, LOG, CMP, MATMUL, CONV1D, CONV1D_BACKWARD_INPUT, CONV1D_BACKWARD_WEIGHT, CONV1D_BACKWARD_BIAS, CONV2D, CONV2D_BACKWARD_INPUT, CONV2D_BACKWARD_WEIGHT, CONV2D_BACKWARD_BIAS, CONV_TRANSPOSE2D, CONV_TRANSPOSE2D_BACKWARD_INPUT, CONV_TRANSPOSE2D_BACKWARD_WEIGHT, CONV_TRANSPOSE2D_BACKWARD_BIAS, CONV3D, CONV3D_BACKWARD_INPUT, CONV3D_BACKWARD_WEIGHT, CONV3D_BACKWARD_BIAS, LAYER_NORM, LAYER_NORM_BACKWARD_INPUT, LAYER_NORM_BACKWARD_WEIGHT, LAYER_NORM_BACKWARD_BIAS, REDUCE, RESHAPE, PERMUTE, SLICE, PAD, WHERE, INDEX, MASK, CUSTOM, FUSED_ELEMENTWISE, FUSED_SOFTMAX, SCATTER_ADD, BROADCAST_TO, ISNAN, SGD_UPDATE, ADAMW_UPDATE_M, ADAMW_UPDATE_V, ADAMW_UPDATE_PARAM, ADAM_UPDATE_M, ADAM_UPDATE_V, ADAM_UPDATE_PARAM.
- `TensorProxy` — lazy tensor; metadata never realizes; arithmetic builds IR; `.numpy()` / `.item()` triggers realize.
- `Session` + `BufferTable` for per-tab isolation.
- NumPy realizer (`_realize.py`): single dispatch table; one Python function per opcode; deterministic across runs.

### Fusion (PRD-006)
- Elementwise chain fusion → `OP_FUSED_ELEMENTWISE`.
- Softmax DAG → `OP_FUSED_SOFTMAX`.
- Introspection: `bg.jit.debug_fused_kernels()`, `bg.jit.debug_unfused_reasons()`.

### Symbolic backward (PRD-007)
- 17 VJP rules (ADD, MUL, DIV, NEG, EXP, LOG, CAST, MATMUL, CONV1D, CONV2D, CONV_TRANSPOSE2D, CONV3D, LAYER_NORM, REDUCE, RESHAPE, PERMUTE, ISNAN).
- Closure-backward kept as the safety net for ops without VJPs.
- `arg["vjp_of"]` tags every emitted backward node — observable by checkpointing.

### Trace cache + safetensors (PRD-008)
- In-memory trace cache (`_trace_cache.py`); hit ratio observable via `bg.jit.trace_cache_stats()`.
- `bg.save_safetensors(state) -> bytes` (browser-friendly), `bg.load_safetensors(blob)`.

### Gradient checkpointing (PRD-009)
- `bg.utils.checkpoint.checkpoint(fn, *args)` (and `torch.utils.checkpoint.checkpoint`).
- IR rewrite at backward time: interior UOps are re-cloned from anchor inputs and re-realized.

### Mixed precision (PRD-010)
- `bg.amp.autocast(device_type, dtype, enabled)` context manager. Tags forward UOps with `arg["autocast_hint"]`.
- Cast-insertion IR pass (`_amp.insert_cast_pass`) wraps tagged ops with explicit CASTs per the ALLOWLIST_F16 / BLOCKLIST_F32 / PROMOTE_OPS policy.
- `bg.amp.GradScaler` — real loss scaler with NaN-triggered backoff + growth-interval doubling.
- `_h_matmul` runs `f16 @ f16` with fp32 accumulator (tensor-core semantics).

### GPUBuffer-backed WGSL realizer (PRD-011.5)
- `bg.realize_webgpu(tensor)` — explicit-realize through the registered bridge.
- `bg.register_webgpu_bridge(bridge)` — pluggable; bridge owns GPUBuffer lifetimes.
- `bg.gpu_plan_summary(tensor)` / `bg.jit.gpu_plan.*` — compiler-facing
  tensor-IR plan scaffold. It records schedule steps, liveness bytes, and the
  single root materialization boundary; it refuses `CUSTOM` by default so core
  framework GPU work cannot hide behind callback escape hatches. The planner
  now folds linear elementwise chains into `FUSED_ELEMENTWISE` schedule steps
  and canonical softmax DAGs into `FUSED_SOFTMAX` schedule steps.
- Whitelisted opcodes: BUFFER, LOAD, CONST, CAST, MATMUL, FUSED_ELEMENTWISE,
  FUSED_SOFTMAX, CONV1D,
  CONV1D_BACKWARD_INPUT, CONV1D_BACKWARD_WEIGHT, CONV1D_BACKWARD_BIAS,
  CONV2D, CONV2D_BACKWARD_INPUT, CONV2D_BACKWARD_WEIGHT,
  CONV2D_BACKWARD_BIAS, CONV_TRANSPOSE2D,
  CONV_TRANSPOSE2D_BACKWARD_INPUT, CONV_TRANSPOSE2D_BACKWARD_WEIGHT,
  CONV_TRANSPOSE2D_BACKWARD_BIAS, CONV3D, CONV3D_BACKWARD_INPUT,
  CONV3D_BACKWARD_WEIGHT, CONV3D_BACKWARD_BIAS, LAYER_NORM,
  LAYER_NORM_BACKWARD_INPUT, LAYER_NORM_BACKWARD_WEIGHT,
  LAYER_NORM_BACKWARD_BIAS, optimizer update ops, CUSTOM.
- `bg.kernels.attention_forward(Q, K, V)` — typed, bounded rank-4 float32
  attention-forward IR with stable CPU semantics and a legacy row-wise
  online-softmax direct-WebGPU route. `bg.kernels.flash_attention` is a
  compatibility alias; it does not claim the Gate 5 block-tiled algorithm or
  FlashAttention-v2.
- `nn.functional.conv1d(...)` / `nn.Conv1d(...)`,
  `nn.functional.conv2d(...)` / `nn.Conv2d(...)`,
  `nn.functional.conv_transpose2d(...)` / `nn.ConvTranspose2d(...)`, and
  `nn.functional.conv3d(...)` / `nn.Conv3d(...)` are primitive IR ops and can
  route forward/backward gradient roots through generic tensor-plan WebGPU.
- `nn.functional.scaled_dot_product_attention(...)` decomposes into primitive
  tensor IR (`MATMUL` -> scale -> `FUSED_SOFTMAX` -> `MATMUL`) and lowers
  through the same tensor-plan WebGPU path without `CUSTOM`.

### CNN parity in JIT
- `nn.Conv1d`, `nn.Conv2d`, `nn.ConvTranspose2d`, and `nn.Conv3d` are available as primitive
  forward/backward IR with CPU handlers, symbolic VJPs, explicit vmap refusals,
  and ONNX refusals.
- `bg.realize_tensor_plan_webgpu(...)` lowers Conv1d/Conv2d/ConvTranspose2d/Conv3d
  forward/backward through the generic tensor-plan bridge (`run_tensor_plan`)
  instead of legacy per-op conv bridge methods.
- `loss.backward(device="webgpu")` realizes symbolic leaf-gradient roots through
  the tensor-plan bridge and refuses closure-only graphs instead of falling
  back to CPU. `resident=True` stores leaf `.grad` tensors as GPUBuffer-backed
  TensorProxy values until explicit `.numpy()` / `.item()`. Default
  `.backward()` selects this resident WebGPU path when the graph reads
  GPU-owned buffers.
- Covered semantics: tuple stride/padding/dilation where applicable, groups,
  output padding for ConvTranspose2d, module state_dict keys, and torch alias
  exposure.
- Conv1d/Conv2d/ConvTranspose2d/Conv3d/LayerNorm forward/backward have generic tensor-plan WebGPU
  coverage.
- `bg.optim.sgd_update(...)`, `bg.optim.adam_update(...)`, and
  `bg.optim.adamw_update(...)` emit functional
  optimizer/update IR and lower through generic tensor-plan WebGPU. They do
  not mutate params/state.
- `Optimizer.step(device="webgpu")` uses the same update IR for SGD without
  momentum, Adam, and AdamW, then writes the materialized result back to CPU
  parameter/state buffers. `SGD.step(device="webgpu", resident=True)` keeps
  no-momentum parameter updates GPU-resident; `Adam.step(device="webgpu",
  resident=True)` and `AdamW.step(device="webgpu", resident=True)` keep
  params/m/v state resident. Default `.step()` selects the resident WebGPU path
  when params/grads are already GPU-owned.
- Perf hardening left for future throughput work: broaden scheduler/codegen
  beyond direct kernels and current elementwise-chain, softmax, and
  SDPA-shaped graph fusion.

### Tiled GEMM + fused codegen + GPU cast (PRD-012a)
- `matmulTiledDirect` — 16×16 tiled GEMM (workgroup-shared A/B tiles). Closes most of the gap PRD-012 was claiming via "megakernels".
- `fusedElementwiseDirect` — runtime WGSL codegen. Walks the ops list, emits a single compute shader, hashed for pipeline cache.
- `cast` (f32→f32) via `CopyBufferToBuffer` — true GPU-only copy, no host round-trip.

### Lab platform alignment (PRD-013)
- `LabManifest` schema + `parseManifest(json)` validator (no ajv dep, hand-written).
- `isSemverCompatible(range, version)` + `assertCompatibleRuntime` with `LabRuntimeMismatch`.
- `bg.lab.{assert_pytorch_match, assert_shape_match, assert_no_nan_inf}` — semantic harness primitives that route through the runtime's `browsergrad` module.

### Functional transforms (PRD-014 + partial 014b)
- `bg.func.grad(fn, argnums)` — functional gradient. Does NOT write `.grad`. Returns lazy TensorProxy.
- `bg.func.vjp(fn, *primals)` — outputs + vjp_fn for vector-valued backward.
- `bg.func.functional_call(module, params_dict, args, kwargs)` — stateless module evaluation.
- `bg.func.vmap(fn)` — JAX-style batching transform. 17 per-opcode rules. Stand-alone vmap works; `vmap(grad(fn))` composition has remaining shape-broadcasting subtleties (PRD-014b polish).
- `torch.func.*` shim via `install_torch_alias()`.

### Custom WGSL kernels (PRD-015)
- `@bg.custom_kernel(wgsl, name, workgroup_size, output_shape_fn, dispatch_shape_fn, num_inputs)` decorator.
- SHA-256 of WGSL = cache key. Forward only.

### ONNX export (PRD-016)
- `bg.onnx.export_inference(tensor, input_buffers=(...))` — pure-Python proto3 encoder (no protobuf wheel).
- 14 ops mapped (ADD/MUL/DIV/NEG/EXP/LOG/MATMUL/WHERE/CAST/REDUCE/RESHAPE/PERMUTE/CMP/BROADCAST_TO) + lifecycle.
- `OnnxUnmappableOp` typed refusal for the rest.

## Surface inventory — `browsergrad-grad` (v0.5.1)

Stable eager library. PyTorch-shaped tensor + autograd, closure backward,
NumPy-backed by default. Current covered surface:

- Tensor creation, dtype aliases, indexing/fancy indexing, broadcasting
  arithmetic, reductions, reshape/view/permute/transpose, gather/scatter,
  einsum, topk/sort, multinomial, no_grad.
- Functional ops: activations, softmax/log_softmax, cross entropy/NLL/MSE/BCE,
  padding/interpolation/normalization/cosine similarity, scaled dot-product
  attention.
- Layers: Linear, Conv1d/2d/3d, ConvTranspose2d, MaxPool2d/AvgPool2d,
  AdaptiveAvgPool2d, BatchNorm1d/2d/3d, GroupNorm, InstanceNorm2d, LayerNorm,
  Dropout/Dropout2d, Flatten, Sequential, Embedding, MultiHeadAttention,
  stacked/bidirectional RNN/LSTM/GRU, common activation modules, loss modules.
- Module ergonomics: train/eval propagation, forward hooks, buffers,
  state_dict/load_state_dict, torch.save/load shim, torch alias install.
- Optimizers/schedulers: SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta plus
  StepLR/CosineAnnealingLR/ReduceLROnPlateau/MultiStepLR/ExponentialLR/OneCycleLR.
- Optional eager forward WebGPU dispatch: pass `device=` from
  `createGradKernelDeviceBridge(device)` for 2D matmul/mm, last-dim softmax,
  last-dim LayerNorm, and unmasked default-scale 2D scaled-dot-product
  attention. Backward remains CPU closure autograd after forward materializes.

See `packages/browsergrad-grad/README.md` for full API and examples.

## Surface inventory — `browsergrad-kernels`

WGSL kernels — each with a JS reference for conformance:
- `matmul` (naive triple-loop), `matmulTiledDirect` (16×16 tiled, the production path)
- `softmax` (stable, along last axis)
- `relu`, `gelu` (elementwise)
- `layernorm` (along last axis, optional gamma/beta)
- `attention` (composed 3-kernel)
- `flash_attention.ts` (legacy row-wise online-softmax attention forward with
  strict real-WebGPU parity; not the Gate 5 block-tiled implementation)
- `fusedElementwiseDirect` — runtime WGSL codegen for arbitrary elementwise chains
- `WebGpuRealizerBridge.conv1d*` / `conv2d*` — f32 Conv1d/Conv2d forward plus
  input/weight/bias backward kernels over resident GPUBuffer handles,
  including stride/padding/dilation and groups

Plus the realizer-tier API:
- `createWebGpuRealizerBridge(device)` — production bridge for browsergrad-jit.
- `runDirect` / `materializeFloat32` / `uploadFloat32` — GPUBuffer-in/out dispatch path.
- `createDevice({ outputBufferPoolSize })` — per-device reusable output-buffer
  pool for direct-dispatch outputs; stats expose pool size, bytes, hits, and
  misses through `device.getStats()`.
- `runTensorGpuPlan` — generic f32 tensor-plan executor for BUFFER/LOAD,
  2-D MATMUL, elementwise chains, RESHAPE, PERMUTE, BROADCAST_TO, and
  REDUCE(sum/mean) rank <= 4, plus `FUSED_ELEMENTWISE` runtime WGSL codegen,
  `FUSED_SOFTMAX` last-axis direct softmax lowering,
  Conv1d/Conv2d/ConvTranspose2d/Conv3d/LayerNorm
  forward/backward and functional SGD/Adam/AdamW updates. It accepts
  the snake_case `browsergrad-jit` plan payload, keeps intermediates
  resident, and materializes only the declared root, matching the GPU-native
  direction. It releases dead owned buffers from liveness metadata before the
  root boundary, returns dead direct-dispatch outputs to the device pool, and
  reports early-release counts/bytes.
- `runTensorGpuPlanResident` — same executor, but returns an owned resident
  root `GPUBuffer`; bridge inputs can be host bytes or existing resident
  handles.
- `WebGpuRealizerBridge.run_tensor_plan` — one graph-level bridge call from JIT
  plan payload + seed buffers into `runTensorGpuPlan`; avoids legacy per-op
  bridge dispatch for supported primitive tensor plans.
- `WebGpuRealizerBridge.run_tensor_plan_resident` — graph-level bridge call
  that mints a resident root handle for `browsergrad-jit` follow-on plans.

## Surface inventory — `browsergrad-runtime`

- `createSession({ pyodideIndexURL, packages, worker?, disableInterruptBuffer? })`
- `session.fs.write/read` (MEMFS via Emscripten)
- `session.exec({ code, timeoutMs, signal, onStdout, onStderr, onAssertion, onArtifact })`
- `session.interrupt()` + `session.canInterrupt` (SAB + cross-origin isolation)
- `session.clearNamespace()`, `session.dispose()`
- Structured assertion + artifact protocols emitted from Python via `import browsergrad as bg`
- Lab manifest: `parseManifest`, `isSemverCompatible`, `assertCompatibleRuntime`, `LabRuntimeMismatch`

## Browser testing

Real-WebGPU CI ships with the kernels package. Run with:

```sh
pnpm --filter @unlocalhosted/browsergrad-kernels test:browser
```

Launches Chromium via Playwright with WebGPU enabled. Tests actual tiled GEMM,
fused elementwise codegen, FlashAttention forward, Conv2d forward, residency
contract, and the `WebGpuRealizerBridge` end-to-end against a real `GPUDevice`.
On macOS the browser is headed (Metal driver only exposed when visible); on
Linux CI set `BG_BROWSER_HEADLESS=1`.

## Performance baselines (NumPy realizer)

From `tests-integration/perf_bench.test.ts` — written to `/tmp/bg-perf-report.md` each run.

| Shape | Time | GFLOPS |
|---|---|---|
| matmul 64×64×64 | 0.41ms | 1.27 |
| matmul 128×128×128 | 1.43ms | 2.93 |
| matmul 256×256×256 | 11.51ms | 2.91 |
| matmul 512×64×256 | 5.26ms | 3.19 |

Trace cache: ~3.6× warm-vs-cold speedup on a chained matmul + reduce.
vmap vs Python for-loop on 32 samples: ~16× speedup.
AMP on NumPy: not faster than f32 (NumPy lacks f16 SIMD); the value is correctness substrate + WGSL-ready cast pass. Wall-clock wins materialise on real GPU.

## Known issues

| Issue | Found by | Status |
|---|---|---|
| Trace cache misses on `requires_grad=True` graphs | Perf bench | Intentional (`_trace_cache.py:147` exclusion). Lifting would let backward graphs cache too. P1 follow-up. |

## PRD coverage

Current status across the 16 PRDs:

| PRD | Status |
|---|---|
| 001-010 | ✅ Shipped (see PROGRESS.md) |
| 011 (WebNN) | ⏳ Draft backend. Only `bg.experimental.webnn.is_available` presence detection exists; ADR-0034 removed the non-executable matmul constructor. |
| 011.5 (WGSL realizer seam) | ✅ Shipped |
| 012 (megakernels) | ⏳ Split: PRD-012a (tiled GEMM + fused codegen + CAST) and PRD-012b (cost model + producer-consumer detection) shipped. PRD-012c remains a draft; ADR-0035 removed its non-executable constructor placeholder. |
| 013 (lab platform) | ✅ Shipped |
| 014 (functional transforms) | ✅ Shipped — `grad`, `vjp`, `functional_call`, full `vmap` with 17 active rules + explicit refusal stubs for random, masking, custom ops, CNN primitives/backward roots, STORE, and optimizer-update roots. `vmap(grad(fn))` composition works. |
| 015 (custom WGSL) | ✅ Shipped |
| 016 (ONNX export) | ✅ Shipped |

## Honest limitations

| Item | Reason |
|---|---|
| **Default `.backward()` through GPU realizer** | CPU-owned graphs keep CPU mutation. GPU-owned graphs make default `.backward()` select `run_tensor_plan_resident` and store `.grad` as GPUBuffer-backed TensorProxy values. `Optimizer.step(device="webgpu")` runs update math through tensor-plan WebGPU, then materializes params/state back to CPU buffers; default `.step()` on GPU-owned params/grads keeps supported params/state resident for SGD/Adam/AdamW. Explicit `realize_tensor_plan_webgpu_resident(...)` can keep tensor-plan roots GPU-resident until `.numpy()` / `.item()`. |
| **f16/bf16 cast kernels** | Future work — current CAST handler is f32→f32 only. |
| **Primitive/WebGPU ConvTranspose in `browsergrad-jit`** | Lazy `browsergrad-jit` now has primitive Conv1d/Conv2d/ConvTranspose2d/Conv3d forward/backward IR with CPU handlers, symbolic VJPs, and generic tensor-plan WebGPU lowering. |
| **torch.cuda.\*, torch.compile, torch.fx** | Out of scope for `install_torch_alias`. Raises `AttributeError`. |
| **Cross-browser WGSL compile-error line/column parsing** | Vendor diagnostic formats differ; ship raw browser messages and call it honest. |
| **vmap of RANDOM** | Needs PRNG key splits (JAX-style PRNGKey). Refuses with clear message; user can hand-write a key-split pattern. |
| **`transformer_block`** | Not exposed. PRD-012c remains a draft until typed recognition/codegen and actual execution evidence exist. |

When any of these become blocking for a real consumer, file an issue against the relevant PRD doc and we'll revisit.
