# BrowserGrad Package Requirements And Low-Level Design

Last updated: 2026-07-06

This document is the package-by-package requirements map for BrowserGrad after
the v0.5.x capability pass. It is intentionally explicit: each package owns a
small production contract, a clear low-level design, and validation gates that
prove the contract. It also names the research and platform facts behind the
architecture so future work does not drift into assumption-led design.

## Research Basis

| Design pressure | External basis | BrowserGrad consequence |
| --- | --- | --- |
| Eager autograd remains useful for teaching and debugging. | PyTorch autograd records a graph during forward execution and runs reverse-mode differentiation from that graph. See [PyTorch Autograd mechanics](https://docs.pytorch.org/docs/2.12/notes/autograd.html). | `browsergrad-grad` keeps closure-based eager autograd as the readable baseline. It should stay CPU-first and easy to inspect. |
| Production acceleration needs staged graphs, not ad hoc per-op hacks. | JAX documents tracing to an IR, lowering to StableHLO, compiling for a target device, then executing; see [JAX AOT lowering and compilation](https://docs.jax.dev/en/latest/aot.html). PyTorch `torch.compile` similarly separates capture, graph breaks, guards, and backend compilation; see [torch.compile](https://docs.pytorch.org/docs/2.12/generated/torch.compile.html). | `browsergrad-jit` owns lazy UOp IR, VJP graph construction, fusion, AMP/checkpoint rewrites, ONNX export, and WebGPU realization. |
| WebGPU kernels must respect browser GPU semantics, not CUDA folklore. | WGSL is statically typed, exposes storage/uniform/workgroup memory, requires uniformity for barriers, and does not share workgroup variables across workgroups. See [WGSL specification](https://www.w3.org/TR/WGSL/). | `browsergrad-kernels` and `browsergrad-compiler` must model address spaces, barriers, feature gates, and readback boundaries explicitly. |
| Browser execution needs worker isolation and cooperative interrupt semantics. | Pyodide interrupts require a Web Worker plus `SharedArrayBuffer` and `pyodide.setInterruptBuffer()`. See [Pyodide interrupting execution](https://pyodide.org/en/stable/usage/keyboard-interrupts.html). | `browsergrad-runtime` owns Worker lifecycle, same-origin Pyodide assets, timeout/abort, and clear cancellation behavior. |
| Conv performance should lower convolution to matrix multiply where simple. | Caffe documents `im2col` as image-to-column lowering used to perform convolution through matrix multiplication. See [Caffe im2col](https://caffe.berkeleyvision.org/tutorial/layers/im2col.html). | `browsergrad-grad` Conv2d uses im2col + batched matmul for correctness and readable performance. |
| Attention acceleration must be IO-aware. | FlashAttention frames exact attention speed around tiling and reducing reads/writes between memory levels. See [FlashAttention](https://arxiv.org/abs/2205.14135). | `browsergrad-kernels` keeps JS reference attention plus WebGPU attention/FlashAttention paths; `jit` uses graph realization for throughput-oriented attention. |
| Recurrent parity is shape/state heavy. | PyTorch RNN/LSTM APIs define `num_layers`, `dropout`, `bidirectional`, direction-expanded hidden state shapes, and layer-specific parameter names. See [torch.nn.RNN](https://docs.pytorch.org/docs/2.12/generated/torch.nn.RNN.html) and [torch.nn.LSTM](https://docs.pytorch.org/docs/2.12/generated/torch.nn.LSTM.html). | `browsergrad-grad` now covers stacked and bidirectional RNN/LSTM/GRU with explicit state-shape, state_dict-key, dropout, and backward tests. |
| Export needs a stable graph format with operator/version semantics. | ONNX IR defines models, graphs, nodes, operators, tensor types, opsets, and versioning. See [ONNX IR specification](https://onnx.ai/onnx/repo-docs/IR.html). | `browsergrad-jit` ONNX export should remain an explicit supported-op subset with clear unmappable-op failures. |
| Browser GPU feature availability changes by adapter/browser. | Chrome WebGPU notes expose feature-specific gates such as `shader-f16`; see [Chrome WebGPU updates](https://developer.chrome.com/blog/new-in-webgpu-120). | `kernels.detectKernelFeatures()` and compiler options must carry adapter facts instead of hard-coded platform assumptions. |
| Browser-native ML must avoid accidental CPU round-trips. | WebGPU readback is asynchronous: `GPUBuffer.mapAsync()` returns a Promise and mapped buffers cannot be used by GPU commands while mapped; see [MDN GPUBuffer.mapAsync](https://developer.mozilla.org/en-US/docs/Web/API/GPUBuffer/mapAsync). Pyodide crosses Python/JS with `JsProxy`/`PyProxy` wrappers; see [Pyodide type translations](https://pyodide.org/en/stable/usage/type-conversions.html). | A GPU-native BrowserGrad path needs one canonical tensor IR, explicit storage/device ownership, GPU-resident parameters/activations/grads/optimizer state, and `.numpy()`/`.item()` as explicit materialization boundaries. |

## Production Architecture

```text
Host app / lab platform
  -> @unlocalhosted/browsergrad-runtime
       Pyodide Worker, exec protocol, fs, cancellation, assertions/artifacts,
       manifest/profile preflight

  -> Python package installers
       @unlocalhosted/browsergrad-grad
         eager NumPy autograd, readable modules, optional explicit device bridge

       @unlocalhosted/browsergrad-jit
         lazy UOp IR, symbolic VJP, transforms, fusion, ONNX, WebGPU realizer

  -> GPU / compiler substrate
       @unlocalhosted/browsergrad-kernels
         WebGPU device helpers, WGSL catalog, JS references, resident buffers

       @unlocalhosted/browsergrad-compiler
         CUDA-lite parser/analyzer -> Kernel IR -> CPU reference -> WGSL/WebGPU

  -> Shared domain helpers
       @unlocalhosted/browsergrad-primitives
         browser-safe text/data/eval/simulation/RL helpers

  -> Release proof
       @unlocalhosted/browsergrad-dogfood
         published-tarball tests against real browser/Pyodide/WebGPU surfaces
```

Non-negotiable boundaries:

- Runtime stays tensor-agnostic.
- Kernels stay Python-agnostic.
- Grad stays eager and readable; it may call WebGPU only through explicit
  `device=` forwarding.
- JIT owns graph execution and throughput GPU work.
- Compiler owns CUDA-lite semantics and diagnostics; platform code must not
  duplicate its planning logic.
- Primitives stay course-agnostic; assignment wrappers live above the package.
- Dogfood tests published artifacts, not local workspace illusions.

## GPU-Native Target Architecture

The long-term WebGPU-backed framework path is one GPU IR, not many escape
hatches. `CUSTOM` callbacks are acceptable as migration scaffolding and for
user kernels, but core framework ops should become primitive tensor IR nodes
with CPU and WebGPU backends.

Target flow:

```text
Python API
  -> eager/lazy tensor frontend
  -> canonical tensor IR
  -> autodiff over IR
  -> optimizer/update IR
  -> scheduler / memory planner
  -> GPU codegen backend
  -> WebGPU runtime
```

Required properties:

1. Tensor owns storage abstraction.
   Storage is `CPU ndarray | GPUBuffer | pending graph value`. Shape, dtype,
   device, layout, lifetime, aliasing, and materialization state are tracked
   from creation.
2. Ops build canonical IR.
   Conv, matmul, norm, attention, elementwise, reductions, indexing, and
   optimizer updates are IR ops, not Python callbacks.
3. Autograd generates backward IR.
   Backward is graph construction, not Python closures. Examples:
   `conv2d_backward_input`, `conv2d_backward_weight`, norm backward, attention
   backward, reduce-bias-grad.
4. Tensor compiler lowers IR to WebGPU.
   CUDA-lite compilation remains for labs and user-authored kernels. Framework
   runtime ops need tensor-aware lowering: shape specialization, layout
   planning, fusion, tiling, memory reuse, dispatch scheduling, and feature
   gates.
5. GPU residency is default.
   Parameters, activations, gradients, and optimizer state stay in `GPUBuffer`.
   CPU readback occurs only for explicit `.numpy()`, `.item()`, debug artifact,
   assertion, or host export.
6. Optimizers are GPU IR and kernels.
   `SGD`, `Adam`, and `AdamW` update parameter buffers in place on GPU. No
   `grad.numpy()` inside `step()`.
7. WebGPU runtime owns execution.
   It manages command encoders, pipeline cache, bind group cache, buffer pool,
   device-loss recovery, adapter feature detection, and f16/subgroup gates.
8. CPU backend remains reference.
   Same IR runs on CPU for correctness/debug. Tests compare CPU reference vs
   WebGPU output and gradients.

Future package shape if BrowserGrad becomes a GPU-native framework:

```text
browsergrad-core
  Tensor, Device, Storage, Graph IR

browsergrad-autograd
  VJP rules -> backward graph

browsergrad-compiler
  tensor IR -> WGSL pipelines

browsergrad-runtime-webgpu
  GPUBuffer pool, scheduler, pipeline cache

browsergrad-nn
  modules as thin IR builders

browsergrad-reference
  CPU NumPy backend
```

This is a direction, not a renaming requirement for the current monorepo. In the
current package split, the pragmatic correction is:

- Keep `browsergrad-grad` educational CPU/reference-first.
- Make `browsergrad-jit` the canonical tensor IR owner.
- Stop expanding core GPU support through ad hoc `CUSTOM` paths.
- Add primitive conv/norm/attention/optimizer IR.
- Compile those IR ops to WebGPU through tensor-aware lowering.
- Use `browsergrad-compiler` CUDA-lite for lab/user kernels, not core framework
  ops.
- Keep `.numpy()` and `.item()` as explicit materialization boundaries.

Conv target path:

```text
nn.Conv2d.forward
  -> IR: conv2d(input, weight, bias, stride, padding, dilation, groups)
  -> GPU compiler:
       choose direct/tiled/im2col/Winograd depending shape
       emit WGSL
       keep output GPU-resident
  -> backward:
       conv2d_backward_input
       conv2d_backward_weight
       reduce bias grad
  -> optimizer:
       adamw_update(weight, grad_weight, m, v)
```

Browser constraints that make this mandatory:

- WebGPU readback is async and synchronization-heavy.
- `GPUBuffer` lifetime must be explicit.
- WGSL has strict address-space, typing, and uniformity rules; compiler code
  must own layout and memory semantics.
- Pyodide/Python crossing is expensive enough that per-op Python callback
  dispatch is the wrong hot path.
- f16/subgroups and limits vary by browser/device, so backend must feature-gate
  instead of assuming CUDA-like capabilities.

## Package Requirements

### `@unlocalhosted/browsergrad-runtime`

Purpose: production host for browser Python labs.

Requirements:

| ID | Requirement | Current status | Acceptance gate |
| --- | --- | --- | --- |
| RT-001 | Create and dispose Pyodide Worker sessions with same-origin Pyodide assets. | In | `packages/browsergrad-runtime` unit/integration tests |
| RT-002 | Execute Python with stdout/stderr, timeout, AbortSignal, and explicit interrupt. | In | timeout/cancel tests; manual browser smoke where interrupt depends on `SharedArrayBuffer` |
| RT-003 | Emit structured assertion/artifact events without interpreting course semantics. | In | assertion/artifact protocol tests |
| RT-004 | Validate lab manifests and semver runtime gates. | In | manifest + semver adversarial tests |
| RT-005 | Validate assignment profiles and produce preflight/run plans. | In | profile parser/planner tests |
| RT-006 | Keep ML/tensor library installation pluggable through public installer calls. | In | grad/jit integration tests |

LLD:

- `createSession()` owns Worker boot, Pyodide package loading, namespace state,
  and the message protocol.
- Python receives a small `browsergrad` module for assertions, artifacts,
  oracle lookup, and assignment context.
- Cancellation path is layered: cooperative interrupt when isolation allows it,
  worker termination fallback otherwise.
- Manifest/profile validation is TypeScript-side and dependency-light so host
  platforms can preflight before launching Python.

Production hardening required:

- Keep every protocol message versioned or backward-compatible.
- Never load Pyodide from an uncontrolled CDN in production docs.
- Add browser smoke tests for cross-origin-isolated interrupt behavior before
  calling cancellation fully production-grade across browsers.

### `@unlocalhosted/browsergrad-grad`

Purpose: readable eager NumPy autograd for curriculum content.

Requirements:

| ID | Requirement | Current status | Acceptance gate |
| --- | --- | --- | --- |
| GR-001 | Closure-based reverse-mode autograd over core tensor ops. | In | surface tests + Pyodide integration |
| GR-002 | CNN layer set: Conv1d/2d/3d, ConvTranspose2d, pooling, im2col Conv2d, tuple shapes, dilation, groups. | In | conv integration tests |
| GR-003 | Norm layer set: BatchNorm1d/2d/3d, LayerNorm, GroupNorm, InstanceNorm2d with stats-aware backward where implemented. | In | norm integration tests |
| GR-004 | Transformer/sequence basics: Embedding, MultiHeadAttention, RNN/LSTM/GRU. | In | kitchen-sink and sequence tests |
| GR-005 | Module ergonomics: `train`, `eval`, hooks, buffers, `state_dict`, `load_state_dict`. | In | module/state tests |
| GR-006 | Torch compatibility shim that covers common tutorial code and refuses unsupported APIs loudly. | In | torch alias tests |
| GR-007 | Explicit WebGPU forward dispatch through `device=` for matmul, softmax, layernorm, attention. | In | kernel device bridge unit + Pyodide tests |
| GR-008 | Multi-layer/bidirectional RNN/LSTM/GRU parity. | In | recurrent integration tests for direction ordering, state shapes, state_dict keys, dropout, and backward |
| GR-009 | GPU-resident eager autograd. | Out of current scope | Device tensor storage + GPU backward kernels + optimizer residency |

LLD:

- Python sources under `src/python/` are edited directly and codegen embeds them
  into TypeScript.
- Tensor ops save backward closures and update leaf gradients on `.backward()`.
- Conv2d lowers patches through im2col and batched matmul to avoid unreadable
  nested-loop hot paths while preserving educational clarity.
- `device=` calls pass concrete arrays to a JS bridge, run selected kernels, and
  materialize results back into CPU tensors. CPU autograd remains the gradient
  authority.

Production hardening required:

- For each new PyTorch-shaped op, add PyTorch parity fixtures where PyTorch is
  the oracle and BrowserGrad failures are explicit.
- Keep unsupported CUDA/distributed/compile/fx/quantization APIs as clear
  refusals.
- Do not expand eager GPU by accident. If tensors can remain GPU-resident, that
  is a new storage/backend project, not a small `device=` extension.

### `@unlocalhosted/browsergrad-jit`

Purpose: lazy PyTorch-shaped IR layer for graph transforms and acceleration.

Requirements:

| ID | Requirement | Current status | Acceptance gate |
| --- | --- | --- | --- |
| JIT-001 | TensorProxy builds UOp graphs lazily and realizes on explicit boundaries. | In | integration tests for `.numpy`, `.item`, `.backward`, optimizer step |
| JIT-002 | NumPy realizer remains universal correctness backend. | In | integration parity tests |
| JIT-003 | Symbolic VJP path with closure fallback where rules are absent. | In | VJP/backward tests |
| JIT-004 | Fusion, AMP, GradScaler, checkpoint rewrites, and functional transforms operate over IR. | In | PRD-specific tests |
| JIT-005 | `vmap`, `grad`, `vjp`, `functional_call` compose without hidden module state. | In | transform tests |
| JIT-006 | WebGPU realizer bridge supports forward opcodes and materializes at boundaries. | In | bridge/mock + real WebGPU tests through kernels |
| JIT-007 | Custom WGSL kernels are cache-keyed, forward-only, and explicit. | In | custom kernel tests |
| JIT-008 | ONNX export emits a supported subset and refuses unmappable ops. | In | ONNX tests |
| JIT-009 | GPU-resident backward/optimizer steps. | Partial | Conv/LayerNorm backward roots, explicit `backward(device="webgpu", resident=True)` GPUBuffer-backed leaf grads, functional SGD/Adam/AdamW update IR, explicit `Optimizer.step(device="webgpu")`, resident `SGD`/`Adam`/`AdamW` step paths, and explicit resident tensor-plan roots can lower through WebGPU; default CPU `.backward()` and memory planner remain |
| JIT-010 | Heavy CNN family parity with grad. | Partial | Conv1d/Conv2d/ConvTranspose2d/Conv3d forward/backward are primitive IR with CPU handlers and symbolic VJPs; these CNN roots lower through generic f32 tensor-plan WebGPU; explicit `backward(device="webgpu", resident=True)` can populate leaf grads through tensor-plan WebGPU without CPU readback; default CPU `.backward()` and full optimizer residency remain |
| JIT-011 | Canonical tensor IR for core framework ops instead of `CUSTOM` GPU escape hatches. | In progress | Primitive IR ops for conv/norm/attention/optimizer updates, CPU handlers/refusals, VJPs where differentiable, GPU tensor-plan lowering, WebGPU lowering, refusal tests |
| JIT-012 | `.numpy()`/`.item()` are the primary materialization boundaries for GPU paths. | Partial | `realize_tensor_plan_webgpu_resident(...)` returns GPUBuffer-backed TensorProxy roots, resident backward stores leaf grads as GPUBuffer-backed TensorProxy values, resident optimizers rebind params/state to GPUBuffer handles, and all materialize only on `.numpy()` / `.item()`; default training path is not fully resident |

LLD:

- `_ir.py` defines opcodes and UOp structure.
- Tensor dunders create IR nodes, not data.
- `_realize.py` topologically executes UOps through backend handlers.
- `_vjp.py` emits gradient UOps for symbolic backward.
- `_fusion.py`, `_amp.py`, checkpointing, and `func` passes rewrite IR.
- `_gpu_plan.py` builds a compiler-facing tensor-IR execution plan with
  liveness/materialization metadata and refuses `CUSTOM` by default. This is
  the path future WebGPU codegen/scheduling should consume instead of growing
  one bridge method per framework op. Plan value IDs are stable schedule-local
  integers so payloads can cross Pyodide/JS without Python object identity.
- `_realize_webgpu.py` talks only to a bridge protocol; `browsergrad-kernels`
  owns actual WebGPU resources. `realize_tensor_plan_webgpu(tensor)` now sends
  one canonical plan plus seed buffers to `bridge.run_tensor_plan(...)`.
  `realize_tensor_plan_webgpu_resident(tensor)` sends the same plan to
  `bridge.run_tensor_plan_resident(...)`, registers the root handle in the
  GPU buffer table, and defers CPU bytes until `.numpy()` / `.item()`. The
  older `realize_webgpu(tensor)` path remains legacy per-op bridge coverage.
- Conv1d, Conv2d, ConvTranspose2d, and Conv3d forward are primitive IR ops with
  NumPy handlers, explicit vmap refusals, and ONNX refusals. Their symbolic
  backwards emit input/weight/bias gradient UOps with CPU reference handlers.
  These CNN forward/backward roots lower through `runTensorGpuPlan()` as generic
  plan ops. `loss.backward(device="webgpu")` realizes symbolic leaf-gradient
  roots through that same bridge and refuses closure-only graphs.
  `loss.backward(device="webgpu", resident=True)` registers leaf `.grad`
  tensors as GPU-resident TensorProxy buffers until explicit materialization.
  Default CPU `.backward()` remains pending.
- `nn.LayerNorm` / `F.layer_norm(...)` emit primitive `LAYER_NORM` IR and
  `LAYER_NORM_BACKWARD_*` symbolic gradient roots. CPU handlers remain the
  reference; forward/input-grad/weight-grad/bias-grad lower through
  `runTensorGpuPlan()` with real browser WebGPU parity tests.
- `bg.optim.sgd_update(...)`, `bg.optim.adam_update(...)`, and
  `bg.optim.adamw_update(...)` emit primitive optimizer/update IR with NumPy
  handlers, vmap/ONNX refusals, tensor-plan lowering, and real WebGPU kernels.
  `Optimizer.step(device="webgpu")` uses those same update IR nodes for SGD
  without momentum, Adam, and AdamW, then writes the materialized result back to
  CPU parameter/state buffers. `SGD.step(device="webgpu", resident=True)` uses
  resident tensor-plan roots to rebind no-momentum parameter buffers to WebGPU
  handles without CPU readback. `Adam.step(device="webgpu", resident=True)` and
  `AdamW.step(device="webgpu", resident=True)` keep parameter, first-moment, and
  second-moment buffers resident.
- Future GPU-native work should promote core ops out of `CUSTOM` into primitive
  tensor IR. `CUSTOM` should remain for user-authored WGSL/lab kernels and
  temporary migration scaffolding, not the final path for framework ops.
- Existing per-op bridge methods are legacy/interim bridge coverage. Do not
  keep expanding GPU support by adding one JS/Python bridge method per new
  framework op. New core GPU work should add tensor-IR lowering and runtime
  scheduling so op families compile through one backend path.

Production hardening required:

- Keep public errors stable and specific.
- Any new opcode requires: IR declaration, NumPy handler, VJP or clear no-backward
  refusal, vmap rule or clear transform refusal, ONNX/export decision, and
  optional WebGPU lowering.
- Preserve graph-break-like behavior: unsupported dynamic Python behavior should
  refuse or realize explicitly, never silently change semantics.

### `@unlocalhosted/browsergrad-kernels`

Purpose: WebGPU/WGSL kernel catalog and GPU resource substrate.

Requirements:

| ID | Requirement | Current status | Acceptance gate |
| --- | --- | --- | --- |
| KER-001 | Ship JS reference implementations for every public kernel. | In | unit conformance tests |
| KER-002 | Provide real WebGPU paths for matmul, softmax, layernorm, attention, fused elementwise, Conv1d/Conv2d explicit bridge kernels, Conv1d/Conv2d/ConvTranspose2d/Conv3d generic tensor plans, and generic WGSL programs. | In | browser tests |
| KER-003 | Provide bridge protocol implementation for `browsergrad-jit`. | In | dogfood bridge lifecycle/method/concurrency tests |
| KER-004 | Provide public resident buffer helpers and prepared sequence APIs. | In | browser hot-loop tests |
| KER-005 | Expose feature detection for adapter-gated behavior such as f16/subgroups. | In | feature detection tests |
| KER-006 | Keep tensor-library dependency at zero. | In | package/dependency check |
| KER-007 | Production FlashAttention forward across supported browsers/devices. | In | strict real-WebGPU parity test against composed attention reference |

LLD:

- Kernels expose both host-tensor convenience APIs and lower-level
  `GPUBuffer`-resident APIs.
- `runDirect()` is the realizer-tier fast path: GPUBuffer in, GPUBuffer out.
- `runTensorGpuPlan()` is the first generic tensor-plan executor: it consumes
  scheduled primitive plan steps, including the snake_case payload emitted by
  `browsergrad-jit`'s `gpu_plan_summary`, keeps intermediates in `GPUBuffer`,
  supports f32 BUFFER/LOAD/MATMUL, elementwise chains, RESHAPE, PERMUTE,
  BROADCAST_TO, REDUCE(sum/mean) for rank <= 4, Conv1d/Conv2d/ConvTranspose2d/Conv3d
  forward/backward, LayerNorm forward/backward, functional SGD/Adam/AdamW
  updates, and materializes only the root.
- `runTensorGpuPlanResident()` shares that executor but returns an owned root
  `GPUBuffer`; `createWebGpuRealizerBridge(...).run_tensor_plan_resident(...)`
  mints a bridge handle so later plans can consume resident roots without
  readback.
- `createWebGpuRealizerBridge(...).run_tensor_plan(plan, inputs, dtype)` exposes
  that executor through the Pyodide bridge as one graph-level call. This is the
  preferred framework GPU direction; new core ops should lower into tensor
  plans/runtime kernels instead of adding one bridge method per op.
- `prepareWgslKernelProgramSequence()` caches pipelines/bind groups for repeated
  dispatch.
- JS references are correctness oracles and CPU fallback, not performance paths.

Production hardening required:

- Every WGSL program must have shape validation, bounds safety, and reference
  parity.
- Device loss, validation errors, and feature-missing cases must report stable
  errors.
- Keep readback explicit. Hidden readbacks destroy the performance model.

### `@unlocalhosted/browsergrad-compiler`

Purpose: CUDA-lite source compiler for browser-native GPU labs.

Requirements:

| ID | Requirement | Current status | Acceptance gate |
| --- | --- | --- | --- |
| CMP-001 | Parse CUDA-lite/CUDA-shaped source without semantic rewrites in the parser. | In | parser unit tests |
| CMP-002 | Analyze symbols, types, memory spaces, feature gates, safety, and unsupported diagnostics. | In | analyzer/diagnostic tests |
| CMP-003 | Lower to backend-neutral Kernel IR. | In | Kernel IR snapshot/semantic tests |
| CMP-004 | Execute CPU reference from the same semantic facts as WGSL. | In | reference parity tests |
| CMP-005 | Emit WGSL and run WebGPU plans through kernels package. | In | browser/WebGPU tests |
| CMP-006 | Support execution plans: single dispatch, grid-sync phases, host dynamic launch, host copy, unsupported. | In | orchestration tests |
| CMP-007 | Support resident buffers and prepared compiled kernels for hot paths. | In | prepared/resident tests |
| CMP-008 | Expose execution-plan summaries and blocker codes for platform UI. | In | summary tests |
| CMP-009 | Grow CUDA compatibility by semantic families, not assignment patches. | In progress | corpus audit and compatibility map tests |
| CMP-010 | Production compiler-backed GPU labs. | In | canonical examples + compiler verify + browser WebGPU + published dogfood tests |
| CMP-011 | Keep CUDA-lite compiler out of the core tensor runtime hot path. | Target | Framework ops lower from tensor IR; CUDA-lite remains lab/user-kernel path |

LLD:

- Pipeline: source -> lexer/parser -> analyzer -> Kernel IR -> CPU reference ->
  WGSL -> WebGPU execution plan -> kernels dispatch.
- Parser owns syntax only.
- Analyzer owns semantics, feature gates, and deterministic diagnostics.
- `semantic_ir.ts` is the target for new compiler passes.
- `runtime_plan.ts` and `webgpu_orchestration.ts` decide whether a kernel can
  become real WebGPU work.
- `prepareCompiledKernelWebGpu()` must reuse the same plan path as one-shot
  execution.
- Canonical lab examples live in `packages/browsergrad-compiler/examples/README.md`
  and are exercised through workspace compiler tests plus published-package
  dogfood tests.
- CUDA-lite is not the main framework compiler. It is the right package for
  CUDA-shaped labs, diagnostics, source-to-Kernel-IR examples, and user kernels.
  Core `nn`/autograd/optimizer ops should lower from canonical tensor IR.

Production hardening required:

- Every emitted diagnostic/blocker needs stable code + source span + compatibility
  family.
- Every claimed feature needs unit + reference + WGSL/browser coverage, or a
  written reason browser coverage is impossible.
- Real-world CUDA claims require corpus audit gates, not hand-picked examples.

### `@unlocalhosted/browsergrad-primitives`

Purpose: course-agnostic browser-safe ML helper facade.

Requirements:

| ID | Requirement | Current status | Acceptance gate |
| --- | --- | --- | --- |
| PRM-001 | Expose stable small helpers for text, data, evaluation, simulation, scaling, RL. | In | unit tests |
| PRM-002 | Keep names generic: reference, comparator, fixture, simulator. | In | API review |
| PRM-003 | Avoid assignment-specific public APIs. | In | package review |
| PRM-004 | Provide subpath exports for bundle-sensitive consumers. | In | package export tests |

LLD:

- One top-level facade plus domain subpaths.
- Helpers are pure TypeScript where possible.
- Course/profile adapters wrap primitives outside this package.

Production hardening required:

- Do not add a new primitive until two consumers or one durable platform need
  proves it is generic.
- Keep fixtures deterministic and browser-safe.

### `@unlocalhosted/browsergrad-dogfood`

Purpose: post-publish verification against real npm artifacts.

Requirements:

| ID | Requirement | Current status | Acceptance gate |
| --- | --- | --- | --- |
| DOG-001 | Install exact npm-published packages, not workspace links. | In | dependency pins |
| DOG-002 | Test browser/WebGPU public surfaces in Chromium. | In | browser test suite |
| DOG-003 | Test Pyodide/Node package installation and cross-package coexistence. | In | node test suite |
| DOG-004 | Cover adversarial package failures: missing files, export drift, lifecycle leaks, numerical edge cases. | In | hypotheses-derived tests |

LLD:

- Private workspace package.
- Depends on exact published versions.
- Browser suite proves real WebGPU and public exports.
- Node suite proves Pyodide installation, grad/jit coexistence, and runtime
  manifest compatibility.

Production hardening required:

- Bump dogfood dependencies only after npm publish.
- Add dogfood coverage for every new public subpath.

## Cross-Package Requirements

| ID | Requirement | Owner | Acceptance |
| --- | --- | --- | --- |
| X-001 | Public imports use package exports only; no `src/` or `dist/` deep imports. | All | pack/dogfood tests |
| X-002 | Generated Python bundles match source. | grad, jit | `pnpm --filter ... codegen` during build |
| X-003 | Unsupported APIs fail loudly with stable errors. | grad, jit, compiler | refusal tests |
| X-004 | WebGPU claims require real browser tests, not unit tests only. | kernels, compiler, jit bridge | Vitest browser/Playwright gates |
| X-005 | Release tarballs contain built output and rewrite workspace dependencies. | all published packages | `pnpm pack`, `test:release-packages`, `npm view` |
| X-006 | Platform-facing capability labels match executable truth. | docs + package owners | docs update + tests in same change |
| X-007 | Browser/device feature gates flow from detection to compile/run. | kernels + compiler | feature matrix tests |
| X-008 | Readback/materialization boundaries are explicit. | kernels, grad, jit, compiler | API review + perf tests |
| X-009 | Core framework GPU ops must be tensor IR ops, not permanent `CUSTOM` callbacks. | jit + kernels | IR/VJP/WebGPU tests per op family |
| X-010 | CPU backend remains reference for every GPU-native core op. | jit + reference/backend owners | CPU-vs-WebGPU forward/backward parity tests |

## Remaining Limits And Removal Requirements

### Direct eager GPU scope

Current limit: `grad` has explicit forward-only `device=` dispatch. It does not
make eager tensors GPU-resident.

Remove the limit only through a dedicated device-aware tensor project:

- Add tensor storage abstraction:
  - CPU ndarray,
  - GPUBuffer handle,
  - materialization state,
  - ownership/lifetime rules.
- Add device-aware autograd:
  - backward kernels for supported GPU forward ops,
  - CPU fallback with explicit transfer,
  - no hidden readbacks inside training loops.
- Add optimizer residency:
  - GPU parameter buffers,
  - GPU gradient buffers,
  - update kernels for SGD/Adam/AdamW.
- Add memory planner:
  - temporary buffer reuse,
  - release strategy,
  - device-loss cleanup.
- Add async execution semantics:
  - command ordering,
  - explicit synchronization/readback,
  - deterministic test hooks.
- Add parity/perf tests:
  - CPU vs GPU numerical parity,
  - backward parity,
  - no-readback hot-loop perf gates,
  - browser/device feature matrix.

Recommendation: do not put this in `browsergrad-grad` unless the product goal
changes. Keep serious GPU graph execution in `browsergrad-jit`; keep `grad`
explicit and teachable.

### JIT CNN WebGPU Lowering

Current limit: `browsergrad-jit` has primitive Conv1d/Conv2d/ConvTranspose2d/
Conv3d forward/backward IR with CPU reference handlers. Those CNN roots lower
through generic tensor-plan WebGPU. Explicit `loss.backward(device="webgpu")`
can populate leaf `.grad` values by realizing symbolic gradient roots through
the tensor-plan bridge. Default `.backward()` still mutates CPU `.grad` buffers,
the WebGPU backward path can keep `.grad` GPU-resident when called with
`resident=True`, LayerNorm backward roots, functional SGD/Adam/AdamW updates,
and explicit `Optimizer.step(device="webgpu")` can run as tensor-plan WebGPU.
No-momentum `SGD.step(device="webgpu", resident=True)` can keep parameter
updates GPU-resident. Adam/AdamW resident mode keeps params/state GPU-resident;
non-resident `Optimizer.step(device="webgpu")` still materializes params/state
back to CPU buffers after the GPU update.
Explicit `realize_tensor_plan_webgpu_resident(...)` can keep tensor-plan roots
GPU-resident across follow-on tensor-plan calls until `.numpy()` / `.item()`;
default training storage is still CPU-owned.

Remove the limit when:

- Make default `.backward()` select the graph backend when tensors/params are
  GPU-resident.
- Make GPU-resident `.grad` the default path for GPU-resident tensors.
- Add a real memory planner/buffer pool for gradient and optimizer lifetimes.
- Broaden scheduler/codegen beyond direct kernels when throughput matters.

## Validation Matrix

Use narrow package gates for development and broad gates before release.

```sh
pnpm --filter @unlocalhosted/browsergrad-runtime test
pnpm --filter @unlocalhosted/browsergrad-grad test
pnpm --filter @unlocalhosted/browsergrad-grad test:integration
pnpm --filter @unlocalhosted/browsergrad-jit test
pnpm --filter @unlocalhosted/browsergrad-jit test:integration
pnpm --filter @unlocalhosted/browsergrad-kernels test
pnpm --filter @unlocalhosted/browsergrad-kernels test:browser
pnpm --filter @unlocalhosted/browsergrad-compiler verify:compiler
pnpm --filter @unlocalhosted/browsergrad-compiler e2e:webgpu:fast
pnpm --filter @unlocalhosted/browsergrad-primitives test
pnpm --dir packages/browsergrad-dogfood test
pnpm test:release-packages
```

Production-ready means:

- package-level build/typecheck/lint/test pass,
- Pyodide integration passes for Python packages,
- browser WebGPU passes for shader claims,
- compiler corpus gates pass for CUDA compatibility claims,
- packed tarball checks pass,
- dogfood passes after publish,
- docs status matches the exact validated surface.
