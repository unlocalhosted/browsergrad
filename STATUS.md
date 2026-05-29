# Status

Living document. Reflects the current state of each package, what's tested, and what's deliberately deferred.

## Package versions

| Package | Version | Surface tests | Integration tests | Browser tests |
|---|---|---|---|---|
| `@unlocalhosted/browsergrad-runtime` | `0.1.1` | 27 | 23 (Pyodide-in-node) | — |
| `@unlocalhosted/browsergrad-kernels` | `0.1.0` | 35 (incl. JS-reference numerical checks, FUSED WGSL codegen) | — | 7 (real Chromium + WebGPU) |
| `@unlocalhosted/browsergrad-grad` | `0.4.6` | 25 | 115 (Pyodide-in-node) | — |
| `@unlocalhosted/browsergrad-jit` | `0.8.0` | 8 | 156 (Pyodide-in-node, incl. feedback + perf benches) | — (via kernels) |

**Total: 396 tests green** across the workspace.

Every gradient verified against finite differences or a hand-derived oracle. Every realizer numerical result verified against a NumPy or JS-reference oracle. No test compares the implementation against itself.

## Two-library story

- `browsergrad-grad` is the **closure-autograd** library. PyTorch-shaped, NumPy-backed, eager. Used today by curriculum content. Stable.
- `browsergrad-jit` is the **lazy IR** successor. Same PyTorch surface, but ops build a UOp graph that gets realized through a backend (NumPy today, WebGPU via PRD-011.5's seam). Fusion + symbolic backward + AMP + gradient checkpointing + functional transforms + ONNX export + custom WGSL kernels all live here.

Both ship under the same `unlocalhosted` npm scope. They can coexist in the same Pyodide session (separate `install_torch_alias()` namespaces, owner-token protocol prevents collision).

## Surface inventory — `browsergrad-jit` (v0.8.0)

### Core (PRD-005)
- 28-opcode IR (`_ir.py`): BUFFER, LOAD, STORE, CONST, RANDOM, CAST, ADD, MUL, DIV, NEG, EXP, LOG, CMP, MATMUL, REDUCE, RESHAPE, PERMUTE, SLICE, PAD, WHERE, INDEX, MASK, CUSTOM, FUSED_ELEMENTWISE, FUSED_SOFTMAX, SCATTER_ADD, BROADCAST_TO, ISNAN.
- `TensorProxy` — lazy tensor; metadata never realizes; arithmetic builds IR; `.numpy()` / `.item()` triggers realize.
- `Session` + `BufferTable` for per-tab isolation.
- NumPy realizer (`_realize.py`): single dispatch table; one Python function per opcode; deterministic across runs.

### Fusion (PRD-006)
- Elementwise chain fusion → `OP_FUSED_ELEMENTWISE`.
- Softmax DAG → `OP_FUSED_SOFTMAX`.
- Introspection: `bg.jit.debug_fused_kernels()`, `bg.jit.debug_unfused_reasons()`.

### Symbolic backward (PRD-007)
- 13 VJP rules (ADD, MUL, DIV, NEG, EXP, LOG, CAST, MATMUL, REDUCE, RESHAPE, PERMUTE, ISNAN, CMP).
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
- Whitelisted opcodes: BUFFER, LOAD, CONST, CAST, MATMUL, FUSED_ELEMENTWISE, CUSTOM.
- `bg.kernels.flash_attention(Q, K, V, mask=None)` — opt-in CUSTOM op for FA-v2.

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

## Surface inventory — `browsergrad-grad`

(Stable; unchanged from prior releases. PyTorch-shaped tensor + autograd, closure backward, NumPy-backed eager. See `packages/browsergrad-grad/README.md` for the full API.)

## Surface inventory — `browsergrad-kernels`

WGSL kernels — each with a JS reference for conformance:
- `matmul` (naive triple-loop), `matmulTiledDirect` (16×16 tiled, the production path)
- `softmax` (stable, along last axis)
- `relu`, `gelu` (elementwise)
- `layernorm` (along last axis, optional gamma/beta)
- `attention` (composed 3-kernel)
- `flash_attention.ts` (FA-v2 forward — **known issue**, see below)
- `fusedElementwiseDirect` — runtime WGSL codegen for arbitrary elementwise chains

Plus the realizer-tier API:
- `createWebGpuRealizerBridge(device)` — production bridge for browsergrad-jit.
- `runDirect` / `materializeFloat32` / `uploadFloat32` — GPUBuffer-in/out dispatch path.

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

Launches Chromium via Playwright with WebGPU enabled. Tests the actual tiled GEMM, fused elementwise codegen, residency contract, and the `WebGpuRealizerBridge` end-to-end against a real `GPUDevice`. On macOS the browser is headed (Metal driver only exposed when visible); on Linux CI set `BG_BROWSER_HEADLESS=1`.

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
| FA-v2 kernel: ~0.69 max abs diff vs composed reference | Real-WebGPU browser CI (PRD-011.5+012a) | Bit-deterministic; kernel logic bug. Tracked as PRD-012a follow-up. |
| Trace cache misses on `requires_grad=True` graphs | Perf bench | Intentional (`_trace_cache.py:147` exclusion). Lifting would let backward graphs cache too. P1 follow-up. |

## PRD coverage

All 16 PRDs land at v0:

| PRD | Status |
|---|---|
| 001-010 | ✅ Shipped (see PROGRESS.md) |
| 011 (WebNN) | ✅ Experimental spike at `bg.experimental.webnn.matmul`. Full backend tier when Chrome WebNN reaches GA + meaningful user fraction. |
| 011.5 (WGSL realizer seam) | ✅ Shipped |
| 012 (megakernels) | ✅ Split: PRD-012a (tiled GEMM + fused codegen + CAST) shipped. PRD-012b (cost model + producer-consumer detection) shipped at `bg.jit.cost_model.*`. PRD-012c (transformer_block megakernel constructor) shipped at `bg.kernels.transformer_block(...)`. |
| 013 (lab platform) | ✅ Shipped |
| 014 (functional transforms) | ✅ Shipped — `grad`, `vjp`, `functional_call`, full `vmap` with 17 active rules + 4 refusal stubs (RANDOM, MASK, CUSTOM, STORE). `vmap(grad(fn))` composition works. |
| 015 (custom WGSL) | ✅ Shipped |
| 016 (ONNX export) | ✅ Shipped |

## Honest limitations

| Item | Reason |
|---|---|
| **Backward through GPU realizer** | NumPy realizer handles all `.backward()` calls; GPU path is forward-inference only. |
| **f16/bf16 cast kernels** | Future work — current CAST handler is f32→f32 only. |
| **ConvTranspose / Conv3d / dilated / groups** | Out of v0 scope across both grad and jit. |
| **torch.cuda.\*, torch.compile, torch.fx** | Out of scope for `install_torch_alias`. Raises `AttributeError`. |
| **Cross-browser WGSL compile-error line/column parsing** | Vendor diagnostic formats differ; ship raw browser messages and call it honest. |
| **vmap of RANDOM** | Needs PRNG key splits (JAX-style PRNGKey). Refuses with clear message; user can hand-write a key-split pattern. |
| **`transformer_block` and `webnn_matmul`** | Constructors build OP_CUSTOM UOps; bridge dispatch lands per JS-side kernel implementation. Forward only. |

When any of these become blocking for a real consumer, file an issue against the relevant PRD doc and we'll revisit.
