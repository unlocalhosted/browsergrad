# Architecture

How the four packages compose, what each layer owns, where the seams live, and why we made the calls we did.

For PRD history (what was scoped, what was deferred, with rationale) see [PROGRESS.md](./PROGRESS.md) and the per-PRD docs under [`docs/prd/`](./docs/prd/). For end-to-end test data + ranked improvement priorities see [FEEDBACK.md](./FEEDBACK.md).

## The layered picture

```
┌────────────────────────────────────────────────────────────────────────┐
│                       craftingattention (lab UI)                       │
│        - lab.json manifest validated at boot                           │
│        - rubric.py emits assertions via bg.lab.assert_*                │
│        - student code runs via session.exec()                          │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │  loadLab + runLab
┌─────────────────────────────────▼──────────────────────────────────────┐
│                  @unlocalhosted/browsergrad-runtime                    │
│  - Spawns Pyodide in a Worker                                          │
│  - exec / fs / interrupt / onAssertion / onArtifact protocol           │
│  - parseManifest + assertCompatibleRuntime (semver gate)               │
│  - Python preamble registers bg.assert_pass/fail/error/log/emit_*      │
└─────────┬────────────────────────────────────────────────────┬─────────┘
          │                                                    │
          │  install_torch_alias()                             │  createWebGpuRealizerBridge(device)
          │                                                    │  (only for jit + WebGPU consumers)
┌─────────▼──────────────────┐    ┌─────────────────┐     ┌────▼──────────────────────────────┐
│  @unlocalhosted/           │    │  @unlocalhosted/│     │  @unlocalhosted/                  │
│  browsergrad-grad          │    │  browsergrad-jit│     │  browsergrad-kernels              │
│  (eager, NumPy-backed)     │    │  (lazy IR)      │     │  (WGSL kernels + JS reference)    │
│                            │    │                 │     │                                   │
│  Tensor + autograd         │    │  TensorProxy +  │     │  matmul / matmulTiled /           │
│  nn.{Linear,Conv,...}      │    │  UOp IR (28 ops)│     │  softmax / layernorm /            │
│  optim.{SGD,Adam,AdamW}    │    │  Fusion         │     │  attention / FA-v2 /              │
│  Closure backward          │◀───┼─optional────────│     │  fusedElementwise (codegen)       │
│                            │    │  Symbolic VJP   │     │                                   │
│                            │    │  AMP + Scaler   │     │  createWebGpuRealizerBridge       │
│                            │    │  Checkpointing  │─────▶  (PRD-011.5 seam)                 │
│                            │    │  bg.func.*      │     │                                   │
│                            │    │  bg.onnx.*      │     │  reference.* (pure-JS oracle)     │
│                            │    │  bg.custom_kernel─────▶  runs user WGSL                   │
└────────────────────────────┘    └─────────────────┘     └───────────────────────────────────┘
```

## Package responsibilities

### `browsergrad-runtime` — host
Owns the **lifecycle**: Pyodide booted in a Worker, the wire protocol between host and Python (stdout/stderr/assertions/artifacts), interrupt + clearNamespace + fs. Also owns the **lab contract**: `LabManifest` schema + parser + semver gate. Knows nothing about tensors.

### `browsergrad-kernels` — GPU layer
Owns the **WGSL**. Every kernel has a JS reference for conformance + a CPU fallback path. No tensor library dependency — it's a primitives catalog. Plus the **realizer-bridge** (`createWebGpuRealizerBridge`) that browsergrad-jit consumes for its WebGPU realizer tier.

### `browsergrad-grad` — eager autograd (stable)
Owns the **eager closure-backward** path. Each op carries a Python closure that runs at `.backward()` time. NumPy-backed, no IR. This is what curriculum content uses today.

### `browsergrad-jit` — lazy IR (active development)
Owns the **UOp IR + lazy execution**. Every arithmetic operation builds an IR node; nothing realizes until you call `.numpy()` / `.item()` / `.backward()` / `optimizer.step()`. The IR enables fusion, symbolic backward, AMP cast-insertion, gradient checkpointing rewrites, functional transforms (vmap/grad/vjp), ONNX export, and pluggable backends.

## Data flow — a forward+backward pass through jit

```
1. user code:  y = x @ w + b; loss = ((y - t)**2).mean()
                  │
                  ▼ (TensorProxy.__matmul__, __add__, etc. build UOps)

2. IR graph:   REDUCE(sum)
                  │
                  ▼
                MUL
                ┌─┴─┐
              SUB   SUB
              ┌┴┐   ┌┴┐
            ADD t   ADD t          ← t = LOAD(BUFFER_t)
            ┌┴┐     ┌┴┐
          MAT b   MAT b           ← b = LOAD(BUFFER_b)
          ┌┴┐     ┌┴┐
        x   w   x   w             ← x, w = LOAD(BUFFER_x), LOAD(BUFFER_w)

3. user calls loss.backward()
   │
   ▼ _tensor_proxy.backward() decides path:
   │
   ├──► symbolic-viable? yes (every op has a VJP rule)
   │    → build a parallel "gradient UOp" graph via _vjp rules
   │    → checkpoint rewrite if any region is open
   │    → realize each leaf-grad UOp via realize(buffer_table)
   │
   └──► closure path (safety net for ops without VJPs)

4. realize() → _amp.insert_cast_pass → _fusion.fuse →
   topological NumPy dispatch → ndarray returned
                            ↑
                            └── OR, if bridge registered:
                                bg.realize_webgpu → _h_matmul / _h_fused_elementwise
                                → kernels.matmulTiledDirect etc.
                                → GPUBuffer materialise at the seam → bytes → ndarray
```

## Key design principles

### 1. Lazy by default, realize at explicit triggers
Arithmetic builds IR. Realization happens at `.numpy()`, `.tolist()`, `.item()`, `.backward()`, optimizer.step, or any Python boolean conversion (`__bool__`, `__float__`, `__int__`). The metadata path (`.shape`, `.dtype`, `.ndim`, `len()`, `repr`) NEVER realizes — important for IDE tooling.

### 2. Three layers of fallback
- **Symbolic VJP path** (fast, fusion-friendly) for ops with registered VJP rules.
- **Closure backward** for ops without VJPs (PRD-005 substrate, kept as the safety net).
- **NumPy realizer** as the universal backend; WebGPU realizer is opt-in via the bridge.

Each layer fails over to the one below with a clear error. Never silently degrades to wrong-result paths.

### 3. The realizer-tier seam is bridge-shaped
`browsergrad-jit` doesn't talk to WebGPU directly — Python in Pyodide doesn't have a WebGPU binding. Instead, jit accepts an arbitrary **bridge object** that satisfies a Protocol (`upload`, `materialize`, `release`, `matmul`, `fused_elementwise`, `cast`, `flash_attention`, `run_user_kernel`). The kernels package ships the production bridge (`createWebGpuRealizerBridge`); tests use a NumPy-backed mock. **This is the load-bearing seam** — it's what lets us validate the Python side in pyodide-in-node without a GPU and the WGSL side in headed Chromium without Python.

### 4. Honest scope cuts
Every PRD passed through a DL/GPU systems review before implementation. Each review killed speculative scope (WebNN's <5% user reach, cross-block megakernel's DRAM-bound block I/O, autotune sweeps cold-start cost) and identified the load-bearing piece (tiled GEMM, fused-elementwise codegen, Flash Attention v2). The review's verdicts live in commits; what shipped is the cut scope, not the original.

### 5. We own the codegen
The fused-elementwise WGSL is assembled from a TypeScript walker over the ops list. No template engine. No WGSL parser. The hash of the ops sequence drives the pipeline cache. Same approach for the ONNX export: hand-rolled proto3 encoder (no protobuf wheel). When a library would have added a dep we couldn't ship in Pyodide, we wrote the encoder ourselves.

## Where the seams are

| Seam | What crosses it | Why it's there |
|---|---|---|
| `session.exec({ code })` | Python source string | Host ↔ Worker; the only string-passing boundary. |
| `import browsergrad` (Pyodide) | Structured assertions/artifacts (JSON) | Lab UI receives student progress events. |
| `bg.register_webgpu_bridge(bridge)` | A Protocol-satisfying object | Pluggable backend; tests use a mock. |
| `_realize.realize(root, buffer_table)` | UOp graph + a per-session BufferTable | Single dispatch table; one Python fn per opcode. |
| `_amp.insert_cast_pass(root)` | UOp graph | Cast-insertion IR rewriter; opt-in by `autocast_hint`. |
| `_fusion.fuse(root, holdout)` | UOp graph | Pattern matcher; produces FUSED_* nodes. |
| `_vjp.get_rule(opcode)` | Function pointer | VJP registry; rules emit UOps tagged `vjp_of`. |
| `bg.save_safetensors(state)` | bytes | Browser-friendly checkpoint format. |
| `bg.onnx.export_inference(root)` | bytes | Pure-Python proto3 encoder. |
| `parseManifest(json)` | A LabManifest object | Lab contract; semver-gated. |

## Testing strategy

| Layer | Test type | Where |
|---|---|---|
| **Pure TS / structural** | vitest unit tests | `packages/*/tests/` |
| **Pyodide-in-node** (Python correctness) | vitest integration | `packages/*/tests-integration/` |
| **Real WebGPU** (kernel + bridge) | vitest browser mode + Playwright Chromium | `packages/browsergrad-kernels/tests-browser/` |
| **End-to-end feedback** | record() harness writes `/tmp/bg-feedback-report.md` | `tests-integration/end_to_end_bench.test.ts` |
| **Performance baseline** | timing sweep writes `/tmp/bg-perf-report.md` | `tests-integration/perf_bench.test.ts` |

The end-to-end + perf benches treat the library **as a user would** — every shipped PRD has one `record(prd, scenario, fn)` entry. When a real workflow breaks, the bench surfaces it as structured data in the markdown report, not just a failing assertion. This is how we found the FA-v2 kernel bug and the two API friction points fixed in v0.8.0.

## Historical refactor decisions

The pre-jit ARCHITECTURE.md tracked five candidate refactors and their grilling-then-implement cycle (single-source version, Python-as-.py-files, NodePyodideTarget adapter, torch_compat split, nn.ts split). Those decisions are preserved in git history. The current architecture above is the result; the load-bearing call from each refactor (codegen pipeline, owner-token alias protocol, per-family chunk codegen) survived into the jit library and is documented there.

## Where to start reading the source

- `packages/browsergrad-jit/src/python/_ir.py` — 28 opcodes + UOp + toposort. Smallest file; biggest leverage.
- `packages/browsergrad-jit/src/python/_realize.py` — dispatch table. One handler per opcode. Read top-to-bottom.
- `packages/browsergrad-jit/src/python/_tensor_proxy.py` — what users actually touch. Lazy semantics; arithmetic dunders build IR.
- `packages/browsergrad-jit/src/python/_vjp.py` — symbolic backward rules. Each rule emits IR.
- `packages/browsergrad-jit/src/python/_realize_webgpu.py` — the WebGPU realizer tier. Bridge-shaped.
- `packages/browsergrad-kernels/src/realizer.ts` — the production bridge.
- `packages/browsergrad-kernels/src/kernels/matmul_tiled.ts` — the tiled GEMM. Read alongside the WGSL.
- `packages/browsergrad-kernels/src/kernels/fused_elementwise.ts` — runtime WGSL codegen. ~120 LOC; representative of the "we own the codegen" stance.

If you're contributing a new opcode: see `docs/prd/PRD-005-jit-foundation.md` for the IR rules + the four-file checklist (declare in `_ir.py`, handler in `_realize.py`, VJP in `_vjp.py` or refuse-via-closure, batching rule in `_vmap.py` or refuse).
