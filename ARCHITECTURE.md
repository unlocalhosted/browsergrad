# Architecture

How the packages compose, what each layer owns, and why the calls were made the way they were.

## Layered picture

```
┌──────────────────────────────────────────────────────────────────┐
│                       Host application                          │
│   (notebook UI, lab platform, demo page, ML playground, …)       │
└─────────────────────────────┬────────────────────────────────────┘
                              │  createSession + exec()
┌─────────────────────────────▼────────────────────────────────────┐
│                @unlocalhosted/browsergrad-runtime                │
│  Spawns Pyodide in a Worker. Exposes exec / fs / interrupt /     │
│  onAssertion / onArtifact. Ships an optional manifest validator  │
│  with a semver gate for platforms that pin runtime versions.     │
└─────────┬────────────────────────────────────────────────────┬───┘
          │  install*()                                        │  createWebGpuRealizerBridge(device)
          │                                                    │
┌─────────▼──────────────────┐  ┌──────────────────┐    ┌──────▼──────────────────────────┐
│  @unlocalhosted/           │  │  @unlocalhosted/ │    │  @unlocalhosted/                │
│  browsergrad-grad          │  │  browsergrad-jit │    │  browsergrad-kernels            │
│  (eager, NumPy-backed)     │  │  (lazy IR)       │    │  (WGSL + JS reference)          │
│                            │  │                  │    │                                 │
│  Tensor + closure backward │  │  TensorProxy +   │    │  matmul / matmulTiled /         │
│  nn.{Linear, Conv, ...}    │  │  UOp IR (28 ops) │    │  softmax / layernorm /          │
│  optim.{SGD, Adam, AdamW}  │  │  Fusion          │    │  attention / FA-v2 /            │
│                            │◀─┤  Symbolic VJP    │    │  fusedElementwise (codegen)     │
│                            │  │  AMP + GradScaler│    │                                 │
│                            │  │  Checkpointing   │────▶  WebGPU realizer bridge         │
│                            │  │  bg.func.*       │    │  (forward-only GPU path)        │
│                            │  │  bg.onnx.*       │    │                                 │
│                            │  │  bg.custom_kernel│────▶  Runs user-supplied WGSL        │
└────────────────────────────┘  └──────────────────┘    └─────────────────────────────────┘
                                                              ▲
                                                              │ Kernel IR / WGSL dispatch
                                                ┌─────────────┴───────────────────────┐
                                                │ @unlocalhosted/browsergrad-compiler │
                                                │ CUDA-lite parser/analyzer, CPU      │
                                                │ reference, WGSL emitter, WebGPU     │
                                                │ runner, corpus compatibility gates  │
                                                └─────────────────────────────────────┘
```

The runtime knows nothing about tensors. The kernels package knows nothing about Python. The two ML libraries pick their own backends — grad is eager NumPy, jit is lazy IR with pluggable realizers. Each piece composes with the others but doesn't depend on them at runtime.
The compiler owns CUDA-lite semantics and lowers learner kernels to Kernel IR,
CPU reference execution, WGSL, and real WebGPU dispatch through the kernels
package. It is course-agnostic; corpus support lives in generic source
normalization, diagnostics, and compatibility gates.

For the multi-course guided-lab layer that sits above these packages, see
`docs/platform/curriculum-platform-architecture.md`.
For the browser-native systems/kernel lab direction, see
`docs/platform/kernel-lab-foundation.md`.

## Package responsibilities

### `browsergrad-runtime` — host layer
Owns the lifecycle: Pyodide booted in a Worker, the wire protocol between host and Python (stdout / stderr / structured assertions / structured artifacts), interrupt + clearNamespace + fs. Ships a `LabManifest` schema + parser + semver gate for platforms that want to pin runtime versions on a per-resource basis. No tensor library dependency.

### `browsergrad-kernels` — GPU layer
Owns the WGSL. Every kernel has a JS reference for conformance + a CPU fallback path. A primitives catalog — no tensor library dependency. Also ships `createWebGpuRealizerBridge`, the production WebGPU bridge that `browsergrad-jit` consumes for its WebGPU realizer tier.

### `browsergrad-compiler` — CUDA-lite compiler layer
Owns CUDA-lite parsing, source/context normalization, semantic analysis,
Kernel IR lowering, lockstep CPU reference execution, WGSL emission, and
WebGPU runner orchestration. It consumes `browsergrad-kernels` dispatch helpers
but keeps CUDA/C++ compatibility heuristics out of platform code. Its
real-world corpus gates compile/codegen pinned CUDA corpora and execute selected
fixtures in Chromium/WebGPU against CPU reference outputs.

### `browsergrad-grad` — eager autograd
Closure-based reverse-mode autograd in Python. Each op carries a closure that runs at `.backward()` time. NumPy-backed, no IR, no lazy semantics. Stable; designed to be readable source code.

### `browsergrad-jit` — lazy IR
The UOp graph + lazy execution path. Every arithmetic op builds an IR node; nothing realizes until `.numpy()` / `.item()` / `.backward()` / `optimizer.step()`. The IR enables fusion, symbolic backward, AMP cast-insertion, gradient-checkpointing IR rewrites, functional transforms (vmap / grad / vjp / functional_call), custom WGSL kernels, ONNX export, and pluggable backends.

### `browsergrad-primitives` — small primitive facade
Canonical public interface for browser-safe text, data, evaluation,
simulation, hosted-training, and RL/math primitives. It keeps generic caller
vocabulary such as references, comparators, fixtures, and simulators. The
legacy leaf packages (`browsergrad-tokenizers`, `browsergrad-data`,
`browsergrad-snapshots`, `browsergrad-scaling`, `browsergrad-simulators`, and
`browsergrad-alignment`) are implementation shards and compatibility surfaces,
not the product identity.

Profile-specific bridge objects that exist only to expose snake_case or JSON
methods to Python rubrics are not primitive interfaces. Keep those wrappers in
runtime tests, profile modules, or platform code; keep package exports useful
for non-course browser ML tools.

## Data flow — a forward+backward pass through jit

```
1. user code:  y = x @ w + b; loss = ((y - t) ** 2).mean()
                  │
                  ▼ (TensorProxy.__matmul__ + __add__ + __sub__ + __pow__ + .mean()
                  │   build UOp nodes; nothing executes)

2. IR graph:   REDUCE(mean)
                  │
                  ▼
                MUL                      ← (y-t)*(y-t) from __pow__(2) lowering
              ┌─┴─┐
              SUB SUB
              ┌┴┐  ┌┴┐
            ADD t   ADD t                ← t = LOAD(BUFFER_t)
            ┌┴┐    ┌┴┐
          MAT b   MAT b                  ← b = LOAD(BUFFER_b)
          ┌┴┐    ┌┴┐
        x   w  x   w                     ← x, w = LOAD(BUFFER_x), LOAD(BUFFER_w)

3. user calls loss.backward()
   │
   ▼ _tensor_proxy.backward() picks path:
   │
   ├──► symbolic-viable? yes (every op on the chain has a VJP rule)
   │    → build a parallel "gradient UOp" graph via _vjp rules
   │    → apply checkpoint rewrite if any region is open
   │    → realize each leaf-grad UOp via realize(buffer_table)
   │
   └──► closure path (safety net for ops without VJPs)

4. realize() → insert_cast_pass (AMP) → fuse() → topological dispatch
                                                  ↓
                            either NumPy handlers OR — if a bridge is
                            registered — the WebGPU realizer that
                            dispatches via runDirect to kernels.matmulTiled,
                            kernels.fusedElementwise, kernels.flashAttention, …
                            then materializes bytes back at the seam.
```

## Key design principles

### 1. Lazy by default, realize at explicit triggers
Arithmetic builds IR. Realization happens at `.numpy()`, `.tolist()`, `.item()`, `.backward()`, `optimizer.step()`, or any Python boolean conversion (`__bool__`, `__float__`, `__int__`). The metadata path (`.shape`, `.dtype`, `.ndim`, `len()`, `repr`) NEVER realizes — important for IDE tooling.

### 2. Three layers of fallback
- **Symbolic VJP path** (fast, fusion-friendly) for ops with registered VJP rules.
- **Closure backward** for ops without VJPs (kept as the safety net since v0.1).
- **NumPy realizer** as the universal backend; WebGPU realizer is opt-in via the bridge.

Each layer fails over to the one below with a clear error. Never silently degrades to a wrong-result path.

### 3. The realizer-tier seam is bridge-shaped
`browsergrad-jit` doesn't talk to WebGPU directly — Python in Pyodide has no WebGPU binding. Instead, jit accepts an arbitrary **bridge object** that satisfies a Protocol (`upload`, `materialize`, `release`, `matmul`, `fused_elementwise`, `cast`, `flash_attention`, `run_user_kernel`). The kernels package ships the production bridge; tests use a NumPy-backed mock. This is the load-bearing seam — it's what lets us validate the Python side in pyodide-in-node without a GPU and the WGSL side in headed Chromium without Python.

### 4. We own the codegen
The fused-elementwise WGSL is assembled from a TypeScript walker over the ops list. No template engine. No WGSL parser. The hash of the ops sequence drives the pipeline cache. Same approach for the ONNX export: hand-rolled proto3 encoder (no protobuf wheel). When a library would have required a dependency unavailable in Pyodide, we wrote the encoder ourselves.

### 5. Honest scope cuts
Every major feature passes through a design review before implementation. Each review kills speculative scope and identifies the load-bearing piece. The verdicts live in commits; what shipped is the cut scope, not the original. See `docs/prd/` for the per-PRD design records.

## Seams

| Seam | What crosses it | Why it's there |
|---|---|---|
| `session.exec({ code })` | Python source string | Host ↔ Worker; the only string-passing boundary. |
| Pyodide's `_bg_native` module | Structured assertions / artifacts (JSON) | Host receives student-progress / lab-rubric events. |
| `createSession({ jsModules })` | Worker-imported JS oracle modules | Platform rubrics call browser-safe JS references from Python. |
| `bg.register_webgpu_bridge(bridge)` | A Protocol-satisfying object | Pluggable GPU backend; tests use a mock. |
| `_realize.realize(root, buffer_table)` | UOp graph + per-session BufferTable | Single dispatch table; one Python function per opcode. |
| `_amp.insert_cast_pass(root)` | UOp graph | Cast-insertion IR rewriter; opt-in via `autocast_hint`. |
| `_fusion.fuse(root, holdout)` | UOp graph | Pattern matcher; produces FUSED_* nodes. |
| `_vjp.get_rule(opcode)` | Function pointer | VJP registry; rules emit UOps tagged `vjp_of`. |
| `bg.save_safetensors(state)` | bytes | Browser-friendly checkpoint format. |
| `bg.onnx.export_inference(root)` | bytes | Pure-Python proto3 encoder. |
| `parseManifest(json)` + `assertCompatibleRuntime` | A `LabManifest` object | Optional contract for platforms that pin runtime versions. |

## Testing strategy

| Layer | Type | Location |
|---|---|---|
| Pure TS / structural | vitest unit | `packages/*/tests/` |
| Pyodide-in-node (Python correctness) | vitest integration | `packages/*/tests-integration/` |
| Real WebGPU (kernel + bridge) | vitest browser + Playwright Chromium | `packages/browsergrad-kernels/tests-browser/` |
| End-to-end feedback | record() harness → structured JSON | `packages/browsergrad-jit/tests-integration/end_to_end_bench.test.ts` |
| Performance baseline | timing sweep → markdown report | `packages/browsergrad-jit/tests-integration/perf_bench.test.ts` |

## Where to start reading the source

- `packages/browsergrad-jit/src/python/_ir.py` — 28 opcodes + UOp + toposort. Smallest file; biggest leverage.
- `packages/browsergrad-jit/src/python/_realize.py` — dispatch table. One handler per opcode. Read top-to-bottom.
- `packages/browsergrad-jit/src/python/_tensor_proxy.py` — what users actually touch. Lazy semantics; arithmetic dunders build IR.
- `packages/browsergrad-jit/src/python/_vjp.py` — symbolic backward rules. Each rule emits IR.
- `packages/browsergrad-jit/src/python/_realize_webgpu.py` — the WebGPU realizer tier. Bridge-shaped.
- `packages/browsergrad-kernels/src/realizer.ts` — the production bridge.
- `packages/browsergrad-kernels/src/kernels/matmul_tiled.ts` — the tiled GEMM. Read alongside the WGSL.
- `packages/browsergrad-kernels/src/kernels/fused_elementwise.ts` — runtime WGSL codegen. ~120 LOC; representative of the "we own the codegen" stance.
- `packages/browsergrad-primitives/src/index.ts` — canonical facade for small
  primitive helpers.

If you're adding a new opcode, the contract is in `docs/prd/PRD-005-jit-foundation.md`: declare in `_ir.py`, handler in `_realize.py`, VJP in `_vjp.py` (or refuse-via-closure), batching rule in `_vmap.py` (or refuse with a clear message).
