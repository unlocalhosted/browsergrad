# PRD-016 — ONNX Export: From Browser Training to Production Inference

| Field | Value |
|---|---|
| **Status** | Draft v1 |
| **Author** | unlocalhosted maintainers |
| **Date** | 2026-05-26 |
| **PRD ID** | PRD-016 |
| **Phase** | P2 stretch goal (Months 10–14 of the 14-month roadmap in PRD.md §6; ships behind `bg.config.experimental_onnx_export=True`) |
| **Package** | `@unlocalhosted/browsergrad-jit` (new submodule `onnx/`); optional companion `@unlocalhosted/browsergrad-onnx-runtime-bridge` for round-trip verification |
| **Depends on** | PRD-005 (IR — the 19-opcode graph we serialise), PRD-007 (symbolic backward — supplies the training-graph export branch), PRD-008 (OPFS + safetensors — the analog for weight serialisation) |
| **Enables** | "Browser to production" deployment path: CoreML on iOS, TensorRT on NVIDIA, OpenVINO on Intel, ONNX Runtime Web on every browser, onnx-mlir → native binary on anything else |
| **Companion docs** | [VISION.md](../../VISION.md) §3 — bridge to deployment · [PRD.md](../../PRD.md) §6 P2 stretch · [PRD-011](PRD-011-webnn-backend.md) §References — ORT-Web WebNN EP is the same partition pattern in reverse |

---

## TL;DR

Today browsergrad is a learning tool: a student can train a model in their browser, see the loss curve, watch gradients flow — and that's where the journey stops. If they want to *deploy* what they trained, there is no path out. PRD-016 builds that path. `torch.onnx.export(model, args, "model.onnx")` — the same call site PyTorch uses ([pytorch.org/docs/stable/onnx.html](https://pytorch.org/docs/stable/onnx.html)) — traces the model via the PRD-005 JIT, walks the resulting IR, and serialises it as an ONNX `ModelProto` ([onnx.ai/onnx/intro/concepts.html](https://onnx.ai/onnx/intro/concepts.html)) targeting opset 18. The IR is a near-perfect match for ONNX: 17 of our 19 opcodes have direct 1:1 ONNX equivalents, two (`LOAD`/`STORE`/`CONST`/`BUFFER`) fold into graph lifecycle, and weights serialise as `TensorProto` initializers analogous to how safetensors loading works in PRD-008. The serialiser is a hand-rolled JavaScript protobuf writer driven by `onnx.proto3` ([github.com/onnx/onnx/blob/main/onnx/onnx.proto3](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3)) — we do not pull Python's `onnx` package into Pyodide because it bloats the wheel by ~5 MB. Verification is in-tab: the exported bytes are immediately re-imported into onnx-runtime-web ([onnxruntime.ai/docs/build/web.html](https://onnxruntime.ai/docs/build/web.html)), forward is run on identical inputs, and outputs must match within 1e-4. With this PRD, a student trains in browsergrad and downloads a `.onnx` that runs unchanged in CoreML, TensorRT, OpenVINO, or any production engine. Without it, browsergrad ends at the browser tab.

---

## Background

### Why ONNX, why now

The roadmap in PRD.md §1 calls out ONNX export as an **explicit non-goal** for v0.5.0 and the P0/P1 windows. That was correct: until the JIT lands (PRD-005) there is no IR to export, and until backward stabilises (PRD-007) the training-graph branch can't exist. P2 is when both prerequisites land, and at that point the cost-benefit reverses. The non-goal becomes a stretch goal precisely because the IR makes export almost free.

The pedagogical case is straightforward. Today a student running browsergrad in a tab can do the following: load weights, train, evaluate, visualise. They cannot do this: ship the trained network to a phone, a microcontroller, a CDN, or any inference engine outside the browser tab. That last step is what production ML *is*. Without it, the runtime is a sandbox; with it, the runtime is a real prototyping environment whose output graduates into real systems.

The technical case is that ONNX is the lingua franca. The [ONNX op catalog](https://github.com/onnx/onnx/blob/main/docs/Operators.md) is consumed by:

- **ONNX Runtime** (Microsoft) — CPU/CUDA/DirectML/CoreML/TensorRT execution providers; powers Bing, Office, and the Windows AI Studio runtime.
- **TensorRT** (NVIDIA) — [nvidia.com/en-us/deep-learning-ai/products/tensorrt](https://www.nvidia.com/en-us/deep-learning-ai/products/tensorrt/); the production path for NVIDIA inference, consumes `.onnx` directly.
- **CoreML** (Apple) — `coremltools` converts `.onnx` → `.mlmodel`; deploys to iPhone/Mac via the Neural Engine.
- **OpenVINO** (Intel) — converts `.onnx` to its IR for CPU/iGPU/NPU inference.
- **onnx-mlir** (IBM) — [github.com/onnx/onnx-mlir](https://github.com/onnx/onnx-mlir) compiles `.onnx` to native binaries via MLIR; the "browser → native executable" path.
- **ONNX Runtime Web** — [onnxruntime.ai/docs/build/web.html](https://onnxruntime.ai/docs/build/web.html) runs `.onnx` in a different browser tab via WASM or WebGPU, useful for the round-trip verification step.

A `.onnx` file is the closest thing the ML ecosystem has to a "binary that runs anywhere." Once browsergrad can emit one, every downstream engine becomes a deployment target — for free, without us writing a single new backend.

### Why our IR is almost ideal for export

PRD-005 §Architecture defines a 19-opcode IR. The [ONNX op catalog](https://github.com/onnx/onnx/blob/main/docs/Operators.md) defines ~190 ops at opset 18; the intersection covers our entire surface with one-to-one mappings:

| browsergrad UOp | ONNX op (opset 18) | Notes |
|---|---|---|
| `ADD`, `MUL`, `DIV`, `NEG` | `Add`, `Mul`, `Div`, `Neg` | Direct. NumPy broadcasting matches ONNX broadcasting. |
| `EXP`, `LOG` | `Exp`, `Log` | Direct. |
| `MATMUL` | `MatMul` (batched) or `Gemm` (2D, with optional alpha/beta) | We emit `MatMul`; ORT optimises to `Gemm` automatically. |
| `REDUCE` (sum/max/min) | `ReduceSum`, `ReduceMax`, `ReduceMin` | Direct. `axes` attribute. |
| `REDUCE` (mean) | `ReduceMean` | Direct. |
| `CAST` | `Cast` | `to` attribute carries dtype enum (`FLOAT=1`, `FLOAT16=10`, `INT64=7`, ...). |
| `RESHAPE` | `Reshape` | Shape is a graph input tensor in ONNX, not an attribute — we materialise it as an initializer. |
| `PERMUTE` | `Transpose` | `perm` attribute. |
| `PAD` | `Pad` | Mode + pads tensor; constant & reflect modes covered. |
| `SLICE` | `Slice` | `starts`/`ends`/`axes`/`steps` as inputs (opset 10+). |
| `WHERE` | `Where` | Direct. |
| `GATHER` | `Gather` / `GatherElements` | Direct; dispatch on `dim` semantics. |
| `LOAD`, `STORE`, `BUFFER`, `CONST` | n/a — graph lifecycle | Folded into `graph.initializer` and `graph.input`/`graph.output`. |

**17 of 19 opcodes have a direct 1:1 mapping.** The remaining four (`LOAD`/`STORE`/`BUFFER`/`CONST`) are bookkeeping nodes that the IR uses to track tensor identity and that ONNX expresses through graph structure — they vanish in the lowered representation.

This is the same mapping pattern PRD-011's WebNN backend uses ([PRD-011 §Architecture](PRD-011-webnn-backend.md)). The fact that both ONNX and WebNN consume our IR with nearly identical translation tables is not coincidence: the IR was designed against the tinygrad-style "minimal primitive set" that all modern op catalogs converge on. Export is the third consumer of the same lowering machinery, after WGSL codegen (PRD-006) and WebNN (PRD-011).

### Why hand-rolled protobuf, not the Python `onnx` library

The obvious approach — `import onnx; onnx.save(...)` — fails for two reasons:

1. **Wheel size.** The `onnx` PyPI package is 16 MB of compiled protobuf bindings plus its Python wrappers. Pyodide must download every wheel; that download cost is paid by every student every cold-start. The PRD-008 cold-start budget is 200 ms over network; a 16 MB wheel blows it by 80×.
2. **Pyodide compatibility.** `onnx` ships native protobuf bindings (`onnx_cpp2py_export.so`) that are not compiled for `wasm32-unknown-emscripten`. The pure-Python fallback exists but is partial. We'd be on the slow path even if we paid the wheel cost.

The cheaper path is a hand-rolled JS-side protobuf writer driven by [onnx.proto3](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3). The schema is small (one `.proto` file, ~600 lines). Proto3 wire format is well-documented ([protobuf.dev/programming-guides/encoding](https://protobuf.dev/programming-guides/encoding/)) — varints, length-delimited fields, packed repeated. Compared against [protobuf-ts](https://github.com/timostamm/protobuf-ts), which auto-generates TypeScript serialisers from `.proto` files, we have two viable implementation paths:

- **Path A (preferred):** generate TS serialisers from `onnx.proto3` with `protobuf-ts` at build time; ship the ~80 KB generated code as part of `browsergrad-jit`. Zero hand-written encoding logic; spec-conformant by construction; updates pin to the ONNX repo tag.
- **Path B (fallback if `protobuf-ts` is unworkable):** hand-write a minimal proto3 writer covering exactly the ~30 `ModelProto`/`GraphProto`/`NodeProto`/`TensorProto`/`AttributeProto`/`ValueInfoProto`/`TypeProto` fields we emit. Estimated 400 LOC; we audit and own it.

We start on Path A, with Path B in reserve if `protobuf-ts` adds runtime baggage we can't justify. Either way, the encoding happens on the JS side: the Python `torch.onnx.export(...)` call collects the IR and the weights, hands them to a JS callback via the existing Pyodide bridge (the same one PRD-008 uses for safetensors), and JS produces the bytes that become the download.

### Inference vs training graph export

Most production ONNX consumers care only about inference. CoreML doesn't import training graphs; TensorRT doesn't import training graphs; ORT can import them but rarely does. So the *default* export emits the forward graph only — exactly what PyTorch's `torch.onnx.export` does ([pytorch.org/docs/stable/onnx.html](https://pytorch.org/docs/stable/onnx.html)).

But we have a second branch the ecosystem doesn't usually offer: PRD-007 builds the backward graph as more 19-opcode IR. Those backward subgraphs are perfectly valid forward graphs from ONNX's perspective. With `export_training_graph=True`, we emit both the forward `ModelProto` and a sibling `<name>.training.onnx` containing the backward + optimizer update graph. This is rare — the [ONNX Training spec extension](https://github.com/onnx/onnx/blob/main/docs/IR.md#training-information) defines `TrainingInfoProto` for exactly this case but is sparsely supported downstream. We emit it because we can, and because the "train in browser, export training graph, fine-tune on device" loop is a credible future workflow even if no downstream engine consumes it today.

---

## User Stories

**U1 — Train in browser, deploy to phone.** A student trains a 4-layer MNIST classifier in browsergrad to 97% test accuracy. They call `torch.onnx.export(model, dummy_input, "mnist.onnx")`. A `.onnx` file (1.2 MB) downloads via the browser. They open it in `coremltools` on their Mac, convert to `mnist.mlmodel`, drop it into an iOS app, and run inference at native Neural Engine speed. Total elapsed time from clicking "export" to mobile inference: ~10 minutes.

**U2 — In-tab round-trip verification.** The same student is sceptical the exported model is correct. Browsergrad's export call returns a verification report: the exported bytes were loaded into onnxruntime-web in the same tab, forward was run on the original `dummy_input`, and the output matched the browsergrad forward to 5.3e-6 max absolute error. The student trusts the file.

**U3 — Edge-case op rejection.** An engineer is exporting a custom transformer that uses a registered op `bg.ops.spiky_softmax` outside the 19-opcode core. `torch.onnx.export(...)` raises `OnnxUnmappableOp: spiky_softmax (node #47) has no ONNX opset-18 equivalent. Options: (1) replace with composed primitives via bg.ops.spiky_softmax_decomposed; (2) keep model in browsergrad for further iteration; (3) contribute a mapping at <repo URL>/onnx/op_table.py`. The engineer follows option 1 and re-exports successfully.

**U4 — Dynamic batch axis.** A developer wants a model that accepts variable batch sizes. They call `torch.onnx.export(model, x, "model.onnx", dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})`. The exported graph carries symbolic batch dimensions, and ORT-web can run it with B=1, B=8, or B=128 without re-export.

**U5 — onnx-mlir to native binary.** A researcher wants to deploy a browsergrad-trained classifier as a static native binary on a Linux server. They run `onnx-mlir --EmitObj model.onnx` to produce `model.o`, link it into their inference daemon, and run at native CPU speed with zero browser/JS dependencies. The browsergrad-trained network becomes a `.so`.

**U6 — Export the training graph.** A course author exporting a small RNN passes `export_training_graph=True`. Two files download: `rnn.onnx` (forward only, 240 KB) and `rnn.training.onnx` (forward + backward + Adam step, 680 KB). The latter is not widely consumable today, but the author archives it as a complete record of what the network actually does end-to-end.

---

## Goals and Non-Goals

### Goals

1. Ship `torch.onnx.export(model, args, path, **kwargs)` matching PyTorch's [export signature](https://pytorch.org/docs/stable/onnx.html) for the subset of kwargs we support: `opset_version`, `dynamic_axes`, `input_names`, `output_names`, `do_constant_folding`, plus our additions `export_training_graph`, `verify`.
2. Cover all 19 IR opcodes with ONNX op-table entries; emit an `OnnxUnmappableOp` for any user-registered op outside the core.
3. Pin opset 18 as the default emission target; gate opset 20+ behind `opset_version=20`.
4. Hand-rolled or `protobuf-ts`-generated JS-side serialisation; **zero new Python wheel weight**. Cold-start delta from PRD-008 baseline must be ≤ 5 ms.
5. Emit weights as `TensorProto` initializers in `graph.initializer`, raw or external-data per the [ONNX External Data spec](https://github.com/onnx/onnx/blob/main/docs/ExternalData.md) for >1 GB models (rare in browser, but supported).
6. **Numerical conformance**: re-import the exported file via `onnxruntime-web` in the same tab and assert forward outputs match the browsergrad forward within 1e-4 absolute, 1e-3 relative. This is a default-on verification step gated by `verify=True` (default `True`).
7. Static-shape export is v1; `dynamic_axes` kwarg for symbolic dimensions is v1 too (matches PyTorch's surface).
8. Training-graph export branch is v1 behind `export_training_graph=True` (default `False`).
9. Trigger a download via `<a download="model.onnx">` from the JS bridge when the call is made in a browser context; return the bytes as a `bytes` object when called in a Node test context.

### Non-Goals

1. **Custom op extension surface.** We do not let users register custom ONNX domains beyond `ai.onnx`. If a user needs a new op, contribute to the IR opcode table, then to the ONNX op-table.
2. **Quantised export.** INT8/FP8 quantisation export is a separate PRD (deferred; ties into PRD-010 mixed precision).
3. **`torch.export.export` API surface.** PyTorch is mid-migration to `torch.export` ([pytorch.org/docs/stable/export.html](https://pytorch.org/docs/stable/export.html)); we stick with the stable `torch.onnx.export` call.
4. **ONNX import.** Loading `.onnx` *into* browsergrad to fine-tune in browser is a credible future PRD but is out of scope here. PRD-008's safetensors path is the v1 weight-import story.
5. **Opset < 18.** Legacy opset support adds a long tail of compat shims for marginal user benefit. If you need opset 11, run an ONNX version converter post-export.
6. **Bundling onnxruntime-web in the main bundle.** ORT-web is heavy (~7 MB). We lazy-load it only when `verify=True` and only on the first call. Users who don't verify pay zero.
7. **WebNN execution provider routing of the verification step.** The verification step uses ORT-web's default (WASM) execution provider for determinism. WebNN-routed verification is a follow-up.

---

## Architecture

### Module layout

```
packages/browsergrad-jit/
  src/python/onnx/
    export.py                # torch.onnx.export entry point
    op_table.py              # IR opcode → ONNX op-name + attribute mapper
    serialise.py             # Python-side trace collection, hands off to JS
    verify.py                # round-trip verification (calls ORT-web via JS bridge)
  src/ts/onnx/
    proto/
      onnx.proto3            # vendored from onnx repo, version-pinned
      onnx.generated.ts      # protobuf-ts output (gitignored, built at npm pack)
    writer.ts                # ModelProto/GraphProto/TensorProto builders
    download.ts              # <a download> trigger in browser, bytes in Node
    ort_bridge.ts            # lazy onnxruntime-web loader for verify path
```

### End-to-end flow

```
User Python: torch.onnx.export(model, x, "model.onnx")
   │
   ▼  export.py
trace model via PRD-005 (forward only, or forward+backward if training_graph)
   │
   ▼
IR graph (19-opcode UOps) + parameter buffers
   │
   ▼  op_table.py
Walk IR topologically, produce ONNX node list (Python-side dicts):
   [{op: "MatMul", inputs: [...], outputs: [...], attrs: {...}}, ...]
   │
   ▼  serialise.py
Bridge call to JS: pyodide.runPython gives JS a structured payload
   {nodes, initializers, inputs, outputs, opset_version, ir_version}
   │
   ▼  ts/onnx/writer.ts
Build ModelProto via generated proto3 encoder
   │
   ▼  ts/onnx/download.ts
Trigger <a download> in browser, or return Uint8Array in Node
   │
   ▼  verify.py + ts/onnx/ort_bridge.ts (if verify=True)
Lazy-load onnxruntime-web, create InferenceSession from bytes,
run forward on the same args, assert |out_onnx - out_bg| < 1e-4
   │
   ▼
Return ExportReport(path, size_bytes, num_nodes, num_params, verify_max_abs_err)
```

### Op-table sketch (`op_table.py`)

```python
def lower_uop(node: UOp, ctx: ExportContext) -> OnnxNode:
    inputs  = [ctx.name_of(inp) for inp in node.inputs]
    outputs = [ctx.fresh_name()]
    if node.op == "ADD":      return OnnxNode("Add",    inputs, outputs)
    if node.op == "MUL":      return OnnxNode("Mul",    inputs, outputs)
    if node.op == "MATMUL":   return OnnxNode("MatMul", inputs, outputs)
    if node.op == "REDUCE":
        onnx_op = {"sum": "ReduceSum", "max": "ReduceMax",
                   "min": "ReduceMin", "mean": "ReduceMean"}[node.arg["op"]]
        return OnnxNode(onnx_op, inputs, outputs,
                        attrs={"axes": [node.arg["axis"]],
                               "keepdims": int(node.arg["keepdims"])})
    if node.op == "RESHAPE":
        shape_init = ctx.add_initializer_int64(node.arg["new_shape"])
        return OnnxNode("Reshape", inputs + [shape_init.name], outputs)
    if node.op == "PERMUTE":
        return OnnxNode("Transpose", inputs, outputs,
                        attrs={"perm": list(node.arg["axes"])})
    if node.op == "PAD":
        pads_init  = ctx.add_initializer_int64(node.arg["pad_width"])
        value_init = ctx.add_initializer_scalar(0.0)
        return OnnxNode("Pad", inputs + [pads_init.name, value_init.name],
                        outputs, attrs={"mode": node.arg["mode"]})
    if node.op == "SLICE":
        starts, ends, axes, steps = _slice_args(node.arg["slices"])
        s_i = ctx.add_initializer_int64(starts)
        e_i = ctx.add_initializer_int64(ends)
        a_i = ctx.add_initializer_int64(axes)
        st_i = ctx.add_initializer_int64(steps)
        return OnnxNode("Slice",
                        inputs + [s_i.name, e_i.name, a_i.name, st_i.name],
                        outputs)
    if node.op == "WHERE":    return OnnxNode("Where",  inputs, outputs)
    if node.op == "GATHER":
        return OnnxNode("Gather", inputs, outputs,
                        attrs={"axis": node.arg["dim"]})
    if node.op == "CAST":
        return OnnxNode("Cast", inputs, outputs,
                        attrs={"to": _dtype_to_onnx_enum(node.arg["dtype"])})
    if node.op == "EXP":      return OnnxNode("Exp", inputs, outputs)
    if node.op == "LOG":      return OnnxNode("Log", inputs, outputs)
    if node.op == "NEG":      return OnnxNode("Neg", inputs, outputs)
    if node.op == "DIV":      return OnnxNode("Div", inputs, outputs)
    if node.op in ("LOAD", "STORE", "BUFFER", "CONST"):
        return None  # folded into graph structure
    raise OnnxUnmappableOp(node)
```

`ExportContext` maintains the name table, dedup of initializers, and the running `opset_version` ceiling.

### Shape inference

ONNX requires every tensor have a declared shape (or symbolic dimension) on every graph value-info. Our IR carries `shape: Tuple[int, ...]` on every UOp by construction (PRD-005 §IR Design). So shape inference is **already done** — we read `node.shape` and emit a `ValueInfoProto` per intermediate. Symbolic dimensions for `dynamic_axes` are emitted as `dim_param: "batch"` instead of `dim_value: <int>`.

### Initializer serialisation

Each `BUFFER` UOp backed by a parameter becomes a `TensorProto` in `graph.initializer`. For typical models (≤ 1 GB parameters), data is inlined as `raw_data: bytes` in little-endian wire format — float32 is 4 bytes per element, int64 is 8 bytes, fp16 is 2 bytes. For models > 1 GB (rare in browser but possible with PRD-008's OPFS-cached weights), we emit `external_data` references and write a sibling `.onnx_data` file per [ONNX External Data Format](https://github.com/onnx/onnx/blob/main/docs/ExternalData.md). The download UI then triggers two `<a download>` clicks, one per file.

The encoding side runs entirely in TS. The Python side bridges raw bytes via Pyodide's `pyodide.toJs({create_proxies: false})` so the `np.ndarray.tobytes()` payload crosses the JS/Python boundary zero-copy.

### Protobuf writer (`ts/onnx/writer.ts`)

Path A using `protobuf-ts`:

```typescript
import { ModelProto, GraphProto, NodeProto, TensorProto } from "./proto/onnx.generated";

export function buildModelProto(payload: ExportPayload): Uint8Array {
  const graph = GraphProto.create({
    node: payload.nodes.map(n => NodeProto.create({
      opType: n.op,
      input:  n.inputs,
      output: n.outputs,
      attribute: encodeAttrs(n.attrs),
      name: n.name,
    })),
    initializer: payload.initializers.map(buildTensorProto),
    input:  payload.inputs.map(buildValueInfo),
    output: payload.outputs.map(buildValueInfo),
    valueInfo: payload.valueInfos.map(buildValueInfo),
  });
  const model = ModelProto.create({
    irVersion: 9n,                       // ONNX IR v9, matches opset 18
    opsetImport: [{ domain: "", version: BigInt(payload.opsetVersion) }],
    producerName: "browsergrad",
    producerVersion: PKG_VERSION,
    graph,
  });
  return ModelProto.toBinary(model);
}
```

Path B (hand-rolled fallback) implements the ~30 wire-format encoders we need (varint, fixed32, fixed64, length-delimited, packed-repeated, embedded-message). ~400 LOC, mirrors the proto3 spec section by section, hand-audited.

### Round-trip verification (`verify.py` + `ts/onnx/ort_bridge.ts`)

```python
def _verify_roundtrip(bytes_: bytes, args, original_outputs):
    from .ort_bridge import run_in_ort_web
    onnx_outputs = run_in_ort_web(bytes_, args)
    max_abs = max(np.max(np.abs(a - b)) for a, b in zip(original_outputs, onnx_outputs))
    if max_abs >= 1e-4:
        raise OnnxVerificationFailed(max_abs=max_abs, threshold=1e-4)
    return VerifyReport(max_abs_err=max_abs)
```

```typescript
let _ortSession: ort.InferenceSession | null = null;
export async function runInOrtWeb(bytes: Uint8Array, args: TensorMap): Promise<TensorMap> {
  if (!_ortLoaded) await import("onnxruntime-web").then(m => { ort = m; });
  _ortSession = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  return await _ortSession.run(args);
}
```

`onnxruntime-web` is a dynamic `import()` — it lands in a separate chunk that only loads if `verify=True`. Cold-start budget unchanged for non-verifying users.

### Training-graph branch

When `export_training_graph=True`, we trace **both** forward and backward via PRD-007's symbolic backward pass. The forward graph is emitted as `model.onnx`. The full forward+backward+optimizer-step graph is emitted as `model.training.onnx` with a `TrainingInfoProto` block per the [ONNX Training Information spec](https://github.com/onnx/onnx/blob/main/docs/IR.md#training-information):

```
TrainingInfoProto {
  initialization_binding: [(param_name, init_tensor_name), ...]
  update_binding:         [(param_name, updated_tensor_name), ...]
  algorithm: GraphProto (the backward + Adam step subgraph)
}
```

This is rarely consumed but matches the spec exactly so any ONNX-Training-aware engine (research engines mostly) can pick it up.

### Dynamic axes

`dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}` tells the exporter to emit symbolic dimensions:

```python
def _dim_proto(size: int, name: Optional[str]) -> Dim:
    if name is not None: return Dim(dim_param=name)
    else:                return Dim(dim_value=size)
```

The IR's static shapes are kept for op attribute emission (e.g. `Reshape` initializers still use concrete numbers), but the *graph-level* input/output value-infos carry the symbolic dims. ORT will accept any concrete size at those axes at runtime.

---

## API Surface

### Python — user-facing

```python
import browsergrad_jit as torch
import browsergrad_jit.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3, padding=1)
        self.fc   = nn.Linear(16 * 28 * 28, 10)
    def forward(self, x):
        return self.fc(torch.relu(self.conv(x)).reshape(x.shape[0], -1))

model = CNN()
# ... train ...

# Static-shape export, defaults
dummy = torch.randn(1, 1, 28, 28)
report = torch.onnx.export(model, dummy, "mnist.onnx")
print(report)
# ExportReport(
#   path="mnist.onnx",
#   size_bytes=1_245_312,
#   num_nodes=8,
#   num_params=128_058,
#   opset_version=18,
#   verify_max_abs_err=5.3e-06,
# )

# Dynamic batch axis
torch.onnx.export(
    model, dummy, "mnist_dyn.onnx",
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    input_names=["input"], output_names=["output"],
)

# Export training graph too
torch.onnx.export(model, dummy, "mnist.onnx", export_training_graph=True)
# downloads: mnist.onnx + mnist.training.onnx
```

### Python — internal

```python
import browsergrad_jit
browsergrad_jit.config.experimental_onnx_export = True   # v1 ships behind flag

# Programmatic access (e.g. for tests):
from browsergrad_jit.onnx import export_bytes
proto_bytes: bytes = export_bytes(model, dummy, verify=False)
```

### TypeScript — internal

```typescript
import { buildModelProto, runInOrtWeb } from "@unlocalhosted/browsergrad-jit/onnx";
```

No user-visible JS API; everything is driven from Python.

---

## Implementation Plan

### Week 1 — Op-table + scaffolding

- [ ] Create `packages/browsergrad-jit/src/python/onnx/` and `src/ts/onnx/`.
- [ ] Vendor `onnx.proto3` from [onnx/onnx@v1.16.x](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3) into `src/ts/onnx/proto/`; pin commit SHA in a sibling `VERSION`.
- [ ] Wire `protobuf-ts` codegen into the build: `pnpm build:proto` regenerates `onnx.generated.ts`.
- [ ] Write `op_table.py` covering all 19 IR opcodes with unit tests on a synthetic 1-node-per-opcode IR.

### Week 2 — Serialisation + download

- [ ] Implement `serialise.py`: walks IR, collects nodes, initializers, value-infos; bridges to JS via Pyodide.
- [ ] Implement `ts/onnx/writer.ts`: `buildModelProto`, `buildTensorProto`, attribute encoding.
- [ ] Implement `ts/onnx/download.ts`: `<a download>` trigger in browser; `Uint8Array` return in Node.
- [ ] Integration test: export `y = x @ W + b` to bytes; parse back with `protobuf-ts` decoder; assert structure (1 MatMul node, 1 Add node, 2 initializers, 1 input, 1 output).

### Week 3 — Round-trip verification

- [ ] Implement `verify.py` + `ts/onnx/ort_bridge.ts` with lazy `import("onnxruntime-web")`.
- [ ] Integration test: export a 2-layer MLP, re-import via ORT-web in the same Pyodide-in-Node test, run forward on the same `dummy`, assert max-abs-err < 1e-4.
- [ ] Cold-start budget test: assert verify=False export adds ≤ 5 ms to baseline cold-start; verify=True adds ≤ 200 ms (lazy ORT-web load).

### Week 4 — Dynamic axes + edge cases

- [ ] Implement `dynamic_axes` kwarg handling.
- [ ] Edge cases: scalar inputs, zero-dim tensors, empty initializer (rare but legal), nested module hierarchies.
- [ ] Integration test: export with `dynamic_axes={"input": {0: "batch"}}`, run ORT-web with B=1 and B=8, both succeed and match.
- [ ] Error path: model containing a non-mappable op raises `OnnxUnmappableOp` with helpful message pointing at the op-table contribution path.

### Week 5 — Training-graph branch + external data

- [ ] Implement `export_training_graph=True`: emit forward + backward + optimizer step as `<name>.training.onnx`; populate `TrainingInfoProto`.
- [ ] Implement external-data path for > 1 GB models (synthetic test with mocked large initializer).
- [ ] Integration test: training graph round-trips through ORT-web with `enableTraining=true` build (or skip with clear "ORT training mode not loaded" if the lab build doesn't include it).

### Week 6 — End-to-end demo + ship

- [ ] Demo notebook in `docs/demos/onnx_export.ipynb`: train MNIST classifier, export to `.onnx`, download.
- [ ] External-engine validation: run the exported `mnist.onnx` through (a) ORT desktop (CPU EP) (b) coremltools convert → mlmodel (c) onnx-mlir compile → executable. Document each in `docs/onnx_export_compat.md`.
- [ ] Conformance suite: 10 representative models (MLP, CNN, RNN, attention, embedding, residual block, normalised conv, softmax-classifier, scaled-dot-product-attention, RMSNorm) export and round-trip within tolerance.
- [ ] Browser matrix test: Chrome, Firefox, Safari, Edge — export call works, download succeeds, verification passes.
- [ ] Publish `@unlocalhosted/browsergrad-jit@0.X.0` with `bg.config.experimental_onnx_export` default `False`.

---

## Acceptance Criteria

| # | Criterion | Measurement |
|---|---|---|
| AC1 | All 19 IR opcodes have a passing `op_table.py` entry (or `LOAD`/`STORE`/`BUFFER`/`CONST` correctly folded) | `op_table_coverage.test.ts` |
| AC2 | `torch.onnx.export` API surface matches PyTorch for `opset_version`, `dynamic_axes`, `input_names`, `output_names`, `do_constant_folding` | `export_api.test.ts` against PyTorch signature snapshot |
| AC3 | Exported `.onnx` for a 2-layer MLP re-imports into ORT-web in the same tab and matches forward within 1e-4 | `roundtrip_mlp.test.ts` |
| AC4 | Exported `.onnx` for a CNN (Conv2d + ReLU + Linear) round-trips within 1e-4 | `roundtrip_cnn.test.ts` |
| AC5 | Exported `.onnx` for a scaled-dot-product-attention block round-trips within 1e-4 | `roundtrip_attention.test.ts` |
| AC6 | Exported `.onnx` for the MNIST demo loads cleanly in ORT desktop (CPU EP, Node CLI) | `external_ort_desktop.spec.ts` |
| AC7 | Exported `.onnx` converts to `.mlmodel` via `coremltools` without errors | offline checklist, captured in `docs/onnx_export_compat.md` |
| AC8 | Exported `.onnx` compiles via `onnx-mlir --EmitObj` without errors | offline checklist |
| AC9 | `dynamic_axes` produces a graph that ORT-web runs at B=1 and B=8 with matching per-batch outputs | `dynamic_axes.test.ts` |
| AC10 | `export_training_graph=True` produces a valid `TrainingInfoProto` block parseable by `onnx.load` | `training_graph.test.ts` (uses `protobuf-ts` decoder on emitted bytes) |
| AC11 | Cold-start delta from PRD-008 baseline ≤ 5 ms when `verify=False` | `coldstart.bench.ts` |
| AC12 | Cold-start delta ≤ 200 ms when `verify=True` (covers lazy ORT-web load) | `coldstart.bench.ts` |
| AC13 | Model with non-mappable op raises `OnnxUnmappableOp` with the op name and contribution-path message | `error_paths.test.ts` |
| AC14 | Opset 18 is the default emission; `opset_version=20` emits an opset-20 import statement | `opset_version.test.ts` |
| AC15 | `<a download>` trigger fires in a browser context; `Uint8Array` returns in Node context | `download_dispatch.spec.ts` (Playwright + Node) |

---

## Test Strategy

### Unit tests (no Pyodide, no ORT-web)

- `op_table_unit.test.ts` — every opcode → `OnnxNode` mapping; synthetic UOp inputs; assert `op_type`, attribute keys, attribute values.
- `writer_unit.test.ts` — `buildModelProto` on hand-built payloads; decode the output bytes back with `protobuf-ts`; structural assertions.
- `proto_pin.test.ts` — vendored `onnx.proto3` SHA matches the pinned version; warn on drift.

### Integration tests (Vitest + real Pyodide-in-Node)

- `roundtrip_*.test.ts` — see AC3–AC5; the primary verification battery.
- `dynamic_axes.test.ts` — see AC9.
- `training_graph.test.ts` — see AC10.
- `error_paths.test.ts` — non-mappable op error message; broken inputs reach raise-time before serialisation.
- `verify_disabled.test.ts` — `verify=False` skips ORT-web load; bytes still produced.

### Browser matrix (Playwright)

- `browser/chrome_export.spec.ts` — Chrome 147: export works, download fires, ORT-web verify passes.
- `browser/firefox_export.spec.ts` — Firefox latest stable: same.
- `browser/safari_export.spec.ts` — Safari 18: same; ORT-web's WASM EP is the verified path.

### External-engine validation (manual checklist, documented)

- ORT desktop CLI on Linux/macOS/Windows.
- `coremltools` convert on macOS.
- `onnx-mlir` compile on Linux.
- TensorRT `trtexec --onnx=mnist.onnx` on NVIDIA hardware (CI-optional; documented as recommended).

### Benchmarks (CI-tracked, not blocking)

- `bench/export_latency.bench.ts` — export latency vs model size; MLP, CNN, mini-transformer.
- `bench/verify_latency.bench.ts` — verification latency (ORT-web cold + warm).
- `bench/roundtrip_drift.bench.ts` — distribution of max-abs-err across 100 random inputs per model.

---

## Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | `protobuf-ts` codegen produces output incompatible with our bundler (esbuild) or with Pyodide's Web Worker context | Medium | Medium | Path B (hand-rolled ~400-LOC writer) in reserve; if either issue surfaces, switch implementations behind the same `writer.ts` interface |
| R2 | onnxruntime-web's WASM bundle balloons cold-start when `verify=True` is the default | Medium | Medium | Lazy `import()`; `verify=True` is documented as opt-in performance cost; provide `verify="async"` mode that runs verify in background and reports via callback |
| R3 | Numerical drift between browsergrad NumPy and ORT-web WASM exceeds 1e-4 for fp16 paths from PRD-010 | Medium | Medium | Tolerance is per-dtype: 1e-4 for fp32, 1e-3 for fp16; documented and tested per-dtype |
| R4 | Vendored `onnx.proto3` drifts from upstream — newer opset uses fields we don't encode | Low | Low | Pin SHA; `proto_pin.test.ts` warns on drift; opset upgrades are a deliberate PR |
| R5 | `coremltools` conversion fails on a corner-case op we emit (e.g. `Pad` with reflect mode on 4-D tensors) | Medium | Low | Conformance table in `docs/onnx_export_compat.md` documents per-engine known issues; users can disable problematic ops via `opset_version` downgrade or model edit |
| R6 | `dynamic_axes` interacts badly with `Reshape` initializers whose shape includes the dynamic dim | Medium | Medium | `Reshape` arg with `-1` for dynamic dims; tested in `dynamic_axes.test.ts`; documented |
| R7 | Training-graph export produces `TrainingInfoProto` that no downstream engine consumes — looks like dead code | Medium | Low | Emit it anyway per spec; document as "archival format"; v1's success metric is round-trip-through-ORT-web only |
| R8 | Large models (> 1 GB) hit `<a download>` size limits in Safari (~512 MB historically) | Low | Medium | External-data path emits a sidecar `.onnx_data` file; both downloads stay under 512 MB for typical educational models |
| R9 | ORT-web verification path is slow on first call (load + compile), confusing students | High | Low | First-call latency is documented; `verify=True` reports `ort_load_ms` separately from `verify_ms` in `ExportReport` |
| R10 | Opset 18 vs 20 op semantics drift for `Reduce*` ops (`axes` moved from attribute to input between 13 → 18) | Medium | Medium | Op-table is opset-aware; `op_table.py` carries opset-version branches for `ReduceSum`/`ReduceMean`/`Pad` where semantics changed |
| R11 | Pyodide's `toJs({create_proxies: false})` on large numpy arrays copies through V8 heap; spike during export | Medium | Low | Stream initializers one at a time; document peak memory ~ 2× model size; mitigate with PRD-008 OPFS staging if needed |
| R12 | User reports a "mysteriously slow" export and we lack telemetry to debug | Medium | Low | `ExportReport` includes per-phase timings (trace, lower, serialise, verify); easy to bisect |

---

## Open Questions

1. **Default `verify` value.** Defaulting to `verify=True` catches bugs early but pays the ORT-web load cost on every export. Defaulting to `verify=False` is fast but lets a broken model slip out unnoticed. Resolution: default `True`, document the ~200 ms cost, provide `verify=False` for hot loops. The educational frame favours correctness-by-default.

2. **Opset 20 default vs 18 default.** Opset 20 has cleaner semantics for several ops but is less widely supported in downstream engines (CoreML lags ~2 opsets behind, TensorRT lags ~1). Resolution: default opset 18 for v1; revisit when CoreML reaches opset 20 (estimated 2027).

3. **External data threshold.** When does inline `raw_data` become `external_data`? PyTorch's threshold is 2 GB (protobuf message limit); we set ours at 1 GB to leave headroom for `Slice`/`Reshape` initializers. Resolution: 1 GB configurable via `external_data_threshold_bytes` kwarg.

4. **Model versioning.** Should the exported `.onnx` carry `producer_name="browsergrad"` and `producer_version=<pkg version>`? Resolution: yes, both. Useful for downstream debugging.

5. **Constant folding.** PyTorch's `do_constant_folding=True` pre-evaluates constant subgraphs at export time, producing smaller models. Our IR's `CONST` and constant-propagation are mostly handled in PRD-006 fusion; we get most of the benefit pre-export. Resolution: honour the kwarg but the implementation is mostly a no-op in v1 (already folded upstream); document as such.

6. **`torch.export.export` future-proofing.** PyTorch is migrating to a new export API; should we shadow it? Resolution: not in v1. Add an open issue to track the migration when the new API stabilises; the underlying IR work is identical so the API change is shallow.

7. **`StringTensorProto` and string inputs.** Some preprocessing graphs accept string tensors. Resolution: out of scope; raise `OnnxUnsupportedDtype` for string tensors; document.

8. **Symbolic shape inference for `dynamic_axes` interaction with `Reshape`.** If a graph has `Reshape` to a shape that includes the dynamic batch axis, the initializer for `Reshape` must use `-1`. Resolution: detected at op-table-time; emit `-1` in that position; tested in AC9.

9. **Lab integration.** Should the craftingattention lab UI surface a "download .onnx" button that calls `torch.onnx.export` from a button handler? Resolution: yes in a follow-up PRD on the lab side; v1 of this PRD is library-only.

---

## References

1. **ONNX intro & concepts** — [onnx.ai/onnx/intro/concepts.html](https://onnx.ai/onnx/intro/concepts.html). The canonical introduction to `ModelProto`/`GraphProto`/`NodeProto`/`TensorProto`; everything in `writer.ts` traces back to this document.

2. **ONNX op catalog (opset 18 normative)** — [github.com/onnx/onnx/blob/main/docs/Operators.md](https://github.com/onnx/onnx/blob/main/docs/Operators.md). Source of every IR-to-ONNX mapping; cross-checked attribute names and signatures.

3. **ONNX Protobuf schema (`onnx.proto3`)** — [github.com/onnx/onnx/blob/main/onnx/onnx.proto3](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3). Vendored into our build, pinned by SHA.

4. **ONNX External Data Format** — [github.com/onnx/onnx/blob/main/docs/ExternalData.md](https://github.com/onnx/onnx/blob/main/docs/ExternalData.md). Specifies the `external_data` reference scheme for > 2 GB models; we adopt the format at our 1 GB threshold.

5. **ONNX Training Information** — [github.com/onnx/onnx/blob/main/docs/IR.md#training-information](https://github.com/onnx/onnx/blob/main/docs/IR.md#training-information). Defines `TrainingInfoProto`; consumed by the `export_training_graph=True` branch.

6. **PyTorch ONNX exporter** — [pytorch.org/docs/stable/onnx.html](https://pytorch.org/docs/stable/onnx.html). The signature we match (`opset_version`, `dynamic_axes`, `input_names`, `output_names`, `do_constant_folding`); our API is a strict subset for v1.

7. **PyTorch `torch.export`** — [pytorch.org/docs/stable/export.html](https://pytorch.org/docs/stable/export.html). The successor API; out of scope for v1 but tracked in Open Question 6.

8. **ONNX Runtime Web** — [onnxruntime.ai/docs/build/web.html](https://onnxruntime.ai/docs/build/web.html). The in-tab round-trip verification engine; lazy-loaded only when `verify=True`.

9. **Protobuf encoding spec** — [protobuf.dev/programming-guides/encoding](https://protobuf.dev/programming-guides/encoding/). Reference for the Path B hand-rolled fallback writer.

10. **protobuf-ts** — [github.com/timostamm/protobuf-ts](https://github.com/timostamm/protobuf-ts). The codegen tool used for Path A; emits TypeScript serialisers from `.proto3` files.

11. **TensorRT ONNX deployment** — [nvidia.com/en-us/deep-learning-ai/products/tensorrt](https://www.nvidia.com/en-us/deep-learning-ai/products/tensorrt/). The NVIDIA production inference path; consumes `.onnx` directly via `trtexec`.

12. **CoreML & coremltools** — [developer.apple.com/documentation/coreml](https://developer.apple.com/documentation/coreml), [github.com/apple/coremltools](https://github.com/apple/coremltools). The iOS/macOS deployment path; converts `.onnx` to `.mlmodel`.

13. **OpenVINO ONNX support** — [docs.openvino.ai/latest/openvino_docs_model_processing_introduction.html](https://docs.openvino.ai/latest/openvino_docs_model_processing_introduction.html). Intel's CPU/iGPU/NPU inference engine; consumes `.onnx`.

14. **onnx-mlir** — [github.com/onnx/onnx-mlir](https://github.com/onnx/onnx-mlir). IBM's `.onnx` → native binary compiler via MLIR; the "browser-trained model becomes a `.so`" path.

15. **ONNX Runtime Web WebNN Execution Provider** — [onnxruntime.ai/docs/execution-providers/WebNN-ExecutionProvider.html](https://onnxruntime.ai/docs/execution-providers/WebNN-ExecutionProvider.html). Same partitioning pattern as PRD-011 in reverse — confirms our IR-to-ONNX mapping table is the inverse of their ONNX-to-WebNN table.

16. **PRD-005 IR opcode table** — establishes the 19 opcodes PRD-016 maps to ONNX; identical surface as PRD-011's WebNN mapping.

17. **PRD-007 symbolic backward** — supplies the backward IR consumed by `export_training_graph=True`; without it the training-graph branch is impossible.

18. **PRD-008 persistent caching (safetensors)** — the analog for *import*; PRD-016 is the export side of the same weight-serialisation story. `TensorProto` in ONNX plays the role safetensors plays for inbound weights.
