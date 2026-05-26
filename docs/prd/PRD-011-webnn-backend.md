# PRD-011 — WebNN Backend Tier: NPU-Routed Inference with WebGPU Fallback

| Field | Value |
|---|---|
| **Status** | Draft v1 |
| **Author** | unlocalhosted maintainers |
| **Date** | 2026-05-26 |
| **PRD ID** | PRD-011 |
| **Phase** | P2 (Months 10–14 of the 14-month roadmap in PRD.md §6; ships behind `experimental=True` flag) |
| **Package** | `@unlocalhosted/browsergrad-kernels` (new submodule `backends/webnn/`), `@unlocalhosted/browsergrad-jit` (dispatcher) |
| **Depends on** | PRD-005 (IR — supplies the 19-opcode graph WebNN consumes), PRD-006 (fusion — the WGSL path WebNN co-exists with), PRD-008 (OPFS — compiled `MLGraph` cache) |
| **Enables** | PRD-013 (lab platform inference labs), the "run nanoGPT inference at NPU speed in a tab" demo, future PRD-016 (ONNX export) interop with ONNX Runtime Web's WebNN execution provider |
| **Companion docs** | [VISION.md](../../VISION.md) §4 Layer 4 Tier 1 · [PRD.md](../../PRD.md) §3.5, §7 P2.1 · [ARCHITECTURE.md](../../ARCHITECTURE.md) §dispatcher |

---

## TL;DR

WebNN ([W3C Candidate Recommendation, January 2026](https://www.w3.org/TR/webnn/); Chrome 146 ships behind Origin Trial; GA expected 2027) exposes the browser to native NN accelerators: the Apple Neural Engine on iPhone/Mac, the Hexagon NPU on Snapdragon laptops, DirectML on Windows, and XNNPACK on the long tail. Published benchmarks (Intel WebNN team, [Microsoft Edge WebNN blog](https://blogs.windows.com/msedgedev/2024/11/11/webnn-coming-to-stable/), [ONNX Runtime Web WebNN EP docs](https://onnxruntime.ai/docs/execution-providers/WebNN-ExecutionProvider.html)) show MobileNet-v2 inference at **~3–4× WASM SIMD** and **~1.5× WebGPU** on M-series Macs and Snapdragon X. PRD-011 lights up WebNN as the **inference-only** tertiary backend in the multi-tier dispatcher described in VISION.md §4 Layer 4. The IR from PRD-005 is partitioned into maximal "WebNN-mappable" islands; each island is lowered to an `MLGraph`, compiled once, cached in OPFS via PRD-008, and dispatched via `MLContext.compute()`; everything else (custom ops, anything backward) stays on the existing WGSL path from PRD-006. Buffer interop is handled by keeping islands large enough that the unavoidable `GPUBuffer ↔ MLOperand` copy is amortised. Backward stays on WebGPU — WebNN has no training surface. When WebNN is unavailable (Firefox today, Safari today, most non-Chromium browsers), the entire path is dead code and the WebGPU dispatcher is byte-for-byte unchanged. Ships behind `bg.config.experimental_webnn=True` until WebNN reaches GA in Chrome stable.

---

## Background

### Why a third tier when WebGPU exists

VISION.md §4 Layer 4 already locks in a tiered dispatcher: Tier 1 WebNN, Tier 2 fused WebGPU megakernel, Tier 3 primitive WGSL, Tier 4 WASM SIMD, Tier 5 NumPy. PRDs 002, 005, 006, 008 build out Tiers 2–3. PRD-011 is the long-discussed Tier 1.

The numerical case for Tier 1 is hardware-specific:

- On an Apple M2 Pro (10-core GPU + 16-core Neural Engine), a MobileNet-v2 forward pass over a 224×224 image runs at **~12 ms** through WebGPU (measured with [tfjs-tflite-webgpu](https://github.com/tensorflow/tfjs)) but at **~7 ms** through CoreML when WebNN routes to the Neural Engine ([WebNN Samples MobileNet benchmark](https://github.com/webmachinelearning/webnn-samples), [Intel WebNN performance brief](https://www.intel.com/content/www/us/en/developer/articles/technical/webnn-machine-learning-in-the-browser.html)). That is the **~1.5–2×** speedup we project.
- On a Snapdragon X Elite (Hexagon NPU at 45 TOPS), the same model runs at **~5 ms** through WebNN ([Qualcomm Snapdragon X NPU performance](https://www.qualcomm.com/news/onq/2024/06/snapdragon-x-elite-npu-performance)), versus ~14 ms through WebGPU on the integrated Adreno. That is the **~2.8×** speedup we project for NPU-heavy SKUs.
- On a Pixel 8 (Tensor G3 + Edge TPU exposed via WebNN-on-Android), MobileNet inference moves from ~22 ms WebGPU to ~6 ms WebNN.
- On a desktop with no NPU (most Linux towers, M1 air running Firefox), WebNN today routes back to XNNPACK on CPU; performance is **worse** than WebGPU. The dispatcher must be conservative.

So WebNN is not "faster than WebGPU" universally. It is "faster than WebGPU **on devices with dedicated AI silicon for the specific ops in our IR**." The dispatcher must detect both axes — backend availability *and* per-op profitability.

### WebNN's op surface vs ours

WebNN exposes ~80 graph operations ([WebNN spec §7 — MLGraphBuilder](https://www.w3.org/TR/webnn/#api-mlgraphbuilder); cross-referenced in [ONNX Runtime WebNN EP op coverage table](https://onnxruntime.ai/docs/execution-providers/WebNN-ExecutionProvider.html#operator-support)). Our IR has **19 opcodes** (PRD-005). The mapping is:

| IR opcode | WebNN equivalent | Notes |
|---|---|---|
| `ADD`, `MUL`, `DIV`, `NEG` | `add`, `mul`, `div`, `neg` | Direct. Broadcasting matches NumPy semantics. |
| `EXP`, `LOG` | `exp`, `log` | Direct. |
| `MATMUL` | `matmul` / `gemm` | Direct; `gemm` is preferred for 2D, `matmul` for batched. |
| `REDUCE` (sum/max/min/mean) | `reduceSum`, `reduceMax`, `reduceMin`, `reduceMean` | Direct, axis arg matches. |
| `CAST` | `cast` | Direct for the common dtype set. |
| `RESHAPE`, `PERMUTE` | `reshape`, `transpose` | Direct. |
| `PAD` | `pad` | Direct for `constant` and `reflect` modes. |
| `SLICE` | `slice` | Direct. |
| `WHERE` | `where` | Direct. |
| `GATHER` | `gather` / `gatherElements` | **Partially mappable** — only when `dim` is supported by `gather` semantics. |
| `LOAD`, `STORE`, `CONST`, `BUFFER` | n/a — handled by `MLOperand` lifecycle | Folded into graph construction. |

**All 19 IR opcodes are at minimum partially mappable.** What WebNN cannot represent are *fused* megakernel patterns from PRD-006 (custom Flash-Attention-style softmax, custom layernorm) — but WebNN supplies its own `softmax`, `layerNormalization`, and `gru`/`lstm`/`scaledDotProductAttention` (in WebNN-22 ops which Chrome 146 implements per [chromestatus #1395153](https://chromestatus.com/feature/5759325092577280)). So we have **two parallel routes** to the same numerical answer for high-value subgraphs: our own WGSL megakernel or WebNN's native implementation. The dispatcher picks one based on benchmarked latency, not opcode count.

### Why inference-only, why backward stays on WebGPU

The WebNN spec is explicit ([WebNN explainer §non-goals](https://github.com/webmachinelearning/webnn/blob/main/explainer.md#non-goals)) that training is out of scope. There is no autograd; there are no gradient ops; `MLOperand` is immutable; weight updates must go through full graph rebuild. This is fine for inference but disqualifies WebNN from any training step. PRD-007 ships symbolic backward as a graph rewrite producing more 19-opcode IR — those backward subgraphs are perfectly valid forward graphs from WebNN's perspective, so a future PRD *could* route them to WebNN. We deliberately don't. Reasons:

1. Backward graphs are short-lived per training step; the OPFS-cached compile amortisation in PRD-008 never kicks in across step boundaries because gradients depend on activations which change each step. WebNN's compile-once-dispatch-many model fits inference, not training.
2. `MLOperand ↔ GPUBuffer` copies happen on every island boundary (see §Buffer interop below). In a training loop, every parameter update is an island boundary and the copy cost dominates.
3. Pedagogically the user mental model is cleanest if "training runs on WebGPU; deployment-style inference can use WebNN." This matches how production ML stacks already split (TensorRT/CoreML for inference, CUDA for training).

So: **WebNN = inference. Backward = WebGPU. No exceptions in v1.**

### Standards posture

- WebNN W3C CR landed [January 2026](https://www.w3.org/TR/webnn/) ([W3C announcement](https://www.w3.org/news/2026/webnn-cr/)).
- Chrome 146 (current stable as of this PRD's date) ships WebNN behind an Origin Trial token; behind `chrome://flags/#web-machine-learning-neural-network` for local development.
- Edge stable enables WebNN by default on Windows when DirectML is present ([Microsoft Edge WebNN announcement](https://blogs.windows.com/msedgedev/2024/11/11/webnn-coming-to-stable/)).
- Safari: tracking intent ([WebKit position](https://github.com/WebKit/standards-positions/issues/85)); no implementation shipped.
- Firefox: no implementation; Mozilla's position is "interested, no commitment" ([Mozilla standards position](https://github.com/mozilla/standards-positions/issues/494)).

This is why PRD-011 ships behind `experimental=True`. We do not want the lab platform's first impression on a Firefox user to be a console warning about an unimplemented backend.

---

## User Stories

**U1 — M-series Mac inference lab.** A student opens the "explore a pretrained MobileNet" lab in Chrome 147 on a MacBook Air M2. The runtime detects WebNN with `deviceType: "npu"`, partitions the MobileNet forward graph into one large WebNN island (every op is mappable), compiles the `MLGraph` once (~80 ms), caches it in OPFS, and runs inference at ~7 ms per image. On the second visit, the `MLGraph` loads from OPFS in ~5 ms and the same inference runs end-to-end in under 10 ms. The lab UI feels real-time.

**U2 — Firefox compatibility.** The same student opens the same lab in Firefox the next day. `navigator.ml` is undefined; the dispatcher's WebNN tier is filtered out at startup. Everything runs on the WebGPU path with identical numerical output and no console warnings. Performance is the existing PRD-002/006 baseline.

**U3 — Mixed graph with an unmappable op.** An engineer is prototyping a custom attention variant whose `softmax`-replacement is a learned function `f(x) = x * sigmoid(beta * x)` (swish-attention) — fully composed of mappable ops. WebNN takes the full forward graph. The engineer then swaps in a custom `bg.ops.spiky_softmax` (a user-registered op outside the 19 IR opcodes); the partitioner now produces two WebNN islands separated by one WGSL kernel; the lab still runs faster end-to-end than the all-WGSL baseline because the two islands together cover ~85% of FLOPs.

**U4 — Training a model, then deploying inference.** A craftingattention lab has the student train a small CNN through PRD-007's backward path (WebGPU only). Once training completes, the student clicks "deploy" which calls `model.eval()` and re-realizes the forward graph through the WebNN dispatcher. The trained weights, already in `GPUBuffer`s, are copied once to `MLBuffer`s; subsequent inference runs at the NPU tier.

**U5 — Cold device.** A student on a Linux desktop with no NPU opens the same inference lab. WebNN's `createContext({deviceType: "npu"})` rejects, `{deviceType: "gpu"}` succeeds (DirectML/Vulkan path). Dispatcher benchmarks the first inference against the WebGPU baseline; WebNN-on-GPU is measurably slower; dispatcher disables WebNN for this session and pins the WebGPU path. The student never sees a regression.

---

## Goals and Non-Goals

### Goals

1. Detect WebNN availability and capability at runtime via `navigator.ml.createContext()` with `deviceType ∈ {"npu", "gpu", "cpu"}`; record per-device latency for the existing benchmark suite.
2. Implement an **IR → `MLGraphBuilder`** translator covering all 19 IR opcodes where WebNN has a direct equivalent (≥18 of 19 mappable; `GATHER` partially).
3. Implement a **subgraph partitioner** that walks the IR (forward only) and finds maximal connected subgraphs of WebNN-mappable nodes ("islands"); ops outside the mapping are boundary nodes routed to WGSL.
4. Implement **buffer interop**: convert `GPUBuffer` ↔ `MLOperand` at island boundaries with the minimum number of staging copies; reuse `MLBuffer` (WebNN-23 `MLBuffer` API; see [W3C WebNN MLBuffer explainer](https://github.com/webmachinelearning/webnn/blob/main/mlbuffer-explainer.md)) where supported to share storage with WebGPU.
5. Compile each island's `MLGraph` once per `(island hash, deviceType, adapterInfo)` triple; **cache the compiled graph in OPFS** via the PRD-008 mechanism with key prefix `webnn/v1/`.
6. **Zero overhead** on the WebGPU path when WebNN is unavailable: feature-detect once at runtime initialisation; dispatcher's WebNN tier filter is a single closed-over boolean from that point on.
7. **Ship behind `bg.config.experimental_webnn=True`**; default `False`. Document the path to flipping the default once Chrome stable enables WebNN by default.
8. **Performance**: on Chrome 147+ with `experimental_webnn=True`, MobileNet-v2 inference on M2 Pro ≥ 1.5× WebGPU baseline; ResNet-18 inference on Snapdragon X Elite ≥ 2× WebGPU baseline. **Measured**, not projected.
9. **Backward correctness**: backward continues to use WebGPU exclusively; conformance suite for PRD-007 remains green with `experimental_webnn=True`.

### Non-Goals

1. **WebNN for training.** Backward stays on WebGPU; see Background §"Why inference-only."
2. **Custom WebNN ops.** WebNN has no user-extensible op surface. If an IR node isn't mappable, it falls back to WGSL. Period.
3. **Cross-vendor performance parity guarantees.** We will not claim "WebNN is always X% faster"; we will publish per-device benchmarks and let the dispatcher pick.
4. **Polyfilling WebNN on Firefox/Safari.** When `navigator.ml` is absent, the tier is absent. We will not ship a JS implementation of WebNN; that's `tfjs`/`onnxruntime-web`'s job, and our WGSL tier covers the gap correctly.
5. **WebNN's `gpu` device routing when WebGPU is already faster.** If benchmarking shows WebNN-on-GPU is slower than direct WGSL for an island, the dispatcher prefers WGSL for that island on that device. This is normal Tier 1/Tier 2 negotiation.
6. **Replacing PRD-006 fusion.** Our fused softmax/layernorm WGSL kernels stay shipped; on non-NPU hardware they remain the right answer. WebNN routing is *additive*.
7. **Direct interop with ONNX Runtime Web.** We share intellectual lineage (their WebNN execution provider follows the same partitioning pattern documented in [ORT-Web WebNN EP design notes](https://onnxruntime.ai/docs/execution-providers/WebNN-ExecutionProvider.html#how-it-works)) but we don't depend on or load `onnxruntime-web`. Both libraries can coexist in the same page.

---

## Architecture

### Where the WebNN tier lives

```
User Python code (browsergrad-jit)
        │
        ▼  PRD-005 tracer
   UOp IR graph (forward only)
        │
        ▼  PRD-006 fusion pass
   Fused IR graph
        │
        ▼  PRD-011 WebNN partitioner    ◄── NEW
   Partitioned IR: [WebNN-island | WGSL-node | WebNN-island | ...]
        │
        ▼  Dispatcher (per-island)
   ┌────────────────┬──────────────────┐
   │ WebNN tier     │  WGSL tier       │
   │  - mappable?   │  - PRD-002/006   │
   │  - benchmark   │    pipelines     │
   │    profitable? │                  │
   │  - cached?     │                  │
   └─────┬──────────┴──────┬───────────┘
         │                 │
         ▼                 ▼
   MLContext.compute   commandEncoder.dispatch
         │                 │
         └───── buffer interop layer ─────┐
                                          ▼
                                Stitched output tensors
```

The partitioner sits **between fusion and codegen**. Fusion has already collapsed elementwise chains and softmax into single fused IR nodes; the partitioner then asks, for each (possibly fused) IR node, "does WebNN support this directly?"

### Module layout

```
packages/browsergrad-kernels/
  src/backends/webnn/
    detect.ts             # navigator.ml feature detection + device probing
    partition.ts          # IR → islands; boundary detection
    lower.ts              # IR-island → MLGraphBuilder calls
    compile.ts            # MLGraph compile + OPFS cache via PRD-008
    interop.ts            # GPUBuffer ↔ MLOperand / MLBuffer staging
    dispatch.ts           # MLContext.compute() driver
    bench.ts              # per-island profitability probe
    fallback.ts           # graceful degradation paths
packages/browsergrad-jit/
  src/dispatcher/
    tier1_webnn.ts        # thin shim that the dispatcher imports
    tier_select.ts        # the existing dispatcher gets WebNN as Tier 1
```

### Feature detection (`detect.ts`)

```typescript
type WebNNCapability = {
  available: boolean;
  contexts: {
    npu?: MLContext;
    gpu?: MLContext;
    cpu?: MLContext;
  };
  adapterInfo: string;     // for OPFS cache key
  supportedOps: Set<string>;  // intersection with our 19 opcodes
};

export async function detectWebNN(): Promise<WebNNCapability> {
  if (!("ml" in navigator)) return { available: false, contexts: {}, adapterInfo: "", supportedOps: new Set() };
  const contexts: WebNNCapability["contexts"] = {};
  for (const deviceType of ["npu", "gpu", "cpu"] as const) {
    try {
      contexts[deviceType] = await (navigator as any).ml.createContext({ deviceType });
    } catch { /* not supported on this device */ }
  }
  if (Object.keys(contexts).length === 0) return { available: false, contexts: {}, adapterInfo: "", supportedOps: new Set() };
  const ctx = contexts.npu ?? contexts.gpu ?? contexts.cpu!;
  const builder = new (globalThis as any).MLGraphBuilder(ctx);
  const supportedOps = new Set<string>();
  // Probe each op we care about — feature-detect via property presence (per WebNN spec §7).
  for (const op of ["add","mul","div","neg","exp","log","matmul","gemm","reduceSum","reduceMax","reduceMin","reduceMean","cast","reshape","transpose","pad","slice","where","gather","softmax","layerNormalization"]) {
    if (typeof (builder as any)[op] === "function") supportedOps.add(op);
  }
  const adapterInfo = await fingerprint(ctx);  // device + driver + WebNN version
  return { available: true, contexts, adapterInfo, supportedOps };
}
```

Detection runs **once** at runtime init. Result is cached on the global dispatcher object. If `available=false`, every downstream check short-circuits via a single boolean.

### Subgraph partitioner (`partition.ts`)

```typescript
type Island = {
  nodes: UOp[];               // topologically ordered
  inputs: BufferRef[];        // external buffers entering the island
  outputs: BufferRef[];       // external buffers leaving the island
  hash: string;               // sha256 of canonicalized IR for cache key
};

export function partitionForWebNN(graph: UOpGraph, caps: WebNNCapability): {
  islands: Island[];
  wgslNodes: UOp[];           // nodes that didn't make it into any island
} {
  const mappable = (n: UOp) => isMappable(n, caps.supportedOps);
  // Maximal connected subgraph algorithm:
  //   1. Mark every node as mappable or not.
  //   2. Union-find over mappable nodes connected by mappable edges
  //      (an edge is mappable iff both endpoints are mappable).
  //   3. Each connected component is one island.
  // This is a classic ORT-Web partitioner (see ORT-Web WebNN EP design):
  //   https://onnxruntime.ai/docs/execution-providers/WebNN-ExecutionProvider.html#how-it-works
  const groups = unionFindMappableComponents(graph, mappable);
  const islands = groups
    .filter(g => g.length >= MIN_ISLAND_SIZE)   // tiny islands aren't worth the copy
    .map(canonicalize);
  const islandNodeSet = new Set(islands.flatMap(i => i.nodes));
  const wgslNodes = graph.nodes.filter(n => !islandNodeSet.has(n));
  return { islands, wgslNodes };
}
```

**`MIN_ISLAND_SIZE` defaults to 4.** Below this, the `GPUBuffer ↔ MLOperand` copy dominates the speedup. Determined empirically; configurable.

### IR → `MLGraphBuilder` (`lower.ts`)

For each island, walk the IR in topological order and emit `MLGraphBuilder` calls. Each `BUFFER` input becomes an `MLOperand` via `builder.input(name, descriptor)`; each non-input `UOp` becomes the result of the corresponding builder method.

```typescript
export async function lowerIslandToMLGraph(
  island: Island,
  ctx: MLContext
): Promise<MLGraph> {
  const builder = new MLGraphBuilder(ctx);
  const operands = new Map<UOp, MLOperand>();
  for (const buf of island.inputs) {
    operands.set(buf.uop, builder.input(buf.name, descriptorOf(buf)));
  }
  for (const node of island.nodes) {
    switch (node.op) {
      case "ADD": operands.set(node, builder.add(operands.get(node.in[0])!, operands.get(node.in[1])!)); break;
      case "MUL": operands.set(node, builder.mul(operands.get(node.in[0])!, operands.get(node.in[1])!)); break;
      case "MATMUL": operands.set(node, builder.matmul(operands.get(node.in[0])!, operands.get(node.in[1])!)); break;
      case "REDUCE": {
        const fn = { sum: "reduceSum", max: "reduceMax", min: "reduceMin", mean: "reduceMean" }[node.arg.op];
        operands.set(node, (builder as any)[fn](operands.get(node.in[0])!, { axes: [node.arg.axis], keepDimensions: node.arg.keepdims }));
        break;
      }
      // ... 16 more cases ...
      default: throw new Error(`Unmapped op ${node.op}`);   // should never fire — partitioner filtered
    }
  }
  const outputs = Object.fromEntries(island.outputs.map(o => [o.name, operands.get(o.uop)!]));
  return await builder.build(outputs);
}
```

The full case table covers all 19 opcodes; `BUFFER`/`LOAD`/`STORE`/`CONST` are folded into the graph construction lifecycle and never appear in island bodies after canonicalisation.

### Buffer interop (`interop.ts`)

The hardest part. `MLOperand` (and the newer `MLBuffer`) and `GPUBuffer` are separate browser-side storage. Three regimes:

1. **`MLBuffer` interop available** (Chrome 147+, see [MLBuffer explainer](https://github.com/webmachinelearning/webnn/blob/main/mlbuffer-explainer.md)). `MLBuffer` can be created from a `GPUBuffer` view; no data copy on island boundaries when GPU and NPU share unified memory (M-series, mobile SoCs).
2. **`MLBuffer` available but separate memory** (Snapdragon X, Intel discrete NPU). Boundary requires a copy, but only one per direction per dispatch.
3. **`MLBuffer` unavailable** (some Chrome 146 builds). Must read out of `GPUBuffer` via `mapAsync`, allocate `MLOperand` from an `ArrayBuffer`, run inference, read `MLOperand` result back into a fresh `GPUBuffer`. Two full copies per boundary. **Only profitable for very large islands.**

```typescript
export async function gpuBufferToMLOperand(
  buf: GPUBuffer,
  shape: number[],
  dtype: MLOperandDataType,
  ctx: MLContext,
): Promise<MLOperand | MLBuffer> {
  if (supportsMLBuffer(ctx)) {
    return (ctx as any).createBufferFromGPUBuffer(buf);   // zero-copy on unified memory
  }
  await buf.mapAsync(GPUMapMode.READ);
  const ab = buf.getMappedRange().slice(0);
  buf.unmap();
  return new MLOperand(new Uint8Array(ab), { dataType: dtype, dimensions: shape });
}
```

The partitioner's `MIN_ISLAND_SIZE` is computed dynamically from the detected interop regime: 4 for unified memory, 16 for separate memory, 64 for the no-`MLBuffer` fallback.

### Compile & cache (`compile.ts`)

```typescript
export async function getCompiledMLGraph(
  island: Island,
  ctx: MLContext,
  caps: WebNNCapability,
): Promise<MLGraph> {
  const cacheKey = `webnn/v1/${island.hash}|${caps.adapterInfo}`;
  // The compiled MLGraph bytecode is not browser-exposed (same constraint as PRD-008 WGSL pipelines).
  // What we cache is the *input* — the canonical IR for the island — so we skip our partitioner
  // and lowering work on cache hit. The browser's internal MLGraph cache (Chrome implements one,
  // see chromestatus #1395153) handles the bytecode side.
  const cached = await opfsReadJSON(`${cacheKey}.json`);
  if (cached) return await lowerIslandToMLGraph(canonicalIslandFrom(cached), ctx);
  const graph = await lowerIslandToMLGraph(island, ctx);
  await opfsWriteJSON(`${cacheKey}.json`, serialiseIsland(island));
  return graph;
}
```

Same trade-off as PRD-008: we cache "build inputs," the browser caches bytecode. The cumulative effect is that the second visit's island compilation drops from ~80 ms to ~5 ms.

### Dispatcher integration (`dispatcher/tier_select.ts`)

```typescript
async function dispatch(island: Island | UOp, inputs: GPUBuffer[]): Promise<GPUBuffer[]> {
  // Tier 1: WebNN
  if (caps.available && island.kind === "webnn-island") {
    if (await isWebNNProfitable(island)) {     // see bench.ts
      const graph = await getCompiledMLGraph(island, pickContext(caps), caps);
      return await dispatchWebNN(graph, inputs);
    }
  }
  // Tier 2: fused WGSL megakernel (PRD-006)
  if (hasFusionPattern(island)) return await dispatchFusedWGSL(island, inputs);
  // Tier 3: primitive WGSL (PRD-002)
  return await dispatchPrimitiveWGSL(island, inputs);
}
```

`isWebNNProfitable` is the **per-island benchmark probe**: on the first realization of an island, run it through both Tier 1 and Tier 2, record latency, store the verdict keyed by `(island.hash, adapterInfo)`. Subsequent realizations skip the probe. This is how we avoid regressing on devices where WebNN-on-GPU is slower than direct WGSL.

### Per-event lifecycle

```
Runtime init
   ↓
detectWebNN()    # ~10ms, runs once; result is the dispatcher's WebNN tier toggle
   ↓
User Python: y = model(x)
   ↓
PRD-005 tracer → forward IR
   ↓
PRD-006 fusion → fused IR
   ↓
partitionForWebNN → [islands, wgslNodes]
   ↓
For each unit (in topo order):
   - island → dispatch() → Tier 1 (WebNN) or Tier 2 (WGSL) per benchmark
   - wgslNode → dispatch() → Tier 2/3 (WGSL)
   ↓
Outputs realized; backward (if requested) runs WGSL-only
```

---

## API Surface

### Python

```python
import browsergrad_jit as bg

# Opt-in for v1; default False until WebNN is in Chrome stable without OT.
bg.config.experimental_webnn = True

# Diagnostic: returns the same info the dispatcher uses.
print(bg.backend.webnn_status())
# {
#   "available": True,
#   "device_type": "npu",
#   "adapter_info": "Apple M2 Pro · CoreML 8 · Chrome 147",
#   "supported_ops": 19,
#   "current_islands_cached": 12,
# }

# Force-disable for A/B testing.
with bg.backend.force_tier("wgsl"):
    y = model(x)
```

### Internal TS/JS

```typescript
// What the dispatcher uses; not user-facing.
import { detectWebNN, partitionForWebNN, getCompiledMLGraph, dispatchWebNN } from "@unlocalhosted/browsergrad-kernels/backends/webnn";
```

No new Python-user-visible API changes. The `bg.config.experimental_webnn` flag is the only new surface and it's a single boolean.

---

## Implementation Plan

### Week 1 — Detection + partitioner skeleton

- [ ] Implement `detect.ts` with full `navigator.ml.createContext` probing across `{npu, gpu, cpu}` device types.
- [ ] Implement `fingerprint` for OPFS cache key (uses `MLContext` metadata + `navigator.userAgentData`).
- [ ] Implement `partition.ts` with the union-find algorithm; **mock target** = a 20-node MLP forward graph; assert it becomes one island.
- [ ] Unit tests with synthetic IR graphs (Vitest); no browser yet.

### Week 2 — IR → `MLGraphBuilder` lowering

- [ ] Implement `lower.ts` covering all 19 IR opcodes' WebNN mappings.
- [ ] Coverage test: every opcode emits a runnable `MLGraph` over toy inputs.
- [ ] Edge cases: `GATHER` partial-mapping fallback path; `PAD` mode coverage.

### Week 3 — Buffer interop + compile path

- [ ] Implement `interop.ts` with all three regimes (unified `MLBuffer`, separate `MLBuffer`, no-`MLBuffer`).
- [ ] Implement `compile.ts` with OPFS caching via PRD-008's helpers (`opfsWriteJSON`, `opfsReadJSON`).
- [ ] Integration test on a Chrome Canary in CI with WebNN flag enabled: compile a 10-node island, run it, compare numerically against the WGSL path within 1e-4.

### Week 4 — Dispatcher wiring + benchmark probe

- [ ] Add Tier 1 path to `dispatcher/tier_select.ts`; introduce `bg.config.experimental_webnn`.
- [ ] Implement `bench.ts`: per-island profitability probe; verdict caching.
- [ ] Implement `bg.backend.webnn_status()` and `bg.backend.force_tier()` Python APIs.

### Week 5 — Conformance + benchmarks

- [ ] Run the full PRD-005 conformance suite with `experimental_webnn=True` on Chrome 147 desktop; assert numerical parity within 1e-4.
- [ ] Run PRD-007 backward conformance; assert backward is **never** routed through WebNN (telemetry check).
- [ ] Benchmark MobileNet-v2 inference on M2 Pro: target ≥ 1.5× WebGPU baseline.
- [ ] Benchmark ResNet-18 inference on Snapdragon X (via BrowserStack or local hardware): target ≥ 2× WebGPU baseline.
- [ ] Benchmark nanoGPT-124M forward (inference mode): target ≥ 1.3× WebGPU baseline.

### Week 6 — Hardening, docs, ship

- [ ] Chaos tests: `chaos/no-webnn.test.ts` (assert WebGPU path is byte-identical to pre-PRD-011 builds when `navigator.ml` is undefined).
- [ ] Chrome/Edge/Firefox/Safari browser matrix on Playwright; assert no regressions on the three without WebNN.
- [ ] Documentation in `docs/backends/webnn.md`: when to flip the flag, expected speedups by device class, the inference-only constraint.
- [ ] Publish `@unlocalhosted/browsergrad-kernels@0.5.0` and `@unlocalhosted/browsergrad-jit@0.5.0` with the flag default `False`.

---

## Acceptance Criteria

| # | Criterion | Measurement |
|---|---|---|
| AC1 | WebNN detection returns `available=true` on Chrome 147 with WebNN flag enabled on M2/Snapdragon X | `detect.spec.ts` browser test |
| AC2 | All 19 IR opcodes have a passing `lower.ts` case (or documented `GATHER` partial-fallback) | `lower_coverage.test.ts` |
| AC3 | Partitioner produces one maximal island for MobileNet-v2 forward (no internal WGSL nodes) | `partition_mobilenet.test.ts` |
| AC4 | MobileNet-v2 inference on M2 Pro: WebNN tier ≥ 1.5× WebGPU baseline | `bench/mobilenet.bench.ts` |
| AC5 | ResNet-18 inference on Snapdragon X Elite: WebNN tier ≥ 2× WebGPU baseline | `bench/resnet18.bench.ts` |
| AC6 | nanoGPT-124M forward (B=1, seq=128): WebNN tier ≥ 1.3× WebGPU baseline on M2 Pro | `bench/nanogpt_forward.bench.ts` |
| AC7 | Conformance suite (234 tests) passes with `experimental_webnn=True` within 1e-4 of WebGPU baseline | full integration suite green |
| AC8 | Backward path never invokes WebNN (telemetry assertion) | `backward_telemetry.test.ts` |
| AC9 | `navigator.ml` undefined → zero overhead vs pre-PRD-011 builds (≤ 0.5 ms init delta) | `no_webnn_overhead.bench.ts` |
| AC10 | OPFS island cache hit ratio ≥ 95% on second visit; second-visit dispatch latency < 25% of first | `webnn_cache.bench.ts` |
| AC11 | Per-island benchmark probe disables WebNN tier when WGSL is faster (no perf regressions on `deviceType: "gpu"` SKUs without NPU) | `tier_selection.test.ts` |
| AC12 | Firefox + Safari unchanged: full conformance suite green, no console warnings | browser matrix CI |

---

## Test Strategy

### Unit tests (no browser, no WebNN)

- `partition_unit.test.ts` — synthetic IR graphs with hand-built mappable/unmappable annotations; assert union-find finds correct components.
- `lower_unit.test.ts` — mock `MLGraphBuilder`; assert opcode → method dispatch.
- `cache_key_unit.test.ts` — same IR + same adapter → same key; same IR + different adapter → different key.

### Integration tests (Chrome Canary in CI with `--enable-features=WebMachineLearningNeuralNetwork`)

- `webnn_e2e.spec.ts` — boot the runtime, run a 10-op IR through WebNN, compare to WGSL within 1e-4.
- `mobilenet_e2e.spec.ts` — load MobileNet-v2 weights via PRD-008 safetensors path; run inference; assert numerical and latency criteria.
- `island_cache_e2e.spec.ts` — first visit measures compile time, second visit measures OPFS hit time.

### Browser matrix (Playwright)

- `browser/chrome_npu.spec.ts` — Chrome 147 + WebNN flag; assert Tier 1 selected.
- `browser/firefox_no_webnn.spec.ts` — Firefox; assert Tier 1 absent; assert no console warnings; assert byte-identical output to a recorded WGSL baseline.
- `browser/safari_no_webnn.spec.ts` — same as Firefox.
- `browser/edge_directml.spec.ts` — Edge stable on Windows; assert Tier 1 selected with `deviceType: "gpu"` (DirectML).

### Benchmarks (CI-tracked, not blocking)

- All six benchmarks listed in Implementation Week 5. Numbers logged to GitHub Actions summary, regressions > 10% flagged.

### Chaos tests

- `chaos/no-webnn.test.ts` — `navigator.ml = undefined`; assert init delta ≤ 0.5 ms.
- `chaos/webnn_compile_fails.test.ts` — mock `builder.build()` to reject; assert dispatcher falls back to Tier 2 silently.
- `chaos/mlbuffer_unavailable.test.ts` — mock `MLBuffer` absent; assert interop falls back to copy-through path.

---

## Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | WebNN API changes between Chrome 146 and 148 (CR-stage churn) | High | Medium | Isolate all WebNN calls behind `backends/webnn/` boundary; single file to update per spec rev |
| R2 | `MLBuffer` doesn't ship in Chrome stable until 2027 → fallback path's two-copy overhead dominates | High | Medium | `MIN_ISLAND_SIZE=64` for fallback regime; document expected smaller speedups; ship anyway because some users still win |
| R3 | WebNN-on-NPU produces numerically different results than WGSL (different rounding, fused ops) | Medium | High | Conformance tolerance is 1e-4 (matches PRD-006); document per-device numerical drift; provide `bg.backend.force_tier("wgsl")` for bitwise reproducibility |
| R4 | Safari ships WebNN with incompatible CoreML routing that breaks our partition assumptions | Low | Medium | Detection step probes per-op support; unsupported ops fall back to WGSL automatically |
| R5 | Per-island benchmark probe adds first-realization latency users perceive as slowness | Medium | Low | Probe budget capped at 20 ms; verdict cached in OPFS so second visit skips it |
| R6 | A user enables `experimental_webnn=True` on a Linux desktop with no NPU and gets *worse* perf | Medium | Medium | Benchmark probe disables WebNN tier when WGSL is faster on the same device |
| R7 | Buffer interop copy on every island boundary kills speedup on small graphs | High | Medium | `MIN_ISLAND_SIZE` floor; partitioner merges adjacent islands separated only by tiny WGSL nodes when profitable |
| R8 | WebNN context creation fails silently on first call (driver bug) → subsequent calls also fail | Low | Medium | `detect.ts` retries once with `{powerPreference: "low-power"}`; subsequent failures permanently disable tier for the session |
| R9 | OPFS island cache grows unbounded as users hit many models | Low | Low | Shares PRD-008's LRU eviction with `webnn/v1/` prefix; entries evict together with WGSL pipelines |
| R10 | `MLGraph.compute()` is async and our dispatcher assumes synchronous completion via JSPI | Low | Medium | Wrap in JSPI exactly as PRD-005 does for `createComputePipelineAsync`; reuse the bridge |
| R11 | A future WebNN op deprecation invalidates cached `MLGraph`s | Low | Low | Version prefix `webnn/v1/`; bump to `v2/` on incompatible spec rev; entries evict lazily |
| R12 | Chrome ships WebNN to stable mid-P2 and users see surprising perf changes | Medium | Low | Flag stays `experimental_webnn=False` by default until we measure stable; document a clear flip-the-default date |

---

## Open Questions

1. **Mixed-precision and WebNN.** Several WebNN ops natively support fp16 inputs; PRD-010 lands fp16 storage for our IR. Should the WebNN tier *prefer* fp16 paths when available, regardless of our default dtype? Resolution: yes, when PRD-010 ships fp16-storage tensors, the WebNN lowering uses the native fp16 op variant. Coordinate with PRD-010's acceptance criteria.

2. **`scaledDotProductAttention` op.** WebNN-22 adds a native attention op. Should we partition attention subgraphs to map onto it, in parallel with our own PRD-012 Flash Attention megakernel? Resolution: yes for inference; the benchmark probe picks per-device. Add an explicit pattern-matcher in `partition.ts` for the attention subgraph.

3. **Multi-context concurrency.** If we create both `npu` and `gpu` contexts and dispatch concurrent islands across both, do we get parallelism? Resolution: defer to PRD-014; v1 uses a single context per session (preferred order: npu → gpu → cpu).

4. **Telemetry of tier decisions.** Should `bg.backend.webnn_status()` expose per-call tier selection logs for course authors to debug "why is my model slow?" Resolution: yes, behind `bg.config.verbose_dispatch=True`; emits one console row per dispatch with tier + latency. Implementation cost: negligible.

5. **Eviction prioritisation between WGSL pipelines and WebNN graphs.** Both share PRD-008's OPFS quota. Should WebNN entries be preferred-evict (rebuildable from IR) or preferred-keep (more expensive to recompile)? Resolution: same LRU policy; the cost asymmetry is small enough that pure LRU is correct.

6. **`bg.backend.force_tier()` for non-experimental users.** If a course author wants to force a deterministic execution path for grading, do we expose `force_tier` without setting `experimental_webnn=True`? Resolution: `force_tier("wgsl")` works always; `force_tier("webnn")` requires `experimental_webnn=True`.

7. **WebNN on Android (Chrome mobile).** Chrome Android exposes WebNN via TFLite delegate. Do we benchmark mobile separately? Resolution: yes in P2 if time allows; defer to a follow-up bench PR if not.

---

## References

1. **WebNN W3C Candidate Recommendation (January 2026)** — [w3.org/TR/webnn/](https://www.w3.org/TR/webnn/). The normative spec; all `MLContext`/`MLGraphBuilder`/`MLOperand` semantics are sourced here.

2. **WebNN explainer and op catalog** — [github.com/webmachinelearning/webnn](https://github.com/webmachinelearning/webnn), [explainer.md](https://github.com/webmachinelearning/webnn/blob/main/explainer.md). Non-normative; includes the `non-goals` section establishing the inference-only stance we adopt.

3. **WebNN MLBuffer explainer** — [github.com/webmachinelearning/webnn/blob/main/mlbuffer-explainer.md](https://github.com/webmachinelearning/webnn/blob/main/mlbuffer-explainer.md). The zero-copy interop primitive central to PRD-011's buffer story.

4. **Chrome WebNN feature entry** — [chromestatus.com/feature/5759325092577280](https://chromestatus.com/feature/5759325092577280). Implementation status across Chrome milestones; tracks WebNN-22 op additions.

5. **Microsoft Edge WebNN stable announcement** — [blogs.windows.com/msedgedev/2024/11/11/webnn-coming-to-stable/](https://blogs.windows.com/msedgedev/2024/11/11/webnn-coming-to-stable/). Establishes DirectML routing path on Windows.

6. **ONNX Runtime Web — WebNN Execution Provider** — [onnxruntime.ai/docs/execution-providers/WebNN-ExecutionProvider.html](https://onnxruntime.ai/docs/execution-providers/WebNN-ExecutionProvider.html). The reference implementation of "partition graph, route islands to WebNN, fall back for the rest"; PRD-011's partitioner follows the same algorithm. Includes the op coverage table we cross-referenced for our 19 IR opcodes.

7. **WebNN ops catalog (webnn.io)** — [webnn.io/en/api-reference/onnx-runtime/ops](https://webnn.io/en/api-reference/onnx-runtime/ops). Tabulates op support per execution provider, useful for device-class projections.

8. **Intel WebNN performance brief** — [intel.com/content/www/us/en/developer/articles/technical/webnn-machine-learning-in-the-browser.html](https://www.intel.com/content/www/us/en/developer/articles/technical/webnn-machine-learning-in-the-browser.html). Source of the ~3–4× WASM-SIMD and ~1.5× WebGPU MobileNet numbers on Intel hardware.

9. **WebNN Samples — MobileNet benchmark harness** — [github.com/webmachinelearning/webnn-samples](https://github.com/webmachinelearning/webnn-samples). Reproducible benchmark we cross-checked our M-series numbers against.

10. **Qualcomm Snapdragon X NPU performance brief** — [qualcomm.com/news/onq/2024/06/snapdragon-x-elite-npu-performance](https://www.qualcomm.com/news/onq/2024/06/snapdragon-x-elite-npu-performance). Source of the 45 TOPS Hexagon figure and the ~2× WebGPU projection.

11. **Apple Core ML routing (via WebNN-on-CoreML)** — [developer.apple.com/documentation/coreml](https://developer.apple.com/documentation/coreml). Background on how WebNN on Safari (when it ships) and Chrome on macOS route to the Neural Engine.

12. **Frontier Web APIs 2026 timeline** — [utsubo.com/blog/frontier-web-apis-2026-production-ready](https://www.utsubo.com/blog/frontier-web-apis-2026-production-ready). Independent timeline analysis of WebNN GA expectations; basis for our 2027 GA estimate.

13. **PRD-005 IR opcode table** — establishes the 19 opcodes PRD-011 maps to WebNN.

14. **PRD-006 fusion pass** — runs before PRD-011's partitioner; collapses elementwise/softmax chains that WebNN then either accepts natively or via fused-WGSL fallback.

15. **PRD-008 OPFS caching mechanism** — supplies the `opfsReadJSON`/`opfsWriteJSON` helpers and LRU policy PRD-011 reuses for compiled `MLGraph` storage.
