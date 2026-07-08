# BrowserGrad Resource Metrics

BrowserGrad reports resource evidence by provenance. A metric is only useful to
platform graphs if callers can tell what was measured, where it came from, and
whether it is exact.

## Confidence Classes

- `exact`: measured from BrowserGrad-owned state or a monotonic runtime clock.
- `estimated`: browser-provided estimate with documented browser limitations.
- `coarse`: completion-boundary timing, not pure execution time.
- `unavailable`: not observable from the current browser execution path.

Hard budgets should use only `exact` metrics. Estimated and coarse metrics are
for warnings, context, and graphs that clearly label their source.

## Runtime Metrics

`session.exec({ resourceMetrics: { enabled: true } })` returns
`ExecResult.resources` and can stream `onResourceSample` events for live graphs.
The final summary includes a `histogramKey` built from caller-provided lab
context, detected browser family when available, runtime packages, backend tier,
and the metric names actually reported. Platforms should store accepted-run
summaries under that key for future percentile graphs.

Current runtime metrics:

- `worker_wall_time_ms`: exact worker execution duration reported by the worker.
- `host_round_trip_ms`: exact host-side request/response duration.
- `host_elapsed_ms`: exact host-side sample placement time.
- `estimated_page_bytes`: estimated browser page memory when
  `performance.measureUserAgentSpecificMemory()` is available; otherwise
  reported as unavailable.

BrowserGrad does not use deprecated `performance.memory` in the default path.

### Adding Whole-Run Timing To A Lab

Use runtime resource metrics for every Pyodide-backed lab, including Python,
NumPy, and PyTorch-shaped BrowserGrad labs:

```ts
const result = await session.exec(studentCode, {
  timeoutMs: 30_000,
  resourceMetrics: {
    enabled: true,
    sampleIntervalMs: 100,
    budgets: {
      wallTimeMs: 10_000,
      estimatedPageBytes: 512 * 1024 * 1024,
    },
    histogram: {
      assignmentId: "linear-regression-lab",
      assignmentVersion: "2026-07-07",
      backendTier: "pyodide",
      runtimePackages: ["numpy", "browsergrad-grad"],
    },
  },
  onResourceSample(sample) {
    renderLiveResourceGraph(sample);
  },
});

renderFinalResourceSummary(result.resources);
```

Platform behavior:

- `timeoutMs` is the watchdog. It stops runaway user code.
- `resourceMetrics.budgets.wallTimeMs` is the grading/graphing budget. It
  evaluates the completed worker wall time.
- `worker_wall_time_ms` is exact wall-clock duration inside the Pyodide worker.
  It includes Python bytecode, NumPy WASM compute, BrowserGrad PyTorch-shaped
  eager/JIT code, package imports performed during the exec, and any user waits.
- `host_round_trip_ms` is exact browser host request/response duration. It
  includes worker scheduling and message transfer overhead, so it is useful for
  platform health graphs, not student algorithm scoring.
- `estimated_page_bytes` is advisory only. It depends on
  `performance.measureUserAgentSpecificMemory()` and may be unavailable.

Do not compare raw times across different histogram keys. A fair percentile
graph needs the same assignment id/version, backend tier, browser family, and
runtime package set.

### Adding Section Timing Inside Pyodide Labs

Whole-run timing is automatic. Section timing is authored by the lab or rubric
because BrowserGrad cannot know which lines are setup, model code, training,
evaluation, or visualization.

Use a monotonic Python timer and emit a structured artifact:

```python
import time
import browsergrad as bg

sections = []

t0 = time.perf_counter()
# setup data
sections.append({"name": "setup", "ms": (time.perf_counter() - t0) * 1000})

t0 = time.perf_counter()
# student NumPy / BrowserGrad torch-shaped code
sections.append({"name": "student_train", "ms": (time.perf_counter() - t0) * 1000})

t0 = time.perf_counter()
# rubric evaluation
sections.append({"name": "rubric_eval", "ms": (time.perf_counter() - t0) * 1000})

bg.emit_json("section_timing", {
    "source": "python-time-perf-counter",
    "confidence": "exact",
    "unit": "ms",
    "sections": sections,
})
```

Platform behavior:

- Store `section_timing` next to `ExecResult.resources`.
- Graph section timings as lab-authored exact wall-clock evidence.
- Keep section timing separate from `ResourceMetricSummary` until BrowserGrad
  has a typed section-metric API.
- Do not use Python section timing as CPU time. It is elapsed wall time inside
  the Pyodide worker.

Recommended section names:

- `setup`
- `student_code`
- `student_train`
- `student_inference`
- `rubric_eval`
- `visualization`
- `package_import`

## WebGPU Metrics

`createWebGpuRealizerBridge(device).resourceSnapshot()` reports exact
BrowserGrad-owned WebGPU buffer accounting:

- `currentOwnedGpuBytes`
- `peakOwnedGpuBytes`
- `totalAllocatedGpuBytes`
- `totalReleasedGpuBytes`
- `aliveHandleCount`
- `logicalTensorPlanPeakBytes` when a resident tensor plan reports one

These values are not total VRAM, total browser GPU memory, or system memory.
They are the bytes BrowserGrad can account for because the bridge created and
released the buffers.

The snapshot also reports whether the device exposes `timestamp-query`.
Bridge-dispatched direct kernels and resident/materialized tensor-plan dispatches
are profiled by default:

- if `timestamp-query` was requested on the `GPUDevice`, completed pass profiles
  report exact `gpuElapsedMs`;
- otherwise completed pass profiles report coarse `queueElapsedMs` from
  `GPUQueue.onSubmittedWorkDone()`;
- `pendingProfileCount` is non-zero while submitted timing readbacks are still
  settling;
- `flushProfiles()` waits for pending timing evidence and returns a fresh
  snapshot.

Do not infer exact GPU elapsed time from `timestampQueryAvailable` alone. Exact
GPU timing requires a completed pass profile with `timingMode:
"timestamp-query"` and `confidence: "exact"`.
When BrowserGrad creates the device, request this with
`createDevice({ requiredFeatures: ["timestamp-query"] })` after checking adapter
support.

### Adding WebGPU Timing To A Kernel Or Tensor Lab

For bridge-backed labs, use the realizer snapshot:

```ts
const bridge = createWebGpuRealizerBridge(device);

const out = bridge.matmul(a, b, m, k, n, "float32");
const beforeFlush = bridge.resourceSnapshot();
const afterFlush = await bridge.flushProfiles();

renderGpuMemoryGraph(afterFlush.currentOwnedGpuBytes, afterFlush.peakOwnedGpuBytes);
renderGpuPassGraph(afterFlush.passProfiles);
```

For direct kernel benchmarks or regression tests, pass a label and await the
returned profile:

```ts
const result = matmulTiledDirect(device, aBuffer, bBuffer, m, k, n, {
  label: "bench:tiled:square-medium:1",
});
const profile = await result.profile;
```

Profile interpretation:

- `confidence: "exact"` and `timingMode: "timestamp-query"`: GPU pass elapsed
  time from WebGPU timestamp queries.
- `confidence: "coarse"` and `timingMode: "queue-completion"`: queue completion
  elapsed time. This includes queue/drain effects and is useful for trends, not
  hard micro-benchmark claims.
- `confidence: "unavailable"`: keep the data point visible as unavailable; do
  not substitute wall time.

BrowserGrad includes a real browser benchmark fixture at
`packages/browsergrad-kernels/tests-browser/resource_metrics_benchmark.test.ts`.
It runs repeated naive-vs-tiled matmul samples across multiple matrix shapes and
checks exact bridge-owned memory accounting.

## Resource-Budget Gates

Assignment profiles may declare a `resource-budget` behavioral gate with these
options:

- `wall_time_ms`
- `browsergrad_owned_gpu_bytes`
- `estimated_page_bytes`
- `wasm_heap_capacity_bytes`
- `webgpu_timestamp_ms`

Exact metrics can hard-fail. Estimated or unavailable metrics should not become
hard failures by default.

Example assignment gate:

```json
{
  "name": "student_resource_limits",
  "kind": "resource-budget",
  "options": {
    "wall_time_ms": 10000,
    "estimated_page_bytes": 536870912
  }
}
```

Routing behavior:

- assignment run planning enables `resourceMetrics`;
- multiple `resource-budget` gates collapse to the tightest value per budget;
- histogram context is filled from assignment id/version, `backendTier:
  "pyodide"`, and session packages;
- only exact completed metrics should fail a hard budget.

Current browser-runtime support:

- `wall_time_ms`: reported and hard-failable through `worker_wall_time_ms`.
- `estimated_page_bytes`: reported when browser memory measurement exists;
  otherwise unavailable; advisory by default.
- `browsergrad_owned_gpu_bytes`: available from WebGPU bridge snapshots, not
  from ordinary `session.exec` yet.
- `wasm_heap_capacity_bytes`: currently unavailable in browser runtime summary.
- `webgpu_timestamp_ms`: available from WebGPU profiles, not from ordinary
  `session.exec` yet.

## Deliberately Unavailable In Browser-First V1

BrowserGrad does not simulate or guess:

- native process RSS
- cgroup memory usage
- total GPU utilization
- total VRAM
- other tabs' GPU work
- driver scheduler time
- hardware counters such as occupancy, cache misses, or bandwidth

External runners may add native metrics later under a separate provenance source.

## Pyodide, NumPy, And PyTorch-Shaped Labs

Pyodide labs run inside a browser Worker. NumPy runs as WASM/native extension
code inside that same worker. BrowserGrad PyTorch-shaped labs run through
BrowserGrad packages installed into the same session. Therefore:

- whole-lab elapsed time: use `worker_wall_time_ms`;
- live graph x-axis: use `host_elapsed_ms` samples;
- platform overhead graph: use `host_round_trip_ms`;
- per-section lab graph: emit `section_timing` artifacts from Python;
- memory graph: use `estimated_page_bytes` when available, labeled estimated;
- hard time budget: use `resource-budget.options.wall_time_ms`;
- runaway stop: use `timeoutMs`;
- exact process RSS, CPU time, and native heap attribution: unavailable in
  browser-first Pyodide.

For PyTorch-shaped BrowserGrad labs, include the teaching runtime package names
in the histogram key, for example:

```ts
resourceMetrics: {
  enabled: true,
  histogram: {
    assignmentId: "mlp-training",
    assignmentVersion: "2026-07-07",
    backendTier: "pyodide",
    runtimePackages: ["numpy", "browsergrad-grad", "browsergrad-jit"],
  },
}
```

If the lab offloads work to WebGPU through BrowserGrad kernels, merge the
runtime summary with the WebGPU bridge snapshot:

- runtime summary answers "how long did the lab execution take?";
- bridge snapshot answers "how much BrowserGrad-owned GPU memory existed?";
- pass profiles answer "how long did BrowserGrad-owned GPU dispatches take?";
- unavailable native metrics remain unavailable.
