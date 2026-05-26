# PRD-008 — Persistent Caching: OPFS Pipeline Cache + Safetensors Streaming

| Field | Value |
|---|---|
| **Status** | Draft v1 |
| **Author** | unlocalhosted maintainers |
| **Date** | 2026-05-26 |
| **PRD ID** | PRD-008 |
| **Phase** | P1 (Months 4–9 of the 14-month roadmap in PRD.md §6) |
| **Package** | `@unlocalhosted/browsergrad-kernels`, `@unlocalhosted/browsergrad-runtime` |
| **Depends on** | PRD-002 (WebGPU kernels), PRD-005 (IR with stable hashing), PRD-006 (fusion produces named pipelines) |
| **Enables** | PRD-013 (lab platform integration with seamless re-runs), the entire LLM-on-browser story |

---

## TL;DR

Every reload of a course exercise today re-compiles every WebGPU pipeline (~50ms each, ~3-5 seconds total for a transformer block) and re-downloads any model weights (often hundreds of MB). This PRD persists both into **OPFS (Origin Private File System)**: compiled WGSL pipelines are keyed by their IR hash + browser fingerprint and cached as the binary representation usable by `device.createComputePipelineAsync`'s cache hint; model weights are stored as the open **safetensors** format and *streamed* into GPU buffers tensor-by-tensor without ever materializing the full file in CPU memory. After this PRD, the second visit to a lab page boots in under a second; a 1B-parameter model that took 30 seconds to load on first visit takes ~2 seconds on the second.

---

## Background

### The cold-start economics today

VISION.md §3 measured the cold path:
- Pyodide boot + browsergrad imports: ~1.2s (covered by PRD-003 service worker).
- First WebGPU pipeline compile per kernel: ~50ms (driver-dependent; see [Chrome WebGPU pipeline caching blog post](https://developer.chrome.com/blog/passing-data-between-cpu-and-gpu)).
- A transformer block has roughly 11 kernel shapes (q/k/v projections, attention scores, softmax, output projection, ffn up, ffn down, residuals, layernorms). Multiple block instances *do* share pipelines if shape signatures are identical.
- A small toy transformer needs ~20-30 distinct pipelines. ~1.5 seconds of recompilation per page reload, repeated.
- Model weights: a 125M parameter model (small GPT-2) is ~500MB in float32, ~250MB in float16, ~125MB in int8. Even ~125MB takes seconds over typical home Wi-Fi.

The browser already has the storage technology to fix both. OPFS (per-origin private filesystem, accessed via the `navigator.storage` API) lets us write arbitrary binary blobs with persistent durability and no eviction (unless the user clears site data). It is supported in all evergreen browsers; see [Chrome OPFS introduction](https://developer.chrome.com/articles/origin-private-file-system).

### What we cache and how

There are two distinct artifact families:

1. **Compiled WebGPU pipelines** — these are the JIT'd, GPU-specific bytecode produced by `device.createComputePipelineAsync` from a `GPUShaderModuleDescriptor` containing WGSL source. We cannot persist the *compiled bytecode* directly because the browser does not expose it. What we *can* cache is:
   - The **WGSL source string** keyed by the fusion-emitted kernel hash.
   - The **pipeline layout descriptor** (binding layouts, entry point name, workgroup size).
   - A **shape registry** noting which (input shape signature) values have already been compiled.
   
   On a cache hit, we read the WGSL+descriptor from OPFS, hand them to `createComputePipelineAsync`, and *let the browser's internal pipeline cache* do the actual bytecode caching. Chrome and Firefox both implement internal pipeline caches keyed by source hash that survive across page loads ([Chrome WebGPU recipes - pipeline caching](https://developer.chrome.com/docs/web-platform/webgpu); [WebGPU pipeline cache discussion in WGPU](https://github.com/gfx-rs/wgpu/discussions/4097)). So OPFS holds the "build inputs," the browser holds the bytecode — but reading our entry from OPFS avoids re-running our fusion + codegen pass.

2. **Model weights in safetensors format** — [safetensors](https://github.com/huggingface/safetensors) is HuggingFace's open binary format: a JSON header describing tensor names, dtypes, shapes, and byte offsets, followed by a contiguous blob of raw tensor data. The format is designed for memory-mapped loading and supports out-of-order partial reads. The reference implementation is in Rust; a pure-JS reader is roughly 100 lines.

Why safetensors specifically:
- **Stream-able** — we can read the header (typically <1 MB), allocate one GPU buffer per tensor, then `fetch` byte ranges into each buffer without ever holding the full model in CPU memory.
- **Self-describing** — no pickled Python state, no eval, no security issues like the classic PyTorch `.pt`/`.bin` formats.
- **Ecosystem** — HuggingFace Hub serves safetensors for the majority of modern models. We can fetch directly from HF or from any CDN.

### Browser storage limits

OPFS storage is governed by the [Storage API](https://developer.mozilla.org/en-US/docs/Web/API/Storage_API) and the browser's storage quota. On modern Chrome/Edge/Firefox the default quota for a site is roughly 60% of available disk space — many gigabytes for typical desktop users; under 1 GB on memory-constrained mobile. We must:
- Track on-disk usage and proactively evict (LRU) before approaching the quota.
- Request "persistent" storage via `navigator.storage.persist()` so the browser does not auto-evict the cache under disk pressure.
- Allow the user (course author) to clear the cache explicitly.

---

## User Stories

**U1 — Returning student.** A student visits a lab page, runs through the exercise, closes the tab. They return the next day. The runtime initializes in under 2 seconds total (Pyodide from service worker cache via PRD-003, kernels from OPFS pipeline cache, weights from OPFS tensor cache). The student notices no waiting.

**U2 — First-time visitor, second exercise.** A new student finishes exercise 1 (which builds a small CNN), then opens exercise 2 (which uses a different CNN with overlapping kernel shapes). Exercise 2's kernels that share signatures with exercise 1 are already compiled — fewer cold pipelines, faster start.

**U3 — Large model assignment.** An assignment downloads a 250 MB safetensors file (small GPT-2). The first visit shows a per-tensor progress bar that streams. The second visit completes in ~2 seconds because all tensors are already in OPFS.

**U4 — Course author cache control.** A course author hits "Clear cache" in their dashboard. All OPFS data for the origin is wiped, allowing them to test the cold-start experience.

**U5 — Quota pressure.** A student has used the lab for 3 months and accumulated 800 MB of cached weights. When they start a 200 MB new exercise on a device near disk full, the cache LRU-evicts the oldest 200 MB silently. The new exercise loads; old exercises that re-visit trigger re-download, but no error surfaces.

---

## Goals and Non-Goals

### Goals

1. Cache compiled WebGPU pipelines (WGSL source + binding layout descriptor) keyed by `hash(WGSL source) + browser/GPU fingerprint`, with persistent storage via OPFS.
2. Implement a safetensors reader that streams tensor data from a URL or OPFS handle directly into `GPUBuffer` objects without intermediate CPU copies of the full file.
3. Provide a `load_model("hf://repo/path/model.safetensors")` Python API that orchestrates download + caching + GPU upload.
4. Track OPFS usage; implement LRU eviction when within 80% of quota.
5. Request persistent storage on first cache write.
6. Provide a course-author/admin API to inspect and clear the cache.
7. Cold-start time after the first visit ≤ 2× the JS overhead alone — measured as `tFirstKernelRun - tPageLoad` < 1.5 seconds for a transformer block, < 5 seconds for the 250 MB GPT-2.

### Non-Goals

1. Caching raw NumPy / Pyodide arrays. The IR layer in PRD-005 reconstructs the graph cheaply from Python; only the GPU-side artifacts (pipelines + weights) are expensive.
2. Cross-origin sharing of caches (this is impossible by browser security model — OPFS is per-origin).
3. Cache *compression* beyond what safetensors itself supports. Lossless compression of float16 weights yields ~1.05× compression ratios and adds CPU cost at load time; not worth it.
4. Cloud sync or backup of the OPFS cache. Out of scope.
5. CDN/edge-side caching of safetensors URLs (orthogonal — the browser HTTP cache already handles this naturally).
6. Real-time training-checkpoint persistence. The cache is for *read-mostly* artifacts (pipelines + immutable weights), not user-generated training state. (Separate small PRD if a course needs it.)

---

## Architecture

### Module layout

```
packages/browsergrad-kernels/
  src/cache/
    opfs.ts                  # thin OPFS wrapper (handle + read/write)
    pipeline-cache.ts        # WGSL pipeline cache: hash -> {wgsl, layout}
    safetensors.ts           # parser + streaming reader
    quota.ts                 # quota tracking + LRU eviction
    persistence.ts           # navigator.storage.persist()
```

### Cache key design

**Pipeline cache key** = `hash(wgsl_source) + "|" + gpuAdapterInfo`, where:
- `hash(wgsl_source)` — SHA-256 of the post-codegen WGSL string. Stable across runs because PRD-006 codegen is deterministic.
- `gpuAdapterInfo` — from `(await navigator.gpu.requestAdapter()).info`: `vendor + architecture + device + description` joined. A cache built on M2 Pro is not used on RTX 4090; the bytecode is GPU-specific so this avoids subtle correctness/perf bugs after a user upgrades drivers or moves machines.

**Tensor cache key** = `hash(source_url) + "/" + tensor_name`, namespaced per model. A single safetensors URL becomes a directory containing one binary file per tensor.

### OPFS Wrapper (`opfs.ts`)

```typescript
const opfsRoot = await navigator.storage.getDirectory();

export async function opfsWriteBinary(path: string, data: Uint8Array): Promise<void> {
  const handle = await opfsRoot.getFileHandle(path, { create: true });
  const writable = await handle.createWritable();
  await writable.write(data);
  await writable.close();
}

export async function opfsReadBinary(path: string): Promise<Uint8Array | null> {
  try {
    const handle = await opfsRoot.getFileHandle(path);
    const file = await handle.getFile();
    return new Uint8Array(await file.arrayBuffer());
  } catch (e) {
    if (e.name === "NotFoundError") return null;
    throw e;
  }
}

export async function opfsList(prefix: string): Promise<string[]> {
  /* recursive walk */
}

export async function opfsDelete(path: string): Promise<void> { /* ... */ }
export async function opfsUsage(): Promise<{ used: number; quota: number }> {
  const e = await navigator.storage.estimate();
  return { used: e.usage ?? 0, quota: e.quota ?? 0 };
}
```

### Pipeline Cache (`pipeline-cache.ts`)

```typescript
type PipelineEntry = {
  wgsl: string;
  layout: GPUPipelineLayoutDescriptor;
  entryPoint: string;
  workgroupSize: [number, number, number];
  lastAccessed: number;  // for LRU
};

export async function getCachedPipeline(
  device: GPUDevice,
  cacheKey: string
): Promise<GPUComputePipeline | null> {
  const path = `pipelines/${cacheKey}.json`;
  const raw = await opfsReadBinary(path);
  if (!raw) return null;
  const entry: PipelineEntry = JSON.parse(new TextDecoder().decode(raw));
  entry.lastAccessed = Date.now();
  await opfsWriteBinary(path, new TextEncoder().encode(JSON.stringify(entry)));
  return device.createComputePipelineAsync({
    layout: device.createPipelineLayout(entry.layout),
    compute: {
      module: device.createShaderModule({ code: entry.wgsl }),
      entryPoint: entry.entryPoint,
    },
  });
}

export async function storePipeline(
  cacheKey: string,
  entry: Omit<PipelineEntry, "lastAccessed">
): Promise<void> {
  const path = `pipelines/${cacheKey}.json`;
  await opfsWriteBinary(
    path,
    new TextEncoder().encode(JSON.stringify({ ...entry, lastAccessed: Date.now() }))
  );
}
```

Integration with PRD-006 fusion: when fusion emits a kernel, compute `cacheKey = hash(wgsl) + "|" + gpuId`, call `getCachedPipeline`. If miss, run `createComputePipelineAsync` and `storePipeline`. The actual compile call to the WebGPU driver still happens, but the browser's internal cache makes the *second* compile cheap (sub-millisecond) — and we avoid re-running our fusion + codegen.

### Safetensors Streaming Reader (`safetensors.ts`)

The safetensors format:
- 8 bytes: little-endian uint64 = header byte length N.
- N bytes: UTF-8 JSON: `{ "tensor_name": { "dtype": "F32", "shape": [...], "data_offsets": [start, end] }, ... }`.
- Remaining bytes: raw tensor data, contiguous, in order specified by data_offsets.

Streaming algorithm:

```typescript
export async function streamSafetensorsToGPU(
  url: string,
  device: GPUDevice,
  onProgress?: (loaded: number, total: number) => void
): Promise<Map<string, GPUBuffer>> {
  // Step 1: HEAD request to get total size.
  const headRes = await fetch(url, { method: "HEAD" });
  const total = parseInt(headRes.headers.get("content-length") ?? "0", 10);

  // Step 2: Range request for the first 8 bytes — header length prefix.
  const lenRes = await fetch(url, { headers: { Range: "bytes=0-7" } });
  const lenBuf = new Uint8Array(await lenRes.arrayBuffer());
  const headerLen = new DataView(lenBuf.buffer).getBigUint64(0, true);
  const headerLenNum = Number(headerLen);

  // Step 3: Range request for the JSON header.
  const headerRes = await fetch(url, {
    headers: { Range: `bytes=8-${7 + headerLenNum}` },
  });
  const headerJson = JSON.parse(await headerRes.text());

  // Step 4: For each tensor, range-request its bytes and stream into a GPUBuffer.
  const buffers = new Map<string, GPUBuffer>();
  const dataOffset = 8 + headerLenNum;
  for (const [name, meta] of Object.entries(headerJson)) {
    if (name === "__metadata__") continue;
    const [startRel, endRel] = (meta as any).data_offsets as [number, number];
    const start = dataOffset + startRel;
    const end = dataOffset + endRel;  // exclusive in spec but inclusive in HTTP Range
    const len = endRel - startRel;

    const buf = device.createBuffer({
      label: `weight:${name}`,
      size: len,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    const dstArr = new Uint8Array(buf.getMappedRange());

    // Stream-fetch and copy into the mapped buffer (no full-file copy in CPU).
    const tensorRes = await fetch(url, {
      headers: { Range: `bytes=${start}-${end - 1}` },
    });
    const reader = tensorRes.body!.getReader();
    let writePos = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      dstArr.set(value, writePos);
      writePos += value.length;
      onProgress?.(start + writePos, total);
    }
    buf.unmap();
    buffers.set(name, buf);
  }
  return buffers;
}
```

OPFS-cached version: before issuing range requests, check if a local cache directory `tensors/{hash(url)}/` exists. If yes, read each tensor's bytes from OPFS into a `GPUBuffer` directly. If no, fetch from network and *also* write to OPFS as each tensor completes. The second visit takes only the OPFS path.

### Quota & LRU Eviction (`quota.ts`)

```typescript
export async function maybeEvict(): Promise<void> {
  const { used, quota } = await opfsUsage();
  if (used < quota * 0.8) return;

  // List all cache entries with lastAccessed timestamps.
  const entries = await collectAllCacheEntries();  // walks pipelines/ and tensors/
  entries.sort((a, b) => a.lastAccessed - b.lastAccessed);

  const targetUsage = quota * 0.6;  // evict down to 60%
  let currentUsage = used;
  for (const entry of entries) {
    if (currentUsage < targetUsage) break;
    await opfsDelete(entry.path);
    currentUsage -= entry.size;
  }
}
```

Triggered before every write that adds more than 10 MB. For pipeline cache writes (typically <100 KB) we skip the check; for safetensor writes we always check.

### Persistence Request (`persistence.ts`)

```typescript
export async function ensurePersistent(): Promise<boolean> {
  if (await navigator.storage.persisted()) return true;
  return await navigator.storage.persist();
}
```

Called once on first cache write per session. Returns `true` if the browser granted persistence (Chrome/Edge auto-grant for installed PWAs; Firefox prompts the user; Safari has limited support).

### Python API

```python
# packages/browsergrad-runtime/src/python/loader.py (new file)
def load_model(url: str, *, dtype="float32", show_progress=True) -> dict:
    """Load a safetensors file from URL or OPFS, returning a dict of TensorProxy."""
    js_load = browsergrad_runtime.bridge.call("loadSafetensors", url)  # async via JSPI
    weight_handles = js_load.result()  # dict of {name: GPUBuffer handle id}
    return {name: TensorProxy.from_gpu_buffer(h, ...) for name, h in weight_handles.items()}

def clear_cache(*, scope: str = "all") -> None:
    """Clear pipeline cache, tensor cache, or both. scope ∈ {'all', 'pipelines', 'tensors'}."""
    ...

def cache_stats() -> dict:
    """Return {used_bytes, quota_bytes, pipeline_count, tensor_count}."""
    ...
```

### Per-event lifecycle

```
Page load
   ↓
PRD-003 service-worker boot (Pyodide ready ~300ms)
   ↓
ensurePersistent()              # one-shot per session
   ↓
User Python code: load_model("hf://gpt2/model.safetensors")
   ↓
First visit:  fetch + stream + write OPFS  → ~5-30s (network-bound)
Second visit: read OPFS                     → ~2s
   ↓
User code: model(x) — kernels compile
   ↓
For each kernel hash:
  getCachedPipeline → hit:  pull WGSL from OPFS, hand to driver (~5ms)
                   → miss: codegen + create + storePipeline (~50-100ms)
   ↓
Steady state: pipelines reused from PRD-002 in-memory cache; OPFS untouched.
```

---

## Implementation Plan

### Week 1 — OPFS scaffolding

- [ ] Implement `opfs.ts`: read, write, list, delete, usage.
- [ ] Implement `persistence.ts`: `ensurePersistent` with feature-detect for Safari.
- [ ] Unit tests with `node:fs` mock (Vitest); browser integration test with Playwright in a follow-up week.

### Week 2 — Pipeline cache

- [ ] Implement `pipeline-cache.ts` with `getCachedPipeline` and `storePipeline`.
- [ ] Wire into PRD-006's kernel emission point in `packages/browsergrad-kernels/src/jit/fusion.ts` (codegen output → `getCachedPipeline` → fallback to compile + store).
- [ ] Add cache key computation: `sha256(wgsl + adapterInfo)` via Web Crypto (`crypto.subtle.digest`).
- [ ] Integration test: run a conv2d twice; assert second call sees a cached entry; total time of second call < 25% of first.

### Week 3 — Safetensors reader (network path)

- [ ] Implement `safetensors.ts` `streamSafetensorsToGPU` for network sources.
- [ ] Stream-fetch with `Range` requests; per-tensor `mappedAtCreation` `GPUBuffer` allocation.
- [ ] Progress events `{loaded, total, currentTensor}`.
- [ ] Integration test: download a 1 MB test safetensors blob hosted in CI; assert all tensors load correctly; assert CPU memory peak < 2× file size.

### Week 4 — Safetensors OPFS cache layer

- [ ] Extend `safetensors.ts`: check OPFS first; if hit, skip network; if miss, write to OPFS as each tensor's stream completes.
- [ ] Add `__metadata__` entry in OPFS noting source URL, total size, last-access timestamp.
- [ ] Integration test: load same model twice; assert second load < 2s for the 1 MB test blob; assert third load after `clear_cache()` re-downloads.

### Week 5 — Quota + eviction + Python API

- [ ] Implement `quota.ts` LRU eviction.
- [ ] Implement `loader.py` exposing `load_model`, `clear_cache`, `cache_stats`.
- [ ] Bridge via JSPI so Python can `await` on streaming progress.
- [ ] Lab UI hook: progress events surface to React UI for the per-tensor progress bar (PRD-003's lab UI).

### Week 6 — Conformance, benchmarks, hardening

- [ ] Browser matrix test: Chrome, Edge, Firefox on macOS/Windows/Linux. Safari best-effort (OPFS works; `persist()` not).
- [ ] Benchmark cold vs warm reload of:
  - Single conv2d → assert warm < 50ms vs cold ~1s.
  - Transformer block (24 distinct kernel shapes) → assert warm < 500ms vs cold ~2.5s.
  - 250 MB safetensors → assert warm < 5s vs cold ~30s on 100 Mbps.
- [ ] Eviction stress test: fill OPFS to 80%, trigger writes, assert eviction drops to ~60% with oldest entries gone.
- [ ] Add `chaos/no-opfs.test.ts`: simulate OPFS unavailable; assert all paths gracefully degrade to no-cache.
- [ ] Publish `@unlocalhosted/browsergrad-kernels@0.3.0` and `@unlocalhosted/browsergrad-runtime@0.3.0`.

---

## Acceptance Criteria

| # | Criterion | Measurement |
|---|---|---|
| AC1 | Pipeline cache hit reduces compile-equivalent time to < 25% of cold | `pipeline_cache_speedup.bench.ts` |
| AC2 | safetensors second-visit load time < 25% of first-visit on Chrome/Firefox/Edge | `tensor_cache_speedup.bench.ts` |
| AC3 | CPU memory peak during 250 MB safetensors load < 50 MB extra (streaming proven) | `streaming_memory.bench.ts` |
| AC4 | Pipeline cache invalidates correctly when WGSL source changes (key includes hash) | `cache_invalidation.test.ts` |
| AC5 | LRU eviction kicks in at 80% quota; reduces to ~60% | `eviction.test.ts` with quota mock |
| AC6 | `ensurePersistent()` returns true on Chrome desktop; gracefully falls back on Safari | `persistence_compat.test.ts` |
| AC7 | `clear_cache()` removes both pipelines and tensors; subsequent load re-downloads | `clear_cache.test.ts` |
| AC8 | Conformance tests still pass with caching enabled | full integration suite green |
| AC9 | safetensors parser handles all `dtype` variants in real HF GPT-2 file: F32, F16, BF16, I64 | `dtype_coverage.test.ts` |
| AC10 | Cold-start lab page (Pyodide + small transformer block) < 1.5s on second visit | `cold_start.bench.ts` running on Chrome desktop |

---

## Test Strategy

### Unit tests (no browser, no Pyodide)

- `safetensors_parser.test.ts` — synthesize safetensors blobs in memory, assert header parsing.
- `opfs_wrapper.test.ts` — mock OPFS via `node:fs` adapter, assert read/write/delete semantics.
- `quota_lru.test.ts` — given a synthetic list of entries with timestamps + sizes, assert eviction picks the right ones.

### Integration tests (Pyodide-in-Node + WebGPU stub or skip)

- `pipeline_cache_integration.test.ts` — runs only when WebGPU is available; uses a no-op WGSL kernel to verify the round-trip without an actual GPU compute.
- `safetensors_e2e.test.ts` — uses a real ~1 MB safetensors blob bundled in the test repo.

### Browser tests (Playwright)

- `browser/cold_warm_boot.spec.ts` — opens a lab page, measures load time, closes tab, reopens, measures again.
- `browser/eviction.spec.ts` — fills OPFS with synthetic entries, triggers eviction.
- `browser/quota_pressure.spec.ts` — uses Playwright's `context.setOfflineStateForTesting` and storage quota APIs to simulate exhaustion.

### Benchmarks (CI-tracked, not blocking)

- All three "speedup" benchmarks above. Numbers logged to GitHub Actions summary.

---

## Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | OPFS not available on older Safari → no caching, but functionality still works | High | Low | Feature-detect; gracefully no-op; document Safari behavior |
| R2 | Browser-internal pipeline cache invalidated by driver update → our OPFS entry hits but actual recompile is still slow | Medium | Low | Acceptable; user sees one slow page after driver update, then warm again |
| R3 | safetensors files served without `Accept-Ranges: bytes` → can't stream | Medium | Medium | Detect via HEAD response; fall back to single full fetch (still works, just slower & more CPU memory) |
| R4 | `mappedAtCreation` GPU buffer + concurrent network fetch may stall on memory-constrained devices | Medium | Medium | Limit to 2 concurrent tensor uploads; allocate sequentially under low-memory hints |
| R5 | LRU eviction picks the file the user is currently loading | Low | High | Add "pinned" flag to in-flight entries; never evict pinned |
| R6 | `navigator.storage.persist()` denied by Firefox user → cache evicts under disk pressure | Medium | Low | Document; recommend course authors keep cached models < 200 MB |
| R7 | Pipeline cache key collision across distinct WGSL sources (SHA-256 birthday) | Negligible | High | Use 256-bit hash truncated to 128 bits in filename; collision rate well below browser update lifetime |
| R8 | Safetensors header > 100 MB (malformed input) leaks memory | Low | Medium | Cap header size at 10 MB before parse |
| R9 | OPFS write fails mid-stream (disk full) → corrupt cache entry | Medium | Medium | Write to `*.tmp` then atomic rename via OPFS `move`; reader skips entries without rename |
| R10 | Concurrent reads of same cache key — race | Medium | Medium | Add per-key in-memory promise map; second concurrent caller awaits the first |

---

## Open Questions

1. **Cache key version prefix.** Should the cache key include a version byte so we can invalidate on incompatible changes? Resolution: prefix every key with `v1/`. PRD-008 ships `v1`; future schema changes bump to `v2` and existing entries become unreachable (then evicted lazily).

2. **Safetensors over HTTP/2 multiplexing.** For ranged fetches, can we issue all tensor requests in parallel? Resolution: yes for HTTP/2 origins; cap at 4 concurrent on HTTP/1.1 to avoid connection-pool exhaustion.

3. **Should `load_model` accept a Python `bytes` object** for offline-served models (e.g. course author embeds a small model in the page bundle)? Resolution: yes; add `load_model_from_bytes(buf)` path, no caching.

4. **Cross-tab cache coherence.** If two tabs of the same lab open simultaneously, they may both miss cache and both fetch. Resolution: rely on OPFS file locks via `createWritable({ mode: "exclusive" })` where supported; otherwise duplicate fetch is acceptable.

5. **Encrypted models.** Some course authors may want to gate cached weights by user auth. Resolution: out of scope; if needed, ship as a thin wrapper that decrypts in-flight before writing to OPFS.

---

## References

1. **Origin Private File System** — [MDN: File System Access API](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system). The persistent storage primitive.

2. **OPFS introduction** — [web.dev: OPFS](https://web.dev/articles/origin-private-file-system). Practical guide and feature support matrix.

3. **safetensors format** — [HuggingFace safetensors](https://github.com/huggingface/safetensors), [format spec](https://github.com/huggingface/safetensors#format).

4. **HTTP Range requests** — [MDN: Range header](https://developer.mozilla.org/en-US/docs/Web/HTTP/Range_requests). Enables tensor-by-tensor streaming.

5. **WebGPU pipeline caching** — [Chrome WebGPU pipeline caching](https://developer.chrome.com/docs/web-platform/webgpu). Browser-internal cache behavior we delegate to.

6. **WebGPU shader compile cost** — [arXiv:2604.02344](https://arxiv.org/abs/2604.02344). Establishes ~50ms per pipeline compile baseline.

7. **Storage API and persistent storage** — [MDN: navigator.storage.persist](https://developer.mozilla.org/en-US/docs/Web/API/StorageManager/persist). Opting out of eviction.

8. **JSPI (JavaScript Promise Integration)** — [V8 blog: JSPI](https://v8.dev/blog/jspi). Enables Python `await` on JS promises through Pyodide.

9. **Storage quota and estimation** — [MDN: navigator.storage.estimate](https://developer.mozilla.org/en-US/docs/Web/API/StorageManager/estimate). Powers our LRU trigger.

10. **PRD-006 fusion kernel emission** — emits the WGSL strings PRD-008 caches.
