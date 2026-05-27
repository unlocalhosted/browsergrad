/**
 * Performance benchmark harness.
 *
 * Sweeps matmul shapes, measures forward latency on the NumPy realizer
 * (current default), counts fused vs unfused kernels in a softmax DAG,
 * and times the trace cache's hit vs miss path. The goal is structured
 * data: shape → ms, plus observability counters.
 *
 * No assertions about absolute numbers — those vary by machine. The
 * assertions are correctness gates (the result is right) and the data
 * goes to /tmp/bg-perf-report.md for inspection.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { writeFileSync } from "node:fs";
import { clearNamespace, getJitTarget } from "./pyodide-host";

describe("perf bench (data only — no perf assertions)", () => {
  beforeAll(async () => {
    await getJitTarget();
  }, 120_000);

  it("collects matmul / softmax / fusion / amp / vmap timing data", async () => {
    const target = await getJitTarget();
    await clearNamespace(target);
    const report = await target.run<{
      matmul_sweep: Array<{ M: number; N: number; K: number; ms: number; gflops: number }>;
      softmax_sweep: Array<{ batch: number; features: number; ms: number; fused: number }>;
      amp_vs_f32: { f32_ms: number; amp_ms: number; max_diff: number };
      trace_cache: { cold_ms: number; warm_ms_avg: number };
      vmap_vs_loop: { vmap_ms: number; loop_ms: number };
    }>(`
import browsergrad_jit as bg
import numpy as np
import time

def time_n(fn, n=5):
    """Return (mean_ms, last_result). Warmup once, then time n calls."""
    fn()  # warmup
    ts = []
    out = None
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return sum(ts) / len(ts), out

# Matmul shape sweep (NumPy realizer; no real GPU in pyodide-in-node).
matmul_sweep = []
for (M, N, K) in [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 64, 256), (32, 32, 1024)]:
    a = bg.from_numpy(np.random.randn(M, K).astype(np.float32))
    b = bg.from_numpy(np.random.randn(K, N).astype(np.float32))
    ms, _ = time_n(lambda: (a @ b).numpy(), n=3)
    flops = 2.0 * M * N * K  # multiply + add per output element
    gflops = (flops / 1e9) / (ms / 1000) if ms > 0 else 0
    matmul_sweep.append({"M": M, "N": N, "K": K, "ms": ms, "gflops": gflops})

# Softmax fusion sweep — confirms PRD-006 fires.
softmax_sweep = []
bg.jit.use_fusion(True)
for batch, features in [(8, 32), (16, 128), (32, 512)]:
    x = bg.from_numpy(np.random.randn(batch, features).astype(np.float32))
    ms, _ = time_n(lambda: bg.nn.functional.softmax(x, dim=-1).numpy(), n=3)
    fused = len(bg.jit.debug_fused_kernels())
    softmax_sweep.append({"batch": batch, "features": features, "ms": ms, "fused": fused})

# AMP vs f32 — same matmul, both paths.
size = 256
a_f32 = bg.from_numpy(np.random.randn(size, size).astype(np.float32))
b_f32 = bg.from_numpy(np.random.randn(size, size).astype(np.float32))
ms_f32, ref_f32 = time_n(lambda: (a_f32 @ b_f32).numpy(), n=3)
def amp_call():
    with bg.amp.autocast(device_type="webgpu", dtype="float16"):
        return (a_f32 @ b_f32).numpy()
ms_amp, ref_amp = time_n(amp_call, n=3)
amp_vs_f32 = {
    "f32_ms": ms_f32,
    "amp_ms": ms_amp,
    "max_diff": float(np.max(np.abs(ref_f32.astype(np.float32) - ref_amp.astype(np.float32)))),
}

# Trace cache cold vs warm.
bg.clear_cache("trace")
bg.jit.use_trace_cache(True)
def trace_workload():
    x = bg.from_numpy(np.random.randn(32, 32).astype(np.float32))
    return (x @ x.T + x.sum() * 0.01).numpy()
t0 = time.perf_counter()
trace_workload()
cold_ms = (time.perf_counter() - t0) * 1000
warm_times = []
for _ in range(5):
    t0 = time.perf_counter()
    trace_workload()
    warm_times.append((time.perf_counter() - t0) * 1000)
trace_cache = {
    "cold_ms": cold_ms,
    "warm_ms_avg": sum(warm_times) / len(warm_times),
}

# vmap vs Python-for-loop for per-sample sums.
B = 32
batched = bg.from_numpy(np.random.randn(B, 16).astype(np.float32))
def vmap_path():
    return bg.func.vmap(lambda x: x.sum())(batched).numpy()
def loop_path():
    out = np.zeros(B, dtype=np.float32)
    arr = batched.numpy()
    for i in range(B):
        out[i] = float(bg.from_numpy(arr[i]).sum().numpy())
    return out
ms_vmap, _ = time_n(vmap_path, n=3)
ms_loop, _ = time_n(loop_path, n=3)

{
    "matmul_sweep": matmul_sweep,
    "softmax_sweep": softmax_sweep,
    "amp_vs_f32": amp_vs_f32,
    "trace_cache": trace_cache,
    "vmap_vs_loop": {"vmap_ms": ms_vmap, "loop_ms": ms_loop},
}
`);

    // Write markdown report.
    const lines: string[] = [];
    lines.push("# Browsergrad-jit performance bench");
    lines.push("");
    lines.push("> NumPy realizer baseline. Runs in pyodide-in-node; no real GPU.");
    lines.push("> Numbers are wall-clock from the Python side. WGSL backend (PRD-011.5)");
    lines.push("> not exercised here — needs browser CI with WebGPU.");
    lines.push("");
    lines.push("## Matmul shape sweep");
    lines.push("");
    lines.push("| M | N | K | ms | GFLOPS |");
    lines.push("|---|---|---|----|--------|");
    for (const row of report.matmul_sweep) {
      lines.push(`| ${row.M} | ${row.N} | ${row.K} | ${row.ms.toFixed(2)} | ${row.gflops.toFixed(2)} |`);
    }
    lines.push("");
    lines.push("## Softmax fusion sweep");
    lines.push("");
    lines.push("| batch | features | ms | fused kernels |");
    lines.push("|-------|----------|----|---------------|");
    for (const row of report.softmax_sweep) {
      lines.push(`| ${row.batch} | ${row.features} | ${row.ms.toFixed(2)} | ${row.fused} |`);
    }
    lines.push("");
    lines.push("## AMP f16 vs f32 (256×256 matmul)");
    lines.push("");
    lines.push("| path | ms | max abs diff vs f32 ref |");
    lines.push("|------|----|------------------------|");
    lines.push(`| f32 | ${report.amp_vs_f32.f32_ms.toFixed(2)} | 0 |`);
    lines.push(`| amp (f16 input, f32 accumulator) | ${report.amp_vs_f32.amp_ms.toFixed(2)} | ${report.amp_vs_f32.max_diff.toExponential(2)} |`);
    lines.push("");
    lines.push("> Note: AMP on NumPy is **not** faster than f32 — NumPy lacks f16 SIMD.");
    lines.push("> The educational value is the cast-pass + fp32 accumulator correctness.");
    lines.push("> Wall-clock wins land when PRD-011.5's WGSL kernels run on a real device.");
    lines.push("");
    lines.push("## Trace cache");
    lines.push("");
    lines.push(`- Cold (first call): ${report.trace_cache.cold_ms.toFixed(2)}ms`);
    lines.push(`- Warm (avg of 5 subsequent calls): ${report.trace_cache.warm_ms_avg.toFixed(2)}ms`);
    lines.push("");
    lines.push("## vmap vs Python loop (32 samples)");
    lines.push("");
    lines.push(`- vmap path: ${report.vmap_vs_loop.vmap_ms.toFixed(2)}ms`);
    lines.push(`- Python for-loop: ${report.vmap_vs_loop.loop_ms.toFixed(2)}ms`);
    lines.push("");
    writeFileSync("/tmp/bg-perf-report.md", lines.join("\n"));

    // Sanity-only assertions.
    expect(report.matmul_sweep.length).toBe(5);
    expect(report.amp_vs_f32.max_diff).toBeLessThan(1.0); // f16 noise bounded
  }, 120_000);
});
