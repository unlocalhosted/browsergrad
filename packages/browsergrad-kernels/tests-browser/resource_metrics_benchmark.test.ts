import { beforeAll, describe, expect, it } from "vitest";
import { createDevice } from "../src/device";
import { matmulDirect, matmulTiledDirect } from "../src/index";
import {
  materializeFloat32,
  releaseDirectBuffer,
  uploadFloat32,
  type DirectDispatchProfile,
} from "../src/runner";
import { createWebGpuRealizerBridge } from "../src/realizer";

interface DeviceCheck {
  available: boolean;
  reason?: string;
  adapterName?: string;
  timestampQuerySupported?: boolean;
}

interface MatmulBenchCase {
  name: string;
  m: number;
  k: number;
  n: number;
  repeats: number;
}

interface BenchSample {
  caseName: string;
  variant: "naive" | "tiled";
  iteration: number;
  timingMode: DirectDispatchProfile["timingMode"];
  confidence: DirectDispatchProfile["confidence"];
  elapsedMs: number;
  outputBytes: number;
}

interface BenchSummary {
  caseName: string;
  variant: "naive" | "tiled";
  samples: number;
  timingMode: DirectDispatchProfile["timingMode"];
  confidence: DirectDispatchProfile["confidence"];
  medianMs: number;
  minMs: number;
  maxMs: number;
  outputBytes: number;
}

const CASES: readonly MatmulBenchCase[] = [
  { name: "skinny-projection", m: 32, k: 128, n: 16, repeats: 5 },
  { name: "square-small", m: 64, k: 64, n: 64, repeats: 5 },
  { name: "square-medium", m: 128, k: 128, n: 128, repeats: 5 },
  { name: "batch-like-wide", m: 48, k: 96, n: 160, repeats: 5 },
];

async function checkDevice(): Promise<DeviceCheck> {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    return { available: false, reason: "navigator.gpu undefined" };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { available: false, reason: "no GPU adapter" };
    return {
      available: true,
      adapterName: adapter.info?.device ?? "unknown",
      timestampQuerySupported: adapter.features.has("timestamp-query"),
    };
  } catch (err) {
    return { available: false, reason: err instanceof Error ? err.message : String(err) };
  }
}

function deterministicMatrix(length: number, seed: number): Float32Array {
  const data = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    data[i] = Math.sin((i + 1) * (seed + 0.17)) * Math.cos((i + 3) * 0.031);
  }
  return data;
}

async function runMatmulVariant(
  device: Awaited<ReturnType<typeof createDevice>>,
  benchCase: MatmulBenchCase,
  variant: BenchSample["variant"],
): Promise<BenchSample[]> {
  const a = deterministicMatrix(benchCase.m * benchCase.k, benchCase.m + benchCase.k);
  const b = deterministicMatrix(benchCase.k * benchCase.n, benchCase.k + benchCase.n);
  const aBuf = uploadFloat32(device, a);
  const bBuf = uploadFloat32(device, b);
  const samples: BenchSample[] = [];
  try {
    for (let i = 0; i < benchCase.repeats + 1; i++) {
      const label = `bench:${variant}:${benchCase.name}:${i}`;
      const result = variant === "naive"
        ? matmulDirect(device, aBuf, bBuf, benchCase.m, benchCase.k, benchCase.n, { label })
        : matmulTiledDirect(device, aBuf, bBuf, benchCase.m, benchCase.k, benchCase.n, { label });
      try {
        const profile = await requireProfile(result.profile, label);
        await materializeFloat32(device, result.buffer, result.byteLength);
        if (i === 0) continue;
        samples.push({
          caseName: benchCase.name,
          variant,
          iteration: i,
          timingMode: profile.timingMode,
          confidence: profile.confidence,
          elapsedMs: profile.gpuElapsedMs ?? profile.queueElapsedMs ?? Number.NaN,
          outputBytes: result.byteLength,
        });
      } finally {
        releaseDirectBuffer(device, result.buffer, result.byteLength);
      }
    }
  } finally {
    aBuf.destroy();
    bBuf.destroy();
  }
  return samples;
}

async function requireProfile(
  profile: Promise<DirectDispatchProfile> | undefined,
  label: string,
): Promise<DirectDispatchProfile> {
  if (!profile) throw new Error(`missing profile for ${label}`);
  const resolved = await profile;
  expect(resolved.label).toBe(label);
  return resolved;
}

function summarize(samples: readonly BenchSample[]): BenchSummary {
  if (samples.length === 0) throw new Error("cannot summarize empty samples");
  const elapsed = samples.map((sample) => sample.elapsedMs).sort((a, b) => a - b);
  return {
    caseName: samples[0]!.caseName,
    variant: samples[0]!.variant,
    samples: samples.length,
    timingMode: samples[0]!.timingMode,
    confidence: samples[0]!.confidence,
    medianMs: elapsed[Math.floor(elapsed.length / 2)]!,
    minMs: elapsed[0]!,
    maxMs: elapsed[elapsed.length - 1]!,
    outputBytes: samples[0]!.outputBytes,
  };
}

describe("real WebGPU resource metrics benchmarks", () => {
  let deviceCheck: DeviceCheck;

  beforeAll(async () => {
    deviceCheck = await checkDevice();
    if (!deviceCheck.available) {
      console.warn(`[skip] WebGPU benchmark unavailable: ${deviceCheck.reason}`);
    } else {
      console.log(`[bench] WebGPU adapter: ${deviceCheck.adapterName}`);
    }
  });

  it("captures repeated naive-vs-tiled matmul timings across matrix shapes", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const summaries: BenchSummary[] = [];
    for (const benchCase of CASES) {
      const naive = await runMatmulVariant(device, benchCase, "naive");
      const tiled = await runMatmulVariant(device, benchCase, "tiled");
      expect(naive).toHaveLength(benchCase.repeats);
      expect(tiled).toHaveLength(benchCase.repeats);
      for (const sample of [...naive, ...tiled]) {
        expect(sample.elapsedMs).toEqual(expect.any(Number));
        expect(Number.isFinite(sample.elapsedMs)).toBe(true);
        expect(sample.elapsedMs).toBeGreaterThanOrEqual(0);
        expect(sample.outputBytes).toBe(benchCase.m * benchCase.n * Float32Array.BYTES_PER_ELEMENT);
        expect(sample.confidence).toMatch(/^(exact|coarse|unavailable)$/);
      }
      summaries.push(summarize(naive), summarize(tiled));
    }

    console.info(`[resource-benchmark] ${JSON.stringify(summaries)}`);
    expect(summaries).toHaveLength(CASES.length * 2);
    expect(summaries.some((summary) => summary.variant === "naive")).toBe(true);
    expect(summaries.some((summary) => summary.variant === "tiled")).toBe(true);
  });

  it("captures exact bridge-owned memory under repeated resident allocations", async () => {
    if (!deviceCheck.available) return;
    const device = await createDevice();
    const bridge = createWebGpuRealizerBridge(device, { profiling: false });
    const live: number[] = [];
    const buffers = [
      new Float32Array(16),
      new Float32Array(256),
      new Float32Array(1024),
      new Float32Array(4096),
    ];

    const handles = buffers.map((buffer) => {
      const handle = bridge.upload(new Uint8Array(buffer.buffer), [buffer.length], "float32");
      live.push(buffer.byteLength);
      const expectedBytes = live.reduce((sum, bytes) => sum + bytes, 0);
      expect(bridge.resourceSnapshot()).toMatchObject({
        currentOwnedGpuBytes: expectedBytes,
        peakOwnedGpuBytes: expectedBytes,
        aliveHandleCount: live.length,
      });
      return handle;
    });

    while (handles.length > 0) {
      const handle = handles.pop()!;
      const released = live.pop()!;
      bridge.release(handle);
      const expectedBytes = live.reduce((sum, bytes) => sum + bytes, 0);
      expect(bridge.resourceSnapshot()).toMatchObject({
        currentOwnedGpuBytes: expectedBytes,
        totalReleasedGpuBytes: buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0) - expectedBytes,
        aliveHandleCount: live.length,
      });
      expect(released).toBeGreaterThan(0);
    }
  });
});
