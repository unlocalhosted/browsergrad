import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import { layoutArtifactPayload, verifyLayoutArtifact } from "@unlocalhosted/browsergrad-semantic-core/layout";
import { kernelArtifactPayload, verifyKernelArtifact } from "@unlocalhosted/browsergrad-semantic-core/kernel";
import { hashSemanticArtifact } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { createDevice } from "../src/device";
import {
  prepareSemanticViewCopyWgsl,
  runSemanticViewCopyWebGpu,
  type PreparedSemanticViewCopyWgsl,
} from "../src/semantic_view_copy";
import { getWgslPipelineCacheStats } from "../src/wgsl_program";

let prepared: PreparedSemanticViewCopyWgsl;

beforeAll(async () => {
  const layout = await verifyLayoutArtifact({
    schema: "browsergrad.layout",
    version: { major: 1, minor: 0 },
    producer: { id: "device-lifecycle-tests", version: "1" },
    artifactId: "dense-layout",
    requiredExtensions: [],
    payload: {
      symbols: [],
      constraints: [],
      allocations: [
        { allocationId: "source", byteLength: constant("16"), memorySpace: { kind: "global" }, alignmentBytes: 4, aliasSetId: "sourceAlias" },
        { allocationId: "destination", byteLength: constant("16"), memorySpace: { kind: "global" }, alignmentBytes: 4, aliasSetId: "destinationAlias" },
      ],
      indexMaps: [
        { indexMapId: "sourceMap", coordinateRank: 2, locationUnit: "element", location: rowMajor(), inBounds: { kind: "bool", value: true } },
        { indexMapId: "destinationMap", coordinateRank: 2, locationUnit: "element", location: rowMajor(), inBounds: { kind: "bool", value: true } },
      ],
      views: [
        { viewId: "sourceView", allocationId: "source", dtype: "f32", byteOffset: constant("0"), shape: [constant("2"), constant("2")], indexMapId: "sourceMap", requiredAlignmentBytes: 4 },
        { viewId: "destinationView", allocationId: "destination", dtype: "f32", byteOffset: constant("0"), shape: [constant("2"), constant("2")], indexMapId: "destinationMap", requiredAlignmentBytes: 4 },
      ],
    },
  });
  const payload = layoutArtifactPayload(layout);
  const kernel = await verifyKernelArtifact({
    schema: "browsergrad.kernel",
    version: { major: 1, minor: 0 },
    producer: { id: "device-lifecycle-tests", version: "1" },
    artifactId: "dense-copy",
    requiredExtensions: [],
    payload: {
      layoutSemanticHash: await hashSemanticArtifact(layout),
      operations: [{
        operationId: "copy",
        kind: "view-copy",
        version: { major: 1, minor: 0 },
        dtype: "f32",
        source: { viewId: payload.views[0]!.viewId, access: "read", invalidSource: { kind: "reject" } },
        destination: { viewId: payload.views[1]!.viewId, access: "write" },
        overlap: { kind: "forbid" },
      }],
    },
  }, { layout });
  prepared = await prepareSemanticViewCopyWgsl(layout, kernel, {
    operationId: kernelArtifactPayload(kernel).operations[0]!.operationId,
  });
});

beforeEach(() => {
  vi.stubGlobal("GPUBufferUsage", { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, MAP_READ: 8 });
  vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 });
  vi.stubGlobal("GPUMapMode", { READ: 1 });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("semantic view-copy WebGPU device lifecycle", () => {
  it.each([
    ["shader", "BG-WEBGPU-VIEW-COPY-SHADER"],
    ["pipeline", "BG-WEBGPU-VIEW-COPY-PIPELINE"],
  ] as const)("distinguishes %s creation and drains both diagnostic phases", async (failure, code) => {
    const fake = createFakeGpu({ failure, validationOnPreparation: true });
    const device = await createDevice({ device: fake.device });

    await expect(run(device)).rejects.toMatchObject({ code });
    expect(fake.popCalls()).toBe(3);
  });

  it("pops every scope after a pop failure and keeps out-of-memory distinct", async () => {
    const popFailure = createFakeGpu({ failure: "shader", rejectPopCall: 1 });
    await expect(run(await createDevice({ device: popFailure.device }))).rejects.toMatchObject({
      code: "BG-WEBGPU-VIEW-COPY-INTERNAL",
      path: "$.device.errorScope",
    });
    expect(popFailure.popCalls()).toBe(3);

    const outOfMemory = createFakeGpu({ failure: "pipeline", outOfMemoryOnPreparation: true });
    await expect(run(await createDevice({ device: outOfMemory.device }))).rejects.toMatchObject({
      code: "BG-WEBGPU-VIEW-COPY-OUT-OF-MEMORY",
    });
    expect(outOfMemory.popCalls()).toBe(3);
  });

  it("bounds one in-flight run, suppresses timed-out results, and releases after cleanup", async () => {
    const fake = createFakeGpu({ deferredCompilation: true });
    const device = await createDevice({ device: fake.device });
    await expect(run(device, { timeoutMs: 1 })).rejects.toMatchObject({
      code: "BG-WEBGPU-VIEW-COPY-TIMEOUT",
    });
    await expect(run(device)).rejects.toMatchObject({
      code: "BG-WEBGPU-VIEW-COPY-RESOURCE-LIMIT",
      path: "$.device.inFlight",
    });

    fake.resolveCompilation();
    await fake.readbackCompleted;
    await nextTask();
    await expect(run(device)).resolves.toMatchObject({ trace: { submitted: true } });
  });

  it("suppresses an aborted post-submit result until readback cleanup finishes", async () => {
    const fake = createFakeGpu({ deferredMap: true });
    const device = await createDevice({ device: fake.device });
    const controller = new AbortController();
    const first = run(device, { signal: controller.signal });
    await fake.submitted;
    controller.abort();
    await expect(first).rejects.toMatchObject({ code: "BG-WEBGPU-VIEW-COPY-CANCELLED" });
    await expect(run(device)).rejects.toMatchObject({ path: "$.device.inFlight" });

    fake.resolveMap();
    await fake.readbackCompleted;
    await nextTask();
    await expect(run(device)).resolves.toMatchObject({ trace: { submitted: true } });
  });

  it("invalidates every wrapper and the shared pipeline cache on device loss", async () => {
    const fake = createFakeGpu({ deferredCompilation: true });
    const firstWrapper = await createDevice({ device: fake.device });
    const secondWrapper = await createDevice({ device: fake.device });
    const first = run(firstWrapper);
    await fake.pipelineStarted;
    expect(getWgslPipelineCacheStats(firstWrapper).pipelineCacheSize).toBe(1);

    fake.loseDevice();
    await expect(first).rejects.toMatchObject({ code: "BG-WEBGPU-VIEW-COPY-DEVICE-LOST" });
    await waitFor(() => getWgslPipelineCacheStats(firstWrapper).pipelineCacheSize === 0);
    await expect(run(secondWrapper)).rejects.toMatchObject({ code: "BG-WEBGPU-VIEW-COPY-DEVICE-LOST" });
    fake.resolveCompilation();
  });
});

function run(
  device: Awaited<ReturnType<typeof createDevice>>,
  options: Parameters<typeof runSemanticViewCopyWebGpu>[3] = {},
) {
  return runSemanticViewCopyWebGpu(device, prepared, {
    sourceWords: new Uint32Array([1, 2, 3, 4]),
    destinationWords: new Uint32Array(4),
  }, options);
}

interface FakeGpuOptions {
  readonly failure?: "shader" | "pipeline";
  readonly deferredCompilation?: boolean;
  readonly deferredMap?: boolean;
  readonly validationOnPreparation?: boolean;
  readonly outOfMemoryOnPreparation?: boolean;
  readonly rejectPopCall?: number;
}

interface FakeGpuControl {
  readonly device: GPUDevice;
  readonly submitted: Promise<void>;
  readonly pipelineStarted: Promise<void>;
  readonly readbackCompleted: Promise<void>;
  resolveCompilation(): void;
  resolveMap(): void;
  loseDevice(): void;
  popCalls(): number;
}

function createFakeGpu(options: FakeGpuOptions = {}): FakeGpuControl {
  const compilation = deferred<void>();
  const mapped = deferred<void>();
  const lost = deferred<GPUDeviceLostInfo>();
  const submitted = deferred<void>();
  const pipelineStarted = deferred<void>();
  const readbackCompleted = deferred<void>();
  let popCalls = 0;
  let scopePhase = 0;
  const scopeStack: GPUErrorFilter[] = [];

  const queue = {
    writeBuffer: () => undefined,
    submit: () => submitted.resolve(),
    onSubmittedWorkDone: async () => undefined,
  };
  const device = {
    features: new Set<string>(),
    limits: {
      maxBufferSize: 1 << 20,
      maxStorageBufferBindingSize: 1 << 20,
      maxComputeWorkgroupsPerDimension: 65_535,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxBindingsPerBindGroup: 8,
      maxStorageBuffersPerShaderStage: 8,
    },
    lost: lost.promise,
    queue,
    pushErrorScope(filter: GPUErrorFilter) {
      scopeStack.push(filter);
      if (scopeStack.length === 1) scopePhase += 1;
    },
    async popErrorScope() {
      popCalls += 1;
      const filter = scopeStack.pop();
      if (options.rejectPopCall === popCalls) throw new Error("forced pop failure");
      if (scopePhase === 1 && filter === "validation" && options.validationOnPreparation) {
        return { message: "forced validation" } as GPUError;
      }
      if (scopePhase === 1 && filter === "out-of-memory" && options.outOfMemoryOnPreparation) {
        return { message: "forced out of memory" } as GPUError;
      }
      return null;
    },
    createBuffer(descriptor: GPUBufferDescriptor) {
      const bytes = new ArrayBuffer(Number(descriptor.size));
      const isReadback = (descriptor.usage & 8) !== 0;
      return {
        mapAsync: () => isReadback && options.deferredMap ? mapped.promise : Promise.resolve(),
        getMappedRange: () => bytes,
        unmap: () => readbackCompleted.resolve(),
        destroy: () => undefined,
      } as unknown as GPUBuffer;
    },
    createBindGroupLayout: () => ({} as GPUBindGroupLayout),
    createPipelineLayout: () => ({} as GPUPipelineLayout),
    createShaderModule: () => ({
      getCompilationInfo: async () => {
        if (options.deferredCompilation) await compilation.promise;
        return {
          messages: options.failure === "shader"
            ? [{ type: "error", message: "forced shader error", lineNum: 1, linePos: 1, offset: 0, length: 1 }]
            : [],
        } as unknown as GPUCompilationInfo;
      },
    } as GPUShaderModule),
    createComputePipelineAsync: async () => {
      pipelineStarted.resolve();
      if (options.deferredCompilation) await compilation.promise;
      if (options.failure === "pipeline") throw new Error("forced pipeline error");
      return {} as GPUComputePipeline;
    },
    createBindGroup: () => ({} as GPUBindGroup),
    createCommandEncoder: () => ({
      beginComputePass: () => ({
        setPipeline: () => undefined,
        setBindGroup: () => undefined,
        dispatchWorkgroups: () => undefined,
        end: () => undefined,
      }),
      copyBufferToBuffer: () => undefined,
      finish: () => ({} as GPUCommandBuffer),
    } as unknown as GPUCommandEncoder),
  } as unknown as GPUDevice;

  return {
    device,
    submitted: submitted.promise,
    pipelineStarted: pipelineStarted.promise,
    readbackCompleted: readbackCompleted.promise,
    resolveCompilation: compilation.resolve,
    resolveMap: mapped.resolve,
    loseDevice: () => lost.resolve({ reason: "unknown", message: "forced device loss" } as GPUDeviceLostInfo),
    popCalls: () => popCalls,
  };
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((innerResolve) => {
    resolve = innerResolve;
  });
  return { promise, resolve: resolve as T extends void ? () => void : typeof resolve };
}

async function waitFor(predicate: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error("condition did not become true");
}

function nextTask(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function constant(value: string) {
  return { kind: "const" as const, value };
}

function rowMajor() {
  return {
    kind: "add" as const,
    terms: [
      { kind: "mul" as const, lhs: { kind: "coordinate" as const, axis: 0 }, rhs: constant("2") },
      { kind: "coordinate" as const, axis: 1 },
    ],
  };
}
