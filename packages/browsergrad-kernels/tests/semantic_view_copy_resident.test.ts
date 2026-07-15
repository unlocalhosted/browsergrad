import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import {
  createVerifiedDensePermutationViewCopyArtifacts,
  createVerifiedViewCopyArtifacts,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import { parseWireI64, type WireI64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { createDevice } from "../src/device";
import { createWebGpuRealizerBridge } from "../src/realizer";
import {
  prepareSemanticViewCopyWgsl,
  runPreparedSemanticViewCopyResident,
  type PreparedSemanticViewCopyWgsl,
  type SemanticViewCopyResidentSource,
} from "../src/semantic_view_copy";
import {
  assertPreparedTensorPlanSemanticRequests,
  preparedSemanticViewCopyForValue,
  runTensorGpuPlanSemantic,
  runTensorGpuPlanResidentSemantic,
} from "../src/tensor_plan_semantics";

const wire = (value: string): WireI64 => parseWireI64(value);
const constant = (value: string) => ({ kind: "const" as const, value: wire(value) });

let prepared: PreparedSemanticViewCopyWgsl;
let offsetDestination: PreparedSemanticViewCopyWgsl;

beforeAll(async () => {
  const permutation = await createVerifiedDensePermutationViewCopyArtifacts({
    inputShape: [wire("2"), wire("3")],
    axes: [1, 0],
    dtype: "f32",
  });
  prepared = await prepareSemanticViewCopyWgsl(permutation.layout, permutation.kernel, {
    operationId: permutation.operationId,
  });

  const offset = await createVerifiedViewCopyArtifacts({
    dtype: "f32",
    symbols: [],
    constraints: [],
    source: {
      layout: { kind: "strided", shape: [constant("2"), constant("3")], strides: [constant("3"), constant("1")] },
      allocation: { byteLength: constant("24"), memorySpace: { kind: "global" }, alignmentBytes: 4 },
      byteOffset: constant("0"),
      requiredAlignmentBytes: 4,
    },
    destination: {
      layout: { kind: "strided", shape: [constant("2"), constant("3")], strides: [constant("3"), constant("1")] },
      allocation: { byteLength: constant("28"), memorySpace: { kind: "global" }, alignmentBytes: 4 },
      byteOffset: constant("4"),
      requiredAlignmentBytes: 4,
    },
    invalidSource: { kind: "reject" },
  });
  offsetDestination = await prepareSemanticViewCopyWgsl(offset.layout, offset.kernel, {
    operationId: offset.operationId,
  });
});

beforeEach(() => {
  vi.stubGlobal("GPUBufferUsage", { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, MAP_READ: 8 });
  vi.stubGlobal("GPUMapMode", { READ: 1 });
  vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("resident semantic view-copy", () => {
  it("dispatches canonical WGSL over resident roots without upload or readback", async () => {
    const fake = createFakeGpu();
    const device = await createDevice({ device: fake.device });
    const source = fake.buffer(24, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    const result = await runPreparedSemanticViewCopyResident(device, prepared, {
      buffer: source,
      byteLength: 24,
    });

    expect(result.byteLength).toBe(24);
    expect(result.buffer).not.toBe(source);
    expect(result.buffer.size).toBe(24);
    expect(fake.shaderSources).toEqual([prepared.program.wgsl]);
    expect(fake.dispatches).toEqual([[1, 1, 1]]);
    expect(fake.submitCount()).toBe(1);
    expect(fake.writeCount()).toBe(0);
    expect(fake.copyCount()).toBe(0);
  });

  it("keeps exact semantic preparation and dispatch profiles with the live bridge handle", async () => {
    const fake = createFakeGpu();
    const device = await createDevice({ device: fake.device });
    const bridge = createWebGpuRealizerBridge(device, { profiling: true });
    const handle = await bridge.run_tensor_plan_resident_semantic(
      permutationPlan(),
      JSON.stringify(permutationRequests()),
      [{ value_id: 0, data: new Uint8Array(24) }],
      "float32",
    );

    const trace = await bridge.semanticTensorPlanExecutionTrace(handle);
    expect(trace.preparation.requests).toHaveLength(1);
    expect(trace.preparation.requests[0]).toMatchObject({
      valueId: 1,
      layoutSemanticHash: prepared.semantic.layoutSemanticHash,
      kernelSemanticHash: prepared.semantic.kernelSemanticHash,
      wgslModuleHash: prepared.wgslModuleHash,
      logicalInvocationCount: [6, 1, 1],
      plannedWorkgroupCount: [1, 1, 1],
    });
    expect(trace.dispatchProfiles).toHaveLength(1);
    expect(trace.dispatchProfiles[0]?.dispatchCount).toEqual(fake.dispatches[0]);
    expect(trace.dispatchProfiles[0]?.workgroupSize).toEqual([64, 1, 1]);
    expect(Object.isFrozen(trace)).toBe(true);
    expect(Object.isFrozen(trace.dispatchProfiles)).toBe(true);
    expect(Object.isFrozen(trace.dispatchProfiles[0]?.dispatchCount)).toBe(true);

    const flushed = await bridge.flushProfiles();
    expect(flushed.pendingProfileCount).toBe(0);
    expect(flushed.passProfiles).toHaveLength(1);
    bridge.release(handle);
    await expect(bridge.semanticTensorPlanExecutionTrace(handle)).rejects.toThrow(
      /unknown handle/,
    );

    const ordinary = bridge.upload(new Uint8Array(24), [2, 3], "float32");
    await expect(bridge.semanticTensorPlanExecutionTrace(ordinary)).rejects.toThrow(
      /has no semantic tensor-plan execution trace/,
    );
    bridge.release(ordinary);
    expect(bridge.resourceSnapshot().currentOwnedGpuBytes).toBe(0);
  });

  it("routes tensor-plan PERMUTE through the prepared semantic WGSL", async () => {
    const fake = createFakeGpu();
    const device = await createDevice({ device: fake.device });
    const source = fake.buffer(24, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const result = await runTensorGpuPlanResidentSemantic(
      device,
      permutationPlan(),
      permutationRequests(),
      [{ valueId: 0, resident: { buffer: source, byteLength: 24 } }],
    );

    expect(result.shape).toEqual([3, 2]);
    expect(result.byteLength).toBe(24);
    expect(result.residentValueId).toBe(1);
    expect(result.buffer).not.toBe(source);
    expect(fake.shaderSources).toEqual([prepared.program.wgsl]);
    expect(fake.shaderSources[0]).not.toContain("tensor_plan_permute");
    expect(fake.submitCount()).toBe(1);
    expect(fake.writeCount()).toBe(0);
    expect(fake.copyCount()).toBe(0);
    assertPreparedTensorPlanSemanticRequests(result.semanticPreparation);
    const executedPreparation = preparedSemanticViewCopyForValue(
      result.semanticPreparation,
      1,
    );
    expect(executedPreparation?.program.wgsl).toBe(fake.shaderSources[0]);
    const dispatchProfiles = await Promise.all(result.profiles);
    expect(dispatchProfiles).toHaveLength(1);
    expect(dispatchProfiles[0]?.dispatchCount).toEqual(fake.dispatches[0]);
    expect(dispatchProfiles[0]?.dispatchCount).toEqual([1, 1, 1]);
    expect(dispatchProfiles[0]?.workgroupSize).toEqual([64, 1, 1]);
  });

  it("settles delayed LIFO scopes before minting a semantic resident handle", async () => {
    const fake = createFakeGpu({ deferredErrorScopes: true });
    const device = await createDevice({ device: fake.device });
    const bridge = createWebGpuRealizerBridge(device, { profiling: true });
    const pending = bridge.run_tensor_plan_resident_semantic(
      permutationPlan(),
      JSON.stringify(permutationRequests()),
      [{ value_id: 0, data: new Uint8Array(24) }],
      "float32",
    );

    await vi.waitFor(() => {
      expect(fake.scopeEvents).toEqual([
        "push:internal",
        "push:out-of-memory",
        "push:validation",
        "pop:validation",
        "pop:out-of-memory",
        "pop:internal",
      ]);
    });
    expect(bridge.aliveHandleCount()).toBe(0);

    fake.settleErrorScopes("validation", "delayed validation failure");
    await expect(pending).rejects.toThrow(/validation.*delayed validation failure/u);
    expect(bridge.aliveHandleCount()).toBe(0);
    expect(fake.destroyCount()).toBeGreaterThan(0);
    expect(device.getStats()).toMatchObject({
      pipelineCacheSize: 0,
      outputBufferPoolBuffers: 0,
      outputBufferPoolBytes: 0,
    });

    const retry = bridge.run_tensor_plan_resident_semantic(
      permutationPlan(),
      JSON.stringify(permutationRequests()),
      [{ value_id: 0, data: new Uint8Array(24) }],
      "float32",
    );
    await vi.waitFor(() => expect(fake.scopeEvents).toHaveLength(12));
    fake.settleErrorScopes();
    const retryHandle = await retry;
    expect(bridge.aliveHandleCount()).toBe(1);
    bridge.release(retryHandle);
    expect(bridge.aliveHandleCount()).toBe(0);
  });

  it("destroys a failed materialization root and admits a clean nonresident retry", async () => {
    const fake = createFakeGpu();
    const device = await createDevice({ device: fake.device });
    const source = fake.buffer(24, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    fake.setMaterializeMapFailure("map failed");

    await expect(runTensorGpuPlanSemantic(
      device,
      permutationPlan(),
      permutationRequests(),
      [{ valueId: 0, resident: { buffer: source, byteLength: 24 } }],
    )).rejects.toThrow(/map failed/u);
    expect(device.getStats()).toMatchObject({
      pipelineCacheSize: 0,
      outputBufferPoolBuffers: 0,
      outputBufferPoolBytes: 0,
    });

    fake.setMaterializeMapFailure();
    const retry = await runTensorGpuPlanSemantic(
      device,
      permutationPlan(),
      permutationRequests(),
      [{ valueId: 0, resident: { buffer: source, byteLength: 24 } }],
    );
    expect(retry.data).toHaveLength(6);
    expect(device.getStats().outputBufferPoolBuffers).toBe(1);
  });

  it("rejects forged prepared objects before touching the device", async () => {
    const fake = createFakeGpu();
    const device = await createDevice({ device: fake.device });
    const forged = { ...prepared } as PreparedSemanticViewCopyWgsl;

    await expect(runPreparedSemanticViewCopyResident(device, forged, {
      buffer: fake.buffer(24, GPUBufferUsage.STORAGE),
      byteLength: 24,
    })).rejects.toMatchObject({
      code: "BG-WEBGPU-VIEW-COPY-INVALID-BINDING",
      path: "$.prepared",
    });
    expect(fake.submitCount()).toBe(0);
  });

  it.each([
    ["declared length", (fake: FakeGpuControl): SemanticViewCopyResidentSource => ({ buffer: fake.buffer(24, GPUBufferUsage.STORAGE), byteLength: 20 }), "$.source.byteLength"],
    ["physical size", (fake: FakeGpuControl): SemanticViewCopyResidentSource => ({ buffer: fake.buffer(20, GPUBufferUsage.STORAGE), byteLength: 24 }), "$.source.buffer.size"],
    ["storage usage", (fake: FakeGpuControl): SemanticViewCopyResidentSource => ({ buffer: fake.buffer(24, GPUBufferUsage.COPY_SRC), byteLength: 24 }), "$.source.buffer.usage"],
  ] as const)("rejects a resident source with invalid %s", async (_label, source, path) => {
    const fake = createFakeGpu();
    const device = await createDevice({ device: fake.device });
    await expect(runPreparedSemanticViewCopyResident(device, prepared, source(fake)))
      .rejects.toMatchObject({ code: "BG-WEBGPU-VIEW-COPY-INVALID-BINDING", path });
    expect(fake.submitCount()).toBe(0);
  });

  it("refuses to allocate an uninitialized partial destination root", async () => {
    const fake = createFakeGpu();
    const device = await createDevice({ device: fake.device });
    await expect(runPreparedSemanticViewCopyResident(device, offsetDestination, {
      buffer: fake.buffer(24, GPUBufferUsage.STORAGE),
      byteLength: 24,
    })).rejects.toMatchObject({
      code: "BG-WEBGPU-VIEW-COPY-UNSUPPORTED-PROFILE",
      path: "$.destination",
    });
    expect(fake.submitCount()).toBe(0);
  });

  it("applies device allocation limits before dispatch", async () => {
    const fake = createFakeGpu({ maxBufferSize: 16 });
    const device = await createDevice({ device: fake.device });
    await expect(runPreparedSemanticViewCopyResident(device, prepared, {
      buffer: fake.buffer(24, GPUBufferUsage.STORAGE),
      byteLength: 24,
    })).rejects.toMatchObject({
      code: "BG-WEBGPU-VIEW-COPY-DEVICE-LIMIT",
      path: "$.device.limits.maxBufferSize",
    });
    expect(fake.submitCount()).toBe(0);
  });
});

interface FakeGpuControl {
  readonly device: GPUDevice;
  readonly shaderSources: string[];
  readonly dispatches: Array<[number, number, number]>;
  readonly scopeEvents: string[];
  buffer(size: number, usage: GPUBufferUsageFlags): GPUBuffer;
  submitCount(): number;
  writeCount(): number;
  copyCount(): number;
  destroyCount(): number;
  settleErrorScopes(scope?: GPUErrorFilter, message?: string): void;
  setMaterializeMapFailure(message?: string): void;
}

function createFakeGpu(options: {
  readonly maxBufferSize?: number;
  readonly deferredErrorScopes?: boolean;
} = {}): FakeGpuControl {
  const shaderSources: string[] = [];
  const dispatches: Array<[number, number, number]> = [];
  let submits = 0;
  let writes = 0;
  let copies = 0;
  let destroys = 0;
  let materializeMapFailure: string | undefined;
  const scopeStack: GPUErrorFilter[] = [];
  const scopeEvents: string[] = [];
  const pendingScopes: Array<{
    readonly scope: GPUErrorFilter;
    readonly resolve: (error: GPUError | null) => void;
  }> = [];
  const buffer = (size: number, usage: GPUBufferUsageFlags) => ({
    size,
    usage,
    destroy: () => { destroys += 1; },
    mapAsync: () => materializeMapFailure === undefined
      ? Promise.resolve()
      : Promise.reject(new Error(materializeMapFailure)),
    getMappedRange: () => new ArrayBuffer(size),
    unmap: () => undefined,
  } as unknown as GPUBuffer);
  const device = {
    features: new Set<string>(),
    limits: {
      maxBufferSize: options.maxBufferSize ?? (1 << 20),
      maxStorageBufferBindingSize: 1 << 20,
      maxComputeWorkgroupsPerDimension: 65_535,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxBindingsPerBindGroup: 8,
      maxStorageBuffersPerShaderStage: 8,
    },
    lost: new Promise<GPUDeviceLostInfo>(() => undefined),
    pushErrorScope: (scope: GPUErrorFilter) => {
      scopeEvents.push(`push:${scope}`);
      scopeStack.push(scope);
    },
    popErrorScope: () => {
      const scope = scopeStack.pop();
      if (scope === undefined) return Promise.reject(new Error("scope stack underflow"));
      scopeEvents.push(`pop:${scope}`);
      if (!options.deferredErrorScopes) return Promise.resolve(null);
      return new Promise<GPUError | null>((resolve) => pendingScopes.push({ scope, resolve }));
    },
    queue: {
      writeBuffer: () => { writes += 1; },
      submit: () => { submits += 1; },
      onSubmittedWorkDone: async () => undefined,
    },
    createBuffer: (descriptor: GPUBufferDescriptor) => buffer(Number(descriptor.size), descriptor.usage),
    createBindGroupLayout: () => ({} as GPUBindGroupLayout),
    createPipelineLayout: () => ({} as GPUPipelineLayout),
    createShaderModule: ({ code }: GPUShaderModuleDescriptor) => {
      shaderSources.push(code);
      return {} as GPUShaderModule;
    },
    createComputePipeline: () => ({} as GPUComputePipeline),
    createBindGroup: () => ({} as GPUBindGroup),
    createCommandEncoder: () => ({
      beginComputePass: () => ({
        setPipeline: () => undefined,
        setBindGroup: () => undefined,
        dispatchWorkgroups: (x: number, y: number, z: number) => { dispatches.push([x, y, z]); },
        end: () => undefined,
      }),
      copyBufferToBuffer: () => { copies += 1; },
      finish: () => ({} as GPUCommandBuffer),
    } as unknown as GPUCommandEncoder),
  } as unknown as GPUDevice;
  return {
    device,
    shaderSources,
    dispatches,
    scopeEvents,
    buffer,
    submitCount: () => submits,
    writeCount: () => writes,
    copyCount: () => copies,
    destroyCount: () => destroys,
    settleErrorScopes: (failedScope, message = "GPU error") => {
      for (const pending of pendingScopes.splice(0)) {
        pending.resolve(pending.scope === failedScope
          ? { message } as GPUError
          : null);
      }
    },
    setMaterializeMapFailure: (message) => {
      materializeMapFailure = message;
    },
  };
}

function permutationRequests() {
  return {
    schema: "browsergrad.jit.tensor-plan-semantic-requests",
    version: { major: 1, minor: 0 },
    requests: [{
      kind: "dense-permutation-view-copy",
      valueId: 1,
      inputShape: ["2", "3"],
      axes: [1, 0],
      dtype: "f32",
    }],
  };
}

function permutationPlan() {
  return {
    steps: [
      { step: 0, value_id: 0, op: "BUFFER", input_ids: [], shape: [2, 3], dtype: "float32", arg: "buffer:x" },
      { step: 1, value_id: 1, op: "PERMUTE", input_ids: [0], shape: [3, 2], dtype: "float32", arg: null },
    ],
    buffers: [
      { value_id: 0, op: "BUFFER", shape: [2, 3], dtype: "float32", bytes: 24, first_step: 0, last_step: 1, materialize: false },
      { value_id: 1, op: "PERMUTE", shape: [3, 2], dtype: "float32", bytes: 24, first_step: 1, last_step: 1, materialize: true },
    ],
    root_id: 1,
    materialization_boundary: "root",
    peak_live_bytes: 48,
    has_custom_ops: false,
  };
}
