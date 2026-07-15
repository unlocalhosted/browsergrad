import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import {
  createVerifiedDensePermutationViewCopyArtifacts,
  createVerifiedViewCopyArtifacts,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import { parseWireI64, type WireI64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { createDevice } from "../src/device";
import {
  prepareSemanticViewCopyWgsl,
  runPreparedSemanticViewCopyResident,
  type PreparedSemanticViewCopyWgsl,
  type SemanticViewCopyResidentSource,
} from "../src/semantic_view_copy";

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

    const result = runPreparedSemanticViewCopyResident(device, prepared, {
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

  it("rejects forged prepared objects before touching the device", async () => {
    const fake = createFakeGpu();
    const device = await createDevice({ device: fake.device });
    const forged = { ...prepared } as PreparedSemanticViewCopyWgsl;

    expect(() => runPreparedSemanticViewCopyResident(device, forged, {
      buffer: fake.buffer(24, GPUBufferUsage.STORAGE),
      byteLength: 24,
    })).toThrow(expect.objectContaining({
      code: "BG-WEBGPU-VIEW-COPY-INVALID-BINDING",
      path: "$.prepared",
    }));
    expect(fake.submitCount()).toBe(0);
  });

  it.each([
    ["declared length", (fake: FakeGpuControl): SemanticViewCopyResidentSource => ({ buffer: fake.buffer(24, GPUBufferUsage.STORAGE), byteLength: 20 }), "$.source.byteLength"],
    ["physical size", (fake: FakeGpuControl): SemanticViewCopyResidentSource => ({ buffer: fake.buffer(20, GPUBufferUsage.STORAGE), byteLength: 24 }), "$.source.buffer.size"],
    ["storage usage", (fake: FakeGpuControl): SemanticViewCopyResidentSource => ({ buffer: fake.buffer(24, GPUBufferUsage.COPY_SRC), byteLength: 24 }), "$.source.buffer.usage"],
  ] as const)("rejects a resident source with invalid %s", async (_label, source, path) => {
    const fake = createFakeGpu();
    const device = await createDevice({ device: fake.device });
    expect(() => runPreparedSemanticViewCopyResident(device, prepared, source(fake)))
      .toThrow(expect.objectContaining({ code: "BG-WEBGPU-VIEW-COPY-INVALID-BINDING", path }));
    expect(fake.submitCount()).toBe(0);
  });

  it("refuses to allocate an uninitialized partial destination root", async () => {
    const fake = createFakeGpu();
    const device = await createDevice({ device: fake.device });
    expect(() => runPreparedSemanticViewCopyResident(device, offsetDestination, {
      buffer: fake.buffer(24, GPUBufferUsage.STORAGE),
      byteLength: 24,
    })).toThrow(expect.objectContaining({
      code: "BG-WEBGPU-VIEW-COPY-UNSUPPORTED-PROFILE",
      path: "$.destination",
    }));
    expect(fake.submitCount()).toBe(0);
  });

  it("applies device allocation limits before dispatch", async () => {
    const fake = createFakeGpu({ maxBufferSize: 16 });
    const device = await createDevice({ device: fake.device });
    expect(() => runPreparedSemanticViewCopyResident(device, prepared, {
      buffer: fake.buffer(24, GPUBufferUsage.STORAGE),
      byteLength: 24,
    })).toThrow(expect.objectContaining({
      code: "BG-WEBGPU-VIEW-COPY-DEVICE-LIMIT",
      path: "$.device.limits.maxBufferSize",
    }));
    expect(fake.submitCount()).toBe(0);
  });
});

interface FakeGpuControl {
  readonly device: GPUDevice;
  readonly shaderSources: string[];
  readonly dispatches: Array<[number, number, number]>;
  buffer(size: number, usage: GPUBufferUsageFlags): GPUBuffer;
  submitCount(): number;
  writeCount(): number;
  copyCount(): number;
}

function createFakeGpu(options: { readonly maxBufferSize?: number } = {}): FakeGpuControl {
  const shaderSources: string[] = [];
  const dispatches: Array<[number, number, number]> = [];
  let submits = 0;
  let writes = 0;
  let copies = 0;
  const buffer = (size: number, usage: GPUBufferUsageFlags) => ({
    size,
    usage,
    destroy: () => undefined,
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
    buffer,
    submitCount: () => submits,
    writeCount: () => writes,
    copyCount: () => copies,
  };
}
