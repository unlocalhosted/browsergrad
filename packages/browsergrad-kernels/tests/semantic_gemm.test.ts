import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  copyCertifiedLogicalGemmExactF32Inputs,
  createVerifiedLogicalGemmExactF32InputCertificate,
  createVerifiedDenseLogicalGemmTileArtifacts,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  createVerifiedLogicalGemmTileSchedule,
} from "@unlocalhosted/browsergrad-semantic-core/schedule";
import { parseWireU64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { createDevice } from "../src/device";
import {
  prepareSemanticGemmWgsl,
  runPreparedSemanticGemmResident,
  runSemanticGemmWebGpu,
  type PreparedSemanticGemmWgsl,
} from "../src/semantic_gemm";
import type { KernelDevice } from "../src/types";

const wire = (value: number) => parseWireU64(String(value));

beforeEach(() => {
  vi.stubGlobal("GPUBufferUsage", { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, MAP_READ: 8 });
  vi.stubGlobal("GPUMapMode", { READ: 1 });
  vi.stubGlobal("GPUShaderStage", { COMPUTE: 1 });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

async function logical(m = 17, n = 19, k = 23) {
  return createVerifiedDenseLogicalGemmTileArtifacts({
    m: wire(m),
    n: wire(n),
    k: wire(k),
    logicalTile: { m: wire(16), n: wire(16), k: wire(16) },
  });
}

describe("semantic GEMM WebGPU preparation", () => {
  it("lowers one logical GEMM through distinct verified physical schedules", async () => {
    const semantics = await logical();
    const schedule8 = await createVerifiedLogicalGemmTileSchedule(semantics.kernel, {
      physicalTile: { m: wire(8), n: wire(8), k: wire(8) },
    });
    const schedule16 = await createVerifiedLogicalGemmTileSchedule(semantics.kernel, {
      physicalTile: { m: wire(16), n: wire(16), k: wire(16) },
    });

    const prepared8 = await prepareSemanticGemmWgsl(
      semantics.layout,
      semantics.kernel,
      schedule8.artifact,
      { operationId: semantics.operationId },
    );
    const repeated8 = await prepareSemanticGemmWgsl(
      semantics.layout,
      semantics.kernel,
      schedule8.artifact,
      { operationId: semantics.operationId },
    );
    const prepared16 = await prepareSemanticGemmWgsl(
      semantics.layout,
      semantics.kernel,
      schedule16.artifact,
      { operationId: semantics.operationId },
    );

    expect(prepared8.semantic.specializationHash).toBe(prepared16.semantic.specializationHash);
    expect(prepared8.scheduled.scheduleSemanticHash).not.toBe(prepared16.scheduled.scheduleSemanticHash);
    expect(prepared8.backendPreparationHash).toBe(repeated8.backendPreparationHash);
    expect(prepared8.wgslModuleHash).toBe(repeated8.wgslModuleHash);
    expect(prepared8.wgslModuleHash).not.toBe(prepared16.wgslModuleHash);
    expect(prepared8.program.workgroupSize).toEqual([8, 8, 1]);
    expect(prepared16.program.workgroupSize).toEqual([16, 16, 1]);
    expect(prepared8.launch.dispatchCount).toEqual([24, 24, 1]);
    expect(prepared16.launch.dispatchCount).toEqual([32, 32, 1]);
    expect(prepared8.workgroupStorageBytes).toBe("512");
    expect(prepared16.workgroupStorageBytes).toBe("2048");
  });

  it("emits cooperative zero-filled staging with uniform barriers and masked stores", async () => {
    const semantics = await logical();
    const schedule = await createVerifiedLogicalGemmTileSchedule(semantics.kernel, {
      physicalTile: { m: wire(8), n: wire(8), k: wire(8) },
    });
    const prepared = await prepareSemanticGemmWgsl(
      semantics.layout,
      semantics.kernel,
      schedule.artifact,
      { operationId: semantics.operationId },
    );
    const wgsl = prepared.program.wgsl;

    expect(wgsl).toContain("var<workgroup> lhs_tile: array<f32, 64>");
    expect(wgsl).toContain("var<workgroup> rhs_tile: array<f32, 64>");
    expect(wgsl).toContain("var value: f32 = 0.0;");
    expect(wgsl).toContain("if (global_row < 17u && tile_inner < (23u - tile_k_base))");
    expect(wgsl).toContain("if (tile_inner < (23u - tile_k_base) && global_column < 19u)");
    expect(wgsl).toContain("if (output_row < 17u && output_column < 19u)");
    expect(wgsl.match(/workgroupBarrier\(\);/gu)).toHaveLength(2);
    expect(wgsl).not.toContain("return;");

    const firstBarrier = wgsl.indexOf("workgroupBarrier();");
    const accumulation = wgsl.indexOf("accumulator = accumulator +");
    const secondBarrier = wgsl.lastIndexOf("workgroupBarrier();");
    const storeMask = wgsl.indexOf("if (output_row <");
    expect(firstBarrier).toBeLessThan(accumulation);
    expect(accumulation).toBeLessThan(secondBarrier);
    expect(secondBarrier).toBeLessThan(storeMask);
  });

  it("binds schedule identity and enforces preparation budgets", async () => {
    const first = await logical();
    const second = await logical(18, 19, 23);
    const firstSchedule = await createVerifiedLogicalGemmTileSchedule(first.kernel, {
      physicalTile: { m: wire(16), n: wire(16), k: wire(16) },
    });

    await expect(prepareSemanticGemmWgsl(
      second.layout,
      second.kernel,
      firstSchedule.artifact,
      { operationId: second.operationId },
    )).rejects.toMatchObject({
      diagnostic: {
        code: "BG-SCHEDULE-KERNEL-HASH-MISMATCH",
        path: "$.schedule.logicalGemmSemanticHash",
      },
    });

    await expect(prepareSemanticGemmWgsl(
      first.layout,
      first.kernel,
      firstSchedule.artifact,
      { operationId: first.operationId, maxWorkgroupInvocations: 255 },
    )).rejects.toMatchObject({ diagnostic: { code: "BG-SCHEDULE-RESOURCE-LIMIT" } });

    await expect(prepareSemanticGemmWgsl(
      first.layout,
      first.kernel,
      firstSchedule.artifact,
      { operationId: first.operationId, maxWorkgroupStorageBytes: 1024 },
    )).rejects.toMatchObject({ diagnostic: { code: "BG-SCHEDULE-RESOURCE-LIMIT" } });

    await expect(prepareSemanticGemmWgsl(
      first.layout,
      first.kernel,
      firstSchedule.artifact,
      { operationId: first.operationId, maxWgslBytes: 1 },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-RESOURCE-LIMIT",
      path: "$.maxWgslBytes",
    });

    await expect(prepareSemanticGemmWgsl(
      first.layout,
      first.kernel,
      firstSchedule.artifact,
      { operationId: first.operationId, maxTransientWorkingSetBytes: 1024 },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-RESOURCE-LIMIT",
      path: "$.maxTransientWorkingSetBytes",
    });
  });

  it("fails resident execution closed before touching unproven GPU buffers", async () => {
    const semantics = await logical();
    const schedule = await createVerifiedLogicalGemmTileSchedule(semantics.kernel, {
      physicalTile: { m: wire(8), n: wire(8), k: wire(8) },
    });
    const prepared = await prepareSemanticGemmWgsl(
      semantics.layout,
      semantics.kernel,
      schedule.artifact,
      { operationId: semantics.operationId },
    );
    await expect(runPreparedSemanticGemmResident(
      {} as KernelDevice,
      prepared,
      {} as never,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-NUMERICAL-PROOF",
      path: "$.inputs",
    });

    await expect(runPreparedSemanticGemmResident(
      {} as KernelDevice,
      { ...prepared } as PreparedSemanticGemmWgsl,
      {} as never,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-INVALID-BINDING",
      path: "$.prepared",
    });
  });

  it("authorizes only authority-retained certified host bytes", async () => {
    const semantics = await logical();
    const schedule = await createVerifiedLogicalGemmTileSchedule(semantics.kernel, {
      physicalTile: { m: wire(8), n: wire(8), k: wire(8) },
    });
    const prepared = await prepareSemanticGemmWgsl(
      semantics.layout,
      semantics.kernel,
      schedule.artifact,
      { operationId: semantics.operationId },
    );
    const lhs = filledF32Bytes(17 * 23, 1);
    const rhs = filledF32Bytes(23 * 19, 1);
    const certified = await createVerifiedLogicalGemmExactF32InputCertificate(
      semantics.layout,
      semantics.kernel,
      { operationId: semantics.operationId, inputs: { lhs, rhs } },
    );
    new DataView(lhs.buffer).setUint32(0, 0x7fc0_0000, true);
    new DataView(rhs.buffer).setUint32(0, 0x8000_0000, true);
    const retained = copyCertifiedLogicalGemmExactF32Inputs(certified.certificate);
    expect(new DataView(retained.lhs.buffer).getFloat32(0, true)).toBe(1);
    expect(new DataView(retained.rhs.buffer).getFloat32(0, true)).toBe(1);

    await expect(runSemanticGemmWebGpu(
      fakeDevice({ maxBufferSize: 1 }),
      prepared,
      certified.certificate,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-DEVICE-LIMIT",
      path: "$.device.limits.maxBufferSize",
    });
  });

  it("rejects an exact-input certificate for a different specialization", async () => {
    const first = await logical();
    const second = await logical(18, 19, 23);
    const schedule = await createVerifiedLogicalGemmTileSchedule(first.kernel, {
      physicalTile: { m: wire(8), n: wire(8), k: wire(8) },
    });
    const prepared = await prepareSemanticGemmWgsl(
      first.layout,
      first.kernel,
      schedule.artifact,
      { operationId: first.operationId },
    );
    const foreign = await createVerifiedLogicalGemmExactF32InputCertificate(
      second.layout,
      second.kernel,
      {
        operationId: second.operationId,
        inputs: {
          lhs: filledF32Bytes(18 * 23, 1),
          rhs: filledF32Bytes(23 * 19, 1),
        },
      },
    );
    await expect(runSemanticGemmWebGpu(
      fakeDevice(),
      prepared,
      foreign.certificate,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-NUMERICAL-PROOF",
      path: "$.certificate.logicalGemmSemanticHash",
    });
  });

  it("rejects copied preparation and closed-record accessors before device effects", async () => {
    const fixture = await executionFixture();
    let deviceTouched = false;
    const untouchedDevice = {
      get gpu() {
        deviceTouched = true;
        throw new Error("device must not be touched");
      },
    } as unknown as KernelDevice;

    await expect(runSemanticGemmWebGpu(
      untouchedDevice,
      { ...fixture.prepared } as PreparedSemanticGemmWgsl,
      fixture.certificate,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-INVALID-BINDING",
      path: "$.prepared",
    });
    expect(deviceTouched).toBe(false);

    let getterRead = false;
    const accessorOptions = Object.defineProperty({}, "timeoutMs", {
      enumerable: true,
      get() {
        getterRead = true;
        return 1;
      },
    });
    await expect(runSemanticGemmWebGpu(
      untouchedDevice,
      fixture.prepared,
      fixture.certificate,
      accessorOptions,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-INVALID-BINDING",
      path: "$.options.timeoutMs",
    });
    expect(getterRead).toBe(false);
    expect(deviceTouched).toBe(false);

    await expect(runSemanticGemmWebGpu(
      untouchedDevice,
      fixture.prepared,
      fixture.certificate,
      { unexpected: true } as never,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-INVALID-BINDING",
      path: "$.options",
    });
    expect(deviceTouched).toBe(false);
  });

  it("rejects preparation accessors and unknown fields without invoking them", async () => {
    const semantics = await logical();
    const schedule = await createVerifiedLogicalGemmTileSchedule(semantics.kernel, {
      physicalTile: { m: wire(8), n: wire(8), k: wire(8) },
    });
    let getterRead = false;
    const request = Object.defineProperties({}, {
      operationId: { enumerable: true, value: semantics.operationId },
      maxWgslBytes: {
        enumerable: true,
        get() {
          getterRead = true;
          return 1024;
        },
      },
    });
    await expect(prepareSemanticGemmWgsl(
      semantics.layout,
      semantics.kernel,
      schedule.artifact,
      request as never,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-INVALID-BINDING",
      path: "$.request.maxWgslBytes",
    });
    expect(getterRead).toBe(false);

    await expect(prepareSemanticGemmWgsl(
      semantics.layout,
      semantics.kernel,
      schedule.artifact,
      { operationId: semantics.operationId, backend: "webgpu" } as never,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-INVALID-BINDING",
      path: "$.request",
    });
  });

  it("honors native pre-cancellation before reading the device or certificate", async () => {
    const fixture = await executionFixture();
    const controller = new AbortController();
    controller.abort();
    let deviceTouched = false;
    const untouchedDevice = {
      get gpu() {
        deviceTouched = true;
        throw new Error("device must not be touched");
      },
    } as unknown as KernelDevice;
    await expect(runSemanticGemmWebGpu(
      untouchedDevice,
      fixture.prepared,
      fixture.certificate,
      { signal: controller.signal },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-CANCELLED",
      path: "$.signal",
    });
    expect(deviceTouched).toBe(false);

    await expect(runSemanticGemmWebGpu(
      untouchedDevice,
      fixture.prepared,
      fixture.certificate,
      { signal: {} as AbortSignal },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-INVALID-BINDING",
      path: "$.options.signal",
    });
    expect(deviceTouched).toBe(false);
  });

  it.each([
    ["storage binding bytes", { maxStorageBufferBindingSize: 1 }, "$.device.limits.maxStorageBufferBindingSize"],
    ["workgroup invocations", { maxComputeInvocationsPerWorkgroup: 63 }, "$.device.limits.workgroupSize"],
    ["workgroup x", { maxComputeWorkgroupSizeX: 7 }, "$.device.limits.workgroupSize"],
    ["workgroup y", { maxComputeWorkgroupSizeY: 7 }, "$.device.limits.workgroupSize"],
    ["workgroup storage", { maxComputeWorkgroupStorageSize: 511 }, "$.device.limits.maxComputeWorkgroupStorageSize"],
    ["bind group bindings", { maxBindingsPerBindGroup: 2 }, "$.device.limits.bindings"],
    ["storage bindings", { maxStorageBuffersPerShaderStage: 2 }, "$.device.limits.bindings"],
    ["dispatch axes", { maxComputeWorkgroupsPerDimension: 2 }, "$.launch"],
  ] as const)("rejects device %s before GPU allocation", async (_label, overrides, path) => {
    const fixture = await executionFixture();
    await expect(runSemanticGemmWebGpu(
      fakeDevice(overrides),
      fixture.prepared,
      fixture.certificate,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-DEVICE-LIMIT",
      path,
    });
  });

  it("classifies scoped validation failures and destroys prepared upload buffers", async () => {
    const fixture = await executionFixture();
    const fake = await scopedPreparationDevice({ validationMessage: "bad binding" });
    await expect(runSemanticGemmWebGpu(
      fake.device,
      fixture.prepared,
      fixture.certificate,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-VALIDATION",
      path: "$.pipeline",
    });
    expect(fake.destroyed()).toBe(3);
  });

  it("times out a pending pipeline without admitting a second in-flight run", async () => {
    const fixture = await executionFixture();
    const fake = await scopedPreparationDevice({ pendingPipeline: true });
    await expect(runSemanticGemmWebGpu(
      fake.device,
      fixture.prepared,
      fixture.certificate,
      { timeoutMs: 1 },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-TIMEOUT",
      path: "$.timeoutMs",
    });
    await expect(runSemanticGemmWebGpu(
      fake.device,
      fixture.prepared,
      fixture.certificate,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-RESOURCE-LIMIT",
      path: "$.device.inFlight",
    });
  });

  it("reports device loss while pipeline preparation is pending", async () => {
    const fixture = await executionFixture();
    const fake = await scopedPreparationDevice({
      pendingPipeline: true,
      lost: Promise.resolve({ reason: "destroyed", message: "gone" } as GPUDeviceLostInfo),
    });
    await expect(runSemanticGemmWebGpu(
      fake.device,
      fixture.prepared,
      fixture.certificate,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-GEMM-DEVICE-LOST",
      path: "$.device",
    });
  });
});

async function executionFixture() {
  const semantics = await logical();
  const schedule = await createVerifiedLogicalGemmTileSchedule(semantics.kernel, {
    physicalTile: { m: wire(8), n: wire(8), k: wire(8) },
  });
  const prepared = await prepareSemanticGemmWgsl(
    semantics.layout,
    semantics.kernel,
    schedule.artifact,
    { operationId: semantics.operationId },
  );
  const certified = await createVerifiedLogicalGemmExactF32InputCertificate(
    semantics.layout,
    semantics.kernel,
    {
      operationId: semantics.operationId,
      inputs: {
        lhs: filledF32Bytes(17 * 23, 1),
        rhs: filledF32Bytes(23 * 19, 1),
      },
    },
  );
  return { prepared, certificate: certified.certificate };
}

function filledF32Bytes(length: number, value: number): Uint8Array {
  const values = new Float32Array(length);
  values.fill(value);
  return new Uint8Array(values.buffer);
}

function fakeDevice(
  overrides: Partial<Record<string, number>> = {},
): KernelDevice {
  const limits = {
    maxBufferSize: 1 << 20,
    maxStorageBufferBindingSize: 1 << 20,
    maxComputeWorkgroupsPerDimension: 65_535,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxComputeWorkgroupSizeY: 256,
    maxComputeWorkgroupStorageSize: 16 * 1024,
    maxBindingsPerBindGroup: 8,
    maxStorageBuffersPerShaderStage: 8,
    ...overrides,
  };
  return {
    gpu: {
      limits,
      features: new Set<string>(),
      lost: new Promise<GPUDeviceLostInfo>(() => undefined),
    } as unknown as GPUDevice,
    getStats: () => ({
      pipelineCacheSize: 0,
      pipelineCacheHits: 0,
      pipelineCacheMisses: 0,
      kernelInvocations: 0,
      outputBufferPoolBuffers: 0,
      outputBufferPoolBytes: 0,
      outputBufferPoolHits: 0,
      outputBufferPoolMisses: 0,
    }),
    clearCache: () => undefined,
  };
}

async function scopedPreparationDevice(options: Readonly<{
  validationMessage?: string;
  pendingPipeline?: boolean;
  lost?: Promise<GPUDeviceLostInfo>;
}> = {}) {
  let destroyed = 0;
  let popIndex = 0;
  const pending = new Promise<never>(() => undefined);
  const gpu = {
    limits: {
      maxBufferSize: 1 << 20,
      maxStorageBufferBindingSize: 1 << 20,
      maxComputeWorkgroupsPerDimension: 65_535,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 256,
      maxComputeWorkgroupStorageSize: 16 * 1024,
      maxBindingsPerBindGroup: 8,
      maxStorageBuffersPerShaderStage: 8,
    },
    features: new Set<string>(),
    lost: options.lost ?? new Promise<GPUDeviceLostInfo>(() => undefined),
    pushErrorScope: () => undefined,
    popErrorScope: () => {
      const index = popIndex++;
      return Promise.resolve(index === 0 && options.validationMessage !== undefined
        ? { message: options.validationMessage } as GPUError
        : null);
    },
    createBuffer: (descriptor: GPUBufferDescriptor) => ({
      size: Number(descriptor.size),
      usage: descriptor.usage,
      destroy: () => { destroyed += 1; },
    } as GPUBuffer),
    queue: {
      writeBuffer: () => undefined,
    },
    createBindGroupLayout: () => ({} as GPUBindGroupLayout),
    createPipelineLayout: () => ({} as GPUPipelineLayout),
    createShaderModule: () => ({
      getCompilationInfo: () => Promise.resolve({ messages: [] }),
    } as unknown as GPUShaderModule),
    createComputePipelineAsync: () => options.pendingPipeline === true
      ? pending
      : Promise.resolve({} as GPUComputePipeline),
    createBindGroup: () => ({} as GPUBindGroup),
  } as unknown as GPUDevice;
  const device = await createDevice({ device: gpu });
  return { device, destroyed: () => destroyed };
}
