import { describe, expect, it } from "vitest";

import {
  createVerifiedDenseAttentionForwardArtifacts,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  createVerifiedAttentionOnlineKvTileSchedule,
} from "@unlocalhosted/browsergrad-semantic-core/schedule";
import { parseWireU64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  SEMANTIC_ATTENTION_WEBGPU_PROFILE,
  prepareSemanticAttentionWgsl,
  runSemanticAttentionWebGpu,
  type PreparedSemanticAttentionWgsl,
} from "../src/semantic_attention";
import type { KernelDevice } from "../src/types";

const wire = (value: number) => parseWireU64(String(value));

async function attention(causal = false, queryLength = 17) {
  return createVerifiedDenseAttentionForwardArtifacts({
    batch: wire(2),
    heads: wire(3),
    queryLength: wire(queryLength),
    keyLength: wire(23),
    queryDepth: wire(16),
    valueDepth: wire(12),
    causal,
  });
}

describe("semantic attention WebGPU preparation", () => {
  it("lowers one logical attention operation through distinct verified schedules", async () => {
    const semantics = await attention();
    const schedule8 = await createVerifiedAttentionOnlineKvTileSchedule(semantics.kernel, {
      physicalTile: { queryRows: wire(8), keyRows: wire(8) },
    });
    const schedule16 = await createVerifiedAttentionOnlineKvTileSchedule(semantics.kernel, {
      physicalTile: { queryRows: wire(8), keyRows: wire(16) },
    });

    const prepared8 = await prepareSemanticAttentionWgsl(
      semantics.layout,
      semantics.kernel,
      schedule8.artifact,
      { operationId: semantics.operationId },
    );
    const repeated8 = await prepareSemanticAttentionWgsl(
      semantics.layout,
      semantics.kernel,
      schedule8.artifact,
      { operationId: semantics.operationId },
    );
    const prepared16 = await prepareSemanticAttentionWgsl(
      semantics.layout,
      semantics.kernel,
      schedule16.artifact,
      { operationId: semantics.operationId },
    );

    expect(prepared8.semantic.specializationHash).toBe(prepared16.semantic.specializationHash);
    expect(prepared8.scheduled.scheduleSemanticHash).not.toBe(
      prepared16.scheduled.scheduleSemanticHash,
    );
    expect(prepared8.backendPreparationHash).toBe(repeated8.backendPreparationHash);
    expect(prepared8.wgslModuleHash).toBe(repeated8.wgslModuleHash);
    expect(prepared8.wgslModuleHash).not.toBe(prepared16.wgslModuleHash);
    expect(prepared8.backendProfile).toBe(SEMANTIC_ATTENTION_WEBGPU_PROFILE);
    expect(prepared8.algorithmProfile).toBe("block-tiled-kv-online-softmax-forward");
    expect(prepared8.preservationLevel).toBe("portable-relegalized");
    expect(prepared8.program.workgroupSize).toEqual([8, 1, 1]);
    expect(prepared16.program.workgroupSize).toEqual([8, 1, 1]);
    expect(prepared8.launch.dispatchCount).toEqual([24, 3, 2]);
    expect(prepared16.launch.dispatchCount).toEqual([24, 3, 2]);
    expect(prepared8.workgroupStorageBytes).toBe("896");
    expect(prepared16.workgroupStorageBytes).toBe("1792");
  });

  it("emits cooperative K/V staging, online recurrence, uniform barriers, and hard masks", async () => {
    const semantics = await attention(true);
    const schedule = await createVerifiedAttentionOnlineKvTileSchedule(semantics.kernel, {
      physicalTile: { queryRows: wire(8), keyRows: wire(8) },
    });
    const prepared = await prepareSemanticAttentionWgsl(
      semantics.layout,
      semantics.kernel,
      schedule.artifact,
      { operationId: semantics.operationId },
    );
    const wgsl = prepared.program.wgsl;

    expect(wgsl).toContain("var<workgroup> key_tile: array<f32, 128>");
    expect(wgsl).toContain("var<workgroup> value_tile: array<f32, 96>");
    expect(wgsl).toContain("var query_private: array<f32, 16>");
    expect(wgsl).toContain("var output_private: array<f32, 12>");
    expect(wgsl).toContain("let valid_key: bool = global_key < 23u && global_key <= query_index;");
    expect(wgsl).toContain("prior_rescale = exp(running_maximum - next_maximum);");
    expect(wgsl).toContain("let weight: f32 = exp(score - next_maximum);");
    expect(wgsl).toContain("destination_values[destination_address] = output_private[value_index] / denominator;");
    expect(wgsl.match(/workgroupBarrier\(\);/gu)).toHaveLength(2);
    expect(wgsl).not.toContain("return;");
    expect(wgsl).not.toContain("rowwise_online_attention");

    const firstBarrier = wgsl.indexOf("workgroupBarrier();");
    const stateUpdate = wgsl.indexOf("denominator *= prior_rescale;");
    const secondBarrier = wgsl.lastIndexOf("workgroupBarrier();");
    const store = wgsl.indexOf("destination_values[destination_address]");
    expect(firstBarrier).toBeLessThan(stateUpdate);
    expect(stateUpdate).toBeLessThan(secondBarrier);
    expect(secondBarrier).toBeLessThan(store);
  });

  it("keeps causal meaning in the semantic hash and out of the physical schedule", async () => {
    const nonCausal = await attention(false);
    const causal = await attention(true);
    const request = { physicalTile: { queryRows: wire(8), keyRows: wire(16) } };
    const [nonCausalSchedule, causalSchedule] = await Promise.all([
      createVerifiedAttentionOnlineKvTileSchedule(nonCausal.kernel, request),
      createVerifiedAttentionOnlineKvTileSchedule(causal.kernel, request),
    ]);
    const [nonCausalPrepared, causalPrepared] = await Promise.all([
      prepareSemanticAttentionWgsl(
        nonCausal.layout,
        nonCausal.kernel,
        nonCausalSchedule.artifact,
        { operationId: nonCausal.operationId },
      ),
      prepareSemanticAttentionWgsl(
        causal.layout,
        causal.kernel,
        causalSchedule.artifact,
        { operationId: causal.operationId },
      ),
    ]);

    expect(nonCausalPrepared.scheduled.schedule).toEqual(causalPrepared.scheduled.schedule);
    expect(nonCausalPrepared.semantic.kernelSemanticHash).not.toBe(
      causalPrepared.semantic.kernelSemanticHash,
    );
    expect(nonCausalPrepared.program.wgsl).toContain(
      "let valid_key: bool = global_key < 23u;",
    );
    expect(nonCausalPrepared.program.wgsl).not.toContain("global_key <= query_index");
    expect(causalPrepared.program.wgsl).toContain("global_key <= query_index");
  });

  it("rejects schedule mismatches, resource excess, accessors, and unknown backend fields", async () => {
    const first = await attention(false);
    const second = await attention(false, 18);
    const schedule = await createVerifiedAttentionOnlineKvTileSchedule(first.kernel, {
      physicalTile: { queryRows: wire(8), keyRows: wire(16) },
    });

    await expect(prepareSemanticAttentionWgsl(
      second.layout,
      second.kernel,
      schedule.artifact,
      { operationId: second.operationId },
    )).rejects.toMatchObject({
      diagnostic: {
        code: "BG-SCHEDULE-KERNEL-HASH-MISMATCH",
        path: "$.schedule.attentionForwardSemanticHash",
      },
    });
    await expect(prepareSemanticAttentionWgsl(
      first.layout,
      first.kernel,
      schedule.artifact,
      { operationId: first.operationId, maxWorkgroupStorageBytes: 1024 },
    )).rejects.toMatchObject({ diagnostic: { code: "BG-SCHEDULE-RESOURCE-LIMIT" } });
    await expect(prepareSemanticAttentionWgsl(
      first.layout,
      first.kernel,
      schedule.artifact,
      { operationId: first.operationId, maxWgslBytes: 1 },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-ATTENTION-RESOURCE-LIMIT",
      path: "$.maxWgslBytes",
    });
    await expect(prepareSemanticAttentionWgsl(
      first.layout,
      first.kernel,
      schedule.artifact,
      { operationId: first.operationId, maxTransientWorkingSetBytes: 1024 },
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-ATTENTION-RESOURCE-LIMIT",
      path: "$.maxTransientWorkingSetBytes",
    });

    let getterRead = false;
    const accessorRequest = Object.defineProperties({}, {
      operationId: { enumerable: true, value: first.operationId },
      maxWgslBytes: {
        enumerable: true,
        get() {
          getterRead = true;
          return 1024;
        },
      },
    });
    await expect(prepareSemanticAttentionWgsl(
      first.layout,
      first.kernel,
      schedule.artifact,
      accessorRequest as never,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-ATTENTION-INVALID-BINDING",
      path: "$.request.maxWgslBytes",
    });
    expect(getterRead).toBe(false);
    await expect(prepareSemanticAttentionWgsl(
      first.layout,
      first.kernel,
      schedule.artifact,
      { operationId: first.operationId, backend: "webgpu" } as never,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-ATTENTION-INVALID-BINDING",
      path: "$.request",
    });
  });

  it("rejects copied plans, hostile bindings, and non-finite inputs before device effects", async () => {
    const fixture = await executionFixture();
    let deviceTouched = false;
    const untouchedDevice = {
      get gpu() {
        deviceTouched = true;
        throw new Error("device must not be touched");
      },
    } as unknown as KernelDevice;

    await expect(runSemanticAttentionWebGpu(
      untouchedDevice,
      { ...fixture.prepared } as PreparedSemanticAttentionWgsl,
      fixture.inputs,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-ATTENTION-INVALID-BINDING",
      path: "$.prepared",
    });
    expect(deviceTouched).toBe(false);

    let getterRead = false;
    const accessorInputs = Object.defineProperties({}, {
      query: { enumerable: true, value: fixture.inputs.query },
      key: {
        enumerable: true,
        get() {
          getterRead = true;
          return fixture.inputs.key;
        },
      },
      value: { enumerable: true, value: fixture.inputs.value },
    });
    await expect(runSemanticAttentionWebGpu(
      untouchedDevice,
      fixture.prepared,
      accessorInputs as never,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-ATTENTION-INVALID-BINDING",
      path: "$.inputs.key",
    });
    expect(getterRead).toBe(false);
    expect(deviceTouched).toBe(false);

    const nonFinite = {
      ...fixture.inputs,
      query: new Uint8Array(fixture.inputs.query),
    };
    new DataView(nonFinite.query.buffer).setFloat32(0, Number.NaN, true);
    await expect(runSemanticAttentionWebGpu(
      untouchedDevice,
      fixture.prepared,
      nonFinite,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-ATTENTION-NUMERICAL-DOMAIN",
      path: "$.inputs.query",
    });
    expect(deviceTouched).toBe(false);
  });

  it("honors cancellation and validates native options before inputs or device access", async () => {
    const fixture = await executionFixture();
    const controller = new AbortController();
    controller.abort();
    let inputRead = false;
    const untouchedInputs = Object.defineProperty({}, "query", {
      enumerable: true,
      get() {
        inputRead = true;
        throw new Error("inputs must not be read");
      },
    });
    let deviceTouched = false;
    const untouchedDevice = {
      get gpu() {
        deviceTouched = true;
        throw new Error("device must not be touched");
      },
    } as unknown as KernelDevice;
    await expect(runSemanticAttentionWebGpu(
      untouchedDevice,
      fixture.prepared,
      untouchedInputs as never,
      { signal: controller.signal },
    )).rejects.toMatchObject({ code: "BG-WEBGPU-ATTENTION-CANCELLED" });
    expect(inputRead).toBe(false);
    expect(deviceTouched).toBe(false);

    let optionRead = false;
    const accessorOptions = Object.defineProperty({}, "timeoutMs", {
      enumerable: true,
      get() {
        optionRead = true;
        return 1;
      },
    });
    await expect(runSemanticAttentionWebGpu(
      untouchedDevice,
      fixture.prepared,
      fixture.inputs,
      accessorOptions,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-ATTENTION-INVALID-BINDING",
      path: "$.options.timeoutMs",
    });
    expect(optionRead).toBe(false);
    expect(deviceTouched).toBe(false);
  });

  it("checks device storage limits after retaining finite input snapshots", async () => {
    const fixture = await executionFixture();
    const lost = new Promise<GPUDeviceLostInfo>(() => undefined);
    const gpu = {
      features: new Set<string>(),
      limits: {
        maxBufferSize: 64,
        maxStorageBufferBindingSize: 64,
        maxComputeWorkgroupsPerDimension: 65_535,
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
        maxComputeWorkgroupStorageSize: 16_384,
        maxBindingsPerBindGroup: 8,
        maxStorageBuffersPerShaderStage: 8,
      },
      lost,
    } as unknown as GPUDevice;
    const device = {
      gpu,
      clearCache() {},
    } as unknown as KernelDevice;

    await expect(runSemanticAttentionWebGpu(
      device,
      fixture.prepared,
      fixture.inputs,
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-ATTENTION-DEVICE-LIMIT",
      path: "$.device.limits.maxBufferSize",
    });
  });
});

async function executionFixture() {
  const semantics = await createVerifiedDenseAttentionForwardArtifacts({
    batch: wire(1),
    heads: wire(1),
    queryLength: wire(3),
    keyLength: wire(5),
    queryDepth: wire(4),
    valueDepth: wire(6),
    causal: true,
  });
  const schedule = await createVerifiedAttentionOnlineKvTileSchedule(semantics.kernel, {
    physicalTile: { queryRows: wire(4), keyRows: wire(4) },
  });
  const prepared = await prepareSemanticAttentionWgsl(
    semantics.layout,
    semantics.kernel,
    schedule.artifact,
    { operationId: semantics.operationId },
  );
  return {
    prepared,
    inputs: {
      query: filledF32Bytes(12, 0.25),
      key: filledF32Bytes(20, -0.5),
      value: filledF32Bytes(30, 0.75),
    },
  };
}

function filledF32Bytes(length: number, value: number): Uint8Array {
  const array = new Float32Array(length);
  array.fill(value);
  return new Uint8Array(array.buffer);
}
