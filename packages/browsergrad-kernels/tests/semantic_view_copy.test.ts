import { describe, expect, it } from "vitest";

import {
  layoutArtifactPayload,
  verifyLayoutArtifact,
  type DimExpr,
  type IndexExpr,
  type PredicateExpr,
  type VerifiedLayoutArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  kernelArtifactPayload,
  prepareViewCopyCpu,
  verifyKernelArtifact,
  type InvalidSourcePolicy,
  type VerifiedKernelArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  hashSemanticArtifact,
  parseWireI64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  SemanticViewCopyWebGpuError,
  prepareSemanticViewCopyWgsl,
  runSemanticViewCopyWebGpu,
  type PrepareSemanticViewCopyWgslRequest,
} from "../src/semantic_view_copy";
import type { KernelDevice } from "../src/types";

const TRUE: PredicateExpr = { kind: "bool", value: true };
const FALSE: PredicateExpr = { kind: "bool", value: false };

interface LayoutInput {
  readonly shape: readonly DimExpr[];
  readonly sourceLocation: IndexExpr;
  readonly sourcePredicate?: PredicateExpr;
  readonly destinationLocation?: IndexExpr;
  readonly sourceLocationUnit?: "element" | "byte";
  readonly destinationLocationUnit?: "element" | "byte";
  readonly sourceByteOffset?: DimExpr;
  readonly destinationByteOffset?: DimExpr;
  readonly sourceBytes: DimExpr;
  readonly destinationBytes: DimExpr;
  readonly symbols?: readonly { readonly id: string; readonly domain: { readonly min: string; readonly max: string } }[];
  readonly dtype?: "f32" | "i32" | "u32";
}

async function verifiedLayout(input: LayoutInput): Promise<VerifiedLayoutArtifact> {
  return verifyLayoutArtifact(JSON.parse(JSON.stringify({
    schema: "browsergrad.layout",
    version: { major: 1, minor: 0 },
    producer: { id: "kernels-view-copy-tests", version: "1" },
    artifactId: "layout",
    requiredExtensions: [],
    payload: {
      symbols: input.symbols ?? [],
      constraints: [],
      allocations: [
        {
          allocationId: "sourceAllocation",
          byteLength: input.sourceBytes,
          memorySpace: { kind: "global" },
          alignmentBytes: 4,
          aliasSetId: "sourceAlias",
        },
        {
          allocationId: "destinationAllocation",
          byteLength: input.destinationBytes,
          memorySpace: { kind: "global" },
          alignmentBytes: 4,
          aliasSetId: "destinationAlias",
        },
      ],
      indexMaps: [
        {
          indexMapId: "sourceMap",
          coordinateRank: input.shape.length,
          locationUnit: input.sourceLocationUnit ?? "element",
          location: input.sourceLocation,
          inBounds: input.sourcePredicate ?? TRUE,
        },
        {
          indexMapId: "destinationMap",
          coordinateRank: input.shape.length,
          locationUnit: input.destinationLocationUnit ?? "element",
          location: input.destinationLocation ?? rowMajor(input.shape),
          inBounds: TRUE,
        },
      ],
      views: [
        {
          viewId: "sourceView",
          allocationId: "sourceAllocation",
          dtype: input.dtype ?? "f32",
          byteOffset: input.sourceByteOffset ?? constant("0"),
          shape: input.shape,
          indexMapId: "sourceMap",
          requiredAlignmentBytes: 4,
        },
        {
          viewId: "destinationView",
          allocationId: "destinationAllocation",
          dtype: input.dtype ?? "f32",
          byteOffset: input.destinationByteOffset ?? constant("0"),
          shape: input.shape,
          indexMapId: "destinationMap",
          requiredAlignmentBytes: 4,
        },
      ],
    },
  })));
}

async function verifiedKernel(
  layout: VerifiedLayoutArtifact,
  invalidSource: InvalidSourcePolicy = { kind: "reject" },
): Promise<VerifiedKernelArtifact> {
  const payload = layoutArtifactPayload(layout);
  const source = payload.views[0];
  const destination = payload.views[1];
  if (source === undefined || destination === undefined) throw new Error("fixture views missing");
  return verifyKernelArtifact({
    schema: "browsergrad.kernel",
    version: { major: 1, minor: 0 },
    producer: { id: "kernels-view-copy-tests", version: "1" },
    artifactId: "kernel",
    requiredExtensions: [],
    payload: {
      layoutSemanticHash: await hashSemanticArtifact(layout),
      operations: [{
        operationId: "copy",
        kind: "view-copy",
        version: { major: 1, minor: 0 },
        dtype: source.dtype,
        source: { viewId: source.viewId, access: "read", invalidSource },
        destination: { viewId: destination.viewId, access: "write" },
        overlap: { kind: "forbid" },
      }],
    },
  }, { layout });
}

async function prepare(
  layout: VerifiedLayoutArtifact,
  kernel: VerifiedKernelArtifact,
  request: Omit<PrepareSemanticViewCopyWgslRequest, "operationId"> = {},
) {
  const operation = kernelArtifactPayload(kernel).operations[0];
  if (operation === undefined) throw new Error("fixture operation missing");
  return prepareSemanticViewCopyWgsl(layout, kernel, { operationId: operation.operationId, ...request });
}

describe("semantic view-copy WGSL lowering", () => {
  it("lowers rank-2 transpose from the canonical maps and shares the CPU specialization hash", async () => {
    const shape = [constant("2"), constant("3")] as const;
    const layout = await verifiedLayout({
      shape,
      sourceLocation: add(multiply(coordinate(1), constant("2")), coordinate(0)),
      sourceBytes: constant("24"),
      destinationBytes: constant("24"),
    });
    const kernel = await verifiedKernel(layout);
    const wgsl = await prepare(layout, kernel);
    const cpu = await prepareViewCopyCpu(layout, kernel, {
      operationId: kernelArtifactPayload(kernel).operations[0]!.operationId,
    });

    expect(wgsl.semantic.specializationHash).toBe(cpu.specializationHash);
    expect(wgsl.program.wgsl).toContain("var<storage, read> source_words: array<u32>");
    expect(wgsl.program.wgsl).toContain("coordinate_1 * 2i");
    expect(wgsl.program.wgsl).toContain("destination_words[destination_word] = copied_bits");
    expect(wgsl.program.wgsl).not.toContain("select(");
    expect(wgsl.launch.dispatchCount).toEqual([6, 1, 1]);
    expect(wgsl.wgslModuleHash).toMatch(/^[0-9a-f]{64}$/u);
    expect(wgsl).toMatchObject({
      plannedTransientGpuBytes: "72",
      plannedTransientHostBytes: "24",
      plannedTransientWorkingSetBytes: "96",
    });
    expect(Object.isFrozen(wgsl.program)).toBe(true);
    expect(Object.isFrozen(wgsl.program.bindings)).toBe(true);
    expect(Object.isFrozen(wgsl.program.bindings[0])).toBe(true);
    expect(Object.isFrozen(wgsl.program.workgroupSize)).toBe(true);
    expect(Object.isFrozen(wgsl.launch.dispatchCount)).toBe(true);
  });

  it("lowers i32 and u32 through the same bit-exact word backend", async () => {
    const modules = new Map<string, string>();
    for (const dtype of ["i32", "u32"] as const) {
      const shape = [constant("2"), constant("3")] as const;
      const layout = await verifiedLayout({
        shape,
        sourceLocation: add(
          multiply(coordinate(1), constant("2")),
          coordinate(0),
        ),
        sourceBytes: constant("24"),
        destinationBytes: constant("24"),
        dtype,
      });
      const prepared = await prepare(layout, await verifiedKernel(layout));
      expect(prepared.semantic.portableProfile).toMatchObject({
        profileId: "browsergrad.view-copy.positive-affine-word32@1",
        dtype,
      });
      expect(prepared.program.wgsl)
        .toContain("destination_words[destination_word] = copied_bits");
      modules.set(dtype, prepared.program.wgsl);
    }
    expect(modules.get("i32")).toBe(modules.get("u32"));
  });

  it("keeps signed padding arithmetic and exact fill bits inside a structured guard", async () => {
    const shape = [constant("4"), constant("5")] as const;
    const sourceLocation = add(
      multiply(add(coordinate(0), constant("-1")), constant("3")),
      add(coordinate(1), constant("-1")),
    );
    const predicate: PredicateExpr = {
      kind: "and",
      values: [
        { kind: "lessEqual", lhs: constant("1"), rhs: coordinate(0) },
        { kind: "lessEqual", lhs: coordinate(0), rhs: constant("2") },
        { kind: "lessEqual", lhs: constant("1"), rhs: coordinate(1) },
        { kind: "lessEqual", lhs: coordinate(1), rhs: constant("3") },
      ],
    };
    const layout = await verifiedLayout({
      shape,
      sourceLocation,
      sourcePredicate: predicate,
      sourceBytes: constant("24"),
      destinationBytes: constant("80"),
    });
    const kernel = await verifiedKernel(layout, {
      kind: "fill",
      value: { kind: "float-bits", dtype: "f32", bits: "7fc01234" },
    });
    const prepared = await prepare(layout, kernel);
    const source = prepared.program.wgsl;

    expect(prepared.sourceLocationRange).toEqual({ minimum: -4n, maximum: 9n });
    expect(source).toContain("var copied_bits: u32 = 0x7fc01234u;");
    expect(source).toContain("if (");
    expect(source.indexOf("let source_word")).toBeGreaterThan(source.indexOf("if ("));
    expect(source).not.toContain("select(");
  });

  it("keeps one backend shape for rank-3 permutation, slice, broadcast, offsets, and byte maps", async () => {
    const rank3 = [constant("2"), constant("3"), constant("4")] as const;
    const permutation = await verifiedLayout({
      shape: rank3,
      sourceLocation: add(
        multiply(coordinate(2), constant("6")),
        multiply(coordinate(0), constant("3")),
        coordinate(1),
      ),
      sourceBytes: constant("96"),
      destinationBytes: constant("96"),
    });
    expect((await prepare(permutation, await verifiedKernel(permutation))).launch.dispatchCount[0]).toBe(24);

    const shape = [constant("2"), constant("2")] as const;
    const slice = await verifiedLayout({
      shape,
      sourceLocation: multiply(rowMajor(shape), constant("2")),
      sourceBytes: constant("28"),
      destinationBytes: constant("16"),
    });
    expect((await prepare(slice, await verifiedKernel(slice))).program.wgsl).toContain("* 2i");

    const broadcast = await verifiedLayout({
      shape,
      sourceLocation: coordinate(1),
      sourceBytes: constant("8"),
      destinationBytes: constant("16"),
    });
    expect((await prepare(broadcast, await verifiedKernel(broadcast))).sourceLocationRange).toEqual({ minimum: 0n, maximum: 1n });

    const byteMap = await verifiedLayout({
      shape,
      sourceLocation: multiply(rowMajor(shape), constant("4")),
      destinationLocation: multiply(rowMajor(shape), constant("4")),
      sourceLocationUnit: "byte",
      destinationLocationUnit: "byte",
      sourceByteOffset: constant("4"),
      destinationByteOffset: constant("4"),
      sourceBytes: constant("20"),
      destinationBytes: constant("20"),
    });
    const bytePrepared = await prepare(byteMap, await verifiedKernel(byteMap));
    expect(bytePrepared.program.wgsl).toContain("/ 4i");
  });

  it("derives binding-sensitive modules and rejects i32 or source-size overflow", async () => {
    const n = symbol("n");
    const shape = [n, constant("2")] as const;
    const bytes = multiplyDim(n, constant("8"));
    const layout = await verifiedLayout({
      shape,
      sourceLocation: rowMajor(shape),
      sourceBytes: bytes,
      destinationBytes: bytes,
      symbols: [{ id: "n", domain: { min: "0", max: "8" } }],
    });
    const kernel = await verifiedKernel(layout);
    const two = await prepare(layout, kernel, { bindings: { n: parseWireI64("2") } });
    const three = await prepare(layout, kernel, { bindings: { n: parseWireI64("3") } });
    expect(two.semantic.specializationHash).not.toBe(three.semantic.specializationHash);
    expect(two.wgslModuleHash).not.toBe(three.wgslModuleHash);

    const overflow = await verifiedLayout({
      shape: [constant("3"), constant("2")],
      sourceLocation: multiply(coordinate(0), constant("1073741824")),
      sourcePredicate: FALSE,
      sourceBytes: constant("4"),
      destinationBytes: constant("24"),
    });
    await expect(prepare(overflow, await verifiedKernel(overflow, {
      kind: "fill",
      value: { kind: "float-bits", dtype: "f32", bits: "00000000" },
    }))).rejects.toMatchObject({ code: "BG-WEBGPU-VIEW-COPY-UNSUPPORTED-PROFILE" });

    await expect(prepare(layout, kernel, {
      bindings: { n: parseWireI64("2") },
      maxWgslBytes: 1,
    })).rejects.toMatchObject({ code: "BG-WEBGPU-VIEW-COPY-RESOURCE-LIMIT" });

    await expect(prepare(layout, kernel, {
      bindings: { n: parseWireI64("2") },
      maxTransientWorkingSetBytes: 63,
    })).rejects.toMatchObject({
      code: "BG-WEBGPU-VIEW-COPY-RESOURCE-LIMIT",
      path: "$.maxTransientWorkingSetBytes",
    });
  });

  it("emits the exact WGSL i32 minimum without an overflowing signed literal", async () => {
    const shape = [constant("1"), constant("1")] as const;
    const layout = await verifiedLayout({
      shape,
      sourceLocation: constant("-2147483648"),
      sourceLocationUnit: "byte",
      sourcePredicate: FALSE,
      sourceBytes: constant("4"),
      destinationBytes: constant("4"),
    });
    const prepared = await prepare(layout, await verifiedKernel(layout, {
      kind: "fill",
      value: { kind: "float-bits", dtype: "f32", bits: "7fc01234" },
    }));
    expect(prepared.program.wgsl).toContain("bitcast<i32>(0x80000000u)");
    expect(prepared.program.wgsl).not.toContain("-2147483648i");
  });

  it("handles zero elements without submission and fails closed on runtime buffers and limits", async () => {
    const n = symbol("n");
    const shape = [n, constant("2")] as const;
    const bytes = multiplyDim(n, constant("8"));
    const layout = await verifiedLayout({
      shape,
      sourceLocation: rowMajor(shape),
      sourceBytes: bytes,
      destinationBytes: bytes,
      symbols: [{ id: "n", domain: { min: "0", max: "8" } }],
    });
    const prepared = await prepare(layout, await verifiedKernel(layout), {
      bindings: { n: parseWireI64("0") },
    });
    const result = await runSemanticViewCopyWebGpu(fakeDevice(), prepared, {
      sourceWords: new Uint32Array(0),
      destinationWords: new Uint32Array(0),
    });
    expect(result.trace).toMatchObject({ submitted: false, elementCount: "0" });
    expect(prepared.program.wgsl).toContain("let coordinate_0: i32 = 0i;");
    expect(prepared.program.wgsl).not.toContain("/ 0u");

    const cancelled = new AbortController();
    cancelled.abort();
    await expect(runSemanticViewCopyWebGpu(fakeDevice(), prepared, {
      sourceWords: new Uint32Array(0),
      destinationWords: new Uint32Array(0),
    }, { signal: cancelled.signal })).rejects.toMatchObject({
      code: "BG-WEBGPU-VIEW-COPY-CANCELLED",
    });

    await expect(runSemanticViewCopyWebGpu(fakeDevice(), { ...prepared }, {
      sourceWords: new Uint32Array(0),
      destinationWords: new Uint32Array(0),
    })).rejects.toMatchObject({
      code: "BG-WEBGPU-VIEW-COPY-INVALID-BINDING",
      path: "$.prepared",
    });

    const nonemptyLayout = await verifiedLayout({
      shape: [constant("2"), constant("2")],
      sourceLocation: rowMajor([constant("2"), constant("2")]),
      sourceBytes: constant("16"),
      destinationBytes: constant("16"),
    });
    const nonempty = await prepare(nonemptyLayout, await verifiedKernel(nonemptyLayout));
    await expect(runSemanticViewCopyWebGpu(fakeDevice(), nonempty, {
      sourceWords: new Uint32Array(3),
      destinationWords: new Uint32Array(4),
    })).rejects.toMatchObject({ code: "BG-WEBGPU-VIEW-COPY-INVALID-BINDING" });

    const overlappingBacking = new ArrayBuffer(32);
    await expect(runSemanticViewCopyWebGpu(fakeDevice(), nonempty, {
      sourceWords: new Uint32Array(overlappingBacking, 0, 4),
      destinationWords: new Uint32Array(overlappingBacking, 8, 4),
    })).rejects.toMatchObject({ code: "BG-WEBGPU-VIEW-COPY-INVALID-BINDING" });

    if (typeof SharedArrayBuffer !== "undefined") {
      await expect(runSemanticViewCopyWebGpu(fakeDevice(), nonempty, {
        sourceWords: new Uint32Array(new SharedArrayBuffer(16)),
        destinationWords: new Uint32Array(4),
      })).rejects.toMatchObject({ code: "BG-WEBGPU-VIEW-COPY-INVALID-BINDING" });
    }

    const spoofed = new Uint32Array(3);
    Object.defineProperty(spoofed, "byteLength", { get: () => 16 });
    await expect(runSemanticViewCopyWebGpu(fakeDevice(), nonempty, {
      sourceWords: spoofed,
      destinationWords: new Uint32Array(4),
    })).rejects.toMatchObject({ code: "BG-WEBGPU-VIEW-COPY-INVALID-BINDING" });

    await expect(runSemanticViewCopyWebGpu(fakeDevice({ maxStorageBufferBindingSize: 8 }), nonempty, {
      sourceWords: new Uint32Array(4),
      destinationWords: new Uint32Array(4),
    })).rejects.toMatchObject({ code: "BG-WEBGPU-VIEW-COPY-DEVICE-LIMIT" });
  });

  it("uses typed backend errors", () => {
    const error = new SemanticViewCopyWebGpuError("BG-WEBGPU-VIEW-COPY-EXECUTION", "$.dispatch", "failed");
    expect(error).toMatchObject({ name: "SemanticViewCopyWebGpuError", path: "$.dispatch" });
  });
});

function fakeDevice(limitOverrides: Partial<Record<string, number>> = {}): KernelDevice {
  const limits = {
    maxBufferSize: 1 << 20,
    maxStorageBufferBindingSize: 1 << 20,
    maxComputeWorkgroupsPerDimension: 65_535,
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
    maxBindingsPerBindGroup: 8,
    maxStorageBuffersPerShaderStage: 8,
    ...limitOverrides,
  };
  return {
    gpu: { limits, features: new Set<string>(), lost: new Promise<GPUDeviceLostInfo>(() => undefined) } as unknown as GPUDevice,
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

function constant(value: string): DimExpr & IndexExpr {
  return { kind: "const", value: parseWireI64(value) };
}

function symbol(id: string): DimExpr {
  return { kind: "symbol", id };
}

function coordinate(axis: number): IndexExpr {
  return { kind: "coordinate", axis };
}

function add(...terms: readonly IndexExpr[]): IndexExpr {
  return { kind: "add", terms };
}

function multiply(lhs: IndexExpr, rhs: IndexExpr): IndexExpr {
  return { kind: "mul", lhs, rhs };
}

function multiplyDim(lhs: DimExpr, rhs: DimExpr): DimExpr {
  return { kind: "mul", lhs, rhs };
}

function rowMajor(shape: readonly DimExpr[]): IndexExpr {
  let result: IndexExpr = coordinate(0);
  for (let axis = 1; axis < shape.length; axis += 1) {
    result = add(multiply(result, dimAsIndex(shape[axis] as DimExpr)), coordinate(axis));
  }
  return result;
}

function dimAsIndex(value: DimExpr): IndexExpr {
  if (value.kind === "const") return { kind: "const", value: value.value };
  if (value.kind === "symbol") return { kind: "dimension", symbolId: value.id };
  throw new Error("fixture row-major helper supports const/symbol dimensions only");
}
