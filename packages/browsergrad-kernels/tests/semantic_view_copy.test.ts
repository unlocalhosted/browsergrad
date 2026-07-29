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
import { prepareSemanticViewCopyDynamicWgsl } from "../src/semantic_view_copy_internal";
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
  readonly dtype?:
    | "bool"
    | "i8"
    | "u8"
    | "i16"
    | "u16"
    | "f16"
    | "bf16"
    | "f32"
    | "i32"
    | "u32"
    | "f64"
    | "i64"
    | "u64";
}

async function verifiedLayout(input: LayoutInput): Promise<VerifiedLayoutArtifact> {
  const dtype = input.dtype ?? "f32";
  const requiredAlignmentBytes =
    dtype === "bool" ||
    dtype === "i8" ||
    dtype === "u8"
      ? 1
      : dtype === "i16" ||
    dtype === "u16" ||
    dtype === "f16" ||
    dtype === "bf16"
      ? 2
      : dtype === "f64" ||
          dtype === "i64" ||
          dtype === "u64"
        ? 8
      : 4;
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
          alignmentBytes: Math.max(4, requiredAlignmentBytes),
          aliasSetId: "sourceAlias",
        },
        {
          allocationId: "destinationAllocation",
          byteLength: input.destinationBytes,
          memorySpace: { kind: "global" },
          alignmentBytes: Math.max(4, requiredAlignmentBytes),
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
          dtype,
          byteOffset: input.sourceByteOffset ?? constant("0"),
          shape: input.shape,
          indexMapId: "sourceMap",
          requiredAlignmentBytes,
        },
        {
          viewId: "destinationView",
          allocationId: "destinationAllocation",
          dtype,
          byteOffset: input.destinationByteOffset ?? constant("0"),
          shape: input.shape,
          indexMapId: "destinationMap",
          requiredAlignmentBytes,
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

  it("lowers one runtime-prefix guard without changing semantic authority", async () => {
    const shape = [constant("65")] as const;
    const layout = await verifiedLayout({
      shape,
      sourceLocation: coordinate(0),
      sourceBytes: constant("260"),
      destinationBytes: constant("260"),
    });
    const kernel = await verifiedKernel(layout);
    const operation = kernelArtifactPayload(kernel).operations[0];
    if (operation === undefined) throw new Error("fixture operation missing");
    const dynamic = await prepareSemanticViewCopyDynamicWgsl(
      layout,
      kernel,
      {
        operationId: operation.operationId,
        workgroupSize: 64,
      },
      "linear-prefix",
    );
    const ordinary = await prepare(layout, kernel, {
      workgroupSize: 64,
    });

    expect(dynamic.semantic.specializationHash)
      .toBe(ordinary.semantic.specializationHash);
    expect(dynamic.wgslModuleHash).not.toBe(ordinary.wgslModuleHash);
    expect(dynamic.program.bindings).toContainEqual({
      kind: "uniform",
      name: dynamic.dynamicUniformName,
      byteLength: 4,
      binding: 2,
    });
    expect(dynamic.program.wgsl).toContain(
      "linear_index >= bg_dynamic_prefix.element_count",
    );
    expect(dynamic.launch.dispatchCount).toEqual([65, 1, 1]);
  });

  it("lowers rank-2 through rank-7 rectangular guards before semantic evaluation", async () => {
    for (const shape of [
      [constant("3"), constant("4")],
      [constant("2"), constant("3"), constant("4")],
      [constant("2"), constant("2"), constant("3"), constant("4")],
      [
        constant("2"),
        constant("2"),
        constant("2"),
        constant("3"),
        constant("4"),
      ],
      [
        constant("2"),
        constant("2"),
        constant("2"),
        constant("2"),
        constant("3"),
        constant("4"),
      ],
      [
        constant("2"),
        constant("2"),
        constant("2"),
        constant("2"),
        constant("2"),
        constant("3"),
        constant("4"),
      ],
    ] as const) {
      const elementCount = shape.length === 2
        ? 12
        : shape.length === 3
          ? 24
          : shape.length === 4
            ? 48
            : shape.length === 5
              ? 96
              : shape.length === 6
                ? 192
                : 384;
      const layout = await verifiedLayout({
        shape,
        sourceLocation: rowMajor(shape),
        sourceBytes: constant(String(elementCount * 4)),
        destinationBytes: constant(String(elementCount * 4)),
      });
      const kernel = await verifiedKernel(layout);
      const operation = kernelArtifactPayload(kernel).operations[0];
      if (operation === undefined) throw new Error("fixture operation missing");
      const dynamic = await prepareSemanticViewCopyDynamicWgsl(
        layout,
        kernel,
        {
          operationId: operation.operationId,
          workgroupSize: 64,
        },
        "rectangular-prefix",
      );

      expect(dynamic.program.bindings).toContainEqual({
        kind: "uniform",
        name: dynamic.dynamicUniformName,
        byteLength: shape.length >= 5 ? 32 : 16,
        binding: 2,
      });
      expect(dynamic.launch.dispatchCount).toEqual(
        shape.length === 2
          ? [4, 3, 1]
          : shape.length === 3
            ? [4, 3, 2]
            : shape.length === 4
              ? [4, 3, 4]
              : shape.length === 5
                ? [4, 3, 8]
                : shape.length === 6
                  ? [4, 3, 16]
                  : [4, 3, 32],
      );
      expect(dynamic.program.wgsl).toContain(
        "global_id.x >= bg_dynamic_region.extent_",
      );
      expect(dynamic.program.wgsl).toContain(
        "global_id.y >= bg_dynamic_region.extent_",
      );
      if (shape.length === 3) {
        expect(dynamic.program.wgsl).toContain(
          "global_id.z >= bg_dynamic_region.extent_0",
        );
      } else if (shape.length === 4) {
        expect(dynamic.program.wgsl).toContain(
          "bg_dynamic_region.extent_0 * bg_dynamic_region.extent_1",
        );
        expect(dynamic.program.wgsl).toContain(
          "global_id.z / bg_dynamic_region.extent_1",
        );
        expect(dynamic.program.wgsl).toContain(
          "global_id.z % bg_dynamic_region.extent_1",
        );
      } else if (shape.length === 5) {
        expect(dynamic.program.wgsl).toContain(
          "(bg_dynamic_region.extent_0 * bg_dynamic_region.extent_1 * bg_dynamic_region.extent_2)",
        );
        expect(dynamic.program.wgsl).toContain(
          "global_id.z / rank5_dynamic_stride_0",
        );
        expect(dynamic.program.wgsl).toContain(
          "rank5_dynamic_remainder_0 % bg_dynamic_region.extent_2",
        );
      } else if (shape.length === 6) {
        expect(dynamic.program.wgsl).toContain(
          "(bg_dynamic_region.extent_0 * bg_dynamic_region.extent_1 * bg_dynamic_region.extent_2 * bg_dynamic_region.extent_3)",
        );
        expect(dynamic.program.wgsl).toContain(
          "global_id.z / rank6_dynamic_stride_0",
        );
        expect(dynamic.program.wgsl).toContain(
          "rank6_dynamic_remainder_1 % bg_dynamic_region.extent_3",
        );
      } else if (shape.length === 7) {
        expect(dynamic.program.wgsl).toContain(
          "(bg_dynamic_region.extent_0 * bg_dynamic_region.extent_1 * bg_dynamic_region.extent_2 * bg_dynamic_region.extent_3 * bg_dynamic_region.extent_4)",
        );
        expect(dynamic.program.wgsl).toContain(
          "global_id.z / rank7_dynamic_stride_0",
        );
        expect(dynamic.program.wgsl).toContain(
          "rank7_dynamic_remainder_2 % bg_dynamic_region.extent_4",
        );
      }
      const guardIndex = dynamic.program.wgsl.indexOf(
        "global_id.x >= bg_dynamic_region.extent_",
      );
      const coordinateIndex = dynamic.program.wgsl.indexOf("let coordinate_0");
      expect(guardIndex).toBeGreaterThanOrEqual(0);
      expect(coordinateIndex).toBeGreaterThan(guardIndex);
      expect(dynamic.program.wgsl).not.toContain("let linear_index");
    }
  });

  it("rejects rectangular launch outside its admitted rank profile", async () => {
    for (const shape of [[constant("4")]] as const) {
      const layout = await verifiedLayout({
        shape,
        sourceLocation: rowMajor(shape),
        sourceBytes: constant("16"),
        destinationBytes: constant("16"),
      });
      const kernel = await verifiedKernel(layout);
      const operation = kernelArtifactPayload(kernel).operations[0];
      if (operation === undefined) throw new Error("fixture operation missing");

      await expect(prepareSemanticViewCopyDynamicWgsl(
        layout,
        kernel,
        { operationId: operation.operationId },
        "rectangular-prefix",
      )).rejects.toMatchObject({
        code: "BG-WEBGPU-VIEW-COPY-UNSUPPORTED-PROFILE",
      });
    }
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

  it("packs exact i16, u16, f16, and bf16 storage without shader-f16", async () => {
    const modules = new Map<string, string>();
    for (const dtype of ["i16", "u16", "f16", "bf16"] as const) {
      const shape = [constant("3"), constant("3")] as const;
      const layout = await verifiedLayout({
        shape,
        sourceLocation: add(
          multiply(coordinate(1), constant("3")),
          coordinate(0),
        ),
        sourceBytes: constant("20"),
        destinationBytes: constant("20"),
        dtype,
      });
      const prepared = await prepare(layout, await verifiedKernel(layout));

      expect(prepared.semantic.portableProfile).toMatchObject({
        profileId:
          "browsergrad.view-copy.positive-affine-rank2-rank3-packed16@1",
        rank: 2,
        dtype,
      });
      expect(prepared.backendProfile).toBe(
        "browsergrad.webgpu.view-copy.packed16@1",
      );
      expect(prepared.backendVersion).toBe("3.6.0");
      expect(prepared.launch.dispatchCount).toEqual([5, 1, 1]);
      expect(prepared.program.wgsl).toContain(
        "fn copy_packed_element(linear_index: u32)",
      );
      expect(prepared.program.wgsl).toContain(
        "let destination_mask: u32 = 0xffffu << destination_shift;",
      );
      expect(prepared.program.wgsl).toContain(
        "copy_packed_element(first_element_index + 1u);",
      );
      expect(prepared.program.wgsl).not.toContain("enable f16;");
      modules.set(dtype, prepared.program.wgsl);
    }
    expect(new Set(modules.values()).size).toBe(1);

    const misalignedDestination = await verifiedLayout({
      shape: [constant("2"), constant("2")],
      sourceLocation: rowMajor([constant("2"), constant("2")]),
      sourceBytes: constant("8"),
      destinationByteOffset: constant("2"),
      destinationBytes: constant("12"),
      dtype: "f16",
    });
    await expect(prepare(
      misalignedDestination,
      await verifiedKernel(misalignedDestination),
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-VIEW-COPY-UNSUPPORTED-PROFILE",
      path: "$.destination.viewByteOffset",
    });

    const dynamicLayout = await verifiedLayout({
      shape: [constant("2"), constant("2")],
      sourceLocation: rowMajor([constant("2"), constant("2")]),
      sourceBytes: constant("8"),
      destinationBytes: constant("8"),
      dtype: "bf16",
    });
    const dynamicKernel = await verifiedKernel(dynamicLayout);
    const operation = kernelArtifactPayload(dynamicKernel).operations[0];
    if (operation === undefined) throw new Error("fixture operation missing");
    await expect(prepareSemanticViewCopyDynamicWgsl(
      dynamicLayout,
      dynamicKernel,
      { operationId: operation.operationId },
      "linear-prefix",
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-VIEW-COPY-UNSUPPORTED-PROFILE",
      path: "$.launchMode",
    });
  });

  it("packs exact bool, i8, and u8 storage four elements per word", async () => {
    const modules = new Map<string, string>();
    for (const dtype of ["bool", "i8", "u8"] as const) {
      const shape = [constant("3"), constant("3")] as const;
      const layout = await verifiedLayout({
        shape,
        sourceLocation: add(
          multiply(coordinate(1), constant("3")),
          coordinate(0),
        ),
        sourceBytes: constant("12"),
        destinationBytes: constant("12"),
        dtype,
      });
      const prepared = await prepare(layout, await verifiedKernel(layout));

      expect(prepared.semantic.portableProfile).toMatchObject({
        profileId:
          "browsergrad.view-copy.positive-affine-rank2-rank3-packed8@1",
        rank: 2,
        dtype,
      });
      expect(prepared.backendProfile).toBe(
        "browsergrad.webgpu.view-copy.packed8@1",
      );
      expect(prepared.backendVersion).toBe("3.6.0");
      expect(prepared.launch.dispatchCount).toEqual([3, 1, 1]);
      expect(prepared.program.wgsl).toContain(
        "let first_element_index: u32 = global_id.x * 4u;",
      );
      expect(prepared.program.wgsl).toContain(
        "let destination_mask: u32 = 0xffu << destination_shift;",
      );
      expect(prepared.program.wgsl).toContain(
        "copy_packed_element(first_element_index + 3u);",
      );
      modules.set(dtype, prepared.program.wgsl);
    }
    expect(new Set(modules.values()).size).toBe(1);
  });

  it("copies exact f64, i64, and u64 storage as two raw words", async () => {
    const modules = new Map<string, string>();
    for (const dtype of ["f64", "i64", "u64"] as const) {
      const shape = [constant("2"), constant("3")] as const;
      const layout = await verifiedLayout({
        shape,
        sourceLocation: add(
          multiply(coordinate(1), constant("2")),
          coordinate(0),
        ),
        sourceBytes: constant("48"),
        destinationBytes: constant("48"),
        dtype,
      });
      const prepared = await prepare(layout, await verifiedKernel(layout));

      expect(prepared.semantic.portableProfile).toMatchObject({
        profileId:
          "browsergrad.view-copy.positive-affine-rank2-rank3-word64@1",
        rank: 2,
        dtype,
      });
      expect(prepared.backendProfile).toBe(
        "browsergrad.webgpu.view-copy.word64@1",
      );
      expect(prepared.backendVersion).toBe("3.6.0");
      expect(prepared.launch.dispatchCount).toEqual([6, 1, 1]);
      expect(prepared.program.wgsl).toContain(
        "destination_words[destination_word] = source_words[source_word];",
      );
      expect(prepared.program.wgsl).toContain(
        "destination_words[destination_word + 1u] = source_words[source_word + 1u];",
      );
      expect(prepared.program.wgsl).not.toContain("enable f64;");
      modules.set(dtype, prepared.program.wgsl);
    }
    expect(new Set(modules.values()).size).toBe(1);

    const dynamicLayout = await verifiedLayout({
      shape: [constant("2"), constant("2")],
      sourceLocation: rowMajor([constant("2"), constant("2")]),
      sourceBytes: constant("32"),
      destinationBytes: constant("32"),
      dtype: "f64",
    });
    const dynamicKernel = await verifiedKernel(dynamicLayout);
    const operation = kernelArtifactPayload(dynamicKernel).operations[0];
    if (operation === undefined) throw new Error("fixture operation missing");
    await expect(prepareSemanticViewCopyDynamicWgsl(
      dynamicLayout,
      dynamicKernel,
      { operationId: operation.operationId },
      "linear-prefix",
    )).rejects.toMatchObject({
      code: "BG-WEBGPU-VIEW-COPY-UNSUPPORTED-PROFILE",
      path: "$.launchMode",
    });
  });

  it("lowers rank-1 packed8, packed16, and word64 source addresses", async () => {
    const cases = [
      {
        dtype: "i8" as const,
        dtypeBytes: 1,
        positiveSourceBytes: 8,
        signedSourceBytes: 4,
        destinationBytes: 4,
        positiveProfile:
          "browsergrad.view-copy.positive-affine-rank1-packed8@1",
        signedProfile:
          "browsergrad.view-copy.signed-affine-rank1-packed8@1",
        backendProfile: "browsergrad.webgpu.view-copy.packed8@1",
        dispatchCount: 1,
      },
      {
        dtype: "bf16" as const,
        dtypeBytes: 2,
        positiveSourceBytes: 12,
        signedSourceBytes: 8,
        destinationBytes: 8,
        positiveProfile:
          "browsergrad.view-copy.positive-affine-rank1-packed16@1",
        signedProfile:
          "browsergrad.view-copy.signed-affine-rank1-packed16@1",
        backendProfile: "browsergrad.webgpu.view-copy.packed16@1",
        dispatchCount: 2,
      },
      {
        dtype: "f64" as const,
        dtypeBytes: 8,
        positiveSourceBytes: 40,
        signedSourceBytes: 24,
        destinationBytes: 24,
        positiveProfile:
          "browsergrad.view-copy.positive-affine-rank1-word64@1",
        signedProfile:
          "browsergrad.view-copy.signed-affine-rank1-word64@1",
        backendProfile: "browsergrad.webgpu.view-copy.word64@1",
        dispatchCount: 3,
      },
    ];
    for (const testCase of cases) {
      for (const signed of [false, true]) {
        const shape = [constant("3")] as const;
        const layout = await verifiedLayout({
          shape,
          sourceLocation: multiply(
            coordinate(0),
            constant(signed ? "-1" : "2"),
          ),
          ...(signed
            ? { sourceByteOffset: constant(String(testCase.dtypeBytes * 2)) }
            : {}),
          sourceBytes: constant(String(
            signed
              ? testCase.signedSourceBytes
              : testCase.positiveSourceBytes,
          )),
          destinationBytes: constant(String(testCase.destinationBytes)),
          dtype: testCase.dtype,
        });
        const prepared = await prepare(layout, await verifiedKernel(layout));

        expect(prepared.semantic.portableProfile).toMatchObject({
          profileId: signed
            ? testCase.signedProfile
            : testCase.positiveProfile,
          rank: 1,
          dtype: testCase.dtype,
        });
        expect(prepared.backendProfile).toBe(testCase.backendProfile);
        expect(prepared.backendVersion).toBe("3.6.0");
        expect(prepared.sourceLocationRange).toEqual(
          signed
            ? { minimum: -2n, maximum: 0n }
            : { minimum: 0n, maximum: 4n },
        );
        expect(prepared.launch.dispatchCount).toEqual([
          testCase.dispatchCount,
          1,
          1,
        ]);
        expect(prepared.program.wgsl).toContain(
          "let coordinate_0: i32 = i32(linear_index);",
        );
        if (signed) {
          expect(prepared.program.wgsl).toContain("* -1i");
        }
      }
    }
  });

  it("lowers rank-4 packed8, packed16, and word64 source addresses", async () => {
    const cases = [
      {
        dtype: "i8" as const,
        dtypeBytes: 1,
        positiveProfile:
          "browsergrad.view-copy.positive-affine-rank4-packed8@1",
        signedProfile:
          "browsergrad.view-copy.signed-affine-rank4-packed8@1",
        backendProfile: "browsergrad.webgpu.view-copy.packed8@1",
        dispatchCount: 4,
      },
      {
        dtype: "bf16" as const,
        dtypeBytes: 2,
        positiveProfile:
          "browsergrad.view-copy.positive-affine-rank4-packed16@1",
        signedProfile:
          "browsergrad.view-copy.signed-affine-rank4-packed16@1",
        backendProfile: "browsergrad.webgpu.view-copy.packed16@1",
        dispatchCount: 8,
      },
      {
        dtype: "f64" as const,
        dtypeBytes: 8,
        positiveProfile:
          "browsergrad.view-copy.positive-affine-rank4-word64@1",
        signedProfile:
          "browsergrad.view-copy.signed-affine-rank4-word64@1",
        backendProfile: "browsergrad.webgpu.view-copy.word64@1",
        dispatchCount: 16,
      },
    ];
    for (const testCase of cases) {
      for (const signed of [false, true]) {
        const shape = [
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
        ] as const;
        const layout = await verifiedLayout({
          shape,
          sourceLocation: signed
            ? add(
              multiply(coordinate(0), constant("-8")),
              multiply(coordinate(1), constant("-4")),
              multiply(coordinate(2), constant("-2")),
              multiply(coordinate(3), constant("-1")),
            )
            : add(
              coordinate(0),
              multiply(coordinate(1), constant("2")),
              multiply(coordinate(2), constant("4")),
              multiply(coordinate(3), constant("8")),
            ),
          ...(signed
            ? {
              sourceByteOffset: constant(String(testCase.dtypeBytes * 15)),
            }
            : {}),
          sourceBytes: constant(String(testCase.dtypeBytes * 16)),
          destinationBytes: constant(String(testCase.dtypeBytes * 16)),
          dtype: testCase.dtype,
        });
        const prepared = await prepare(layout, await verifiedKernel(layout));

        expect(prepared.semantic.portableProfile).toMatchObject({
          profileId: signed
            ? testCase.signedProfile
            : testCase.positiveProfile,
          rank: 4,
          dtype: testCase.dtype,
        });
        expect(prepared.backendProfile).toBe(testCase.backendProfile);
        expect(prepared.backendVersion).toBe("3.6.0");
        expect(prepared.sourceLocationRange).toEqual(
          signed
            ? { minimum: -15n, maximum: 0n }
            : { minimum: 0n, maximum: 15n },
        );
        expect(prepared.launch.dispatchCount).toEqual([
          testCase.dispatchCount,
          1,
          1,
        ]);
        expect(prepared.program.wgsl).toContain(
          "let coordinate_3: i32 = i32(inner_remainder % 2u);",
        );
        if (signed) {
          expect(prepared.program.wgsl).toContain("* -8i");
        }
      }
    }
  });

  it("lowers rank-5 packed8, packed16, and word64 source addresses", async () => {
    const cases = [
      {
        dtype: "i8" as const,
        dtypeBytes: 1,
        positiveProfile:
          "browsergrad.view-copy.positive-affine-rank5-packed8@1",
        signedProfile:
          "browsergrad.view-copy.signed-affine-rank5-packed8@1",
        backendProfile: "browsergrad.webgpu.view-copy.packed8@1",
        dispatchCount: 8,
      },
      {
        dtype: "bf16" as const,
        dtypeBytes: 2,
        positiveProfile:
          "browsergrad.view-copy.positive-affine-rank5-packed16@1",
        signedProfile:
          "browsergrad.view-copy.signed-affine-rank5-packed16@1",
        backendProfile: "browsergrad.webgpu.view-copy.packed16@1",
        dispatchCount: 16,
      },
      {
        dtype: "f64" as const,
        dtypeBytes: 8,
        positiveProfile:
          "browsergrad.view-copy.positive-affine-rank5-word64@1",
        signedProfile:
          "browsergrad.view-copy.signed-affine-rank5-word64@1",
        backendProfile: "browsergrad.webgpu.view-copy.word64@1",
        dispatchCount: 32,
      },
    ];
    for (const testCase of cases) {
      for (const signed of [false, true]) {
        const shape = [
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
        ] as const;
        const layout = await verifiedLayout({
          shape,
          sourceLocation: signed
            ? add(
              multiply(coordinate(0), constant("-16")),
              multiply(coordinate(1), constant("-8")),
              multiply(coordinate(2), constant("-4")),
              multiply(coordinate(3), constant("-2")),
              multiply(coordinate(4), constant("-1")),
            )
            : add(
              coordinate(0),
              multiply(coordinate(1), constant("2")),
              multiply(coordinate(2), constant("4")),
              multiply(coordinate(3), constant("8")),
              multiply(coordinate(4), constant("16")),
            ),
          ...(signed
            ? {
              sourceByteOffset: constant(String(testCase.dtypeBytes * 31)),
            }
            : {}),
          sourceBytes: constant(String(testCase.dtypeBytes * 32)),
          destinationBytes: constant(String(testCase.dtypeBytes * 32)),
          dtype: testCase.dtype,
        });
        const prepared = await prepare(layout, await verifiedKernel(layout));

        expect(prepared.semantic.portableProfile).toMatchObject({
          profileId: signed
            ? testCase.signedProfile
            : testCase.positiveProfile,
          rank: 5,
          dtype: testCase.dtype,
        });
        expect(prepared.backendProfile).toBe(testCase.backendProfile);
        expect(prepared.backendVersion).toBe("3.6.0");
        expect(prepared.sourceLocationRange).toEqual(
          signed
            ? { minimum: -31n, maximum: 0n }
            : { minimum: 0n, maximum: 31n },
        );
        expect(prepared.launch.dispatchCount).toEqual([
          testCase.dispatchCount,
          1,
          1,
        ]);
        expect(prepared.program.wgsl).toContain(
          "let coordinate_4: i32 = i32(rank5_remainder_2 % 2u);",
        );
        if (signed) {
          expect(prepared.program.wgsl).toContain("* -16i");
        }
      }
    }
  });

  it("lowers rank-6 packed8, packed16, and word64 source addresses", async () => {
    const cases = [
      {
        dtype: "i8" as const,
        dtypeBytes: 1,
        positiveProfile:
          "browsergrad.view-copy.positive-affine-rank6-packed8@1",
        signedProfile:
          "browsergrad.view-copy.signed-affine-rank6-packed8@1",
        backendProfile: "browsergrad.webgpu.view-copy.packed8@1",
        dispatchCount: 16,
      },
      {
        dtype: "bf16" as const,
        dtypeBytes: 2,
        positiveProfile:
          "browsergrad.view-copy.positive-affine-rank6-packed16@1",
        signedProfile:
          "browsergrad.view-copy.signed-affine-rank6-packed16@1",
        backendProfile: "browsergrad.webgpu.view-copy.packed16@1",
        dispatchCount: 32,
      },
      {
        dtype: "f64" as const,
        dtypeBytes: 8,
        positiveProfile:
          "browsergrad.view-copy.positive-affine-rank6-word64@1",
        signedProfile:
          "browsergrad.view-copy.signed-affine-rank6-word64@1",
        backendProfile: "browsergrad.webgpu.view-copy.word64@1",
        dispatchCount: 64,
      },
    ];
    for (const testCase of cases) {
      for (const signed of [false, true]) {
        const shape = [
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
        ] as const;
        const layout = await verifiedLayout({
          shape,
          sourceLocation: signed
            ? add(
              multiply(coordinate(0), constant("-32")),
              multiply(coordinate(1), constant("-16")),
              multiply(coordinate(2), constant("-8")),
              multiply(coordinate(3), constant("-4")),
              multiply(coordinate(4), constant("-2")),
              multiply(coordinate(5), constant("-1")),
            )
            : add(
              coordinate(0),
              multiply(coordinate(1), constant("2")),
              multiply(coordinate(2), constant("4")),
              multiply(coordinate(3), constant("8")),
              multiply(coordinate(4), constant("16")),
              multiply(coordinate(5), constant("32")),
            ),
          ...(signed
            ? {
              sourceByteOffset: constant(String(testCase.dtypeBytes * 63)),
            }
            : {}),
          sourceBytes: constant(String(testCase.dtypeBytes * 64)),
          destinationBytes: constant(String(testCase.dtypeBytes * 64)),
          dtype: testCase.dtype,
        });
        const prepared = await prepare(layout, await verifiedKernel(layout));

        expect(prepared.semantic.portableProfile).toMatchObject({
          profileId: signed
            ? testCase.signedProfile
            : testCase.positiveProfile,
          rank: 6,
          dtype: testCase.dtype,
        });
        expect(prepared.backendProfile).toBe(testCase.backendProfile);
        expect(prepared.backendVersion).toBe("3.6.0");
        expect(prepared.sourceLocationRange).toEqual(
          signed
            ? { minimum: -63n, maximum: 0n }
            : { minimum: 0n, maximum: 63n },
        );
        expect(prepared.launch.dispatchCount).toEqual([
          testCase.dispatchCount,
          1,
          1,
        ]);
        expect(prepared.program.wgsl).toContain(
          "let coordinate_5: i32 = i32(rank6_remainder_3 % 2u);",
        );
        if (signed) {
          expect(prepared.program.wgsl).toContain("* -32i");
        }
      }
    }
  });

  it("lowers rank-7 packed8, packed16, and word64 source addresses", async () => {
    const cases = [
      {
        dtype: "i8" as const,
        dtypeBytes: 1,
        positiveProfile:
          "browsergrad.view-copy.positive-affine-rank7-packed8@1",
        signedProfile:
          "browsergrad.view-copy.signed-affine-rank7-packed8@1",
        backendProfile: "browsergrad.webgpu.view-copy.packed8@1",
        dispatchCount: 32,
      },
      {
        dtype: "bf16" as const,
        dtypeBytes: 2,
        positiveProfile:
          "browsergrad.view-copy.positive-affine-rank7-packed16@1",
        signedProfile:
          "browsergrad.view-copy.signed-affine-rank7-packed16@1",
        backendProfile: "browsergrad.webgpu.view-copy.packed16@1",
        dispatchCount: 64,
      },
      {
        dtype: "f64" as const,
        dtypeBytes: 8,
        positiveProfile:
          "browsergrad.view-copy.positive-affine-rank7-word64@1",
        signedProfile:
          "browsergrad.view-copy.signed-affine-rank7-word64@1",
        backendProfile: "browsergrad.webgpu.view-copy.word64@1",
        dispatchCount: 128,
      },
    ];
    for (const testCase of cases) {
      for (const signed of [false, true]) {
        const shape = [
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
        ] as const;
        const layout = await verifiedLayout({
          shape,
          sourceLocation: signed
            ? add(
              multiply(coordinate(0), constant("-64")),
              multiply(coordinate(1), constant("-32")),
              multiply(coordinate(2), constant("-16")),
              multiply(coordinate(3), constant("-8")),
              multiply(coordinate(4), constant("-4")),
              multiply(coordinate(5), constant("-2")),
              multiply(coordinate(6), constant("-1")),
            )
            : add(
              coordinate(0),
              multiply(coordinate(1), constant("2")),
              multiply(coordinate(2), constant("4")),
              multiply(coordinate(3), constant("8")),
              multiply(coordinate(4), constant("16")),
              multiply(coordinate(5), constant("32")),
              multiply(coordinate(6), constant("64")),
            ),
          ...(signed
            ? {
              sourceByteOffset: constant(String(testCase.dtypeBytes * 127)),
            }
            : {}),
          sourceBytes: constant(String(testCase.dtypeBytes * 128)),
          destinationBytes: constant(String(testCase.dtypeBytes * 128)),
          dtype: testCase.dtype,
        });
        const prepared = await prepare(layout, await verifiedKernel(layout));

        expect(prepared.semantic.portableProfile).toMatchObject({
          profileId: signed
            ? testCase.signedProfile
            : testCase.positiveProfile,
          rank: 7,
          dtype: testCase.dtype,
        });
        expect(prepared.backendProfile).toBe(testCase.backendProfile);
        expect(prepared.backendVersion).toBe("3.6.0");
        expect(prepared.sourceLocationRange).toEqual(
          signed
            ? { minimum: -127n, maximum: 0n }
            : { minimum: 0n, maximum: 127n },
        );
        expect(prepared.launch.dispatchCount).toEqual([
          testCase.dispatchCount,
          1,
          1,
        ]);
        expect(prepared.program.wgsl).toContain(
          "let coordinate_6: i32 = i32(rank7_remainder_4 % 2u);",
        );
        if (signed) {
          expect(prepared.program.wgsl).toContain("* -64i");
        }
      }
    }
  });

  it("lowers signed-affine packed8, packed16, and word64 source addresses", async () => {
    const cases = [
      {
        dtype: "i8" as const,
        sourceByteOffset: constant("5"),
        sourceBytes: constant("8"),
        destinationBytes: constant("8"),
        profileId:
          "browsergrad.view-copy.signed-affine-rank2-rank3-packed8@1",
        backendProfile: "browsergrad.webgpu.view-copy.packed8@1",
        addressScale: "* 1i",
      },
      {
        dtype: "bf16" as const,
        sourceByteOffset: constant("10"),
        sourceBytes: constant("12"),
        destinationBytes: constant("12"),
        profileId:
          "browsergrad.view-copy.signed-affine-rank2-rank3-packed16@1",
        backendProfile: "browsergrad.webgpu.view-copy.packed16@1",
        addressScale: "* 2i",
      },
      {
        dtype: "f64" as const,
        sourceByteOffset: constant("40"),
        sourceBytes: constant("48"),
        destinationBytes: constant("48"),
        profileId:
          "browsergrad.view-copy.signed-affine-rank2-rank3-word64@1",
        backendProfile: "browsergrad.webgpu.view-copy.word64@1",
        addressScale: "* 8i",
      },
    ];
    for (const testCase of cases) {
      const shape = [constant("2"), constant("3")] as const;
      const layout = await verifiedLayout({
        shape,
        sourceLocation: add(
          multiply(coordinate(0), constant("-3")),
          multiply(coordinate(1), constant("-1")),
        ),
        sourceByteOffset: testCase.sourceByteOffset,
        sourceBytes: testCase.sourceBytes,
        destinationBytes: testCase.destinationBytes,
        dtype: testCase.dtype,
      });
      const prepared = await prepare(layout, await verifiedKernel(layout));

      expect(prepared.semantic.portableProfile).toMatchObject({
        profileId: testCase.profileId,
        rank: 2,
        dtype: testCase.dtype,
      });
      expect(prepared.backendProfile).toBe(testCase.backendProfile);
      expect(prepared.backendVersion).toBe("3.6.0");
      expect(prepared.sourceLocationRange).toEqual({
        minimum: -5n,
        maximum: 0n,
      });
      expect(prepared.program.wgsl).toContain("* -");
      expect(prepared.program.wgsl).toContain(testCase.addressScale);
      expect(prepared.program.wgsl).toContain(
        "u32(source_byte_address / 4i)",
      );
    }
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

  it("lowers signed-affine rank-1 through rank-7 source strides through exact i32 addresses", async () => {
    const cases = [
      {
        shape: [constant("4")] as const,
        sourceLocation: multiply(coordinate(0), constant("-1")),
        sourceByteOffset: constant("12"),
        sourceBytes: constant("16"),
        destinationBytes: constant("16"),
        dtype: "u32" as const,
        expectedRange: { minimum: -3n, maximum: 0n },
        profileId: "browsergrad.view-copy.signed-affine-rank1-word32@1",
      },
      {
        shape: [constant("2"), constant("3")] as const,
        sourceLocation: add(
          multiply(coordinate(0), constant("-3")),
          multiply(coordinate(1), constant("-1")),
        ),
        sourceByteOffset: constant("20"),
        sourceBytes: constant("24"),
        destinationBytes: constant("24"),
        dtype: "f32" as const,
        expectedRange: { minimum: -5n, maximum: 0n },
        profileId:
          "browsergrad.view-copy.signed-affine-rank2-rank3-word32@1",
      },
      {
        shape: [constant("2"), constant("2"), constant("3")] as const,
        sourceLocation: add(
          multiply(coordinate(0), constant("-6")),
          multiply(coordinate(1), constant("-3")),
          multiply(coordinate(2), constant("-1")),
        ),
        sourceByteOffset: constant("44"),
        sourceBytes: constant("48"),
        destinationBytes: constant("48"),
        dtype: "u32" as const,
        expectedRange: { minimum: -11n, maximum: 0n },
        profileId:
          "browsergrad.view-copy.signed-affine-rank2-rank3-word32@1",
      },
      {
        shape: [
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
        ] as const,
        sourceLocation: add(
          multiply(coordinate(0), constant("-8")),
          multiply(coordinate(1), constant("-4")),
          multiply(coordinate(2), constant("-2")),
          multiply(coordinate(3), constant("-1")),
        ),
        sourceByteOffset: constant("60"),
        sourceBytes: constant("64"),
        destinationBytes: constant("64"),
        dtype: "f32" as const,
        expectedRange: { minimum: -15n, maximum: 0n },
        profileId:
          "browsergrad.view-copy.signed-affine-rank4-rank5-word32@1",
      },
      {
        shape: [
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
        ] as const,
        sourceLocation: add(
          multiply(coordinate(0), constant("-32")),
          multiply(coordinate(1), constant("-16")),
          multiply(coordinate(2), constant("-8")),
          multiply(coordinate(3), constant("-4")),
          multiply(coordinate(4), constant("-2")),
          multiply(coordinate(5), constant("-1")),
        ),
        sourceByteOffset: constant("252"),
        sourceBytes: constant("256"),
        destinationBytes: constant("256"),
        dtype: "u32" as const,
        expectedRange: { minimum: -63n, maximum: 0n },
        profileId:
          "browsergrad.view-copy.signed-affine-rank6-word32@1",
      },
      {
        shape: [
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
        ] as const,
        sourceLocation: add(
          multiply(coordinate(0), constant("-64")),
          multiply(coordinate(1), constant("-32")),
          multiply(coordinate(2), constant("-16")),
          multiply(coordinate(3), constant("-8")),
          multiply(coordinate(4), constant("-4")),
          multiply(coordinate(5), constant("-2")),
          multiply(coordinate(6), constant("-1")),
        ),
        sourceByteOffset: constant("508"),
        sourceBytes: constant("512"),
        destinationBytes: constant("512"),
        dtype: "u32" as const,
        expectedRange: { minimum: -127n, maximum: 0n },
        profileId:
          "browsergrad.view-copy.signed-affine-rank7-word32@1",
      },
      {
        shape: [
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
          constant("2"),
        ] as const,
        sourceLocation: add(
          multiply(coordinate(0), constant("-16")),
          multiply(coordinate(1), constant("-8")),
          multiply(coordinate(2), constant("-4")),
          multiply(coordinate(3), constant("-2")),
          multiply(coordinate(4), constant("-1")),
        ),
        sourceByteOffset: constant("124"),
        sourceBytes: constant("128"),
        destinationBytes: constant("128"),
        dtype: "i32" as const,
        expectedRange: { minimum: -31n, maximum: 0n },
        profileId:
          "browsergrad.view-copy.signed-affine-rank4-rank5-word32@1",
      },
    ];
    for (const testCase of cases) {
      const layout = await verifiedLayout(testCase);
      const prepared = await prepare(layout, await verifiedKernel(layout));
      expect(prepared.semantic.portableProfile).toMatchObject({
        profileId: testCase.profileId,
        rank: testCase.shape.length,
        dtype: testCase.dtype,
      });
      expect(prepared.sourceLocationRange).toEqual(testCase.expectedRange);
      expect(prepared.program.wgsl).toContain("* -");
      expect(prepared.program.wgsl).toContain("u32(");
    }

    const signedPredicateLayout = await verifiedLayout({
      shape: [constant("2"), constant("3")],
      sourceLocation: add(
        multiply(coordinate(0), constant("3")),
        coordinate(1),
      ),
      sourcePredicate: {
        kind: "lessEqual",
        lhs: constant("-1"),
        rhs: multiply(coordinate(1), constant("-1")),
      },
      sourceBytes: constant("24"),
      destinationBytes: constant("24"),
    });
    const signedPredicate = await prepare(
      signedPredicateLayout,
      await verifiedKernel(signedPredicateLayout, {
        kind: "fill",
        value: { kind: "float-bits", dtype: "f32", bits: "7fc01234" },
      }),
    );
    expect(signedPredicate.semantic.portableProfile.profileId).toBe(
      "browsergrad.view-copy.signed-affine-rank2-rank3-word32@1",
    );
    expect(signedPredicate.program.wgsl).toContain("* -1i");
    expect(signedPredicate.program.wgsl).toContain(
      "if ((-1i <= (coordinate_1 * -1i)))",
    );
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

  it("lowers exact rank-1 and rank-4 through rank-8 coordinates without changing rank-2/rank-3 WGSL", async () => {
    const rank1Shape = [constant("4")] as const;
    const rank1 = await verifiedLayout({
      shape: rank1Shape,
      sourceLocation: multiply(coordinate(0), constant("2")),
      sourceBytes: constant("28"),
      destinationBytes: constant("16"),
    });
    const rank1Prepared = await prepare(rank1, await verifiedKernel(rank1));
    expect(rank1Prepared.semantic.portableProfile).toMatchObject({
      profileId:
        "browsergrad.view-copy.positive-affine-rank1-rank4-word32@1",
      rank: 1,
      dtype: "f32",
    });
    expect(rank1Prepared.program.wgsl)
      .toContain("let coordinate_0: i32 = i32(linear_index);");

    const rank4Shape = [
      constant("2"),
      constant("2"),
      constant("2"),
      constant("2"),
    ] as const;
    const rank4 = await verifiedLayout({
      shape: rank4Shape,
      sourceLocation: add(
        coordinate(0),
        multiply(coordinate(1), constant("2")),
        multiply(coordinate(2), constant("4")),
        multiply(coordinate(3), constant("8")),
      ),
      sourceBytes: constant("64"),
      destinationBytes: constant("64"),
    });
    const rank4Prepared = await prepare(rank4, await verifiedKernel(rank4));
    expect(rank4Prepared.semantic.portableProfile).toMatchObject({
      profileId:
        "browsergrad.view-copy.positive-affine-rank1-rank4-word32@1",
      rank: 4,
      dtype: "f32",
    });
    expect(rank4Prepared.program.wgsl).toContain(
      "let coordinate_3: i32 = i32(inner_remainder % 2u);",
    );
    expect(rank4Prepared.launch.dispatchCount).toEqual([16, 1, 1]);

    const rank5Shape = [
      constant("2"),
      constant("2"),
      constant("2"),
      constant("2"),
      constant("2"),
    ] as const;
    const rank5 = await verifiedLayout({
      shape: rank5Shape,
      sourceLocation: add(
        coordinate(0),
        multiply(coordinate(1), constant("2")),
        multiply(coordinate(2), constant("4")),
        multiply(coordinate(3), constant("8")),
        multiply(coordinate(4), constant("16")),
      ),
      sourceBytes: constant("128"),
      destinationBytes: constant("128"),
    });
    const rank5Prepared = await prepare(rank5, await verifiedKernel(rank5));
    expect(rank5Prepared.semantic.portableProfile).toMatchObject({
      profileId:
        "browsergrad.view-copy.positive-affine-rank5-word32@1",
      rank: 5,
      dtype: "f32",
    });
    expect(rank5Prepared.program.wgsl).toContain(
      "let coordinate_4: i32 = i32(rank5_remainder_2 % 2u);",
    );
    expect(rank5Prepared.launch.dispatchCount).toEqual([32, 1, 1]);

    const rank6Shape = [
      constant("2"),
      constant("2"),
      constant("2"),
      constant("2"),
      constant("2"),
      constant("2"),
    ] as const;
    const rank6 = await verifiedLayout({
      shape: rank6Shape,
      sourceLocation: add(
        coordinate(0),
        multiply(coordinate(1), constant("2")),
        multiply(coordinate(2), constant("4")),
        multiply(coordinate(3), constant("8")),
        multiply(coordinate(4), constant("16")),
        multiply(coordinate(5), constant("32")),
      ),
      sourceBytes: constant("256"),
      destinationBytes: constant("256"),
    });
    const rank6Prepared = await prepare(rank6, await verifiedKernel(rank6));
    expect(rank6Prepared.semantic.portableProfile).toMatchObject({
      profileId:
        "browsergrad.view-copy.positive-affine-rank6-word32@1",
      rank: 6,
      dtype: "f32",
    });
    expect(rank6Prepared.program.wgsl).toContain(
      "let coordinate_5: i32 = i32(rank6_remainder_3 % 2u);",
    );
    expect(rank6Prepared.launch.dispatchCount).toEqual([64, 1, 1]);

    const rank7Shape = [
      constant("2"),
      constant("2"),
      constant("2"),
      constant("2"),
      constant("2"),
      constant("2"),
      constant("2"),
    ] as const;
    const rank7 = await verifiedLayout({
      shape: rank7Shape,
      sourceLocation: add(
        coordinate(0),
        multiply(coordinate(1), constant("2")),
        multiply(coordinate(2), constant("4")),
        multiply(coordinate(3), constant("8")),
        multiply(coordinate(4), constant("16")),
        multiply(coordinate(5), constant("32")),
        multiply(coordinate(6), constant("64")),
      ),
      sourceBytes: constant("512"),
      destinationBytes: constant("512"),
    });
    const rank7Prepared = await prepare(rank7, await verifiedKernel(rank7));
    expect(rank7Prepared.semantic.portableProfile).toMatchObject({
      profileId:
        "browsergrad.view-copy.positive-affine-rank7-word32@1",
      rank: 7,
      dtype: "f32",
    });
    expect(rank7Prepared.program.wgsl).toContain(
      "let coordinate_6: i32 = i32(rank7_remainder_4 % 2u);",
    );
    expect(rank7Prepared.launch.dispatchCount).toEqual([128, 1, 1]);

    const rank8Shape = [
      constant("2"),
      constant("2"),
      constant("2"),
      constant("2"),
      constant("2"),
      constant("2"),
      constant("2"),
      constant("2"),
    ] as const;
    const rank8 = await verifiedLayout({
      shape: rank8Shape,
      sourceLocation: add(
        coordinate(0),
        multiply(coordinate(1), constant("2")),
        multiply(coordinate(2), constant("4")),
        multiply(coordinate(3), constant("8")),
        multiply(coordinate(4), constant("16")),
        multiply(coordinate(5), constant("32")),
        multiply(coordinate(6), constant("64")),
        multiply(coordinate(7), constant("128")),
      ),
      sourceBytes: constant("1024"),
      destinationBytes: constant("1024"),
    });
    const rank8Prepared = await prepare(rank8, await verifiedKernel(rank8));
    expect(rank8Prepared.semantic.portableProfile).toMatchObject({
      profileId:
        "browsergrad.view-copy.positive-affine-rank8-word32@1",
      rank: 8,
      dtype: "f32",
    });
    expect(rank8Prepared.program.wgsl).toContain(
      "let coordinate_7: i32 = i32(rank8_remainder_5 % 2u);",
    );
    expect(rank8Prepared.launch.dispatchCount).toEqual([256, 1, 1]);
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
