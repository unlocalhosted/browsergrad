import { describe, expect, it } from "vitest";
import {
  createVerifiedViewCopyArtifacts,
  type VerifiedViewCopyArtifacts,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import { parseWireI64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CUDA_LITE_VIEW_COPY_BINDING_PROFILE,
  CudaLiteViewCopyBindingError,
  createCudaLiteViewCopyBindingCompileCacheKey,
  prepareCudaLiteViewCopyBinding,
  type PreparedCudaLiteViewCopyBinding,
} from "../../src/index";
import { unwrapPreparedCudaLiteViewCopyBinding } from "../../src/semantic_view_copy_bindings";

describe("prepared CUDA-lite view-copy bindings", () => {
  it("binds both artifact roles and all semantic identities", async () => {
    const artifacts = await paddedRank2Artifacts("7fc01234");
    const prepared = await prepareCudaLiteViewCopyBinding(artifacts.layout, artifacts.kernel, {
      operationId: artifacts.operationId,
      sourceParameter: "input",
      destinationParameter: "output",
      indexing: "row-major-flat",
    });

    expect(prepared).toEqual(expect.objectContaining({
      profile: CUDA_LITE_VIEW_COPY_BINDING_PROFILE,
      operationId: artifacts.operationId,
      sourceParameter: "input",
      destinationParameter: "output",
      layoutSemanticHash: artifacts.layoutSemanticHash,
      kernelSemanticHash: artifacts.kernelSemanticHash,
      logicalShape: ["4", "5"],
      elementCount: "20",
      sourceAllocationByteLength: "24",
      destinationAllocationByteLength: "80",
    }));
    expect(prepared.specializationHash).toMatch(/^[0-9a-f]{64}$/u);
    expect(prepared.bindingProjectionHash).toMatch(/^[0-9a-f]{64}$/u);
    expect(Object.isFrozen(prepared)).toBe(true);
    expect(Object.isFrozen(prepared.logicalShape)).toBe(true);
    expect(Object.isFrozen(prepared.dimensionBindings)).toBe(true);
    expect(unwrapPreparedCudaLiteViewCopyBinding(prepared).specialization.filledElements).toBe(14n);
  });

  it("separates source routing, exact fill policy, and operation authority", async () => {
    const firstArtifacts = await paddedRank2Artifacts("7fc01234");
    const secondArtifacts = await paddedRank2Artifacts("7fc05678");
    const first = await prepareCudaLiteViewCopyBinding(firstArtifacts.layout, firstArtifacts.kernel, request(firstArtifacts));
    const rerouted = await prepareCudaLiteViewCopyBinding(firstArtifacts.layout, firstArtifacts.kernel, {
      ...request(firstArtifacts),
      sourceParameter: "source_words",
      destinationParameter: "destination_words",
    });
    const second = await prepareCudaLiteViewCopyBinding(secondArtifacts.layout, secondArtifacts.kernel, request(secondArtifacts));

    expect(first.layoutSemanticHash).toBe(second.layoutSemanticHash);
    expect(first.kernelSemanticHash).not.toBe(second.kernelSemanticHash);
    expect(first.specializationHash).not.toBe(second.specializationHash);
    expect(first.bindingProjectionHash).not.toBe(second.bindingProjectionHash);
    expect(first.bindingProjectionHash).not.toBe(rerouted.bindingProjectionHash);
    expect(createCudaLiteViewCopyBindingCompileCacheKey("source", first))
      .not.toBe(createCudaLiteViewCopyBindingCompileCacheKey("source", second));
    expect(createCudaLiteViewCopyBindingCompileCacheKey("source", first))
      .not.toBe(createCudaLiteViewCopyBindingCompileCacheKey("source", rerouted));
  });

  it("fails closed for malformed requests, artifact substitution, and forged authority", async () => {
    const artifacts = await paddedRank2Artifacts("7fc01234");
    const other = await paddedRank3Artifacts("7fc01234");

    await expect(prepareCudaLiteViewCopyBinding(artifacts.layout, artifacts.kernel, {
      ...request(artifacts),
      destinationParameter: "input",
    })).rejects.toMatchObject({
      code: "BG-COMPILER-VIEW-COPY-BINDING-INVALID-REQUEST",
      path: "$.destinationParameter",
    });
    await expect(prepareCudaLiteViewCopyBinding(artifacts.layout, other.kernel, request(artifacts)))
      .rejects.toMatchObject({ code: "BG-COMPILER-VIEW-COPY-BINDING-INVALID-ARTIFACT" });
    await expect(prepareCudaLiteViewCopyBinding(artifacts.layout, artifacts.kernel, {
      ...request(artifacts),
      operationId: "missing",
    })).rejects.toMatchObject({ code: "BG-COMPILER-VIEW-COPY-BINDING-INVALID-ARTIFACT" });

    const forged = Object.freeze({
      profile: CUDA_LITE_VIEW_COPY_BINDING_PROFILE,
      operationId: artifacts.operationId,
    }) as unknown as PreparedCudaLiteViewCopyBinding;
    expect(() => unwrapPreparedCudaLiteViewCopyBinding(forged)).toThrow(CudaLiteViewCopyBindingError);
    expect(() => unwrapPreparedCudaLiteViewCopyBinding(forged)).toThrow(
      /BG-COMPILER-VIEW-COPY-BINDING-UNVERIFIED-PREPARED/u,
    );
    expect(() => createCudaLiteViewCopyBindingCompileCacheKey("source", forged)).toThrow(
      /BG-COMPILER-VIEW-COPY-BINDING-UNVERIFIED-PREPARED/u,
    );
  });

  it("passes specialization resource and cancellation controls through the authority boundary", async () => {
    const artifacts = await paddedRank2Artifacts("7fc01234");
    await expect(prepareCudaLiteViewCopyBinding(artifacts.layout, artifacts.kernel, request(artifacts), {
      maxElements: 19,
    })).rejects.toMatchObject({ code: "BG-COMPILER-VIEW-COPY-BINDING-INVALID-ARTIFACT" });

    const controller = new AbortController();
    controller.abort();
    await expect(prepareCudaLiteViewCopyBinding(artifacts.layout, artifacts.kernel, request(artifacts), {
      signal: controller.signal,
    })).rejects.toMatchObject({ code: "BG-COMPILER-VIEW-COPY-BINDING-INVALID-ARTIFACT" });
  });
});

function request(artifacts: VerifiedViewCopyArtifacts) {
  return {
    operationId: artifacts.operationId,
    sourceParameter: "input",
    destinationParameter: "output",
    indexing: "row-major-flat" as const,
  };
}

async function paddedRank2Artifacts(fillBits: string): Promise<VerifiedViewCopyArtifacts> {
  return createVerifiedViewCopyArtifacts({
    dtype: "f32",
    symbols: [],
    constraints: [],
    source: {
      layout: {
        kind: "pad",
        source: strided([2, 3]),
        low: [constant(1), constant(1)],
        high: [constant(1), constant(1)],
      },
      allocation: globalAllocation(24),
      byteOffset: constant(0),
      requiredAlignmentBytes: 4,
    },
    destination: {
      layout: paddedDestination([2, 3]),
      allocation: globalAllocation(80),
      byteOffset: constant(0),
      requiredAlignmentBytes: 4,
    },
    invalidSource: {
      kind: "fill",
      value: { kind: "float-bits", dtype: "f32", bits: fillBits },
    },
  }, { producer: { id: "compiler-view-copy-binding-test", version: "1" } });
}

async function paddedRank3Artifacts(fillBits: string): Promise<VerifiedViewCopyArtifacts> {
  return createVerifiedViewCopyArtifacts({
    dtype: "f32",
    symbols: [],
    constraints: [],
    source: {
      layout: {
        kind: "pad",
        source: strided([2, 2, 2]),
        low: [constant(1), constant(1), constant(1)],
        high: [constant(1), constant(1), constant(1)],
      },
      allocation: globalAllocation(32),
      byteOffset: constant(0),
      requiredAlignmentBytes: 4,
    },
    destination: {
      layout: paddedDestination([2, 2, 2]),
      allocation: globalAllocation(256),
      byteOffset: constant(0),
      requiredAlignmentBytes: 4,
    },
    invalidSource: {
      kind: "fill",
      value: { kind: "float-bits", dtype: "f32", bits: fillBits },
    },
  }, { producer: { id: "compiler-view-copy-binding-test", version: "1" } });
}

function strided(shape: readonly number[]) {
  return {
    kind: "strided" as const,
    shape: shape.map(constant),
    strides: shape.map((_, axis) => constant(
      shape.slice(axis + 1).reduce((product, extent) => product * extent, 1),
    )),
  };
}

function paddedDestination(sourceShape: readonly number[]) {
  const destinationShape = sourceShape.map((extent) => ({
    kind: "add" as const,
    terms: [constant(1), constant(extent), constant(1)],
  }));
  const destinationExtents = sourceShape.map((extent) => extent + 2);
  return {
    kind: "strided" as const,
    shape: destinationShape,
    strides: destinationExtents.map((_, axis) => constant(
      destinationExtents.slice(axis + 1).reduce((product, extent) => product * extent, 1),
    )),
  };
}

function constant(value: number) {
  return { kind: "const" as const, value: parseWireI64(String(value)) };
}

function globalAllocation(byteLength: number) {
  return {
    byteLength: constant(byteLength),
    memorySpace: { kind: "global" as const },
    alignmentBytes: 4,
  };
}
