import { describe, expect, it } from "vitest";
import {
  layoutArtifactPayload,
  verifyLayoutArtifact,
  type MemorySpace,
  type VerifiedLayoutArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import { hashSemanticArtifact, parseWireI64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CUDA_LITE_LAYOUT_BINDING_PROFILE,
  CudaLiteLayoutBindingError,
  createCudaLiteLayoutBindingCompileCacheKey,
  prepareCudaLiteLayoutBindings,
  type CudaLiteLayoutBindingRequest,
  type PreparedCudaLiteLayoutBindings,
} from "../../src/index";

interface LayoutFixtureOptions {
  readonly producer?: string;
  readonly artifactId?: string;
  readonly memorySpace?: MemorySpace;
  readonly secondLocation?: "identity" | "transpose";
}

async function layoutFixture(options: LayoutFixtureOptions = {}): Promise<VerifiedLayoutArtifact> {
  const secondLocation = options.secondLocation === "identity"
    ? add(multiply(coordinate(0), constant("3")), coordinate(1))
    : add(multiply(coordinate(1), constant("2")), coordinate(0));
  return verifyLayoutArtifact({
    schema: "browsergrad.layout",
    version: { major: 1, minor: 0 },
    producer: { id: options.producer ?? "compiler-layout-binding-tests", version: "1" },
    artifactId: options.artifactId ?? "layout",
    requiredExtensions: [],
    payload: {
      symbols: [{ id: "batch", domain: { min: "1", max: "8" } }],
      constraints: [],
      allocations: [{
        allocationId: "inputAllocation",
        byteLength: constant("96"),
        memorySpace: options.memorySpace ?? { kind: "global" },
        alignmentBytes: 16,
        aliasSetId: "inputAlias",
      }],
      indexMaps: [
        {
          indexMapId: "identityMap",
          coordinateRank: 2,
          locationUnit: "element",
          location: add(multiply(coordinate(0), constant("3")), coordinate(1)),
          inBounds: { kind: "bool", value: true },
        },
        {
          indexMapId: "secondMap",
          coordinateRank: 2,
          locationUnit: "element",
          location: secondLocation,
          inBounds: { kind: "bool", value: true },
        },
      ],
      views: [
        {
          viewId: "identityView",
          allocationId: "inputAllocation",
          dtype: "f32",
          byteOffset: constant("0"),
          shape: [constant("2"), constant("3")],
          indexMapId: "identityMap",
          requiredAlignmentBytes: 4,
        },
        {
          viewId: "secondView",
          allocationId: "inputAllocation",
          dtype: "f32",
          byteOffset: { kind: "mul", lhs: { kind: "symbol", id: "batch" }, rhs: constant("4") },
          shape: [constant("2"), constant("3")],
          indexMapId: "secondMap",
          requiredAlignmentBytes: 4,
        },
      ],
    },
  });
}

function request(
  artifact: VerifiedLayoutArtifact,
  viewIndex = 1,
  parameter = "input",
): CudaLiteLayoutBindingRequest {
  const view = layoutArtifactPayload(artifact).views[viewIndex];
  if (view === undefined) throw new Error("fixture view missing");
  return {
    parameter,
    viewId: view.viewId,
    access: "read" as const,
    indexing: "row-major-flat" as const,
    dimensionBindings: { [layoutArtifactPayload(artifact).symbols[0]!.id]: parseWireI64("2") },
  };
}

describe("prepared CUDA-lite layout bindings", () => {
  it("retains verified semantic identity and resolved immutable binding facts", async () => {
    const artifact = await layoutFixture();
    const prepared = await prepareCudaLiteLayoutBindings(artifact, [request(artifact)]);

    expect(prepared.profile).toBe(CUDA_LITE_LAYOUT_BINDING_PROFILE);
    expect(prepared.layoutSemanticHash).toBe(await hashSemanticArtifact(artifact));
    expect(prepared.bindingProjectionHash).toMatch(/^[0-9a-f]{64}$/u);
    expect(prepared.bindings).toEqual([expect.objectContaining({
      parameter: "input",
      access: "read",
      indexing: "row-major-flat",
      dtype: "f32",
      dtypeBytes: 4,
      locationUnit: "element",
      logicalShape: ["2", "3"],
      viewByteOffset: "8",
      allocationByteLength: "96",
    })]);
    expect(Object.isFrozen(prepared)).toBe(true);
    expect(Object.isFrozen(prepared.bindings)).toBe(true);
    expect(Object.isFrozen(prepared.bindings[0])).toBe(true);
    expect(Object.isFrozen(prepared.bindings[0]!.logicalShape)).toBe(true);
    expect(Object.isFrozen(prepared.bindings[0]!.dimensionBindings)).toBe(true);
  });

  it("canonicalizes request order and dimension-binding key order", async () => {
    const artifact = await layoutFixture();
    const payload = layoutArtifactPayload(artifact);
    const symbolId = payload.symbols[0]!.id;
    const first = await prepareCudaLiteLayoutBindings(artifact, [
      { ...request(artifact, 1, "z"), dimensionBindings: { [symbolId]: parseWireI64("2") } },
      request(artifact, 0, "a"),
    ]);
    const second = await prepareCudaLiteLayoutBindings(artifact, [
      request(artifact, 0, "a"),
      { ...request(artifact, 1, "z"), dimensionBindings: Object.assign(Object.create(null), { [symbolId]: parseWireI64("2") }) },
    ]);

    expect(first.bindings.map((binding) => binding.parameter)).toEqual(["a", "z"]);
    expect(first.bindingProjectionHash).toBe(second.bindingProjectionHash);
    expect(createCudaLiteLayoutBindingCompileCacheKey("source", first, { workgroupSize: [8, 1, 1] }))
      .toBe(createCudaLiteLayoutBindingCompileCacheKey("source", second, { workgroupSize: [8, 1, 1] }));
  });

  it("separates cache identity for semantic layouts, view projections, and compile options", async () => {
    const artifact = await layoutFixture();
    const changed = await layoutFixture({ secondLocation: "identity" });
    const second = await prepareCudaLiteLayoutBindings(artifact, [request(artifact, 1)]);
    const identity = await prepareCudaLiteLayoutBindings(artifact, [request(artifact, 0)]);
    const changedSecond = await prepareCudaLiteLayoutBindings(changed, [request(changed, 1)]);

    expect(second.bindingProjectionHash).not.toBe(identity.bindingProjectionHash);
    expect(second.layoutSemanticHash).not.toBe(changedSecond.layoutSemanticHash);
    expect(createCudaLiteLayoutBindingCompileCacheKey("source", second, { workgroupSize: [8, 1, 1] }))
      .not.toBe(createCudaLiteLayoutBindingCompileCacheKey("source", second, { workgroupSize: [16, 1, 1] }));
    expect(createCudaLiteLayoutBindingCompileCacheKey("source", second))
      .not.toBe(createCudaLiteLayoutBindingCompileCacheKey("source", changedSecond));
  });

  it("uses semantic rather than producer identity", async () => {
    const firstArtifact = await layoutFixture({ producer: "producer-a", artifactId: "artifact-a" });
    const secondArtifact = await layoutFixture({ producer: "producer-b", artifactId: "artifact-b" });
    const first = await prepareCudaLiteLayoutBindings(firstArtifact, [request(firstArtifact)]);
    const second = await prepareCudaLiteLayoutBindings(secondArtifact, [request(secondArtifact)]);

    expect(first.layoutSemanticHash).toBe(second.layoutSemanticHash);
    expect(first.bindingProjectionHash).toBe(second.bindingProjectionHash);
  });

  it("rejects duplicate parameters, unsupported spaces, malformed requests, and resource overflow", async () => {
    const artifact = await layoutFixture();
    await expect(prepareCudaLiteLayoutBindings(artifact, [request(artifact), request(artifact)]))
      .rejects.toMatchObject({ code: "BG-COMPILER-LAYOUT-BINDING-DUPLICATE-PARAMETER" });

    const host = await layoutFixture({ memorySpace: { kind: "host" } });
    await expect(prepareCudaLiteLayoutBindings(host, [request(host)]))
      .rejects.toMatchObject({ code: "BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-MEMORY-SPACE" });

    await expect(prepareCudaLiteLayoutBindings(artifact, [{
      ...request(artifact),
      access: "write" as never,
    }])).rejects.toMatchObject({ code: "BG-COMPILER-LAYOUT-BINDING-INVALID-REQUEST", path: "$.bindings[0].access" });

    await expect(prepareCudaLiteLayoutBindings(artifact, [request(artifact)], { maxBindings: 0 }))
      .rejects.toMatchObject({ code: "BG-COMPILER-LAYOUT-BINDING-RESOURCE-LIMIT", path: "$.maxBindings" });
    await expect(prepareCudaLiteLayoutBindings(artifact, [request(artifact)], { maxBindings: 1 }))
      .resolves.toBeDefined();
  });

  it("rejects forged prepared objects at the compile-cache boundary", () => {
    const forged = Object.freeze({
      profile: CUDA_LITE_LAYOUT_BINDING_PROFILE,
      layoutSemanticHash: "0".repeat(64),
      bindingProjectionHash: "1".repeat(64),
      bindings: [],
    }) as unknown as PreparedCudaLiteLayoutBindings;

    expect(() => createCudaLiteLayoutBindingCompileCacheKey("source", forged)).toThrow(CudaLiteLayoutBindingError);
    expect(() => createCudaLiteLayoutBindingCompileCacheKey("source", forged)).toThrow(
      /BG-COMPILER-LAYOUT-BINDING-UNVERIFIED-PREPARED/u,
    );
  });
});

function constant(value: string) {
  return { kind: "const" as const, value };
}

function coordinate(axis: number) {
  return { kind: "coordinate" as const, axis };
}

function multiply(lhs: ReturnType<typeof coordinate>, rhs: ReturnType<typeof constant>) {
  return { kind: "mul" as const, lhs, rhs };
}

function add(...terms: readonly (ReturnType<typeof coordinate> | ReturnType<typeof multiply>)[]) {
  return { kind: "add" as const, terms };
}
