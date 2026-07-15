import { describe, expect, it } from "vitest";
import {
  DENSE_PERMUTATION_VIEW_COPY_FIXTURES,
  fixtureExtentNumbers,
  fixtureWords,
} from "../../../../test-support/dense-permutation-view-copy-fixtures";
import {
  layoutArtifactPayload,
  traceViewCoordinate,
  verifyLayoutArtifact,
  type IndexExpr,
  type MemorySpace,
  type PredicateExpr,
  type VerifiedLayoutArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  encodeWireI64,
  hashSemanticArtifact,
  parseWireI64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import { createVerifiedDensePermutationViewCopyArtifacts } from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  CUDA_LITE_LAYOUT_BINDING_PROFILE,
  CudaLiteCompilerError,
  CudaLiteLayoutBindingError,
  compileCudaLiteKernelWithLayoutBindings,
  createCudaLiteLayoutBindingCompileCacheKey,
  prepareCudaLiteLayoutBindings,
  runCompiledKernelSemanticReference,
  runCompiledKernelWebGpu,
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
    const symbolId = layoutArtifactPayload(artifact).symbols[0]!.id;
    const rebound = await prepareCudaLiteLayoutBindings(artifact, [{
      ...request(artifact, 1),
      dimensionBindings: { [symbolId]: parseWireI64("3") },
    }]);

    expect(second.bindingProjectionHash).not.toBe(identity.bindingProjectionHash);
    expect(second.layoutSemanticHash).not.toBe(changedSecond.layoutSemanticHash);
    expect(second.layoutSemanticHash).toBe(rebound.layoutSemanticHash);
    expect(second.bindingProjectionHash).not.toBe(rebound.bindingProjectionHash);
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

interface CompilerViewFixture {
  readonly shape: readonly number[];
  readonly location: IndexExpr;
  readonly locationUnit?: "element" | "byte";
  readonly byteOffset?: number;
  readonly allocationBytes?: number;
  readonly predicate?: PredicateExpr;
}

async function compilerViewFixture(input: CompilerViewFixture): Promise<VerifiedLayoutArtifact> {
  const shape = input.shape.map((value) => constant(String(value)));
  return verifyLayoutArtifact({
    schema: "browsergrad.layout",
    version: { major: 1, minor: 0 },
    producer: { id: "compiler-layout-lowering-tests", version: "1" },
    artifactId: "compiler-view",
    requiredExtensions: [],
    payload: {
      symbols: [],
      constraints: [],
      allocations: [{
        allocationId: "inputAllocation",
        byteLength: constant(String(input.allocationBytes ?? 256)),
        memorySpace: { kind: "global" },
        alignmentBytes: 16,
        aliasSetId: "inputAlias",
      }],
      indexMaps: [{
        indexMapId: "inputMap",
        coordinateRank: shape.length,
        locationUnit: input.locationUnit ?? "element",
        location: input.location,
        inBounds: input.predicate ?? { kind: "bool", value: true },
      }],
      views: [{
        viewId: "inputView",
        allocationId: "inputAllocation",
        dtype: "f32",
        byteOffset: constant(String(input.byteOffset ?? 0)),
        shape,
        indexMapId: "inputMap",
        requiredAlignmentBytes: 4,
      }],
    },
  });
}

const DIRECT_VIEW_COPY = `
__global__ void copy_view(const float* input, float* output) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 6u) output[i] = input[i];
}`;

describe("CUDA-lite verified layout lowering", () => {
  it.each([
    {
      name: "rank-2 transpose with view byte offset",
      fixture: {
        shape: [2, 3],
        location: addIndex(multiplyIndex(coordinate(1), constant("2")), coordinate(0)),
        byteOffset: 8,
      },
    },
    {
      name: "positive strided slice",
      fixture: {
        shape: [2, 3],
        location: addIndex(
          multiplyIndex(addIndex(coordinate(0), constant("1")), constant("4")),
          multiplyIndex(coordinate(1), constant("2")),
        ),
      },
    },
    {
      name: "read-only broadcast",
      fixture: {
        shape: [2, 3],
        location: coordinate(1),
      },
    },
    {
      name: "byte-unit map",
      fixture: {
        shape: [2, 3],
        locationUnit: "byte" as const,
        byteOffset: 4,
        location: addIndex(
          multiplyIndex(coordinate(0), constant("24")),
          multiplyIndex(coordinate(1), constant("8")),
        ),
      },
    },
    {
      name: "rank-3 permutation",
      fixture: {
        shape: [1, 2, 3],
        location: addIndex(
          multiplyIndex(coordinate(2), constant("2")),
          coordinate(1),
          multiplyIndex(coordinate(0), constant("6")),
        ),
      },
    },
  ])("lowers $name through identical CPU memory traces", async ({ fixture }) => {
    const artifact = await compilerViewFixture(fixture);
    const payload = layoutArtifactPayload(artifact);
    const view = payload.views[0]!;
    const prepared = await prepareCudaLiteLayoutBindings(artifact, [{
      parameter: "input",
      viewId: view.viewId,
      access: "read",
      indexing: "row-major-flat",
    }]);
    const compiled = compileCudaLiteKernelWithLayoutBindings(DIRECT_VIEW_COPY, prepared, {
      workgroupSize: [8, 1, 1],
    });
    const source = Float32Array.from({ length: 64 }, (_, index) => index + 0.25);
    const output = new Float32Array(6);
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { input: source, output } },
      { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
      { trace: "full" },
    );
    const expectedIndices = Array.from({ length: 6 }, (_, flat) => {
      const coordinates = unflattenFlat(flat, fixture.shape);
      const trace = traceViewCoordinate(artifact, {
        viewId: view.viewId,
        coordinates: coordinates.map((value) => encodeWireI64(BigInt(value))),
      });
      expect(trace.accessInBounds).toBe(true);
      return Number(BigInt(trace.rootByteStart) / 4n);
    });
    const readIndices = result.trace.flatMap((thread) => (
      thread.reads.filter((read) => read.name === "input").map((read) => read.index)
    ));

    expect(readIndices).toEqual(expectedIndices);
    expect([...result.buffers.output as Float32Array]).toEqual(expectedIndices.map((index) => source[index]));
    expect(compiled.wgsl).toBeDefined();
    expect(compiled.wgslProgram?.name).toBe(
      `__bg_layout_${prepared.layoutSemanticHash}_${prepared.bindingProjectionHash}_copy_view`,
    );
    expect(compiled.verifiedKernelIr.ir).toBe(compiled.kernelIr);
    expect(compiled.typeCheckedKernelIr.ir).toBe(compiled.kernelIr);
    expect(compiled.wgslLegalizedKernelIr.ir).toBe(compiled.kernelIr);
    expect(compiled.preparedLayoutBindings).toBe(prepared);
    expect(Object.isFrozen(compiled)).toBe(true);
    expect(compiled.layoutBindingCompileCacheKey).toBe(
      createCudaLiteLayoutBindingCompileCacheKey(DIRECT_VIEW_COPY, prepared, { workgroupSize: [8, 1, 1] }),
    );
  });

  it("consumes the canonical constructor predicate when it is true over the logical domain", async () => {
    const artifacts = await createVerifiedDensePermutationViewCopyArtifacts({
      inputShape: [parseWireI64("2"), parseWireI64("3")],
      axes: [1, 0],
      dtype: "f32",
    }, { producer: { id: "compiler-constructor-test", version: "1" } });
    const prepared = await preparedInput(artifacts.layout);
    const compiled = compileCudaLiteKernelWithLayoutBindings(DIRECT_VIEW_COPY, prepared, {
      workgroupSize: [8, 1, 1],
    });
    const result = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: {
          input: new Float32Array([1, 2, 3, 4, 5, 6]),
          output: new Float32Array(6),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    );

    expect(prepared.layoutSemanticHash).toBe(artifacts.layoutSemanticHash);
    expect(prepared.layoutSemanticHash).toBe(await hashSemanticArtifact(artifacts.layout));
    expect([...result.buffers.output as Float32Array]).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it.each(DENSE_PERMUTATION_VIEW_COPY_FIXTURES.cases)(
    "consumes the shared $id semantic hashes and physical-word mapping",
    async (fixture) => {
      const artifacts = await createVerifiedDensePermutationViewCopyArtifacts({
        inputShape: fixture.request.inputShape.map((extent) => parseWireI64(extent)),
        axes: fixture.request.axes,
        dtype: fixture.request.dtype,
      }, { producer: { id: "compiler-shared-fixture", version: "1" } });
      const prepared = await preparedInput(artifacts.layout);
      const payload = layoutArtifactPayload(artifacts.layout);
      const sourceView = payload.views.find((view) => view.viewId === prepared.bindings[0]?.viewId);
      if (sourceView === undefined) throw new Error("shared fixture source view missing");
      const outputShape = fixtureExtentNumbers(fixture.outputShape);
      const sourceWords = fixtureWords(fixture.sourceWords);
      const tracedWords = Array.from({ length: fixture.expectedOutputWords.length }, (_, flat) => {
        const coordinates = unflattenFlat(flat, outputShape);
        const trace = traceViewCoordinate(artifacts.layout, {
          viewId: sourceView.viewId,
          coordinates: coordinates.map((value) => encodeWireI64(BigInt(value))),
        });
        expect(trace.accessInBounds).toBe(true);
        return sourceWords[Number(BigInt(trace.rootByteStart) / 4n)]!;
      });

      expect(artifacts.layoutSemanticHash).toBe(fixture.layoutSemanticHash);
      expect(artifacts.kernelSemanticHash).toBe(fixture.kernelSemanticHash);
      expect(prepared.layoutSemanticHash).toBe(fixture.layoutSemanticHash);
      expect(tracedWords).toEqual([...fixtureWords(fixture.expectedOutputWords)]);
    },
  );

  it("specializes dynamic dimension bindings into physical offsets and compile identity", async () => {
    const artifact = await layoutFixture();
    const prepared = await prepareCudaLiteLayoutBindings(artifact, [request(artifact)]);
    const compiled = compileCudaLiteKernelWithLayoutBindings(DIRECT_VIEW_COPY, prepared, {
      workgroupSize: [8, 1, 1],
    });
    const result = runCompiledKernelSemanticReference(
      compiled,
      {
        buffers: {
          input: Float32Array.from({ length: 24 }, (_, index) => index),
          output: new Float32Array(6),
        },
      },
      { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
    );

    expect(result.trace.flatMap((thread) => (
      thread.reads.filter((read) => read.name === "input").map((read) => read.index)
    ))).toEqual([2, 4, 6, 3, 5, 7]);
    expect([...result.buffers.output as Float32Array]).toEqual([2, 4, 6, 3, 5, 7]);
    expect(compiled.layoutBindingCompileCacheKey).toContain(prepared.bindingProjectionHash);
  });

  it("fails closed without an exact dominant logical-domain guard", async () => {
    const artifact = await compilerViewFixture({
      shape: [2, 3],
      location: addIndex(multiplyIndex(coordinate(1), constant("2")), coordinate(0)),
    });
    const prepared = await preparedInput(artifact);
    const missingGuard = DIRECT_VIEW_COPY.replace("if (i < 6u) ", "");
    const wrongGuard = DIRECT_VIEW_COPY.replace("i < 6u", "i < 7u");

    expect(() => compileCudaLiteKernelWithLayoutBindings(missingGuard, prepared, { workgroupSize: [8, 1, 1] }))
      .toThrow(/BG-COMPILER-LAYOUT-BINDING-MISSING-GUARD/u);
    expect(() => compileCudaLiteKernelWithLayoutBindings(wrongGuard, prepared, { workgroupSize: [8, 1, 1] }))
      .toThrow(/BG-COMPILER-LAYOUT-BINDING-MISSING-GUARD/u);

    try {
      compileCudaLiteKernelWithLayoutBindings(missingGuard, prepared, { workgroupSize: [8, 1, 1] });
      throw new Error("expected layout-bound compilation to fail");
    } catch (error) {
      expect(error).toBeInstanceOf(CudaLiteCompilerError);
      expect((error as CudaLiteCompilerError).diagnostics).toEqual([
        expect.objectContaining({
          code: "BG-COMPILER-LAYOUT-BINDING-MISSING-GUARD",
          span: expect.objectContaining({ line: expect.any(Number), column: expect.any(Number) }),
        }),
      ]);
    }
  });

  it("rejects signed or mutated logical indices, aliases, legacy offsets, and writes", async () => {
    const artifact = await compilerViewFixture({ shape: [2, 3], location: coordinate(1) });
    const prepared = await preparedInput(artifact);
    const signed = DIRECT_VIEW_COPY.replace("unsigned int i", "int i");
    const mutated = DIRECT_VIEW_COPY.replace(
      "if (i < 6u)",
      "i += 1u; if (i < 6u)",
    );
    const aliased = DIRECT_VIEW_COPY.replace(
      "if (i < 6u) output[i] = input[i];",
      "const float* alias = input; if (i < 6u) output[i] = alias[i];",
    );
    const writeSource = DIRECT_VIEW_COPY.replace("const float* input", "float* input")
      .replace("output[i] = input[i]", "input[i] = output[i]");

    expect(() => compileCudaLiteKernelWithLayoutBindings(signed, prepared, { workgroupSize: [8, 1, 1] }))
      .toThrow(/BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX/u);
    expect(() => compileCudaLiteKernelWithLayoutBindings(mutated, prepared, { workgroupSize: [8, 1, 1] }))
      .toThrow(/BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX/u);
    expect(() => compileCudaLiteKernelWithLayoutBindings(aliased, prepared, { workgroupSize: [8, 1, 1] }))
      .toThrow(/BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-USE/u);
    expect(() => compileCudaLiteKernelWithLayoutBindings(DIRECT_VIEW_COPY, prepared, {
      workgroupSize: [8, 1, 1],
      pointerBaseOffsets: { input: 1 },
    })).toThrow(/BG-COMPILER-LAYOUT-BINDING-POINTER-OFFSET/u);
    expect(() => compileCudaLiteKernelWithLayoutBindings(writeSource, prepared, { workgroupSize: [8, 1, 1] }))
      .toThrow(/BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-PARAMETER/u);
  });

  it("rejects logical-index address escapes and operation-level writes", async () => {
    const artifact = await compilerViewFixture({ shape: [2, 3], location: coordinate(1) });
    const prepared = await preparedInput(artifact);
    const helperEscape = `
__device__ void bump(unsigned int* value) { *value = 99u; }
__global__ void copy_view(const float* input, float* output) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 6u) { bump(&i); output[threadIdx.x] = input[i]; }
}`;
    const inlineAsmWrite = `
__global__ void copy_view(const float* input, float* output) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 6u) {
    asm volatile("mov.u32 %0, 99;" : "=r"(i));
    output[threadIdx.x] = input[i];
  }
}`;

    expect(() => compileCudaLiteKernelWithLayoutBindings(helperEscape, prepared, { workgroupSize: [8, 1, 1] }))
      .toThrow(/BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX/u);
    expect(() => compileCudaLiteKernelWithLayoutBindings(inlineAsmWrite, prepared, { workgroupSize: [8, 1, 1] }))
      .toThrow(/BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX/u);
  });

  it("requires authorized compiled proofs and full verified allocation extents at runtime", async () => {
    const artifact = await compilerViewFixture({
      shape: [2, 3],
      location: addIndex(multiplyIndex(coordinate(1), constant("2")), coordinate(0)),
      allocationBytes: 256,
    });
    const prepared = await preparedInput(artifact);
    const compiled = compileCudaLiteKernelWithLayoutBindings(DIRECT_VIEW_COPY, prepared, { workgroupSize: [8, 1, 1] });
    const shortInput = {
      buffers: { input: new Float32Array(63), output: new Float32Array(6) },
    };
    const launch = { gridDim: [1, 1, 1], blockDim: [8, 1, 1] } as const;

    expect(() => runCompiledKernelSemanticReference(compiled, shortInput, launch))
      .toThrow(/BG-COMPILER-LAYOUT-BINDING-RUNTIME-BUFFER/u);
    expect(() => runCompiledKernelSemanticReference(
      compiled,
      { buffers: { input: new Int32Array(64) as never, output: new Float32Array(6) } },
      launch,
    )).toThrow(/BG-COMPILER-LAYOUT-BINDING-RUNTIME-BUFFER/u);
    await expect(runCompiledKernelWebGpu({} as never, compiled, shortInput, launch))
      .rejects.toThrow(/BG-COMPILER-LAYOUT-BINDING-RUNTIME-BUFFER/u);
    await expect(runCompiledKernelWebGpu(
      {} as never,
      compiled,
      {
        buffers: { output: new Float32Array(6) },
        residentBuffers: { input: { buffer: {} as never, byteLength: 252, valueType: "f32" } },
      },
      launch,
    )).rejects.toThrow(/BG-COMPILER-LAYOUT-BINDING-RUNTIME-BUFFER/u);

    const forged = { ...compiled };
    expect(() => runCompiledKernelSemanticReference(
      forged,
      { buffers: { input: new Float32Array(64), output: new Float32Array(6) } },
      launch,
    )).toThrow(/BG-COMPILER-LAYOUT-BINDING-UNVERIFIED-COMPILED/u);
  });

  it("puts complete layout and binding proof hashes in program identity", async () => {
    const transposeArtifact = await compilerViewFixture({
      shape: [2, 3],
      location: addIndex(multiplyIndex(coordinate(1), constant("2")), coordinate(0)),
    });
    const identityArtifact = await compilerViewFixture({
      shape: [2, 3],
      location: addIndex(multiplyIndex(coordinate(0), constant("3")), coordinate(1)),
    });
    const transpose = await preparedInput(transposeArtifact);
    const identity = await preparedInput(identityArtifact);
    const transposeCompiled = compileCudaLiteKernelWithLayoutBindings(DIRECT_VIEW_COPY, transpose, { workgroupSize: [8, 1, 1] });
    const identityCompiled = compileCudaLiteKernelWithLayoutBindings(DIRECT_VIEW_COPY, identity, { workgroupSize: [8, 1, 1] });

    expect(transposeCompiled.wgslProgram?.name).toContain(transpose.layoutSemanticHash);
    expect(transposeCompiled.wgslProgram?.name).toContain(transpose.bindingProjectionHash);
    expect(identityCompiled.wgslProgram?.name).toContain(identity.layoutSemanticHash);
    expect(identityCompiled.wgslProgram?.name).toContain(identity.bindingProjectionHash);
    expect(transposeCompiled.wgslProgram?.name).not.toBe(identityCompiled.wgslProgram?.name);
  });

  it("rejects conditional, negative, unaligned, overflowing, and unsupported-rank maps before execution", async () => {
    const conditional = await compilerViewFixture({
      shape: [2, 3],
      location: coordinate(1),
      predicate: { kind: "lessEqual", lhs: coordinate(1), rhs: constant("1") },
    });
    const negative = await compilerViewFixture({
      shape: [2, 3],
      location: addIndex(constant("-1"), coordinate(1)),
    });
    const unaligned = await compilerViewFixture({
      shape: [2, 3],
      locationUnit: "byte",
      location: coordinate(1),
    });
    const overflow = await compilerViewFixture({
      shape: [2, 3],
      location: multiplyIndex(coordinate(1), constant("4294967296")),
      allocationBytes: 30_000_000_000,
    });
    const rankOne = await compilerViewFixture({ shape: [6], location: coordinate(0) });
    const rankFour = await compilerViewFixture({ shape: [1, 1, 2, 3], location: coordinate(3) });
    const conditionalPrepared = await preparedInput(conditional);
    const negativePrepared = await preparedInput(negative);
    const unalignedPrepared = await preparedInput(unaligned);
    const overflowPrepared = await preparedInput(overflow);
    const rankOnePrepared = await preparedInput(rankOne);
    const rankFourPrepared = await preparedInput(rankFour);

    expect(() => compileCudaLiteKernelWithLayoutBindings(DIRECT_VIEW_COPY, conditionalPrepared, { workgroupSize: [8, 1, 1] }))
      .toThrow(/BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-PREDICATE/u);
    expect(() => compileCudaLiteKernelWithLayoutBindings(DIRECT_VIEW_COPY, negativePrepared, { workgroupSize: [8, 1, 1] }))
      .toThrow(/BG-COMPILER-LAYOUT-BINDING-INTEGER-RANGE/u);
    expect(() => compileCudaLiteKernelWithLayoutBindings(DIRECT_VIEW_COPY, unalignedPrepared, { workgroupSize: [8, 1, 1] }))
      .toThrow(/BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX-MAP/u);
    expect(() => compileCudaLiteKernelWithLayoutBindings(DIRECT_VIEW_COPY, overflowPrepared, { workgroupSize: [8, 1, 1] }))
      .toThrow(/BG-COMPILER-LAYOUT-BINDING-INTEGER-RANGE/u);
    expect(() => compileCudaLiteKernelWithLayoutBindings(DIRECT_VIEW_COPY, rankOnePrepared, { workgroupSize: [8, 1, 1] }))
      .toThrow(/BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX-MAP/u);
    expect(() => compileCudaLiteKernelWithLayoutBindings(DIRECT_VIEW_COPY, rankFourPrepared, { workgroupSize: [8, 1, 1] }))
      .toThrow(/BG-COMPILER-LAYOUT-BINDING-UNSUPPORTED-INDEX-MAP/u);
  });
});

async function preparedInput(artifact: VerifiedLayoutArtifact) {
  return prepareCudaLiteLayoutBindings(artifact, [{
    parameter: "input",
    viewId: layoutArtifactPayload(artifact).views[0]!.viewId,
    access: "read",
    indexing: "row-major-flat",
  }]);
}

function constant(value: string) {
  return { kind: "const" as const, value: parseWireI64(value) };
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

function multiplyIndex(lhs: IndexExpr, rhs: IndexExpr): IndexExpr {
  return { kind: "mul", lhs, rhs };
}

function addIndex(...terms: readonly IndexExpr[]): IndexExpr {
  return { kind: "add", terms };
}

function unflattenFlat(flat: number, shape: readonly number[]): number[] {
  return shape.map((extent, axis) => {
    const stride = shape.slice(axis + 1).reduce((product, value) => product * value, 1);
    return Math.floor(flat / stride) % extent;
  });
}
