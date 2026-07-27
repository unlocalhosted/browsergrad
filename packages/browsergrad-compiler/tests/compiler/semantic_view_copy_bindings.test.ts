import { describe, expect, it } from "vitest";
import {
  createVerifiedViewCopyArtifacts,
  prepareViewCopyCpu,
  type VerifiedViewCopyArtifacts,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  hostGraphArtifactPayload,
  prepareHostGraphProgram,
} from "@unlocalhosted/browsergrad-semantic-core/graph";
import { parseWireI64 } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CUDA_LITE_VIEW_COPY_BINDING_PROFILE,
  CUDA_LITE_VIEW_COPY_HOST_GRAPH_PROFILE,
  CudaLiteCompilerError,
  CudaLiteViewCopyBindingError,
  compileCudaLiteKernelWithViewCopyBinding,
  createCudaLiteViewCopyHostGraph,
  createCudaLiteViewCopyBindingCompileCacheKey,
  prepareCudaLiteViewCopyBinding,
  runCompiledKernelSemanticReference,
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

describe("CUDA-lite view-copy host graphs", () => {
  it("constructs a bounded multi-dispatch pipeline from opaque bindings", async () => {
    const artifacts = await denseRank2Artifacts();
    const first = await prepareCudaLiteViewCopyBinding(
      artifacts.layout,
      artifacts.kernel,
      request(artifacts),
    );
    const second = await prepareCudaLiteViewCopyBinding(
      artifacts.layout,
      artifacts.kernel,
      request(artifacts),
    );
    const constructed = await createCudaLiteViewCopyHostGraph([
      first,
      second,
    ]);
    const payload = hostGraphArtifactPayload(constructed.artifact);
    const prepared = await prepareHostGraphProgram(constructed.artifact);

    expect(constructed).toMatchObject({
      profile: CUDA_LITE_VIEW_COPY_HOST_GRAPH_PROFILE,
      dispatchCount: 2,
      inputResourceId: "input",
      outputResourceId: "output",
      temporaryResourceIds: ["temporary/0"],
    });
    expect(payload.program.resources).toEqual([
      {
        resourceId: "input",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "f32",
        byteLength: "24",
        alignmentBytes: 4,
      },
      {
        resourceId: "output",
        role: "output",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "f32",
        byteLength: "24",
        alignmentBytes: 4,
      },
      {
        resourceId: "temporary/0",
        role: "temporary",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "f32",
        byteLength: "24",
        alignmentBytes: 4,
      },
    ]);
    expect(prepared).toMatchObject({
      graphSemanticHash: constructed.graphSemanticHash,
      dispatchCount: 2,
      collectiveCount: 0,
      topologicalNodeIds: ["dispatch/0", "dispatch/1"],
    });
  });

  it("retains resolved dimension bindings and allocation geometry", async () => {
    const artifacts = await dynamicRank2Artifacts();
    const binding = await prepareCudaLiteViewCopyBinding(
      artifacts.layout,
      artifacts.kernel,
      {
        ...request(artifacts),
        dimensionBindings: { n: parseWireI64("3") },
      },
    );
    const constructed = await createCudaLiteViewCopyHostGraph([binding]);
    const payload = hostGraphArtifactPayload(constructed.artifact);

    expect(payload.program.resources).toEqual([
      {
        resourceId: "input",
        role: "input",
        multiplicity: "per-rank",
        initialization: "external-input",
        dtype: "f32",
        byteLength: "24",
        alignmentBytes: 4,
      },
      {
        resourceId: "output",
        role: "output",
        multiplicity: "per-rank",
        initialization: "zero-fill",
        dtype: "f32",
        byteLength: "24",
        alignmentBytes: 4,
      },
    ]);
    expect(payload.program.nodes[0]).toMatchObject({
      kind: "dispatch",
      dimensionBindings: { n: "3" },
    });
  });

  it("rejects empty, forged, hostile, and incompatible pipelines", async () => {
    await expect(createCudaLiteViewCopyHostGraph([])).rejects.toMatchObject({
      code: "BG-COMPILER-VIEW-COPY-BINDING-INVALID-REQUEST",
    });

    const dense = await denseRank2Artifacts();
    const denseBinding = await prepareCudaLiteViewCopyBinding(
      dense.layout,
      dense.kernel,
      request(dense),
    );
    const forged = Object.freeze({
      profile: CUDA_LITE_VIEW_COPY_BINDING_PROFILE,
    }) as unknown as PreparedCudaLiteViewCopyBinding;
    await expect(createCudaLiteViewCopyHostGraph([forged]))
      .rejects.toMatchObject({
        code: "BG-COMPILER-VIEW-COPY-BINDING-UNVERIFIED-PREPARED",
      });

    const padded = await paddedRank2Artifacts("7fc01234");
    const paddedBinding = await prepareCudaLiteViewCopyBinding(
      padded.layout,
      padded.kernel,
      request(padded),
    );
    await expect(createCudaLiteViewCopyHostGraph([
      paddedBinding,
      denseBinding,
    ])).rejects.toMatchObject({
      code: "BG-COMPILER-VIEW-COPY-BINDING-INVALID-REQUEST",
    });

    let reads = 0;
    const hostile = [denseBinding];
    Object.defineProperty(hostile, "0", {
      enumerable: true,
      get() {
        reads += 1;
        return denseBinding;
      },
    });
    await expect(createCudaLiteViewCopyHostGraph(hostile))
      .rejects.toMatchObject({
        code: "BG-COMPILER-VIEW-COPY-BINDING-INVALID-REQUEST",
      });
    expect(reads).toBe(0);
  });
});

describe("CUDA-lite structured view-copy lowering", () => {
  it("executes rank-2 padding as exact raw words behind a structured source guard", async () => {
    const artifacts = await paddedRank2Artifacts("7fc01234");
    const prepared = await prepareCudaLiteViewCopyBinding(artifacts.layout, artifacts.kernel, request(artifacts));
    const sourceText = directViewCopySource(20);
    const compiled = compileCudaLiteKernelWithViewCopyBinding(sourceText, prepared, { workgroupSize: [32, 1, 1] });
    const source = Uint32Array.from([0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000]);
    const originalSource = new Uint32Array(source);
    const output = new Uint32Array(20);
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { input: source, output } },
      { gridDim: [1, 1, 1], blockDim: [32, 1, 1] },
      { trace: "full" },
    );
    const canonicalDestination = new Uint8Array(80);
    const canonical = await prepareViewCopyCpu(artifacts.layout, artifacts.kernel, {
      operationId: artifacts.operationId,
    });
    const canonicalTrace = canonical.execute({
      source: new Uint8Array(source.buffer),
      destination: canonicalDestination,
    });
    const expected = paddedWords2d(source, 4, 5, 1, 1, 2, 3, 0x7fc01234);
    const sourceReads = result.trace.flatMap((thread) => (
      thread.reads.filter((read) => read.name === "input").map((read) => read.index)
    ));

    expect([...result.buffers.output as Uint32Array]).toEqual(expected);
    expect([...result.buffers.output as Uint32Array]).toEqual([...new Uint32Array(canonicalDestination.buffer)]);
    expect([...source]).toEqual([...originalSource]);
    expect(sourceReads).toEqual([0, 1, 2, 3, 4, 5]);
    expect(canonicalTrace).toMatchObject({ readElements: "6", filledElements: "14" });
    expect(compiled.kernelIr.params.map((parameter) => parameter.valueType)).toEqual(["uint", "uint"]);
    expect(compiled.wgsl).toContain("array<u32>");
    expect(compiled.wgsl).toContain("if (");
    expect(compiled.wgsl).toContain("i32(");
    const entryWgsl = mainWgsl(compiled.wgsl);
    expect(entryWgsl).not.toContain("select(");
    expect(entryWgsl).toContain("i32(0) <= i32(bitcast<i32>");
    expect(entryWgsl).not.toContain("0u <= u32");
    expect(entryWgsl.indexOf("input[")).toBeGreaterThan(entryWgsl.indexOf("if (", entryWgsl.indexOf("if (") + 1));
    expect(entryWgsl).toContain("2143294004u");
    expect(compiled.wgslProgram?.name).toContain(prepared.layoutSemanticHash);
    expect(compiled.wgslProgram?.name).toContain(prepared.kernelSemanticHash);
    expect(compiled.wgslProgram?.name).toContain(prepared.specializationHash);
    expect(compiled.wgslProgram?.name).toContain(prepared.bindingProjectionHash);
    expect(compiled.preparedViewCopyBinding).toBe(prepared);
    expect(compiled.viewCopyBindingCompileCacheKey).toBe(
      createCudaLiteViewCopyBindingCompileCacheKey(sourceText, prepared, { workgroupSize: [32, 1, 1] }),
    );
  });

  it("executes rank-3 padding through the same artifact-specialized lowering", async () => {
    const artifacts = await paddedRank3Artifacts("7fc05678");
    const prepared = await prepareCudaLiteViewCopyBinding(artifacts.layout, artifacts.kernel, request(artifacts));
    const compiled = compileCudaLiteKernelWithViewCopyBinding(directViewCopySource(64), prepared, {
      workgroupSize: [64, 1, 1],
    });
    const source = Uint32Array.from({ length: 8 }, (_, index) => 0x3f800000 + index);
    const output = new Uint32Array(64);
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { input: source, output } },
      { gridDim: [1, 1, 1], blockDim: [64, 1, 1] },
      { trace: "full" },
    );
    const expected = paddedWords3d(source, 4, 4, 4, 1, 2, 2, 2, 0x7fc05678);
    const sourceReads = result.trace.flatMap((thread) => (
      thread.reads.filter((read) => read.name === "input").map((read) => read.index)
    ));

    expect([...result.buffers.output as Uint32Array]).toEqual(expected);
    expect(sourceReads).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(expected.filter((word) => word === 0x7fc05678)).toHaveLength(56);
    expect(mainWgsl(compiled.wgsl)).not.toContain("select(");
  });

  it("preserves nonzero allocation offsets and untouched root canaries", async () => {
    const artifacts = await paddedRank2Artifacts("7fc0abcd", {
      sourceOffsetWords: 1,
      sourceSuffixWords: 1,
      destinationOffsetWords: 1,
      destinationSuffixWords: 1,
    });
    const prepared = await prepareCudaLiteViewCopyBinding(artifacts.layout, artifacts.kernel, request(artifacts));
    const compiled = compileCudaLiteKernelWithViewCopyBinding(directViewCopySource(20), prepared, {
      workgroupSize: [32, 1, 1],
    });
    const source = Uint32Array.from([
      0xdeadbeef,
      0x3f800000,
      0x40000000,
      0x40400000,
      0x40800000,
      0x40a00000,
      0x40c00000,
      0xcafebabe,
    ]);
    const sourceBefore = new Uint32Array(source);
    const destination = new Uint32Array(22);
    destination.fill(0xa5a5a5a5);
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { input: source, output: destination } },
      { gridDim: [1, 1, 1], blockDim: [32, 1, 1] },
      { trace: "full" },
    );
    const canonical = await prepareViewCopyCpu(artifacts.layout, artifacts.kernel, {
      operationId: artifacts.operationId,
    });
    const canonicalDestination = new Uint8Array(destination.byteLength);
    new Uint32Array(canonicalDestination.buffer).fill(0xa5a5a5a5);
    canonical.execute({
      source: new Uint8Array(source.buffer),
      destination: canonicalDestination,
    });
    const output = result.buffers.output as Uint32Array;
    const sourceReads = result.trace.flatMap((thread) => (
      thread.reads.filter((read) => read.name === "input").map((read) => read.index)
    ));
    const destinationWrites = result.trace.flatMap((thread) => (
      thread.writes.filter((write) => write.name === "output").map((write) => write.index)
    ));

    expect([...output]).toEqual([...new Uint32Array(canonicalDestination.buffer)]);
    expect([...output]).toEqual([
      0xa5a5a5a5,
      ...paddedWords2d(source.subarray(1, 7), 4, 5, 1, 1, 2, 3, 0x7fc0abcd),
      0xa5a5a5a5,
    ]);
    expect([...source]).toEqual([...sourceBefore]);
    expect(sourceReads).toEqual([1, 2, 3, 4, 5, 6]);
    expect(destinationWrites).toEqual(Array.from({ length: 20 }, (_, index) => index + 1));
  });

  it("fills an always-false source map without issuing any source read", async () => {
    const artifacts = await alwaysFalseArtifacts("7fc0beef");
    const prepared = await prepareCudaLiteViewCopyBinding(artifacts.layout, artifacts.kernel, request(artifacts));
    const compiled = compileCudaLiteKernelWithViewCopyBinding(directViewCopySource(4), prepared, {
      workgroupSize: [4, 1, 1],
    });
    const source = Uint32Array.of(0xdeadbeef);
    const sourceBefore = new Uint32Array(source);
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { input: source, output: new Uint32Array(4) } },
      { gridDim: [1, 1, 1], blockDim: [4, 1, 1] },
      { trace: "full" },
    );
    const sourceReads = result.trace.flatMap((thread) => (
      thread.reads.filter((read) => read.name === "input")
    ));
    const canonicalDestination = new Uint8Array(16);
    const canonical = await prepareViewCopyCpu(artifacts.layout, artifacts.kernel, {
      operationId: artifacts.operationId,
    });
    const canonicalTrace = canonical.execute({
      source: new Uint8Array(source.buffer),
      destination: canonicalDestination,
    });
    const specialization = unwrapPreparedCudaLiteViewCopyBinding(prepared).specialization;

    expect([...(result.buffers.output as Uint32Array)]).toEqual(Array(4).fill(0x7fc0beef));
    expect([...(result.buffers.output as Uint32Array)]).toEqual([...new Uint32Array(canonicalDestination.buffer)]);
    expect(sourceReads).toEqual([]);
    expect([...source]).toEqual([...sourceBefore]);
    expect(specialization).toMatchObject({ readElements: 0n, filledElements: 4n });
    expect(canonicalTrace).toMatchObject({ readElements: "0", filledElements: "4" });
    expect(mainWgsl(compiled.wgsl).indexOf("input[")).toBeGreaterThan(
      mainWgsl(compiled.wgsl).indexOf("if (", mainWgsl(compiled.wgsl).indexOf("if (") + 1),
    );
  });

  it("rejects possibly-invalid reads when the verified operation has reject policy", async () => {
    const artifacts = await paddedRank2Artifacts();
    const caught = await prepareCudaLiteViewCopyBinding(
      artifacts.layout,
      artifacts.kernel,
      request(artifacts),
    ).then(() => undefined, (error: unknown) => error);

    expect(caught).toMatchObject({
      code: "BG-COMPILER-VIEW-COPY-BINDING-INVALID-ARTIFACT",
      path: "$.source[0,0]",
    });
    expect((caught as Error & { readonly cause?: unknown }).cause).toMatchObject({
      diagnostic: {
        code: "BG-KERNEL-INVALID-ACCESS",
        path: "$.source[0,0]",
      },
    });

    const dense = await denseRank2Artifacts();
    const prepared = await prepareCudaLiteViewCopyBinding(dense.layout, dense.kernel, request(dense));
    const compiled = compileCudaLiteKernelWithViewCopyBinding(directViewCopySource(6), prepared, {
      workgroupSize: [8, 1, 1],
    });
    const source = Uint32Array.from([1, 2, 3, 4, 5, 6]);
    const result = runCompiledKernelSemanticReference(
      compiled,
      { buffers: { input: source, output: new Uint32Array(6) } },
      { gridDim: [1, 1, 1], blockDim: [8, 1, 1] },
      { trace: "full" },
    );
    expect([...(result.buffers.output as Uint32Array)]).toEqual([...source]);
    expect(unwrapPreparedCudaLiteViewCopyBinding(prepared).specialization)
      .toMatchObject({ readElements: 6n, filledElements: 0n });
  });

  it("rejects semantic widening around the exact guarded materializing copy", async () => {
    const artifacts = await paddedRank2Artifacts("7fc01234");
    const prepared = await prepareCudaLiteViewCopyBinding(artifacts.layout, artifacts.kernel, request(artifacts));
    const valid = directViewCopySource(20);
    const cases = [
      valid.replace("if (i < 20u) ", ""),
      valid.replace("i < 20u", "i < 21u"),
      valid.replace("output[i] = input[i]", "output[i] = input[i] + 1.0f"),
      valid.replace("output[i] = input[i]", "output[0] = input[i]"),
      valid.replace("if (i < 20u) output[i] = input[i];", "if (i < 20u) { output[i] = input[i]; output[i] = input[i]; }"),
      valid.replace("if (i < 20u)", "if (i < 20u) if (threadIdx.x < 32u)"),
      valid.replace("\n}", "\n  return;\n}"),
    ];

    for (const source of cases) {
      expect(() => compileCudaLiteKernelWithViewCopyBinding(source, prepared, { workgroupSize: [32, 1, 1] }))
        .toThrow(CudaLiteCompilerError);
      expect(() => compileCudaLiteKernelWithViewCopyBinding(source, prepared, { workgroupSize: [32, 1, 1] }))
        .toThrow(/BG-COMPILER-VIEW-COPY-BINDING-(?:MISSING-GUARD|UNSUPPORTED-SOURCE)/u);
    }
  });

  it("requires exact non-aliased raw allocation roots at runtime", async () => {
    const artifacts = await paddedRank2Artifacts("7fc01234");
    const prepared = await prepareCudaLiteViewCopyBinding(artifacts.layout, artifacts.kernel, request(artifacts));
    const compiled = compileCudaLiteKernelWithViewCopyBinding(directViewCopySource(20), prepared, {
      workgroupSize: [32, 1, 1],
    });
    const launch = { gridDim: [1, 1, 1], blockDim: [32, 1, 1] } as const;
    const root = new ArrayBuffer(104);

    expect(() => runCompiledKernelSemanticReference(
      compiled,
      { buffers: { input: new Uint32Array(5), output: new Uint32Array(20) } },
      launch,
    )).toThrow(/BG-COMPILER-VIEW-COPY-BINDING-RUNTIME-BUFFER/u);
    expect(() => runCompiledKernelSemanticReference(
      compiled,
      { buffers: { input: new Float32Array(6) as never, output: new Uint32Array(20) } },
      launch,
    )).toThrow(/BG-COMPILER-VIEW-COPY-BINDING-RUNTIME-BUFFER/u);
    if (typeof SharedArrayBuffer !== "undefined") {
      expect(() => runCompiledKernelSemanticReference(
        compiled,
        {
          buffers: {
            input: new Uint32Array(new SharedArrayBuffer(24)),
            output: new Uint32Array(20),
          },
        },
        launch,
      )).toThrow(/BG-COMPILER-VIEW-COPY-BINDING-RUNTIME-BUFFER/u);
    }
    expect(() => runCompiledKernelSemanticReference(
      compiled,
      { buffers: { input: new Uint32Array(root, 0, 6), output: new Uint32Array(root, 24, 20) } },
      launch,
    )).toThrow(/BG-COMPILER-VIEW-COPY-BINDING-RUNTIME-BUFFER/u);
    expect(() => runCompiledKernelSemanticReference(
      { ...compiled },
      { buffers: { input: new Uint32Array(6), output: new Uint32Array(20) } },
      launch,
    )).toThrow(/BG-COMPILER-VIEW-COPY-BINDING-UNVERIFIED-COMPILED/u);
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

async function paddedRank2Artifacts(
  fillBits?: string,
  offsets: {
    readonly sourceOffsetWords?: number;
    readonly sourceSuffixWords?: number;
    readonly destinationOffsetWords?: number;
    readonly destinationSuffixWords?: number;
  } = {},
): Promise<VerifiedViewCopyArtifacts> {
  const sourceOffsetWords = offsets.sourceOffsetWords ?? 0;
  const destinationOffsetWords = offsets.destinationOffsetWords ?? 0;
  const sourceSuffixWords = offsets.sourceSuffixWords ?? 0;
  const destinationSuffixWords = offsets.destinationSuffixWords ?? 0;
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
      allocation: globalAllocation((sourceOffsetWords + 6 + sourceSuffixWords) * 4),
      byteOffset: constant(sourceOffsetWords * 4),
      requiredAlignmentBytes: 4,
    },
    destination: {
      layout: paddedDestination([2, 3]),
      allocation: globalAllocation((destinationOffsetWords + 20 + destinationSuffixWords) * 4),
      byteOffset: constant(destinationOffsetWords * 4),
      requiredAlignmentBytes: 4,
    },
    invalidSource: fillBits === undefined ? { kind: "reject" } : {
      kind: "fill",
      value: { kind: "float-bits", dtype: "f32", bits: fillBits },
    },
  }, { producer: { id: "compiler-view-copy-binding-test", version: "1" } });
}

async function denseRank2Artifacts(): Promise<VerifiedViewCopyArtifacts> {
  return createVerifiedViewCopyArtifacts({
    dtype: "f32",
    symbols: [],
    constraints: [],
    source: {
      layout: strided([2, 3]),
      allocation: globalAllocation(24),
      byteOffset: constant(0),
      requiredAlignmentBytes: 4,
    },
    destination: {
      layout: strided([2, 3]),
      allocation: globalAllocation(24),
      byteOffset: constant(0),
      requiredAlignmentBytes: 4,
    },
    invalidSource: { kind: "reject" },
  }, { producer: { id: "compiler-view-copy-binding-test", version: "1" } });
}

async function dynamicRank2Artifacts(): Promise<VerifiedViewCopyArtifacts> {
  const symbol = () => ({ kind: "symbol" as const, id: "n" });
  const byteLength = () => ({
    kind: "mul" as const,
    lhs: symbol(),
    rhs: constant(8),
  });
  const layout = () => ({
    kind: "strided" as const,
    shape: [symbol(), constant(2)],
    strides: [constant(2), constant(1)],
  });
  return createVerifiedViewCopyArtifacts({
    dtype: "f32",
    symbols: [{ id: "n", domain: { min: parseWireI64("0"), max: parseWireI64("4") } }],
    constraints: [],
    source: {
      layout: layout(),
      allocation: {
        byteLength: byteLength(),
        memorySpace: { kind: "global" },
        alignmentBytes: 4,
      },
      byteOffset: constant(0),
      requiredAlignmentBytes: 4,
    },
    destination: {
      layout: layout(),
      allocation: {
        byteLength: byteLength(),
        memorySpace: { kind: "global" },
        alignmentBytes: 4,
      },
      byteOffset: constant(0),
      requiredAlignmentBytes: 4,
    },
    invalidSource: { kind: "reject" },
  }, { producer: { id: "compiler-view-copy-binding-test", version: "1" } });
}

async function alwaysFalseArtifacts(fillBits: string): Promise<VerifiedViewCopyArtifacts> {
  return createVerifiedViewCopyArtifacts({
    dtype: "f32",
    symbols: [],
    constraints: [],
    source: {
      layout: {
        kind: "pad",
        source: strided([0, 0]),
        low: [constant(1), constant(1)],
        high: [constant(1), constant(1)],
      },
      allocation: globalAllocation(4),
      byteOffset: constant(0),
      requiredAlignmentBytes: 4,
    },
    destination: {
      layout: paddedDestination([0, 0]),
      allocation: globalAllocation(16),
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

function directViewCopySource(elementCount: number): string {
  return `
__global__ void copy_view(const float* input, float* output) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${elementCount}u) output[i] = input[i];
}`;
}

function paddedWords2d(
  source: Uint32Array,
  rows: number,
  columns: number,
  lowRow: number,
  lowColumn: number,
  sourceRows: number,
  sourceColumns: number,
  fill: number,
): number[] {
  return Array.from({ length: rows * columns }, (_, flat) => {
    const row = Math.floor(flat / columns);
    const column = flat % columns;
    const sourceRow = row - lowRow;
    const sourceColumn = column - lowColumn;
    return sourceRow >= 0 && sourceRow < sourceRows && sourceColumn >= 0 && sourceColumn < sourceColumns
      ? source[sourceRow * sourceColumns + sourceColumn]!
      : fill;
  });
}

function paddedWords3d(
  source: Uint32Array,
  depth: number,
  rows: number,
  columns: number,
  low: number,
  sourceDepth: number,
  sourceRows: number,
  sourceColumns: number,
  fill: number,
): number[] {
  return Array.from({ length: depth * rows * columns }, (_, flat) => {
    const outputDepth = Math.floor(flat / (rows * columns));
    const outputRow = Math.floor(flat / columns) % rows;
    const outputColumn = flat % columns;
    const sourceZ = outputDepth - low;
    const sourceY = outputRow - low;
    const sourceX = outputColumn - low;
    return sourceZ >= 0 && sourceZ < sourceDepth &&
      sourceY >= 0 && sourceY < sourceRows &&
      sourceX >= 0 && sourceX < sourceColumns
      ? source[(sourceZ * sourceRows + sourceY) * sourceColumns + sourceX]!
      : fill;
  });
}

function mainWgsl(wgsl: string | undefined): string {
  if (wgsl === undefined) throw new Error("expected WGSL output");
  const start = wgsl.lastIndexOf("@compute");
  if (start < 0) throw new Error("expected WGSL compute entry");
  return wgsl.slice(start);
}
