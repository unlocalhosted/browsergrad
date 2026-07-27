import { describe, expect, it } from "vitest";

import {
  HOST_GRAPH_CPU_PROFILE,
  HostGraphCpuError,
  createVerifiedHostGraphArtifact,
  prepareHostGraphCpu,
  type HostGraphCpuInputBinding,
  type HostGraphProgram,
  type VerifiedHostGraphArtifact,
} from "../../src/graph";
import {
  createVerifiedDensePermutationViewCopyArtifacts,
  type VerifiedViewCopyArtifacts,
} from "../../src/kernel";
import {
  parseWireI64,
  parseWireU64,
  type WireU64,
} from "../../src/schema";

const wire = (value: string): WireU64 => parseWireU64(value);

async function identityArtifacts(
  dtype: "f32" | "i32" | "u32" = "f32",
): Promise<VerifiedViewCopyArtifacts> {
  return createVerifiedDensePermutationViewCopyArtifacts({
    inputShape: [parseWireI64("2")],
    axes: [0],
    dtype,
  });
}

function artifactOptions(artifacts: VerifiedViewCopyArtifacts) {
  return {
    kernelArtifacts: [artifacts.kernel],
    layoutArtifacts: [artifacts.layout],
  };
}

function resource(
  resourceId: string,
  role: "input" | "temporary" | "output",
  dtype: "f32" | "i32" | "u32",
) {
  return {
    resourceId,
    role,
    multiplicity: "per-rank" as const,
    initialization: role === "input"
      ? "external-input" as const
      : "zero-fill" as const,
    dtype,
    byteLength: wire("8"),
    alignmentBytes: 4,
  };
}

function dispatch(
  artifacts: VerifiedViewCopyArtifacts,
  nodeId: string,
  source: string,
  destination: string,
  dependsOn: readonly string[],
) {
  return {
    nodeId,
    kind: "dispatch" as const,
    dependsOn,
    semanticArtifactHash: artifacts.kernelSemanticHash,
    entrypointId: artifacts.operationId,
    dimensionBindings: {},
    bindings: [
      {
        semanticResourceId: artifacts.source.viewId,
        graphResourceId: source,
      },
      {
        semanticResourceId: artifacts.destination.viewId,
        graphResourceId: destination,
      },
    ],
  };
}

function pipelineProgram(
  artifacts: VerifiedViewCopyArtifacts,
  dtype: "f32" | "i32" | "u32",
): HostGraphProgram {
  return {
    kind: "host-graph",
    version: { major: 1, minor: 0 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire("1"),
    resources: [
      resource("input", "input", dtype),
      resource("temporary", "temporary", dtype),
      resource("output", "output", dtype),
    ],
    nodes: [
      dispatch(artifacts, "first", "input", "temporary", []),
      dispatch(artifacts, "second", "temporary", "output", ["first"]),
    ],
  };
}

function collectiveProgram(
  artifacts: VerifiedViewCopyArtifacts,
  dtype: "f32" | "i32" | "u32",
  reduction: "sum" | "min" | "max",
): HostGraphProgram {
  const numericalPolicy = dtype === "f32"
    ? "rank-order-f32" as const
    : reduction === "sum"
      ? "rank-order-wrapping-32" as const
      : "exact-32-bit" as const;
  return {
    kind: "host-graph",
    version: { major: 1, minor: 0 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: wire("2"),
    resources: [
      resource("input", "input", dtype),
      resource("output", "output", dtype),
    ],
    nodes: [
      dispatch(artifacts, "copy", "input", "output", []),
      {
        nodeId: "reduce",
        kind: "all-reduce",
        dependsOn: ["copy"],
        resourceId: "output",
        reduction,
        dtype,
        numericalPolicy,
        participants: [wire("0"), wire("1")],
        result: "replicated-to-all-participants",
      },
    ],
  };
}

async function verified(
  program: HostGraphProgram,
  artifacts: VerifiedViewCopyArtifacts,
): Promise<VerifiedHostGraphArtifact> {
  return (await createVerifiedHostGraphArtifact(
    program,
    artifactOptions(artifacts),
  )).artifact;
}

function input(
  rank: number,
  bytes: Uint8Array,
): HostGraphCpuInputBinding {
  return {
    rank: wire(String(rank)),
    resourceId: "input",
    bytes,
  };
}

function f32Bytes(values: readonly number[]): Uint8Array {
  const result = new Uint8Array(values.length * 4);
  const view = new DataView(result.buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return result;
}

function i32Bytes(values: readonly number[]): Uint8Array {
  const result = new Uint8Array(values.length * 4);
  const view = new DataView(result.buffer);
  values.forEach((value, index) => view.setInt32(index * 4, value, true));
  return result;
}

function u32Bytes(values: readonly number[]): Uint8Array {
  const result = new Uint8Array(values.length * 4);
  const view = new DataView(result.buffer);
  values.forEach((value, index) => view.setUint32(index * 4, value, true));
  return result;
}

function readF32(bytes: Uint8Array): number[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Array.from(
    { length: bytes.byteLength / 4 },
    (_, index) => view.getFloat32(index * 4, true),
  );
}

function readI32(bytes: Uint8Array): number[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Array.from(
    { length: bytes.byteLength / 4 },
    (_, index) => view.getInt32(index * 4, true),
  );
}

function readU32(bytes: Uint8Array): number[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return Array.from(
    { length: bytes.byteLength / 4 },
    (_, index) => view.getUint32(index * 4, true),
  );
}

describe("host graph CPU reference", () => {
  it("executes an authority-bound multi-dispatch pipeline", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(pipelineProgram(artifacts, "f32"), artifacts);
    const prepared = await prepareHostGraphCpu(
      graph,
      artifactOptions(artifacts),
    );
    const source = f32Bytes([1.25, -2.5]);
    const result = await prepared.execute({ inputs: [input(0, source)] });

    expect(prepared).toMatchObject({
      profile: HOST_GRAPH_CPU_PROFILE,
      rankCount: 1n,
      inputResourceIds: ["input"],
      outputResourceIds: ["output"],
      elementOperations: 4n,
    });
    expect(result).toMatchObject({
      profile: HOST_GRAPH_CPU_PROFILE,
      failureModel: "fail-stop-no-partial-output-commit",
      executedNodeIds: ["first", "second"],
      elementOperations: "4",
    });
    expect(readF32(result.outputs[0]!.bytes)).toEqual([1.25, -2.5]);
  });

  it("reduces finite f32 in ascending participant-rank order", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(
      collectiveProgram(artifacts, "f32", "sum"),
      artifacts,
    );
    const prepared = await prepareHostGraphCpu(
      graph,
      artifactOptions(artifacts),
    );
    const result = await prepared.execute({
      inputs: [
        input(1, f32Bytes([2.25, 5])),
        input(0, f32Bytes([1.5, -2])),
      ],
    });

    expect(result.outputs.map((output) => output.rank)).toEqual(["0", "1"]);
    expect(result.outputs.map((output) => readF32(output.bytes))).toEqual([
      [3.75, 3],
      [3.75, 3],
    ]);
  });

  it("implements wrapping i32 sum and exact u32 max", async () => {
    const signed = await identityArtifacts("i32");
    const signedGraph = await verified(
      collectiveProgram(signed, "i32", "sum"),
      signed,
    );
    const signedCpu = await prepareHostGraphCpu(
      signedGraph,
      artifactOptions(signed),
    );
    const signedResult = await signedCpu.execute({
      inputs: [
        input(0, i32Bytes([2_147_483_647, -2])),
        input(1, i32Bytes([1, -3])),
      ],
    });
    expect(signedResult.outputs.map((output) => readI32(output.bytes))).toEqual([
      [-2_147_483_648, -5],
      [-2_147_483_648, -5],
    ]);

    const unsigned = await identityArtifacts("u32");
    const unsignedGraph = await verified(
      collectiveProgram(unsigned, "u32", "max"),
      unsigned,
    );
    const unsignedCpu = await prepareHostGraphCpu(
      unsignedGraph,
      artifactOptions(unsigned),
    );
    const unsignedResult = await unsignedCpu.execute({
      inputs: [
        input(0, u32Bytes([1, 0xffff_ffff])),
        input(1, u32Bytes([2, 5])),
      ],
    });
    expect(
      unsignedResult.outputs.map((output) => readU32(output.bytes)),
    ).toEqual([
      [2, 0xffff_ffff],
      [2, 0xffff_ffff],
    ]);
  });

  it("snapshots inputs and rejects non-finite collective values before commit", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(
      collectiveProgram(artifacts, "f32", "sum"),
      artifacts,
    );
    const prepared = await prepareHostGraphCpu(
      graph,
      artifactOptions(artifacts),
    );
    const first = f32Bytes([1, 2]);
    const second = f32Bytes([3, 4]);
    const pending = prepared.execute({
      inputs: [input(0, first), input(1, second)],
    });
    first.fill(0);
    second.fill(0);
    const result = await pending;
    expect(result.outputs.map((output) => readF32(output.bytes))).toEqual([
      [4, 6],
      [4, 6],
    ]);

    const nonFinite = f32Bytes([Number.NaN, 1]);
    const original = new Uint8Array(nonFinite);
    await expect(prepared.execute({
      inputs: [input(0, nonFinite), input(1, f32Bytes([1, 2]))],
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-NUMERICAL-DOMAIN",
    });
    expect(nonFinite).toEqual(original);
  });

  it("fails closed for incomplete, duplicate, shared, and misaligned inputs", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(
      collectiveProgram(artifacts, "f32", "sum"),
      artifacts,
    );
    const prepared = await prepareHostGraphCpu(
      graph,
      artifactOptions(artifacts),
    );
    await expect(prepared.execute({
      inputs: [input(0, f32Bytes([1, 2]))],
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-INVALID-BINDING",
    });
    await expect(prepared.execute({
      inputs: [
        input(0, f32Bytes([1, 2])),
        input(0, f32Bytes([3, 4])),
      ],
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-INVALID-BINDING",
    });

    if (typeof SharedArrayBuffer !== "undefined") {
      await expect(prepared.execute({
        inputs: [
          input(0, new Uint8Array(new SharedArrayBuffer(8))),
          input(1, f32Bytes([3, 4])),
        ],
      })).rejects.toMatchObject({
        code: "BG-GRAPH-CPU-UNSUPPORTED-PROFILE",
      });
    }
    const misaligned = new Uint8Array(new ArrayBuffer(9), 1, 8);
    await expect(prepared.execute({
      inputs: [
        input(0, misaligned),
        input(1, f32Bytes([3, 4])),
      ],
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-INVALID-BINDING",
    });
  });

  it("requires exact graph and semantic authority without invoking accessors", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(pipelineProgram(artifacts, "f32"), artifacts);
    const copied = JSON.parse(JSON.stringify(graph)) as
      VerifiedHostGraphArtifact;
    await expect(prepareHostGraphCpu(
      copied,
      artifactOptions(artifacts),
    )).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-INVALID-AUTHORITY",
    });
    await expect(prepareHostGraphCpu(graph, {
      kernelArtifacts: [],
      layoutArtifacts: [artifacts.layout],
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-INVALID-BINDING",
    });

    let reads = 0;
    const hostile = artifactOptions(artifacts);
    Object.defineProperty(hostile, "kernelArtifacts", {
      enumerable: true,
      get() {
        reads += 1;
        return [artifacts.kernel];
      },
    });
    await expect(prepareHostGraphCpu(graph, hostile))
      .rejects.toBeInstanceOf(HostGraphCpuError);
    expect(reads).toBe(0);
  });

  it("enforces preparation, execution, and cancellation budgets", async () => {
    const artifacts = await identityArtifacts();
    const graph = await verified(pipelineProgram(artifacts, "f32"), artifacts);
    await expect(prepareHostGraphCpu(graph, {
      ...artifactOptions(artifacts),
      maxWorkingBytes: 1,
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-RESOURCE-LIMIT",
    });
    await expect(prepareHostGraphCpu(graph, {
      ...artifactOptions(artifacts),
      maxElementOperations: 1,
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-RESOURCE-LIMIT",
    });

    const prepared = await prepareHostGraphCpu(
      graph,
      artifactOptions(artifacts),
    );
    const controller = new AbortController();
    controller.abort();
    await expect(prepared.execute({
      inputs: [input(0, f32Bytes([1, 2]))],
      signal: controller.signal,
    })).rejects.toMatchObject({
      code: "BG-GRAPH-CPU-ABORTED",
    });
  });
});
