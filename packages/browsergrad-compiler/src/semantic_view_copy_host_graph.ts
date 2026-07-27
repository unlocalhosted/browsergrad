import {
  HOST_GRAPH_MAX_NODES,
  createVerifiedHostGraphArtifact,
  type HostGraphProgram,
  type HostGraphResource,
  type VerifiedHostGraphArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/graph";
import type {
  VerifiedKernelArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import type {
  VerifiedLayoutArtifact,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import {
  encodeWireU64,
  parseWireU64,
  SemanticSchemaError,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CudaLiteViewCopyBindingError,
  unwrapPreparedCudaLiteViewCopyBinding,
  type PreparedCudaLiteViewCopyBinding,
} from "./semantic_view_copy_bindings.js";

export const CUDA_LITE_VIEW_COPY_HOST_GRAPH_PROFILE =
  "browsergrad.compiler.view-copy-host-graph@1" as const;

export interface ConstructedCudaLiteViewCopyHostGraph {
  readonly profile: typeof CUDA_LITE_VIEW_COPY_HOST_GRAPH_PROFILE;
  readonly artifact: VerifiedHostGraphArtifact;
  readonly graphSemanticHash: string;
  readonly dispatchCount: number;
  readonly inputResourceId: "input";
  readonly outputResourceId: "output";
  readonly temporaryResourceIds: readonly string[];
}

/**
 * Compiler-owned lowering from already prepared view-copy bindings into one
 * verified linear multi-dispatch host graph. Resource geometry, effects,
 * semantic view IDs, and dimension bindings are derived from opaque prepared
 * authority; callers cannot restate them.
 */
export async function createCudaLiteViewCopyHostGraph(
  preparedBindings: readonly PreparedCudaLiteViewCopyBinding[],
): Promise<ConstructedCudaLiteViewCopyHostGraph> {
  const bindings = snapshotPreparedBindings(preparedBindings);
  const records = bindings.map((prepared) =>
    unwrapPreparedCudaLiteViewCopyBinding(prepared));
  const kernelArtifacts = new Map<string, VerifiedKernelArtifact>();
  const layoutArtifacts = new Map<string, VerifiedLayoutArtifact>();
  for (const record of records) {
    const specialization = record.specialization;
    if (!kernelArtifacts.has(specialization.kernelSemanticHash)) {
      kernelArtifacts.set(
        specialization.kernelSemanticHash,
        record.kernelArtifact,
      );
    }
    if (!layoutArtifacts.has(specialization.layoutSemanticHash)) {
      layoutArtifacts.set(
        specialization.layoutSemanticHash,
        record.layoutArtifact,
      );
    }
  }

  const resources = pipelineResources(records);
  const temporaryResourceIds = Object.freeze(
    resources
      .filter((resource) => resource.role === "temporary")
      .map((resource) => resource.resourceId),
  );
  const program: HostGraphProgram = {
    kind: "host-graph",
    version: { major: 1, minor: 0 },
    failureModel: "fail-stop-no-partial-output-commit",
    rankCount: parseWireU64("1"),
    resources,
    nodes: records.map((record, index) => {
      const specialization = record.specialization;
      return {
        nodeId: nodeId(index),
        kind: "dispatch",
        dependsOn: index === 0 ? [] : [nodeId(index - 1)],
        semanticArtifactHash: specialization.kernelSemanticHash,
        entrypointId: specialization.operation.operationId,
        dimensionBindings: Object.fromEntries(
          Object.entries(specialization.bindings),
        ),
        bindings: [
          {
            semanticResourceId: specialization.operation.source.viewId,
            graphResourceId: resourceId(index, records.length),
          },
          {
            semanticResourceId:
              specialization.operation.destination.viewId,
            graphResourceId: resourceId(index + 1, records.length),
          },
        ],
      };
    }),
  };

  try {
    const constructed = await createVerifiedHostGraphArtifact(program, {
      kernelArtifacts: [...kernelArtifacts.values()],
      layoutArtifacts: [...layoutArtifacts.values()],
      producer: {
        id: "browsergrad.compiler.view-copy-host-graph",
        version: "1",
      },
      artifactId: "cuda-lite-view-copy-host-graph",
    });
    return Object.freeze({
      profile: CUDA_LITE_VIEW_COPY_HOST_GRAPH_PROFILE,
      artifact: constructed.artifact,
      graphSemanticHash: constructed.graphSemanticHash,
      dispatchCount: records.length,
      inputResourceId: "input",
      outputResourceId: "output",
      temporaryResourceIds,
    });
  } catch (cause) {
    const path = cause instanceof SemanticSchemaError
      ? cause.diagnostic.path ?? "$"
      : "$";
    throw new CudaLiteViewCopyBindingError(
      "BG-COMPILER-VIEW-COPY-BINDING-INVALID-ARTIFACT",
      path,
      `host graph construction failed: ${
        cause instanceof Error ? cause.message : "unknown semantic error"
      }`,
      { cause },
    );
  }
}

function pipelineResources(
  records: readonly ReturnType<
    typeof unwrapPreparedCudaLiteViewCopyBinding
  >[],
): readonly HostGraphResource[] {
  return Object.freeze(Array.from(
    { length: records.length + 1 },
    (_, index) => {
      const producer = index === 0
        ? undefined
        : records[index - 1]?.specialization.destination;
      const consumer = index === records.length
        ? undefined
        : records[index]?.specialization.source;
      const accessor = producer ?? consumer;
      if (accessor === undefined) {
        invalid("$", "prepared pipeline resource geometry disappeared");
      }
      if (producer !== undefined &&
          consumer !== undefined &&
          (producer.dtype !== consumer.dtype ||
           producer.allocationByteLength !==
             consumer.allocationByteLength)) {
        invalid(
          `$[${index}]`,
          "adjacent dispatches require identical intermediate dtype and allocation byte length",
        );
      }
      return Object.freeze({
        resourceId: resourceId(index, records.length),
        role: index === 0
          ? "input" as const
          : index === records.length
            ? "output" as const
            : "temporary" as const,
        multiplicity: "per-rank" as const,
        initialization: index === 0
          ? "external-input" as const
          : "zero-fill" as const,
        dtype: accessor.dtype,
        byteLength: encodeWireU64(accessor.allocationByteLength),
        alignmentBytes: Math.max(
          accessor.allocationAlignmentBytes,
          accessor.requiredAlignmentBytes,
          producer?.allocationAlignmentBytes ?? 1,
          producer?.requiredAlignmentBytes ?? 1,
          consumer?.allocationAlignmentBytes ?? 1,
          consumer?.requiredAlignmentBytes ?? 1,
        ),
      });
    },
  ));
}

function snapshotPreparedBindings(
  value: readonly PreparedCudaLiteViewCopyBinding[],
): readonly PreparedCudaLiteViewCopyBinding[] {
  if (!Array.isArray(value) ||
      Object.getPrototypeOf(value) !== Array.prototype ||
      value.length === 0 ||
      value.length > HOST_GRAPH_MAX_NODES) {
    invalid(
      "$",
      `prepared bindings must be a plain array with 1-${HOST_GRAPH_MAX_NODES} entries`,
    );
  }
  const descriptors = Object.getOwnPropertyDescriptors(value);
  const keys = Reflect.ownKeys(descriptors);
  if (keys.length !== value.length + 1 ||
      keys.some((key) =>
        key !== "length" &&
        (typeof key !== "string" ||
         !/^(?:0|[1-9][0-9]*)$/u.test(key) ||
         Number(key) >= value.length))) {
    invalid("$", "prepared bindings must be a dense data-only array");
  }
  return Object.freeze(Array.from(
    { length: value.length },
    (_, index) => {
      const descriptor = descriptors[String(index)];
      if (descriptor === undefined || !("value" in descriptor) ||
          descriptor.enumerable !== true) {
        invalid(
          `$[${index}]`,
          "prepared binding must be an enumerable data property",
        );
      }
      return descriptor.value as PreparedCudaLiteViewCopyBinding;
    },
  ));
}

function resourceId(index: number, dispatchCount: number): string {
  if (index === 0) return "input";
  if (index === dispatchCount) return "output";
  return `temporary/${index - 1}`;
}

function nodeId(index: number): string {
  return `dispatch/${index}`;
}

function invalid(path: string, message: string): never {
  throw new CudaLiteViewCopyBindingError(
    "BG-COMPILER-VIEW-COPY-BINDING-INVALID-REQUEST",
    path,
    message,
  );
}
