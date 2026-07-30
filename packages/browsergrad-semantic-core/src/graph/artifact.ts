import { canonicalizeJson, compareCanonicalStrings } from "../schema/canonical-json.js";
import {
  kernelArtifactPayload,
  type VerifiedKernelArtifact,
} from "../kernel/artifact.js";
import {
  BUILTIN_DTYPES,
  type BuiltinDTypeId,
} from "../layout/dtype.js";
import type { VerifiedLayoutArtifact } from "../layout/artifact.js";
import { prepareViewAccessor } from "../layout/prepare.js";
import {
  GRAPH_DIAGNOSTIC_CODES,
  SemanticSchemaError,
} from "../schema/diagnostics.js";
import {
  unwrapVerifiedArtifact,
  validateWireEnvelope,
  verifyWireArtifact,
  type VerifiedArtifact,
  type WireEnvelope,
} from "../schema/envelope.js";
import { hashSemanticArtifact } from "../schema/hash.js";
import {
  encodeWireU64,
  parseWireI64,
  parseWireU64,
  wireIntegerToBigInt,
  type WireI64,
  type WireU64,
} from "../schema/integers.js";
import {
  decodeWireJson,
  isJsonObject,
  type JsonArray,
  type JsonObject,
  type JsonValue,
} from "../schema/json.js";
import {
  resolveDecodeLimits,
  type DecodeLimits,
} from "../schema/limits.js";
import {
  HOST_GRAPH_FAILURE_MODEL,
  type HostGraphAllReduceNode,
  type HostGraphCollectiveDType,
  type HostGraphCollectiveNumericalPolicy,
  type HostGraphCollectiveReduction,
  type HostGraphConditionalBodyNode,
  type HostGraphConditionalNode,
  type HostGraphCopyNode,
  type HostGraphDynamicDispatchControl,
  type HostGraphDynamicExtentControl,
  type HostGraphDynamicDispatchNode,
  type HostGraphDispatchNode,
  type HostGraphDispatchResourceBinding,
  type HostGraphEventNode,
  type HostGraphExecutableNode,
  type HostGraphInputPredicate,
  type HostGraphMaterializeNode,
  type HostGraphNode,
  type HostGraphProgram,
  type HostGraphRepeatBodyNode,
  type HostGraphRepeatNode,
  type HostGraphResource,
  type HostGraphResourceConditionalNode,
  type HostGraphResourceDynamicDispatchNode,
  type HostGraphResourceDynamicDispatchSource,
  type HostGraphResourceDynamicExtentSource,
  type HostGraphResourceRectangularDynamicDispatchNode,
  type HostGraphResourcePredicate,
  type HostGraphResourceRepeatNode,
  type HostGraphResourceRepeatSource,
  type HostGraphResourceRole,
  type HostGraphRuntimeControlPredicate,
  type HostGraphRuntimeRepeatControl,
} from "./model.js";

export const HOST_GRAPH_ARTIFACT_SCHEMA = "browsergrad.host-graph";
export const HOST_GRAPH_ARTIFACT_MAJOR = 1;
export const HOST_GRAPH_ARTIFACT_MINOR = 30;
export const HOST_GRAPH_MAX_RESOURCES = 256;
export const HOST_GRAPH_MAX_NODES = 256;
export const HOST_GRAPH_MAX_EDGES = 4_096;
export const HOST_GRAPH_MAX_REPEAT_BODY_NODES = 64;
export const HOST_GRAPH_MAX_CONDITIONAL_BODY_NODES = 64;
export const HOST_GRAPH_MAX_REPEAT_ITERATIONS = 1_024;
export const HOST_GRAPH_MAX_RUNTIME_CONTROLS = 64;
export const HOST_GRAPH_MIN_RECTANGULAR_DYNAMIC_RANK = 2;
export const HOST_GRAPH_MAX_RECTANGULAR_DYNAMIC_RANK = 8;
const HOST_GRAPH_MAX_LEGACY_RECTANGULAR_DYNAMIC_RANK = 3;
const HOST_GRAPH_MAX_RANK_FOUR_RECTANGULAR_DYNAMIC_RANK = 4;
const HOST_GRAPH_MAX_RANK_FIVE_RECTANGULAR_DYNAMIC_RANK = 5;
const HOST_GRAPH_MAX_RANK_SIX_RECTANGULAR_DYNAMIC_RANK = 6;
const HOST_GRAPH_MAX_RANK_SEVEN_RECTANGULAR_DYNAMIC_RANK = 7;
export const HOST_GRAPH_MAX_RESOURCE_CONDITIONALS = 1;
export const HOST_GRAPH_MAX_RESOURCE_FEEDBACK_NODES = 4;
const HOST_GRAPH_MAX_THREE_STAGE_RESOURCE_FEEDBACK_NODES = 3;
const HOST_GRAPH_MAX_FANOUT_RESOURCE_FEEDBACK_NODES = 2;
const HOST_GRAPH_MAX_LEGACY_RESOURCE_FEEDBACK_NODES = 1;
export const HOST_GRAPH_MAX_EXPANDED_NODES = 16_384;
export const HOST_GRAPH_MAX_RANKS = 256;
export const HOST_GRAPH_MAX_SEMANTIC_ARTIFACTS = 256;
export const HOST_GRAPH_MAX_TOTAL_RESOURCE_BYTES = 1_073_741_824n;
export const HOST_GRAPH_MAX_ALIGNMENT_BYTES = 4_096;

const AUTHORITY = Object.freeze({
  schema: HOST_GRAPH_ARTIFACT_SCHEMA,
  major: HOST_GRAPH_ARTIFACT_MAJOR,
});
const ID = /^[A-Za-z][A-Za-z0-9._:/-]{0,127}$/u;
const HASH = /^[0-9a-f]{64}$/u;
const RESOURCE_ROLES = new Set<HostGraphResourceRole>([
  "input",
  "temporary",
  "output",
]);
const REDUCTIONS = new Set<HostGraphCollectiveReduction>([
  "sum",
  "min",
  "max",
]);
const COLLECTIVE_DTYPES = new Set<HostGraphCollectiveDType>([
  "f32",
  "i32",
  "u32",
]);
const GRAPH_DTYPES = new Set<BuiltinDTypeId>(
  Object.keys(BUILTIN_DTYPES) as BuiltinDTypeId[],
);
const ANALYSES = new WeakMap<object, HostGraphAnalysis>();
const PREPARED = new WeakSet<object>();

export type HostGraphArtifactPayloadV1 = JsonObject & {
  readonly program: HostGraphProgram;
};

export type VerifiedHostGraphArtifact =
  VerifiedArtifact<HostGraphArtifactPayloadV1>;

export interface HostGraphArtifactVerificationOptions {
  readonly kernelArtifacts: readonly VerifiedKernelArtifact[];
  readonly layoutArtifacts: readonly VerifiedLayoutArtifact[];
  readonly limits?: Partial<DecodeLimits>;
}

export interface PreparedHostGraphProgram {
  readonly artifact: VerifiedHostGraphArtifact;
  readonly graphSemanticHash: string;
  readonly failureModel: typeof HOST_GRAPH_FAILURE_MODEL;
  readonly rankCount: bigint;
  readonly resourceCount: number;
  readonly nodeCount: number;
  readonly edgeCount: number;
  readonly dispatchCount: number;
  readonly dynamicDispatchCount: number;
  readonly resourceDynamicDispatchCount: number;
  readonly collectiveCount: number;
  readonly copyCount: number;
  readonly materializationCount: number;
  readonly eventCount: number;
  readonly eventIds: readonly string[];
  readonly repeatCount: number;
  readonly repeatIterationCount: number;
  readonly runtimeRepeatCount: number;
  readonly resourceRepeatCount: number;
  readonly conditionalCount: number;
  readonly resourceConditionalCount: number;
  readonly runtimeControlIds: readonly string[];
  readonly expandedNodeCount: number;
  readonly topologicalNodeIds: readonly string[];
  readonly outputResourceIds: readonly string[];
}

interface HostGraphAnalysis {
  readonly rankCount: bigint;
  readonly edgeCount: number;
  readonly dispatchCount: number;
  readonly dynamicDispatchCount: number;
  readonly resourceDynamicDispatchCount: number;
  readonly collectiveCount: number;
  readonly copyCount: number;
  readonly materializationCount: number;
  readonly eventCount: number;
  readonly eventIds: readonly string[];
  readonly repeatCount: number;
  readonly repeatIterationCount: number;
  readonly runtimeRepeatCount: number;
  readonly resourceRepeatCount: number;
  readonly conditionalCount: number;
  readonly resourceConditionalCount: number;
  readonly runtimeControlIds: readonly string[];
  readonly expandedNodeCount: number;
  readonly topologicalNodeIds: readonly string[];
  readonly outputResourceIds: readonly string[];
}

interface ResourceUse {
  readonly nodeId: string;
  readonly access: HostGraphResourceAccess;
  readonly guaranteesWrite: boolean;
  readonly path: string;
}

type HostGraphResourceAccess = "read" | "write" | "read-write";

interface HostGraphResourceEffect {
  readonly resourceId: string;
  readonly access: HostGraphResourceAccess;
  readonly requiresPriorWriter?: boolean;
  readonly guaranteesWrite?: boolean;
}

interface DispatchOperationBinding {
  readonly layoutArtifact: VerifiedLayoutArtifact;
  readonly sourceSemanticResourceId: string;
  readonly destinationSemanticResourceId: string;
  readonly dtype: BuiltinDTypeId;
}

interface ResolvedDispatchGeometry {
  readonly logicalShape: readonly bigint[];
  readonly elementCount: bigint;
  readonly sourceByteLength: WireU64;
  readonly destinationByteLength: WireU64;
  readonly sourceAlignmentBytes: number;
  readonly destinationAlignmentBytes: number;
}

interface ParsedDispatchFields {
  readonly nodeId: string;
  readonly dependsOn: readonly string[];
  readonly semanticArtifactHash: string;
  readonly entrypointId: string;
  readonly dimensionBindings: Readonly<Record<string, WireI64>>;
  readonly bindings: readonly HostGraphDispatchResourceBinding[];
}

type SemanticArtifactCatalog = ReadonlyMap<
  string,
  ReadonlyMap<string, DispatchOperationBinding>
>;

export async function verifyHostGraphArtifact(
  value: unknown,
  options: HostGraphArtifactVerificationOptions,
): Promise<VerifiedHostGraphArtifact> {
  const limits = resolveDecodeLimits(options.limits);
  const kernelArtifacts = snapshotKernelArtifacts(
    options.kernelArtifacts,
  );
  const layoutArtifacts = snapshotLayoutArtifacts(
    options.layoutArtifacts,
  );
  const envelope = validateWireEnvelope(value, {
    schema: HOST_GRAPH_ARTIFACT_SCHEMA,
    supportedMajor: HOST_GRAPH_ARTIFACT_MAJOR,
    supportedMinor: HOST_GRAPH_ARTIFACT_MINOR,
    knownRequiredExtensions: new Set(),
    limits,
  });
  const semanticCatalog = await semanticArtifactCatalog(
    kernelArtifacts,
    layoutArtifacts,
    limits,
  );
  const program = parseProgram(envelope.payload, envelope.version.minor);
  const analysis = analyzeProgram(program, semanticCatalog, limits);
  const normalizedEnvelope: WireEnvelope<JsonValue> = {
    ...envelope,
    payload: { program } as JsonValue,
  };
  canonicalizeJson(normalizedEnvelope, { limits });
  const verified = verifyWireArtifact(normalizedEnvelope, {
    schema: HOST_GRAPH_ARTIFACT_SCHEMA,
    supportedMajor: HOST_GRAPH_ARTIFACT_MAJOR,
    supportedMinor: HOST_GRAPH_ARTIFACT_MINOR,
    knownRequiredExtensions: new Set(),
    limits,
    validatePayload: (payload) => payload,
  }, AUTHORITY) as VerifiedHostGraphArtifact;
  ANALYSES.set(verified as object, analysis);
  return verified;
}

export async function decodeHostGraphArtifact(
  bytes: Uint8Array,
  options: HostGraphArtifactVerificationOptions,
): Promise<VerifiedHostGraphArtifact> {
  return verifyHostGraphArtifact(
    decodeWireJson(
      bytes,
      options.limits === undefined ? {} : { limits: options.limits },
    ),
    options,
  );
}

export function hostGraphArtifactPayload(
  artifact: VerifiedHostGraphArtifact,
): HostGraphArtifactPayloadV1 {
  const envelope = unwrapVerifiedArtifact(artifact, AUTHORITY);
  if (envelope.schema !== HOST_GRAPH_ARTIFACT_SCHEMA ||
      envelope.version.major !== HOST_GRAPH_ARTIFACT_MAJOR) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidArtifact,
      "$",
      "verified artifact is not a browsergrad.host-graph@1 artifact",
    );
  }
  return envelope.payload;
}

export async function prepareHostGraphProgram(
  artifact: VerifiedHostGraphArtifact,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<PreparedHostGraphProgram> {
  hostGraphArtifactPayload(artifact);
  const analysis = ANALYSES.get(artifact as object);
  if (analysis === undefined) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidArtifact,
      "$",
      "host graph preparation requires verifier-issued artifact authority",
    );
  }
  const limits = resolveDecodeLimits(options.limits);
  const payload = hostGraphArtifactPayload(artifact);
  const prepared = Object.freeze({
    artifact,
    graphSemanticHash: await hashSemanticArtifact(artifact, { limits }),
    failureModel: payload.program.failureModel,
    rankCount: analysis.rankCount,
    resourceCount: payload.program.resources.length,
    nodeCount: payload.program.nodes.length,
    edgeCount: analysis.edgeCount,
    dispatchCount: analysis.dispatchCount,
    dynamicDispatchCount: analysis.dynamicDispatchCount,
    resourceDynamicDispatchCount: analysis.resourceDynamicDispatchCount,
    collectiveCount: analysis.collectiveCount,
    copyCount: analysis.copyCount,
    materializationCount: analysis.materializationCount,
    eventCount: analysis.eventCount,
    eventIds: Object.freeze([...analysis.eventIds]),
    repeatCount: analysis.repeatCount,
    repeatIterationCount: analysis.repeatIterationCount,
    runtimeRepeatCount: analysis.runtimeRepeatCount,
    resourceRepeatCount: analysis.resourceRepeatCount,
    conditionalCount: analysis.conditionalCount,
    resourceConditionalCount: analysis.resourceConditionalCount,
    runtimeControlIds: Object.freeze([...analysis.runtimeControlIds]),
    expandedNodeCount: analysis.expandedNodeCount,
    topologicalNodeIds: Object.freeze([...analysis.topologicalNodeIds]),
    outputResourceIds: Object.freeze([...analysis.outputResourceIds]),
  });
  PREPARED.add(prepared);
  return prepared;
}

export function requirePreparedHostGraphProgram(
  prepared: PreparedHostGraphProgram,
): void {
  if (typeof prepared !== "object" || prepared === null ||
      !PREPARED.has(prepared as object)) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      "$",
      "host graph execution requires an exact prepared graph from this module instance",
    );
  }
}

function parseProgram(
  value: JsonValue,
  envelopeMinor: number,
): HostGraphProgram {
  const payload = closedObject(value, ["program"], "$.payload");
  const object = closedObject(
    field(payload, "program", "$.payload"),
    [
      "kind",
      "version",
      "failureModel",
      "rankCount",
      "resources",
      "nodes",
    ],
    "$.payload.program",
  );
  if (object.kind !== "host-graph") {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidArtifact,
      "$.payload.program.kind",
      "host graph kind must be host-graph",
    );
  }
  const version = closedObject(
    field(object, "version", "$.payload.program"),
    ["major", "minor"],
    "$.payload.program.version",
  );
  if (
    version.major !== 1 ||
    typeof version.minor !== "number" ||
    !Number.isInteger(version.minor) ||
    version.minor < 0 ||
    version.minor > HOST_GRAPH_ARTIFACT_MINOR
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      "$.payload.program.version",
      "host graph program reader supports versions 1.0 through 1.30 only",
    );
  }
  const programMinor =
    version.minor as HostGraphProgram["version"]["minor"];
  if (programMinor !== envelopeMinor) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidArtifact,
      "$.payload.program.version",
      "host graph program minor version must match its artifact envelope",
    );
  }
  if (object.failureModel !== HOST_GRAPH_FAILURE_MODEL) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      "$.payload.program.failureModel",
      `initial host graph profile requires ${HOST_GRAPH_FAILURE_MODEL}`,
    );
  }
  const rankCount = positiveWire(
    field(object, "rankCount", "$.payload.program"),
    "$.payload.program.rankCount",
  );
  if (wireIntegerToBigInt(rankCount) > BigInt(HOST_GRAPH_MAX_RANKS)) {
    resource(
      "$.payload.program.rankCount",
      `rank count exceeds ${HOST_GRAPH_MAX_RANKS}`,
    );
  }
  const resources = parseResources(
    field(object, "resources", "$.payload.program"),
    wireIntegerToBigInt(rankCount),
  );
  const nodes = parseNodes(
    field(object, "nodes", "$.payload.program"),
    programMinor,
  );
  return {
    kind: "host-graph",
    version: { major: 1, minor: programMinor },
    failureModel: HOST_GRAPH_FAILURE_MODEL,
    rankCount,
    resources,
    nodes,
  };
}

function parseResources(
  value: JsonValue,
  rankCount: bigint,
): readonly HostGraphResource[] {
  const values = arrayValue(value, "$.payload.program.resources");
  if (values.length === 0 || values.length > HOST_GRAPH_MAX_RESOURCES) {
    resource(
      "$.payload.program.resources",
      `resource count must be between 1 and ${HOST_GRAPH_MAX_RESOURCES}`,
    );
  }
  let totalResourceBytes = 0n;
  const resources = values.map((item, index) => {
    const path = `$.payload.program.resources[${index}]`;
    const object = closedObject(
      item,
      [
        "resourceId",
        "role",
        "multiplicity",
        "initialization",
        "dtype",
        "byteLength",
        "alignmentBytes",
      ],
      path,
    );
    const resourceId = identifier(
      field(object, "resourceId", path),
      `${path}.resourceId`,
    );
    const role = stringValue(field(object, "role", path), `${path}.role`);
    if (!RESOURCE_ROLES.has(role as HostGraphResourceRole)) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.role`,
        "resource role must be input, temporary, or output",
      );
    }
    if (object.multiplicity !== "per-rank") {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.multiplicity`,
        "initial host graph resources require per-rank multiplicity",
      );
    }
    const expectedInitialization: HostGraphResource["initialization"] =
      role === "input"
      ? "external-input"
      : "zero-fill";
    if (object.initialization !== expectedInitialization) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.initialization`,
        `${role} resources require ${expectedInitialization} initialization`,
      );
    }
    const dtype = stringValue(field(object, "dtype", path), `${path}.dtype`);
    if (!GRAPH_DTYPES.has(dtype as BuiltinDTypeId)) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.dtype`,
        `resource dtype must be one of ${[...GRAPH_DTYPES].join(", ")}`,
      );
    }
    const byteLength = positiveOrZeroWire(
      field(object, "byteLength", path),
      `${path}.byteLength`,
    );
    totalResourceBytes += wireIntegerToBigInt(byteLength) * rankCount;
    if (totalResourceBytes > HOST_GRAPH_MAX_TOTAL_RESOURCE_BYTES) {
      resource(
        "$.payload.program.resources",
        `aggregate resource bytes across ranks exceed ${HOST_GRAPH_MAX_TOTAL_RESOURCE_BYTES}`,
      );
    }
    const alignmentBytes = alignment(
      field(object, "alignmentBytes", path),
      `${path}.alignmentBytes`,
    );
    return {
      resourceId,
      role: role as HostGraphResourceRole,
      multiplicity: "per-rank" as const,
      initialization: expectedInitialization,
      dtype: dtype as BuiltinDTypeId,
      byteLength,
      alignmentBytes,
    };
  }).sort((left, right) =>
    compareCanonicalStrings(left.resourceId, right.resourceId));
  unique(
    resources.map((item) => item.resourceId),
    "$.payload.program.resources",
    "resource",
  );
  if (!resources.some((item) => item.role === "output")) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidArtifact,
      "$.payload.program.resources",
      "host graph requires at least one output resource",
    );
  }
  return resources;
}

function parseNodes(
  value: JsonValue,
  programMinor: HostGraphProgram["version"]["minor"],
): readonly HostGraphNode[] {
  const values = arrayValue(value, "$.payload.program.nodes");
  if (values.length === 0 || values.length > HOST_GRAPH_MAX_NODES) {
    resource(
      "$.payload.program.nodes",
      `node count must be between 1 and ${HOST_GRAPH_MAX_NODES}`,
    );
  }
  const nodes = values.map((item, index) =>
    parseNode(
      item,
      `$.payload.program.nodes[${index}]`,
      programMinor,
    ))
    .sort((left, right) =>
      compareCanonicalStrings(left.nodeId, right.nodeId));
  unique(
    nodes.map((node) => node.nodeId),
    "$.payload.program.nodes",
    "node",
  );
  unique(
    nodes.flatMap((node) => [
      node.nodeId,
      ...(node.kind === "repeat"
        ? node.body.map((bodyNode) => bodyNode.nodeId)
        : node.kind === "conditional"
          ? [
              ...node.thenBody.map((bodyNode) => bodyNode.nodeId),
              ...node.elseBody.map((bodyNode) => bodyNode.nodeId),
            ]
        : []),
    ]),
    "$.payload.program.nodes",
    "top-level or control-body node",
  );
  unique(
    nodes
      .filter((node): node is HostGraphEventNode => node.kind === "event")
      .map((node) => node.eventId),
    "$.payload.program.nodes",
    "event",
  );
  return nodes;
}

function parseNode(
  value: JsonValue,
  path: string,
  programMinor: HostGraphProgram["version"]["minor"],
  controlBody?: string,
): HostGraphNode {
  const object = objectValue(value, path);
  const kind = stringValue(field(object, "kind", path), `${path}.kind`);
  if (kind === "dispatch") return parseDispatchNode(object, path);
  if (kind === "dynamic-dispatch") {
    if (controlBody !== undefined) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.kind`,
        `${controlBody} bodies cannot contain dynamic dispatch`,
      );
    }
    if (programMinor < 9) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.kind`,
        "dynamic dispatch nodes require host graph program version 1.9",
      );
    }
    return parseDynamicDispatchNode(object, path, programMinor);
  }
  if (kind === "all-reduce") return parseAllReduceNode(object, path);
  if (kind === "copy") {
    if (programMinor < 1) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.kind`,
        "copy nodes require host graph program version 1.1",
      );
    }
    return parseCopyNode(object, path);
  }
  if (kind === "materialize") {
    if (controlBody !== undefined) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.kind`,
        `${controlBody} bodies cannot contain materialization`,
      );
    }
    if (programMinor < 2) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.kind`,
        "materialize nodes require host graph program version 1.2",
      );
    }
    return parseMaterializeNode(object, path);
  }
  if (kind === "event") {
    if (controlBody !== undefined) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.kind`,
        `${controlBody} bodies cannot contain events`,
      );
    }
    if (programMinor < 3) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.kind`,
        "event nodes require host graph program version 1.3",
      );
    }
    return parseEventNode(object, path);
  }
  if (kind === "repeat") {
    if (controlBody !== undefined) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.kind`,
        `${controlBody} bodies cannot contain nested control`,
      );
    }
    if (programMinor < 4) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.kind`,
        "repeat nodes require host graph program version 1.4",
      );
    }
    return parseRepeatNode(object, path, programMinor);
  }
  if (kind === "conditional") {
    if (controlBody !== undefined) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.kind`,
        `${controlBody} bodies cannot contain nested control`,
      );
    }
    if (programMinor < 5) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.kind`,
        "conditional nodes require host graph program version 1.5",
      );
    }
    return parseConditionalNode(object, path, programMinor);
  }
  invalid(
    GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
    `${path}.kind`,
    "host graph profile supports dispatch, all-reduce, version-1.1 copy, version-1.2 materialize, version-1.3 event, version-1.4 fixed repeat, version-1.5 through 1.7 conditional, version-1.8 runtime repeat, version-1.9 dynamic dispatch, version-1.10 resource repeat, version-1.11 resource dynamic dispatch, version-1.12 runtime rectangular dynamic dispatch, version-1.13 resource rectangular dynamic dispatch, version-1.14 request-time rank-4 rectangular dispatch, version-1.15 produced-resource rank-4 rectangular dispatch, version-1.16 request-time rank-5 rectangular dispatch, version-1.17 produced-resource rank-5 rectangular dispatch, version-1.18 request-time rank-6 rectangular dispatch, version-1.19 produced-resource rank-6 rectangular dispatch, version-1.20 request-time rank-7 rectangular dispatch, version-1.21 produced-resource rank-7 rectangular dispatch, version-1.22 request-time rank-8 rectangular dispatch, version-1.23 produced-resource rank-8 rectangular dispatch, version-1.24 shared produced-resource linear-dispatch fanout, version-1.25 shared produced-resource rectangular-dispatch fanout, version-1.26 two-stage produced-resource linear feedback, version-1.27 three-stage produced-resource linear feedback, version-1.28 four-stage produced-resource linear feedback, version-1.29 shared conditional/repeat feedback, and version-1.30 sequential conditional-to-repeat feedback nodes",
  );
}

function parseRepeatNode(
  value: JsonObject,
  path: string,
  programMinor: HostGraphProgram["version"]["minor"],
): HostGraphRepeatNode {
  const mode = stringValue(field(value, "mode", path), `${path}.mode`);
  if (mode === "resource-u32-count-sequential") {
    if (programMinor < 10) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.mode`,
        "resource count repeats require host graph program version 1.10",
      );
    }
    const object = closedObject(
      value,
      [
        "nodeId",
        "kind",
        "dependsOn",
        "iterationSource",
        "maxIterationCount",
        "body",
        "mode",
      ],
      path,
    );
    const maxIterationCount = positiveWire(
      field(object, "maxIterationCount", path),
      `${path}.maxIterationCount`,
    );
    verifyRepeatIterationBound(
      maxIterationCount,
      `${path}.maxIterationCount`,
    );
    return {
      nodeId: identifier(field(object, "nodeId", path), `${path}.nodeId`),
      kind: "repeat",
      dependsOn: dependencies(field(object, "dependsOn", path), path),
      iterationSource: parseResourceRepeatSource(
        field(object, "iterationSource", path),
        `${path}.iterationSource`,
      ),
      maxIterationCount,
      body: parseLinearControlBody(
        field(object, "body", path),
        `${path}.body`,
        "resource count repeat",
        HOST_GRAPH_MAX_REPEAT_BODY_NODES,
      ) as readonly HostGraphRepeatBodyNode[],
      mode: "resource-u32-count-sequential",
    };
  }
  if (mode === "runtime-u32-count-sequential") {
    if (programMinor < 8) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.mode`,
        "runtime control repeats require host graph program version 1.8",
      );
    }
    const object = closedObject(
      value,
      [
        "nodeId",
        "kind",
        "dependsOn",
        "iterationControl",
        "maxIterationCount",
        "body",
        "mode",
      ],
      path,
    );
    const maxIterationCount = positiveWire(
      field(object, "maxIterationCount", path),
      `${path}.maxIterationCount`,
    );
    verifyRepeatIterationBound(
      maxIterationCount,
      `${path}.maxIterationCount`,
    );
    return {
      nodeId: identifier(field(object, "nodeId", path), `${path}.nodeId`),
      kind: "repeat",
      dependsOn: dependencies(field(object, "dependsOn", path), path),
      iterationControl: parseRuntimeRepeatControl(
        field(object, "iterationControl", path),
        `${path}.iterationControl`,
      ),
      maxIterationCount,
      body: parseLinearControlBody(
        field(object, "body", path),
        `${path}.body`,
        "runtime control repeat",
        HOST_GRAPH_MAX_REPEAT_BODY_NODES,
      ) as readonly HostGraphRepeatBodyNode[],
      mode: "runtime-u32-count-sequential",
    };
  }
  const object = closedObject(
    value,
    ["nodeId", "kind", "dependsOn", "iterationCount", "body", "mode"],
    path,
  );
  if (mode !== "fixed-count-sequential") {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.mode`,
      "repeat mode must be fixed-count-sequential, runtime-u32-count-sequential, or resource-u32-count-sequential",
    );
  }
  const iterationCount = positiveWire(
    field(object, "iterationCount", path),
    `${path}.iterationCount`,
  );
  verifyRepeatIterationBound(iterationCount, `${path}.iterationCount`);
  const body = parseLinearControlBody(
    field(object, "body", path),
    `${path}.body`,
    "fixed-count repeat",
    HOST_GRAPH_MAX_REPEAT_BODY_NODES,
  ) as readonly HostGraphRepeatBodyNode[];
  return {
    nodeId: identifier(field(object, "nodeId", path), `${path}.nodeId`),
    kind: "repeat",
    dependsOn: dependencies(field(object, "dependsOn", path), path),
    iterationCount,
    body,
    mode: "fixed-count-sequential",
  };
}

function parseResourceRepeatSource(
  value: JsonValue,
  path: string,
): HostGraphResourceRepeatSource {
  const object = closedObject(
    value,
    ["resourceId", "rank", "mode"],
    path,
  );
  if (object.mode !== "u32-count") {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.mode`,
      "resource repeat source mode must be u32-count",
    );
  }
  return {
    resourceId: identifier(
      field(object, "resourceId", path),
      `${path}.resourceId`,
    ),
    rank: parseWireU64(field(object, "rank", path), `${path}.rank`),
    mode: "u32-count",
  };
}

function verifyRepeatIterationBound(
  value: WireU64,
  path: string,
): void {
  if (
    wireIntegerToBigInt(value) >
    BigInt(HOST_GRAPH_MAX_REPEAT_ITERATIONS)
  ) {
    resource(
      path,
      `repeat iteration count exceeds ${HOST_GRAPH_MAX_REPEAT_ITERATIONS}`,
    );
  }
}

function parseRuntimeRepeatControl(
  value: JsonValue,
  path: string,
): HostGraphRuntimeRepeatControl {
  const object = closedObject(value, ["controlId", "mode"], path);
  if (object.mode !== "u32-count") {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.mode`,
      "runtime repeat control mode must be u32-count",
    );
  }
  return {
    controlId: identifier(
      field(object, "controlId", path),
      `${path}.controlId`,
    ),
    mode: "u32-count",
  };
}

function parseConditionalNode(
  value: JsonObject,
  path: string,
  programMinor: HostGraphProgram["version"]["minor"],
): HostGraphConditionalNode {
  const object = closedObject(
    value,
    [
      "nodeId",
      "kind",
      "dependsOn",
      "predicate",
      "thenBody",
      "elseBody",
      "mode",
    ],
    path,
  );
  if (
    object.mode !== "input-u32-branch-sequential" &&
    object.mode !== "runtime-u32-branch-sequential" &&
    object.mode !== "resource-u32-branch-sequential"
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.mode`,
      "conditional mode must be input-u32-branch-sequential, runtime-u32-branch-sequential, or resource-u32-branch-sequential",
    );
  }
  if (
    object.mode === "runtime-u32-branch-sequential" &&
    programMinor < 6
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.mode`,
      "runtime control conditionals require host graph program version 1.6",
    );
  }
  if (
    object.mode === "resource-u32-branch-sequential" &&
    programMinor < 7
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.mode`,
      "resource conditionals require host graph program version 1.7",
    );
  }
  const label = object.mode === "input-u32-branch-sequential"
    ? "input conditional"
    : object.mode === "runtime-u32-branch-sequential"
      ? "runtime control conditional"
      : "resource conditional";
  const thenBody = parseLinearControlBody(
    field(object, "thenBody", path),
    `${path}.thenBody`,
    label,
    HOST_GRAPH_MAX_CONDITIONAL_BODY_NODES,
  ) as readonly HostGraphConditionalBodyNode[];
  const elseBody = parseLinearControlBody(
    field(object, "elseBody", path),
    `${path}.elseBody`,
    label,
    HOST_GRAPH_MAX_CONDITIONAL_BODY_NODES,
  ) as readonly HostGraphConditionalBodyNode[];
  if (
    thenBody.length !== elseBody.length ||
    thenBody.some((node, index) => node.kind !== elseBody[index]?.kind)
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.elseBody`,
      "conditional branches require equal-length executable-kind structure",
    );
  }
  const common = {
    nodeId: identifier(field(object, "nodeId", path), `${path}.nodeId`),
    kind: "conditional" as const,
    dependsOn: dependencies(field(object, "dependsOn", path), path),
    thenBody,
    elseBody,
  };
  if (object.mode === "input-u32-branch-sequential") {
    return {
      ...common,
      predicate: parseInputPredicate(
        field(object, "predicate", path),
        `${path}.predicate`,
      ),
      mode: "input-u32-branch-sequential",
    };
  }
  if (object.mode === "runtime-u32-branch-sequential") {
    return {
      ...common,
      predicate: parseRuntimeControlPredicate(
        field(object, "predicate", path),
        `${path}.predicate`,
      ),
      mode: "runtime-u32-branch-sequential",
    };
  }
  return {
    ...common,
    predicate: parseResourcePredicate(
      field(object, "predicate", path),
      `${path}.predicate`,
    ),
    mode: "resource-u32-branch-sequential",
  };
}

function parseInputPredicate(
  value: JsonValue,
  path: string,
): HostGraphInputPredicate {
  const object = closedObject(value, ["resourceId", "rank", "mode"], path);
  if (object.mode !== "u32-nonzero") {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.mode`,
      "input predicate mode must be u32-nonzero",
    );
  }
  return {
    resourceId: identifier(
      field(object, "resourceId", path),
      `${path}.resourceId`,
    ),
    rank: parseWireU64(field(object, "rank", path), `${path}.rank`),
    mode: "u32-nonzero",
  };
}

function parseRuntimeControlPredicate(
  value: JsonValue,
  path: string,
): HostGraphRuntimeControlPredicate {
  const object = closedObject(value, ["controlId", "mode"], path);
  if (object.mode !== "u32-nonzero") {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.mode`,
      "runtime control predicate mode must be u32-nonzero",
    );
  }
  return {
    controlId: identifier(
      field(object, "controlId", path),
      `${path}.controlId`,
    ),
    mode: "u32-nonzero",
  };
}

function parseResourcePredicate(
  value: JsonValue,
  path: string,
): HostGraphResourcePredicate {
  const object = closedObject(value, ["resourceId", "rank", "mode"], path);
  if (object.mode !== "u32-nonzero") {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.mode`,
      "resource predicate mode must be u32-nonzero",
    );
  }
  return {
    resourceId: identifier(
      field(object, "resourceId", path),
      `${path}.resourceId`,
    ),
    rank: parseWireU64(field(object, "rank", path), `${path}.rank`),
    mode: "u32-nonzero",
  };
}

function parseLinearControlBody(
  value: JsonValue,
  path: string,
  label: string,
  maxNodes: number,
): readonly HostGraphExecutableNode[] {
  const values = arrayValue(value, path);
  if (values.length === 0 || values.length > maxNodes) {
    resource(
      path,
      `${label} body node count must be between 1 and ${maxNodes}`,
    );
  }
  const body = values.map((item, index) => {
    const node = parseNode(item, `${path}[${index}]`, 5, label);
    if (
      node.kind === "materialize" ||
      node.kind === "event" ||
      node.kind === "repeat" ||
      node.kind === "conditional"
    ) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}[${index}].kind`,
        `${label} bodies support dispatch, all-reduce, and copy only`,
      );
    }
    return node;
  }) as HostGraphExecutableNode[];
  unique(body.map((node) => node.nodeId), path, `${label} body node`);
  for (const [index, node] of body.entries()) {
    const expected = index === 0 ? [] : [body[index - 1]!.nodeId];
    if (
      node.dependsOn.length !== expected.length ||
      node.dependsOn.some((dependency, dependencyIndex) =>
        dependency !== expected[dependencyIndex])
    ) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.invalidAccess,
        `${path}[${index}].dependsOn`,
        index === 0
          ? `first ${label} body node must have no internal dependency`
          : `${label} body node must depend only on immediately preceding node ${expected[0]}`,
      );
    }
  }
  return Object.freeze(body);
}

function parseEventNode(
  value: JsonObject,
  path: string,
): HostGraphEventNode {
  const object = closedObject(
    value,
    ["nodeId", "kind", "dependsOn", "eventId", "mode"],
    path,
  );
  if (object.mode !== "completion-after-dependencies") {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.mode`,
      "event mode must be completion-after-dependencies",
    );
  }
  return {
    nodeId: identifier(field(object, "nodeId", path), `${path}.nodeId`),
    kind: "event",
    dependsOn: dependencies(field(object, "dependsOn", path), path),
    eventId: identifier(
      field(object, "eventId", path),
      `${path}.eventId`,
    ),
    mode: "completion-after-dependencies",
  };
}

function parseMaterializeNode(
  value: JsonObject,
  path: string,
): HostGraphMaterializeNode {
  const object = closedObject(
    value,
    ["nodeId", "kind", "dependsOn", "resourceId", "mode"],
    path,
  );
  if (object.mode !== "host-readback-after-graph-success") {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.mode`,
      "materialize mode must be host-readback-after-graph-success",
    );
  }
  return {
    nodeId: identifier(field(object, "nodeId", path), `${path}.nodeId`),
    kind: "materialize",
    dependsOn: dependencies(field(object, "dependsOn", path), path),
    resourceId: identifier(
      field(object, "resourceId", path),
      `${path}.resourceId`,
    ),
    mode: "host-readback-after-graph-success",
  };
}

function parseCopyNode(
  value: JsonObject,
  path: string,
): HostGraphCopyNode {
  const object = closedObject(
    value,
    [
      "nodeId",
      "kind",
      "dependsOn",
      "sourceResourceId",
      "destinationResourceId",
      "mode",
    ],
    path,
  );
  if (object.mode !== "whole-allocation-bytes-per-rank") {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.mode`,
      "copy mode must be whole-allocation-bytes-per-rank",
    );
  }
  return {
    nodeId: identifier(field(object, "nodeId", path), `${path}.nodeId`),
    kind: "copy",
    dependsOn: dependencies(field(object, "dependsOn", path), path),
    sourceResourceId: identifier(
      field(object, "sourceResourceId", path),
      `${path}.sourceResourceId`,
    ),
    destinationResourceId: identifier(
      field(object, "destinationResourceId", path),
      `${path}.destinationResourceId`,
    ),
    mode: "whole-allocation-bytes-per-rank",
  };
}

function parseDispatchNode(
  value: JsonObject,
  path: string,
): HostGraphDispatchNode {
  const object = closedObject(
    value,
    [
      "nodeId",
      "kind",
      "dependsOn",
      "semanticArtifactHash",
      "entrypointId",
      "dimensionBindings",
      "bindings",
    ],
    path,
  );
  return {
    ...parseDispatchFields(object, path),
    kind: "dispatch",
  };
}

function parseDynamicDispatchNode(
  value: JsonObject,
  path: string,
  programMinor: HostGraphProgram["version"]["minor"],
): HostGraphDynamicDispatchNode {
  const mode = stringValue(field(value, "mode", path), `${path}.mode`);
  if (mode === "resource-u32-rectangular-prefix") {
    if (programMinor < 13) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.mode`,
        "resource rectangular dynamic dispatch requires host graph program version 1.13",
      );
    }
    const object = closedObject(
      value,
      [
        "nodeId",
        "kind",
        "dependsOn",
        "semanticArtifactHash",
        "entrypointId",
        "dimensionBindings",
        "bindings",
        "launchSources",
        "maxExtents",
        "mode",
      ],
      path,
    );
    const maxExtents = parseRectangularDynamicMaximum(
      field(object, "maxExtents", path),
      `${path}.maxExtents`,
      programMinor >= 23
        ? HOST_GRAPH_MAX_RECTANGULAR_DYNAMIC_RANK
        : programMinor >= 21
          ? HOST_GRAPH_MAX_RANK_SEVEN_RECTANGULAR_DYNAMIC_RANK
          : programMinor >= 19
            ? HOST_GRAPH_MAX_RANK_SIX_RECTANGULAR_DYNAMIC_RANK
            : programMinor >= 17
              ? HOST_GRAPH_MAX_RANK_FIVE_RECTANGULAR_DYNAMIC_RANK
              : programMinor >= 15
                ? HOST_GRAPH_MAX_RANK_FOUR_RECTANGULAR_DYNAMIC_RANK
                : HOST_GRAPH_MAX_LEGACY_RECTANGULAR_DYNAMIC_RANK,
    );
    return {
      ...parseDispatchFields(object, path),
      kind: "dynamic-dispatch",
      launchSources: parseResourceRectangularDynamicSources(
        field(object, "launchSources", path),
        `${path}.launchSources`,
        maxExtents.length,
      ),
      maxExtents,
      mode: "resource-u32-rectangular-prefix",
    };
  }
  if (mode === "runtime-u32-rectangular-prefix") {
    if (programMinor < 12) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.mode`,
        "rectangular dynamic dispatch requires host graph program version 1.12",
      );
    }
    const object = closedObject(
      value,
      [
        "nodeId",
        "kind",
        "dependsOn",
        "semanticArtifactHash",
        "entrypointId",
        "dimensionBindings",
        "bindings",
        "launchControls",
        "maxExtents",
        "mode",
      ],
      path,
    );
    const maxExtents = parseRectangularDynamicMaximum(
      field(object, "maxExtents", path),
      `${path}.maxExtents`,
      programMinor >= 22
        ? HOST_GRAPH_MAX_RECTANGULAR_DYNAMIC_RANK
        : programMinor >= 20
          ? HOST_GRAPH_MAX_RANK_SEVEN_RECTANGULAR_DYNAMIC_RANK
          : programMinor >= 18
            ? HOST_GRAPH_MAX_RANK_SIX_RECTANGULAR_DYNAMIC_RANK
            : programMinor >= 16
              ? HOST_GRAPH_MAX_RANK_FIVE_RECTANGULAR_DYNAMIC_RANK
              : programMinor >= 14
                ? HOST_GRAPH_MAX_RANK_FOUR_RECTANGULAR_DYNAMIC_RANK
                : HOST_GRAPH_MAX_LEGACY_RECTANGULAR_DYNAMIC_RANK,
    );
    return {
      ...parseDispatchFields(object, path),
      kind: "dynamic-dispatch",
      launchControls: parseRectangularDynamicControls(
        field(object, "launchControls", path),
        `${path}.launchControls`,
        maxExtents.length,
      ),
      maxExtents,
      mode: "runtime-u32-rectangular-prefix",
    };
  }
  if (mode === "resource-u32-prefix-elements") {
    if (programMinor < 11) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${path}.mode`,
        "resource dynamic dispatch requires host graph program version 1.11",
      );
    }
    const object = closedObject(
      value,
      [
        "nodeId",
        "kind",
        "dependsOn",
        "semanticArtifactHash",
        "entrypointId",
        "dimensionBindings",
        "bindings",
        "launchSource",
        "maxElementCount",
        "mode",
      ],
      path,
    );
    return {
      ...parseDispatchFields(object, path),
      kind: "dynamic-dispatch",
      launchSource: parseResourceDynamicDispatchSource(
        field(object, "launchSource", path),
        `${path}.launchSource`,
      ),
      maxElementCount: parseDynamicDispatchMaximum(object, path),
      mode: "resource-u32-prefix-elements",
    };
  }
  const object = closedObject(
    value,
    [
      "nodeId",
      "kind",
      "dependsOn",
      "semanticArtifactHash",
      "entrypointId",
      "dimensionBindings",
      "bindings",
      "launchControl",
      "maxElementCount",
      "mode",
    ],
    path,
  );
  if (object.mode !== "runtime-u32-prefix-elements") {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.mode`,
      "dynamic dispatch mode must be runtime-u32-prefix-elements, resource-u32-prefix-elements, runtime-u32-rectangular-prefix, or resource-u32-rectangular-prefix",
    );
  }
  return {
    ...parseDispatchFields(object, path),
    kind: "dynamic-dispatch",
    launchControl: parseDynamicDispatchControl(
      field(object, "launchControl", path),
      `${path}.launchControl`,
    ),
    maxElementCount: parseDynamicDispatchMaximum(object, path),
    mode: "runtime-u32-prefix-elements",
  };
}

function parseRectangularDynamicMaximum(
  value: JsonValue,
  path: string,
  maxRank: number,
): readonly WireU64[] {
  const values = arrayValue(value, path);
  if (
    values.length < HOST_GRAPH_MIN_RECTANGULAR_DYNAMIC_RANK ||
    values.length > maxRank
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      path,
      `rectangular dynamic dispatch rank must be between ${HOST_GRAPH_MIN_RECTANGULAR_DYNAMIC_RANK} and ${maxRank}`,
    );
  }
  return Object.freeze(values.map((extent, axis) => {
    const parsed = positiveWire(extent, `${path}[${axis}]`);
    if (wireIntegerToBigInt(parsed) > 0xffff_ffffn) {
      resource(
        `${path}[${axis}]`,
        "rectangular dynamic dispatch extent exceeds the u32 control domain",
      );
    }
    return parsed;
  }));
}

function parseRectangularDynamicControls(
  value: JsonValue,
  path: string,
  rank: number,
): readonly HostGraphDynamicExtentControl[] {
  const values = arrayValue(value, path);
  if (values.length !== rank) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidArtifact,
      path,
      `rectangular dynamic dispatch requires exactly ${rank} axis controls`,
    );
  }
  const controls = values.map((candidate, index) => {
    const controlPath = `${path}[${index}]`;
    const object = closedObject(
      candidate,
      ["axis", "controlId", "mode"],
      controlPath,
    );
    if (object.mode !== "u32-prefix-extent") {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${controlPath}.mode`,
        "rectangular dynamic dispatch controls must use u32-prefix-extent",
      );
    }
    const axis = rectangularDynamicAxis(
      field(object, "axis", controlPath),
      `${controlPath}.axis`,
      rank,
    );
    return {
      axis,
      controlId: identifier(
        field(object, "controlId", controlPath),
        `${controlPath}.controlId`,
      ),
      mode: "u32-prefix-extent" as const,
    };
  }).sort((left, right) => left.axis - right.axis);
  if (controls.some((control, axis) => control.axis !== axis)) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.duplicateId,
      path,
      "rectangular dynamic dispatch must bind every axis exactly once",
    );
  }
  unique(
    controls.map((control) => control.controlId),
    path,
    "rectangular dynamic dispatch control",
  );
  return Object.freeze(controls.map((control) => Object.freeze(control)));
}

function parseResourceRectangularDynamicSources(
  value: JsonValue,
  path: string,
  rank: number,
): readonly HostGraphResourceDynamicExtentSource[] {
  const values = arrayValue(value, path);
  if (values.length !== rank) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidArtifact,
      path,
      `resource rectangular dynamic dispatch requires exactly ${rank} axis sources`,
    );
  }
  const sources = values.map((candidate, index) => {
    const sourcePath = `${path}[${index}]`;
    const object = closedObject(
      candidate,
      ["axis", "resourceId", "rank", "mode"],
      sourcePath,
    );
    if (object.mode !== "u32-prefix-extent") {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        `${sourcePath}.mode`,
        "resource rectangular dynamic dispatch sources must use u32-prefix-extent",
      );
    }
    return {
      axis: rectangularDynamicAxis(
        field(object, "axis", sourcePath),
        `${sourcePath}.axis`,
        rank,
      ),
      resourceId: identifier(
        field(object, "resourceId", sourcePath),
        `${sourcePath}.resourceId`,
      ),
      rank: parseWireU64(
        field(object, "rank", sourcePath),
        `${sourcePath}.rank`,
      ),
      mode: "u32-prefix-extent" as const,
    };
  }).sort((left, right) => left.axis - right.axis);
  if (sources.some((source, axis) => source.axis !== axis)) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.duplicateId,
      path,
      "resource rectangular dynamic dispatch must bind every axis exactly once",
    );
  }
  unique(
    sources.map((source) => source.resourceId),
    path,
    "resource rectangular dynamic dispatch source resource",
  );
  return Object.freeze(sources.map((source) => Object.freeze(source)));
}

function rectangularDynamicAxis(
  value: JsonValue,
  path: string,
  rank: number,
): number {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < 0 ||
    value >= rank
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidArtifact,
      path,
      `rectangular dynamic dispatch axis must be an integer in [0, ${rank})`,
    );
  }
  return value;
}

function parseDynamicDispatchMaximum(
  object: JsonObject,
  path: string,
): WireU64 {
  const maxElementCount = positiveWire(
    field(object, "maxElementCount", path),
    `${path}.maxElementCount`,
  );
  if (wireIntegerToBigInt(maxElementCount) > 0xffff_ffffn) {
    resource(
      `${path}.maxElementCount`,
      "dynamic dispatch maximum exceeds the u32 control domain",
    );
  }
  return maxElementCount;
}

function parseResourceDynamicDispatchSource(
  value: JsonValue,
  path: string,
): HostGraphResourceDynamicDispatchSource {
  const object = closedObject(
    value,
    ["resourceId", "rank", "mode"],
    path,
  );
  if (object.mode !== "u32-prefix-element-count") {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.mode`,
      "resource dynamic dispatch source mode must be u32-prefix-element-count",
    );
  }
  return {
    resourceId: identifier(
      field(object, "resourceId", path),
      `${path}.resourceId`,
    ),
    rank: parseWireU64(field(object, "rank", path), `${path}.rank`),
    mode: "u32-prefix-element-count",
  };
}

function parseDynamicDispatchControl(
  value: JsonValue,
  path: string,
): HostGraphDynamicDispatchControl {
  const object = closedObject(value, ["controlId", "mode"], path);
  if (object.mode !== "u32-prefix-element-count") {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.mode`,
      "dynamic dispatch control mode must be u32-prefix-element-count",
    );
  }
  return {
    controlId: identifier(
      field(object, "controlId", path),
      `${path}.controlId`,
    ),
    mode: "u32-prefix-element-count",
  };
}

function parseDispatchFields(
  object: JsonObject,
  path: string,
): ParsedDispatchFields {
  const bindings = arrayValue(
    field(object, "bindings", path),
    `${path}.bindings`,
  )
    .map((item, index) =>
      parseDispatchBinding(item, `${path}.bindings[${index}]`))
    .sort((left, right) =>
      compareCanonicalStrings(
        left.semanticResourceId,
        right.semanticResourceId,
      ));
  if (bindings.length === 0 ||
      bindings.length > HOST_GRAPH_MAX_RESOURCES) {
    resource(
      `${path}.bindings`,
      `dispatch binding count must be between 1 and ${HOST_GRAPH_MAX_RESOURCES}`,
    );
  }
  unique(
    bindings.map((binding) => binding.semanticResourceId),
    `${path}.bindings`,
    "semantic resource binding",
  );
  return {
    nodeId: identifier(field(object, "nodeId", path), `${path}.nodeId`),
    dependsOn: dependencies(field(object, "dependsOn", path), path),
    semanticArtifactHash: hashValue(
      field(object, "semanticArtifactHash", path),
      `${path}.semanticArtifactHash`,
    ),
    entrypointId: identifier(
      field(object, "entrypointId", path),
      `${path}.entrypointId`,
    ),
    dimensionBindings: parseDimensionBindings(
      field(object, "dimensionBindings", path),
      `${path}.dimensionBindings`,
    ),
    bindings,
  };
}

function parseDimensionBindings(
  value: JsonValue,
  path: string,
): Readonly<Record<string, WireI64>> {
  const object = objectValue(value, path);
  const result: Record<string, WireI64> = Object.create(null) as
    Record<string, WireI64>;
  const keys = Object.keys(object).sort(compareCanonicalStrings);
  if (keys.length > HOST_GRAPH_MAX_RESOURCES) {
    resource(
      path,
      `dimension binding count exceeds ${HOST_GRAPH_MAX_RESOURCES}`,
    );
  }
  for (const key of keys) {
    if (!ID.test(key)) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.invalidArtifact,
        path,
        "dimension binding keys must use the closed identifier grammar",
      );
    }
    result[key] = parseWireI64(object[key], `${path}.${key}`);
  }
  return Object.freeze(result);
}

function parseDispatchBinding(
  value: JsonValue,
  path: string,
): HostGraphDispatchResourceBinding {
  const object = closedObject(
    value,
    ["semanticResourceId", "graphResourceId"],
    path,
  );
  return {
    semanticResourceId: identifier(
      field(object, "semanticResourceId", path),
      `${path}.semanticResourceId`,
    ),
    graphResourceId: identifier(
      field(object, "graphResourceId", path),
      `${path}.graphResourceId`,
    ),
  };
}

function parseAllReduceNode(
  value: JsonObject,
  path: string,
): HostGraphAllReduceNode {
  const object = closedObject(
    value,
    [
      "nodeId",
      "kind",
      "dependsOn",
      "resourceId",
      "reduction",
      "dtype",
      "numericalPolicy",
      "participants",
      "result",
    ],
    path,
  );
  const reduction = stringValue(
    field(object, "reduction", path),
    `${path}.reduction`,
  );
  if (!REDUCTIONS.has(reduction as HostGraphCollectiveReduction)) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.reduction`,
      "all-reduce reduction must be sum, min, or max",
    );
  }
  const dtype = stringValue(field(object, "dtype", path), `${path}.dtype`);
  if (!COLLECTIVE_DTYPES.has(dtype as HostGraphCollectiveDType)) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.dtype`,
      "all-reduce dtype must be f32, i32, or u32",
    );
  }
  const numericalPolicy = stringValue(
    field(object, "numericalPolicy", path),
    `${path}.numericalPolicy`,
  );
  const expectedPolicy = collectiveNumericalPolicy(
    dtype as HostGraphCollectiveDType,
    reduction as HostGraphCollectiveReduction,
  );
  if (numericalPolicy !== expectedPolicy) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.numericalPolicy`,
      `all-reduce ${dtype} ${reduction} requires ${expectedPolicy}`,
    );
  }
  const participants = parseParticipants(
    field(object, "participants", path),
    `${path}.participants`,
  );
  if (object.result !== "replicated-to-all-participants") {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.result`,
      "all-reduce result must be replicated-to-all-participants",
    );
  }
  return {
    nodeId: identifier(field(object, "nodeId", path), `${path}.nodeId`),
    kind: "all-reduce",
    dependsOn: dependencies(field(object, "dependsOn", path), path),
    resourceId: identifier(
      field(object, "resourceId", path),
      `${path}.resourceId`,
    ),
    reduction: reduction as HostGraphCollectiveReduction,
    dtype: dtype as HostGraphCollectiveDType,
    numericalPolicy: numericalPolicy as HostGraphCollectiveNumericalPolicy,
    participants,
    result: "replicated-to-all-participants",
  };
}

function parseParticipants(
  value: JsonValue,
  path: string,
): readonly WireU64[] {
  const participants = arrayValue(value, path).map((item, index) =>
    positiveOrZeroWire(item, `${path}[${index}]`))
    .sort((left, right) => {
      const a = wireIntegerToBigInt(left);
      const b = wireIntegerToBigInt(right);
      return a < b ? -1 : a > b ? 1 : 0;
    });
  if (participants.length < 2 ||
      participants.length > HOST_GRAPH_MAX_RANKS) {
    resource(
      path,
      `all-reduce participant count must be between 2 and ${HOST_GRAPH_MAX_RANKS}`,
    );
  }
  unique(participants, path, "participant");
  return participants;
}

function dependencies(
  value: JsonValue,
  nodePath: string,
): readonly string[] {
  const path = `${nodePath}.dependsOn`;
  const dependencies = arrayValue(value, path).map((item, index) =>
    identifier(item, `${path}[${index}]`))
    .sort(compareCanonicalStrings);
  if (dependencies.length > HOST_GRAPH_MAX_NODES) {
    resource(path, `dependency count exceeds ${HOST_GRAPH_MAX_NODES}`);
  }
  unique(dependencies, path, "dependency");
  return dependencies;
}

function analyzeProgram(
  program: HostGraphProgram,
  semanticCatalog: SemanticArtifactCatalog,
  limits: DecodeLimits,
): HostGraphAnalysis {
  const rankCount = wireIntegerToBigInt(program.rankCount);
  const resources = new Map(
    program.resources.map((resource) => [resource.resourceId, resource]),
  );
  const nodes = new Map(program.nodes.map((node) => [node.nodeId, node]));
  let edgeCount = 0;
  let dispatchCount = 0;
  let dynamicDispatchCount = 0;
  let resourceDynamicDispatchCount = 0;
  let collectiveCount = 0;
  let copyCount = 0;
  let materializationCount = 0;
  let eventCount = 0;
  let repeatCount = 0;
  let repeatIterationCount = 0;
  let runtimeRepeatCount = 0;
  let resourceRepeatCount = 0;
  let conditionalCount = 0;
  let resourceConditionalCount = 0;
  const runtimeControlIds = new Set<string>();
  let expandedNodeCount = 0;
  const dispatchEffects = new Map<
    HostGraphDispatchNode | HostGraphDynamicDispatchNode,
    readonly HostGraphResourceEffect[]
  >();
  const repeatEffects = new Map<
    HostGraphRepeatNode,
    readonly HostGraphResourceEffect[]
  >();
  const conditionalEffects = new Map<
    HostGraphConditionalNode,
    readonly HostGraphResourceEffect[]
  >();
  const dispatchGeometry = new Map<string, ResolvedDispatchGeometry>();
  for (const [index, node] of program.nodes.entries()) {
    const path = `$.payload.program.nodes[${index}]`;
    const candidates = node.kind === "repeat"
      ? node.body.map((candidate, bodyIndex) => ({
          candidate,
          candidatePath: `${path}.body[${bodyIndex}]`,
        }))
      : node.kind === "conditional"
        ? [
            ...node.thenBody.map((candidate, bodyIndex) => ({
              candidate,
              candidatePath: `${path}.thenBody[${bodyIndex}]`,
            })),
            ...node.elseBody.map((candidate, bodyIndex) => ({
              candidate,
              candidatePath: `${path}.elseBody[${bodyIndex}]`,
            })),
          ]
        : [{ candidate: node, candidatePath: path }];
    for (const { candidate, candidatePath } of candidates) {
      if (candidate.kind === "all-reduce") {
        verifyCollectiveBinding(
          candidate,
          resources,
          rankCount,
          candidatePath,
        );
      } else if (candidate.kind === "copy") {
        verifyCopyBinding(candidate, resources, candidatePath);
      } else if (candidate.kind === "materialize") {
        verifyMaterializeBinding(candidate, resources, candidatePath);
      }
    }
  }
  for (const [index, node] of program.nodes.entries()) {
    const path = `$.payload.program.nodes[${index}]`;
    edgeCount += node.dependsOn.length;
    if (edgeCount > HOST_GRAPH_MAX_EDGES) {
      resource(
        "$.payload.program.nodes[*].dependsOn",
        `dependency edge count exceeds ${HOST_GRAPH_MAX_EDGES}`,
      );
    }
    for (const dependency of node.dependsOn) {
      if (dependency === node.nodeId) {
        invalid(
          GRAPH_DIAGNOSTIC_CODES.cycle,
          `${path}.dependsOn`,
          `node ${node.nodeId} cannot depend on itself`,
        );
      }
      if (!nodes.has(dependency)) {
        invalid(
          GRAPH_DIAGNOSTIC_CODES.danglingReference,
          `${path}.dependsOn`,
          `node ${node.nodeId} depends on missing node ${dependency}`,
        );
      }
    }
    if (
      node.kind === "dispatch" ||
      node.kind === "dynamic-dispatch" ||
      node.kind === "all-reduce" ||
      node.kind === "copy"
    ) {
      const effects = verifyExecutableNodeBinding(
        node,
        resources,
        rankCount,
        semanticCatalog,
        limits,
        dispatchGeometry,
        path,
      );
      if (node.kind === "dynamic-dispatch") {
        dynamicDispatchCount += 1;
        if (
          node.mode === "runtime-u32-prefix-elements" ||
          node.mode === "runtime-u32-rectangular-prefix"
        ) {
          const controlIds = node.mode === "runtime-u32-prefix-elements"
            ? [node.launchControl.controlId]
            : node.launchControls.map((control) => control.controlId);
          for (const controlId of controlIds) {
            runtimeControlIds.add(controlId);
          }
          if (runtimeControlIds.size > HOST_GRAPH_MAX_RUNTIME_CONTROLS) {
            resource(
              node.mode === "runtime-u32-prefix-elements"
                ? `${path}.launchControl.controlId`
                : `${path}.launchControls`,
              `runtime control count exceeds ${HOST_GRAPH_MAX_RUNTIME_CONTROLS}`,
            );
          }
          dispatchEffects.set(node, effects);
        } else {
          resourceDynamicDispatchCount += 1;
          const launchSources =
            node.mode === "resource-u32-prefix-elements"
              ? [node.launchSource]
              : node.launchSources;
          for (const [sourceIndex, source] of launchSources.entries()) {
            verifyResourceDynamicDispatchSource(
              source,
              resources,
              rankCount,
              node.mode === "resource-u32-prefix-elements"
                ? `${path}.launchSource`
                : `${path}.launchSources[${sourceIndex}]`,
            );
          }
          verifyResourceFeedbackBound(
            resourceConditionalCount +
              resourceRepeatCount +
              resourceDynamicDispatchCount,
            `${path}.mode`,
            program.version.minor,
          );
          dispatchEffects.set(
            node,
            resourceDynamicDispatchEffects(node, effects, path),
          );
        }
      } else if (node.kind === "dispatch") {
        dispatchEffects.set(node, effects);
      }
      ({ dispatchCount, collectiveCount, copyCount } =
        addExecutableCounts(
          node,
          1,
          dispatchCount,
          collectiveCount,
          copyCount,
        ));
      expandedNodeCount += 1;
    } else if (node.kind === "materialize") {
      verifyMaterializeBinding(node, resources, path);
      materializationCount += 1;
      expandedNodeCount += 1;
    } else if (node.kind === "event") {
      eventCount += 1;
      expandedNodeCount += 1;
    } else if (node.kind === "repeat") {
      repeatCount += 1;
      const iterationCount = Number(wireIntegerToBigInt(
        node.mode === "fixed-count-sequential"
          ? node.iterationCount
          : node.maxIterationCount,
      ));
      repeatIterationCount += iterationCount;
      if (node.mode === "runtime-u32-count-sequential") {
        runtimeRepeatCount += 1;
        runtimeControlIds.add(node.iterationControl.controlId);
        if (runtimeControlIds.size > HOST_GRAPH_MAX_RUNTIME_CONTROLS) {
          resource(
            `${path}.iterationControl.controlId`,
            `runtime control count exceeds ${HOST_GRAPH_MAX_RUNTIME_CONTROLS}`,
          );
        }
      } else if (node.mode === "resource-u32-count-sequential") {
        resourceRepeatCount += 1;
        verifyResourceRepeatSource(
          node.iterationSource,
          resources,
          rankCount,
          `${path}.iterationSource`,
        );
        verifyResourceFeedbackBound(
          resourceConditionalCount +
            resourceRepeatCount +
            resourceDynamicDispatchCount,
          `${path}.mode`,
          program.version.minor,
        );
      }
      const bodyExpandedNodeCount = iterationCount * node.body.length;
      expandedNodeCount += bodyExpandedNodeCount;
      if (expandedNodeCount > HOST_GRAPH_MAX_EXPANDED_NODES) {
        resource(
          `${path}.iterationCount`,
          `expanded node count exceeds ${HOST_GRAPH_MAX_EXPANDED_NODES}`,
        );
      }
      const bodyEffects = new Map<
        HostGraphRepeatBodyNode,
        readonly HostGraphResourceEffect[]
      >();
      for (const [bodyIndex, bodyNode] of node.body.entries()) {
        const bodyPath = `${path}.body[${bodyIndex}]`;
        edgeCount += bodyNode.dependsOn.length;
        if (edgeCount > HOST_GRAPH_MAX_EDGES) {
          resource(
            `${path}.body[*].dependsOn`,
            `dependency edge count exceeds ${HOST_GRAPH_MAX_EDGES}`,
          );
        }
        const effects = verifyExecutableNodeBinding(
          bodyNode,
          resources,
          rankCount,
          semanticCatalog,
          limits,
          dispatchGeometry,
          bodyPath,
        );
        bodyEffects.set(bodyNode, effects);
        if (bodyNode.kind === "dispatch") {
          dispatchEffects.set(bodyNode, effects);
        }
        ({ dispatchCount, collectiveCount, copyCount } =
          addExecutableCounts(
            bodyNode,
            iterationCount,
            dispatchCount,
            collectiveCount,
            copyCount,
          ));
      }
      const effects = aggregateLinearEffects(
        node.body,
        bodyEffects,
        "repeat",
      );
      repeatEffects.set(
        node,
        node.mode === "fixed-count-sequential"
          ? effects
          : node.mode === "runtime-u32-count-sequential"
            ? runtimeRepeatEffects(effects)
            : resourceRepeatEffects(node, effects, path),
      );
    } else {
      conditionalCount += 1;
      verifyConditionalPredicate(node, resources, rankCount, path);
      if (node.mode === "resource-u32-branch-sequential") {
        resourceConditionalCount += 1;
        if (
          resourceConditionalCount >
          HOST_GRAPH_MAX_RESOURCE_CONDITIONALS
        ) {
          resource(
            `${path}.mode`,
            `resource conditional count exceeds ${HOST_GRAPH_MAX_RESOURCE_CONDITIONALS}`,
          );
        }
        verifyResourceFeedbackBound(
          resourceConditionalCount +
            resourceRepeatCount +
            resourceDynamicDispatchCount,
          `${path}.mode`,
          program.version.minor,
        );
      }
      if (node.mode === "runtime-u32-branch-sequential") {
        runtimeControlIds.add(node.predicate.controlId);
        if (runtimeControlIds.size > HOST_GRAPH_MAX_RUNTIME_CONTROLS) {
          resource(
            `${path}.predicate.controlId`,
            `runtime control count exceeds ${HOST_GRAPH_MAX_RUNTIME_CONTROLS}`,
          );
        }
      }
      expandedNodeCount += node.thenBody.length;
      const branchEffects: Array<readonly HostGraphResourceEffect[]> = [];
      for (const [branchName, body] of [
        ["thenBody", node.thenBody],
        ["elseBody", node.elseBody],
      ] as const) {
        const bodyEffects = new Map<
          HostGraphConditionalBodyNode,
          readonly HostGraphResourceEffect[]
        >();
        for (const [bodyIndex, bodyNode] of body.entries()) {
          const bodyPath = `${path}.${branchName}[${bodyIndex}]`;
          edgeCount += bodyNode.dependsOn.length;
          if (edgeCount > HOST_GRAPH_MAX_EDGES) {
            resource(
              `${path}.${branchName}[*].dependsOn`,
              `dependency edge count exceeds ${HOST_GRAPH_MAX_EDGES}`,
            );
          }
          const effects = verifyExecutableNodeBinding(
            bodyNode,
            resources,
            rankCount,
            semanticCatalog,
            limits,
            dispatchGeometry,
            bodyPath,
          );
          bodyEffects.set(bodyNode, effects);
          if (bodyNode.kind === "dispatch") {
            dispatchEffects.set(bodyNode, effects);
          }
          if (branchName === "thenBody") {
            ({ dispatchCount, collectiveCount, copyCount } =
              addExecutableCounts(
                bodyNode,
                1,
                dispatchCount,
                collectiveCount,
                copyCount,
              ));
          }
        }
        branchEffects.push(aggregateLinearEffects(
          body,
          bodyEffects,
          `conditional ${branchName}`,
        ));
      }
      conditionalEffects.set(node, mergeConditionalEffects(
        node,
        branchEffects[0]!,
        branchEffects[1]!,
      ));
    }
    if (expandedNodeCount > HOST_GRAPH_MAX_EXPANDED_NODES) {
      resource(
        "$.payload.program.nodes",
        `expanded node count exceeds ${HOST_GRAPH_MAX_EXPANDED_NODES}`,
      );
    }
  }
  verifyResourceFeedbackProfile(
    program,
    resourceConditionalCount,
    resourceRepeatCount,
    resourceDynamicDispatchCount,
    dispatchEffects,
    conditionalEffects,
  );
  const topologicalNodeIds = topologicalOrder(program.nodes, nodes);
  const ancestors = dependencyAncestors(topologicalNodeIds, nodes);
  verifyMaterializationContract(program);
  verifyEffects(
    program.nodes,
    topologicalNodeIds,
    nodes,
    resources,
    ancestors,
    dispatchEffects,
    repeatEffects,
    conditionalEffects,
  );
  return Object.freeze({
    rankCount,
    edgeCount,
    dispatchCount,
    dynamicDispatchCount,
    resourceDynamicDispatchCount,
    collectiveCount,
    copyCount,
    materializationCount,
    eventCount,
    eventIds: Object.freeze(topologicalNodeIds.flatMap((nodeId) => {
      const node = nodes.get(nodeId);
      return node?.kind === "event" ? [node.eventId] : [];
    })),
    repeatCount,
    repeatIterationCount,
    runtimeRepeatCount,
    resourceRepeatCount,
    conditionalCount,
    resourceConditionalCount,
    runtimeControlIds: Object.freeze(
      [...runtimeControlIds].sort(compareCanonicalStrings),
    ),
    expandedNodeCount,
    topologicalNodeIds: Object.freeze(topologicalNodeIds),
    outputResourceIds: Object.freeze(
      program.version.minor < 2
        ? program.resources
          .filter((resource) => resource.role === "output")
          .map((resource) => resource.resourceId)
        : program.nodes
          .filter((node): node is HostGraphMaterializeNode =>
            node.kind === "materialize")
          .map((node) => node.resourceId)
          .sort(compareCanonicalStrings),
    ),
  });
}

function verifyExecutableNodeBinding(
  node: HostGraphExecutableNode | HostGraphDynamicDispatchNode,
  resources: ReadonlyMap<string, HostGraphResource>,
  rankCount: bigint,
  semanticCatalog: SemanticArtifactCatalog,
  limits: DecodeLimits,
  dispatchGeometry: Map<string, ResolvedDispatchGeometry>,
  path: string,
): readonly HostGraphResourceEffect[] {
  if (node.kind === "all-reduce") {
    verifyCollectiveBinding(node, resources, rankCount, path);
    return Object.freeze([Object.freeze({
      resourceId: node.resourceId,
      access: "read-write" as const,
    })]);
  }
  if (node.kind === "copy") {
    verifyCopyBinding(node, resources, path);
    return Object.freeze([
      Object.freeze({
        resourceId: node.sourceResourceId,
        access: "read" as const,
      }),
      Object.freeze({
        resourceId: node.destinationResourceId,
        access: "write" as const,
      }),
    ]);
  }
  const operations = semanticCatalog.get(node.semanticArtifactHash);
  if (operations === undefined) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.semanticArtifactMismatch,
      `${path}.semanticArtifactHash`,
      "dispatch references no supplied opaque verified semantic artifact",
    );
  }
  const operation = operations.get(node.entrypointId);
  if (operation === undefined) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.semanticArtifactMismatch,
      `${path}.entrypointId`,
      "dispatch entrypoint is absent from its supplied verified semantic artifact",
    );
  }
  const bindings = new Map(
    node.bindings.map((binding) => [
      binding.semanticResourceId,
      binding.graphResourceId,
    ]),
  );
  if (
    bindings.size !== 2 ||
    !bindings.has(operation.sourceSemanticResourceId) ||
    !bindings.has(operation.destinationSemanticResourceId)
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.semanticArtifactMismatch,
      `${path}.bindings`,
      "initial dispatch profile requires exact source and destination view bindings from the verified view-copy operation",
    );
  }
  const sourceResourceId = bindings.get(operation.sourceSemanticResourceId);
  const destinationResourceId =
    bindings.get(operation.destinationSemanticResourceId);
  if (sourceResourceId === undefined || destinationResourceId === undefined) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.semanticArtifactMismatch,
      `${path}.bindings`,
      "view-copy resource binding disappeared during verification",
    );
  }
  if (sourceResourceId === destinationResourceId) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidAccess,
      `${path}.bindings`,
      "view-copy source and destination must bind distinct graph resources",
    );
  }
  const geometryKey = `${node.semanticArtifactHash}\0${node.entrypointId}\0${
    canonicalizeJson(node.dimensionBindings, { limits })
  }`;
  let geometry = dispatchGeometry.get(geometryKey);
  if (geometry === undefined) {
    geometry = resolveDispatchGeometry(operation, node, limits, path);
    dispatchGeometry.set(geometryKey, geometry);
  }
  if (
    node.kind === "dynamic-dispatch" &&
    node.mode !== "runtime-u32-rectangular-prefix" &&
    node.mode !== "resource-u32-rectangular-prefix" &&
    wireIntegerToBigInt(node.maxElementCount) > geometry.elementCount
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      `${path}.maxElementCount`,
      `dynamic dispatch maximum exceeds semantic element count ${geometry.elementCount}`,
    );
  }
  if (
    node.kind === "dynamic-dispatch" &&
    (
      node.mode === "runtime-u32-rectangular-prefix" ||
      node.mode === "resource-u32-rectangular-prefix"
    )
  ) {
    verifyRectangularDynamicGeometry(node, geometry, path);
  }
  const sourceResource = resources.get(sourceResourceId);
  const destinationResource = resources.get(destinationResourceId);
  if (sourceResource === undefined || destinationResource === undefined) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.danglingReference,
      `${path}.bindings`,
      "view-copy dispatch binds a missing graph resource",
    );
  }
  if (
    sourceResource.dtype !== operation.dtype ||
    destinationResource.dtype !== operation.dtype
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      `${path}.bindings`,
      `view-copy ${operation.dtype} operation requires matching source and destination resource dtypes`,
    );
  }
  if (
    sourceResource.byteLength !== geometry.sourceByteLength ||
    destinationResource.byteLength !== geometry.destinationByteLength
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      `${path}.bindings`,
      "view-copy graph resources must preserve the exact verified allocation byte lengths",
    );
  }
  if (
    sourceResource.alignmentBytes < geometry.sourceAlignmentBytes ||
    sourceResource.alignmentBytes % geometry.sourceAlignmentBytes !== 0 ||
    destinationResource.alignmentBytes <
      geometry.destinationAlignmentBytes ||
    destinationResource.alignmentBytes %
      geometry.destinationAlignmentBytes !== 0
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      `${path}.bindings`,
      "view-copy graph resources do not satisfy verified allocation alignment",
    );
  }
  return Object.freeze([
    Object.freeze({
      resourceId: sourceResourceId,
      access: "read" as const,
    }),
    Object.freeze({
      resourceId: destinationResourceId,
      access: "write" as const,
    }),
  ]);
}

function addExecutableCounts(
  node: HostGraphExecutableNode | HostGraphDynamicDispatchNode,
  multiplier: number,
  dispatchCount: number,
  collectiveCount: number,
  copyCount: number,
): Readonly<{
  dispatchCount: number;
  collectiveCount: number;
  copyCount: number;
}> {
  return {
    dispatchCount:
      dispatchCount +
      (node.kind === "dispatch" || node.kind === "dynamic-dispatch"
        ? multiplier
        : 0),
    collectiveCount:
      collectiveCount + (node.kind === "all-reduce" ? multiplier : 0),
    copyCount: copyCount + (node.kind === "copy" ? multiplier : 0),
  };
}

function aggregateLinearEffects(
  body: readonly HostGraphExecutableNode[],
  bodyEffects: ReadonlyMap<
    HostGraphExecutableNode,
    readonly HostGraphResourceEffect[]
  >,
  label: string,
): readonly HostGraphResourceEffect[] {
  const aggregate = new Map<string, {
    read: boolean;
    write: boolean;
    requiresPriorWriter: boolean;
  }>();
  for (const node of body) {
    const effects = bodyEffects.get(node);
    if (effects === undefined) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.semanticArtifactMismatch,
        "$.payload.program.nodes",
        `${label} body effects for ${node.nodeId} were not derived`,
      );
    }
    for (const effect of effects) {
      const state = aggregate.get(effect.resourceId) ?? {
        read: false,
        write: false,
        requiresPriorWriter: false,
      };
      const readsBeforeBodyWriter =
        reads(effect.access) &&
        !state.write &&
        effect.requiresPriorWriter !== false;
      state.read ||= reads(effect.access);
      state.write ||= writes(effect.access);
      state.requiresPriorWriter ||= readsBeforeBodyWriter;
      aggregate.set(effect.resourceId, state);
    }
  }
  return Object.freeze([...aggregate.entries()]
    .sort(([left], [right]) => compareCanonicalStrings(left, right))
    .map(([resourceId, state]) => Object.freeze({
      resourceId,
      access: state.read && state.write
        ? "read-write" as const
        : state.write
          ? "write" as const
          : "read" as const,
      requiresPriorWriter: state.requiresPriorWriter,
      guaranteesWrite: state.write,
    })));
}

function runtimeRepeatEffects(
  effects: readonly HostGraphResourceEffect[],
): readonly HostGraphResourceEffect[] {
  return Object.freeze(effects.map((effect) => Object.freeze({
    ...effect,
    // A request-time count of zero executes no body node, so writes performed
    // only by this repeat cannot dominate later reads or satisfy ownership.
    guaranteesWrite: false,
  })));
}

function resourceRepeatEffects(
  node: HostGraphResourceRepeatNode,
  effects: readonly HostGraphResourceEffect[],
  path: string,
): readonly HostGraphResourceEffect[] {
  if (effects.some((effect) =>
    effect.resourceId === node.iterationSource.resourceId)) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      `${path}.body`,
      "resource repeat body cannot access its captured iteration source",
    );
  }
  return Object.freeze([
    Object.freeze({
      resourceId: node.iterationSource.resourceId,
      access: "read" as const,
      requiresPriorWriter: true,
      guaranteesWrite: false,
    }),
    ...runtimeRepeatEffects(effects),
  ].sort((left, right) =>
    compareCanonicalStrings(left.resourceId, right.resourceId)));
}

function resourceDynamicDispatchEffects(
  node: Extract<
    HostGraphDynamicDispatchNode,
    {
      readonly mode:
        | "resource-u32-prefix-elements"
        | "resource-u32-rectangular-prefix";
    }
  >,
  effects: readonly HostGraphResourceEffect[],
  path: string,
): readonly HostGraphResourceEffect[] {
  const launchSources = node.mode === "resource-u32-prefix-elements"
    ? [node.launchSource]
    : node.launchSources;
  const launchResourceIds = new Set(
    launchSources.map((source) => source.resourceId),
  );
  if (effects.some((effect) => launchResourceIds.has(effect.resourceId))) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      `${path}.bindings`,
      "resource dynamic dispatch cannot access its captured launch sources",
    );
  }
  return Object.freeze([
    ...effects,
    ...launchSources.map((source) => Object.freeze({
      resourceId: source.resourceId,
      access: "read" as const,
      requiresPriorWriter: true,
      guaranteesWrite: false,
    })),
  ]);
}

function verifyResourceDynamicDispatchSource(
  source:
    | HostGraphResourceDynamicDispatchSource
    | HostGraphResourceDynamicExtentSource,
  resources: ReadonlyMap<string, HostGraphResource>,
  rankCount: bigint,
  path: string,
): void {
  const resource = resources.get(source.resourceId);
  if (resource === undefined) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.danglingReference,
      `${path}.resourceId`,
      `resource dynamic dispatch source references missing resource ${source.resourceId}`,
    );
  }
  if (
    resource.role !== "temporary" ||
    resource.initialization !== "zero-fill" ||
    resource.dtype !== "u32" ||
    resource.byteLength !== "4" ||
    resource.alignmentBytes < 4
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      `${path}.resourceId`,
      "resource dynamic dispatch source requires one 4-byte-aligned zero-filled temporary u32 resource",
    );
  }
  if (wireIntegerToBigInt(source.rank) >= rankCount) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      `${path}.rank`,
      "resource dynamic dispatch source rank is outside the graph rank count",
    );
  }
}

function verifyResourceRepeatSource(
  source: HostGraphResourceRepeatSource,
  resources: ReadonlyMap<string, HostGraphResource>,
  rankCount: bigint,
  path: string,
): void {
  const resource = resources.get(source.resourceId);
  if (resource === undefined) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.danglingReference,
      `${path}.resourceId`,
      `resource repeat source references missing resource ${source.resourceId}`,
    );
  }
  if (
    resource.role !== "temporary" ||
    resource.initialization !== "zero-fill" ||
    resource.dtype !== "u32" ||
    resource.byteLength !== "4" ||
    resource.alignmentBytes < 4
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      `${path}.resourceId`,
      "resource repeat source requires one 4-byte-aligned zero-filled temporary u32 resource",
    );
  }
  if (wireIntegerToBigInt(source.rank) >= rankCount) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      `${path}.rank`,
      "resource repeat source rank is outside the graph rank count",
    );
  }
}

function verifyResourceFeedbackBound(
  count: number,
  path: string,
  programMinor: HostGraphProgram["version"]["minor"],
): void {
  const maximum = programMinor >= 28
    ? HOST_GRAPH_MAX_RESOURCE_FEEDBACK_NODES
    : programMinor >= 27
      ? HOST_GRAPH_MAX_THREE_STAGE_RESOURCE_FEEDBACK_NODES
      : programMinor >= 24
        ? HOST_GRAPH_MAX_FANOUT_RESOURCE_FEEDBACK_NODES
        : HOST_GRAPH_MAX_LEGACY_RESOURCE_FEEDBACK_NODES;
  if (count > maximum) {
    resource(
      path,
      `resource feedback node count exceeds ${maximum}`,
    );
  }
}

function verifyResourceFeedbackProfile(
  program: HostGraphProgram,
  resourceConditionalCount: number,
  resourceRepeatCount: number,
  resourceDynamicDispatchCount: number,
  dispatchEffects: ReadonlyMap<
    HostGraphDispatchNode | HostGraphDynamicDispatchNode,
    readonly HostGraphResourceEffect[]
  >,
  conditionalEffects: ReadonlyMap<
    HostGraphConditionalNode,
    readonly HostGraphResourceEffect[]
  >,
): void {
  const feedbackCount =
    resourceConditionalCount +
    resourceRepeatCount +
    resourceDynamicDispatchCount;
  if (feedbackCount < 2) return;
  if (
    resourceConditionalCount === 1 &&
    resourceRepeatCount === 1 &&
    resourceDynamicDispatchCount === 0
  ) {
    const conditional = program.nodes.find((
      node,
    ): node is HostGraphResourceConditionalNode =>
      node.kind === "conditional" &&
      node.mode === "resource-u32-branch-sequential"
    );
    const repeat = program.nodes.find((
      node,
    ): node is HostGraphResourceRepeatNode =>
      node.kind === "repeat" &&
      node.mode === "resource-u32-count-sequential"
    );
    if (
      conditional === undefined ||
      repeat === undefined
    ) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.invalidBinding,
        "$.payload.program.nodes",
        "mixed resource feedback lost its conditional or repeat node",
      );
    }
    const sharesSelection =
      conditional.predicate.resourceId ===
        repeat.iterationSource.resourceId &&
      conditional.predicate.rank === repeat.iterationSource.rank;
    if (sharesSelection) {
      if (program.version.minor < 29) {
        invalid(
          GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
          "$.payload.program.nodes",
          "shared conditional/repeat resource feedback requires host graph program version 1.29",
        );
      }
      return;
    }
    if (
      program.version.minor === 29 &&
      conditional.predicate.resourceId ===
        repeat.iterationSource.resourceId
    ) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.invalidBinding,
        "$.payload.program.nodes",
        "version-1.29 mixed resource feedback requires one exact shared conditional/repeat source and rank",
      );
    }
    if (program.version.minor < 30) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        "$.payload.program.nodes",
        "sequential conditional-to-repeat resource feedback requires host graph program version 1.30",
      );
    }
    if (
      conditional.predicate.resourceId ===
        repeat.iterationSource.resourceId
    ) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.invalidBinding,
        "$.payload.program.nodes",
        "version-1.30 sequential conditional-to-repeat feedback requires distinct predicate and repeat-count resources",
      );
    }
    const repeatSourceEffect = (conditionalEffects.get(conditional) ?? [])
      .find((effect) =>
        effect.resourceId === repeat.iterationSource.resourceId
      );
    if (
      repeatSourceEffect === undefined ||
      !writes(repeatSourceEffect.access) ||
      repeatSourceEffect.guaranteesWrite !== true
    ) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.invalidBinding,
        "$.payload.program.nodes",
        "version-1.30 sequential conditional-to-repeat feedback requires both conditional branches to guarantee-write the distinct repeat-count source",
      );
    }
    return;
  }
  const dispatches = program.nodes.filter((
    node,
  ): node is
    | HostGraphResourceDynamicDispatchNode
    | HostGraphResourceRectangularDynamicDispatchNode =>
    node.kind === "dynamic-dispatch" &&
    (
      node.mode === "resource-u32-prefix-elements" ||
      node.mode === "resource-u32-rectangular-prefix"
    )
  );
  if (
    resourceConditionalCount !== 0 ||
    resourceRepeatCount !== 0 ||
    (
      resourceDynamicDispatchCount !== 2 &&
      resourceDynamicDispatchCount !== 3 &&
      resourceDynamicDispatchCount !== 4
    ) ||
    dispatches.length !== resourceDynamicDispatchCount
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      "$.payload.program.nodes",
      "resource feedback profile requires exactly two, three, or four top-level resource-controlled dynamic dispatches",
    );
  }
  if (resourceDynamicDispatchCount >= 3) {
    const requiredMinor = resourceDynamicDispatchCount === 4 ? 28 : 27;
    const versionLabel = `version-1.${requiredMinor}`;
    if (program.version.minor < requiredMinor) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        "$.payload.program.nodes",
        `${resourceDynamicDispatchCount}-stage sequential resource feedback requires host graph program version 1.${requiredMinor}`,
      );
    }
    if (dispatches.some((dispatch) =>
      dispatch.mode !== "resource-u32-prefix-elements"
    )) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        "$.payload.program.nodes",
        `${versionLabel} sequential resource feedback requires exactly ${resourceDynamicDispatchCount} linear resource dispatches`,
      );
    }
    const linearDispatches =
      dispatches as readonly HostGraphResourceDynamicDispatchNode[];
    const sourceIds = linearDispatches.map((dispatch) =>
      dispatch.launchSource.resourceId);
    if (new Set(sourceIds).size !== linearDispatches.length) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.invalidBinding,
        "$.payload.program.nodes",
        `${versionLabel} sequential resource feedback requires ${resourceDynamicDispatchCount} distinct launch-source resources`,
      );
    }
    const producerEdges = linearDispatches.flatMap((producer) =>
      linearDispatches.flatMap((consumer) =>
        producer === consumer ||
          !(dispatchEffects.get(producer) ?? []).some((effect) =>
            effect.resourceId === consumer.launchSource.resourceId &&
            writes(effect.access))
          ? []
          : [{ producer, consumer }])
    );
    if (!isExactLinearProducerChain(linearDispatches, producerEdges)) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.invalidBinding,
        "$.payload.program.nodes",
        `${versionLabel} sequential resource feedback requires one exact ${resourceDynamicDispatchCount}-dispatch producer chain`,
      );
    }
    return;
  }
  if (program.version.minor < 24) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      "$.payload.program.nodes",
      "resource feedback fanout requires host graph program version 1.24",
    );
  }
  const [first, second] = dispatches;
  if (
    first === undefined ||
    second === undefined ||
    first.mode !== second.mode
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      "$.payload.program.nodes",
      "resource feedback fanout dispatches must use the same source mode",
    );
  }
  if (
    first.mode === "resource-u32-prefix-elements" &&
    second.mode === "resource-u32-prefix-elements"
  ) {
    const selectionDiffers =
      first.launchSource.resourceId !== second.launchSource.resourceId ||
      first.launchSource.rank !== second.launchSource.rank ||
      first.launchSource.mode !== second.launchSource.mode ||
      first.maxElementCount !== second.maxElementCount;
    if (!selectionDiffers) return;
    if (program.version.minor < 26) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.invalidBinding,
        "$.payload.program.nodes",
        "version-1.24 resource feedback fanout dispatches must share one exact launch source and artifact maximum",
      );
    }
    const firstEffects = dispatchEffects.get(first) ?? [];
    const secondEffects = dispatchEffects.get(second) ?? [];
    const firstProducesSecond = firstEffects.some((effect) =>
      effect.resourceId === second.launchSource.resourceId &&
      writes(effect.access));
    const secondProducesFirst = secondEffects.some((effect) =>
      effect.resourceId === first.launchSource.resourceId &&
      writes(effect.access));
    if (firstProducesSecond === secondProducesFirst) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.invalidBinding,
        "$.payload.program.nodes",
        "version-1.26 sequential resource feedback requires exactly one dynamic dispatch to produce the other's distinct launch source",
      );
    }
    return;
  }
  if (
    program.version.minor < 25 ||
    first.mode !== "resource-u32-rectangular-prefix" ||
    second.mode !== "resource-u32-rectangular-prefix"
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      "$.payload.program.nodes",
      "rectangular resource feedback fanout requires host graph program version 1.25",
    );
  }
  if (
    first.maxExtents.length !== second.maxExtents.length ||
    first.maxExtents.some((extent, index) =>
      extent !== second.maxExtents[index]) ||
    first.launchSources.length !== second.launchSources.length ||
    first.launchSources.some((source, index) => {
      const peer = second.launchSources[index];
      return peer === undefined ||
        source.axis !== peer.axis ||
        source.resourceId !== peer.resourceId ||
        source.rank !== peer.rank ||
        source.mode !== peer.mode;
    })
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      "$.payload.program.nodes",
      "version-1.25 rectangular resource feedback fanout dispatches must share exact ordered sources and artifact maxima",
    );
  }
}

function isExactLinearProducerChain(
  dispatches: readonly HostGraphResourceDynamicDispatchNode[],
  edges: readonly {
    readonly producer: HostGraphResourceDynamicDispatchNode;
    readonly consumer: HostGraphResourceDynamicDispatchNode;
  }[],
): boolean {
  if (edges.length !== dispatches.length - 1) return false;
  const incoming = new Map(
    dispatches.map((dispatch) => [
      dispatch,
      [] as HostGraphResourceDynamicDispatchNode[],
    ]),
  );
  const outgoing = new Map(
    dispatches.map((dispatch) => [
      dispatch,
      [] as HostGraphResourceDynamicDispatchNode[],
    ]),
  );
  for (const { producer, consumer } of edges) {
    outgoing.get(producer)?.push(consumer);
    incoming.get(consumer)?.push(producer);
  }
  const roots = dispatches.filter((dispatch) =>
    incoming.get(dispatch)?.length === 0);
  const leaves = dispatches.filter((dispatch) =>
    outgoing.get(dispatch)?.length === 0);
  if (roots.length !== 1 || leaves.length !== 1) return false;
  const root = roots[0]!;
  const leaf = leaves[0]!;
  if (
    dispatches.some((dispatch) => {
      const incomingCount = incoming.get(dispatch)?.length ?? 0;
      const outgoingCount = outgoing.get(dispatch)?.length ?? 0;
      if (dispatch === root) {
        return incomingCount !== 0 || outgoingCount !== 1;
      }
      if (dispatch === leaf) {
        return incomingCount !== 1 || outgoingCount !== 0;
      }
      return incomingCount !== 1 || outgoingCount !== 1;
    })
  ) {
    return false;
  }
  const visited = new Set<HostGraphResourceDynamicDispatchNode>();
  let current: HostGraphResourceDynamicDispatchNode | undefined = root;
  while (current !== undefined) {
    if (visited.has(current)) return false;
    visited.add(current);
    current = outgoing.get(current)?.[0];
  }
  return visited.size === dispatches.length && visited.has(leaf);
}

function verifyConditionalPredicate(
  node: HostGraphConditionalNode,
  resources: ReadonlyMap<string, HostGraphResource>,
  rankCount: bigint,
  path: string,
): void {
  if (node.mode === "runtime-u32-branch-sequential") return;
  const predicate = resources.get(node.predicate.resourceId);
  if (predicate === undefined) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.danglingReference,
      `${path}.predicate.resourceId`,
      `conditional predicate references missing resource ${node.predicate.resourceId}`,
    );
  }
  const expectedRole = node.mode === "resource-u32-branch-sequential"
    ? "temporary"
    : "input";
  if (
    predicate.role !== expectedRole ||
    predicate.dtype !== "u32" ||
    predicate.byteLength !== "4" ||
    predicate.alignmentBytes < 4
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      `${path}.predicate.resourceId`,
      node.mode === "resource-u32-branch-sequential"
        ? "resource conditional predicate requires one 4-byte-aligned zero-filled temporary u32 resource"
        : "input conditional predicate requires one 4-byte-aligned external-input u32 resource",
    );
  }
  if (wireIntegerToBigInt(node.predicate.rank) >= rankCount) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      `${path}.predicate.rank`,
      "conditional predicate rank is outside the graph rank count",
    );
  }
}

function mergeConditionalEffects(
  node: HostGraphConditionalNode,
  thenEffects: readonly HostGraphResourceEffect[],
  elseEffects: readonly HostGraphResourceEffect[],
): readonly HostGraphResourceEffect[] {
  const thenByResource = new Map(thenEffects.map((effect) => [
    effect.resourceId,
    effect,
  ]));
  const elseByResource = new Map(elseEffects.map((effect) => [
    effect.resourceId,
    effect,
  ]));
  const predicateResourceId = node.mode !==
      "runtime-u32-branch-sequential"
    ? node.predicate.resourceId
    : undefined;
  const resourceIds = new Set([
    ...(predicateResourceId === undefined ? [] : [predicateResourceId]),
    ...thenByResource.keys(),
    ...elseByResource.keys(),
  ]);
  return Object.freeze([...resourceIds]
    .sort(compareCanonicalStrings)
    .map((resourceId) => {
      const thenEffect = thenByResource.get(resourceId);
      const elseEffect = elseByResource.get(resourceId);
      const predicateRead = resourceId === predicateResourceId;
      const read = predicateRead ||
        (thenEffect !== undefined && reads(thenEffect.access)) ||
        (elseEffect !== undefined && reads(elseEffect.access));
      const write =
        (thenEffect !== undefined && writes(thenEffect.access)) ||
        (elseEffect !== undefined && writes(elseEffect.access));
      const requiresPriorWriter =
        (predicateRead &&
          node.mode === "resource-u32-branch-sequential") ||
        (thenEffect?.requiresPriorWriter ?? false) ||
        (elseEffect?.requiresPriorWriter ?? false);
      const guaranteesWrite =
        thenEffect !== undefined &&
        writes(thenEffect.access) &&
        (thenEffect.guaranteesWrite ?? true) &&
        elseEffect !== undefined &&
        writes(elseEffect.access) &&
        (elseEffect.guaranteesWrite ?? true);
      return Object.freeze({
        resourceId,
        access: read && write
          ? "read-write" as const
          : write
            ? "write" as const
            : "read" as const,
        requiresPriorWriter,
        guaranteesWrite,
      });
    }));
}

function verifyMaterializeBinding(
  node: HostGraphMaterializeNode,
  resources: ReadonlyMap<string, HostGraphResource>,
  path: string,
): void {
  const resource = resources.get(node.resourceId);
  if (resource === undefined) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.danglingReference,
      `${path}.resourceId`,
      `materialize references missing resource ${node.resourceId}`,
    );
  }
  if (resource.role !== "output") {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidAccess,
      `${path}.resourceId`,
      "materialize may expose only a declared output resource",
    );
  }
}

function verifyMaterializationContract(
  program: HostGraphProgram,
): void {
  if (program.version.minor < 2) return;
  const materializations = program.nodes.filter(
    (node): node is HostGraphMaterializeNode =>
      node.kind === "materialize",
  );
  const byResource = new Map<string, HostGraphMaterializeNode>();
  for (const node of materializations) {
    if (byResource.has(node.resourceId)) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.invalidAccess,
        "$.payload.program.nodes",
        `output resource ${node.resourceId} is materialized more than once`,
      );
    }
    byResource.set(node.resourceId, node);
  }
  for (const resource of program.resources) {
    if (resource.role === "output" && !byResource.has(resource.resourceId)) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.invalidAccess,
        "$.payload.program.nodes",
        `output resource ${resource.resourceId} requires exactly one materialize node in host graph version 1.2 or newer`,
      );
    }
  }
  const materializationIds = new Set(
    materializations.map((node) => node.nodeId),
  );
  for (const [index, node] of program.nodes.entries()) {
    const dependency = node.dependsOn.find((nodeId) =>
      materializationIds.has(nodeId));
    if (dependency !== undefined) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.invalidAccess,
        `$.payload.program.nodes[${index}].dependsOn`,
        `materialize node ${dependency} is terminal and cannot have dependents`,
      );
    }
  }
}

function verifyCopyBinding(
  node: HostGraphCopyNode,
  resources: ReadonlyMap<string, HostGraphResource>,
  path: string,
): void {
  if (node.sourceResourceId === node.destinationResourceId) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidAccess,
      `${path}.destinationResourceId`,
      "copy source and destination resources must be distinct",
    );
  }
  const source = resources.get(node.sourceResourceId);
  const destination = resources.get(node.destinationResourceId);
  if (source === undefined || destination === undefined) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.danglingReference,
      source === undefined
        ? `${path}.sourceResourceId`
        : `${path}.destinationResourceId`,
      "copy references a missing graph resource",
    );
  }
  if (
    source.dtype !== destination.dtype ||
    source.byteLength !== destination.byteLength
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      path,
      "copy requires identical source and destination dtype and allocation byte length",
    );
  }
}

function verifyCollectiveBinding(
  node: HostGraphAllReduceNode,
  resources: ReadonlyMap<string, HostGraphResource>,
  rankCount: bigint,
  path: string,
): void {
  const resource = resources.get(node.resourceId);
  if (resource === undefined) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.danglingReference,
      `${path}.resourceId`,
      `all-reduce references missing resource ${node.resourceId}`,
    );
  }
  if (resource.dtype !== node.dtype) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      `${path}.dtype`,
      `all-reduce dtype ${node.dtype} does not match resource dtype ${resource.dtype}`,
    );
  }
  if (wireIntegerToBigInt(resource.byteLength) % 4n !== 0n) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidCollective,
      `${path}.resourceId`,
      "initial all-reduce resources require a whole number of 32-bit elements",
    );
  }
  for (const [participantIndex, participant] of
    node.participants.entries()) {
    if (wireIntegerToBigInt(participant) >= rankCount) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.invalidCollective,
        `${path}.participants[${participantIndex}]`,
        `participant ${participant} is outside rank count ${rankCount}`,
      );
    }
  }
}

function resolveDispatchGeometry(
  operation: DispatchOperationBinding,
  node: HostGraphDispatchNode | HostGraphDynamicDispatchNode,
  limits: DecodeLimits,
  path: string,
): ResolvedDispatchGeometry {
  try {
    const source = prepareViewAccessor(operation.layoutArtifact, {
      viewId: operation.sourceSemanticResourceId,
      bindings: node.dimensionBindings,
      limits,
    });
    const destination = prepareViewAccessor(operation.layoutArtifact, {
      viewId: operation.destinationSemanticResourceId,
      bindings: node.dimensionBindings,
      limits,
    });
    if (source.dtype !== operation.dtype ||
        destination.dtype !== operation.dtype) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.semanticArtifactMismatch,
        path,
        "verified kernel and layout dtype meaning diverged",
      );
    }
    if (source.memorySpace.kind !== "global" ||
        destination.memorySpace.kind !== "global") {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
        path,
        "initial host graph dispatch profile requires global-memory views",
      );
    }
    if (source.allocationByteLength >
          HOST_GRAPH_MAX_TOTAL_RESOURCE_BYTES ||
        destination.allocationByteLength >
          HOST_GRAPH_MAX_TOTAL_RESOURCE_BYTES) {
      resource(
        path,
        `resolved dispatch allocation bytes exceed ${HOST_GRAPH_MAX_TOTAL_RESOURCE_BYTES}`,
      );
    }
    return Object.freeze({
      logicalShape: Object.freeze([...destination.logicalShape]),
      elementCount: destination.logicalShape.reduce(
        (total, extent) => total * extent,
        1n,
      ),
      sourceByteLength: encodeWireU64(source.allocationByteLength),
      destinationByteLength: encodeWireU64(
        destination.allocationByteLength,
      ),
      sourceAlignmentBytes: Math.max(
        source.allocationAlignmentBytes,
        source.requiredAlignmentBytes,
      ),
      destinationAlignmentBytes: Math.max(
        destination.allocationAlignmentBytes,
        destination.requiredAlignmentBytes,
      ),
    });
  } catch (error) {
    if (error instanceof SemanticSchemaError &&
        error.diagnostic.code.startsWith("BG-GRAPH-")) {
      throw error;
    }
    if (error instanceof SemanticSchemaError &&
        error.diagnostic.code.endsWith("-RESOURCE-LIMIT")) {
      resource(
        `${path}.dimensionBindings`,
        `dispatch dimension binding exceeded a semantic resource limit: ${
          error.message
        }`,
      );
    }
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      `${path}.dimensionBindings`,
      `dispatch dimension binding failed: ${
        error instanceof Error ? error.message : "unknown semantic error"
      }`,
    );
  }
}

function verifyRectangularDynamicGeometry(
  node: Extract<
    HostGraphDynamicDispatchNode,
    {
      readonly mode:
        | "runtime-u32-rectangular-prefix"
        | "resource-u32-rectangular-prefix";
    }
  >,
  geometry: ResolvedDispatchGeometry,
  path: string,
): void {
  if (
    node.maxExtents.length !== geometry.logicalShape.length ||
    (
      node.mode === "runtime-u32-rectangular-prefix"
        ? node.launchControls.length
        : node.launchSources.length
    ) !== geometry.logicalShape.length
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidBinding,
      `${path}.maxExtents`,
      `rectangular dynamic rank ${node.maxExtents.length} does not match semantic rank ${geometry.logicalShape.length}`,
    );
  }
  for (const [axis, extent] of node.maxExtents.entries()) {
    const maximum = wireIntegerToBigInt(extent);
    const semanticExtent = geometry.logicalShape[axis];
    if (semanticExtent === undefined || maximum > semanticExtent) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.invalidBinding,
        `${path}.maxExtents[${axis}]`,
        `rectangular dynamic extent ${maximum} exceeds semantic extent ${semanticExtent ?? "missing"}`,
      );
    }
  }
}

function topologicalOrder(
  graphNodes: readonly HostGraphNode[],
  nodes: ReadonlyMap<string, HostGraphNode>,
): string[] {
  const indegree = new Map<string, number>();
  const dependents = new Map<string, string[]>();
  for (const node of graphNodes) {
    indegree.set(node.nodeId, node.dependsOn.length);
    for (const dependency of node.dependsOn) {
      const list = dependents.get(dependency) ?? [];
      list.push(node.nodeId);
      dependents.set(dependency, list);
    }
  }
  const ready = graphNodes
    .filter((node) => node.dependsOn.length === 0)
    .map((node) => node.nodeId)
    .sort(compareCanonicalStrings);
  const ordered: string[] = [];
  while (ready.length > 0) {
    const nodeId = ready.shift();
    if (nodeId === undefined) break;
    ordered.push(nodeId);
    const next = (dependents.get(nodeId) ?? []).sort(compareCanonicalStrings);
    for (const dependent of next) {
      const remaining = (indegree.get(dependent) ?? 0) - 1;
      indegree.set(dependent, remaining);
      if (remaining === 0) {
        ready.push(dependent);
        ready.sort(compareCanonicalStrings);
      }
    }
  }
  if (ordered.length !== nodes.size) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.cycle,
      "$.payload.program.nodes",
      "host graph dependency relation contains a cycle",
    );
  }
  return ordered;
}

function dependencyAncestors(
  topologicalNodeIds: readonly string[],
  nodes: ReadonlyMap<string, HostGraphNode>,
): ReadonlyMap<string, ReadonlySet<string>> {
  const ancestors = new Map<string, ReadonlySet<string>>();
  for (const nodeId of topologicalNodeIds) {
    const node = nodes.get(nodeId);
    if (node === undefined) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.danglingReference,
        "$.payload.program.nodes",
        `topological node ${nodeId} disappeared`,
      );
    }
    const values = new Set<string>();
    for (const dependency of node.dependsOn) {
      values.add(dependency);
      for (const ancestor of ancestors.get(dependency) ?? []) {
        values.add(ancestor);
      }
    }
    ancestors.set(nodeId, values);
  }
  return ancestors;
}

function verifyEffects(
  graphNodes: readonly HostGraphNode[],
  topologicalNodeIds: readonly string[],
  nodes: ReadonlyMap<string, HostGraphNode>,
  resources: ReadonlyMap<string, HostGraphResource>,
  ancestors: ReadonlyMap<string, ReadonlySet<string>>,
  dispatchEffects: ReadonlyMap<
    HostGraphDispatchNode | HostGraphDynamicDispatchNode,
    readonly HostGraphResourceEffect[]
  >,
  repeatEffects: ReadonlyMap<
    HostGraphRepeatNode,
    readonly HostGraphResourceEffect[]
  >,
  conditionalEffects: ReadonlyMap<
    HostGraphConditionalNode,
    readonly HostGraphResourceEffect[]
  >,
): void {
  const uses = new Map<string, ResourceUse[]>();
  const nodeIndexes = new Map(
    graphNodes.map((node, index) => [node.nodeId, index]),
  );
  for (const nodeId of topologicalNodeIds) {
    const node = nodes.get(nodeId);
    const nodeIndex = nodeIndexes.get(nodeId);
    if (node === undefined || nodeIndex === undefined) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.danglingReference,
        "$.payload.program.nodes",
        `effect analysis node ${nodeId} disappeared`,
      );
    }
    const path = `$.payload.program.nodes[${nodeIndex}]`;
    for (const [effectIndex, effect] of
      nodeEffects(
        node,
        dispatchEffects,
        repeatEffects,
        conditionalEffects,
      ).entries()) {
      const effectPath =
        node.kind === "dynamic-dispatch" &&
            node.mode === "resource-u32-prefix-elements" &&
            effect.resourceId === node.launchSource.resourceId
          ? `${path}.launchSource`
          : node.kind === "dynamic-dispatch" &&
              node.mode === "resource-u32-rectangular-prefix" &&
              node.launchSources.some((source) =>
                source.resourceId === effect.resourceId)
            ? `${path}.launchSources`
          : node.kind === "dispatch" || node.kind === "dynamic-dispatch"
            ? `${path}.bindings[${effectIndex}]`
        : node.kind === "all-reduce"
          ? `${path}.resourceId`
          : node.kind === "copy"
            ? effectIndex === 0
              ? `${path}.sourceResourceId`
              : `${path}.destinationResourceId`
            : node.kind === "materialize"
              ? `${path}.resourceId`
              : node.kind === "repeat"
                ? `${path}.body`
                : `${path}.predicate`;
      const resource = resources.get(effect.resourceId);
      if (resource === undefined) {
        invalid(
          GRAPH_DIAGNOSTIC_CODES.danglingReference,
          effectPath,
          `node ${node.nodeId} references missing resource ${effect.resourceId}`,
        );
      }
      if (resource.role === "input" && writes(effect.access)) {
        invalid(
          GRAPH_DIAGNOSTIC_CODES.invalidAccess,
          effectPath,
          `input resource ${resource.resourceId} is read-only`,
        );
      }
      const prior = uses.get(effect.resourceId) ?? [];
      if (resource.role !== "input" &&
          (effect.requiresPriorWriter ?? reads(effect.access)) &&
          !prior.some((use) =>
            use.guaranteesWrite &&
            ancestors.get(node.nodeId)?.has(use.nodeId) === true)) {
        invalid(
          GRAPH_DIAGNOSTIC_CODES.readBeforeWrite,
          effectPath,
          `resource ${resource.resourceId} is read before an ordered writer`,
        );
      }
      for (const use of prior) {
        if (!conflicts(use.access, effect.access)) continue;
        const ordered =
          ancestors.get(node.nodeId)?.has(use.nodeId) === true ||
          ancestors.get(use.nodeId)?.has(node.nodeId) === true;
        if (!ordered) {
          invalid(
            GRAPH_DIAGNOSTIC_CODES.effectConflict,
            effectPath,
            `resource ${resource.resourceId} has unordered conflicting effects in ${use.nodeId} and ${node.nodeId}`,
          );
        }
      }
      prior.push({
        nodeId: node.nodeId,
        access: effect.access,
        guaranteesWrite:
          effect.guaranteesWrite ?? writes(effect.access),
        path: effectPath,
      });
      uses.set(effect.resourceId, prior);
    }
  }
  for (const resource of resources.values()) {
    if (resource.role === "input") continue;
    const resourceUses = uses.get(resource.resourceId) ?? [];
    if (!resourceUses.some((use) => use.guaranteesWrite)) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.readBeforeWrite,
        "$.payload.program.resources",
        `${resource.role} resource ${resource.resourceId} has no writer`,
      );
    }
  }
}

function nodeEffects(
  node: HostGraphNode,
  dispatchEffects: ReadonlyMap<
    HostGraphDispatchNode | HostGraphDynamicDispatchNode,
    readonly HostGraphResourceEffect[]
  >,
  repeatEffects: ReadonlyMap<
    HostGraphRepeatNode,
    readonly HostGraphResourceEffect[]
  >,
  conditionalEffects: ReadonlyMap<
    HostGraphConditionalNode,
    readonly HostGraphResourceEffect[]
  >,
): readonly HostGraphResourceEffect[] {
  if (node.kind === "dispatch" || node.kind === "dynamic-dispatch") {
    const effects = dispatchEffects.get(node);
    if (effects === undefined) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.semanticArtifactMismatch,
        "$.payload.program.nodes",
        `dispatch effects for ${node.nodeId} were not derived`,
      );
    }
    return effects;
  }
  if (node.kind === "all-reduce") {
    return [{ resourceId: node.resourceId, access: "read-write" }];
  }
  if (node.kind === "copy") {
    return [
      { resourceId: node.sourceResourceId, access: "read" },
      { resourceId: node.destinationResourceId, access: "write" },
    ];
  }
  if (node.kind === "materialize") {
    return [{ resourceId: node.resourceId, access: "read" }];
  }
  if (node.kind === "repeat") {
    const effects = repeatEffects.get(node);
    if (effects === undefined) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.semanticArtifactMismatch,
        "$.payload.program.nodes",
        `repeat effects for ${node.nodeId} were not derived`,
      );
    }
    return effects;
  }
  if (node.kind === "conditional") {
    const effects = conditionalEffects.get(node);
    if (effects === undefined) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.semanticArtifactMismatch,
        "$.payload.program.nodes",
        `conditional effects for ${node.nodeId} were not derived`,
      );
    }
    return effects;
  }
  return [];
}

function reads(access: HostGraphResourceAccess): boolean {
  return access === "read" || access === "read-write";
}

function writes(access: HostGraphResourceAccess): boolean {
  return access === "write" || access === "read-write";
}

function conflicts(
  left: HostGraphResourceAccess,
  right: HostGraphResourceAccess,
): boolean {
  return writes(left) || writes(right);
}

function collectiveNumericalPolicy(
  dtype: HostGraphCollectiveDType,
  reduction: HostGraphCollectiveReduction,
): HostGraphCollectiveNumericalPolicy {
  if (dtype === "f32") return "rank-order-f32";
  if (reduction === "sum") return "rank-order-wrapping-32";
  return "exact-32-bit";
}

async function semanticArtifactCatalog(
  kernelArtifacts: readonly VerifiedKernelArtifact[],
  layoutArtifacts: readonly VerifiedLayoutArtifact[],
  limits: DecodeLimits,
): Promise<SemanticArtifactCatalog> {
  const layoutEntries = await Promise.all(
    layoutArtifacts.map(async (artifact) => Object.freeze({
      hash: await hashSemanticArtifact(artifact, { limits }),
      artifact,
    })),
  );
  unique(
    layoutEntries.map((entry) => entry.hash),
    "$options.layoutArtifacts",
    "layout artifact hash",
  );
  const layouts = new Map(
    layoutEntries.map((entry) => [entry.hash, entry.artifact]),
  );
  const entries = await Promise.all(
    kernelArtifacts.map(async (artifact) => {
      const hash = await hashSemanticArtifact(artifact, { limits });
      const payload = kernelArtifactPayload(artifact);
      const layoutArtifact = layouts.get(payload.layoutSemanticHash);
      if (layoutArtifact === undefined) {
        invalid(
          GRAPH_DIAGNOSTIC_CODES.semanticArtifactMismatch,
          "$options.layoutArtifacts",
          `kernel ${hash} references no supplied opaque verified layout artifact`,
        );
      }
      const operations = new Map<string, DispatchOperationBinding>();
      for (const operation of payload.operations) {
        operations.set(operation.operationId, Object.freeze({
          layoutArtifact,
          sourceSemanticResourceId: operation.source.viewId,
          destinationSemanticResourceId: operation.destination.viewId,
          dtype: operation.dtype,
        }));
      }
      return Object.freeze({
        hash,
        operations: operations as ReadonlyMap<
          string,
          DispatchOperationBinding
        >,
      });
    }),
  );
  unique(
    entries.map((entry) => entry.hash),
    "$options.kernelArtifacts",
    "semantic artifact hash",
  );
  return new Map(entries.map((entry) => [entry.hash, entry.operations]));
}

function snapshotKernelArtifacts(
  value: readonly VerifiedKernelArtifact[],
): readonly VerifiedKernelArtifact[] {
  return snapshotArtifacts(
    value,
    "$options.kernelArtifacts",
  );
}

function snapshotLayoutArtifacts(
  value: readonly VerifiedLayoutArtifact[],
): readonly VerifiedLayoutArtifact[] {
  return snapshotArtifacts(
    value,
    "$options.layoutArtifacts",
  );
}

function snapshotArtifacts<Artifact>(
  value: readonly Artifact[],
  path: string,
): readonly Artifact[] {
  if (!Array.isArray(value) ||
      Object.getPrototypeOf(value) !== Array.prototype ||
      value.length > HOST_GRAPH_MAX_SEMANTIC_ARTIFACTS) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.resourceLimit,
      path,
      `artifacts must be a plain array with at most ${HOST_GRAPH_MAX_SEMANTIC_ARTIFACTS} entries`,
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
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidArtifact,
      path,
      "artifacts must be dense and contain no named or symbolic properties",
    );
  }
  return Object.freeze(Array.from(
    { length: value.length },
    (_, index) => {
      const descriptor = descriptors[String(index)];
      if (descriptor === undefined || !("value" in descriptor) ||
          descriptor.enumerable !== true) {
        invalid(
          GRAPH_DIAGNOSTIC_CODES.invalidArtifact,
          `${path}[${index}]`,
          "semantic artifact entries must be enumerable data properties",
        );
      }
      return descriptor.value as Artifact;
    },
  ));
}

function closedObject(
  value: JsonValue,
  allowed: readonly string[],
  path: string,
): JsonObject {
  const object = objectValue(value, path);
  const known = new Set(allowed);
  const unknown = Object.keys(object).filter((key) => !known.has(key));
  if (unknown.length > 0) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unknownField,
      path,
      `unknown closed-record fields: ${unknown.sort(compareCanonicalStrings).join(", ")}`,
    );
  }
  for (const fieldName of allowed) {
    if (object[fieldName] === undefined) {
      invalid(
        GRAPH_DIAGNOSTIC_CODES.invalidArtifact,
        `${path}.${fieldName}`,
        "required field is missing",
      );
    }
  }
  return object;
}

function objectValue(value: JsonValue, path: string): JsonObject {
  if (!isJsonObject(value)) {
    invalid(GRAPH_DIAGNOSTIC_CODES.invalidArtifact, path, "expected object");
  }
  return value;
}

function arrayValue(value: JsonValue, path: string): JsonArray {
  if (!Array.isArray(value)) {
    invalid(GRAPH_DIAGNOSTIC_CODES.invalidArtifact, path, "expected array");
  }
  return value;
}

function field(object: JsonObject, name: string, path: string): JsonValue {
  const value = object[name];
  if (value === undefined) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidArtifact,
      `${path}.${name}`,
      "required field is missing",
    );
  }
  return value;
}

function stringValue(value: JsonValue, path: string): string {
  if (typeof value !== "string") {
    invalid(GRAPH_DIAGNOSTIC_CODES.invalidArtifact, path, "expected string");
  }
  return value;
}

function identifier(value: JsonValue, path: string): string {
  const result = stringValue(value, path);
  if (!ID.test(result)) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidArtifact,
      path,
      "identifier must use the closed 1-128 character ASCII grammar",
    );
  }
  return result;
}

function hashValue(value: JsonValue, path: string): string {
  const result = stringValue(value, path);
  if (!HASH.test(result)) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidArtifact,
      path,
      "semantic artifact hash must be 64 lowercase hexadecimal digits",
    );
  }
  return result;
}

function alignment(value: JsonValue, path: string): number {
  if (typeof value !== "number" ||
      !Number.isSafeInteger(value) ||
      value < 1 ||
      value > HOST_GRAPH_MAX_ALIGNMENT_BYTES ||
      (value & (value - 1)) !== 0) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidArtifact,
      path,
      `alignment must be a power of two between 1 and ${HOST_GRAPH_MAX_ALIGNMENT_BYTES}`,
    );
  }
  return value;
}

function positiveWire(value: JsonValue, path: string): WireU64 {
  const result = parseWireU64(value, path);
  if (wireIntegerToBigInt(result) === 0n) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.invalidArtifact,
      path,
      "value must be positive",
    );
  }
  return result;
}

function positiveOrZeroWire(value: JsonValue, path: string): WireU64 {
  return encodeWireU64(wireIntegerToBigInt(parseWireU64(value, path)));
}

function unique(
  values: readonly string[],
  path: string,
  label: string,
): void {
  if (new Set(values).size !== values.length) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.duplicateId,
      path,
      `${label} values must be unique`,
    );
  }
}

function invalid(
  code: `BG-GRAPH-${string}`,
  path: string,
  message: string,
): never {
  throw new SemanticSchemaError({
    code,
    stage: "verification",
    severity: "error",
    path,
    message,
  });
}

function resource(path: string, message: string): never {
  invalid(GRAPH_DIAGNOSTIC_CODES.resourceLimit, path, message);
}
