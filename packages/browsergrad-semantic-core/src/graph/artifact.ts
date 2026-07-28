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
  type HostGraphResourcePredicate,
  type HostGraphResourceRole,
  type HostGraphRuntimeControlPredicate,
} from "./model.js";

export const HOST_GRAPH_ARTIFACT_SCHEMA = "browsergrad.host-graph";
export const HOST_GRAPH_ARTIFACT_MAJOR = 1;
export const HOST_GRAPH_ARTIFACT_MINOR = 7;
export const HOST_GRAPH_MAX_RESOURCES = 256;
export const HOST_GRAPH_MAX_NODES = 256;
export const HOST_GRAPH_MAX_EDGES = 4_096;
export const HOST_GRAPH_MAX_REPEAT_BODY_NODES = 64;
export const HOST_GRAPH_MAX_CONDITIONAL_BODY_NODES = 64;
export const HOST_GRAPH_MAX_REPEAT_ITERATIONS = 1_024;
export const HOST_GRAPH_MAX_RUNTIME_CONTROLS = 64;
export const HOST_GRAPH_MAX_RESOURCE_CONDITIONALS = 1;
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
  readonly collectiveCount: number;
  readonly copyCount: number;
  readonly materializationCount: number;
  readonly eventCount: number;
  readonly eventIds: readonly string[];
  readonly repeatCount: number;
  readonly repeatIterationCount: number;
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
  readonly collectiveCount: number;
  readonly copyCount: number;
  readonly materializationCount: number;
  readonly eventCount: number;
  readonly eventIds: readonly string[];
  readonly repeatCount: number;
  readonly repeatIterationCount: number;
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
  readonly sourceByteLength: WireU64;
  readonly destinationByteLength: WireU64;
  readonly sourceAlignmentBytes: number;
  readonly destinationAlignmentBytes: number;
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
    collectiveCount: analysis.collectiveCount,
    copyCount: analysis.copyCount,
    materializationCount: analysis.materializationCount,
    eventCount: analysis.eventCount,
    eventIds: Object.freeze([...analysis.eventIds]),
    repeatCount: analysis.repeatCount,
    repeatIterationCount: analysis.repeatIterationCount,
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
    (version.minor !== 0 &&
      version.minor !== 1 &&
      version.minor !== 2 &&
      version.minor !== 3 &&
      version.minor !== 4 &&
      version.minor !== 5 &&
      version.minor !== 6 &&
      version.minor !== 7)
  ) {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      "$.payload.program.version",
      "host graph program reader supports versions 1.0 through 1.7 only",
    );
  }
  if (version.minor !== envelopeMinor) {
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
    version.minor,
  );
  return {
    kind: "host-graph",
    version: { major: 1, minor: version.minor },
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
    return parseRepeatNode(object, path);
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
    "host graph profile supports dispatch, all-reduce, version-1.1 copy, version-1.2 materialize, version-1.3 event, version-1.4 repeat, and version-1.5 through 1.7 conditional nodes",
  );
}

function parseRepeatNode(
  value: JsonObject,
  path: string,
): HostGraphRepeatNode {
  const object = closedObject(
    value,
    ["nodeId", "kind", "dependsOn", "iterationCount", "body", "mode"],
    path,
  );
  if (object.mode !== "fixed-count-sequential") {
    invalid(
      GRAPH_DIAGNOSTIC_CODES.unsupportedProfile,
      `${path}.mode`,
      "repeat mode must be fixed-count-sequential",
    );
  }
  const iterationCount = positiveWire(
    field(object, "iterationCount", path),
    `${path}.iterationCount`,
  );
  if (
    wireIntegerToBigInt(iterationCount) >
    BigInt(HOST_GRAPH_MAX_REPEAT_ITERATIONS)
  ) {
    resource(
      `${path}.iterationCount`,
      `repeat iteration count exceeds ${HOST_GRAPH_MAX_REPEAT_ITERATIONS}`,
    );
  }
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
    kind: "dispatch",
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
  let collectiveCount = 0;
  let copyCount = 0;
  let materializationCount = 0;
  let eventCount = 0;
  let repeatCount = 0;
  let repeatIterationCount = 0;
  let conditionalCount = 0;
  let resourceConditionalCount = 0;
  const runtimeControlIds = new Set<string>();
  let expandedNodeCount = 0;
  const dispatchEffects = new Map<
    HostGraphDispatchNode,
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
      if (node.kind === "dispatch") dispatchEffects.set(node, effects);
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
      const iterationCount = Number(wireIntegerToBigInt(node.iterationCount));
      repeatIterationCount += iterationCount;
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
      repeatEffects.set(node, aggregateLinearEffects(
        node.body,
        bodyEffects,
        "repeat",
      ));
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
  node: HostGraphExecutableNode,
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
  node: HostGraphExecutableNode,
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
      dispatchCount + (node.kind === "dispatch" ? multiplier : 0),
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
  node: HostGraphDispatchNode,
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
    HostGraphDispatchNode,
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
      const effectPath = node.kind === "dispatch"
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
    HostGraphDispatchNode,
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
  if (node.kind === "dispatch") {
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
