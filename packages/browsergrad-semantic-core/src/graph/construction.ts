import { canonicalizeJson } from "../schema/canonical-json.js";
import type { VerifiedKernelArtifact } from "../kernel/artifact.js";
import type { VerifiedLayoutArtifact } from "../layout/artifact.js";
import type { WireProducer } from "../schema/envelope.js";
import { hashSemanticArtifact } from "../schema/hash.js";
import {
  isJsonObject,
  parseWireJson,
  type JsonObject,
  type JsonValue,
} from "../schema/json.js";
import {
  DEFAULT_DECODE_LIMITS,
  resolveDecodeLimits,
  type DecodeLimits,
} from "../schema/limits.js";
import {
  HOST_GRAPH_ARTIFACT_SCHEMA,
  verifyHostGraphArtifact,
  type VerifiedHostGraphArtifact,
} from "./artifact.js";
import type { HostGraphProgram } from "./model.js";

const DEFAULT_PRODUCER = Object.freeze({
  id: "browsergrad.semantic-core.host-graph-construction",
  version: "1",
});

export interface HostGraphConstructionOptions {
  readonly kernelArtifacts: readonly VerifiedKernelArtifact[];
  readonly layoutArtifacts: readonly VerifiedLayoutArtifact[];
  readonly producer?: WireProducer;
  readonly artifactId?: string;
  readonly limits?: Partial<DecodeLimits>;
}

export interface ConstructedHostGraphArtifact {
  readonly artifact: VerifiedHostGraphArtifact;
  readonly graphSemanticHash: string;
}

interface NormalizedOptions {
  readonly kernelArtifacts: readonly VerifiedKernelArtifact[];
  readonly layoutArtifacts: readonly VerifiedLayoutArtifact[];
  readonly producer: WireProducer;
  readonly artifactId: string;
  readonly limits: DecodeLimits;
}

/**
 * Constructs one verified graph from semantic meaning only. Node ordering,
 * dependency closure, effect legality, collective policy, and artifact
 * references are normalized by the verifier instead of trusted from callers.
 */
export async function createVerifiedHostGraphArtifact(
  request: HostGraphProgram,
  options: HostGraphConstructionOptions,
): Promise<ConstructedHostGraphArtifact> {
  const normalized = normalizeOptions(options);
  const program = parseWireJson(
    canonicalizeJson(request, { limits: normalized.limits }),
    { limits: normalized.limits },
  );
  const artifact = await verifyHostGraphArtifact({
    schema: HOST_GRAPH_ARTIFACT_SCHEMA,
    version: { major: 1, minor: 0 },
    producer: normalized.producer,
    artifactId: normalized.artifactId,
    requiredExtensions: [],
    payload: { program },
  }, {
    kernelArtifacts: normalized.kernelArtifacts,
    layoutArtifacts: normalized.layoutArtifacts,
    limits: normalized.limits,
  });
  return Object.freeze({
    artifact,
    graphSemanticHash: await hashSemanticArtifact(artifact, {
      limits: normalized.limits,
    }),
  });
}

function normalizeOptions(
  options: HostGraphConstructionOptions,
): NormalizedOptions {
  const values = inspectOptions(options);
  const config = parseWireJson(
    canonicalizeJson({
      ...(values.producer === undefined
        ? {}
        : { producer: values.producer }),
      ...(values.artifactId === undefined
        ? {}
        : { artifactId: values.artifactId }),
      ...(values.limits === undefined ? {} : { limits: values.limits }),
    }),
  );
  if (!isJsonObject(config)) {
    throw new TypeError("host graph construction options must be an object");
  }
  const limits = normalizeLimits(config.limits);
  return Object.freeze({
    kernelArtifacts: values.kernelArtifacts as
      readonly VerifiedKernelArtifact[],
    layoutArtifacts: values.layoutArtifacts as
      readonly VerifiedLayoutArtifact[],
    producer: config.producer === undefined
      ? DEFAULT_PRODUCER
      : normalizeProducer(config.producer),
    artifactId: config.artifactId === undefined
      ? "host-graph"
      : nonemptyString(config.artifactId, "$options.artifactId"),
    limits,
  });
}

function inspectOptions(
  options: HostGraphConstructionOptions,
): Record<string, unknown> {
  if (typeof options !== "object" || options === null ||
      Object.getPrototypeOf(options) !== Object.prototype) {
    throw new TypeError("host graph construction options must be a plain object");
  }
  const allowed = new Set([
    "kernelArtifacts",
    "layoutArtifacts",
    "producer",
    "artifactId",
    "limits",
  ]);
  const descriptors = Object.getOwnPropertyDescriptors(options);
  const keys = Reflect.ownKeys(descriptors);
  if (keys.some((key) => typeof key !== "string" || !allowed.has(key))) {
    throw new TypeError("host graph construction options contain unknown fields");
  }
  const kernelArtifacts = descriptors.kernelArtifacts;
  if (kernelArtifacts === undefined ||
      !("value" in kernelArtifacts) ||
      kernelArtifacts.enumerable !== true) {
    throw new TypeError(
      "host graph construction requires kernelArtifacts as an enumerable data property",
    );
  }
  const layoutArtifacts = descriptors.layoutArtifacts;
  if (layoutArtifacts === undefined ||
      !("value" in layoutArtifacts) ||
      layoutArtifacts.enumerable !== true) {
    throw new TypeError(
      "host graph construction requires layoutArtifacts as an enumerable data property",
    );
  }
  const result: Record<string, unknown> = {
    kernelArtifacts: kernelArtifacts.value,
    layoutArtifacts: layoutArtifacts.value,
  };
  for (const key of ["producer", "artifactId", "limits"] as const) {
    const descriptor = descriptors[key];
    if (descriptor === undefined) continue;
    if (!("value" in descriptor) || descriptor.enumerable !== true) {
      throw new TypeError(
        `host graph construction option ${key} must be an enumerable data property`,
      );
    }
    result[key] = descriptor.value;
  }
  return result;
}

function normalizeProducer(value: JsonValue): WireProducer {
  const object = closedObject(
    value,
    ["id", "version"],
    "$options.producer",
  );
  return Object.freeze({
    id: nonemptyString(object.id, "$options.producer.id"),
    version: nonemptyString(object.version, "$options.producer.version"),
  });
}

function normalizeLimits(value: JsonValue | undefined): DecodeLimits {
  if (value === undefined) return DEFAULT_DECODE_LIMITS;
  const object = closedObject(
    value,
    Object.keys(DEFAULT_DECODE_LIMITS),
    "$options.limits",
  );
  return resolveDecodeLimits(object as unknown as Partial<DecodeLimits>);
}

function closedObject(
  value: JsonValue,
  allowedFields: readonly string[],
  path: string,
): JsonObject {
  if (!isJsonObject(value)) {
    throw new TypeError(`${path} must be an object`);
  }
  const allowed = new Set(allowedFields);
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length > 0) {
    throw new TypeError(`${path} contains unknown fields: ${unknown.sort().join(", ")}`);
  }
  return value;
}

function nonemptyString(value: JsonValue | undefined, path: string): string {
  if (typeof value !== "string" || value.length === 0 ||
      value.length > 4_096) {
    throw new TypeError(`${path} must be a bounded non-empty string`);
  }
  return value;
}
