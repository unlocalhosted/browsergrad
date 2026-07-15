import { canonicalizeJson } from "../schema/canonical-json.js";
import { KERNEL_DIAGNOSTIC_CODES, SemanticSchemaError } from "../schema/diagnostics.js";
import {
  unwrapVerifiedArtifact,
  validateWireEnvelope,
  verifyWireArtifact,
  type VerifiedArtifact,
  type WireEnvelope,
} from "../schema/envelope.js";
import { parseFloatBits } from "../schema/float-bits.js";
import { hashCanonicalJson, hashSemanticArtifact } from "../schema/hash.js";
import { decodeWireJson, isJsonObject, type JsonObject, type JsonValue } from "../schema/json.js";
import { type DecodeLimits, resolveDecodeLimits } from "../schema/limits.js";
import { unwrapLayoutArtifact, type VerifiedLayoutArtifact } from "../layout/artifact.js";
import { getBuiltinDType, type BuiltinDTypeId } from "../layout/dtype.js";
import type { InvalidSourcePolicy, ViewCopyOperation } from "./model.js";

export const KERNEL_ARTIFACT_SCHEMA = "browsergrad.kernel";
export const KERNEL_ARTIFACT_MAJOR = 1;
export const KERNEL_ARTIFACT_MINOR = 0;
const KERNEL_ARTIFACT_AUTHORITY = Object.freeze({ schema: KERNEL_ARTIFACT_SCHEMA, major: KERNEL_ARTIFACT_MAJOR });

export type KernelArtifactPayloadV1 = JsonObject & {
  readonly layoutSemanticHash: string;
  readonly operations: readonly ViewCopyOperation[];
};

export type VerifiedKernelArtifact = VerifiedArtifact<KernelArtifactPayloadV1>;

export interface KernelArtifactVerificationOptions {
  readonly layout: VerifiedLayoutArtifact;
  readonly limits?: Partial<DecodeLimits>;
}

interface RawKernelArtifact {
  readonly layoutSemanticHash: string;
  readonly operations: readonly ViewCopyOperation[];
}

export async function verifyKernelArtifact(
  value: unknown,
  options: KernelArtifactVerificationOptions,
): Promise<VerifiedKernelArtifact> {
  const limits = resolveDecodeLimits(options.limits);
  const envelope = validateWireEnvelope(value, {
    schema: KERNEL_ARTIFACT_SCHEMA,
    supportedMajor: KERNEL_ARTIFACT_MAJOR,
    supportedMinor: KERNEL_ARTIFACT_MINOR,
    knownRequiredExtensions: new Set(),
    limits,
  });
  const raw = parsePayload(envelope.payload, limits);
  const actualLayoutHash = await hashSemanticArtifact(options.layout, { limits });
  if (raw.layoutSemanticHash !== actualLayoutHash) {
    invalid(KERNEL_DIAGNOSTIC_CODES.layoutHashMismatch, "$.payload.layoutSemanticHash", "kernel artifact does not reference the supplied verified layout semantics");
  }

  const provisional = remapOperationIds(raw, "provisional");
  const scopeDigest = await hashCanonicalJson({
    domain: "browsergrad.kernel-id-scope.v1",
    payload: provisional,
  }, { limits });
  const normalized = remapOperationIds(raw, scopeDigest);
  verifySemantics(normalized, options.layout, limits);

  const normalizedEnvelope: WireEnvelope<JsonValue> = {
    ...envelope,
    payload: normalized as unknown as JsonValue,
  };
  canonicalizeJson(normalizedEnvelope, { limits });
  return verifyWireArtifact(normalizedEnvelope, {
    schema: KERNEL_ARTIFACT_SCHEMA,
    supportedMajor: KERNEL_ARTIFACT_MAJOR,
    supportedMinor: KERNEL_ARTIFACT_MINOR,
    knownRequiredExtensions: new Set(),
    limits,
    validatePayload: (payload) => payload,
  }, KERNEL_ARTIFACT_AUTHORITY) as VerifiedKernelArtifact;
}

export async function decodeKernelArtifact(
  bytes: Uint8Array,
  options: KernelArtifactVerificationOptions,
): Promise<VerifiedKernelArtifact> {
  return verifyKernelArtifact(decodeWireJson(bytes, options.limits === undefined ? {} : { limits: options.limits }), options);
}

export function kernelArtifactPayload(artifact: VerifiedKernelArtifact): KernelArtifactPayloadV1 {
  const envelope = unwrapVerifiedArtifact(artifact, KERNEL_ARTIFACT_AUTHORITY);
  if (envelope.schema !== KERNEL_ARTIFACT_SCHEMA || envelope.version.major !== KERNEL_ARTIFACT_MAJOR) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, "$", "verified artifact is not a browsergrad.kernel@1 artifact");
  }
  return envelope.payload;
}

function parsePayload(value: JsonValue, limits: DecodeLimits): RawKernelArtifact {
  const object = closedObject(value, ["layoutSemanticHash", "operations"], "$.payload");
  const layoutSemanticHash = stringValue(field(object, "layoutSemanticHash", "$.payload"), "$.payload.layoutSemanticHash");
  if (!/^[0-9a-f]{64}$/u.test(layoutSemanticHash)) {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, "$.payload.layoutSemanticHash", "layout semantic hash must be 64 lowercase hexadecimal digits");
  }
  const operationValues = arrayValue(field(object, "operations", "$.payload"), "$.payload.operations");
  if (operationValues.length !== 1) invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, "$.payload.operations", "kernel v1 requires exactly one standalone operation; sequencing belongs to the host graph");
  if (operationValues.length > limits.maxArrayLength) resource("$.payload.operations", `operation count exceeds ${limits.maxArrayLength}`);
  const operations = operationValues.map((operation, index) => parseOperation(operation, `$.payload.operations[${index}]`));
  uniqueIds(operations.map((operation) => operation.operationId), "$.payload.operations");
  return { layoutSemanticHash, operations };
}

function parseOperation(value: JsonValue, path: string): ViewCopyOperation {
  const object = closedObject(value, ["operationId", "kind", "version", "dtype", "source", "destination", "overlap"], path);
  if (object.kind !== "view-copy") invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, `${path}.kind`, "kernel v1 supports only view-copy operations");
  const version = closedObject(field(object, "version", path), ["major", "minor"], `${path}.version`);
  if (version.major !== 1 || version.minor !== 0) {
    invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `${path}.version`, "view-copy reader supports operation version 1.0 only");
  }
  const dtypeValue = stringValue(field(object, "dtype", path), `${path}.dtype`);
  const dtype = getBuiltinDType(dtypeValue, `${path}.dtype`).id;
  const source = closedObject(field(object, "source", path), ["viewId", "access", "invalidSource"], `${path}.source`);
  if (source.access !== "read") invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, `${path}.source.access`, "view-copy source effect must be read");
  const destination = closedObject(field(object, "destination", path), ["viewId", "access"], `${path}.destination`);
  if (destination.access !== "write") invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, `${path}.destination.access`, "view-copy destination effect must be write");
  const overlap = closedObject(field(object, "overlap", path), ["kind"], `${path}.overlap`);
  if (overlap.kind !== "forbid") invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, `${path}.overlap.kind`, "kernel v1 overlap policy must be forbid");
  return {
    operationId: localId(field(object, "operationId", path), `${path}.operationId`),
    kind: "view-copy",
    version: { major: 1, minor: 0 },
    dtype: dtype as BuiltinDTypeId,
    source: {
      viewId: localId(field(source, "viewId", `${path}.source`), `${path}.source.viewId`),
      access: "read",
      invalidSource: parseInvalidSource(field(source, "invalidSource", `${path}.source`), `${path}.source.invalidSource`),
    },
    destination: {
      viewId: localId(field(destination, "viewId", `${path}.destination`), `${path}.destination.viewId`),
      access: "write",
    },
    overlap: { kind: "forbid" },
  };
}

function parseInvalidSource(value: JsonValue, path: string): InvalidSourcePolicy {
  if (!isJsonObject(value) || typeof value.kind !== "string") {
    invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, path, "invalid-source policy must be a tagged object");
  }
  if (value.kind === "reject") {
    exactFields(value, ["kind"], path);
    return { kind: "reject" };
  }
  if (value.kind === "fill") {
    exactFields(value, ["kind", "value"], path);
    return { kind: "fill", value: parseFloatBits(field(value, "value", path), `${path}.value`) };
  }
  invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, `${path}.kind`, `unknown invalid-source policy ${JSON.stringify(value.kind)}`);
}

function remapOperationIds(raw: RawKernelArtifact, scope: string): KernelArtifactPayloadV1 {
  return {
    layoutSemanticHash: raw.layoutSemanticHash,
    operations: raw.operations.map((operation, index) => ({
      ...operation,
      operationId: scope === "provisional"
        ? `@kernel-operation/${index}`
        : `bg.entity.kernel-operation.scope-sha256.${scope}.ordinal.${index}`,
    })),
  } as unknown as KernelArtifactPayloadV1;
}

function verifySemantics(
  payload: KernelArtifactPayloadV1,
  layoutArtifact: VerifiedLayoutArtifact,
  limits: DecodeLimits,
): void {
  const layout = unwrapLayoutArtifact(layoutArtifact);
  const views = new Map(layout.views.map((view) => [view.viewId, view]));
  const allocations = new Map(layout.allocations.map((allocation) => [allocation.allocationId, allocation]));
  for (const [index, operation] of payload.operations.entries()) {
    const path = `$.payload.operations[${index}]`;
    const source = views.get(operation.source.viewId);
    const destination = views.get(operation.destination.viewId);
    if (source === undefined) invalid(KERNEL_DIAGNOSTIC_CODES.danglingReference, `${path}.source.viewId`, `unknown source view ${operation.source.viewId}`);
    if (destination === undefined) invalid(KERNEL_DIAGNOSTIC_CODES.danglingReference, `${path}.destination.viewId`, `unknown destination view ${operation.destination.viewId}`);
    if (source.dtype !== operation.dtype || destination.dtype !== operation.dtype) {
      invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `${path}.dtype`, "operation, source, and destination dtypes must match exactly");
    }
    if (destination.shape.length !== source.shape.length) {
      invalid(KERNEL_DIAGNOSTIC_CODES.shapeMismatch, path, "source and destination view-copy ranks must match");
    }
    if (canonicalizeJson(source.shape, { limits }) !== canonicalizeJson(destination.shape, { limits })) {
      invalid(KERNEL_DIAGNOSTIC_CODES.shapeMismatch, path, "source and destination logical shapes must be canonically identical");
    }
    const sourceAllocation = allocations.get(source.allocationId);
    const destinationAllocation = allocations.get(destination.allocationId);
    if (sourceAllocation === undefined || destinationAllocation === undefined) throw new Error("internal: verified layout allocation disappeared");
    if (destinationAllocation.memorySpace.kind === "constant") {
      invalid(KERNEL_DIAGNOSTIC_CODES.unsupportedProfile, `${path}.destination`, "view-copy destination cannot use read-only constant memory");
    }
    if (sourceAllocation.allocationId === destinationAllocation.allocationId || sourceAllocation.aliasSetId === destinationAllocation.aliasSetId) {
      invalid(KERNEL_DIAGNOSTIC_CODES.aliasConflict, `${path}.overlap`, "forbid-overlap view copies require disjoint allocation and alias-set identities");
    }
    if (operation.source.invalidSource.kind === "fill" && operation.source.invalidSource.value.dtype !== operation.dtype) {
      invalid(KERNEL_DIAGNOSTIC_CODES.invalidFill, `${path}.source.invalidSource.value.dtype`, "fill dtype must match the view-copy dtype exactly");
    }
  }
}

function closedObject(value: JsonValue, fields: readonly string[], path: string): JsonObject {
  if (!isJsonObject(value)) invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, path, "expected object");
  exactFields(value, fields, path);
  return value;
}

function exactFields(object: JsonObject, fields: readonly string[], path: string): void {
  const allowed = new Set(fields);
  const unknown = Object.keys(object).filter((key) => !allowed.has(key));
  if (unknown.length > 0) invalid(KERNEL_DIAGNOSTIC_CODES.unknownField, path, `unknown fields: ${unknown.sort().join(", ")}`);
  for (const fieldName of fields) {
    if (object[fieldName] === undefined) invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, `${path}.${fieldName}`, "required field is missing");
  }
}

function field(object: JsonObject, name: string, path: string): JsonValue {
  const value = object[name];
  if (value === undefined) invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, `${path}.${name}`, "required field is missing");
  return value;
}

function arrayValue(value: JsonValue, path: string): readonly JsonValue[] {
  if (!Array.isArray(value)) invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, path, "expected array");
  return value;
}

function stringValue(value: JsonValue, path: string): string {
  if (typeof value !== "string") invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, path, "expected string");
  return value;
}

function localId(value: JsonValue, path: string): string {
  const result = stringValue(value, path);
  if (!/^[A-Za-z_][A-Za-z0-9_.:-]{0,191}$/u.test(result)) invalid(KERNEL_DIAGNOSTIC_CODES.invalidArtifact, path, "invalid local or canonical identifier");
  return result;
}

function uniqueIds(ids: readonly string[], path: string): void {
  const seen = new Set<string>();
  for (const [index, id] of ids.entries()) {
    if (seen.has(id)) invalid(KERNEL_DIAGNOSTIC_CODES.duplicateId, `${path}[${index}]`, `duplicate ID ${id}`);
    seen.add(id);
  }
}

function resource(path: string, message: string): never {
  invalid(KERNEL_DIAGNOSTIC_CODES.resourceLimit, path, message);
}

function invalid(code: `BG-KERNEL-${string}`, path: string, message: string): never {
  throw new SemanticSchemaError({ code, stage: "verification", severity: "error", message, path });
}
