import {
  SCHEMA_DIAGNOSTIC_CODES,
  SemanticSchemaError,
  deepFreezeJson,
  hashCanonicalJson,
  resolveDecodeLimits,
  sha256Hex,
  wireIntegerToBigInt,
  type DecodeLimits,
  type JsonObject,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapVerifiedCppCuteFrontendArtifact,
  unwrapVerifiedCppCuteFrontendArtifactResource,
  type VerifiedCppCuteFrontendArtifact,
  type VerifiedCppCuteFrontendArtifactResource,
} from "./cpp_cute_frontend_artifact.js";
import { findCppCutePreparedFrontendProfileBindingMismatch } from "./cpp_cute_frontend_profile_binding.js";
import {
  copyPreparedCppCuteFrontendSourceSnapshots,
  unwrapPreparedCppCuteFrontendRequest,
  type PreparedCppCuteFrontendRequest,
} from "./cpp_cute_frontend_request.js";
import { unwrapPreparedCppCuteFrontendProfile } from "./cpp_cute_frontend_profile.js";
import type {
  CppCuteFrontendPayloadV3,
  CppCuteInputClosureV3,
} from "./cpp_cute_frontend_types.js";

export const CPP_CUTE_FRONTEND_REQUEST_BINDING_SCHEMA =
  "browsergrad.compiler.cpp-cute.frontend-request-binding";
export const CPP_CUTE_FRONTEND_REQUEST_BINDING_MAJOR = 1;
export const CPP_CUTE_FRONTEND_REQUEST_BINDING_MINOR = 0;

const PREPARED_BINDINGS = new WeakMap<object, StoredCppCuteFrontendRequestBinding>();

export type CppCuteFrontendRequestSelectionV1 =
  | (JsonObject & {
      readonly kind: "resolved";
      readonly requestId: string;
      readonly anchorTokenSha256: string;
      readonly resolvedEntryId: string;
      readonly resolvedRootSourceEntityId: string;
      readonly anchorMatch: "spelling" | "expansion";
    })
  | (JsonObject & {
      readonly kind: "rejected";
      readonly requestId: string;
      readonly anchorTokenSha256: string;
      readonly blockingDiagnosticIds: readonly string[];
    });

/** Host-derived semantic relation. Producer observations belong to separate verified evidence. */
export interface CppCuteFrontendRequestBindingBodyV1 extends JsonObject {
  readonly schema: typeof CPP_CUTE_FRONTEND_REQUEST_BINDING_SCHEMA;
  readonly version: JsonObject & {
    readonly major: typeof CPP_CUTE_FRONTEND_REQUEST_BINDING_MAJOR;
    readonly minor: typeof CPP_CUTE_FRONTEND_REQUEST_BINDING_MINOR;
  };
  readonly requestId: string;
  readonly artifactId: string;
  readonly artifactBytesSha256: string;
  readonly artifactByteLength: WireU64;
  readonly selection: CppCuteFrontendRequestSelectionV1;
}

export interface CppCuteFrontendRequestBindingV1 extends CppCuteFrontendRequestBindingBodyV1 {
  readonly bindingId: string;
}

export interface PrepareCppCuteFrontendRequestBindingOptions {
  readonly limits?: Partial<DecodeLimits>;
  readonly signal?: AbortSignal;
}

declare const preparedCppCuteFrontendRequestBindingBrand: unique symbol;

export interface PreparedCppCuteFrontendRequestBinding {
  readonly [preparedCppCuteFrontendRequestBindingBrand]: true;
  readonly bindingId: string;
  readonly bindingHash: string;
  readonly profileHash: string;
  readonly requestId: string;
  readonly artifactId: string;
  readonly artifactBytesSha256: string;
  readonly inputClosureSha256: string;
  readonly outcome: "accepted" | "rejected";
}

export interface PreparedCppCuteFrontendRequestBindingRecord {
  readonly request: PreparedCppCuteFrontendRequest;
  readonly artifactResource: VerifiedCppCuteFrontendArtifactResource;
  readonly artifact: VerifiedCppCuteFrontendArtifact;
  readonly binding: CppCuteFrontendRequestBindingV1;
}

type StoredCppCuteFrontendRequestBinding = PreparedCppCuteFrontendRequestBindingRecord;

export type CppCuteFrontendRequestBindingErrorCode =
  | "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-CANCELLED"
  | "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-INVALID"
  | "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-REQUEST-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-ARTIFACT-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-INPUT-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-SELECTION-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-CONFORMANCE-MISMATCH"
  | "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-RESOURCE-LIMIT"
  | "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-UNVERIFIED";

export class CppCuteFrontendRequestBindingError extends Error {
  constructor(
    readonly code: CppCuteFrontendRequestBindingErrorCode,
    readonly path: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${code}: ${message}`, options);
    this.name = "CppCuteFrontendRequestBindingError";
  }
}

/** Derive one producer-neutral relation from opaque request and artifact authorities. */
export async function prepareCppCuteFrontendRequestBinding(
  request: PreparedCppCuteFrontendRequest,
  artifactResource: VerifiedCppCuteFrontendArtifactResource,
  options: PrepareCppCuteFrontendRequestBindingOptions = {},
): Promise<PreparedCppCuteFrontendRequestBinding> {
  const normalizedOptions = normalizeOptions(options);
  throwIfAborted(normalizedOptions.signal);
  const requestRecord = unwrapPreparedCppCuteFrontendRequest(request);
  const profileRecord = unwrapPreparedCppCuteFrontendProfile(requestRecord.profile);
  const artifact = unwrapVerifiedCppCuteFrontendArtifactResource(artifactResource);
  const artifactRecord = unwrapVerifiedCppCuteFrontendArtifact(artifact);
  const payload = artifactRecord.envelope.payload;
  if (artifact.compilationContractHash !== request.compilationContractHash) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-ARTIFACT-MISMATCH",
      "$.artifact.compilationContractHash",
      "artifact compilation contract differs from prepared request",
    );
  }
  const profileMismatch = findCppCutePreparedFrontendProfileBindingMismatch(payload, profileRecord);
  if (profileMismatch !== null) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-ARTIFACT-MISMATCH",
      profileMismatch.path,
      profileMismatch.message,
    );
  }
  await verifySourceAuthority(request, payload, normalizedOptions.signal);
  throwIfAborted(normalizedOptions.signal);
  const selection = expectedSelectionFor(request, payload);
  verifyDetachedConformance(request, artifact);
  const body: CppCuteFrontendRequestBindingBodyV1 = {
    schema: CPP_CUTE_FRONTEND_REQUEST_BINDING_SCHEMA,
    version: {
      major: CPP_CUTE_FRONTEND_REQUEST_BINDING_MAJOR,
      minor: CPP_CUTE_FRONTEND_REQUEST_BINDING_MINOR,
    },
    requestId: request.requestId,
    artifactId: artifact.artifactId,
    artifactBytesSha256: artifact.artifactBytesSha256,
    artifactByteLength: artifact.artifactByteLength,
    selection,
  };
  const bindingHash = await deriveCppCuteFrontendRequestBindingHash(body, {
    limits: normalizedOptions.limits,
  });
  throwIfAborted(normalizedOptions.signal);
  const bindingId = `bg.cpp.frontend-request-binding.sha256.${bindingHash}`;
  const binding = deepFreezeJson({ ...body, bindingId });
  const prepared = Object.freeze({
    bindingId,
    bindingHash,
    profileHash: request.profileHash,
    requestId: request.requestId,
    artifactId: artifact.artifactId,
    artifactBytesSha256: artifact.artifactBytesSha256,
    inputClosureSha256: artifact.inputClosureSha256,
    outcome: artifact.outcome,
  }) as PreparedCppCuteFrontendRequestBinding;
  PREPARED_BINDINGS.set(prepared, Object.freeze({
    request,
    artifactResource,
    artifact,
    binding,
  }));
  return prepared;
}

export function unwrapPreparedCppCuteFrontendRequestBinding(
  prepared: PreparedCppCuteFrontendRequestBinding,
): PreparedCppCuteFrontendRequestBindingRecord {
  if (typeof prepared !== "object" || prepared === null) unverified();
  const record = PREPARED_BINDINGS.get(prepared as object);
  if (record === undefined) unverified();
  return record;
}

export async function deriveCppCuteFrontendRequestBindingHash(
  binding: CppCuteFrontendRequestBindingV1 | CppCuteFrontendRequestBindingBodyV1,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  return hashJson({
    domain: "browsergrad.compiler.cpp-cute.frontend-request-binding.v1",
    binding: {
      schema: binding.schema,
      version: binding.version,
      requestId: binding.requestId,
      artifactId: binding.artifactId,
      artifactBytesSha256: binding.artifactBytesSha256,
      artifactByteLength: binding.artifactByteLength,
      selection: binding.selection,
    },
  }, "$.bindingId", options.limits);
}

async function verifySourceAuthority(
  request: PreparedCppCuteFrontendRequest,
  payload: CppCuteFrontendPayloadV3,
  signal: AbortSignal | undefined,
): Promise<void> {
  const requestRecord = unwrapPreparedCppCuteFrontendRequest(request);
  const snapshots = copyPreparedCppCuteFrontendSourceSnapshots(request);
  const sourceFiles = payload.inputs.files.filter((file) => file.owner.kind === "source");
  if (sourceFiles.length !== requestRecord.request.files.length || snapshots.length !== requestRecord.request.files.length) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-INPUT-MISMATCH",
      "$.artifact.inputs.files",
      "artifact source-owned input cardinality differs from prepared request authority",
    );
  }
  for (const [index, descriptor] of requestRecord.request.files.entries()) {
    throwIfAborted(signal);
    const snapshot = snapshots[index];
    const actual = sourceFiles.find((candidate) => candidate.virtualPath === descriptor.virtualPath);
    if (snapshot === undefined || snapshot.virtualPath !== descriptor.virtualPath) {
      mismatch(
        "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-INPUT-MISMATCH",
        `$.request.files[${index}]`,
        "prepared request source snapshot order differs from its descriptor",
      );
    }
    const snapshotSha256 = await sha256Hex(snapshot.bytes);
    if (snapshotSha256 !== descriptor.contentSha256 ||
        BigInt(snapshot.bytes.byteLength) !== wireIntegerToBigInt(descriptor.byteLength)) {
      mismatch(
        "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-INPUT-MISMATCH",
        `$.request.files[${index}]`,
        "prepared request source snapshot differs from its descriptor",
      );
    }
    if (actual === undefined || actual.fileId !== descriptor.fileId || actual.role !== descriptor.role ||
        actual.contentSha256 !== descriptor.contentSha256 || actual.byteLength !== descriptor.byteLength ||
        actual.includeRootId !== descriptor.includeRootId || actual.owner.kind !== "source") {
      mismatch(
        "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-INPUT-MISMATCH",
        `$.artifact.inputs.files[${index}]`,
        "artifact source differs from exact prepared request descriptor and bytes",
      );
    }
  }
  const mainFile = payload.inputs.files.find((file) => file.fileId === payload.inputs.mainFileId);
  if (mainFile?.virtualPath !== request.mainVirtualPath || mainFile.role !== "main-source") {
    mismatch(
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-INPUT-MISMATCH",
      "$.artifact.inputs.mainFileId",
      "artifact main source differs from prepared request",
    );
  }
}

function expectedSelectionFor(
  request: PreparedCppCuteFrontendRequest,
  payload: CppCuteFrontendPayloadV3,
): CppCuteFrontendRequestSelectionV1 {
  const requestEntry = unwrapPreparedCppCuteFrontendRequest(request).request.entryRequests[0];
  if (requestEntry === undefined) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-REQUEST-MISMATCH",
      "$.request.entryRequests",
      "prepared request lost its entry selection",
    );
  }
  if (payload.outcome.kind === "accepted") {
    const selectedEntryId = payload.outcome.selectedEntryIds.length === 1
      ? payload.outcome.selectedEntryIds[0]
      : undefined;
    const selectedEntry = selectedEntryId === undefined
      ? undefined
      : payload.entries.find((entry) => entry.entryId === selectedEntryId);
    if (selectedEntry === undefined || selectedEntry.kind !== requestEntry.kind) {
      mismatch(
        "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-SELECTION-MISMATCH",
        "$.artifact.outcome.selectedEntryIds",
        "accepted artifact must resolve exactly one entry of requested family",
      );
    }
    const resolvedRoot = resolveRequestedSourceRoot(requestEntry, selectedEntry, payload);
    return deepFreezeJson({
      kind: "resolved",
      requestId: requestEntry.requestId,
      anchorTokenSha256: requestEntry.anchor.tokenSha256,
      resolvedEntryId: selectedEntry.entryId,
      resolvedRootSourceEntityId: resolvedRoot.sourceEntityId,
      anchorMatch: resolvedRoot.anchorMatch,
    });
  }
  return deepFreezeJson({
    kind: "rejected",
    requestId: requestEntry.requestId,
    anchorTokenSha256: requestEntry.anchor.tokenSha256,
    blockingDiagnosticIds: payload.outcome.blockingDiagnosticIds,
  });
}

function resolveRequestedSourceRoot(
  requestEntry: ReturnType<typeof unwrapPreparedCppCuteFrontendRequest>["request"]["entryRequests"][number],
  selectedEntry: CppCuteFrontendPayloadV3["entries"][number],
  payload: CppCuteFrontendPayloadV3,
): { readonly sourceEntityId: string; readonly anchorMatch: "spelling" | "expansion" } {
  const matches = selectedEntry.selectedRootDeclarationIds.flatMap((declarationId) => {
    const declaration = payload.declarations.find((candidate) => candidate.declarationId === declarationId);
    if (declaration === undefined || declaration.kind !== requestEntry.declarationKind ||
        declaration.origin.kind !== "source" || declaration.identitySpanId === null) return [];
    const span = payload.spans.find((candidate) => candidate.spanId === declaration.identitySpanId);
    if (span === undefined) return [];
    const anchorMatch = sourceAnchorMatch(requestEntry.anchor, span, payload.inputs);
    if (anchorMatch === null) return [];
    const sourceEntity = payload.sourceEntities.find((candidate) =>
      candidate.entityKind === declaration.kind &&
      candidate.canonicalIdentity === declaration.canonicalUsr &&
      sameSourceOrigin(candidate.origin, declaration.origin));
    return sourceEntity === undefined ? [] : [{ sourceEntityId: sourceEntity.sourceEntityId, anchorMatch }];
  });
  if (matches.length !== 1) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-SELECTION-MISMATCH",
      "$.request.entryRequests[0].anchor",
      "request anchor must resolve exactly one selected artifact root of requested declaration kind",
    );
  }
  return matches[0]!;
}

function sourceAnchorMatch(
  anchor: ReturnType<typeof unwrapPreparedCppCuteFrontendRequest>["request"]["entryRequests"][number]["anchor"],
  span: CppCuteFrontendPayloadV3["spans"][number],
  inputs: CppCuteInputClosureV3,
): "spelling" | "expansion" | null {
  const anchorBegin = wireIntegerToBigInt(anchor.beginByte);
  const anchorEnd = wireIntegerToBigInt(anchor.endByte);
  for (const kind of ["spelling", "expansion"] as const) {
    const range = span[kind];
    const file = inputs.files.find((candidate) => candidate.fileId === range.fileId);
    if (file?.virtualPath === anchor.virtualPath &&
        wireIntegerToBigInt(range.startByte) === anchorBegin &&
        wireIntegerToBigInt(range.endByte) === anchorEnd) return kind;
  }
  return null;
}

function sameSourceOrigin(
  left: CppCuteFrontendPayloadV3["sourceEntities"][number]["origin"],
  right: CppCuteFrontendPayloadV3["declarations"][number]["origin"],
): boolean {
  return left.kind === right.kind && (left.kind === "source"
    ? right.kind === "source" && left.spanId === right.spanId
    : right.kind === "implicit" && left.anchorSpanId === right.anchorSpanId && left.reason === right.reason);
}

function verifyDetachedConformance(
  request: PreparedCppCuteFrontendRequest,
  artifact: VerifiedCppCuteFrontendArtifact,
): void {
  const expected = unwrapPreparedCppCuteFrontendRequest(request).detached.conformance;
  if (expected === null) return;
  if (expected.expectedArtifactSha256 !== null &&
      expected.expectedArtifactSha256 !== artifact.artifactBytesSha256) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-CONFORMANCE-MISMATCH",
      "$.detached.conformance.expectedArtifactSha256",
      "artifact bytes differ from detached conformance assertion",
    );
  }
  if (expected.expectedOpenedHeaderSetSha256 !== null &&
      expected.expectedOpenedHeaderSetSha256 !== artifact.headerSetSha256) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-CONFORMANCE-MISMATCH",
      "$.detached.conformance.expectedOpenedHeaderSetSha256",
      "opened header set differs from detached conformance assertion",
    );
  }
  if (expected.expectedInputClosureSha256 !== null &&
      expected.expectedInputClosureSha256 !== artifact.inputClosureSha256) {
    mismatch(
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-CONFORMANCE-MISMATCH",
      "$.detached.conformance.expectedInputClosureSha256",
      "input closure differs from detached conformance assertion",
    );
  }
}

function normalizeOptions(options: PrepareCppCuteFrontendRequestBindingOptions): {
  readonly limits: DecodeLimits;
  readonly signal: AbortSignal | undefined;
} {
  const descriptors = plainDataRecord(options, "$.options", ["limits", "signal"]);
  const rawLimits = optionalDescriptorValue(descriptors, "limits");
  let limits: DecodeLimits;
  try {
    limits = resolveDecodeLimits(rawLimits as Partial<DecodeLimits> | undefined);
  } catch (error) {
    resource("$.options.limits", "invalid semantic decode limits", { cause: error });
  }
  const rawSignal = optionalDescriptorValue(descriptors, "signal");
  if (rawSignal !== undefined && !(rawSignal instanceof AbortSignal)) {
    invalid("$.options.signal", "signal must be AbortSignal");
  }
  return Object.freeze({ limits, signal: rawSignal as AbortSignal | undefined });
}

function plainDataRecord(
  value: unknown,
  path: string,
  allowedKeys: readonly string[],
): PropertyDescriptorMap {
  if (typeof value !== "object" || value === null || Array.isArray(value)) invalid(path, "expected plain object");
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) invalid(path, "expected plain object");
  const descriptors = Object.getOwnPropertyDescriptors(value);
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== "string" || !allowedKeys.includes(key)) invalid(path, "unknown option field");
    const descriptor = descriptors[key];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalid(`${path}.${key}`, "expected enumerable data property");
    }
  }
  return descriptors;
}

function optionalDescriptorValue(descriptors: PropertyDescriptorMap, name: string): unknown {
  const descriptor = descriptors[name];
  return descriptor !== undefined && "value" in descriptor ? descriptor.value : undefined;
}

async function hashJson(value: JsonObject, path: string, limits?: Partial<DecodeLimits>): Promise<string> {
  try {
    return await hashCanonicalJson(value, limits === undefined ? {} : { limits });
  } catch (error) {
    if (error instanceof SemanticSchemaError && error.diagnostic.code === SCHEMA_DIAGNOSTIC_CODES.resourceLimit) {
      resource(error.diagnostic.path ?? path, "canonical hash exceeded fixed resource limits", { cause: error });
    }
    invalid(path, "canonical hash input is invalid", { cause: error });
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    fail(
      "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-CANCELLED",
      "$.signal",
      "frontend request binding was aborted",
    );
  }
}

function unverified(): never {
  fail(
    "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-UNVERIFIED",
    "$",
    "operation requires opaque prepared frontend request binding",
  );
}

function resource(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-REQUEST-BINDING-RESOURCE-LIMIT", path, message, options);
}

function invalid(path: string, message: string, options?: ErrorOptions): never {
  fail("BG-COMPILER-CPP-CUTE-REQUEST-BINDING-INVALID", path, message, options);
}

function mismatch(
  code: Extract<CppCuteFrontendRequestBindingErrorCode,
    | "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-REQUEST-MISMATCH"
    | "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-ARTIFACT-MISMATCH"
    | "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-INPUT-MISMATCH"
    | "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-SELECTION-MISMATCH"
    | "BG-COMPILER-CPP-CUTE-REQUEST-BINDING-CONFORMANCE-MISMATCH">,
  path: string,
  message: string,
): never {
  fail(code, path, message);
}

function fail(
  code: CppCuteFrontendRequestBindingErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteFrontendRequestBindingError(code, path, message, options);
}
