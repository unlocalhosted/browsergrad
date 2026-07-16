import {
  canonicalJsonBytes,
  decodeWireJson,
  encodeWireU64,
  hashCanonicalJson,
  resolveDecodeLimits,
  sha256Hex,
  validateWireEnvelope,
  type DecodeLimits,
  type JsonValue,
  type WireU64,
  type WireEnvelope,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
  CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
  CPP_CUTE_FRONTEND_ARTIFACT_SCHEMA,
  type CppCuteFrontendPayloadV1,
} from "./cpp_cute_frontend_types.js";
import {
  cppCuteFrontendArtifactFailure,
  DEFAULT_CPP_CUTE_FRONTEND_ARTIFACT_LIMITS,
  parseCppCuteFrontendPayload,
  type CppCuteFrontendArtifactLimits,
} from "./cpp_cute_frontend_parse.js";
import {
  verifyCppCuteFrontendPayload,
  type VerifiedCppCuteInputHashes,
} from "./cpp_cute_frontend_verify.js";

const VERIFIED_ARTIFACTS = new WeakMap<object, VerifiedCppCuteFrontendArtifactRecord>();
const VERIFIED_ARTIFACT_RESOURCES = new WeakMap<object, VerifiedCppCuteFrontendArtifact>();
const STABLE_ID_KIND = /^[a-z][a-z0-9-]*$/u;

declare const verifiedCppCuteFrontendArtifactBrand: unique symbol;

export interface VerifiedCppCuteFrontendArtifact {
  readonly [verifiedCppCuteFrontendArtifactBrand]: true;
  readonly artifactId: string;
  readonly artifactHash: string;
  readonly transportHash: string;
  readonly artifactBytesSha256: string;
  readonly artifactByteLength: WireU64;
  readonly profileHash: string;
  readonly sourceSetSha256: string;
  readonly headerSetSha256: string;
  readonly inputClosureSha256: string;
  readonly outcome: "accepted" | "rejected";
}

declare const verifiedCppCuteFrontendArtifactResourceBrand: unique symbol;

/** Opaque proof that exact canonical artifact bytes passed strict decoding. */
export interface VerifiedCppCuteFrontendArtifactResource {
  readonly [verifiedCppCuteFrontendArtifactResourceBrand]: true;
  readonly artifactId: string;
  readonly artifactHash: string;
  readonly artifactBytesSha256: string;
  readonly artifactByteLength: WireU64;
}

export interface VerifyCppCuteFrontendArtifactOptions {
  readonly limits?: Partial<DecodeLimits>;
  readonly artifactLimits?: Partial<CppCuteFrontendArtifactLimits>;
  readonly signal?: AbortSignal;
}

export interface VerifiedCppCuteFrontendArtifactRecord {
  readonly envelope: WireEnvelope<CppCuteFrontendPayloadV1>;
  readonly inputHashes: VerifiedCppCuteInputHashes;
  readonly artifactHash: string;
  readonly transportHash: string;
  readonly artifactBytesSha256: string;
  readonly artifactByteLength: WireU64;
}

export async function verifyCppCuteFrontendArtifact(
  value: unknown,
  options: VerifyCppCuteFrontendArtifactOptions = {},
): Promise<VerifiedCppCuteFrontendArtifact> {
  throwIfAborted(options.signal);
  const limits = resolveDecodeLimits(options.limits);
  const artifactLimits = resolveArtifactLimits(options.artifactLimits);
  const envelope = validateWireEnvelope(value, {
    schema: CPP_CUTE_FRONTEND_ARTIFACT_SCHEMA,
    supportedMajor: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
    supportedMinor: CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
    knownRequiredExtensions: new Set(),
    limits,
  });
  if (envelope.version.minor !== CPP_CUTE_FRONTEND_ARTIFACT_MINOR) {
    cppCuteFrontendArtifactFailure(
      "BG-COMPILER-CPP-CUTE-ARTIFACT-UNSUPPORTED-VERSION",
      "$.version.minor",
      `closed artifact reader supports ${CPP_CUTE_FRONTEND_ARTIFACT_MAJOR}.${CPP_CUTE_FRONTEND_ARTIFACT_MINOR} only`,
    );
  }
  if (envelope.optionalMetadata !== undefined) {
    cppCuteFrontendArtifactFailure(
      "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      "$.optionalMetadata",
      "frontend artifact v1 forbids optional metadata; evidence and provenance are detached records",
    );
  }
  const payload = parseCppCuteFrontendPayload(envelope.payload, artifactLimits);
  const inputHashes = await verifyCppCuteFrontendPayload(payload, { limits });
  throwIfAborted(options.signal);
  const artifactHash = await hashCppCuteFrontendSemantics(payload, envelope.requiredExtensions, { limits });
  const expectedArtifactId = `bg.artifact.cpp-cute-frontend.sha256.${artifactHash}`;
  if (envelope.artifactId !== expectedArtifactId) {
    cppCuteFrontendArtifactFailure(
      "BG-COMPILER-CPP-CUTE-ARTIFACT-HASH-MISMATCH",
      "$.artifactId",
      `artifactId must equal ${expectedArtifactId}`,
    );
  }
  const normalizedEnvelope: WireEnvelope<CppCuteFrontendPayloadV1> = {
    schema: CPP_CUTE_FRONTEND_ARTIFACT_SCHEMA,
    version: {
      major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
      minor: CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
    },
    producer: envelope.producer,
    artifactId: expectedArtifactId,
    payload,
    requiredExtensions: [],
  };
  const canonicalBytes = canonicalJsonBytes(normalizedEnvelope, { limits });
  const artifactBytesSha256 = await sha256Hex(canonicalBytes);
  const artifactByteLength = encodeWireU64(BigInt(canonicalBytes.byteLength));
  const transportHash = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.frontend-artifact-transport.v1",
    envelope: normalizedEnvelope,
  }, { limits });
  throwIfAborted(options.signal);
  const verified = Object.freeze({
    artifactId: expectedArtifactId,
    artifactHash,
    transportHash,
    artifactBytesSha256,
    artifactByteLength,
    profileHash: payload.profileHash,
    sourceSetSha256: inputHashes.sourceSetSha256,
    headerSetSha256: inputHashes.headerSetSha256,
    inputClosureSha256: inputHashes.closureSha256,
    outcome: payload.outcome.kind,
  }) as VerifiedCppCuteFrontendArtifact;
  VERIFIED_ARTIFACTS.set(verified, Object.freeze({
    envelope: normalizedEnvelope,
    inputHashes,
    artifactHash,
    transportHash,
    artifactBytesSha256,
    artifactByteLength,
  }));
  return verified;
}

export async function decodeCppCuteFrontendArtifact(
  bytes: Uint8Array,
  options: VerifyCppCuteFrontendArtifactOptions = {},
): Promise<VerifiedCppCuteFrontendArtifactResource> {
  throwIfAborted(options.signal);
  const snapshot = new Uint8Array(bytes);
  const artifact = await verifyCppCuteFrontendArtifact(
    decodeWireJson(snapshot, options.limits === undefined ? {} : { limits: options.limits }),
    options,
  );
  const canonical = canonicalCppCuteFrontendArtifactBytes(
    artifact,
    options.limits === undefined ? {} : { limits: options.limits },
  );
  if (!equalBytes(snapshot, canonical)) {
    cppCuteFrontendArtifactFailure(
      "BG-COMPILER-CPP-CUTE-ARTIFACT-NONCANONICAL-BYTES",
      "$bytes",
      "producer artifact bytes must exactly equal the canonical normalized envelope",
    );
  }
  const resource = Object.freeze({
    artifactId: artifact.artifactId,
    artifactHash: artifact.artifactHash,
    artifactBytesSha256: artifact.artifactBytesSha256,
    artifactByteLength: artifact.artifactByteLength,
  }) as VerifiedCppCuteFrontendArtifactResource;
  VERIFIED_ARTIFACT_RESOURCES.set(resource, artifact);
  return resource;
}

export function unwrapVerifiedCppCuteFrontendArtifactResource(
  resource: VerifiedCppCuteFrontendArtifactResource,
): VerifiedCppCuteFrontendArtifact {
  if (typeof resource !== "object" || resource === null) unverified();
  const artifact = VERIFIED_ARTIFACT_RESOURCES.get(resource as object);
  if (artifact === undefined) unverified();
  return artifact;
}

export function canonicalCppCuteFrontendArtifactResourceBytes(
  resource: VerifiedCppCuteFrontendArtifactResource,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Uint8Array {
  return canonicalCppCuteFrontendArtifactBytes(unwrapVerifiedCppCuteFrontendArtifactResource(resource), options);
}

export function canonicalCppCuteFrontendArtifactBytes(
  artifact: VerifiedCppCuteFrontendArtifact,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Uint8Array {
  return canonicalJsonBytes(unwrapVerifiedCppCuteFrontendArtifact(artifact).envelope, options);
}

export function unwrapVerifiedCppCuteFrontendArtifact(
  artifact: VerifiedCppCuteFrontendArtifact,
): VerifiedCppCuteFrontendArtifactRecord {
  if (typeof artifact !== "object" || artifact === null) unverified();
  const record = VERIFIED_ARTIFACTS.get(artifact as object);
  if (record === undefined) unverified();
  return record;
}

export async function deriveCppCuteFrontendArtifactId(
  payload: CppCuteFrontendPayloadV1,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  const digest = await hashCppCuteFrontendSemantics(payload, [], options);
  return `bg.artifact.cpp-cute-frontend.sha256.${digest}`;
}

export async function deriveCppCuteStableId(
  kind: string,
  value: JsonValue,
  options: { readonly limits?: Partial<DecodeLimits> } = {},
): Promise<string> {
  if (!STABLE_ID_KIND.test(kind)) {
    cppCuteFrontendArtifactFailure(
      "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      "$.kind",
      "stable ID kind must match ^[a-z][a-z0-9-]*$",
    );
  }
  const digest = await hashCanonicalJson({
    domain: `browsergrad.compiler.cpp-cute.${kind}-id.v1`,
    value,
  }, options);
  return `bg.cpp.${kind}.sha256.${digest}`;
}

async function hashCppCuteFrontendSemantics(
  payload: CppCuteFrontendPayloadV1,
  requiredExtensions: readonly string[],
  options: { readonly limits?: Partial<DecodeLimits> },
): Promise<string> {
  return hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.frontend-artifact.v1",
    schema: CPP_CUTE_FRONTEND_ARTIFACT_SCHEMA,
    version: {
      major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
      minor: CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
    },
    requiredExtensions: [...requiredExtensions].sort(),
    payload,
  }, options);
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) return false;
  let difference = 0;
  for (let index = 0; index < left.byteLength; index += 1) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}

function resolveArtifactLimits(
  overrides: Partial<CppCuteFrontendArtifactLimits> = {},
): CppCuteFrontendArtifactLimits {
  const unknown = Object.keys(overrides).filter((key) => !Object.hasOwn(DEFAULT_CPP_CUTE_FRONTEND_ARTIFACT_LIMITS, key));
  if (unknown.length > 0) {
    cppCuteFrontendArtifactFailure(
      "BG-COMPILER-CPP-CUTE-ARTIFACT-INVALID",
      "$.options.artifactLimits",
      `unknown artifact limit fields: ${unknown.sort().join(", ")}`,
    );
  }
  const result = {
    ...DEFAULT_CPP_CUTE_FRONTEND_ARTIFACT_LIMITS,
    ...overrides,
  };
  for (const key of Object.keys(DEFAULT_CPP_CUTE_FRONTEND_ARTIFACT_LIMITS) as Array<keyof CppCuteFrontendArtifactLimits>) {
    const value = result[key];
    const maximum = DEFAULT_CPP_CUTE_FRONTEND_ARTIFACT_LIMITS[key];
    if (!Number.isSafeInteger(value) || value <= 0 || value > maximum) {
      cppCuteFrontendArtifactFailure(
        "BG-COMPILER-CPP-CUTE-ARTIFACT-RESOURCE-LIMIT",
        `$.options.artifactLimits.${key}`,
        `${key} may only lower the implementation ceiling ${maximum}`,
      );
    }
  }
  return Object.freeze(result);
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted === true) {
    cppCuteFrontendArtifactFailure(
      "BG-COMPILER-CPP-CUTE-ARTIFACT-CANCELLED",
      "$.signal",
      "C++/CuTe artifact verification was aborted",
    );
  }
}

function unverified(): never {
  cppCuteFrontendArtifactFailure(
    "BG-COMPILER-CPP-CUTE-ARTIFACT-UNVERIFIED",
    "$",
    "C++/CuTe frontend operation requires an opaque verified artifact",
  );
}
