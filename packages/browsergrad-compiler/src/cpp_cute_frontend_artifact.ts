import {
  canonicalizeJson,
  decodeWireJson,
  hashCanonicalJson,
  resolveDecodeLimits,
  validateWireEnvelope,
  type DecodeLimits,
  type JsonValue,
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
const STABLE_ID_KIND = /^[a-z][a-z0-9-]*$/u;

declare const verifiedCppCuteFrontendArtifactBrand: unique symbol;

export interface VerifiedCppCuteFrontendArtifact {
  readonly [verifiedCppCuteFrontendArtifactBrand]: true;
  readonly artifactId: string;
  readonly artifactHash: string;
  readonly transportHash: string;
  readonly profileHash: string;
  readonly sourceSetSha256: string;
  readonly headerSetSha256: string;
  readonly inputClosureSha256: string;
  readonly outcome: "accepted" | "rejected";
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
  canonicalizeJson(normalizedEnvelope, { limits });
  const transportHash = await hashCanonicalJson({
    domain: "browsergrad.compiler.cpp-cute.frontend-artifact-transport.v1",
    envelope: normalizedEnvelope,
  }, { limits });
  throwIfAborted(options.signal);
  const verified = Object.freeze({
    artifactId: expectedArtifactId,
    artifactHash,
    transportHash,
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
  }));
  return verified;
}

export async function decodeCppCuteFrontendArtifact(
  bytes: Uint8Array,
  options: VerifyCppCuteFrontendArtifactOptions = {},
): Promise<VerifiedCppCuteFrontendArtifact> {
  throwIfAborted(options.signal);
  return verifyCppCuteFrontendArtifact(
    decodeWireJson(bytes, options.limits === undefined ? {} : { limits: options.limits }),
    options,
  );
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
