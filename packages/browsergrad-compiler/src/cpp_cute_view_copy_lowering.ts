import {
  createVerifiedViewCopyArtifacts,
  type VerifiedViewCopyArtifacts,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import {
  SemanticSchemaError,
  encodeWireI64,
  parseWireU64,
  wireIntegerToBigInt,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapAuthorizedCppCuteFrontendArtifact,
  type AuthorizedCppCuteFrontendArtifact,
} from "./cpp_cute_frontend_authorization.js";
import {
  CppCuteViewCopyLoweringError,
  cppCuteViewCopyFailure,
  normalizeCppCuteViewCopyOptions,
  prepareVerifiedCppCuteViewCopySemantics,
  throwIfCppCuteViewCopyAborted,
  type PrepareVerifiedCppCuteViewCopySemanticsOptions,
} from "./cpp_cute_view_copy_semantics.js";

export {
  CppCuteViewCopyLoweringError,
  type CppCuteViewCopyLoweringErrorCode,
} from "./cpp_cute_view_copy_semantics.js";

const CAPTURED_OBJECT = Object;
const CAPTURED_REFLECT = Reflect;
const NATIVE_OBJECT_GET_OWN_PROPERTY_DESCRIPTORS = Object.getOwnPropertyDescriptors;
const NATIVE_OBJECT_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const NATIVE_OBJECT_FREEZE = Object.freeze;
const NATIVE_REFLECT_APPLY = Reflect.apply;
const NATIVE_REFLECT_OWN_KEYS = Reflect.ownKeys;

export interface LowerAuthorizedCppCuteViewCopyEntryRequest {
  readonly entryId: string;
  readonly sourceAllocationByteLength: WireU64;
  readonly destinationAllocationByteLength: WireU64;
  readonly sourceByteOffset: WireU64;
  readonly destinationByteOffset: WireU64;
}

export type LowerAuthorizedCppCuteViewCopyEntryOptions =
  PrepareVerifiedCppCuteViewCopySemanticsOptions;

interface NormalizedRequest {
  readonly entryId: string;
  readonly sourceAllocationByteLength: bigint;
  readonly destinationAllocationByteLength: bigint;
  readonly sourceByteOffset: bigint;
  readonly destinationByteOffset: bigint;
}

/**
 * Producer-neutral semantic transition from one exact authorized frontend
 * result. Runtime allocation capacities remain explicit host facts; CuTe
 * `cosize` is never treated as pointer storage authority.
 */
export async function lowerAuthorizedCppCuteViewCopyEntry(
  authorization: AuthorizedCppCuteFrontendArtifact,
  request: LowerAuthorizedCppCuteViewCopyEntryRequest,
  options: LowerAuthorizedCppCuteViewCopyEntryOptions = {},
): Promise<VerifiedViewCopyArtifacts> {
  const normalizedOptions = normalizeCppCuteViewCopyOptions(options);
  throwIfCppCuteViewCopyAborted(normalizedOptions.signal);
  const normalizedRequest = normalizeRequest(request);
  const authorized = unwrapAuthorizedCppCuteFrontendArtifact(authorization);
  const semantics = await prepareVerifiedCppCuteViewCopySemantics(
    authorized.artifact,
    { entryId: normalizedRequest.entryId },
    NATIVE_OBJECT_FREEZE({
      limits: normalizedOptions.limits,
      ...(normalizedOptions.signal === undefined ? {} : { signal: normalizedOptions.signal }),
    }),
  );
  validateStorage(
    normalizedRequest.sourceAllocationByteLength,
    normalizedRequest.sourceByteOffset,
    semantics.sourceSpanElements,
    "$.request.sourceAllocationByteLength",
    "$.request.sourceByteOffset",
  );
  validateStorage(
    normalizedRequest.destinationAllocationByteLength,
    normalizedRequest.destinationByteOffset,
    semantics.destinationSpanElements,
    "$.request.destinationAllocationByteLength",
    "$.request.destinationByteOffset",
  );
  throwIfCppCuteViewCopyAborted(normalizedOptions.signal);
  try {
    const artifacts = await createVerifiedViewCopyArtifacts({
      dtype: semantics.dtype,
      symbols: [],
      constraints: [],
      source: {
        layout: semantics.sourceLayout,
        allocation: {
          byteLength: constant(normalizedRequest.sourceAllocationByteLength),
          memorySpace: { kind: "global" },
          alignmentBytes: 4,
        },
        byteOffset: constant(normalizedRequest.sourceByteOffset),
        requiredAlignmentBytes: 4,
      },
      destination: {
        layout: semantics.destinationLayout,
        allocation: {
          byteLength: constant(normalizedRequest.destinationAllocationByteLength),
          memorySpace: { kind: "global" },
          alignmentBytes: 4,
        },
        byteOffset: constant(normalizedRequest.destinationByteOffset),
        requiredAlignmentBytes: 4,
      },
      invalidSource: { kind: "reject" },
    }, {
      producer: { id: "browsergrad.compiler.cpp-cute-view-copy-lowering", version: "1" },
      layoutArtifactId: "authorized-cpp-cute-view-copy-layout",
      kernelArtifactId: "authorized-cpp-cute-view-copy-kernel",
      limits: normalizedOptions.limits,
    });
    throwIfCppCuteViewCopyAborted(normalizedOptions.signal);
    return artifacts;
  } catch (cause) {
    if (cause instanceof CppCuteViewCopyLoweringError) throw cause;
    if (!(cause instanceof SemanticSchemaError)) throw cause;
    cppCuteViewCopyFailure(
      cause.diagnostic.code.endsWith("RESOURCE-LIMIT")
        ? "BG-COMPILER-CPP-CUTE-VIEW-COPY-RESOURCE-LIMIT"
        : "BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-LAYOUT",
      cause.diagnostic.path ?? "$.artifacts",
      `shared semantic-core rejected the authorized view-copy: ${cause.message}`,
      { cause },
    );
  }
}

function validateStorage(
  byteLength: bigint,
  byteOffset: bigint,
  spanElements: bigint,
  byteLengthPath: string,
  byteOffsetPath: string,
): void {
  if (byteLength % 4n !== 0n) {
    invalidRequest(
      byteLengthPath,
      "32-bit scalar allocation byte length must be 4-byte aligned",
    );
  }
  if (byteOffset % 4n !== 0n) {
    invalidRequest(
      byteOffsetPath,
      "32-bit scalar view byte offset must be 4-byte aligned",
    );
  }
  const required = byteOffset + spanElements * 4n;
  if (required > byteLength) {
    cppCuteViewCopyFailure(
      "BG-COMPILER-CPP-CUTE-VIEW-COPY-UNSUPPORTED-LAYOUT",
      byteLengthPath,
      `explicit allocation has ${byteLength} bytes but the verified affine address span requires ${required}`,
    );
  }
}

function normalizeRequest(request: LowerAuthorizedCppCuteViewCopyEntryRequest): NormalizedRequest {
  const descriptors = closedDataRecord(request, [
    "entryId",
    "sourceAllocationByteLength",
    "destinationAllocationByteLength",
    "sourceByteOffset",
    "destinationByteOffset",
  ], "$.request");
  const entryId = descriptors.entryId?.value;
  if (typeof entryId !== "string" || entryId.length === 0) invalidRequest("$.request.entryId", "entryId must be a non-empty string");
  const sourceAllocationByteLength = unsigned(descriptors.sourceAllocationByteLength?.value, "$.request.sourceAllocationByteLength", true);
  const destinationAllocationByteLength = unsigned(descriptors.destinationAllocationByteLength?.value, "$.request.destinationAllocationByteLength", true);
  const sourceByteOffset = unsigned(descriptors.sourceByteOffset?.value, "$.request.sourceByteOffset", false);
  const destinationByteOffset = unsigned(descriptors.destinationByteOffset?.value, "$.request.destinationByteOffset", false);
  return NATIVE_OBJECT_FREEZE({
    entryId,
    sourceAllocationByteLength,
    destinationAllocationByteLength,
    sourceByteOffset,
    destinationByteOffset,
  });
}

function closedDataRecord(
  value: unknown,
  expectedKeys: readonly string[],
  path: string,
): Readonly<Record<string, PropertyDescriptor>> {
  let prototype: object | null;
  let descriptors: PropertyDescriptorMap;
  let keys: readonly PropertyKey[];
  try {
    prototype = NATIVE_REFLECT_APPLY(NATIVE_OBJECT_GET_PROTOTYPE_OF, CAPTURED_OBJECT, [value]) as object | null;
    descriptors = NATIVE_REFLECT_APPLY(NATIVE_OBJECT_GET_OWN_PROPERTY_DESCRIPTORS, CAPTURED_OBJECT, [value]) as PropertyDescriptorMap;
    keys = NATIVE_REFLECT_APPLY(NATIVE_REFLECT_OWN_KEYS, CAPTURED_REFLECT, [value]) as readonly PropertyKey[];
  } catch (cause) {
    invalidRequest(path, "request inspection failed", { cause });
  }
  if (prototype !== CAPTURED_OBJECT.prototype && prototype !== null) invalidRequest(path, "expected a plain object");
  if (keys.some((key) => typeof key !== "string" || !expectedKeys.includes(key)) ||
      keys.length !== expectedKeys.length || expectedKeys.some((key) => !keys.includes(key))) {
    invalidRequest(path, `expected only ${expectedKeys.join(", ")}`);
  }
  for (const key of keys) {
    const descriptor = descriptors[String(key)];
    if (descriptor === undefined || descriptor.enumerable !== true || !("value" in descriptor)) {
      invalidRequest(`${path}.${String(key)}`, "expected an enumerable data property without accessors");
    }
  }
  return descriptors;
}

function unsigned(value: unknown, path: string, positive: boolean): bigint {
  try {
    const parsed = wireIntegerToBigInt(parseWireU64(value, path));
    if (positive && parsed === 0n) invalidRequest(path, "allocation byte length must be positive");
    if (parsed > 0x7fff_ffff_ffff_ffffn) invalidRequest(path, "value exceeds signed semantic expression range");
    return parsed;
  } catch (cause) {
    if (cause instanceof CppCuteViewCopyLoweringError) throw cause;
    invalidRequest(path, "expected a canonical unsigned wire integer", { cause });
  }
}

function constant(value: bigint) {
  return { kind: "const" as const, value: encodeWireI64(value) };
}

function invalidRequest(path: string, message: string, options?: ErrorOptions): never {
  cppCuteViewCopyFailure("BG-COMPILER-CPP-CUTE-VIEW-COPY-INVALID-REQUEST", path, message, options);
}
