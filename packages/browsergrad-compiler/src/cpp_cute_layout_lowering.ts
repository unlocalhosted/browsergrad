import {
  traceLayoutExpressionCoordinate,
  type LayoutExpressionCoordinateRequest,
  type LayoutExpressionCoordinateTrace,
  type PreparedLayoutExpression,
} from "@unlocalhosted/browsergrad-semantic-core/layout";
import { hashCanonicalJson } from "@unlocalhosted/browsergrad-semantic-core/schema";
import { unwrapVerifiedCppCuteFrontendArtifact } from "./cpp_cute_frontend_artifact.js";
import {
  unwrapAuthorizedCppCuteFrontendArtifact,
  type AuthorizedCppCuteFrontendArtifact,
} from "./cpp_cute_frontend_authorization.js";
import type {
  CppCuteAffineLayoutFactV1,
  CppCuteFrontendEntryV1,
  CppCuteMacroExpansionV1,
  CppCuteSourceSpanV1,
} from "./cpp_cute_frontend_types.js";
import {
  CppCuteLayoutLoweringError,
  normalizeCppCuteLayoutOptions,
  prepareCppCuteLayoutSemantics,
  throwIfCppCuteLayoutAborted,
  validateCppCuteLayoutRequest,
  type CppCuteLayoutLoweringErrorCode,
  type LowerAuthorizedCppCuteLayoutEntryOptions,
  type LowerAuthorizedCppCuteLayoutEntryRequest,
} from "./cpp_cute_layout_semantics.js";

export {
  CppCuteLayoutLoweringError,
  prepareVerifiedCppCuteLayoutSemantics,
  type CppCuteLayoutLoweringErrorCode,
  type LowerAuthorizedCppCuteLayoutEntryOptions,
  type LowerAuthorizedCppCuteLayoutEntryRequest,
  type PreparedVerifiedCppCuteLayoutSemantics,
} from "./cpp_cute_layout_semantics.js";

declare const loweredCppCuteLayoutBrand: unique symbol;

export interface LoweredCppCuteLayoutEntry {
  readonly [loweredCppCuteLayoutBrand]: never;
  readonly layoutSemanticHash: string;
  readonly indexMapId: string;
  readonly coordinateRank: number;
  readonly originHash: string;
}

export interface LoweredCppCuteLayoutEntryRecord {
  readonly authorization: AuthorizedCppCuteFrontendArtifact;
  readonly preparedLayout: PreparedLayoutExpression;
  readonly entry: Extract<CppCuteFrontendEntryV1, { readonly kind: "layout" }>;
  readonly fact: CppCuteAffineLayoutFactV1;
  readonly originSpanRecords: readonly CppCuteSourceSpanV1[];
  readonly macroExpansionRecords: readonly CppCuteMacroExpansionV1[];
}

interface LoweredRecord extends LoweredCppCuteLayoutEntryRecord {
  readonly originHash: string;
}

const LOWERED_LAYOUTS = new WeakMap<object, LoweredRecord>();

class LoweredCppCuteLayoutEntryValue {
  readonly layoutSemanticHash: string;
  readonly indexMapId: string;
  readonly coordinateRank: number;
  readonly originHash: string;

  constructor(record: LoweredRecord) {
    this.layoutSemanticHash = record.preparedLayout.layoutSemanticHash;
    this.indexMapId = record.preparedLayout.indexMapId;
    this.coordinateRank = record.preparedLayout.coordinateRank;
    this.originHash = record.originHash;
    LOWERED_LAYOUTS.set(this, record);
    Object.freeze(this);
  }
}

export async function lowerAuthorizedCppCuteLayoutEntry(
  authorization: AuthorizedCppCuteFrontendArtifact,
  request: LowerAuthorizedCppCuteLayoutEntryRequest,
  options: LowerAuthorizedCppCuteLayoutEntryOptions = {},
): Promise<LoweredCppCuteLayoutEntry> {
  const authorized = unwrapAuthorizedCppCuteFrontendArtifact(authorization);
  const normalizedOptions = normalizeCppCuteLayoutOptions(options);
  throwIfCppCuteLayoutAborted(normalizedOptions.signal);
  const entryId = validateCppCuteLayoutRequest(request);
  const artifact = unwrapVerifiedCppCuteFrontendArtifact(authorized.artifact);
  const semantic = await prepareCppCuteLayoutSemantics(
    artifact.envelope.payload,
    entryId,
    normalizedOptions,
  );
  const { entry, fact, preparedLayout, origin } = semantic;
  let originHash: string;
  try {
    originHash = await hashCanonicalJson({
      domain: "browsergrad.compiler.cpp-cute.layout-origin.v2",
      layoutSemanticHash: preparedLayout.layoutSemanticHash,
      frontendArtifactId: authorization.artifactId,
      frontendArtifactHash: authorization.artifactHash,
      frontendArtifactBytesSha256: authorization.artifactBytesSha256,
      frontendArtifactByteLength: authorization.artifactByteLength,
      profileHash: authorization.profileHash,
      compilationContractHash: authorization.compilationContractHash,
      sourceSetSha256: authorization.sourceSetSha256,
      headerSetSha256: authorization.headerSetSha256,
      inputClosureSha256: authorization.inputClosureSha256,
      evidenceKind: authorization.evidenceKind,
      evidenceHash: authorization.evidenceHash,
      entryId: entry.entryId,
      factId: fact.factId,
      resultDeclarationId: fact.resultDeclarationId,
      origin: fact.origin,
      spans: origin.spans,
      macroExpansions: origin.macroExpansions,
    }, { limits: normalizedOptions.limits });
  } catch (error) {
    failure(
      "BG-COMPILER-CPP-CUTE-LAYOUT-RESOURCE-LIMIT",
      "$.artifact.layoutFact.origin",
      "layout origin projection exceeded canonical hashing limits",
      { cause: error },
    );
  }
  throwIfCppCuteLayoutAborted(normalizedOptions.signal);
  const record: LoweredRecord = Object.freeze({
    authorization,
    preparedLayout,
    entry,
    fact,
    originSpanRecords: origin.spans,
    macroExpansionRecords: origin.macroExpansions,
    originHash,
  });
  return new LoweredCppCuteLayoutEntryValue(record) as unknown as LoweredCppCuteLayoutEntry;
}

export function traceLoweredCppCuteLayoutCoordinate(
  lowered: LoweredCppCuteLayoutEntry,
  request: LayoutExpressionCoordinateRequest,
): LayoutExpressionCoordinateTrace {
  return traceLayoutExpressionCoordinate(unwrapLoweredCppCuteLayoutEntry(lowered).preparedLayout, request);
}

export function unwrapLoweredCppCuteLayoutEntry(
  lowered: LoweredCppCuteLayoutEntry,
): LoweredCppCuteLayoutEntryRecord {
  if ((typeof lowered !== "object" && typeof lowered !== "function") || lowered === null) {
    failure("BG-COMPILER-CPP-CUTE-LAYOUT-UNVERIFIED", "$", "expected a compiler-authorized lowered CuTe layout");
  }
  const record = LOWERED_LAYOUTS.get(lowered as object);
  if (record === undefined) {
    failure("BG-COMPILER-CPP-CUTE-LAYOUT-UNVERIFIED", "$", "lowered CuTe layout was not created by this module instance");
  }
  return record;
}

function failure(
  code: CppCuteLayoutLoweringErrorCode,
  path: string,
  message: string,
  options?: ErrorOptions,
): never {
  throw new CppCuteLayoutLoweringError(code, path, message, options);
}
