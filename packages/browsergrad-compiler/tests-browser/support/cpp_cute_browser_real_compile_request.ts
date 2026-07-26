import {
  encodeWireU64,
  sha256Hex,
  type WireU64,
} from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  CPP_CUTE_FRONTEND_REQUEST_SCHEMA,
  deriveCppCuteFrontendEntryRequestId,
  deriveCppCuteFrontendRequestHash,
  deriveCppCuteFrontendSourceFileId,
  prepareCppCuteFrontendRequest,
  type CppCuteFrontendEntryRequestV1,
  type CppCuteFrontendRequestBodyV1,
  type CppCuteFrontendRequestLimitsV1,
  type CppCuteFrontendRequestSourceFileV1,
  type CppCuteFrontendRequestV1,
  type PreparedCppCuteFrontendRequest,
} from "../../src/cpp_cute_frontend_request.js";
import type {
  PreparedCppCuteFrontendProfile,
} from "../../src/cpp_cute_frontend_profile.js";
import {
  CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
  CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
  type CppCuteAffineLayoutFactV1,
  type CppCuteHierarchyV1,
} from "../../src/cpp_cute_frontend_types.js";
import type {
  CppCuteBrowserRealCompileCase,
} from "../../src/cpp_cute_browser_real_compile_cases.js";

/** One producer-neutral request builder shared by all real browser lanes. */
export async function prepareCppCuteBrowserRealCompileViewCopyRequest(
  profile: PreparedCppCuteFrontendProfile,
  compileCase: CppCuteBrowserRealCompileCase,
): Promise<PreparedCppCuteFrontendRequest> {
  const bytes = new TextEncoder().encode(compileCase.source);
  const source = await sourceFile(compileCase, bytes);
  const tokenBegin = compileCase.source.indexOf("copy_views");
  if (tokenBegin < 0) {
    throw new Error("pinned real source lost its view-copy declaration");
  }
  const anchor = {
    virtualPath: compileCase.virtualPath,
    beginByte: encodeWireU64(BigInt(tokenBegin)),
    endByte: encodeWireU64(BigInt(tokenBegin + "copy_views".length)),
    tokenSha256: await sha256Hex(
      bytes.subarray(tokenBegin, tokenBegin + "copy_views".length),
    ),
  };
  const entryBody = {
    requestId: `bg.cpp.entry-request.sha256.${"0".repeat(64)}`,
    kind: "view-copy" as const,
    declarationKind: "function" as const,
    anchor,
  };
  const entryRequest: CppCuteFrontendEntryRequestV1 = {
    ...entryBody,
    requestId: await deriveCppCuteFrontendEntryRequestId(entryBody),
  };
  const body: CppCuteFrontendRequestBodyV1 = {
    schema: CPP_CUTE_FRONTEND_REQUEST_SCHEMA,
    version: { major: 1, minor: 0 },
    compilationContractHash: profile.compilationContractHash,
    mainVirtualPath: compileCase.virtualPath,
    files: [source],
    entryRequests: [entryRequest],
    expectedArtifact: {
      schema: "browsergrad.compiler.cpp-cute.frontend-artifact",
      version: {
        major: CPP_CUTE_FRONTEND_ARTIFACT_MAJOR,
        minor: CPP_CUTE_FRONTEND_ARTIFACT_MINOR,
      },
    },
    limits: requestLimits(profile),
  };
  const request: CppCuteFrontendRequestV1 = {
    ...body,
    requestId:
      `bg.cpp.frontend-request.sha256.${await deriveCppCuteFrontendRequestHash(body)}`,
  };
  return prepareCppCuteFrontendRequest(profile, request, [{
    virtualPath: compileCase.virtualPath,
    bytes,
  }]);
}

export function cppCuteBrowserRealCompileStaticLayoutProjection(
  fact: CppCuteAffineLayoutFactV1,
): Readonly<{ shape: readonly string[]; strides: readonly string[] }> {
  const flatten = (
    hierarchy: CppCuteHierarchyV1,
    path: string,
  ): readonly string[] => {
    if (hierarchy.kind !== "tuple") {
      throw new Error(`${path} is not one flat static tuple`);
    }
    return Object.freeze(hierarchy.elements.map((element, index) => {
      if (element.kind !== "scalar" ||
          element.value.kind !== "integer") {
        throw new Error(`${path}[${index}] is not one static integer leaf`);
      }
      return element.value.value;
    }));
  };
  return Object.freeze({
    shape: flatten(fact.shape, "$.shape"),
    strides: flatten(fact.stride, "$.stride"),
  });
}

async function sourceFile(
  compileCase: CppCuteBrowserRealCompileCase,
  bytes: Uint8Array,
): Promise<CppCuteFrontendRequestSourceFileV1> {
  const body = {
    role: "main-source" as const,
    virtualPath: compileCase.virtualPath,
    contentSha256: await sha256Hex(bytes),
    byteLength: wire(bytes.byteLength),
    includeRootId: null,
  };
  return {
    ...body,
    fileId: await deriveCppCuteFrontendSourceFileId(body),
  };
}

function requestLimits(
  profile: PreparedCppCuteFrontendProfile,
): CppCuteFrontendRequestLimitsV1 {
  const limits = profile.extractionLimits;
  return {
    maxSourceFiles: limits.maxSourceFiles,
    maxSourceBytes: limits.maxSourceBytes,
    // The configured libc++ pass opens just over 1,024 distinct files.
    maxHeaderFiles: 2_048,
    maxHeaderBytes: limits.maxHeaderBytes,
    maxIncludeDepth: limits.maxIncludeDepth,
    maxMacroExpansions: limits.maxMacroExpansions,
    maxPreprocessedTokens: limits.maxPreprocessedTokens,
    maxAstNodes: limits.maxAstNodes,
    maxConstexprSteps: limits.maxConstexprSteps,
    maxTemplateInstantiations: limits.maxTemplateInstantiations,
    maxTemplateDepth: limits.maxTemplateDepth,
    maxDeclarations: 16_384,
    maxTypes: 16_384,
    maxConstants: 16_384,
    maxLayouts: 1_024,
    maxTensors: 1_024,
    maxOperations: 4_096,
    maxTargetIntrinsics: 1_024,
    maxDiagnostics: 4_096,
    maxOutputBytes: limits.maxOutputBytes,
  };
}

function wire(value: number): WireU64 {
  return encodeWireU64(BigInt(value));
}
