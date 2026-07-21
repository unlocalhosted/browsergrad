import {
  createVerifiedDenseLogicalGemmTileArtifacts,
  type VerifiedLogicalGemmTileArtifacts,
} from "@unlocalhosted/browsergrad-semantic-core/kernel";
import { SemanticSchemaError } from "@unlocalhosted/browsergrad-semantic-core/schema";
import {
  unwrapAuthorizedCppCuteFrontendArtifact,
  type AuthorizedCppCuteFrontendArtifact,
} from "./cpp_cute_frontend_authorization.js";
import {
  CppCuteLogicalGemmTileLoweringError,
  cppCuteLogicalGemmTileFailure,
  normalizeCppCuteLogicalGemmTileOptions,
  prepareVerifiedCppCuteLogicalGemmTileSemantics,
  throwIfCppCuteLogicalGemmTileAborted,
  type PrepareVerifiedCppCuteLogicalGemmTileSemanticsOptions,
  type PrepareVerifiedCppCuteLogicalGemmTileSemanticsRequest,
} from "./cpp_cute_logical_gemm_semantics.js";

export {
  CppCuteLogicalGemmTileLoweringError,
  prepareVerifiedCppCuteLogicalGemmTileSemantics,
  type CppCuteLogicalGemmTileLoweringErrorCode,
  type PreparedVerifiedCppCuteLogicalGemmTileSemantics,
  type PrepareVerifiedCppCuteLogicalGemmTileSemanticsOptions,
  type PrepareVerifiedCppCuteLogicalGemmTileSemanticsRequest,
} from "./cpp_cute_logical_gemm_semantics.js";

export type LowerAuthorizedCppCuteLogicalGemmTileEntryRequest =
  PrepareVerifiedCppCuteLogicalGemmTileSemanticsRequest;

export type LowerAuthorizedCppCuteLogicalGemmTileEntryOptions =
  PrepareVerifiedCppCuteLogicalGemmTileSemanticsOptions;

/**
 * Verified typed-artifact lowering only. The current native Clang extractor
 * does not emit logical-gemm-tile facts, and this transition grants neither a
 * physical schedule nor backend execution authority.
 */
export async function lowerAuthorizedCppCuteLogicalGemmTileEntry(
  authorization: AuthorizedCppCuteFrontendArtifact,
  request: LowerAuthorizedCppCuteLogicalGemmTileEntryRequest,
  options: LowerAuthorizedCppCuteLogicalGemmTileEntryOptions = {},
): Promise<VerifiedLogicalGemmTileArtifacts> {
  const normalizedOptions = normalizeCppCuteLogicalGemmTileOptions(options);
  throwIfCppCuteLogicalGemmTileAborted(normalizedOptions.signal);
  const authorized = unwrapAuthorizedCppCuteFrontendArtifact(authorization);
  const semantics = await prepareVerifiedCppCuteLogicalGemmTileSemantics(
    authorized.artifact,
    request,
    Object.freeze({
      limits: normalizedOptions.limits,
      ...(normalizedOptions.signal === undefined ? {} : { signal: normalizedOptions.signal }),
    }),
  );
  throwIfCppCuteLogicalGemmTileAborted(normalizedOptions.signal);
  try {
    const artifacts = await createVerifiedDenseLogicalGemmTileArtifacts({
      m: semantics.m,
      n: semantics.n,
      k: semantics.k,
      logicalTile: semantics.fact.logicalTile,
    }, {
      producer: { id: "browsergrad.compiler.cpp-cute-logical-gemm-lowering", version: "1" },
      layoutArtifactId: "authorized-cpp-cute-logical-gemm-layout",
      kernelArtifactId: "authorized-cpp-cute-logical-gemm-kernel",
      limits: normalizedOptions.limits,
    });
    throwIfCppCuteLogicalGemmTileAborted(normalizedOptions.signal);
    return artifacts;
  } catch (cause) {
    if (cause instanceof CppCuteLogicalGemmTileLoweringError) throw cause;
    if (!(cause instanceof SemanticSchemaError)) throw cause;
    cppCuteLogicalGemmTileFailure(
      cause.diagnostic.code.endsWith("RESOURCE-LIMIT")
        ? "BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-RESOURCE-LIMIT"
        : "BG-COMPILER-CPP-CUTE-LOGICAL-GEMM-UNSUPPORTED-PROFILE",
      cause.diagnostic.path ?? "$.artifacts",
      `shared semantic-core rejected the authorized logical GEMM: ${cause.message}`,
      { cause },
    );
  }
}
